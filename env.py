import gym
import time
import numpy as np
from util import hyper
from gym import spaces

import torch
import torch.nn.functional as F

torch.manual_seed(42)
torch.cuda.manual_seed(42)
np.random.seed(42)


class ADEnv(gym.Env):
    """
    Customized environment for anomaly detection
    """
    def __init__(self,dataset: np.ndarray,sampling_Du=1000,prob_1=0.5,prob_2=0.5,label_normal=0,label_anomaly=1, name="default"):
        """
        Initialize anomaly environment for DPLAN algorithm.
        :param dataset: Input dataset in the form of 2-D array. The Last column is the label.
        :param sampling_Du: Number of sampling on D_u for the generator g_u
        :param prob_au: Probability of performing g_a.
        :param label_normal: label of normal instances
        :param label_anomaly: label of anomaly instances
        """
        super().__init__()
        self.name=name

        # hyperparameters:
        self.num_S=sampling_Du
        self.normal=label_normal
        self.anomaly=label_anomaly
        self.prob_1=prob_1
        self.prob_2=prob_2
        self.contamination_rate = hyper['contamination_rate']
        self.normal_rate = 1 - self.contamination_rate  

        # Dataset infos: D_a and D_u
        self.m,self.n=dataset.shape
        self.n_feature=self.n-1
        self.n_samples=self.m
        self.x=dataset[:,:self.n_feature]
        self.y=dataset[:,self.n_feature]
        self.dataset=dataset
        self.index_u=np.where(self.y==self.normal)[0]
        self.index_a=np.where(self.y==self.anomaly)[0]

        self.number_of_anomaly_choose = 60 
        self.weight = np.ones(self.n_samples)


        # observation space:
        self.observation_space=spaces.Discrete(self.m)

        # action space: 0 or 1
        self.action_space=spaces.Discrete(2)

        # initial state
        self.counts=None
        self.state=None
        self.DQN=None

        # instances' pseudo-labels on the representation space
        self.places = None
        # instances' position rewards
        self.scores = None

        # initialize consistent punishment
        self.correlation_punishment = 0.2



    def generate_state(self,action,s_t):
        # sampling function for D_u
        S=np.random.choice(self.index_u,self.num_S)
        in_index = np.concatenate(np.where(self.places[S] == 0))
        out_index = np.concatenate(np.where(self.places[S] == 1))
        if len(out_index == 0): 
            S=np.random.choice(self.index_u,self.num_S)
            in_index = np.concatenate(np.where(self.places[S] == 0))
            out_index = np.concatenate(np.where(self.places[S] == 1))

        # calculate distance in the space of last hidden layer of DQN
        all_x=self.x[np.append(S,s_t)]

        all_dqn_s = self.DQN.get_latent(all_x)
        all_dqn_s = all_dqn_s.cpu().detach().numpy()
        dqn_s=all_dqn_s[:-1]
        dqn_st=all_dqn_s[-1]

        dist = np.sqrt(np.sum(np.square(dqn_s-dqn_st),axis=1))
        
        # choose anomalous instance
        if np.random.rand() < self.prob_1:
            if np.random.rand() < self.prob_2:
                index = np.random.choice(self.index_a)
            else:
                index = np.random.choice(out_index)
                    
        # choose normal instance
        else:
            if action == 1:
                loc = np.argmin(dist[in_index])
            elif action == 0:
                loc = np.argmax(dist[in_index])
            
            index = S[in_index[loc]]
            index = np.random.choice(in_index)

        return index
    
    # obtain c,R in the hypersphere boundary
    def DeepSVDD(self,x,threshold):
        c = np.sum(x[self.index_u],axis=0) / len(self.index_u)
        dist_c = np.sqrt(np.sum(np.square(x[self.index_u] - c),axis=1))
        sorted_dist = np.argsort(dist_c)
        index = int(threshold * len(self.index_u))
        R = dist_c[sorted_dist[index]]
        return R,c


    def get_DeepSVDD_in_or_out(self,x,R,c):
        # places = np.zeros(self.n_samples)
        places = np.zeros(x.shape[0])

        dists = np.sqrt(np.sum(np.square(x - c),axis=1)) - R

        index_out = np.where(dists > 0)
        index_in = np.where(dists <= 0)

        # obtain pseudo-labels
        places[index_out] = 1
        places[index_in] = 0
        
        scores = dists / (2*R)

        return places,scores


    # 只判断是否在球内还是在球外
    def DQN_DeepSVDD(self,model):
        # get the output of penulti-layer
        latent_x=model.get_latent(self.x)
        latent_x=latent_x.cpu().detach().numpy()
        R,c = self.DeepSVDD(latent_x,self.normal_rate)
        places,scores = self.get_DeepSVDD_in_or_out(latent_x,R,c)
        self.places = places
        self.scores = scores

    # 合并为一个奖励
    def total_reward(self,action,s_t):
        # Anomaly-biased External Handcrafted Reward Function h
        if (action==1) & (s_t in self.index_a) & (self.places[s_t] == 0):
            return 1+self.scores[s_t]
        elif (action==1) & (s_t in self.index_a) & (self.places[s_t] == 1):
            return 1+self.scores[s_t]
        elif (action==0) & (s_t in self.index_a) & (self.places[s_t] == 0):
            return -1+self.scores[s_t]
        elif (action==0) & (s_t in self.index_a) & (self.places[s_t] == 1):
            return -1+self.scores[s_t]
        elif (action==0) & (s_t in self.index_u) & (self.places[s_t] == 0):
            return -self.scores[s_t]
        elif (action==0) & (s_t in self.index_u) & (self.places[s_t] == 1):
            return -self.correlation_punishment
        elif (action==1) & (s_t in self.index_u) & (self.places[s_t] == 0):
            return -self.correlation_punishment
        elif (action==1) & (s_t in self.index_u) & (self.places[s_t] == 1):
            return self.scores[s_t]

    
    def get_consistent_punishment(self):
        num_a = len(self.index_a)
        num_u = 100 - num_a
        if num_u > 0:
            S = np.random.choice(self.index_u,num_u)
            all_S = np.append(self.index_a,S)  
        else:
            all_S = self.index_a

        all_x = self.x[all_S]
        with torch.no_grad():
            q_value = F.softmax(self.DQN(all_x),dim=1).detach().cpu().numpy()[:,1]
        DeepSVDD_value = self.scores[all_S]
        corr_matrix = np.corrcoef(q_value, DeepSVDD_value)
        correlation_value = corr_matrix[0, 1]
        correlation_value = (correlation_value+1) / 2 
        # return correlation_value
        self.correlation_punishment = correlation_value * 0.2

    def step(self,action):
        self.state = int(self.state)
        # store former state
        s_t=self.state
    
        s_tp1=self.generate_state(action,s_t)

        # change to the next state
        self.state=s_tp1
        self.state = int(self.state)
        self.counts+=1

        # calculate the reward
        reward=self.total_reward(action,s_t)
        # done: whether terminal or not
        done=False

        # info
        info={"State t":s_t, "Action t": action, "State t+1":s_tp1}

        return self.state, reward, done, info

    def reset(self):
        # reset the status of environment
        self.counts=0
        # the first observation is uniformly sampled from the D_u
        self.state=np.random.choice(self.index_u)

        return self.state