from util import hyper,write_results,write_dataset_best_para
from env import ADEnv
from DPLAN import DPLAN
import torch
import os
import pandas as pd
import numpy as np
import time
from visual import visual


torch.manual_seed(42)
torch.cuda.manual_seed(42)
np.random.seed(42)

import warnings
warnings.filterwarnings("ignore", message="invalid value encountered in divide")



BASE_PATH = '/home/shz/ours/Data'



LABEL_NORMAL = 0
LABEL_ANOMALY = 1
CONTAMINATION_RATE  = hyper['contamination_rate']
NUM_ANOMALY_KNOWS = hyper['num_anomaly_knows']
NUM_RUNS = hyper['runs']

dataset = 'NB15'
subsets = ['NB15']

MODELS_PATH = '/home/shz/ours/DRDeepSAD/model_parameters/{}/{}_{}'.format(dataset,NUM_ANOMALY_KNOWS,CONTAMINATION_RATE)
RESULTS_PATH = '/home/shz/ours/DRDeepSAD/result/{}/{}_{}'.format(dataset,NUM_ANOMALY_KNOWS,CONTAMINATION_RATE)
PROCESS_PATH = '/home/shz/ours/DRDeepSAD/process/{}/{}_{}'.format(dataset,NUM_ANOMALY_KNOWS,CONTAMINATION_RATE)
FIGURE_PATH = '/home/shz/ours/DRDeepSAD/figure/{}/{}_{}'.format(dataset,NUM_ANOMALY_KNOWS,CONTAMINATION_RATE)


os.environ['CUDA_VISIBLE_DEVICES'] = '0'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
results_filename = os.path.join(RESULTS_PATH, 'results.csv')
best_para_filename = os.path.join(RESULTS_PATH, 'best_para.csv')

if not os.path.exists(MODELS_PATH):
    os.makedirs(MODELS_PATH)
if not os.path.exists(RESULTS_PATH):
    os.makedirs(RESULTS_PATH)
if not os.path.exists(PROCESS_PATH):
    os.makedirs(PROCESS_PATH)
if not os.path.exists(FIGURE_PATH):
    os.makedirs(FIGURE_PATH)   
   

with open(results_filename, 'w') as f:
        f.write('dataset,subset,para,pr_mean,pr_std,roc_mean,roc_std,pr_max,roc_max,pr_min,roc_min\n')

        
prob1 = 0.6
prob2 = 0.7
count = 0




best_pr = 0
best_para = None

paras = []
for learning_rate in [7.5e-5]:
    for batch_size in [128]:
        paras.append((learning_rate,batch_size))

for subset in subsets:
    for para in paras:
        hyper['learning_rate'] = para[0]
        hyper['batch_size'] = para[1]

        count += 1
        test_path = '/home/shz/ours/Data/{}/seen/dataset/{}_{}/test_for_all.csv'.format(dataset,NUM_ANOMALY_KNOWS,CONTAMINATION_RATE)
        test_set = pd.read_csv(test_path).values


        data_train_path = '/home/shz/ours/Data/{}/seen/dataset/{}_{}/train.csv'.format(dataset,NUM_ANOMALY_KNOWS,CONTAMINATION_RATE)
        training_set = pd.read_csv(data_train_path).values

        data_val_path = '/home/shz/ours/Data/{}/seen/dataset/{}_{}/val.csv'.format(dataset,NUM_ANOMALY_KNOWS,CONTAMINATION_RATE)
        val_set = pd.read_csv(data_val_path).values

        pr_auc_history = []
        roc_auc_history = []
        for j in range(NUM_RUNS):
            print("#######################################################################")
            # print(f'Running {dataset} {subset} {i}...')
            print(f'Running {subset} {j}...')
            model_id = f'_{subset}_run_{j}'
            
            start_time = time.time()

            env = ADEnv(
                dataset=training_set,
                sampling_Du=hyper['sampling_du'],
                prob_1=prob1,
                prob_2=prob2,
                label_normal=LABEL_NORMAL,
                label_anomaly=LABEL_ANOMALY
            )

            dplan = DPLAN(
                env=env,
                test_set=test_set,
                val_set=val_set,
                destination_path=MODELS_PATH,
                device = device,
                double_dqn=True
            )

            data,best_para = dplan.fit(reset_nets = True)

            # compute train time
            end_time = time.time()
            run_time = start_time - end_time
            print("run_time:",run_time)

            df = pd.DataFrame(data)
            output_path1=os.path.join(PROCESS_PATH,"{}_{}.csv".format(subset,j))
            df.to_csv(output_path1,index=False)

            if hyper["num_anomaly_knows"] <= 5: 
                dplan.policy_net.load_state_dict(best_para)
            # 对全部的异常进行预测
            roc,pr = dplan.model_performance_all()
            print(f'Finished run {j} with pr: {pr} and auc-roc: {roc}...')
            pr_auc_history.append(pr)
            roc_auc_history.append(roc)

            destination_filename = model_id + '.pth'
            dplan.save_model(destination_filename)
            print()
            print('--------------------------------------------------\n')

            
        print('--------------------------------------------------\n')
        write_results(pr_auc_history,roc_auc_history,subset,results_filename,para)
        if np.mean(np.array(pr_auc_history)) > best_pr:
            best_pr = np.mean(np.array(pr_auc_history))
            best_para = para
    write_dataset_best_para(best_para_filename,subset,best_para)

