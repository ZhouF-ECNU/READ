import torch
from sklearn.metrics import roc_auc_score, average_precision_score, confusion_matrix,roc_curve
from sklearn.metrics import precision_score, recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import PrecisionRecallDisplay
import matplotlib.pyplot as plt
import numpy as np


hyper = {
    'n_episodes': 2,
    'steps_per_episode' : 2000,
    'max_memory' : 100_000, 
    'eps_max' : 1,
    'eps_min' : 0.1,
    'eps_decay' : 5000,
    'hidden_size1' :128,
    'hidden_size2' :10,
    'learning_rate' : 7.5e-5,
    'momentum' : 0.95,
    'min_squared_gradient' : 0.005,
    'warmup_steps' : 200,
    'gamma' : 0.9,
    'batch_size' : 128,
    'target_update' : 400,
    'theta_update' : 100,
    'num_anomaly_knows' : 10,
    'contamination_rate' : 0.05,
    'runs' : 5, 
    'S_size' : 2000,
    'sampling_du' : 1000, 
    'prob_au' : 0.4,
    'validation_frequency' : 100,
    'weight_decay' : 1e-3,
    'val_record' : 10,
    'clip_grad' : 125,
    'soft_update' : 0.9,
}


def plot_roc_pr(test_set,policy_net):
    test_X, test_y=test_set[:,:-1], test_set[:,-1]
    pred_y=policy_net(test_X).detach().numpy()[:,1]
    fpr, tpr, _ = roc_curve(test_y, pred_y)
    plt.plot(fpr, tpr)
    plt.show()

    display = PrecisionRecallDisplay.from_predictions(test_y, pred_y, name="DQN")
    _ = display.ax_.set_title("2-class Precision-Recall curve")

   
def test_model(test_set,policy_net):
    policy_net.eval()
    test_X, test_y=test_set[:,:-1], test_set[:,-1]
    with torch.no_grad():
        pred_y=policy_net(test_X).detach().cpu().numpy()[:,1]

    policy_net.train()
    return get_roc_pr(test_y, pred_y)

def get_roc_pr(test_y, pred_y):
    roc = roc_auc_score(test_y, pred_y)
    pr = average_precision_score(test_y, pred_y)
    return roc,pr

def get_precision_recall(test_y, pred_y):
    # roc = roc_auc_score(test_y, pred_y)
    # pr = average_precision_score(test_y, pred_y)
    precision = precision_score(test_y, pred_y)
    recall = recall_score(test_y, pred_y)
    return recall,precision

def get_F1(test_y, pred_y):
    f1 = f1_score(test_y,pred_y)
    return f1

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def write_results(pr_auc_history,roc_auc_history,dataset,path,para):
    pr_auc_history = np.array(pr_auc_history)
    roc_auc_history = np.array(roc_auc_history)
    pr_mean = np.mean(pr_auc_history)
    auc_mean = np.mean(roc_auc_history)

    pr_max = np.max(pr_auc_history)
    auc_max = np.max(roc_auc_history)
    pr_min = np.min(pr_auc_history)
    auc_min = np.min(roc_auc_history)

    pr_std = np.std(pr_auc_history)
    auc_std = np.std(roc_auc_history)
    line = f'{dataset},{para},{pr_mean},{pr_std},{auc_mean},{auc_std},{pr_max},{auc_max},{pr_min},{auc_min}\n'
    
    with open(path, 'a') as f:
        f.write(line)

def write_dataset_best_para(path,dataset,para):
    line = f'{dataset},{para}\n'
    with open(path, 'a') as f:
        f.write(line)

def write_reward(path,r_i,r_e):
    with open(path, 'a') as f:
        f.write(f'{r_i},{r_e},')








