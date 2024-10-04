import time
import json
import torch
import numpy as np
import pandas as pd

from dataset import OnlineToyDataset, plot_olympic_example
from methods.oheflow.models.flow_matching import sample as cfm_sampler
from methods.oheflow.models.modules import MLPDiffusion, Model
from methods.oheflow.models.flow_matching import ConditionalFlowMatcher

def get_model(
    model_name,
    model_params,
    n_num_features,
    category_sizes
): 
    print(model_name)
    if model_name == 'mlp':
        model = MLPDiffusion(**model_params)
    else:
        raise "Unknown model!"
    return model

@torch.no_grad()
def split_num_cat_target(syn_data, info, num_inverse, cat_inverse):
    task_type = info['task_type']

    num_col_idx = info['num_col_idx']
    cat_col_idx = info['cat_col_idx']
    target_col_idx = info['target_col_idx']

    n_num_feat = len(num_col_idx)
    n_cat_feat = len(cat_col_idx)

    if task_type == 'regression':
        n_num_feat += len(target_col_idx)
    else:
        n_cat_feat += len(target_col_idx)

    syn_num = syn_data[:, :n_num_feat]
    syn_cat = syn_data[:, n_num_feat:]

    syn_num = num_inverse(syn_num).astype(np.float32)
    syn_cat = cat_inverse(syn_cat)
    
    if info['task_type'] == 'regression':
        syn_target = syn_num[:, :len(target_col_idx)]
        syn_num = syn_num[:, len(target_col_idx):]
    
    else:
        print(syn_cat.shape)
        syn_target = syn_cat[:, :len(target_col_idx)]
        syn_cat = syn_cat[:, len(target_col_idx):]

    return syn_num, syn_cat, syn_target

def recover_data(syn_num, syn_cat, syn_target, info):

    num_col_idx = info['num_col_idx']
    cat_col_idx = info['cat_col_idx']
    target_col_idx = info['target_col_idx']


    idx_mapping = info['idx_mapping']
    idx_mapping = {int(key): value for key, value in idx_mapping.items()}

    syn_df = pd.DataFrame()

    if info['task_type'] == 'regression':
        for i in range(len(num_col_idx) + len(cat_col_idx) + len(target_col_idx)):
            if i in set(num_col_idx):
                syn_df[i] = syn_num[:, idx_mapping[i]] 
            elif i in set(cat_col_idx):
                syn_df[i] = syn_cat[:, idx_mapping[i] - len(num_col_idx)]
            else:
                syn_df[i] = syn_target[:, idx_mapping[i] - len(num_col_idx) - len(cat_col_idx)]


    else:
        for i in range(len(num_col_idx) + len(cat_col_idx) + len(target_col_idx)):
            if i in set(num_col_idx):
                syn_df[i] = syn_num[:, idx_mapping[i]]
            elif i in set(cat_col_idx):
                syn_df[i] = syn_cat[:, idx_mapping[i] - len(num_col_idx)]
            else:
                syn_df[i] = syn_target[:, idx_mapping[i] - len(num_col_idx) - len(cat_col_idx)]

    return syn_df


def sample(
    model_save_path,
    sample_save_path,
    real_data_path,
    dataname,
    num_samples = 0,
    model_type = 'mlp',
    model_params = None,
    num_numerical_features = 0,
    device = torch.device('cuda:0'),
):

    dataset = OnlineToyDataset(dataname)

    K = np.array(dataset.get_category_sizes())
    num_numerical_features = dataset.get_numerical_sizes()
    d_in = np.sum(K) + num_numerical_features
    model_params['d_in'] = d_in

    flow_net = get_model(
        model_type,
        model_params,
        num_numerical_features,
        category_sizes=dataset.get_category_sizes()
    )

    model_path =f'{model_save_path}/model.pt'
    flow_net.load_state_dict(
        torch.load(model_path, map_location="cpu")
    )

    cfm = ConditionalFlowMatcher(sigma=0.0, pred_x1=False)
    model = Model(
        flow_net,
        cfm,
        num_numerical_features,
        K
    )
    model.to(device)
    model.eval()

    num_samples = 20000

    for step in [2, 10, 100, 500]:
        x_gen = cfm_sampler(model, num_samples, d_in, N=step, device=device, use_tqdm=True)
        plot_olympic_example(x_gen, f'{model_save_path}/olympic_oheflow_steps={step}.png')