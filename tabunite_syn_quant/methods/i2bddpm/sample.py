import torch
import numpy as np
import pandas as pd
import os
import json
import time

from dataset import OnlineToyDataset
from methods.i2bddpm.models.gaussian_multinomial_distribution import GaussianMultinomialDiffusion
from methods.i2bddpm.models.modules import MLPDiffusion

import src

def bits_needed(categories):
    return np.ceil(np.log2(categories)).astype(int)

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
    syn_cat = torch.tensor(syn_cat, dtype=torch.uint8)

    # print("syn num: " + str(syn_num))
    # print("syn cat: " + str(syn_cat[0:10]))

    syn_num = num_inverse(syn_num).astype(np.float32)
    syn_cat = cat_inverse(syn_cat)
    # syn_cat = syn_cat.type(torch.int8)


    if info['task_type'] == 'regression':
        syn_target = syn_num[:, :len(target_col_idx)]
        syn_num = syn_num[:, len(target_col_idx):]
    
    else:
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

def to_good_ohe(ohe, X):
    indices = np.cumsum([0] + ohe._n_features_outs)
    Xres = []
    for i in range(1, len(indices)):
        x_ = np.max(X[:, indices[i - 1]:indices[i]], axis=1)
        t = X[:, indices[i - 1]:indices[i]] - x_.reshape(-1, 1)
        Xres.append(np.where(t >= 0, 1, 0))
    return np.hstack(Xres)

def sample(
    model_save_path,
    dataname,
    batch_size = 2000,
    num_samples = 0,
    model_type = 'mlp',
    model_params = None,
    num_timesteps = 1000,
    gaussian_loss_type = 'mse',
    scheduler = 'cosine',
    num_numerical_features = 0,
    device = torch.device('cuda:0'),
    ddim = False,
    steps = 1000,
):

    dataset = OnlineToyDataset(dataname)

    K = np.array(dataset.get_category_sizes())
    num_numerical_features = dataset.get_numerical_sizes()
    num_bits_per_cat_feature = bits_needed(K) if len(K) > 0 else np.array([0])
    d_in = np.sum(num_bits_per_cat_feature) + num_numerical_features
    model_params['d_in'] = d_in

    model = get_model(
        model_type,
        model_params,
        num_numerical_features,
        category_sizes=dataset.get_category_sizes()
    )
   
    model_path =f'{model_save_path}/model.pt'

    model.load_state_dict(
        torch.load(model_path, map_location="cpu")
    )

    num_samples = 20000
    for num_timesteps in [2, 10, 100, 500]:
        
        diffusion = GaussianMultinomialDiffusion(
            K,
            num_numerical_features=num_numerical_features,
            denoise_fn=model, num_timesteps=num_timesteps, 
            gaussian_loss_type=gaussian_loss_type, scheduler=scheduler, device=device,
            num_bits_per_cat_feature=num_bits_per_cat_feature
        )

        diffusion.to(device)
        diffusion.eval()

        if not ddim:
            x_gen = diffusion.sample_all(num_samples, batch_size, ddim=False, steps = num_timesteps)
        else:
            x_gen = diffusion.sample_all(num_samples, batch_size, ddim=True)
            
        x_gen = x_gen.numpy()
        