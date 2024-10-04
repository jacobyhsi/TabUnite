import torch
import numpy as np
import pandas as pd
import os
import json
import time

from dataset import OnlineToyDataset
from methods.dicddpm.models.gaussian_multinomial_distribution import GaussianMultinomialDiffusion
from methods.dicddpm.models.modules import MLPDiffusion

import src

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

def get_model(
    model_name,
    model_params,
    n_num_features,
    category_sizes,
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
    sample_save_path,
    dataname,
    steps = 1000,
    batch_size = 1024,
    model_type='mlp',
    model_params = None,
    num_timesteps = 1000,
    gaussian_loss_type = 'mse',
    scheduler = 'cosine',
    device=torch.device('cuda:0'),
    num_samples = 0,
    num_numerical_features = 0,
    ddim = False,
):
    emb_dim = 1

    dataset = OnlineToyDataset(dataname)
    K = dataset.get_category_sizes()
    num_numerical_features = dataset.get_numerical_sizes()

    cat_len = 0
    for item in K:
        cat_len += 1 if item <=2 else emb_dim

    d_in = num_numerical_features + cat_len
    model_params['d_in'] = d_in


    model = get_model(
        model_type,
        model_params,
        num_numerical_features,
        category_sizes=K,
    )
   
    model_path =f'{model_save_path}/model.pt'
    model.load_state_dict(
        torch.load(model_path, map_location="cpu")
    )
    

    diffusion = GaussianMultinomialDiffusion(
        np.array(K),
        num_numerical_features=num_numerical_features,
        denoise_fn=model, num_timesteps=num_timesteps, 
        gaussian_loss_type=gaussian_loss_type, scheduler=scheduler, device=device,
        categories=K,
        emb_dim=emb_dim,
        d_in=d_in
    )

    diffusion.to(device)
    diffusion.eval()
    
    num_samples = 20000
    num_timesteps = 1000

    if not ddim:
        x_gen = diffusion.sample_all(num_samples, batch_size, ddim=False, steps = num_timesteps)
    else:
        x_gen = diffusion.sample_all(num_samples, batch_size, ddim=True)
        
    x_gen = x_gen.numpy()
    acc = dataset.evaluate(x_gen)
    print(f'Accuracy: {acc}')
    