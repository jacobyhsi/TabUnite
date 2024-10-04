import os
import argparse
from methods.i2bddpm.sample import sample

import src


def main(args):
    dataname = args.dataname
    device = f'cuda:{args.gpu}'

    curr_dir = os.path.dirname(os.path.abspath(__file__))
    config_path = f'{curr_dir}/configs/{dataname}.toml'
    model_save_path = f'{curr_dir}/ckpt/{dataname}'
    real_data_path = f'data/{dataname}'
    sample_save_path = args.save_path

    args.train = True
    
    raw_config = src.load_config(config_path)

    ''' 
    Modification of configs
    '''
    print('START SAMPLING')
    
    sample(
        **raw_config['diffusion_params'],
        model_save_path=model_save_path,
        dataname=dataname,
        model_type=raw_config['model_type'],
        model_params=raw_config['model_params'],
        device=device,
        num_samples=raw_config['sample']['num_samples'],
        batch_size=raw_config['sample']['batch_size'],
        num_numerical_features=raw_config['num_numerical_features']
    )

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--dataname', type = str, default = 'olympic')
    parser.add_argument('--gpu', type = int, default=0)
    parser.add_argument('--ddim', action = 'store_true', default = False, help='Whether to use ddim sampling.')
    parser.add_argument('--steps', type=int, default = 1000)

    args = parser.parse_args()