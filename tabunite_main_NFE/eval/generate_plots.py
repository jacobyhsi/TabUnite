import matplotlib.pyplot as plt
import json

    # python main.py --dataname adult --method i2bddpm --mode sample --save_path "synthetic/aucnfe_adult/i2bddpm_${i}.csv" --gpu 0 --steps ${i}
    # python main.py --dataname adult --method i2bflow --mode sample --save_path "synthetic/aucnfe_adult/i2bflow_${i}.csv" --gpu 0 --steps ${i}
    # python main.py --dataname adult --method dicddpm --mode sample --save_path "synthetic/aucnfe_adult/dicddpm_${i}.csv" --gpu 0 --steps ${i}
    # python main.py --dataname adult --method dicflow --mode sample --save_path "synthetic/aucnfe_adult/dicflow_${i}.csv" --gpu 0 --steps ${i}
    # python main.py --dataname adult --method pskddpm --mode sample --save_path "synthetic/aucnfe_adult/pskddpm_${i}.csv" --gpu 0 --steps ${i}
    # python main.py --dataname adult --method pskflow --mode sample --save_path "synthetic/aucnfe_adult/pskflow_${i}.csv" --gpu 0 --steps ${i}
    # python main.py --dataname adult --method oheddpm --mode sample --save_path "synthetic/aucnfe_adult/oheddpm_${i}.csv" --gpu 0 --steps ${i}
    # python main.py --dataname adult --method oheflow --mode sample --save_path "synthetic/aucnfe_adult/oheflow_${i}.csv" --gpu 0 --steps ${i}
    # python main.py --dataname adult --method tabddpm --mode sample --save_path "synthetic/aucnfe_adult/tabddpm_${i}.csv" --gpu 0 --steps ${i}
    # python main.py --dataname adult --method tabflow --mode sample --save_path "synthetic/aucnfe_adult/tabflow_${i}.csv" --gpu 0 --steps ${i}
def summarise_mle_results():
    steps = [2, 3, 4, 5, 6, 7, 8, 10, 12, 14, 16, 20, 24, 28, 32, 40, 48, 56, 64, 80, 96, 112, 128, 160, 192, 224, 256, 320, 384, 448, 512, 640, 768, 896, 1000]
    # methods = ['i2bddpm', 'i2bflow', 'dicddpm', 'dicflow', 'tabsyn']
    methods = ['pskddpm', 'pskflow', 'tabddpm', 'tabddim', 'tabflow']
    method2name = {
        # 'i2bddpm': 'TabUnite(i2b)-DDPM',
        # 'i2bflow': 'TabUnite(i2b)-Flow',
        # 'dicddpm': 'TabUnite(dic)-DDPM',
        # 'dicflow': 'TabUnite(dic)-Flow',
        # 'tabsyn': 'TabSyn'
        'pskddpm': 'TabUnite(psk)-DDPM',
        'pskflow': 'TabUnite(psk)-Flow',
        'tabddpm': 'TabDDPM',
        'tabddim': 'TabDDIM',
        'tabflow': 'TabFlow',
    }

    results = []
    for method in methods:
        aurocs = []
        for step in steps:
            with open(f'{method}_{step}.json') as f:
                scores = json.load(f)
                # if args.dataname == 'beijing' or args.dataname == 'news':
                #     auroc = scores['best_rmse_scores']['XGBRegressor']['RMSE']
                # else:
                auroc = scores['best_auroc_scores']['XGBClassifier']['roc_auc']
                aurocs.append(auroc)
        results.append(aurocs)

    # plot
    for i, method in enumerate(methods):
        plt.plot(range(len(steps)), results[i], label=method2name[method])
    # plt.title(f'Adult', fontsize=13)
    plt.xlabel('NFEs', fontsize=18)
    # plt.ylabel('AUROC/RMSE')
    plt.ylabel('AUC', fontsize=18)
    plt.grid(color='gray', linestyle='--', linewidth=0.5, alpha=0.5)
    plt.legend(framealpha=0.4, frameon=True, fontsize=7, loc='lower right')
    plt.ylim(0.54, 0.96)
    xticks = [0, 2, 6, 10, 14, 18, 22, 26, 30, 34]
    plt.yticks(fontsize=16)
    plt.xticks(xticks, [2, 4, 8, 16, 32, 64, 128, 256, 512, 1000], fontsize=16)
    print('saving fig')
    plt.savefig(f'/voyager/projects/jacobyhsi/TabUnite/tabunite_main/eval/mle/plots/main.pdf', dpi=300, bbox_inches='tight', pad_inches=0.01)
    plt.close()
    
    
def summarise_density_results():
    steps = [2, 3, 4, 5, 6, 7, 8, 10, 12, 14, 16, 20, 24, 28, 32, 40, 48, 56, 64, 80, 96, 112, 128, 160, 192, 224, 256, 320, 384, 448, 512, 640, 768, 896, 1000]
    # methods = ['i2bddpm', 'i2bflow', 'dicddpm', 'dicflow', 'tabsyn']
    methods = ['pskddpm', 'pskflow', 'tabddpm', 'tabddim', 'tabflow']
    method2name = {
        # 'i2bddpm': 'TabUnite(i2b)-DDPM',
        # 'i2bflow': 'TabUnite(i2b)-Flow',
        # 'dicddpm': 'TabUnite(dic)-DDPM',
        # 'dicflow': 'TabUnite(dic)-Flow',
        # 'tabsyn': 'TabSyn'
        'pskddpm': 'TabUnite(psk)-DDPM',
        'pskflow': 'TabUnite(psk)-Flow',
        'tabddpm': 'TabDDPM',
        'tabddim': 'TabDDIM',
        'tabflow': 'TabFlow',
    }

    results = []
    for method in methods:
        errors = []
        for step in steps:
            # with open(f'density/err_adult/{method}_{step}.json.txt') as f:
            with open(f'density/err_adult/{method}_{step}.txt') as f:
                error = float(f.readline().strip())
                errors.append(error)
        results.append(errors)

    # plot
    for i, method in enumerate(methods):
        plt.plot(range(len(steps)), results[i], label=method2name[method])
    plt.xlabel('NFEs', fontsize=18)
    # plt.ylabel('Average Low-Order Statistic Error', fontsize=13)
    plt.ylabel('Average Error', fontsize=16)
    plt.grid(color='gray', linestyle='--', linewidth=0.5, alpha=0.5)
    plt.legend(framealpha=0.4, frameon=True, fontsize=7, loc='center right')
    # plt.legend(framealpha=0.4, frameon=True, fontsize=18)
    xticks = [0, 2, 6, 10, 14, 18, 22, 26, 30, 34]
    plt.yticks(fontsize=16)
    plt.xticks(xticks, [2, 4, 8, 16, 32, 64, 128, 256, 512, 1000], fontsize=16)
    plt.savefig(f'/voyager/projects/jacobyhsi/TabUnite/tabunite_main/eval/density/plots/main.pdf', dpi=300, bbox_inches='tight', pad_inches=0.01)
    plt.close()
    
if __name__ == '__main__':
    # summarise_mle_results()
    summarise_density_results()