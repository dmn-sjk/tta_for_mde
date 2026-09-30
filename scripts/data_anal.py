import numpy as np
import json
import os
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats
from matplotlib.ticker import FormatStrFormatter


sns.set_theme()
sns.set_context("paper")


path = 'playground_new/single_pairs_experiment/waymo_sunny_Dawn_Dusk/batch_idx_0/data.npy'
EXPERIMENTS_PATH = os.path.join('playground_new', 'single_pairs_experiment_ambMask_autoblur')
# EXPERIMENTS_PATH = os.path.join('playground_new', 'single_stereo_experiment_gtT_ambMask_autoblur')
# EXPERIMENTS_PATH = os.path.join('playground_new', 'single_pairs_experiment_ambMask_autoblur')
SAVE = True
FONT_SIZE = 25
BIGGER_FONT_SIZE = FONT_SIZE + 6


def load_data(path):
    return np.load(path, allow_pickle=True)[()]

def get_folders_in_dir(path, substring=''):
    return [d for d in os.listdir(path) if substring in d]

def scatter_loss_vs_acc(data, loss_key='unsup_loss', save=False, save_dir=''):
    fig, ax = plt.subplots(figsize=(11, 6))
    # ax.yaxis.set_major_formatter(FormatStrFormatter('%.3f'))
    
    ax.scatter(data[loss_key], data['rmse'])
    
    plt.yticks(fontsize=FONT_SIZE)
    plt.xticks(fontsize=FONT_SIZE)
    
    plt.grid(axis='both', linestyle='-', linewidth=2)
    # plt.legend(loc='best', fontsize=BIGGER_FONT_SIZE)
    plt.xlabel("Photometric loss", fontsize=BIGGER_FONT_SIZE)
    plt.ylabel("RMSE [m]", fontsize=BIGGER_FONT_SIZE)
    plt.tight_layout()

    if save:
        plt.savefig(os.path.join(save_dir, loss_key + '_vs_rmse.png'))

def scatter_image_features_vs(dataset_data, vs='unsup_loss', save=False, save_dir=''):
    
    vs_data = []
    img_feat_data ={
        'img_entropy': [],
        'img_density': []
    }
    for batch_name, batch_vals in dataset_data.items():
        vs_data.extend(batch_vals[vs])
        for img_feat in img_feat_data.keys():
            img_feat_data[img_feat].extend([batch_vals[img_feat][0][1]] * len(batch_vals[vs]))
    
    for img_feat, feat_data in img_feat_data.items():
        fig, ax = plt.subplots(figsize=(11, 6))
        ax.scatter(feat_data, vs_data)
        
        plt.yticks(fontsize=FONT_SIZE)
        plt.xticks(fontsize=FONT_SIZE)
        
        plt.grid(axis='both', linestyle='-', linewidth=2)
        # plt.legend(loc='best', fontsize=BIGGER_FONT_SIZE)
        plt.xlabel(img_feat.upper().replace('_', ' '), fontsize=BIGGER_FONT_SIZE)
        plt.ylabel(vs.upper(), fontsize=BIGGER_FONT_SIZE)
        plt.tight_layout()

        if save:
            plt.savefig(os.path.join(save_dir, f'{vs}_vs_{img_feat}.png'))

def plot_batch(data, save=False, save_dir=''):
    scatter_loss_vs_acc(data, loss_key='unsup_loss', save=save, save_dir=save_dir)
    if 'unsup_loss_wo_identity' in data.keys():
        scatter_loss_vs_acc(data, loss_key='unsup_loss_wo_identity', save=save, save_dir=save_dir)
    if 'unsup_loss_wo_identity_&_smooth' in data.keys():
        scatter_loss_vs_acc(data, loss_key='unsup_loss_wo_identity_&_smooth', save=save, save_dir=save_dir)

def plot_dataset(dataset_data, save=False, save_dir=''):
    scatter_image_features_vs(dataset_data, vs='unsup_loss', save=save, save_dir=save_dir)
    if 'unsup_loss_wo_identity' in dataset_data.keys():
        scatter_image_features_vs(dataset_data, vs='unsup_loss_wo_identity', save=save, save_dir=save_dir)
    if 'unsup_loss_wo_identity_&_smooth' in dataset_data.keys():
        scatter_image_features_vs(dataset_data, vs='unsup_loss_wo_identity_&_smooth', save=save, save_dir=save_dir)
    scatter_image_features_vs(dataset_data, vs='rmse', save=save, save_dir=save_dir)
    
def get_correlation(data, loss_key='unsup_loss'):
    return stats.pearsonr(data['rmse'], data[loss_key])

def save_json(data, save_path):
    with open(save_path, 'w') as f:
        json.dump(data, f, indent=4)


def main():
    datasets = get_folders_in_dir(EXPERIMENTS_PATH)
    for dataset in datasets:
        batches = get_folders_in_dir(os.path.join(EXPERIMENTS_PATH, dataset), substring='batch_idx')
        dataset_data = {}
        correlations = {
            'correlation': [],
            'p_value': [],
        }
        correlations_wo = {
            'correlation': [],
            'p_value': [],
        }
        for batch in batches:
            batch_path = os.path.join(EXPERIMENTS_PATH, dataset, batch)
            save_dir = os.path.join(batch_path, 'plots')
            os.makedirs(save_dir, exist_ok=True)
            
            data = load_data(os.path.join(batch_path, 'data.npy'))
            dataset_data[batch] = data
            plot_batch(data, save=SAVE, save_dir=save_dir)
            
            corr, p_val = get_correlation(data, loss_key='unsup_loss')
            corr_wo, p_val_wo = get_correlation(data, loss_key='unsup_loss_wo_identity')
            correlations['correlation'].append(corr)
            correlations['p_value'].append(p_val)
            correlations_wo['correlation'].append(corr_wo)
            correlations_wo['p_value'].append(p_val_wo)
            save_json({'correlation': corr, 'p_value': p_val}, os.path.join(batch_path, 'unsup_loss_correlation.json'))
            save_json({'correlation': corr_wo, 'p_value': p_val_wo}, os.path.join(batch_path, 'unsup_loss_wo_identity_correlation.json'))
            
            print(save_dir)
            # break
        
        save_dir = os.path.join(EXPERIMENTS_PATH, dataset, 'plots')
        os.makedirs(save_dir, exist_ok=True)
        # plot_dataset(dataset_data, save=SAVE, save_dir=save_dir)
        
        for name, corrs in {'unsup_loss_correlation': correlations, 
                            'unsup_loss_wo_identity_correlation': correlations_wo}.items():
            js_save = {
                'correlations': [float(c) for c in corrs['correlation']],
                'correlation_stats': {
                    'mean': float(np.mean(corrs['correlation'])),
                    'std': float(np.mean(corrs['correlation'])),
                    'median': float(np.mean(corrs['correlation'])),
                },
                'p_values': [float(p) for p in corrs['p_value']],
                'p_value_stats': {
                    'min': float(np.min(corrs['p_value'])),
                    'max': float(np.max(corrs['p_value'])),
                    'median': float(np.median(corrs['p_value'])),
                    'percent_significant': float(sum(p < 0.05 for p in corrs['p_value']) / len(corrs['p_value']) * 100),
                }
                
            }
            save_json(js_save, os.path.join(EXPERIMENTS_PATH, dataset, f'{name}.json'))

        # break

def test(data):
    for key, vals in data.items():
        print(key)
        print(len(vals))
        print(vals[:10])
        print('--'*50)

if __name__ == "__main__":
    main()
    # test(load_data(path))