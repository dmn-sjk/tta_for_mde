import os
import json
import matplotlib.pyplot as plt
from typing import Dict
import pandas as pd
import seaborn as sns
import numpy as np
import os


LOGS_DIR = "/net/pr2/projects/plgrid/plgguseby/dmns/ada-depth/exp_logs"

AUG_LABELS = [
    'sim_clr',
    'sim_clr_ratios',
    'sim_clr_ratios_gausNoise',
    'sim_clr_ratios_gausNoise_remGray',
    'sim_clr_ratios_gausNoise_remGray_jitterParams', 
    'sim_clr_ratios_gausNoise_remGray_jitterParams_remRand',
    'Geom.+Photo.',
]

FOLDERS = {
    'waymo_all6': {
        AUG_LABELS[0]: 'consistency_waymo_all6__AUGS_CONFIGsim_clr_LEARNING_RATE0.0003_FRAME_IDS0-1_SCALE_ALIGNMENT',
        AUG_LABELS[1]: 'consistency_waymo_all6__AUGS_CONFIGsim_clr_ratios_LEARNING_RATE0.0003_FRAME_IDS0-1_SCALE_ALIGNMENT',
        AUG_LABELS[2]: 'consistency_waymo_all6__AUGS_CONFIGsim_clr_ratios_gausNoise_LEARNING_RATE0.0003_FRAME_IDS0-1_SCALE_ALIGNMENT',
        AUG_LABELS[3]: 'consistency_waymo_all6__AUGS_CONFIGsim_clr_ratios_gausNoise_remGray_LEARNING_RATE0.0003_FRAME_IDS0-1_SCALE_ALIGNMENT',
        AUG_LABELS[4]: 'consistency_waymo_all6__AUGS_CONFIGsim_clr_ratios_gausNoise_remGray_jitterParams_LEARNING_RATE0.0003_FRAME_IDS0-1_SCALE_ALIGNMENT',
        AUG_LABELS[5]: 'consistency_waymo_all6__AUGS_CONFIGsim_clr_ratios_gausNoise_remGray_jitterParams_remRand_LEARNING_RATE0.0003_FRAME_IDS0-1_SCALE_ALIGNMENT',
        AUG_LABELS[6]: 'consistency_waymo_all6__AUGS_CONFIGjitter_gausBlur_gausNoise_geom_LEARNING_RATE0.0003_FRAME_IDS0-1_SCALE_ALIGNMENT',
    },
    'waymo_sunny_day5': {
        AUG_LABELS[0]: 'consistency_waymo_sunny_day5__AUGS_CONFIGsim_clr_LEARNING_RATE0.0003_FRAME_IDS0-1_SCALE_ALIGNMENT',
        AUG_LABELS[1]: 'consistency_waymo_sunny_day5__AUGS_CONFIGsim_clr_ratios_LEARNING_RATE0.0003_FRAME_IDS0-1_SCALE_ALIGNMENT',
        AUG_LABELS[2]: 'consistency_waymo_sunny_day5__AUGS_CONFIGsim_clr_ratios_gausNoise_LEARNING_RATE0.0003_FRAME_IDS0-1_SCALE_ALIGNMENT',
        AUG_LABELS[3]: 'consistency_waymo_sunny_day5__AUGS_CONFIGsim_clr_ratios_gausNoise_remGray_LEARNING_RATE0.0003_FRAME_IDS0-1_SCALE_ALIGNMENT',
        AUG_LABELS[4]: 'consistency_waymo_sunny_day5__AUGS_CONFIGsim_clr_ratios_gausNoise_remGray_jitterParams_LEARNING_RATE0.0003_FRAME_IDS0-1_SCALE_ALIGNMENT',
        AUG_LABELS[5]: 'consistency_waymo_sunny_day5__AUGS_CONFIGsim_clr_ratios_gausNoise_remGray_jitterParams_remRand_LEARNING_RATE0.0003_FRAME_IDS0-1_SCALE_ALIGNMENT',
        AUG_LABELS[6]: 'consistency_waymo_sunny_day5__AUGS_CONFIGjitter_gausBlur_gausNoise_geom_LEARNING_RATE0.0003_FRAME_IDS0-1_SCALE_ALIGNMENT',
    },
    'waymo_sunny_night5': {
        AUG_LABELS[0]: 'consistency_waymo_sunny_night5__AUGS_CONFIGsim_clr_LEARNING_RATE0.0003_FRAME_IDS0-1_SCALE_ALIGNMENT',
        AUG_LABELS[1]: 'consistency_waymo_sunny_night5__AUGS_CONFIGsim_clr_ratios_LEARNING_RATE0.0003_FRAME_IDS0-1_SCALE_ALIGNMENT',
        AUG_LABELS[2]: 'consistency_waymo_sunny_night5__AUGS_CONFIGsim_clr_ratios_gausNoise_LEARNING_RATE0.0003_FRAME_IDS0-1_SCALE_ALIGNMENT',
        AUG_LABELS[3]: 'consistency_waymo_sunny_night5__AUGS_CONFIGsim_clr_ratios_gausNoise_remGray_LEARNING_RATE0.0003_FRAME_IDS0-1_SCALE_ALIGNMENT',
        AUG_LABELS[4]: 'consistency_waymo_sunny_night5__AUGS_CONFIGsim_clr_ratios_gausNoise_remGray_jitterParams_LEARNING_RATE0.0003_FRAME_IDS0-1_SCALE_ALIGNMENT',
        AUG_LABELS[5]: 'consistency_waymo_sunny_night5__AUGS_CONFIGsim_clr_ratios_gausNoise_remGray_jitterParams_remRand_LEARNING_RATE0.0003_FRAME_IDS0-1_SCALE_ALIGNMENT',
        AUG_LABELS[6]: 'consistency_waymo_sunny_night5__AUGS_CONFIGjitter_gausBlur_gausNoise_geom_LEARNING_RATE0.0003_FRAME_IDS0-1_SCALE_ALIGNMENT',
    },
    'waymo_rainy5': {
        AUG_LABELS[0]: 'consistency_waymo_rainy5__AUGS_CONFIGsim_clr_LEARNING_RATE0.0003_FRAME_IDS0-1_SCALE_ALIGNMENT',
        AUG_LABELS[1]: 'consistency_waymo_rainy5__AUGS_CONFIGsim_clr_ratios_LEARNING_RATE0.0003_FRAME_IDS0-1_SCALE_ALIGNMENT',
        AUG_LABELS[2]: 'consistency_waymo_rainy5__AUGS_CONFIGsim_clr_ratios_gausNoise_LEARNING_RATE0.0003_FRAME_IDS0-1_SCALE_ALIGNMENT',
        AUG_LABELS[3]: 'consistency_waymo_rainy5__AUGS_CONFIGsim_clr_ratios_gausNoise_remGray_LEARNING_RATE0.0003_FRAME_IDS0-1_SCALE_ALIGNMENT',
        AUG_LABELS[4]: 'consistency_waymo_rainy5__AUGS_CONFIGsim_clr_ratios_gausNoise_remGray_jitterParams_LEARNING_RATE0.0003_FRAME_IDS0-1_SCALE_ALIGNMENT',
        AUG_LABELS[5]: 'consistency_waymo_rainy5__AUGS_CONFIGsim_clr_ratios_gausNoise_remGray_jitterParams_remRand_LEARNING_RATE0.0003_FRAME_IDS0-1_SCALE_ALIGNMENT',
        AUG_LABELS[6]: 'consistency_waymo_rainy5__AUGS_CONFIGjitter_gausBlur_gausNoise_geom_LEARNING_RATE0.0003_FRAME_IDS0-1_SCALE_ALIGNMENT',
    },
    'dgp': {
        AUG_LABELS[0]: 'consistency_dgp__AUGS_CONFIGsim_clr_LEARNING_RATE1e-05_FRAME_IDS0-1_SCALE_ALIGNMENT',
        AUG_LABELS[1]: 'consistency_dgp__AUGS_CONFIGsim_clr_ratios_LEARNING_RATE1e-05_FRAME_IDS0-1_SCALE_ALIGNMENT',
        AUG_LABELS[2]: 'consistency_dgp__AUGS_CONFIGsim_clr_ratios_gausNoise_LEARNING_RATE1e-05_FRAME_IDS0-1_SCALE_ALIGNMENT',
        AUG_LABELS[3]: 'consistency_dgp__AUGS_CONFIGsim_clr_ratios_gausNoise_remGray_LEARNING_RATE1e-05_FRAME_IDS0-1_SCALE_ALIGNMENT',
        AUG_LABELS[4]: 'consistency_dgp__AUGS_CONFIGsim_clr_ratios_gausNoise_remGray_jitterParams_LEARNING_RATE1e-05_FRAME_IDS0-1_SCALE_ALIGNMENT',
        AUG_LABELS[5]: 'consistency_dgp__AUGS_CONFIGsim_clr_ratios_gausNoise_remGray_jitterParams_remRand_LEARNING_RATE1e-05_FRAME_IDS0-1_SCALE_ALIGNMENT',
        AUG_LABELS[6]: 'consistency_dgp__AUGS_CONFIGjitter_gausBlur_gausNoise_geom_LEARNING_RATE1e-05_FRAME_IDS0-1_SCALE_ALIGNMENT',
    },
}

METRICS = [
    "abs_rel",
    "sq_rel",
    "rmse",
    "rmse_log",
    "a1",
    "a2",
    "a3",
    ]

def load_result(data_path) -> Dict[str, Dict[str, float]]:
    """
    In the returned dict, mean and std keys contain dict of metric names and their values.  
    """
    with open(data_path) as f:
        data = json.load(f)
    return {
        "mean": data["mean"]["supervised_wo_gt_median_scaling"]["error"],
        "std": data["std"]["supervised_wo_gt_median_scaling"]["error"]
    }

  
def load_data():
    data = {}
    
    for dataset, folder_dict in FOLDERS.items():
        for aug_label, folder in folder_dict.items():
            data_path = os.path.join(LOGS_DIR, folder, 'summary.json')
            if not os.path.exists(data_path):
                print(f"{data_path} NO DATA!")
                continue
        
            data.setdefault(dataset, {})
            data[dataset][aug_label] = load_result(data_path)
    return data

def plot_performance_heatmaps(data, metrics):
    # 1. Flatten the dictionary into a DataFrame
    datasets = list(data.keys())
    aug_configs = data[datasets[0]].keys()
    
    records = []
    for ds in data.keys():
        for aug in data[ds].keys():
            if ds in data and aug in data[ds]:
                entry = data[ds][aug]
                record = {'dataset': ds, 'aug_config': aug}
                for m in metrics:
                    record[f"{m}_mean"] = entry['mean'].get(m, np.nan)
                    record[f"{m}_std"] = entry['std'].get(m, np.nan)
                records.append(record)
    
    df = pd.DataFrame(records)

    for metric in metrics:
        is_accuracy = metric.startswith('a') and len(metric) == 2
        
        # Pivot the values
        mean_pivot = df.pivot(index="aug_config", columns="dataset", values=f"{metric}_mean")
        std_pivot = df.pivot(index="aug_config", columns="dataset", values=f"{metric}_std")
        
        # Reindex to maintain user-specified order
        mean_pivot = mean_pivot.reindex(index=aug_configs, columns=datasets)
        std_pivot = std_pivot.reindex(index=aug_configs, columns=datasets)

        # 2. Add Average Column
        # We calculate the mean across the columns (axis=1) for each augmentation
        mean_pivot['AVERAGE'] = mean_pivot.mean(axis=1)
        std_pivot['AVERAGE'] = std_pivot.mean(axis=1)
        
        # Update dataset list to include the new column for plotting
        plot_cols = datasets + ['AVERAGE']

        # 3. Column-wise Normalization (including the new Average column)
        mean_norm = (mean_pivot - mean_pivot.min()) / (mean_pivot.max() - mean_pivot.min())
        std_norm = (std_pivot - std_pivot.min()) / (std_pivot.max() - std_pivot.min())
        
        mean_norm = mean_norm.fillna(0.5)
        std_norm = std_norm.fillna(0.5)

        # 4. Visualization
        fig, axes = plt.subplots(1, 2, figsize=(22, 8))
        fig.suptitle(f'Metric: {metric.upper()} (Normalized per Column)', fontsize=18, fontweight='bold')

        cmap_mean = "YlGn" if is_accuracy else "YlGn_r"
        
        # Plot Mean Heatmap
        sns.heatmap(mean_norm, annot=mean_pivot, fmt=".4f", cmap=cmap_mean, ax=axes[0],
                    cbar_kws={'label': 'Best performance (Dark Green)'})
        axes[0].set_title(f"Mean {metric} (with Row Average)", fontsize=14)
        
        # Visual separator for the Average column
        axes[0].axvline(len(datasets), color='white', lw=5) 

        # Plot Std Heatmap
        sns.heatmap(std_norm, annot=std_pivot, fmt=".4f", cmap="Purples_r", ax=axes[1],
                    cbar_kws={'label': 'Lowest Variance (Dark Purple)'})
        axes[1].set_title(f"Standard Deviation {metric} (with Row Average)", fontsize=14)
        axes[1].axvline(len(datasets), color='white', lw=5)

        plt.tight_layout(rect=[0, 0.03, 1, 0.95])

        script_name = os.path.splitext(os.path.basename(__file__))[0]
        plot_dir = os.path.join("plots", script_name)
        os.makedirs(plot_dir, exist_ok=True)
        plot_name = f"{metric.upper()}_heatmap"
        save_path = os.path.join(plot_dir, f"{plot_name}.pdf")
        plt.savefig(save_path, bbox_inches='tight')
        save_path = os.path.join(plot_dir, f"{plot_name}.png")
        plt.savefig(save_path, bbox_inches='tight')

if __name__ == "__main__":
    data = load_data()
    # Execute the plotting function
    plot_performance_heatmaps(data, METRICS)