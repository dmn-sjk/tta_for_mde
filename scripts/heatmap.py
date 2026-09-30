
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd

metric_name_to_idx = {
            'AbsRel': 0,
            'SqRel': 1,
            'RMSE': 2,
            'RSMElog': 3,
            'sigma1': 4,
            'sigma2': 5,
            'sigma3': 6
}

idx_to_metric_name = [
            'AbsRel',
            'SqRel',
            'RMSE',
            'RSMElog',
            'sigma1',
            'sigma2',
            'sigma3'
]

results = {
    'DDAD': {
        'Source':[
            0.144, 1.516, 7.950, 0.228, 0.788, 0.938, 0.976
        ],
        'Photo.':[
            0.135, 1.280, 6.844, 0.199, 0.837, 0.958, 0.984
        ],
        'Geom.':[
            0.143, 1.394, 7.526, 0.215, 0.795, 0.949, 0.983

        ],
        'Geom.+Photo.':[
            0.142, 1.310, 6.983, 0.204, 0.812, 0.957, 0.985
        ],
        'IM':[
            0.141, 1.389, 7.462, 0.216, 0.806, 0.948, 0.981

        ],
        'SRP':[
            0.139, 1.330, 7.113, 0.208, 0.821, 0.953, 0.983

        ],
        'SimCLR':[
            0.142, 1.404, 7.549, 0.212, 0.799, 0.945, 0.983

        ],
    },
    'All-6': {
        'Source':[
                    0.260, 4.765, 12.130, 0.457, 0.580, 0.762, 0.831

        ],
        'Photo.':[
                    0.208, 3.167, 9.046, 0.267, 0.691, 0.919, 0.961

        ],
        'Geom.':[
                    0.208, 3.107, 10.104, 0.320, 0.662, 0.859, 0.918

        ],
        'Geom.+Photo.':[
                    0.190, 2.600, 9.170, 0.277, 0.724, 0.896, 0.947

        ],
        'IM':[
                    0.210, 2.925, 9.522, 0.314, 0.643, 0.880, 0.935

        ],
        'SRP':[
                    0.193, 2.267, 8.194, 0.256, 0.672, 0.924, 0.969

        ],
        'SimCLR':[
                    0.193, 2.569, 8.075, 0.241, 0.722, 0.936, 0.972

        ],
    },
    'Rainy-5': {
        'Source':[
                    0.244, 3.439, 10.021, 0.374, 0.601, 0.815, 0.890

        ],
        'Photo.':[
                    0.239, 2.991, 9.164, 0.311, 0.572, 0.855, 0.939

        ],
        'Geom.':[
                    0.213, 2.479, 8.351, 0.286, 0.685, 0.878, 0.951

        ],
        'Geom.+Photo.':[
                    0.216, 2.698, 8.121, 0.269, 0.712, 0.880, 0.945

        ],
        'IM':[
                    0.234, 2.680, 8.502, 0.285, 0.618, 0.882, 0.948

        ],
        'SRP':[
                    0.255, 3.438, 8.775, 0.293, 0.624, 0.875, 0.936

        ],
        'SimCLR':[
                    0.250, 3.166, 9.011, 0.296, 0.608, 0.873, 0.940

        ],
    },
    'Sunny-Day-5': {
        'Source':[
                    0.162, 2.076, 7.346, 0.211, 0.826, 0.949, 0.977

        ],
        'Photo.':[
                    0.187, 2.346, 7.416, 0.229, 0.772, 0.943, 0.975

        ],
        'Geom.':[
                    0.165, 2.034, 7.093, 0.209, 0.828, 0.949, 0.977

        ],
        'Geom.+Photo.':[
                    0.165, 2.083, 7.206, 0.209, 0.825, 0.949, 0.978

        ],
        'IM':[
                    0.173, 2.215, 7.290, 0.217, 0.814, 0.946, 0.976

        ],
        'SRP':[
                    0.179, 2.337, 7.480, 0.220, 0.808, 0.944, 0.975

        ],
        'SimCLR':[
                    0.196, 2.447, 7.878, 0.234, 0.755, 0.938, 0.977

        ],
    },
    'Sunny-Night-5': {
        'Source':[
                    0.199, 2.116, 9.558, 0.272, 0.607, 0.903, 0.972

        ],
        'Photo.':[
                    0.198, 1.870, 8.429, 0.245, 0.620, 0.939, 0.986

        ],
        'Geom.':[
                    0.186, 1.791, 8.525, 0.239, 0.668, 0.939, 0.986

        ],
        'Geom.+Photo.':[
                    0.185, 1.895, 8.803, 0.241, 0.680, 0.926, 0.983

        ],
        'IM':[
                    0.192, 1.811, 8.759, 0.249, 0.606, 0.935, 0.985

        ],
        'SRP':[
                    0.197, 1.838, 8.777, 0.252, 0.581, 0.936, 0.985

        ],
        'SimCLR':[
                    0.200, 1.990, 8.922, 0.252, 0.615, 0.928, 0.984

        ],
    },
}

metric_labels = {
    'RMSE': r'RMSE$\downarrow$',
    'AbsRel': r'AbsRel$\downarrow$',
    'SqRel': r'SqRel$\downarrow$',
    'RSMElog': r'RMSElog$\downarrow$',
    'sigma1': '$\\sigma < 1.25 \\uparrow$',
    'sigma2': '$\\sigma < 1.25^2 \\uparrow$',
    'sigma3': '$\\sigma < 1.25^3 \\uparrow$',
}

import matplotlib
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42

def heatmaps():
    FONTSIZE = 15
    BIGGER_FONTSIZE = FONTSIZE + 2
    TILE_FONTSIZE = 16  # Font size for values inside tiles
    TICK_FONTSIZE = 16  # Font size for axis ticks
    LEGEND_FONTSIZE = 16  # Font size for colorbar legend
    
    # Get all datasets and methods
    datasets = list(results.keys())
    methods = list(results[datasets[0]].keys())
    
    # Create a figure with subplots for each metric
    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    axes = axes.flatten()
    
    # Create heatmap for each metric
    for metric_idx, metric_name in enumerate(idx_to_metric_name):
        ax = axes[metric_idx]
        
        # Create a matrix for this metric
        metric_data = []
        for dataset in datasets:
            row = []
            for method in methods:
                value = results[dataset][method][metric_idx]
                row.append(value)
            metric_data.append(row)
        
        # Convert to DataFrame for better labeling - swap rows and columns
        df = pd.DataFrame(metric_data, 
                         index=datasets, 
                         columns=methods)
        
        # Transpose the DataFrame to swap x and y axes
        df = df.T
        
        # Choose colormap based on metric type
        if metric_name.startswith('sigma'):
            cmap = 'summer_r'  # Reversed summer for sigma metrics
        else:
            cmap = 'summer'    # Regular summer for other metrics
        
        # Create heatmap
        sns.heatmap(df, 
                   annot=True, 
                   fmt='.3f', 
                   cmap=cmap,
                   ax=ax,
                   annot_kws={'size': TILE_FONTSIZE},  # Font size for values inside tiles
                   cbar_kws={'shrink': 0.8})
        
        ax.set_title(f'{metric_name}', fontsize=FONTSIZE, fontweight='bold')
        ax.set_xlabel('Datasets')  # Now datasets are on x-axis
        ax.set_ylabel('Methods')   # Now methods are on y-axis
        
        # Rotate x-axis labels for better readability
        ax.tick_params(axis='x', rotation=45, labelsize=TICK_FONTSIZE)
        ax.tick_params(axis='y', rotation=0, labelsize=TICK_FONTSIZE)
    
    # Remove the last empty subplot
    fig.delaxes(axes[7])
    
    plt.tight_layout()
    plt.savefig(f'plots/heatmaps.pdf', dpi=300, bbox_inches='tight')
    
    # Also create individual heatmaps for better visibility
    for metric_idx, metric_name in enumerate(idx_to_metric_name):
        plt.figure(figsize=(10, 6))
        
        # Create a matrix for this metric
        metric_data = []
        for dataset in datasets:
            row = []
            for method in methods:
                value = results[dataset][method][metric_idx]
                row.append(value)
            metric_data.append(row)
        
        # Convert to DataFrame
        df = pd.DataFrame(metric_data, 
                         index=datasets, 
                         columns=methods)
        
        # Transpose the DataFrame to swap x and y axes
        df = df.T
        
        # Choose colormap based on metric type
        if metric_name.startswith('sigma'):
            cmap = 'summer_r'  # Reversed summer for sigma metrics
        else:
            cmap = 'summer'    # Regular summer for other metrics
        
        # Create heatmap
        sns.heatmap(df, 
                   annot=True, 
                   fmt='.3f', 
                   cmap=cmap,
                   annot_kws={'size': TILE_FONTSIZE},  # Font size for values inside tiles
                   cbar_kws={'shrink': 0.8})
        
        plt.title(metric_labels[metric_name], 
                 fontsize=BIGGER_FONTSIZE, fontweight='bold')
        plt.xlabel('Benchmarks', fontsize=FONTSIZE)  # Now datasets are on x-axis
        plt.ylabel('Augmentation Methods', fontsize=FONTSIZE)  # Now methods are on y-axis
        
        # Rotate x-axis labels and set font size
        plt.xticks(rotation=45, fontsize=TICK_FONTSIZE)
        plt.yticks(rotation=0, fontsize=TICK_FONTSIZE)
        
        # Set colorbar font size
        cbar = plt.gcf().get_axes()[-1]  # Get the colorbar axis
        cbar.tick_params(labelsize=LEGEND_FONTSIZE)
        
        plt.tight_layout()   
        plt.savefig(f'plots/heatmaps_{metric_name}.pdf', dpi=300, bbox_inches='tight')
        # plt.show()


if __name__ == "__main__":
    heatmaps()