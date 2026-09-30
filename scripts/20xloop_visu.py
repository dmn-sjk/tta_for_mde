import matplotlib.pyplot as plt
import os

all6_wo_gt_scaling = {
    'RMSE': {
        'Source': [
            12.130,
            12.130,
            12.130,
            12.130,
            12.130,
            12.130,
            12.130,
            12.130,
            12.130,
            12.130,
            12.130,
        ],
        'AdaDepth': [
            10.633,
            10.469,
            10.301,
            10.225,
            10.195,
            10.159,
            10.136,
            10.103,
            10.084,
            10.075,
            10.076,
            10.086,
            10.096,
            10.104,
            10.111,
            10.124,
            10.136,
            10.150,
            10.156,
            10.165
            
        ],
        'MAE': [
            9.339,
            9.002,
            8.943,
            10.152,
            9.986,
            9.896,
            9.737,
            9.628,
            9.549,
            9.527,
            9.444
        ],
        'MAE_noStochasticReset': [
            9.404,
            9.297,
            9.267,
            9.246,
            9.233,
            9.226,
            9.220,
            9.215,
            9.208,
            9.201,
            9.194
        ],
        'MAE_frozenDecoder_protoReg': [
            9.152,
            9.270,
            9.369,
            9.498,
            9.773,
            9.870,
            9.855,
            9.801,
            9.697,
            9.688,
            9.684
        ],
        'MAE_final': [
            10.085,
            10.162,
            10.149,
            10.136,
            10.111,
            10.093,
            10.077,
            10.063,
            10.049,
            10.035,
            10.023,
            10.011,
            9.999,
            9.987,
            9.977,
            9.969,
            9.961,
            9.954,
            9.947,
            9.941,
        ],
        'MAE_final_10xLR': [
            23.393,
            24.345,
            24.688,
            24.838,
            24.904,
            24.943,
            24.975,
            25.004,
            25.032,
            25.060,
            25.084,
            25.107,
            25.127,
            25.147,
            25.166,
            25.188,
            25.210,
            25.230,
            25.248,
            25.263
        ],
        'MAE_final_5xLR': [
            10.898,
            11.276,
            11.338,
            11.367,
            11.393,
            11.415,
            11.434,
            11.448,
            11.456,
            11.460,
            11.462,
            11.460,
            11.458,
            11.456,
            11.454,
            11.452,
            11.451,
            11.450,
            11.449,
            11.445
        ],
        'AC': [
            9.004,
            8.785,
            8.766,
            8.746,
            8.779,
            8.836,
            8.885,
            8.935,
            8.987,
            9.040,
            9.099,
            9.155,
            9.232,
            9.313,
            9.390,
            9.457,
            9.516,
            9.572,
            9.622,
            9.668
        ]
    },
    'Median Ratio': {
        'Source': [
            1.427,
            1.427,
            1.427,
            1.427,
            1.427,
            1.427,
            1.427,
            1.427,
            1.427,
            1.427,
            1.427,
        ],
        'AdaDepth': [
            1.229,
            1.233,
            1.230,
            1.228,
            1.229,
            1.228,
            1.227,
            1.226,
            1.226,
            1.225,
            1.227,
            1.228,
            1.231,
            1.233,
            1.235,
            1.237,
            1.239,
            1.240,
            1.242,
            1.243
        ],
        'MAE': [
            1.115,
            1.102,
            1.087,
            1.138,
            1.114,
            1.093,
            1.072,
            1.056,
            1.044,
            1.038,
            1.027
        ],
        'MAE_noStochasticReset': [
            1.052,
            1.053,
            1.064,
            1.069,
            1.073,
            1.075,
            1.077,
            1.078,
            1.079,
            1.079,
            1.079
        ],
        'MAE_frozenDecoder_protoReg': [
            1.105,
            1.104,
            1.110,
            1.113,
            1.124,
            1.125,
            1.121,
            1.116,
            1.110,
            1.108,
            1.107 
        ],
        'MAE_final': [
            1.120,
            1.168,
            1.185,
            1.196,
            1.203,
            1.208,
            1.211,
            1.214,
            1.216,
            1.218,
            1.219,
            1.221,
            1.221,
            1.222,
            1.222,
            1.222,
            1.223,
            1.223,
            1.223,
            1.224
        ],
        'MAE_final_10xLR': [
            3.736,
            3.976,
            4.085,
            4.115,
            4.104,
            4.086,
            4.076,
            4.074,
            4.077,
            4.082,
            4.089,
            4.094,
            4.101,
            4.110,
            4.121,
            4.140,
            4.158,
            4.175,
            4.190,
            4.203
        ],
        'MAE_final_5xLR': [
            1.127,
            1.144,
            1.149,
            1.151,
            1.153,
            1.154,
            1.156,
            1.157,
            1.157,
            1.157,
            1.157,
            1.157,
            1.157,
            1.157,
            1.157,
            1.156,
            1.156,
            1.156,
            1.156,
            1.156
        ],
        'AC': [
            1.079,
            1.058,
            1.050,
            1.045,
            1.043,
            1.044,
            1.044,
            1.044,
            1.045,
            1.046,
            1.047,
            1.050,
            1.053,
            1.057,
            1.061,
            1.064,
            1.068,
            1.071,
            1.073,
            1.076
        ]
    },
}


sunny_day_5_wo_gt_scaling = {
    'RMSE': {
        'MAE_final': [
            7.458,
            7.615,
            7.756,
            7.867,
            7.958,
            8.047,
            8.136,
            8.219,
            8.297,
            8.364,
            8.433,
            8.507,
            8.579,
            8.648,
            8.718,
            8.792,
            8.874,
            8.964,
            9.054,
            9.147
        ],
        'AdaDepth': [
            7.222,
            7.331,
            7.516,
            7.690,
            7.865,
            8.001,
            8.100,
            8.162,
            8.196,
            8.213,
            8.222,
            8.224,
            8.217,
            8.202,
            8.181,
            8.158,
            8.134,
            8.109,
            8.088,
            8.069
        ],
    },
    'Median Ratio': {
        'MAE_final': [
            0.942,
            0.944,
            0.946,
            0.946,
            0.947,
            0.948,
            0.948,
            0.948,
            0.949,
            0.950,
            0.950,
            0.951,
            0.952,
            0.952,
            0.953,
            0.954,
            0.955,
            0.956,
            0.956,
            0.957
        ],
        'AdaDepth': [
            0.957,
            0.949,
            0.939,
            0.932,
            0.925,
            0.920,
            0.917,
            0.914,
            0.913,
            0.911,
            0.911,
            0.911,
            0.911,
            0.911,
            0.912,
            0.914,
            0.916,
            0.917,
            0.919,
            0.919
        ],
    },
}



# # Create plots directory if it doesn't exist
# os.makedirs('plots', exist_ok=True)

# # Create figure with two subplots
# plt.figure(figsize=(15, 6))

# # Plot RMSE
# plt.subplot(1, 2, 1)
# for method, values in all6_wo_gt_scaling['RMSE'].items():
#     plt.plot(range(len(values)), values, marker='o', label=method)
# plt.title('RMSE over Iterations')
# plt.xlabel('Iteration')
# plt.ylabel('RMSE')
# plt.grid(True)
# plt.legend()

# # Plot Median Ratio
# plt.subplot(1, 2, 2)
# for method, values in all6_wo_gt_scaling['Median Ratio'].items():
#     plt.plot(range(len(values)), values, marker='o', label=method)
# plt.title('Median Ratio over Iterations')
# plt.xlabel('Iteration')
# plt.ylabel('Median Ratio')
# plt.grid(True)
# plt.legend()

# # Adjust layout to prevent overlap
# plt.tight_layout()

# # Save the figure
# plt.savefig('plots/metrics_comparison.png', dpi=300, bbox_inches='tight')
# plt.close()

# # Create individual plots for each metric
# for metric in ['RMSE', 'Median Ratio']:
#     plt.figure(figsize=(10, 6))
#     for method, values in all6_wo_gt_scaling[metric].items():
#         plt.plot(range(len(values)), values, marker='o', label=method)
#     plt.title(f'{metric} over Iterations')
#     plt.xlabel('Iteration')
#     plt.ylabel(metric)
#     plt.grid(True)
#     plt.legend()
#     plt.tight_layout()
#     plt.savefig(f'plots/{metric.lower().replace(" ", "_")}.png', dpi=300, bbox_inches='tight')
#     plt.close()
    
def main_plot_old(method_key, data, prefix=''):
    # Create a new figure
    plt.figure(figsize=(7, 2))
    
    # Plot RMSE for MAE_final
    x = range(1, 21)  # Adjust x to range from 1 to 20
    y1 = data['RMSE'][method_key]
    plt.plot(x, y1, marker='o', color='b', label='RMSE')
    smaller_font_size = 12
    bigger_font_size = smaller_font_size + 2
    plt.xlabel('Loop No.', fontsize=bigger_font_size)
    plt.ylabel('RMSE', color='b', fontsize=bigger_font_size)
    # plt.title('RMSE and Median Ratio of MAE_final over Iterations')
    plt.grid(True)
    
    # Set x-ticks from 1 to 20
    plt.xticks(range(1, 21), fontsize=smaller_font_size)
    
    # Limit x-axis from 1 to 20
    plt.xlim(1, 20)
    
    # Set y-ticks at the beginning, middle, and end of the visible axis range for RMSE
    y1_min, y1_max = plt.ylim()
    y1_mid = (y1_min + y1_max) / 2
    plt.yticks([round(y1_min, 2), round(y1_mid, 2), round(y1_max, 2)], fontsize=smaller_font_size)
    
    # Create a second y-axis for Median Ratio
    ax2 = plt.gca().twinx()
    y2 = data['Median Ratio'][method_key]
    ax2.plot(x, y2, marker='x', color='r', label='Median Ratio')
    ax2.set_ylabel('Median Ratio', color='r', fontsize=bigger_font_size)
    
    # Set y-ticks at the beginning, middle, and end of the visible axis range for Median Ratio
    y2_min, y2_max = ax2.get_ylim()
    y2_mid = (y2_min + y2_max) / 2
    ax2.set_yticks([round(y2_min, 2), round(y2_mid, 2), round(y2_max, 2)])
    ax2.tick_params(axis='y', labelsize=smaller_font_size)
    
    # Add legends
    # plt.legend(loc='upper left')
    # ax2.legend(loc='upper right')
    
    # Adjust layout
    plt.tight_layout()
    
    # Save the plot
    plt.savefig(f'plots/{prefix}_{method_key}_metrics.png', dpi=300, bbox_inches='tight')
    plt.close()
    
def main_plot():
    smaller_font_size = 12
    bigger_font_size = smaller_font_size + 2
    # Create plots directory if it doesn't exist
    os.makedirs('plots', exist_ok=True)
    
    # Create figure with two subplots side by side
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 2))
    
    # Define methods to plot
    methods = ['AdaDepth', 
            #    'MAE_final'
               'AC'
               ]
    labels = {
        # 'MAE_final': 'MIC', 
        'AC': 'AC (Ours)', 
        'AdaDepth': "ICRA'23"}
    colors = {
        # 'MAE_final': 'tab:green', 
        'AC': 'tab:green', 
        'AdaDepth': 'tab:orange'}
    markers = {
        # 'MAE_final': 'o', 
        'AC': 'o', 
        'AdaDepth': 's'}
    
    # Plot RMSE (left plot)
    for i, method in enumerate(methods):
        values = all6_wo_gt_scaling['RMSE'][method]
        x = range(1, len(values) + 1)
        ax1.plot(x, values, marker=markers[method], color=colors[method], label=labels[method], linewidth=2, markersize=6)
    
    ax1.set_xlabel('Loop No.', fontsize=bigger_font_size)
    ax1.set_ylabel('RMSE', fontsize=bigger_font_size)
    ax1.grid(True, alpha=0.3)
    # ax1.legend(fontsize=11)
    ax1.set_xlim(1, 20)
    ax1.set_xticks(range(1, 21))
    y1_min, y1_max = ax1.get_ylim()
    y1_mid = (y1_min + y1_max) / 2
    ax1.set_yticks([round(y1_min, 2), round(y1_mid, 2), round(y1_max, 2)])
    ax1.tick_params(axis='both', labelsize=smaller_font_size)
    
    # Plot Median Ratio (right plot)
    for i, method in enumerate(methods):
        values = all6_wo_gt_scaling['Median Ratio'][method]
        x = range(1, len(values) + 1)
        ax2.plot(x, values, marker=markers[method], color=colors[method], label=labels[method], linewidth=2, markersize=6)
    
    ax2.set_xlabel('Loop No.', fontsize=bigger_font_size)
    ax2.set_ylabel('Median Ratio', fontsize=bigger_font_size)
    ax2.grid(True, alpha=0.3)
    ax2.legend(fontsize=bigger_font_size)
    ax2.set_xlim(1, 20)
    ax2.set_xticks(range(1, 21))
    y2_min, y2_max = ax2.get_ylim()
    y2_mid = (y2_min + y2_max) / 2
    ax2.set_yticks([round(y2_min, 2), round(y2_mid, 2), round(y2_max, 2)])
    ax2.tick_params(axis='both', labelsize=smaller_font_size)
    
    # Adjust layout
    plt.tight_layout()
    
    # Save the plot
    plt.savefig('plots/20xloop_all6_ac_vs_adadepth.pdf', dpi=300, bbox_inches='tight')
    plt.show()
    plt.close()

def main_plot_vertical():
    smaller_font_size = 12
    bigger_font_size = smaller_font_size + 2
    # Create plots directory if it doesn't exist
    os.makedirs('plots', exist_ok=True)
    
    # Create figure with two subplots side by side
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(6, 4))
    
    # Define methods to plot
    methods = ['AdaDepth', 
            #    'MAE_final'
               'AC'
               ]
    labels = {
        # 'MAE_final': 'MIC', 
        'AC': 'AC (Ours)', 
        'AdaDepth': "ICRA'23"}
    colors = {
        # 'MAE_final': 'tab:green', 
        'AC': 'tab:green', 
        'AdaDepth': 'tab:orange'}
    markers = {
        # 'MAE_final': 'o', 
        'AC': 'o', 
        'AdaDepth': 's'}
    
    # Plot RMSE (left plot)
    for i, method in enumerate(methods):
        values = all6_wo_gt_scaling['RMSE'][method]
        x = range(1, len(values) + 1)
        ax1.plot(x, values, marker=markers[method], color=colors[method], label=labels[method], linewidth=2, markersize=6)
    
    ax1.set_xlabel('Loop No.', fontsize=bigger_font_size)
    ax1.set_ylabel('RMSE', fontsize=bigger_font_size)
    ax1.grid(True, alpha=0.3)
    # ax1.legend(fontsize=11)
    ax1.set_xlim(1, 20)
    ax1.set_xticks(range(1, 21))
    y1_min, y1_max = ax1.get_ylim()
    y1_mid = (y1_min + y1_max) / 2
    ax1.set_yticks([round(y1_min, 2), round(y1_mid, 2), round(y1_max, 2)])
    ax1.tick_params(axis='both', labelsize=smaller_font_size)
    
    # Plot Median Ratio (right plot)
    for i, method in enumerate(methods):
        values = all6_wo_gt_scaling['Median Ratio'][method]
        x = range(1, len(values) + 1)
        ax2.plot(x, values, marker=markers[method], color=colors[method], label=labels[method], linewidth=2, markersize=6)
    
    ax2.set_xlabel('Loop No.', fontsize=bigger_font_size)
    ax2.set_ylabel('Median Ratio', fontsize=bigger_font_size)
    ax2.grid(True, alpha=0.3)
    ax2.legend(fontsize=bigger_font_size)
    ax2.set_xlim(1, 20)
    ax2.set_xticks(range(1, 21))
    y2_min, y2_max = ax2.get_ylim()
    y2_mid = (y2_min + y2_max) / 2
    ax2.set_yticks([round(y2_min, 2), round(y2_mid, 2), round(y2_max, 2)])
    ax2.tick_params(axis='both', labelsize=smaller_font_size)
    
    # Adjust layout
    plt.tight_layout()
    
    # Save the plot
    plt.savefig('plots/20xloop_all6_ac_vs_adadepth.pdf', dpi=300, bbox_inches='tight')
    plt.show()
    plt.close()

import matplotlib
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42

def main_plot_left():
    smaller_font_size = 12
    bigger_font_size = smaller_font_size + 2
    # Create plots directory if it doesn't exist
    os.makedirs('plots', exist_ok=True)
    
    # Create figure with two subplots side by side
    fig, ax1 = plt.subplots(1, 1, figsize=(6, 2))
    
    # Define methods to plot
    methods = ['AdaDepth', 
            #    'MAE_final'
               'AC'
               ]
    labels = {
        # 'MAE_final': 'MIC', 
        'AC': 'AC (Ours)', 
        'AdaDepth': "ICRA'23"}
    colors = {
        # 'MAE_final': 'tab:green', 
        'AC': 'tab:green', 
        'AdaDepth': 'tab:orange'}
    markers = {
        # 'MAE_final': 'o', 
        'AC': 'o', 
        'AdaDepth': 's'}
    
    # Plot RMSE (left plot)
    for i, method in enumerate(methods):
        values = all6_wo_gt_scaling['RMSE'][method]
        x = range(1, len(values) + 1)
        ax1.plot(x, values, marker=markers[method], color=colors[method], label=labels[method], linewidth=2, markersize=6)
    
    ax1.set_xlabel('Loop No.', fontsize=bigger_font_size)
    ax1.set_ylabel('RMSE', fontsize=bigger_font_size)
    ax1.grid(True, alpha=0.3)
    # ax1.legend(fontsize=11)
    ax1.set_xlim(1, 20)
    ax1.set_xticks(range(1, 21))
    y1_min, y1_max = ax1.get_ylim()
    y1_mid = (y1_min + y1_max) / 2
    ax1.set_yticks([round(y1_min, 2), round(y1_mid, 2), round(y1_max, 2)])
    ax1.tick_params(axis='both', labelsize=smaller_font_size)
    
    # # Plot Median Ratio (right plot)
    # for i, method in enumerate(methods):
    #     values = all6_wo_gt_scaling['Median Ratio'][method]
    #     x = range(1, len(values) + 1)
    #     ax2.plot(x, values, marker=markers[method], color=colors[method], label=labels[method], linewidth=2, markersize=6)
    
    # ax2.set_xlabel('Loop No.', fontsize=bigger_font_size)
    # ax2.set_ylabel('Median Ratio', fontsize=bigger_font_size)
    # ax2.grid(True, alpha=0.3)
    ax1.legend(fontsize=bigger_font_size)
    # ax2.set_xlim(1, 20)
    # ax2.set_xticks(range(1, 21))
    # y2_min, y2_max = ax2.get_ylim()
    # y2_mid = (y2_min + y2_max) / 2
    # ax2.set_yticks([round(y2_min, 2), round(y2_mid, 2), round(y2_max, 2)])
    # ax2.tick_params(axis='both', labelsize=smaller_font_size)
    
    # Adjust layout
    plt.tight_layout()
    
    # Save the plot
    plt.savefig('plots/20xloop_all6_mic_vs_adadepth_left.pdf', dpi=300, bbox_inches='tight')
    plt.show()
    plt.close()

if __name__ == '__main__':
    # main_plot_old('MAE_final')
    # main_plot_old('MAE_final_10xLR')
    # main_plot_old('MAE_final_5xLR')
    # main_plot_old('AdaDepth')

    main_plot_vertical()
    # main_plot_left()
    # main_plot_old('MAE_final', sunny_day_5_wo_gt_scaling, 'sunny_day_5')
    # main_plot_old('AdaDepth', sunny_day_5_wo_gt_scaling, 'sunny_day_5')
    