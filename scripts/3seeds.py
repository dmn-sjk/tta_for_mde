import statistics

data = {
    'CoTTA': { # 1e-7
        'DDAD': {
            'AbsRel': [
                0.144, 0.144
            ],
            'SqRel': [
                1.523, 1.523
            ],
            'RMSE': [
                7.983, 7.982
            ],
            'RMSElog': [
                0.229, 0.229
            ],
            'a1': [
                0.786, 0.786
            ],
            'a2': [
                0.937, 0.937
            ],
            'a3': [
                0.985, 0.976
            ],
        },
        'All-6': {
            'AbsRel': [
                0.262, 0.262
            ],
            'SqRel': [
                4.814, 4.818
            ],
            'RMSE': [
                12.190, 12.194
            ],
            'RMSElog': [
                0.461, 0.462
            ],
            'a1': [
                0.577, 0.577
            ],
            'a2': [
                0.759, 0.759
            ],
            'a3': [
                0.828, 0.828
            ],

        },
        'Rainy-5': {
            'AbsRel': [
                0.248, 0.248
            ],
            'SqRel': [
                3.481, 3.482
            ],
            'RMSE': [
                10.105, 10.107
            ],
            'RMSElog': [
                0.384, 0.384
            ],
            'a1': [
                0.558, 0.558
            ],
            'a2': [
                0.811, 0.811
            ],
            'a3': [
                0.887, 0.887
            ],
            
        },
        'Sunny-Day-5': {
            'AbsRel': [
                0.173, 0.173
            ],
            'SqRel': [
                2.148, 2.148
            ],
            'RMSE': [
                7.420, 7.420
            ],
            'RMSElog': [
                0.222, 0.222
            ],
            'a1': [
                0.791, 0.791
            ],
            'a2': [
                0.946, 0.946
            ],
            'a3': [
                0.976, 0.976
            ],
            
        },
        'Sunny-Night-5': {
            'AbsRel': [
                0.200, 0.200
            ],
            'SqRel': [
                2.144, 2.144
            ],
            'RMSE': [
                9.630, 9.630
            ],
            'RMSElog': [
                0.274, 0.274
            ],
            'a1': [
                0.601, 0.601
            ],
            'a2': [
                0.901, 0.901
            ],
            'a3': [
                0.971, 0.971
            ],
            
        },
    },
    'ICRA\'23': { # 1e-5
        'DDAD': {
            'AbsRel': [
                0.118, 0.117, 0.115
            ],
            'SqRel': [
                1.192, 1.181, 1.187
            ],
            'RMSE': [
                6.715, 6.694, 6.740
            ],
            'RMSElog': [
                0.185, 0.185, 0.186
            ],
            'a1': [
                0.858, 0.860, 0.858
            ],
            'a2': [
                0.960, 0.961, 0.960
            ],
            'a3': [
                0.985, 0.986, 0.985
            ],
        },
        'All-6': {
            'AbsRel': [
                0.219, 0.226, 0.217
            ],
            'SqRel': [
                3.659, 3.877, 3.638
            ],
            'RMSE': [
                10.767, 11.038, 10.734
            ],
            'RMSElog': [
                0.353, 0.370, 0.349
            ],
            'a1': [
                0.627, 0.618, 0.635
            ],
            'a2': [
                0.815, 0.799, 0.817
            ],
            'a3': [
                0.887, 0.876, 0.888
            ],

        },
        'Rainy-5': {
            'AbsRel': [
                0.232, 0.234, 0.233
            ],
            'SqRel': [
                2.681, 2.759, 2.754
            ],
            'RMSE': [
                8.833, 8.942, 8.893
            ],
            'RMSElog': [
                0.314, 0.319, 0.318
            ],
            'a1': [
                0.555, 0.554, 0.558
            ],
            'a2': [
                0.852, 0.847, 0.852
            ],
            'a3': [
                0.940, 0.936, 0.937
            ],
            
        },
        'Sunny-Day-5': {
            'AbsRel': [
                0.169, 0.168, 0.168
            ],
            'SqRel': [
                2.187, 2.160, 2.189
            ],
            'RMSE': [
                7.244, 7.217, 7.255
            ],
            'RMSElog': [
                0.212, 0.211, 0.212
            ],
            'a1': [
                0.818, 0.818, 0.818
            ],
            'a2': [
                0.947, 0.947, 0.947
            ],
            'a3': [
                0.976, 0.976, 0.976
            ],
            
        },
        'Sunny-Night-5': {
            'AbsRel': [
                0.167, 0.162, 0.165
            ],
            'SqRel': [
                1.738, 1.713, 1.728
            ],
            'RMSE': [
                8.595, 8.549, 8.560
            ],
            'RMSElog': [
                0.231, 0.227, 0.229
            ],
            'a1': [
                0.725, 0.738, 0.729
            ],
            'a2': [
                0.930, 0.930, 0.930
            ],
            'a3': [
                0.984, 0.983, 0.984
            ],
            
        },
    },
    'AC': { # MIC but AC
        'DDAD': { # 1e-4
            'AbsRel': [
                0.133, 0.133, 0.134
            ],
            'SqRel': [
                1.323, 1.307, 1.327
            ],
            'RMSE': [
                7.322, 7.206, 7.343
            ],
            'RMSElog': [
                0.207, 0.205, 0.208
            ],
            'a1': [
                0.821, 0.825, 0.821
            ],
            'a2': [
                0.951, 0.952, 0.951
            ],
            'a3': [
                0.983, 0.983, 0.982
            ],
            
        },
        'All-6': { # 5e-4
            'AbsRel': [
                0.191, 0.203, 0.195
            ],
            'SqRel': [
                2.195, 2.643, 2.363
            ],
            'RMSE': [
                7.818, 8.738, 8.050
            ],
            'RMSElog': [
                0.242, 0.265, 0.248
            ],
            'a1': [
                0.689, 0.662, 0.681
            ],
            'a2': [
                0.940, 0.907, 0.934
            ],
            'a3': [
                0.976, 0.961, 0.974
            ],
            
        },
        'Rainy-5': { # 5e-4
            'AbsRel': [
                0.228, 0.245, 0.240
            ],
            'SqRel': [
                2.586, 2.979, 3.020
            ],
            'RMSE': [
                8.725, 8.681, 8.647
            ],
            'RMSElog': [
                0.287, 0.294, 0.288
            ],
            'a1': [
                0.582, 0.569, 0.594
            ],
            'a2': [
                0.874, 0.879, 0.878
            ],
            'a3': [
                0.955, 0.948, 0.945
            ],
            
        },
        'Sunny-Day-5': { # 5e-4
            'AbsRel': [
                0.186, 0.185, 0.184
            ],
            'SqRel': [
                2.298, 2.265, 2.250
            ],
            'RMSE': [
                7.426, 7.432, 7.373
            ],
            'RMSElog': [
                0.230, 0.230, 0.228
            ],
            'a1': [
                0.768, 0.767, 0.772
            ],
            'a2': [
                0.941, 0.941, 0.942
            ],
            'a3': [
                0.975, 0.975, 0.975
            ],  
            
        },
        'Sunny-Night-5': { # 5e-4
            'AbsRel': [
                0.192, 0.207, 0.192
            ],
            'SqRel': [
                1.748, 2.009, 1.752
            ],
            'RMSE': [
                8.462, 9.386, 8.512
            ],
            'RMSElog': [
                0.244, 0.270, 0.245
            ],
            'a1': [
                0.603, 0.509, 0.597
            ],
            'a2': [
                0.943, 0.920, 0.942
            ],
            'a3': [
                0.987, 0.981, 0.987
            ],        
        },
    },
#     'MIC_maskRatio0.3': {
#         'DDAD': { # 1e-4
#             'AbsRel': [
#                 0.134, 0.134, 0.135
#             ],
#             'SqRel': [
#                 1.324, 1.345, 1.347
#             ],
#             'RMSE': [
#                 7.377, 7.457, 7.461
#             ],
#             'RMSElog': [
#                 0.209, 0.211, 0.211
#             ],
#             'a1': [
#                 0.820, 0.817, 0.816
#             ],
#             'a2': [
#                 0.951, 0.950, 0.950
#             ],
#             'a3': [
#                 0.983, 0.982, 0.982
#             ],
            
#         },
#         'All-6': { # 5e-4
#             'AbsRel': [
#                 0.201, 0.232, 0.198
#             ],
#             'SqRel': [
#                 2.489, 3.618, 2.509
#             ],
#             'RMSE': [
#                 8.331, 10.455, 8.562
#             ],
#             'RMSElog': [
#                 0.255, 0.338, 0.256
#             ],
#             'a1': [
#                 0.662, 0.615, 0.666
#             ],
#             'a2': [
#                 0.929, 0.850, 0.923
#             ],
#             'a3': [
#                 0.971, 0.913, 0.969
#             ],
            
#         },
#         'Rainy-5': { # 5e-4
#             'AbsRel': [
#                 0.234, 0.240, 0.233
#             ],
#             'SqRel': [
#                 2.631, 3.053, 2.930
#             ],
#             'RMSE': [
#                 8.520, 8.624, 8.319
#             ],
#             'RMSElog': [
#                 0.286, 0.285, 0.280
#             ],
#             'a1': [
#                 0.574, 0.599, 0.610
#             ],
#             'a2': [
#                 0.882, 0.880, 0.885
#             ],
#             'a3': [
#                 0.956, 0.947, 0.948
#             ],
            
#         },
#         'Sunny-Day-5': { # 5e-4
#             'AbsRel': [
#                 0.195, 0.194, 0.192
#             ],
#             'SqRel': [
#                 2.529, 2.514, 2.449
#             ],
#             'RMSE': [
#                 7.855, 7.830, 7.786
#             ],
#             'RMSElog': [
#                 0.237, 0.236, 0.235
#             ],
#             'a1': [
#                 0.750, 0.753, 0.755
#             ],
#             'a2': [
#                 0.942, 0.942, 0.943
#             ],
#             'a3': [
#                 0.975, 0.975, 0.975
#             ],  
            
#         },
#         'Sunny-Night-5': { # 5e-4
#             'AbsRel': [
#                 0.193, 0.192, 0.194
#             ],
#             'SqRel': [
#                 1.847, 1.842, 1.847
#             ],
#             'RMSE': [
#                 8.849, 8.878, 8.839
#             ],
#             'RMSElog': [
#                 0.250, 0.249, 0.250
#             ],
#             'a1': [
#                 0.603, 0.615, 0.604
#             ],
#             'a2': [
#                 0.934, 0.933, 0.934
#             ],
#             'a3': [
#                 0.985, 0.984, 0.985
#             ],
#         },
#     },
#     'MIC_maskRatio0.7': {
#         'DDAD': { # 1e-4
#             'AbsRel': [
#                 0.133, 0.133, 0.134
#             ],
#             'SqRel': [
#                 1.308, 1.301, 1.323
#             ],
#             'RMSE': [
#                 7.150, 7.142, 7.222
#             ],
#             'RMSElog': [
#                 0.204, 0.203, 0.206
#             ],
#             'a1': [
#                 0.826, 0.826, 0.822
#             ],
#             'a2': [
#                 0.952, 0.952, 0.951
#             ],
#             'a3': [
#                 0.983, 0.983, 0.983
#             ],
            
#         },
#         'All-6': { # 5e-4
#             'AbsRel': [
#                 0.184, 0.196, 0.185
#             ],
#             'SqRel': [
#                 2.079, 2.361, 2.094
#             ],
#             'RMSE': [
#                 7.710, 8.389, 7.787
#             ],
#             'RMSElog': [
#                 0.237, 0.261, 0.237
#             ],
#             'a1': [
#                 0.711, 0.665, 0.702
#             ],
#             'a2': [
#                 0.938, 0.919, 0.939
#             ],
#             'a3': [
#                 0.976, 0.966, 0.977
#             ],
            
#         },
#         'Rainy-5': { # 5e-4
#             'AbsRel': [
#                 0.255, 0.251, 0.248
#             ],
#             'SqRel': [
#                 3.469, 3.182, 3.343
#             ],
#             'RMSE': [
#                 8.728, 8.782, 8.724
#             ],
#             'RMSElog': [
#                 0.293, 0.295, 0.291
#             ],
#             'a1': [
#                 0.593, 0.581, 0.597
#             ],
#             'a2': [
#                 0.878, 0.865, 0.874
#             ],
#             'a3': [
#                 0.939, 0.942, 0.941
#             ],
            
#         },
#         'Sunny-Day-5': { # 5e-4
#             'AbsRel': [
#                 0.183, 0.185, 0.189
#             ],
#             'SqRel': [
#                 2.192, 2.247, 2.396
#             ],
#             'RMSE': [
#                 7.397, 7.463, 7.521
#             ],
#             'RMSElog': [
#                 0.229, 0.230, 0.233
#             ],
#             'a1': [
#                 0.766, 0.763, 0.767
#             ],
#             'a2': [
#                 0.945, 0.943, 0.940
#             ],
#             'a3': [
#                 0.976, 0.976, 0.974
#             ],  
            
#         },
#         'Sunny-Night-5': { # 5e-4
#             'AbsRel': [
#                 0.204, 0.203, 0.203
#             ],
#             'SqRel': [
#                 1.897, 1.903, 1.879
#             ],
#             'RMSE': [
#                 8.958, 8.974, 8.905
#             ],
#             'RMSElog': [
#                 0.261, 0.260, 0.261
#             ],
#             'a1': [
#                 0.518, 0.528, 0.519
#             ],
#             'a2': [
#                 0.933, 0.932, 0.933
#             ],
#             'a3': [
#                 0.985, 0.984, 0.985
#             ],
#         },
#     },
#     'MIC_maskRatio0.1': {
#         'DDAD': { # 1e-4
#             'AbsRel': [
#                 0.133, 0.135, 0.135
#             ],
#             'SqRel': [
#                 1.343, 1.368, 1.381
#             ],
#             'RMSE': [
#                 7.511, 7.581, 7.656
#             ],
#             'RMSElog': [
#                 0.211, 0.213, 0.215
#             ],
#             'a1': [
#                 0.818, 0.813, 0.811
#             ],
#             'a2': [
#                 0.949, 0.948, 0.947
#             ],
#             'a3': [
#                 0.982, 0.981, 0.981
#             ],
            
#         },
#         'All-6': { # 5e-4
#             'AbsRel': [
#                 0.279, 0.202, 0.208
#             ],
#             'SqRel': [
#                 5.244, 2.564, 2.839
#             ],
#             'RMSE': [
#                 12.498, 9.021, 9.174
#             ],
#             'RMSElog': [
#                 0.461, 0.274, 0.293
#             ],
#             'a1': [
#                 0.563, 0.651, 0.653
#             ],
#             'a2': [
#                 0.748, 0.899, 0.890
#             ],
#             'a3': [
#                 0.829, 0.957, 0.946
#             ],
            
#         },
#         'Rainy-5': { # 5e-4
#             'AbsRel': [
#                 0.242, 0.251, 0.258
#             ],
#             'SqRel': [
#                 3.329, 3.251, 3.515
#             ],
#             'RMSE': [
#                 8.540, 8.744, 8.783
#             ],
#             'RMSElog': [
#                 0.281, 0.293, 0.298
#             ],
#             'a1': [
#                 0.620, 0.578, 0.577
#             ],
#             'a2': [
#                 0.885, 0.878, 0.875
#             ],
#             'a3': [
#                 0.947, 0.946, 0.941
#             ],
            
#         },
#         'Sunny-Day-5': { # 5e-4
#             'AbsRel': [
#                 0.184, 0.181, 0.182
#             ],
#             'SqRel': [
#                 2.359, 2.243, 2.299
#             ],
#             'RMSE': [
#                 7.549, 7.464, 7.492
#             ],
#             'RMSElog': [
#                 0.232, 0.229, 0.229
#             ],
#             'a1': [
#                 0.771, 0.776, 0.774
#             ],
#             'a2': [
#                 0.940, 0.941, 0.942
#             ],
#             'a3': [
#                 0.975, 0.975, 0.975
#             ],  
            
#         },
#         'Sunny-Night-5': { # 5e-4
#             'AbsRel': [
#                 0.196, 0.194, 0.196
#             ],
#             'SqRel': [
#                 1.899, 1.875, 1.872
#             ],
#             'RMSE': [
#                 8.660, 8.464, 8.417
#             ],
#             'RMSElog': [
#                 0.248, 0.243, 0.245
#             ],
#             'a1': [
#                 0.614, 0.632, 0.614
#             ],
#             'a2': [
#                 0.938, 0.943, 0.944
#             ],
#             'a3': [
#                 0.985, 0.986, 0.986
#             ],
#         },
#     },
#     'MIC_maskRatio0.9': {
#         'DDAD': { # 1e-4
#             'AbsRel': [
#                 0.135, 0.135, 0.136
#             ],
#             'SqRel': [
#                 1.362, 1.343, 1.359
#             ],
#             'RMSE': [
#                 6.957, 6.868, 6.989
#             ],
#             'RMSElog': [
#                 0.202, 0.200, 0.202
#             ],
#             'a1': [
#                 0.826, 0.830, 0.826
#             ],
#             'a2': [
#                 0.949, 0.951, 0.950
#             ],
#             'a3': [
#                 0.983, 0.983, 0.983
#             ],
            
#         },
#         'All-6': { # 5e-4
#             'AbsRel': [
#                 0.209, 0.198, 0.216
#             ],
#             'SqRel': [
#                 2.609, 2.399, 3.015
#             ],
#             'RMSE': [
#                 8.203, 7.944, 8.593
#             ],
#             'RMSElog': [
#                 0.261, 0.250, 0.263
#             ],
#             'a1': [
#                 0.651, 0.676, 0.655
#             ],
#             'a2': [
#                 0.923, 0.931, 0.926
#             ],
#             'a3': [
#                 0.970, 0.972, 0.968
#             ],
            
#         },
#         'Rainy-5': { # 5e-4
#             'AbsRel': [
#                 0.241, 0.248, 0.251
#             ],
#             'SqRel': [
#                 2.874, 3.008, 2.984
#             ],
#             'RMSE': [
#                 9.931, 9.963, 10.030
#             ],
#             'RMSElog': [
#                 0.322, 0.324, 0.330
#             ],
#             'a1': [
#                 0.524, 0.535, 0.503
#             ],
#             'a2': [
#                 0.838, 0.828, 0.827
#             ],
#             'a3': [
#                 0.938, 0.936, 0.937
#             ],
            
#         },
#         'Sunny-Day-5': { # 5e-4
#             'AbsRel': [
#                 0.236, 0.207, 0.241
#             ],
#             'SqRel': [
#                 3.107, 2.423, 3.545
#             ],
#             'RMSE': [
#                 9.071, 8.129, 9.633
#             ],
#             'RMSElog': [
#                 0.274, 0.245, 0.282
#             ],
#             'a1': [
#                 0.654, 0.707, 0.669
#             ],
#             'a2': [
#                 0.910, 0.930, 0.900
#             ],
#             'a3': [
#                 0.970, 0.978, 0.962
#             ],  
            
#         },
#         'Sunny-Night-5': { # 5e-4
#             'AbsRel': [
#                 0.221, 0.205, 0.335
#             ],
#             'SqRel': [
#                 2.559, 2.166, 5.692
#             ],
#             'RMSE': [
#                 10.714, 9.609, 15.338
#             ],
#             'RMSElog': [
#                 0.299, 0.266, 0.507
#             ],
#             'a1': [
#                 0.514, 0.598, 0.243
#             ],
#             'a2': [
#                 0.869, 0.905, 0.654
#             ],
#             'a3': [
#                 0.965, 0.980, 0.791
#             ],
    #     },
    # },
}

efficiency = {
    'CoTTA': {
        'Wall-clock Time [ms]': 1508.6,
        'Memory [MB]': 9663.2
    },
    'ICRA\'23': {
        'Wall-clock Time [ms]': 363.1,
        'Memory [MB]': 10433.2
    },
    'AC': {
        'Wall-clock Time [ms]': 165.3,
        'Memory [MB]': 7017.7
    }
}
        

# Iterate over each method
for method, datasets in data.items():
    print(f"Method: {method}")
    
    # Dictionary to store all values for each metric across datasets
    method_metrics = {metric: [] for metric in datasets[next(iter(datasets))].keys()}
    
    # Iterate over each dataset within the method
    for dataset, metrics in datasets.items():
        # Prepare a list to store the mean of each metric
        metric_means = []
        
        # Iterate over each metric within the dataset
        for metric, values in metrics.items():
            mean_value = statistics.mean(values)
            metric_means.append(f"{mean_value:.3f}")
            
            # Add all values to the method metrics
            method_metrics[metric].extend(values)
        
        # Print the method, dataset, and all metric means in a single line
        print(f"& {method} & {dataset} & " + " & ".join(metric_means) + " \\\\")
    
    # Calculate and print the mean for each metric across all datasets in the method
    overall_means = [f"{statistics.mean(all_values):.3f}" for all_values in method_metrics.values()]
    print(f"& {method} & Mean & " + " & ".join(overall_means) + " \\\\")
    
    print()  # Newline for better readability


import matplotlib.pyplot as plt
import statistics
import os
import numpy as np

# Ensure the plots directory exists
os.makedirs('plots', exist_ok=True)

# Calculate average metrics for each method
average_metrics = {}
for method, datasets in data.items():
    method_metrics = {metric: [] for metric in datasets[next(iter(datasets))].keys()}
    
    for dataset, metrics in datasets.items():
        for metric, values in metrics.items():
            # Average the values for each metric
            mean_value = statistics.mean(values)
            method_metrics[metric].append(mean_value)
    
    # Calculate the overall average for each metric across all datasets
    average_metrics[method] = {metric: statistics.mean(values) for metric, values in method_metrics.items()}

# Define colors and markers for methods and datasets
method_colors = {
    'CoTTA': 'tab:blue',
    'ICRA\'23': 'tab:orange',
    'AC': 'tab:green'
}

dataset_markers = {
    'DDAD': 's',
    'All-6': 'p',
    'Rainy-5': 'P',
    'Sunny-Day-5': 'X',
    'Sunny-Night-5': 'D'
}

labels = {
    'RMSE': r'RMSE$\downarrow$',
    'a1': '$\\sigma < 1.25 \\uparrow$',
    'a2': 'a2',
    'a3': 'a3',
    'AbsRel': 'AbsRel',
    'SqRel': 'SqRel',
    'RMSElog': 'RMSElog'
}

from matplotlib.font_manager import FontProperties

import matplotlib
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42

SMALLER_FONT = 12
BIGGER_FONT = SMALLER_FONT + 2

def teaser_1():
    # used_metrics = ['AbsRel', 'SqRel', 'RMSE', 'RMSElog', 'a1', 'a2', 'a3']
    used_metrics = ['RMSE', 'a1']
    num_metrics = len(used_metrics)


    # Plotting
    efficiency_metrics = ['Wall-clock Time [ms]', 'Memory [MB]']
    fig, axs = plt.subplots(num_metrics, len(efficiency_metrics), figsize=(5, 4), sharey='row')
    for eff_met_idx, eff_metric in enumerate(efficiency_metrics):
        # num_metrics = len(data[next(iter(data))][next(iter(data[next(iter(data))]))].keys())
        # fig, axs = plt.subplots(num_metrics, 1, figsize=(5, 5), sharex=True)
        
        # for i, metric in enumerate(data[next(iter(data))][next(iter(data[next(iter(data))]))].keys()):
        for i, metric in enumerate(used_metrics):
            for method, datasets in data.items():
                x_value = efficiency[method][eff_metric]
                for dataset, metrics in datasets.items():
                    if metric not in used_metrics:
                        continue
                    y_values = metrics[metric]
                    y_val = np.mean(y_values)
                    
                    axs[i, eff_met_idx].scatter(
                        x_value, y_val,
                        label=f'{method} - {dataset}',
                        color=method_colors[method],
                        marker=dataset_markers[dataset],
                        alpha=0.7
                    )
            
            axs[i, eff_met_idx].grid(True)
            if eff_met_idx == 0:
                axs[i, eff_met_idx].set_ylabel(labels[metric], rotation=0, labelpad=35, fontsize=BIGGER_FONT)
            
            # Set different x-axis limits for left and right plots
            if eff_met_idx == 0:  # Left plots
                axs[i, eff_met_idx].set_xlim(0, 1700)  # Example limits for 'Wall-clock Time [ms]'
                axs[i, eff_met_idx].get_shared_x_axes().join(axs[i, 0], axs[i, eff_met_idx])
            else:  # Right plots
                axs[i, eff_met_idx].set_xlim(6000, 11000)  # Example limits for 'Memory [MB]'
                axs[i, eff_met_idx].get_shared_x_axes().join(axs[i, 1], axs[i, eff_met_idx])
            
            # Remove x-tick labels for upper plots
            if i < num_metrics - 1:
                axs[i, eff_met_idx].set_xticklabels([])
        
            axs[i, eff_met_idx].tick_params(axis='both', which='major', labelsize=SMALLER_FONT)

        axs[-1, eff_met_idx].set_xlabel(eff_metric, fontsize=SMALLER_FONT)
        # plt.suptitle(f'Metrics vs {eff_metric}')
        # plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.tight_layout(pad=1.0, h_pad=0.5, w_pad=0.7)
        # Enable LaTeX rendering in Matplotlib
        # plt.rcParams['text.usetex'] = True
        

        # Create a custom legend for methods
        # method_handles = [plt.Line2D([0], [0], marker='o', color='w', label=(r'\textbf{MIC}' if method == 'MIC' else method),
        method_handles = [plt.Line2D([0], [0], marker='o', color='w', label=method,
                                    markerfacecolor=method_colors[method], markersize=10)
                        for method in method_colors]

        # Create a custom legend for datasets
        dataset_handles = [plt.Line2D([0], [0], marker=dataset_markers[dataset], color='w', label=dataset,
                                    markerfacecolor='gray', markersize=10)
                        for dataset in dataset_markers]

        # Position the method legend aligned with the top of the plots
        fig.legend(handles=method_handles, loc='upper left', bbox_to_anchor=(0.98, 0.99), title='Methods', 
                fontsize=BIGGER_FONT, title_fontsize=BIGGER_FONT)

        # Position the dataset legend closer to the method legend
        fig.legend(handles=dataset_handles, loc='upper left', bbox_to_anchor=(0.98, 0.67), title='Benchmarks', 
                fontsize=BIGGER_FONT, title_fontsize=BIGGER_FONT)

    plt.savefig(f'plots/teaser.pdf', bbox_inches='tight')
    plt.close()
    
def teaser_2():
    # used_metrics = ['AbsRel', 'SqRel', 'RMSE', 'RMSElog', 'a1', 'a2', 'a3']
    used_metrics = ['RMSE', 'a1']
    num_metrics = len(used_metrics)

    # Plotting
    efficiency_metrics = ['Wall-clock Time [ms]', 'Memory [MB]']
    x_axis_efficiency_metric_idx = 1
    the_other_efficiency_metric_idx = 0
    fig, axs = plt.subplots(num_metrics, 1, figsize=(5, 4), sharey='row')

    # Calculate size scaling factor for memory values
    memory_values = [efficiency[method][efficiency_metrics[the_other_efficiency_metric_idx]] for method in average_metrics.keys()]
    min_memory = min(memory_values)
    max_memory = max(memory_values)
    size_range = (50, 300)  # Min and max point sizes
    
    for row, metric in enumerate(used_metrics):
        for method in average_metrics.keys():
            # Calculate point size based on memory usage
            memory_size = efficiency[method][efficiency_metrics[the_other_efficiency_metric_idx]]
            point_size = size_range[0] + (size_range[1] - size_range[0]) * (memory_size - min_memory) / (max_memory - min_memory)
            
            axs[row].scatter(
                efficiency[method][efficiency_metrics[x_axis_efficiency_metric_idx]], 
                average_metrics[method][metric], 
                label=method if 'MIC' not in method else f'{method} (Ours)',
                s=point_size,
                alpha=0.7
            )
            
        axs[row].grid(True)
        axs[row].set_ylabel(labels[metric], rotation=0, labelpad=35, fontsize=BIGGER_FONT)
        # axs[row].set_xlim(0, 1700)  # Example limits for 'Wall-clock Time [ms]'
        # axs[row].set_xlim(7000, 10500)  # Example limits for 'Wall-clock Time [ms]'
            
            
        axs[row].tick_params(axis='both', which='major', labelsize=SMALLER_FONT)
    
    axs[0].set_xticklabels([])
    axs[-1].set_xlabel(efficiency_metrics[x_axis_efficiency_metric_idx], fontsize=SMALLER_FONT)
    axs[0].set_ylim(7,10)
    axs[-1].set_ylim(0.6,0.9)
    
    # Add legend
    # axs[0].legend(loc='upper right', fontsize=BIGGER_FONT, bbox_to_anchor=(0.98, 0.99))
    
    # Add size legend for memory values
    # Create dummy scatter plots for the size legend
    legend_elements = []
    legend_sizes = [min_memory, (min_memory + max_memory) / 2, max_memory]
    # legend_labels = [f'{size:.0f} MB' for size in legend_sizes]
    legend_labels = [f'{size:.0f} ms' for size in legend_sizes]
    
    for size, label in zip(legend_sizes, legend_labels):
        point_size = size_range[0] + (size_range[1] - size_range[0]) * (size - min_memory) / (max_memory - min_memory)
        legend_elements.append(plt.scatter([], [], s=point_size, c='gray', alpha=0.7, label=label))
    
    # Add the size legend to the second subplot
    # axs[1].legend(handles=legend_elements, loc='upper right', title='Memory Usage', 
    #               fontsize=BIGGER_FONT, title_fontsize=BIGGER_FONT, bbox_to_anchor=(0.98, 0.67))
        
    method_handles = [plt.Line2D([0], [0], marker='o', color='w', label=method if 'AC' not in method else f'{method} (Ours)',
                                markerfacecolor=method_colors[method], markersize=10)
                    for method in method_colors]
    # Position the method legend aligned with the top of the plots
    fig.legend(handles=method_handles, loc='upper left', bbox_to_anchor=(0.96, 0.97), title='Methods', 
            fontsize=BIGGER_FONT, title_fontsize=BIGGER_FONT)

    # Position the dataset legend closer to the method legend
    # fig.legend(handles=legend_elements, loc='upper left', bbox_to_anchor=(0.98, 0.67), title='Memory Usage', 
    fig.legend(handles=legend_elements, loc='upper left', bbox_to_anchor=(0.96, 0.66), title='Wall-clock Time', 
            fontsize=BIGGER_FONT, title_fontsize=BIGGER_FONT)
    
    # plt.tight_layout(pad=1.0, h_pad=0.5, w_pad=0.7)
    plt.tight_layout()

    plt.savefig(f'plots/teaser.pdf', bbox_inches='tight')
    # plt.savefig(f'plots/teaser.png', bbox_inches='tight')
    plt.close()

    
if __name__ == "__main__":
    teaser_2()
