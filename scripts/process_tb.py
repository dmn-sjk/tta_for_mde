import os
import numpy as np
import matplotlib.pyplot as plt
from tensorboard.backend.event_processing import event_accumulator
import seaborn as sns

sns.set_theme()
sns.set_context("paper")

FONT_SIZE = 25
BIGGER_FONT_SIZE = FONT_SIZE + 6


key_to_label = {
    'not_mean_unsupl/rms': 'RMSE',
    'unsup_loss': 'self-supervised loss',
    'error/not_mean_sup/rms': 'RMSE'
}

dataset_to_scenes_file = {
    'dgp': 'dgp.txt',
    'kitti': 'kitti_depth.txt',
    'waymo': [
        'waymo_sunny_Day.txt',
        'waymo_sunny_Dawn_Dusk.txt',
        'waymo_rain_Dawn_Dusk.txt',
        'waymo_sunny_Night.txt'
        ]
}

method_to_label = {
    'ssl_frozen': 'No adaptation',
    'ssl_naive': 'Photometric loss',
    'adadepth': 'AdaDepth'
}

def normalize(metric_vals):
    vals = np.array(metric_vals)
    vals = (vals - np.min(vals)) / (np.max(vals) - np.min(vals))
    return vals

def plot_tensorboard_metrics(log_dir_metrics_dict, save_dir, ver_lines_at=[], 
                             plot_name='plot', normalize=True, y_label='Value Normalized'):
    """
    Reads two plot values from a TensorBoard log file and creates a single plot with both curves.

    Args:
        log_dir (str): Path to the TensorBoard log directory.
        metric1 (str): Name of the first metric to plot.
        metric2 (str): Name of the second metric to plot.
    """
    plt.figure(figsize=(14, 10))

    for log_dir, metrics in log_dir_metrics_dict.items():
        event_acc = event_accumulator.EventAccumulator(log_dir)
        event_acc.Reload()
        
        for metric in metrics:
            label = metric['label']
            metric = metric['metric']
            metric_data = [[], []]
            for v in event_acc.Scalars(metric):
                metric_data[0].append(v.step)
                metric_data[1].append(v.value)

            if normalize:
                metric_data[1] = normalize(metric_data[1])
        
    
            # plt.plot(metric_data[0], metric_data[1], label=key_to_label[metric])
            plt.plot(metric_data[0], metric_data[1], label=label, linewidth=3.0)

    plt.xlabel('Frames', fontsize=BIGGER_FONT_SIZE)
    plt.ylabel(y_label, fontsize=BIGGER_FONT_SIZE)
    # plt.title(os.path.basename(log_dir))
    
    plt.yticks(fontsize=FONT_SIZE)
    plt.xticks(fontsize=FONT_SIZE)
    
    for x in ver_lines_at:
        plt.axvline(x = x, color = 'r', linestyle='--')
    
    plt.legend(loc='best', fontsize=BIGGER_FONT_SIZE)
    plt.grid(True)
    plt.tight_layout()
    
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    plt.savefig(os.path.join(save_dir, f"{plot_name}.png"))

def get_dataset(dir):
    if 'dgp' in dir:
        return 'dgp'
    elif 'waymo' in dir:
        return 'waymo'
    elif 'kitti_c' in dir:
        return 'kitti_c'
    else:
        return 'kitti'

def get_method(dir):
    methods = [
        'ssl_naive',
        'ssl_frozen',
        'adadepth',
        'custom'
    ]
    for method in methods:
        if method in dir:
            return method
    if 'sslnaive' in dir:
        return 'ssl_naive'
    elif 'sslfrozen' in dir:
        return 'ssl_frozen'
    raise NotImplementedError()

def get_scenes_end(dataset):
    scenes_file = dataset_to_scenes_file[dataset]
    
    if not isinstance(scenes_file, list):
        scenes_path = os.path.join('dataset_imgs/3frames_scenes', scenes_file)
        prev_scene = None
        ends = []
        with open(scenes_path, 'r') as f:
            for i, line in enumerate(f):
                curr_scene = line.strip()

                if prev_scene is None:
                   pass 
                elif prev_scene != curr_scene:
                    ends.append(i - 1)
                
                prev_scene = curr_scene
        return ends
    else:
        ends = []
        idx = -1
        for scene_file in scenes_file:
            scene_path = os.path.join('dataset_imgs/3frames_scenes', scene_file)
            with open(scene_path, 'r') as f:
                while f.readline() != "":
                    idx += 1
            ends.append(idx)
        return ends[:-1]


if __name__ == "__main__":
    # exp_dirs = [
    #     # './exp_logs/kitti2dgp_ssl_frozen',
    #     # './exp_logs/ssl_naive_dgp__LEARNING_RATE1e-05',
    #     './exp_logs/ssl_frozen_waymo_',
    #     # './exp_logs/ssl_naive_waymo__LEARNING_RATE1e-05',
    #     # './exp_logs/kitti2kitti_sslfrozen',
    #     # './exp_logs/kitti2kitti_sslnaive_LR1e-5',
    # ]
    
    
    # metrics = [
    #     # 'not_mean_unsupl/rms',
    #     'error/not_mean_sup/rms'
    #     # 'not_mean_unsupl/log_rms',
    #     # 'not_mean_unsupl/sq_rel',
    #     # 'not_mean_unsupl/abs_rel',
    #     # 'not_mean_unsupl/a3',
    #     # 'not_mean_unsupl/a2',
    #     # 'not_mean_unsupl/a1'
    #     ]
    
    
    # for exp_dir in exp_dirs:
    #     for metric in metrics:
    #         dataset = get_dataset(exp_dir)
    #         ends = get_scenes_end(dataset)
    #         method = get_method(exp_dir)
    #         log_dir_metrics_dict = {exp_dir: [metric]}
    #         plot_tensorboard_metrics(log_dir_metrics_dict, 
    #                                  os.path.join('playground_new/plots', dataset),
    #                                  method, ver_lines_at=ends,
    #                                  plot_name=f"{method}-{metric}_|_{'unsup_loss'}.png".replace('/', '_'))
    
    kitti_seqs = [
        '2011_09_26_0117',
        '2011_09_26_0096',
        '2011_09_26_0086',
    ]
    exp_dirs = [
        './exp_logs/kitti2kitti_ssl_frozen_{}',
        './exp_logs/kitti2kitti_ssl_naive_{}_LR1e-4',
        './exp_logs/kitti2kitti_ssl_naive_{}_LR1e-5',
        './exp_logs/kitti2kitti_ssl_naive_{}_LR1e-6',
    ]
    for kitti_seq in kitti_seqs:
        log_dir_metric_dict = {}
        for exp_dir in exp_dirs:
            method = get_method(exp_dir)
            dataset = get_dataset(exp_dir)
            label = method_to_label[method]
            if method == 'ssl_naive':
                label += ' ' + exp_dir[-6:]
            log_dir_metric_dict[exp_dir.format(kitti_seq)] = [{'metric': 'error/sup/rms',
                                                               'label': label}]

        plot_tensorboard_metrics(log_dir_metric_dict,
                                os.path.join('playground_new/plots', dataset + '_seq'),
                                plot_name=kitti_seq, normalize=False, y_label='RMSE')
