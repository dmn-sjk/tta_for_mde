#!/usr/bin python

import itertools
import subprocess
import os
from copy import deepcopy
import time


# RUN_SCRIPT_DIR = '/' + os.path.join(*__file__.split('/')[:-1])
RUN_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
JOBS_DIR = os.path.join('/path/to/scratch', 'slurm_dmns', 'ada_depth')
# JOBS_DIR = os.path.join('/path/to/scratch', 'slurm_dmns', 'test_time_adaptation')

# /* TO MODIFY ---------------------------------------------
SLURM = True
CPUS_PER_TASK = 10
MEM_PER_CPU = 3


# DATADIR = '/path/to/project'
DATADIR = '/path/to/scratch/datasets'
# EXP_LOGS_DIR = '/path/to/scratch/ada-depth/exp_logs'
EXP_LOGS_DIR = '/path/to/project/ada-depth/exp_logs'
CKPTS_DIR = '/path/to/project/ada-depth/ckpts'


DATA_FOLDERS = {
    'kitti': 'KITTI/kitti_data',
    'kitti_depth': 'KITTI/kitti_data',
    'kitti_c': 'KITTI',
    'waymo': 'waymo',
    'waymo_all6': 'waymo',
    'waymo_rainy5': 'waymo',
    'waymo_sunny_day5': 'waymo',
    'waymo_sunny_night5': 'waymo',
    'dgp': 'ddad_train_val',
    'driving_stereo': 'driving_stereo',
    'shift': 'shift',
    'shift_discr1:11': 'shift',
}


# the dict of datasets, in the lists inside the dict should be setting names
datasets = [
    # 'kitti',
    # 'kitti_depth',
    # 'waymo',
    'waymo_all6',
    # 'waymo_rainy5',
    # 'waymo_sunny_day5',
    # 'waymo_sunny_night5',
    # 'dgp',
    # 'kitti_c',
    # 'driving_stereo',
    # 'shift',
    # 'shift_discr1:11',
]


# RUN_NAME = 'all6_dynamicBN'
# RUN_NAME = 'all6_testBN'
RUN_NAME = 'supModelSSL_GTEgo_30xloop'
# RUN_NAME = 'supModelFrozen'
# RUN_NAME = 'customSeq'

RNG_SEED=1

# all params should be in a form of array, except RUN_NAME 
configs = {
    # 'ssl_frozen': {
    #     'RUN_NAME': RUN_NAME,
    #     # 'frame_ids': ['0 -1']
    # },
    'custom': {
        'RUN_NAME': RUN_NAME,
        # 'frame_ids': ['0 -1']
        'scales': ['0'],
        'gt_transform': ['']
    },
    # 'ssl_naive': {
    #     'RUN_NAME': RUN_NAME,
    #     'learning_rate': [
    #         # 1e-3,
    #         # 1e-4,
    #         1e-5,
    #         # 1e-6
    #         ],
    #     # 'frame_ids': ['0 -1']
    # },
    # 'adadepth': {
    #     'RUN_NAME': RUN_NAME,
    #     'learning_rate': [
    #         # 1e-3,
    #         # 1e-4,
    #         1e-5,
    #         # 1e-6
    #         ],
    # }
}

# argument names from the config dict above to put into the RUN_NAME string 
args_to_run_name = [
    'frame_ids',
    'learning_rate',
    'sup_model',
    'model_type',
    'scale_alignment',
    'eval_split',
    'alpha',
    'augs_config',
    'masking',
    'loops',
    ]

# common_args = f"--load_weights_folder {os.path.join(CKPTS_DIR, 'kitti_sup/models/weights_19')} \
#     --models_to_load encoder depth \
#     --reg_path {os.path.join(CKPTS_DIR, 'kitti_unsup/models/weights_19')} \
#     --num_workers 0"
    
common_args = f"--models_to_load encoder depth \
    --sup_model_path {os.path.join(CKPTS_DIR, 'kitti_sup/models/weights_19')} \
    --ssl_model_path {os.path.join(CKPTS_DIR, 'kitti_unsup/models/weights_19')} \
    --num_workers 0 \
    --seed {RNG_SEED}"

# common_args = f"--models_to_load encoder depth \
#     --sup_model_path {os.path.join(CKPTS_DIR, 'newcrf_seg/models/weights_1')} \
#     --ssl_model_path {os.path.join(CKPTS_DIR, 'kitti_unsup/models/weights_19')} \
#     --num_workers 0"

# TO MODIFY */ ---------------------------------------------

def main():
    if SLURM:
        if not os.path.exists(JOBS_DIR):
            os.mkdir(JOBS_DIR)
    
    for dataset in datasets:
        perform_experiments(dataset)
                
    print('Done!')
            
def run_command(command, method, run_name, dataset):
    if SLURM:
        t = time.localtime()
        current_time = time.strftime("%H:%M:%S", t)

        job_folder = os.path.join(JOBS_DIR, run_name + '_' + f'seed{RNG_SEED}' + '_' +  current_time)
        if not os.path.exists(job_folder):
            os.mkdir(job_folder)

        job_file = os.path.join(job_folder, f"{run_name}.job")
        output_file = os.path.join(job_folder, f"{run_name}.out")
        error_file = os.path.join(job_folder, f"{run_name}.err")
        
        with open(job_file, 'w') as fh:
            fh.writelines("#!/bin/bash\n")
            fh.writelines(f"#SBATCH --job-name={run_name}.job\n")
            fh.writelines(f"#SBATCH --output={output_file}\n")
            fh.writelines(f"#SBATCH --error={error_file}\n")
            fh.writelines("#SBATCH --gpus=1\n")
            fh.writelines("#SBATCH --account=YOUR_GRANT_ACCOUNT-gpu-a100\n")
            fh.writelines("#SBATCH --partition=plgrid-gpu-a100\n")
            fh.writelines("#SBATCH --gres=gpu\n")
            fh.writelines("#SBATCH --time=48:00:00\n")
            fh.writelines(f"#SBATCH --cpus-per-task={CPUS_PER_TASK}\n")
            fh.writelines(f"#SBATCH --mem-per-cpu={MEM_PER_CPU}G\n")


            fh.writelines(f"cd {RUN_SCRIPT_DIR}\n")
            fh.writelines("module load Miniconda3/4.9.2\n")
            fh.writelines("eval \"$(conda shell.bash hook)\"\n")
            fh.writelines("conda activate /path/to/project/conda_envs/ada_depth_shift\n")
            
            fh.writelines(command)
            fh.writelines("\n")

        os.system(f"sbatch {job_file}")
    else:
        # Stream output live (stderr merged into stdout to preserve ordering).
        process = subprocess.Popen(
            command,
            shell=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,  # line-buffered (best-effort; depends on child process buffering too)
        )

        assert process.stdout is not None
        for line in process.stdout:
            print(line, end="", flush=True)

        returncode = process.wait()
        if returncode != 0:
            print("--ERROR--" * 20)

def perform_experiments(dataset):
    for method, params in configs.items():

        base_run_name = method + '_' + dataset + '_'
        tmp_params = deepcopy(params)
        if "RUN_NAME" in params.keys():
            base_run_name = base_run_name + tmp_params["RUN_NAME"]
            del tmp_params["RUN_NAME"]
        
        for params_vals in itertools.product(*tmp_params.values()):
            run_name = base_run_name
            arguments = f'--log_dir {EXP_LOGS_DIR} --dataset {dataset} --adaptation_method {method} '
            for param_name, val in zip(tmp_params.keys(), params_vals):
                arguments += '--' + param_name + ' ' + str(val) + ' '

                if param_name in args_to_run_name:
                    if len(run_name) != 0:
                        run_name = run_name + '_' + param_name.upper() + str(val).replace(' ', '')
                    else:
                        run_name = param_name.upper() + str(val).replace(' ', '')

            arguments += '--data_path ' + os.path.join(DATADIR, DATA_FOLDERS[dataset]) + ' '

            if len(run_name) > 0:
                arguments += '--model_name' + ' ' + run_name + ' '
            
            if 'kitti' in dataset:
                arguments += '--png '
                
            command = f"python adaptation.py {arguments}{common_args}"
            
            print(command)
            
            run_command(command, method, run_name, dataset)
                
if __name__ == "__main__":
    main()

