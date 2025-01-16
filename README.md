# DejaVu-Omni
## Table of Contents
=================

- [DejaVu-Omni](#dejavu-omni)
  - [Table of Contents](#table-of-contents)
  - [Code](#code)
    - [Install](#install)
    - [Usage](#usage)
  - [Datasets](#datasets)
  - [Deployment and Failure Injection Scripts of Train-Ticket](#deployment-and-failure-injection-scripts-of-train-ticket)
  - [Citation](#citation)

  
## Code
### Install
1. All the software requirements are already pre-installed in the Docker image below. The requirements are also listed in `requirements.txt` and `requirements-dev.txt`. Note that `DGL 0.8` is not released yet when I did this work, so I installed `DGL 0.8` manually from the source code. PyTorch version should be equal to or greater than 1.11.0.
   ```bash
   docker pull lizytalk/dejavu
   ```
2. Pull the code from GitHub
   ```bash
   git pull https://github.com/NetManAIOps/DejaVu-Omni.git DejaVu-Omni
   ```
3. Download the datasets following the link in the GitHub repo and extract the datasets into `./DejaVu-Omni/data`
4. I use the command `realpath` in the example commands below, which is not bundled in macOS and Windows. On macOS, you can install it by `brew install coreutils`.
5. Start a Docker container with our image and enter its shell
   ```bash
   docker run -it --rm -v $(realpath DejaVu-Omni):/workspace lizytalk/dejavu bash
   ```
6. Run `direnv allow` in the shell of the Docker container to set the environment variables.
7. Run experiments in the shell of the Docker container following the usage table as follows.


### Usage

#### Root Cause Localization (RCL) & Failure Classification (FC)
For RCL and FC, we can use three datasets (A, B, and C). All scripts for DejaVu-Omni and baselines can be found in `scripts/failure_diagnosis`. Run `bash scripts/failure_diagnosis/run_all.sh` for all algorithms on three datasets.

The commands would print a `one-line summary` in the end, including the following fields: `RCL_A@1`, `RCL_A@2`, `RCL_A@3`, `RCL_A@5`, `RCL_MAR`, `FC_Precision`, `FC_Recall`, `FC_F1`, `Time`, `Epoch`, `Valid Epoch`, `output_dir`, `val_loss`, `val_MAR`, `val_A@1`, `command`, `git_commit_url`, which are the desrired results.

#### Unseen Failure Type Detection (UFTD)
For UFTD, we can use three datasets (A, B, and C). All scripts for DejaVu-Omni and baselines can be found in `scripts/uftd`. Run `bash scripts/unseen_failure_type_detection/run_all.sh` for all algorithms on three datasets.

Results will be found in a f1.1_score.txt in the output directory.

#### System Change Adaption (SCA)
For SCA, we can dataset D. Firstly drift metrics using `notebooks/system_change_adaption.ipynb`, and then run RCL and FC for failures after system changes. All scripts for DejaVu-Omni and baselines can be found in `scripts/system_change_adaption`. Run `bash scripts/system_change_adaption/run_all.sh` for all algorithms.

The commands would print a `one-line summary`, including the following fields: `drift_RCL_A@1`, `drift_RCL_A@2`, `drift_RCL_A@3`, `drift_RCL_A@5`, `drift_RCL_MAR`, `drift_FC_Precision`, `drift_FC_Recall`, `drift_FC_F1`, `non_drift_RCL_A@1`, `non_drift_RCL_A@2`, `non_drift_RCL_A@3`, `non_drift_RCL_A@5`, `non_drift_RCL_MAR`, `non_drift_FC_Precision`, `non_drift_FC_Recall`, `non_drift_FC_F1`, `RCL_A@3_down`, `RCL_MAR_up`, `FC_F1_down`,`Time`, `Epoch`, `Valid Epoch`, `output_dir`, `val_loss`, `val_MAR`, `val_A@1`, `command`, `git_commit_url`, which are the desrired results.

## Datasets

The datasets A, B, C, D are public at :
- https://www.dropbox.com/scl/fo/2jl3iem1dfuo3s7na7ebg/ALeZNJrcSg_jWvZsyPhBVMA?rlkey=ccfy5tnuwl18smrxt2m5lkgie&st=5q8yhby7&dl=0
In each dataset, `graph.yml` or `graphs/*.yml` are FDGs, `metrics.csv` and `metrics.pkl` are metrics, `metrics.norm.csv` and `metrics.norm.pkl` are normalized metrics, and `faults.csv` is failures (including ground truths).
`FDG.pkl` is a pickle of the FDG object, which contains all the above data.
Note that the pickle files are not compatible in different Python and Pandas versions. So if you cannot load the pickles, just ignore and delete them. They are only used to speed up data load.
Particulally, in dataset D which is used for concept drift adaption after system changes, `system_change.json` contains a change list, in which present a change with its start time and end time.

## Deployment and Failure Injection Scripts of Train-Ticket
https://github.com/lizeyan/train-ticket