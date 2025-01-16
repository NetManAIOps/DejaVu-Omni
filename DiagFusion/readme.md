## Diagfusion

## Pre: Build up Environment

> need python version==3.8

```
pip install -r requirements.txt
```

> Tips: fasttext=0.9.2 on win32, please download `.whl` file, click [me](https://www.lfd.uci.edu/~gohlke/pythonlibs/#fasttext)

> then run `pip install [fasttext.whl]`

> use pip to install fasttext=0.9.2 on linux (ubuntu)
## 1. Run Diagfusion

### 1.1 On Gaia Dataset

```shell
python main.py --config gaia_config.yaml

```

### 1.2 On AIops22 Dataset

```shell
python main.py --config aiops22_config.yaml

```

### 1.3 Preprocess AIops22 Data
+ copy 22Aiops dataset to the root directory, for example
```bash
copy -r your_dataset/ ./
```
+ In `config/aiops22_config.yaml`, modify the value of `raw_data.dataset_entry`, for example
```yaml
...
raw_data:
    dataset_entry: dirname_of_your_dataset
    ...
```
+ Run command as follow
```shell
python main_aiops22_process_data.py --config aiops22_config.yaml
```
### 1.4 Preprocess platform Data
+ copy platform dataset to the root directory, for example
```bash
copy -r your_dataset/ ./
```
+ In `config/platform_config.yaml`, modify the value of `raw_data.dataset_entry`, for example
```yaml
...
raw_data:
    dataset_entry: dirname_of_your_dataset
    ...
```
+ Run command as follow
```shell
python main_platform_process_data.py --config platform_config.yaml
```

## 2. Directory structure
```
├.
├── config 配置文件
├── data 预处理数据存储位置 & dgl结果存储位置
│   ├── aiops22 2022年挑战赛初赛
│   │   └── demo
│   │       └── demo_1100
│   │           ├── anomalies
│   │           ├── dgl
│   │           │   └── stratification_10
│   │           │       └── 9
│   │           │           ├── evaluations
│   │           │           │   ├── anomaly
│   │           │           │   └── instance
│   │           │           └── preds
│   │           ├── fasttext
│   │           │   └── temp
│   │           └── parse
│   ├── gaia GAIA数据集
│   │   └── demo
│   │       └── demo_1100
│   │           ├── anomalies
│   │           ├── dgl
│   │           │   └── stratification_10
│   │           │       └── 9
│   │           │           ├── evaluations
│   │           │           │   ├── anomaly
│   │           │           │   └── instance
│   │           │           └── preds
│   │           ├── fasttext
│   │           │   └── temp
│   │           └── parse
│   └── platform NKUAiops 平台数据集
│       └── demo
│           └── demo_1100
│               ├── anomalies
│               ├── dgl
│               │   └── stratification_10
│               │       └── 9
│               │           ├── evaluations
│               │           │   ├── anomaly
│               │           │   └── instance
│               │           └── preds
│               ├── fasttext
│               │   └── temp
│               └── parse
├── detector 3sigma检测代码
├── drain3 drain代码
├── models GNN模型
└── transforms 数据预处理与格式转换
    ├── events 转换为event
    ├── feature 特征相关工具
    ├── process_on_aiops 2022挑战赛初赛预处理代码
    ├── process_on_gaia 最初代码的GAIA数据集预处理代码
    └── process_on_platform NKUAiops数据集预处理代码
```
