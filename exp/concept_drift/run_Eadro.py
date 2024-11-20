import torch
from loguru import logger
from pyprof import profile, Profiler

from failure_dependency_graph import FDG
from Eadro.config import EadroConfig
from Eadro.model import EadroModel
from Eadro.workflow import BaseModel
from metric_preprocess import MetricPreprocessor
import os


@profile('Eadro')
def exp_Eadro(config: EadroConfig):
    with open(os.path.join(config.data_dir, 'drift.txt'), 'r') as f:
        config.drift_time = int(f.read())
        print(f"config.drift_time: {config.drift_time}")
    config.dataset_split_method = 'drift'
    # config.dataset_split_ratio = (0.52, 0.2, 0.28)
    config.dataset_split_ratio = (0.7, 0.3, 0)
    config.input_clip_val = 20
    config.flush_dataset_cache = False
    print(f"[DEBUG] config:\n{config}")

    logger.add(config.output_dir / 'log')
    fdg = FDG.load(
        dir_path=config.data_dir,
        graph_path=config.graph_config_path, metrics_path=config.metrics_path, faults_path=config.faults_path,
        return_real_path=False,
        use_anomaly_direction_constraint=config.use_anomaly_direction_constraint,
        flush_dataset_cache=config.flush_dataset_cache
    )
    mp = MetricPreprocessor(fdg, granularity=60)
    device='cuda' if torch.cuda.is_available() else 'cpu'
    eadro = EadroModel(cdp=fdg, config=config, mp=mp)
    train_dl, valid_dl, test_dl = eadro.train_dataloader, eadro.val_dataloader, eadro.test_dataloader

    model = BaseModel(eadro, lr=config.init_lr, epoches=config.max_epoch, patience=config.early_stopping_epoch_patience, result_dir=config.output_dir, logger=logger, device=device)
    metrics, converge = model.fit(train_dl, valid_dl, test_dl, evaluation_epoch=config.valid_epoch_freq)
    logger.info(
        f"* Best score got at epoch: {converge}\n"
        f"A@1: {metrics['A@1'] * 100:.2f}\n"
        f"A@2: {metrics['A@2'] * 100:.2f}\n"
        f"A@3: {metrics['A@3'] * 100:.2f}\n"
        f"A@5: {metrics['A@5'] * 100:.2f}\n"
        f"MAR: {metrics['MAR']:.2f}\n"
    )
    return metrics


if __name__ == '__main__':
    with profile("Experiment", report_printer=logger.info):
        config = EadroConfig().parse_args()
        metrics = exp_Eadro(config)
    profiler = Profiler.get("/Experiment/Eadro")
    logger.info(
        f"command output one-line summary: "
        f"{metrics['A@1'] * 100:.2f},{metrics['A@2'] * 100:.2f},{metrics['A@3'] * 100:.2f},"
        f"{metrics['A@5'] * 100:.2f},{metrics['MAR']:.2f},"
        f"{profiler.total},{''},{''},{config.output_dir},{''},{''},{''},"
        f"{config.get_reproducibility_info()['command_line']},"
        f"{config.get_reproducibility_info().get('git_url', '')}"
    )