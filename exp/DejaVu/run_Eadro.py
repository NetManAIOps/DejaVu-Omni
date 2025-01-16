import torch
from loguru import logger
from pyprof import profile, Profiler

from failure_dependency_graph import FDG
from Eadro.config import EadroConfig
from Eadro.model import EadroModel
from Eadro.workflow import BaseModel
from metric_preprocess import MetricPreprocessor
from DejaVu.workflow import format_result_string


@profile('Eadro')
def exp_Eadro(config: EadroConfig):
    logger.add(config.output_dir / 'log')
    logger.info(config)
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
    return metrics


if __name__ == '__main__':
    with profile("Experiment", report_printer=logger.info):
        config = EadroConfig().parse_args()
        metrics = exp_Eadro(config)
    profiler = Profiler.get("/Experiment/Eadro")
    logger.info(format_result_string(metrics, profiler, config))