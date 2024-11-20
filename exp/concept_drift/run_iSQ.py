from loguru import logger
from pyprof import profile, Profiler

from DejaVu.evaluation_metrics import get_evaluation_metrics_dict
from DejaVu.workflow import format_result_string
from failure_dependency_graph import FDG, FDGModelInterface
from iSQUAD.config import ISQUADConfig
from iSQUAD.iSQ import ISQUARD
from metric_preprocess import MetricPreprocessor
import os


@profile('exp_iSQ', report_printer=print)
def exp_iSQ(config: ISQUADConfig):
    with open(os.path.join(config.data_dir, 'drift.txt'), 'r') as f:
        config.drift_time = int(f.read())
        print(f"config.drift_time: {config.drift_time}")
    config.dataset_split_method = 'drift'
    config.dataset_split_ratio = (0.7, 0.3, 0)
    config.input_clip_val = 20
    config.flush_dataset_cache = False
    print(f"[DEBUG] config:\n{config}")

    logger.add(config.output_dir / 'log')
    base = FDGModelInterface(config)
    cdp = base.fdg
    fe = base.metric_preprocessor
    isq = ISQUARD(fdg=cdp, config=config, mp=fe)
    y_preds = isq()
    y_trues = []
    for fault_id in isq.test_fault_ids:
        y_trues.append(set(map(cdp.instance_to_gid, cdp.failures_df.iloc[fault_id]["root_cause_node"].split(";"))))
    for y_true, y_pred in zip(y_trues, y_preds):
        logger.info(
            f"{';'.join(map(cdp.gid_to_instance, y_true)):<30}"
            f"|{', '.join(map(cdp.gid_to_instance, y_pred[:5])):<50}"
        )
    metrics = get_evaluation_metrics_dict(y_trues, y_preds, max_rank=cdp.n_failure_instances)
    return metrics


def main(config: ISQUADConfig):
    metrics = exp_iSQ(config)
    profiler = Profiler.get('/exp_iSQ')
    config.dataset_split_method = 'type'
    logger.info(format_result_string(metrics, profiler, config))


if __name__ == '__main__':
    main(ISQUADConfig().parse_args())
