from loguru import logger
from pyprof import profile, Profiler

from DejaVu.config import DejaVuConfig
from DejaVu.evaluation_metrics import get_evaluation_metrics_dict
from DejaVu.workflow import format_result_string
from JSS20.system_graph import GraphLibrary
from failure_dependency_graph import FDGModelInterface
import os


@profile('JSS20', report_printer=print)
def jss20(config: DejaVuConfig):
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
    mp = base.metric_preprocessor
    train_ids = base.train_failure_ids
    test_ids = base.test_failure_ids
    graph_library = GraphLibrary(cdp, train_ids[:], mp)
    labels = []
    preds = []
    for fid, (_, fault) in zip(test_ids, cdp.failures_df.iloc[test_ids[:]].iterrows()):
        labels.append({fault['root_cause_node']})
        preds.append(graph_library.query(fid))
    metrics = get_evaluation_metrics_dict(labels, preds, max_rank=cdp.n_failure_instances)
    logger.info(metrics)
    return metrics


def main(config: DejaVuConfig):
    metrics = jss20(config)
    profiler = Profiler().get('/JSS20')
    config.dataset_split_method = 'type'
    logger.info(format_result_string(metrics, profiler, config))


if __name__ == '__main__':
    main(DejaVuConfig().parse_args())
