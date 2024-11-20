from random_walk_single_metric import workflow
from random_walk_single_metric.config import RandomWalkSingleMetricConfig
import os


if __name__ == '__main__':
    config = RandomWalkSingleMetricConfig().parse_args()
    with open(os.path.join(config.data_dir, 'drift.txt'), 'r') as f:
        config.drift_time = int(f.read())
        print(f"config.drift_time: {config.drift_time}")
    config.dataset_split_method = 'drift'
    config.dataset_split_ratio = (0.7, 0.3, 0)
    config.input_clip_val = 20
    config.flush_dataset_cache = False
    print(f"[DEBUG] config:\n{config}")

    workflow(config)
