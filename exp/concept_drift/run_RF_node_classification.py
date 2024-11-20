from sklearn.ensemble import RandomForestClassifier

from DejaVu.config import DejaVuConfig
from DejaVu.models.get_model import ClassifierProtocol
from failure_dependency_graph import FDG
from DejaVu.workflow import train_exp_sklearn_classifier
import os


def get_model(cdp: FDG, config: DejaVuConfig) -> ClassifierProtocol:
    return RandomForestClassifier(verbose=0)

if __name__ == '__main__':
    config = DejaVuConfig().parse_args()
    with open(os.path.join(config.data_dir, 'drift.txt'), 'r') as f:
        config.drift_time = int(f.read())
        print(f"config.drift_time: {config.drift_time}")
    config.dataset_split_method = 'drift'
    config.dataset_split_ratio = (0.7, 0.3, 0)
    config.input_clip_val = 20
    config.flush_dataset_cache = False
    print(f"[DEBUG] config:\n{config}")

    train_exp_sklearn_classifier(config, get_model)
