from DejaVu.config import DejaVuConfig
from DejaVu.models import get_GAT_model
from DejaVu.workflow import train_exp_CFL
import os

if __name__ == '__main__':
    # logger.info("Disable JIT because of DGL")
    # set_jit_enabled(False)
    config = DejaVuConfig().parse_args()
    with open(os.path.join(config.data_dir, 'drift.txt'), 'r') as f:
        config.drift_time = int(f.read())
        print(f"config.drift_time: {config.drift_time}")
    config.dataset_split_method = 'drift'
    config.dataset_split_ratio = (0.52, 0.2, 0.28)
    config.input_clip_val = 20
    print(f"[DEBUG] config:\n{config}")
    train_exp_CFL(config, get_GAT_model, plot_model=False)
