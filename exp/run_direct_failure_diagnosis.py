from DejaVu.config import DejaVuConfig
from DejaVu.models.get_model import get_DNN_model
from DejaVu.workflow import train_exp_CFL

if __name__ == '__main__':
    config = DejaVuConfig().parse_args()
    config.max_epoch = 500  # 3000
    config.early_stopping_epoch_patience = 100  # 500
    config.flush_dataset_cache = False
    print(f"[DEBUG] config:\n{config}")
    train_exp_CFL(config, get_DNN_model, plot_model=False)
