from DejaVu.config import DejaVuConfig
from failure_dependency_graph import get_non_recur_lists
from failure_dependency_graph.failure_dependency_graph import FDG
from DejaVu.models import get_GAT_model
from DejaVu.workflow import train_exp_CFL
from pathlib import Path
import random


if __name__ == '__main__':
    # logger.info("Disable JIT because of DGL")
    # set_jit_enabled(False)
    config = DejaVuConfig().parse_args()
    config.dataset_split_method='recur'
    dataset = config.data_dir.parts[-1]
    fdg, real_paths = FDG.load(
        dir_path=config.data_dir,
        graph_path=config.graph_config_path, metrics_path=config.metrics_path, faults_path=config.faults_path,
        return_real_path=True, 
        use_anomaly_direction_constraint=config.use_anomaly_direction_constraint,
        loaded_FDG=None,
        flush_dataset_cache=config.flush_dataset_cache,
    )
    non_recurring_fault_set_list = get_non_recur_lists(fdg.failures_df)

    method = ''
    if config.recur_score:
        method = 'dejavu_omni' if config.recur_loss_weight[0] > 0 else 'dejavu_omni_wo_cl'
    else:
        if config.recur_loss == 'mhgl':
            method = 'mhgl'
        elif config.recur_loss == 'contrative':
            method = 'dejavu_omni_wo_nfd'
        elif config.recur_loss == 'kmeans':
            method = 'kmeans'
        elif config.recur_loss == 'gmm':
            method = 'gmm'
    for i, non_recurring_fault_set in enumerate(non_recurring_fault_set_list):
        j  = random.randint(0, len(non_recurring_fault_set_list)-1)
        while j==i:
            j  = random.randint(0, len(non_recurring_fault_set_list)-1)
        config.non_recur_index_train = i
        config.non_recur_index_test = j
        non_recurring_fault_set_test = non_recurring_fault_set_list[j]
        config.output_dir = Path(config.output_base_path / method / f"{'_'.join(non_recurring_fault_set)}.{'_'.join(non_recurring_fault_set_test)}").resolve()
        train_exp_CFL(
            config,
            get_GAT_model,
            plot_model=False
        )
