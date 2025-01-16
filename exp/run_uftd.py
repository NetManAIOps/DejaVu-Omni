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
    config.max_epoch = 500  # 3000
    config.early_stopping_epoch_patience = 100  # 500
    dataset = config.data_dir.parts[-1]
    if dataset == 'A1':
        non_recurring_fault_set_list = [{'DB Session'}, {'DB State'}, {'OS Network'}]
    elif dataset == 'A2':
        non_recurring_fault_set_list = [{'DB Session'}, {'DB State'}, {'OS Network'}, {'Docker CPU'}]
    elif dataset == 'B':
        non_recurring_fault_set_list = [{'OS Memory'}, {'OS Disk'}]
    elif dataset == 'C':
        non_recurring_fault_set_list = [{'Node CPU'}, {'Node Memory'}, {'Pod CPU'}, {'Pod Memory'}]
    print(f"Dataset: {dataset}, non_recurring_fault_set_list: {non_recurring_fault_set_list}")


    for i, non_recurring_fault_set in enumerate(non_recurring_fault_set_list):
        j  = random.randint(0, len(non_recurring_fault_set_list)-1)
        while j==i:
            j  = random.randint(0, len(non_recurring_fault_set_list)-1)
        config.non_recur_index_train = i
        config.non_recur_index_test = j
        non_recurring_fault_set_test = non_recurring_fault_set_list[j]
        config.output_dir = Path(config.output_base_path / f"{'_'.join(non_recurring_fault_set)}.{'_'.join(non_recurring_fault_set_test)}").resolve()
        train_exp_CFL(
            config,
            get_GAT_model,
            plot_model=False
        )
