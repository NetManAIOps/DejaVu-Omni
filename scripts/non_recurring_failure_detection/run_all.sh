cd /DejaVu-Omni
datasets=("A1" "A2" "B" "C" "D")
for dataset in "${datasets[@]}"
do
    echo "Processing dataset: $dataset"
    # DejaVu-Omni
    python exp/run_non_recurring_failure_detect.py --data_dir /DejaVu-Omni/data/${dataset}/ --output_base_path /DejaVu-Omni/output/non_recurring_failure_detection/${dataset}/ --dataset_split_ratio 0.4 0.2 0.4 --dataset_split_method recur --recur_score True --recur_loss contrative --recur_loss_weight 0.05 0
    # DejaVu-Omni w/o CL
    python exp/run_non_recurring_failure_detect.py --data_dir /DejaVu-Omni/data/${dataset}/ --output_base_path /DejaVu-Omni/output/non_recurring_failure_detection/${dataset}/ --dataset_split_ratio 0.4 0.2 0.4 --dataset_split_method recur --recur_score True --recur_loss contrative --recur_loss_weight 0 0
    # DejaVu-Omni w/o NFD
    python exp/run_non_recurring_failure_detect.py --data_dir /DejaVu-Omni/data/${dataset}/ --output_base_path /DejaVu-Omni/output/non_recurring_failure_detection/${dataset}/ --dataset_split_ratio 0.4 0.2 0.4 --dataset_split_method recur --recur_score False --recur_loss contrative --recur_loss_weight 0.05 0
    # MHGL
    python exp/run_non_recurring_failure_detect.py --data_dir /DejaVu-Omni/data/${dataset}/ --output_base_path /DejaVu-Omni/output/non_recurring_failure_detection/${dataset}/ --dataset_split_ratio 0.4 0.2 0.4 --dataset_split_method recur --recur_score False --recur_loss mhgl --recur_loss_weight 0.05 0.05
    # Kmeans
    python exp/run_non_recurring_failure_detect.py --data_dir /DejaVu-Omni/data/${dataset}/ --output_base_path /DejaVu-Omni/output/non_recurring_failure_detection/${dataset}/ --dataset_split_ratio 0.4 0.2 0.4 --dataset_split_method recur --recur_score False --recur_loss kmeans --recur_loss_weight 0 0
    # GMM
    python exp/run_non_recurring_failure_detect.py --data_dir /DejaVu-Omni/data/${dataset}/ --output_base_path /DejaVu-Omni/output/non_recurring_failure_detection/${dataset}/ --dataset_split_ratio 0.4 0.2 0.4 --dataset_split_method recur --recur_score False --recur_loss gmm --recur_loss_weight 0 0
    # NewUnit
done
