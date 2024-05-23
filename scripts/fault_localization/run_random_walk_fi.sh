cd /SSF
datasets=("A1" "A2" "B" "C" "D")
for dataset in "${datasets[@]}"
do
    echo "Processing dataset: $dataset"
    python exp/DejaVu/run_random_walk_failure_instance.py --data_dir /SSF/data/${dataset}/ --window_size 60 10 --anomaly_score_aggregation_method=min --corr_aggregation_method=max --output_base_path /SSF/output/fault_localization/${dataset}/RandomWalkFI/ --dataset_split_ratio 0.4 0.2 0.4 
done