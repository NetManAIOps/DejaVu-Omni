cd /SSF
datasets=("A1" "A2" "B" "C" "D")
for dataset in "${datasets[@]}"
do
    echo "Processing dataset: $dataset"
    # DejaVu-Omni
    python exp/run_GAT_node_classification.py -H=4 -L=8 -fe=GRU -bal=True --data_dir /SSF/data/${dataset}/ --output_base_path /SSF/output/fault_localization/${dataset}/dejavu_omni/ --dataset_split_ratio 0.4 0.2 0.4 
    # Eadro
    python exp/DejaVu/run_Eadro.py --data_dir /SSF/data/${dataset}/ --output_base_path /SSF/output/fault_localization/${dataset}/eadro/ --dataset_split_ratio 0.4 0.2 0.4
    # JSS'20
    python exp/DejaVu/run_JSS20.py --data_dir /SSF/data/${dataset}/ --output_base_path /SSF/output/fault_localization/${dataset}/JSS20/ --dataset_split_ratio 0.4 0.2 0.4
    # iSQUAD
    python exp/DejaVu/run_iSQ.py --data_dir /SSF/data/${dataset}/ --output_base_path /SSF/output/fault_localization/${dataset}/iSQUAD/ --dataset_split_ratio 0.4 0.2 0.4 
    # Decision Tree
    python exp/run_DT_node_classification.py --data_dir /SSF/data/${dataset}/ --output_base_path /SSF/output/fault_localization/${dataset}/eadro/ --dataset_split_ratio 0.4 0.2 0.4
    # RandomWalk@Metric
    python exp/DejaVu/run_random_walk_single_metric.py --data_dir /SSF/data/${dataset}/ --window_size 60 10 --score_aggregation_method=min --output_base_path /SSF/output/fault_localization/${dataset}/RandomWalkMetric/ --dataset_split_ratio 0.4 0.2 0.4 
    # RandomWalk@FI
    python exp/DejaVu/run_random_walk_failure_instance.py --data_dir /SSF/data/${dataset}/ --window_size 60 10 --anomaly_score_aggregation_method=min --corr_aggregation_method=max --output_base_path /SSF/output/fault_localization/${dataset}/RandomWalkFI/ --dataset_split_ratio 0.4 0.2 0.4 
done
