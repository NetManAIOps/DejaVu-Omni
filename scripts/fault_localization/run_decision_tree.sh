cd /DejaVu-Omni
datasets=("A1" "A2" "B" "C" "D")
for dataset in "${datasets[@]}"
do
    echo "Processing dataset: $dataset"
    python exp/run_DT_node_classification.py --data_dir /DejaVu-Omni/data/${dataset}/ --output_base_path /DejaVu-Omni/output/fault_localization/${dataset}/eadro/ --dataset_split_ratio 0.4 0.2 0.4 
done
