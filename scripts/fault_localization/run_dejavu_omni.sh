cd /DejaVu-Omni
datasets=("A1" "A2" "B" "C" "D")
for dataset in "${datasets[@]}"
do
    echo "Processing dataset: $dataset"
    python exp/run_GAT_node_classification.py -H=4 -L=8 -fe=GRU -bal=True --data_dir /DejaVu-Omni/data/${dataset}/ --output_base_path /DejaVu-Omni/output/fault_localization/${dataset}/dejavu_omni/ --dataset_split_ratio 0.4 0.2 0.4 
done