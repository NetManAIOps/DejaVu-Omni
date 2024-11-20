WORKDIR=$(dirname "$(dirname "$(dirname "$(realpath "$0")")")")
cd ${WORKDIR}

datasets=("A2" "B" "C" "D")
for i in {1..1}
do
    for dataset in "${datasets[@]}"
    do
        echo "Processing dataset: $dataset, for $i time..."
        # python exp/run_GAT_node_classification.py -H=4 -L=8 -fe=GRU -bal=True --data_dir ${WORKDIR}/data/${dataset}/ --output_base_path ${WORKDIR}/output/fault_localization/${dataset}/dejavu_omni_GRU_aug/ --dataset_split_ratio 0.4 0.2 0.4 --FI_feature_dim 3 -aug=True
        python exp/run_GAT_node_classification.py -H=4 -L=8 -fe=GRU -bal=True --data_dir ${WORKDIR}/data/${dataset}/ --output_base_path ${WORKDIR}/output/fault_localization/${dataset}/dejavu_omni_MoE/ --dataset_split_ratio 0.4 0.2 0.4 --FI_feature_dim 3
    done
done