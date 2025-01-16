WORKDIR=$(dirname "$(dirname "$(dirname "$(realpath "$0")")")")
cd ${WORKDIR}

GPU="5"
datasets=("A1" "A2" "B" "C")


for dataset in "${datasets[@]}"
do
    echo "Processing dataset: $dataset"
    # ours
    python exp/run_uftd.py -H=4 -L=8 -fe=TCN -bal=True --data_dir ${WORKDIR}/data/${dataset}/ --output_base_path ${WORKDIR}/output/uftd/${dataset}/ours/ --dataset_split_ratio 0.4 0.2 0.4  --FI_feature_dim 3  --input_clip_val 10.0 --dataset_split_method recur --recur_score True --recur_loss contrative --recur_loss_weight 0.05 0 --gpu $GPU
done
