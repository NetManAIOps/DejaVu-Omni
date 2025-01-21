WORKDIR=$(dirname "$(dirname "$(dirname "$(realpath "$0")")")")
cd ${WORKDIR}

GPU="5"
datasets=("A1" "A2" "B" "C")


for dataset in "${datasets[@]}"
do
    echo "Processing dataset: $dataset"
    # w/o GAT
    python exp/run_direct_uftd.py -fe=TCN -bal=True --data_dir ${WORKDIR}/data/${dataset}/ --output_base_path ${WORKDIR}/output/ablation_study/uftd/${dataset}/wo_GAT/ --dataset_split_ratio 0.4 0.2 0.4 --FI_feature_dim 3 --input_clip_val 10.0 --input_clip_val 10.0 --recur_score True --recur_loss contrastive --recur_loss_weight 0.05 0 --dataset_split_method recur --gpu $GPU
    # w/o L_c
    python exp/run_uftd.py -H=4 -L=8 -fe=TCN -bal=True --data_dir ${WORKDIR}/data/${dataset}/ --output_base_path ${WORKDIR}/output/ablation_study/uftd/${dataset}/wo_Lc/ --dataset_split_ratio 0.4 0.2 0.4 --FI_feature_dim 3  --input_clip_val 10.0 --dataset_split_method recur --gpu $GPU
    # GRU + 1-D CNN
    python exp/run_uftd.py -H=4 -L=8 -fe=GRU -bal=True --data_dir ${WORKDIR}/data/${dataset}/ --output_base_path ${WORKDIR}/output/ablation_study/uftd/${dataset}/GRU/ --dataset_split_ratio 0.4 0.2 0.4 --FI_feature_dim 3 --input_clip_val 10.0 --recur_score True --recur_loss contrastive --recur_loss_weight 0.05 0 --dataset_split_method recur --gpu $GPU
    # w/o data augmentation
    python exp/run_uftd.py -H=4 -L=8 -fe=TCN --data_dir ${WORKDIR}/data/${dataset}/ --output_base_path ${WORKDIR}/output/ablation_study/uftd/${dataset}/wo_aug/ --dataset_split_ratio 0.4 0.2 0.4 --FI_feature_dim 3 --input_clip_val 10.0 --recur_score True --recur_loss contrastive --recur_loss_weight 0.05 0 --dataset_split_method recur --gpu $GPU

    # ours
    python exp/run_uftd.py -H=4 -L=8 -fe=TCN -bal=True --data_dir ${WORKDIR}/data/${dataset}/ --output_base_path ${WORKDIR}/output/uftd/${dataset}/ours/ --dataset_split_ratio 0.4 0.2 0.4  --FI_feature_dim 3  --input_clip_val 10.0 --dataset_split_method recur --recur_score True --recur_loss contrastive --recur_loss_weight 0.05 0 --gpu $GPU
    # MHGL
    python exp/run_uftd.py -H=4 -L=8 -fe=TCN -bal=True --data_dir ${WORKDIR}/data/${dataset}/ --output_base_path ${WORKDIR}/output/uftd/${dataset}/mhgl/ --dataset_split_ratio 0.4 0.2 0.4 --FI_feature_dim 3  --input_clip_val 10.0 --dataset_split_method recur --recur_score False --recur_loss mhgl --recur_loss_weight 0.05 0.05 --gpu $GPU
    # Kmeans
    python exp/run_uftd.py -H=4 -L=8 -fe=TCN -bal=True --data_dir ${WORKDIR}/data/${dataset}/ --output_base_path ${WORKDIR}/output/uftd/${dataset}/kmeans/ --dataset_split_ratio 0.4 0.2 0.4 --FI_feature_dim 3  --input_clip_val 10.0 --dataset_split_method recur --recur_score False --recur_loss kmeans --recur_loss_weight 0 0 --gpu $GPU
    # GMM
    python exp/run_uftd.py -H=4 -L=8 -fe=TCN -bal=True --data_dir ${WORKDIR}/data/${dataset}/ --output_base_path ${WORKDIR}/output/uftd/${dataset}/gmm/ --dataset_split_ratio 0.4 0.2 0.4 --FI_feature_dim 3  --input_clip_val 10.0 --dataset_split_method recur --recur_score False --recur_loss gmm --recur_loss_weight 0 0 --gpu $GPU
done
