WORKDIR=$(dirname "$(dirname "$(dirname "$(realpath "$0")")")")
cd ${WORKDIR}

GPU="5"
datasets=("A1" "A2" "B" "C")

for dataset in "${datasets[@]}"
do
    echo "Processing dataset: $dataset"
    # w/o L_c
    python exp/run_GAT_failure_diagnosis.py -H=4 -L=8 -fe=TCN -bal=True --data_dir ${WORKDIR}/data/${dataset}/ --output_base_path ${WORKDIR}/output/ablation_study/failure_diagnosis/${dataset}/wo_Lc/ --dataset_split_ratio 0.4 0.2 0.4 --FI_feature_dim 3  --input_clip_val 10.0 --gpu $GPU
    # w/o GAT
    python exp/run_direct_failure_diagnosis.py -fe=TCN -bal=True --data_dir ${WORKDIR}/data/${dataset}/ --output_base_path ${WORKDIR}/output/ablation_study/failure_diagnosis/${dataset}/wo_GAT/ --dataset_split_ratio 0.4 0.2 0.4 --FI_feature_dim 3 --input_clip_val 10.0 --recur_score True --recur_loss contrative --recur_loss_weight 0.05 0 --gpu $GPU
    # w/o data augmentation
    python exp/run_GAT_failure_diagnosis.py -H=4 -L=8 -fe=TCN --data_dir ${WORKDIR}/data/${dataset}/ --output_base_path ${WORKDIR}/output/ablation_study/failure_diagnosis/${dataset}/wo_aug/ --dataset_split_ratio 0.4 0.2 0.4 --FI_feature_dim 3 --input_clip_val 10.0 --recur_score True --recur_loss contrative --recur_loss_weight 0.05 0 --gpu $GPU
    # GRU + 1-D CNN
    python exp/run_GAT_failure_diagnosis.py -H=4 -L=8 -fe=GRU -bal=True --data_dir ${WORKDIR}/data/${dataset}/ --output_base_path ${WORKDIR}/output/ablation_study/failure_diagnosis/${dataset}/GRU/ --dataset_split_ratio 0.4 0.2 0.4 --FI_feature_dim 3 --input_clip_val 10.0 --recur_score True --recur_loss contrative --recur_loss_weight 0.05 0 --gpu $GPU
    
    # ours
    python exp/run_GAT_failure_diagnosis.py -H=4 -L=8 -fe=TCN -bal=True --data_dir ${WORKDIR}/data/${dataset}/ --output_base_path ${WORKDIR}/output/failure_diagnosis/${dataset}/ours/ --dataset_split_ratio 0.4 0.2 0.4 --FI_feature_dim 3  --input_clip_val 10.0 --recur_score True --recur_loss contrative --recur_loss_weight 0.05 0 --gpu $GPU
    # Eadro
    python exp/DejaVu/run_Eadro.py --data_dir ${WORKDIR}/data/${dataset}/ --output_base_path ${WORKDIR}/output/failure_diagnosis/${dataset}/eadro/ --dataset_split_ratio 0.4 0.2 0.4
    # iSQUAD
    python exp/DejaVu/run_iSQ.py --data_dir ${WORKDIR}/data/${dataset}/ --output_base_path ${WORKDIR}/output/failure_diagnosis/${dataset}/iSQUAD/ --dataset_split_ratio 0.4 0.2 0.4 
    # JSS20
    python exp/DejaVu/run_JSS20.py --data_dir ${WORKDIR}/data/${dataset}/ --output_base_path ${WORKDIR}/output/failure_diagnosis/${dataset}/JSS20/ --dataset_split_ratio 0.4 0.2 0.4 
    # Decision Tree
    python exp/run_DT_failure_diagnosis.py --data_dir ${WORKDIR}/data/${dataset}/ --output_base_path ${WORKDIR}/output/failure_diagnosis/${dataset}/decision_tree/ --dataset_split_ratio 0.4 0.2 0.4
done