WORKDIR=$(dirname "$(dirname "$(dirname "$(realpath "$0")")")")
cd ${WORKDIR}

GPU="5"

## w/o CPD
python exp/run_concept_drift.py -H=4 -L=8 -fe=TCN -bal=True --data_dir ${WORKDIR}/data/D/ --output_base_path ${WORKDIR}/output/ablation_study/concept_drift_adaption/wo_CPD/ --metrics_path ${WORKDIR}/data/D/metrics.norm.drift.mean.pkl --input_clip_val 10.0  --max_epoch 500 --early_stopping_epoch_patience 100 --gpu $GPU
## w/o MA
python exp/run_concept_drift.py -H=4 -L=8 -fe=TCN -bal=True --data_dir ${WORKDIR}/data/D/ --output_base_path ${WORKDIR}/output/ablation_study/concept_drift_adaption/wo_MA/ --metrics_path ${WORKDIR}/data/D/metrics.norm.pkl --max_epoch 500 --early_stopping_epoch_patience 100 --gpu $GPU
## w Stepwise
python exp/run_concept_drift.py -H=4 -L=8 -fe=TCN -bal=True --data_dir ${WORKDIR}/data/D/ --output_base_path ${WORKDIR}/output/ablation_study/concept_drift_adaption/w_stepwise/ --metrics_path ${WORKDIR}/data/D/metrics.norm.drift.stepwise.pkl --max_epoch 500 --early_stopping_epoch_patience 100 --gpu $GPU

## ours
python exp/run_concept_drift.py -H=4 -L=8 -fe=TCN -bal=True --data_dir ${WORKDIR}/data/D/ --output_base_path ${WORKDIR}/output/concept_drift_adaption/ours/ --metrics_path ${WORKDIR}/data/D/metrics.norm.detect.drift.mean.pkl --input_clip_val 10.0  --max_epoch 500 --early_stopping_epoch_patience 100 --gpu $GPU
## Eadro
python exp/concept_drift/run_Eadro.py --data_dir ${WORKDIR}/data/D/ --output_base_path ${WORKDIR}/output/concept_drift_adaption/eadro/ --metrics_path ${WORKDIR}/data/D/metrics.norm.pkl --max_epoch 500 --early_stopping_epoch_patience 100 --gpu $GPU
## JSS'20
python exp/concept_drift/run_JSS20.py --data_dir ${WORKDIR}/data/D/ --output_base_path ${WORKDIR}/output/concept_drift_adaption/JSS20/ --metrics_path ${WORKDIR}/data/D/metrics.norm.pkl --gpu $GPU
## iSQUAD
python exp/concept_drift/run_iSQ.py --data_dir ${WORKDIR}/data/D/ --output_base_path ${WORKDIR}/output/concept_drift_adaption/iSQ/ --metrics_path ${WORKDIR}/data/D/metrics.norm.pkl --gpu $GPU
## Decision Tree
python exp/concept_drift/run_DT_failure_diagnosis.py --data_dir ${WORKDIR}/data/D/ --output_base_path ${WORKDIR}/output/concept_drift_adaption/decision_tree/ --metrics_path ${WORKDIR}/data/D/metrics.norm.pkl --input_clip_val 10.0 --gpu $GPU