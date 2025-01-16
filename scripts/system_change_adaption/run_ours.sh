WORKDIR=$(dirname "$(dirname "$(dirname "$(realpath "$0")")")")
cd ${WORKDIR}

GPU="5"

## ours
python exp/run_sca.py -H=4 -L=8 -fe=TCN -bal=True --data_dir ${WORKDIR}/data/D/ --output_base_path ${WORKDIR}/output/sca/ours/ --metrics_path ${WORKDIR}/data/D/metrics.norm.detect.drift.mean.pkl --input_clip_val 10.0  --max_epoch 500 --early_stopping_epoch_patience 100 --gpu $GPU
