WORKDIR=$(dirname "$(dirname "$(dirname "$(realpath "$0")")")")
cd ${WORKDIR}
python exp/concept_drift/run_random_walk_single_metric.py --data_dir ${WORKDIR}/data/E/ --output_base_path ${WORKDIR}/output/concept_drift_adaption/random_walk_single_metric/ --metrics_path ${WORKDIR}/data/E/metrics.norm.pkl