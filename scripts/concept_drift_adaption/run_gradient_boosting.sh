WORKDIR=$(dirname "$(dirname "$(dirname "$(realpath "$0")")")")
cd ${WORKDIR}
python exp/concept_drift/run_GB_node_classification.py --data_dir ${WORKDIR}/data/E/ --output_base_path ${WORKDIR}/output/concept_drift_adaption/gradient_boosting/ --metrics_path ${WORKDIR}/data/E/metrics.norm.pkl