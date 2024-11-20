WORKDIR=$(dirname "$(dirname "$(dirname "$(realpath "$0")")")")
cd ${WORKDIR}
python exp/concept_drift/run_iSQ.py --data_dir ${WORKDIR}/data/E/ --output_base_path ${WORKDIR}/output/concept_drift_adaption/iSQ/ --metrics_path ${WORKDIR}/data/E/metrics.norm.pkl