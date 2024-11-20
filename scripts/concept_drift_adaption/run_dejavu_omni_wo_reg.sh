WORKDIR=$(dirname "$(dirname "$(dirname "$(realpath "$0")")")")
cd ${WORKDIR}
python exp/run_concept_drift.py --data_dir ${WORKDIR}/data/E/ --output_base_path ${WORKDIR}/output/concept_drift_adaption/dejavu_omni_wo_reg/ --metrics_path ${WORKDIR}/data/E/metrics.norm.pkl