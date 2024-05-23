cd ..
python exp/run_concept_drift.py --data_dir /SSF/data/E/ --output_base_path /SSF/output/concept_drift_adaption/dejavu_omni/ --metrics_path /SSF/data/E/metrics.norm.drift.ours.pkl
python exp/run_concept_drift.py --data_dir /SSF/data/E/ --output_base_path /SSF/output/concept_drift_adaption/icpp19/ --metrics_path /SSF/data/E/metrics.norm.drift.icpp.pkl
python exp/run_concept_drift.py --data_dir /SSF/data/E/ --output_base_path /SSF/output/concept_drift_adaption/stepwise/ --metrics_path /SSF/data/E/metrics.norm.drift.stepwise.pkl
python exp/run_concept_drift.py --data_dir /SSF/data/E/ --output_base_path /SSF/output/concept_drift_adaption/dejavu_omni_wo_reg/ --metrics_path /SSF/data/E/metrics.norm.pkl