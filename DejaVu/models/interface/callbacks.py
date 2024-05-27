import copy
import io
import time
import numpy as np
import pandas as pd
from datetime import datetime
from functools import reduce
from typing import Callable, Any, Optional

import pytorch_lightning as pl
import pytz
from loguru import logger
from DejaVu.evaluation_metrics import top_1_accuracy, top_2_accuracy, top_3_accuracy, top_k_accuracy, MAR, get_rank
from DejaVu.models.interface.model_interface import DejaVuModelInterface
from failure_dependency_graph import FDG


class CFLLoggerCallback(pl.Callback):
    def __init__(self, epoch_freq: int = 1, print_func: Callable[[str], Any] = logger.info,
                 second_freq: float = 10):
        self.print = print_func
        self.epoch_freq = epoch_freq
        self.second_freq = second_freq
        self._last_display_time = time.time()
        self._need_display_on_this_epoch = False

    def on_train_epoch_start(self, trainer: 'pl.Trainer', pl_module: 'pl.LightningModule') -> None:
        if pl_module.current_epoch % self.epoch_freq != 0 and time.time() - self._last_display_time < self.second_freq:
            self._need_display_on_this_epoch = False
        else:
            self._last_display_time = time.time()
            self._need_display_on_this_epoch = True

    def on_train_epoch_end(
            self, trainer: 'pl.Trainer', pl_module: DejaVuModelInterface, unused: Optional = None
    ) -> None:
        if not self._need_display_on_this_epoch:
            return
        self.print(
            f"epoch={pl_module.current_epoch:<5.0f} "
            f"loss={trainer.callback_metrics.get('loss'):<10.4f}"
            f"A@1={trainer.callback_metrics.get('A@1', -1) * 100:<5.2f}% "
            f"A@2={trainer.callback_metrics.get('A@2', -1) * 100:<5.2f}% "
            f"A@3={trainer.callback_metrics.get('A@3', -1) * 100:<5.2f}% "
            f"A@5={trainer.callback_metrics.get('A@5', -1) * 100:<5.2f}% "
            f"MAR={trainer.callback_metrics.get('MAR', -1):<5.2f} "
        )

    def on_validation_epoch_end(self, trainer: 'pl.Trainer', pl_module: DejaVuModelInterface) -> None:
        self.print(
            f"epoch={pl_module.current_epoch:<5.0f} "
            f"val_loss={trainer.callback_metrics.get('val_loss', -1):<10.4f} "
            f"A@1={trainer.callback_metrics.get('A@1', -1) * 100:<5.2f}% "
            f"A@2={trainer.callback_metrics.get('A@2', -1) * 100:<5.2f}% "
            f"A@3={trainer.callback_metrics.get('A@3', -1) * 100:<5.2f}% "
            f"A@5={trainer.callback_metrics.get('A@5', -1) * 100:<5.2f}% "
            f"MAR={trainer.callback_metrics.get('MAR', -1):<5.2f} "
        )

    def on_test_epoch_end(self, trainer: 'pl.Trainer', pl_module: DejaVuModelInterface) -> None:
        labels_list = pl_module.labels_list
        preds_list = pl_module.preds_list
        output = io.StringIO()
        print(
            f"{'All failures:':<25}"
            f"A@1={top_1_accuracy(labels_list, preds_list) * 100:<5.2f}% "
            f"A@2={top_2_accuracy(labels_list, preds_list) * 100:<5.2f}% "
            f"A@3={top_3_accuracy(labels_list, preds_list) * 100:<5.2f}% "
            f"A@5={top_k_accuracy(labels_list, preds_list, k=5) * 100:<5.2f}% "
            f"MAR={MAR(labels_list, preds_list, max_rank=pl_module.fdg.n_failure_instances):<5.2f} ",
            file=output
        )
        if pl_module.config.dataset_split_method == 'recur':
            probs_list = pl_module.probs_list
            probs_list = [list(1./(1. + np.exp([-prob for prob in probs]))) for probs in probs_list]
            non_recurring_list = pl_module.non_recurring_list
            non_recurring_list_test = pl_module.non_recurring_list_test
            recurring_labels_list, recurring_preds_list = [], []
            non_recurring_labels_list, non_recurring_preds_list = [], []
            for fault_id, labels, preds in zip(pl_module.test_failure_ids, labels_list, preds_list):
                if fault_id in non_recurring_list or fault_id in non_recurring_list_test:
                    non_recurring_labels_list.append(labels)
                    non_recurring_preds_list.append(preds)
                else:
                    recurring_labels_list.append(labels)
                    recurring_preds_list.append(preds)
            non_recur_metrics = {
                "non_recur_A@1": top_1_accuracy(non_recurring_labels_list, non_recurring_preds_list),
                "non_recur_A@2": top_2_accuracy(non_recurring_labels_list, non_recurring_preds_list),
                "non_recur_A@3": top_3_accuracy(non_recurring_labels_list, non_recurring_preds_list),
                "non_recur_A@5": top_k_accuracy(non_recurring_labels_list, non_recurring_preds_list, k=5),
                "non_recur_MAR": MAR(non_recurring_labels_list, non_recurring_preds_list, max_rank=pl_module.fdg.n_failure_instances),
            }
            recur_metrics = {
                "recur_A@1": top_1_accuracy(recurring_labels_list, recurring_preds_list),
                "recur_A@2": top_2_accuracy(recurring_labels_list, recurring_preds_list),
                "recur_A@3": top_3_accuracy(recurring_labels_list, recurring_preds_list),
                "recur_A@5": top_k_accuracy(recurring_labels_list, recurring_preds_list, k=5),
                "recur_MAR": MAR(recurring_labels_list, recurring_preds_list, max_rank=pl_module.fdg.n_failure_instances),
            }
            print(
                f"{'Recurring failures:':<25}"
                f"A@1={recur_metrics['recur_A@1'] * 100:<5.2f}% "
                f"A@2={recur_metrics['recur_A@2'] * 100:<5.2f}% "
                f"A@3={recur_metrics['recur_A@3'] * 100:<5.2f}% "
                f"A@5={recur_metrics['recur_A@5'] * 100:<5.2f}% "
                f"MAR={recur_metrics['recur_MAR']:<5.2f} ",
                file=output
            )
            print(
                f"{'Non-recurring failures:':<25}"
                f"A@1={non_recur_metrics['non_recur_A@1'] * 100:<5.2f}% "
                f"A@2={non_recur_metrics['non_recur_A@2'] * 100:<5.2f}% "
                f"A@3={non_recur_metrics['non_recur_A@3'] * 100:<5.2f}% "
                f"A@5={non_recur_metrics['non_recur_A@5'] * 100:<5.2f}% "
                f"MAR={non_recur_metrics['non_recur_MAR']:<5.2f} ",
                file=output
            )
        elif pl_module.config.dataset_split_method == 'drift':
            probs_list = pl_module.probs_list
            probs_list = [list(1./(1. + np.exp([-prob for prob in probs]))) for probs in probs_list]
            drift_list = pl_module.drift_list
            drift_labels_list, drift_preds_list = [], []
            non_drift_labels_list, non_drift_preds_list = [], []
            for fault_id, labels, preds in zip(pl_module.test_failure_ids, labels_list, preds_list):
                if fault_id in drift_list:
                    drift_labels_list.append(labels)
                    drift_preds_list.append(preds)
                else:
                    non_drift_labels_list.append(labels)
                    non_drift_preds_list.append(preds)
            drift_metrics = {
                "drift_A@1": top_1_accuracy(drift_labels_list, drift_preds_list),
                "drift_A@2": top_2_accuracy(drift_labels_list, drift_preds_list),
                "drift_A@3": top_3_accuracy(drift_labels_list, drift_preds_list),
                "drift_A@5": top_k_accuracy(drift_labels_list, drift_preds_list, k=5),
                "drift_MAR": MAR(drift_labels_list, drift_preds_list, max_rank=pl_module.fdg.n_failure_instances),
            }
            non_drift_metrics = {
                "non_drift_A@1": top_1_accuracy(non_drift_labels_list, non_drift_preds_list),
                "non_drift_A@2": top_2_accuracy(non_drift_labels_list, non_drift_preds_list),
                "non_drift_A@3": top_3_accuracy(non_drift_labels_list, non_drift_preds_list),
                "non_drift_A@5": top_k_accuracy(non_drift_labels_list, non_drift_preds_list, k=5),
                "non_drift_MAR": MAR(non_drift_labels_list, non_drift_preds_list, max_rank=pl_module.fdg.n_failure_instances),
            }
            print(
                f"{'Failures after drift:':<25}"
                f"A@1={drift_metrics['drift_A@1'] * 100:<5.2f}% "
                f"A@2={drift_metrics['drift_A@2'] * 100:<5.2f}% "
                f"A@3={drift_metrics['drift_A@3'] * 100:<5.2f}% "
                f"A@5={drift_metrics['drift_A@5'] * 100:<5.2f}% "
                f"MAR={drift_metrics['drift_MAR']:<5.2f} ",
                file=output
            )
            print(
                f"{'Failures before drift:':<25}"
                f"A@1={non_drift_metrics['non_drift_A@1'] * 100:<5.2f}% "
                f"A@2={non_drift_metrics['non_drift_A@2'] * 100:<5.2f}% "
                f"A@3={non_drift_metrics['non_drift_A@3'] * 100:<5.2f}% "
                f"A@5={non_drift_metrics['non_drift_A@5'] * 100:<5.2f}% "
                f"MAR={non_drift_metrics['non_drift_MAR']:<5.2f} ",
                file=output
            )

        train_rc_ids = reduce(
            lambda a, b: a | b,
            [
                {pl_module.fdg.instance_to_gid(_) for _ in pl_module.fdg.root_cause_instances_of(fid)}
                for fid in pl_module.train_failure_ids
            ]
        )

        cdp: FDG = pl_module.fdg
        tz = pytz.timezone('Asia/Shanghai')

        if pl_module.config.dataset_split_method == 'recur':
            print(
                f"|{'id':4}|{'':<5}|{'FR':<3}|{'AR':<3}|{'recurring':<9}|{'recurring failure':<17}|{'timestamp':<25}|"
                f"{'root cause':<30}|{'rank-1':<20}|{'rank-2':<20}|{'rank-3':<20}|",
                file=output
            )
            fault_ids, is_recurs, is_corrects = [], [], []
            means, vars = [], []
            prob1s, prob2s, prob3s = [], [], []
            for probs, preds, fault_id, labels in zip(probs_list, preds_list, pl_module.test_failure_ids, labels_list):
                is_correct = preds[0] in labels
                ranks = get_rank(labels, preds, pl_module.fdg.n_failure_instances)
                is_recurring = all([_ in train_rc_ids for _ in labels])
                is_recurring_failure = fault_id not in non_recurring_list and fault_id not in non_recurring_list_test
                print(
                    f"|{fault_id:<4.0f}|"
                    f"{'✅' if is_correct else '❌':<4}|"
                    f"{min(ranks):3.0f}|"
                    f"{sum(ranks) / len(ranks):3.0f}|"
                    f"{is_recurring!s:<9}|"
                    f"{is_recurring_failure!s:<17}|"
                    f"{datetime.fromtimestamp(cdp.failure_at(fault_id)['timestamp']).astimezone(tz).isoformat():<25}|"
                    f"{','.join([cdp.gid_to_instance(_) for _ in labels]):<30}|"
                    f"{cdp.gid_to_instance(preds[0]):<20}|"
                    f"{cdp.gid_to_instance(preds[1]):<20}|"
                    f"{cdp.gid_to_instance(preds[2]):<20}|",
                    file=output
                )
                probs.sort(reverse=True)
                mean = np.mean(probs)
                var = np.var(probs)
                fault_ids.append(fault_id)
                is_recurs.append('recur' if is_recurring_failure else 'non-recur')
                is_corrects.append('True' if is_correct else 'False')
                means.append(mean)
                vars.append(var)
                prob1s.append(probs[0])
                prob2s.append(probs[1])
                prob3s.append(probs[2])
            probs_dict = {}
            probs_dict['fault_id'], probs_dict['failure_type'], probs_dict['correct'] = fault_ids, is_recurs, is_corrects
            probs_dict['prob-1'], probs_dict['prob-2'], probs_dict['prob-3'] = prob1s, prob2s, prob3s
            probs_dict['mean'], probs_dict['var'] = means, vars
            pl_module.probs_df = pd.DataFrame(probs_dict).sort_values(by=['fault_id'])
        elif pl_module.config.dataset_split_method == 'drift':
            print(
                f"|{'id':4}|{'':<5}|{'FR':<3}|{'AR':<3}|{'recurring':<9}|{'drift':<8}|{'timestamp':<25}|"
                f"{'root cause':<30}|{'rank-1':<20}|{'rank-2':<20}|{'rank-3':<20}|",
                file=output
            )
            fault_ids, is_drifts, is_corrects = [], [], []
            means, vars = [], []
            prob1s, prob2s, prob3s = [], [], []
            for probs, preds, fault_id, labels in zip(probs_list, preds_list, pl_module.test_failure_ids, labels_list):
                is_correct = preds[0] in labels
                ranks = get_rank(labels, preds, pl_module.fdg.n_failure_instances)
                is_recurring = all([_ in train_rc_ids for _ in labels])
                is_drift = fault_id in drift_list
                print(
                    f"|{fault_id:<4.0f}|"
                    f"{'✅' if is_correct else '❌':<4}|"
                    f"{min(ranks):3.0f}|"
                    f"{sum(ranks) / len(ranks):3.0f}|"
                    f"{is_recurring!s:<9}|"
                    f"{is_drift!s:<8}|"
                    f"{datetime.fromtimestamp(cdp.failure_at(fault_id)['timestamp']).astimezone(tz).isoformat():<25}|"
                    f"{','.join([cdp.gid_to_instance(_) for _ in labels]):<30}|"
                    f"{cdp.gid_to_instance(preds[0]):<20}|"
                    f"{cdp.gid_to_instance(preds[1]):<20}|"
                    f"{cdp.gid_to_instance(preds[2]):<20}|",
                    file=output
                )
                probs.sort(reverse=True)
                mean = np.mean(probs)
                var = np.var(probs)
                fault_ids.append(fault_id)
                is_drifts.append('after-drift' if is_drift else 'before-drift')
                is_corrects.append('True' if is_correct else 'False')
                means.append(mean)
                vars.append(var)
                prob1s.append(probs[0])
                prob2s.append(probs[1])
                prob3s.append(probs[2])
            probs_dict = {}
            probs_dict['fault_id'], probs_dict['is_drift'], probs_dict['correct'] = fault_ids, is_drifts, is_corrects
            probs_dict['prob-1'], probs_dict['prob-2'], probs_dict['prob-3'] = prob1s, prob2s, prob3s
            probs_dict['mean'], probs_dict['var'] = means, vars
            pl_module.probs_df = pd.DataFrame(probs_dict).sort_values(by=['fault_id'])
        else:
            print(
                f"|{'id':4}|{'':<5}|{'FR':<3}|{'AR':<3}|{'recurring':<9}|{'timestamp':<25}|"
                f"{'root cause':<70}|{'rank-1':<20}|{'rank-2':<20}|{'rank-3':<20}|",
                file=output
            )
            for preds, fault_id, labels in zip(preds_list, pl_module.test_failure_ids, labels_list):
                is_correct = preds[0] in labels
                ranks = get_rank(labels, preds, pl_module.fdg.n_failure_instances)
                is_recurring = all([_ in train_rc_ids for _ in labels])
                print(
                    f"|{fault_id:<4.0f}|"
                    f"{'✅' if is_correct else '❌':<5}|"
                    f"{min(ranks):3.0f}|"
                    f"{sum(ranks) / len(ranks):3.0f}|"
                    f"{is_recurring!s:<9}|"
                    f"{datetime.fromtimestamp(cdp.failure_at(fault_id)['timestamp']).astimezone(tz).isoformat():<25}|"
                    f"{','.join([cdp.gid_to_instance(_) for _ in labels]):<70}|"
                    f"{cdp.gid_to_instance(preds[0]):<20}|"
                    f"{cdp.gid_to_instance(preds[1]):<20}|"
                    f"{cdp.gid_to_instance(preds[2]):<20}|",
                    file=output
                )
        self.print(f"\n{output.getvalue()}")


class TestCallback(pl.Callback):
    def __init__(self, model: DejaVuModelInterface, second_freq: float = 30, epoch_freq: int = 100):
        super(TestCallback, self).__init__()
        self.epoch_freq = epoch_freq
        self.second_freq = second_freq
        self.trainer = pl.Trainer(
            callbacks=[
                CFLLoggerCallback()
            ]
        )
        self.model = copy.deepcopy(model)
        setattr(self.model, "metric_preprocessor", model.metric_preprocessor)

        self._last_time = time.time()
        self._last_epoch = 0

    def on_validation_epoch_end(self, trainer: 'pl.Trainer', pl_module: DejaVuModelInterface) -> None:
        epoch = pl_module.current_epoch
        if epoch - self._last_epoch > self.epoch_freq or time.time() - self._last_time > self.second_freq:
            self._last_time = time.time()
            self._last_epoch = epoch
            # self.model.load_state_dict(pl_module.state_dict())
            # logger.info(
            #     f"\n=======================Start Test at Epoch {pl_module.current_epoch} ======================"
            # )
            # logger.info(f"load model from {trainer.checkpoint_callback.best_model_path}")
            # self.trainer.test(
            #     self.model, dataloaders=pl_module.test_dataloader(), verbose=False,
            #     ckpt_path=trainer.checkpoint_callback.best_model_path
            # )
            # logger.info(
            #     f"\n======================= End  Test at Epoch {pl_module.current_epoch} ======================"
            # )
            try:
                self.model.load_state_dict(pl_module.state_dict())
                logger.info(
                    f"\n=======================Start Test at Epoch {pl_module.current_epoch} ======================"
                )
                logger.info(f"load model from {trainer.checkpoint_callback.best_model_path}")
                self.trainer.test(
                    self.model, dataloaders=pl_module.test_dataloader(), verbose=False,
                    ckpt_path=trainer.checkpoint_callback.best_model_path
                )
                logger.info(
                    f"\n======================= End  Test at Epoch {pl_module.current_epoch} ======================"
                )
            except Exception as e:
                logger.error(f"Encounter Exception during test: {e!r}")
