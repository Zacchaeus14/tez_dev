import os, json
import numpy as np
from accelerate.logging import get_logger

from tez import enums
from tez.callbacks import Callback
from tez.logger import logger
from glob import glob

logger = get_logger(__name__)


class EarlyStopping(Callback):
    def __init__(self, monitor, model_path, patience=5, mode="min", delta=0.001, save_weights_only=False,
                 save_total_limit=1):
        self.monitor = monitor
        self.patience = patience
        self.counter = 0
        self.mode = mode
        self.best_score = None
        self.early_stop = False
        self.delta = delta
        self.save_weights_only = save_weights_only
        self.model_path = model_path
        self.history = []
        if self.mode == "min":
            self.val_score = np.Inf
        else:
            self.val_score = -np.Inf

        if self.monitor.startswith("train_"):
            self.model_state = "train"
            self.monitor_value = self.monitor[len("train_"):]
        elif self.monitor.startswith("valid_"):
            self.model_state = "valid"
            self.monitor_value = self.monitor[len("valid_"):]
        else:
            raise Exception("monitor must start with train_ or valid_")
        self.save_total_limit = save_total_limit
        self.save_counter = 0
        self.save_scores = {}

    def check(self, tez_trainer):
        epoch_score = tez_trainer.metrics[self.model_state][self.monitor_value]
        if self.save_total_limit != 1:
            self.save_scores[self.save_counter] = epoch_score
            self.save_checkpoint(epoch_score, tez_trainer)
            self.save_counter += 1
            return
        if self.mode == "min":
            score = -1.0 * epoch_score
        else:
            score = np.copy(epoch_score)

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(epoch_score, tez_trainer)
        elif score < self.best_score + self.delta:
            self.counter += 1
            logger.info(f"EarlyStopping counter: {self.counter}/{self.patience}")
            if self.counter >= self.patience:
                tez_trainer.model_state = enums.ModelState.END
        else:
            self.best_score = score
            self.save_checkpoint(epoch_score, tez_trainer)
            self.counter = 0

    def on_valid_epoch_end(self, tez_trainer):
        if tez_trainer.config.val_strategy == "epoch":
            return
        self.check(tez_trainer)

    def on_epoch_end(self, tez_trainer):
        if tez_trainer.config.val_strategy == "batch":
            return
        self.check(tez_trainer)

    def save_checkpoint(self, epoch_score, tez_trainer):
        if epoch_score in [-np.inf, np.inf, -np.nan, np.nan]:
            return
        improvement_string = f"{self.val_score:.5f} -> {epoch_score:.5f}. Saving model!"
        logger.info(improvement_string)
        self.history.append(improvement_string)
        if self.save_total_limit == 1:
            tez_trainer.save(self.model_path, weights_only=self.save_weights_only)
        else:
            tez_trainer.save(f'{self.model_path}_{self.save_counter}', weights_only=self.save_weights_only)
            self.evict_checkpoint()
            with open(f'{self.model_path}-save_scores.json', 'w') as f:
                json.dump(self.save_scores, f, indent=4)
        self.val_score = epoch_score

    def evict_checkpoint(self):
        fps = glob(f'{self.model_path}_*')
        if self.save_total_limit == -1 or len(fps) <= self.save_total_limit:
            return
        counters = [int(fp.split('_')[-1]) for fp in fps]
        scores = [self.save_scores[c] for c in counters]
        cs = list(zip(counters, scores))
        cs.sort(key=lambda x: x[1])
        if self.mode == "min":
            fp_del = f'{self.model_path}_{cs[-1][0]}'
        else:
            fp_del = f'{self.model_path}_{cs[0][0]}'
        logger.info(f"save total limit = {self.save_total_limit}, found {len(fps)}, deleting", fp_del)
        os.remove(fp_del)
