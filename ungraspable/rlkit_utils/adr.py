from abc import ABC
from collections import deque

import numpy as np
from rlkit.core import logger
from rlkit.samplers.data_collector.base import PathCollector

from ungraspable.rlkit_utils.rlkit_custom import get_custom_generic_path_information


class Param(object):
    def __init__(self, name, expl_env, eval_env, target_val, inc,
                 buffer_length=1, init_val=None, pair=None, priority=1, obs_key=None):
        # obs_key is used when creating the environment
        self.name = name
        self.pair = pair
        self.expl_env = expl_env
        self.eval_env = eval_env
        self.target_val = np.array(target_val)
        self.inc = inc
        self.priority = priority
        if init_val is None:
            self.curr_val = getattr(self.eval_env, self.name)
        else:
            self.curr_val = init_val
            self.reset_boundary()
        self.init_val = self.curr_val
        self.finished = True if (self.curr_val == target_val) else False
        self.buffer = deque(maxlen=buffer_length)
        self.pair_val = None

    def get_snapshot(self):
        # For saving
        snapshot = dict()
        snapshot['name'] = self.name
        snapshot['pair'] = self.pair
        snapshot['target_val'] = self.target_val
        snapshot['inc'] = self.inc
        snapshot['finished'] = self.finished
        snapshot['curr_val'] = self.curr_val
        snapshot['buffer'] = self.buffer
        return snapshot

    def try_boundary(self):
        setattr(self.expl_env, self.name, self.curr_val)
        if self.pair is not None:
            assert self.pair_val is None
            self.pair_val = getattr(self.expl_env, self.pair)
            setattr(self.expl_env, self.pair, self.curr_val)
        return self.curr_val

    def update_boundary(self, score, threshold_upper, threshold_lower):
        # Change back the pair parameter first
        if self.pair is not None:
            setattr(self.expl_env, self.pair, self.pair_val)
            self.pair_val = None

        # Set to new parameters or reset to old parameters
        self.buffer.append(score)
        average_score = sum(self.buffer) / len(self.buffer)
        min_score = min(self.buffer)
        if len(self.buffer) == self.buffer.maxlen and min_score >= threshold_upper:
            self.set_new_boundary()
            self.buffer.clear()
            return True
        elif self.curr_val != self.init_val and \
                len(self.buffer) == self.buffer.maxlen and average_score < threshold_lower:
            self.set_new_boundary(decrement=True)
            self.buffer.clear()
            return False
        else:
            self.reset_boundary()
            return None

    def set_new_boundary(self, decrement=False):
        if decrement:
            self.curr_val = self.curr_val - self.inc
            if (self.inc < 0 and self.curr_val >= self.init_val) or \
                    (self.inc > 0 and self.curr_val <= self.init_val):
                self.curr_val = self.init_val
            logger.log(f"ADR: Decreased boundary. {self.name}: {self.curr_val}.")
        else:
            if self.curr_val == self.target_val:
                self.finished = True
                logger.log(f"ADR: Finished boundary. {self.name}: {self.curr_val}.")
            self.curr_val = self.curr_val + self.inc
            if (self.inc < 0 and self.curr_val <= self.target_val) or \
                    (self.inc > 0 and self.curr_val >= self.target_val):
                self.curr_val = self.target_val
            logger.log(f"ADR: Updated new boundary. {self.name}: {self.curr_val}.")
        setattr(self.eval_env, self.name, self.curr_val)
        setattr(self.expl_env, self.name, self.curr_val)

    def reset_boundary(self):
        setattr(self.eval_env, self.name, self.curr_val)
        setattr(self.expl_env, self.name, self.curr_val)


class AdrPathCollector(PathCollector, ABC):
    def __init__(self, base_path_collector: PathCollector, expl_env, eval_env, parameters,
                 threshold_key=None, threshold_value_upper=None, threshold_value_lower=None, buffer_length=1,
                 adr_prob=0.5, enable_early_stopping=False):
        super().__init__()
        self._base_path_collector = base_path_collector

        self.params = dict()
        self.unfinished_keys = []
        self.finished_keys = []
        self.setup_parameters(parameters, buffer_length, expl_env, eval_env)
        self.unfinished_keys_prob = []
        self.update_unfinished_keys_prob()
        self.update_history = []
        self.increase_counts = 0
        self.decrease_counts = 0

        self.threshold_key = threshold_key
        self.threshold_value_upper = threshold_value_upper
        self.threshold_value_lower = threshold_value_lower
        self.adr_prob = adr_prob

        self.enable_early_stopping = enable_early_stopping

    def setup_parameters(self, parameters, buffer_length, expl_env, eval_env):
        for param_key in parameters:
            self.params[param_key] = Param(param_key, buffer_length=buffer_length, expl_env=expl_env, eval_env=eval_env,
                                           **parameters[param_key])
            if self.params[param_key].finished:
                self.finished_keys.append(param_key)
            else:
                self.unfinished_keys.append(param_key)

        logger.log(f"ADR: Finished keys '{self.finished_keys}'. Remaining keys {self.unfinished_keys}.")

    def update_unfinished_keys_prob(self):
        prob = []
        for k in self.unfinished_keys:
            prob.append(self.params[k].priority)
        self.unfinished_keys_prob = np.array(prob)

    def collect_new_paths(
            self,
            max_path_length,
            num_steps,
            discard_incomplete_paths,
    ):
        if np.random.rand() > self.adr_prob or len(self.unfinished_keys) == 0:
            return self._base_path_collector.collect_new_paths(max_path_length, num_steps, discard_incomplete_paths)

        # ADR Procedure: Randomly pick a boundary and set env to the new boundary
        param_key = np.random.choice(self.unfinished_keys,
                                     p=self.unfinished_keys_prob / np.sum(self.unfinished_keys_prob))
        eval_param = self.params[param_key].try_boundary()
        paths = self._base_path_collector.collect_new_paths(max_path_length, num_steps, discard_incomplete_paths)
        paths_stats = get_custom_generic_path_information(paths)
        result = self.params[param_key].update_boundary(paths_stats[self.threshold_key],
                                                        self.threshold_value_upper,
                                                        self.threshold_value_lower)
        if result is True:
            self.increase_counts += 1
        elif result is False:
            self.decrease_counts += 1

        if self.params[param_key].finished is True:
            # Remove from list if this parameter reaches the target value
            self.unfinished_keys.remove(param_key)
            self.finished_keys.append(param_key)
            self.update_unfinished_keys_prob()
            logger.log(f"ADR: Removing key '{param_key}'. Remaining keys {self.unfinished_keys}.")
        return paths

    def end_epoch(self, epoch):
        self._base_path_collector.end_epoch(epoch)
        self.update_history.append(self.increase_counts)

    def get_diagnostics(self):
        log_dict = {'increase_counts': self.increase_counts, 'decrease_counts': self.decrease_counts}
        for k, v in self.params.items():
            log_dict[k] = v.curr_val

        logger.record_dict(log_dict, prefix="ADR/")
        return self._base_path_collector.get_diagnostics()

    def get_snapshot(self):
        snapshot = self._base_path_collector.get_snapshot()
        snapshot['adr_params'] = {k: self.params[k].get_snapshot() for k in self.params.keys()}
        return snapshot

    def get_epoch_paths(self):
        return self._base_path_collector.get_epoch_paths()

    def early_stopping(self, epoch):
        if not self.enable_early_stopping:
            return False

        if len(self.update_history) > 0:
            adr_results = np.array(self.update_history)
            if len(adr_results) > 50 and adr_results[-1] == adr_results[-10]:
                # Terminate experiment if there is no adr update for 10 epochs.
                return True
        return False


def get_obs_keys(adr_config):
    # Convert adr config into a list of keys to be included in the observation
    obs_key_list = []
    for param_key in adr_config['parameters']:
        param = adr_config['parameters'][param_key]
        if 'obs_key' in param and param['obs_key'] not in obs_key_list:
            obs_key_list.append(param['obs_key'])

    logger.log(f"Adaptive policy with additional obs:{obs_key_list}")
    return obs_key_list
