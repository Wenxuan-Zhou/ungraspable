"""
Set of functions and classes that are modified versions of existing ones in rlkit
Modified just for logging purpose
"""
import abc

from rlkit.core import logger, eval_util
import gtimer as gt
from rlkit.core.rl_algorithm import _get_epoch_timings
from rlkit.torch.torch_rl_algorithm import TorchBatchRLAlgorithm
from collections import OrderedDict
import numpy as np
import rlkit.pythonplusplus as ppp


class CustomTorchBatchRLAlgorithm(TorchBatchRLAlgorithm, metaclass=abc.ABCMeta):
    def _get_snapshot(self):
        """
        We don't save 'env' after each iteration (will crash since cython can't
        compress MujocoEnv into C primitives OTS)
        """
        snapshot = {}
        for k, v in self.trainer.get_snapshot().items():
            snapshot['trainer/' + k] = v
        for k, v in self.expl_data_collector.get_snapshot().items():
            if k == 'env':
                continue
            snapshot['exploration/' + k] = v
        for k, v in self.eval_data_collector.get_snapshot().items():
            if k == 'env':
                continue
            snapshot['evaluation/' + k] = v
        for k, v in self.replay_buffer.get_snapshot().items():
            snapshot['replay_buffer/' + k] = v
        return snapshot

    def _log_stats(self, epoch):
        logger.log("Epoch {} finished".format(epoch), with_timestamp=True)

        """
        Replay Buffer
        """
        logger.record_dict(
            self.replay_buffer.get_diagnostics(),
            prefix='replay_buffer/'
        )

        """
        Trainer
        """
        logger.record_dict(self.trainer.get_diagnostics(), prefix='trainer/')

        """
        Exploration
        """
        logger.record_dict(
            self.expl_data_collector.get_diagnostics(),
            prefix='exploration/'
        )
        expl_paths = self.expl_data_collector.get_epoch_paths()
        logger.record_dict(
            get_custom_generic_path_information(expl_paths),
            prefix="exploration/",
        )
        """
        Evaluation
        """
        logger.record_dict(
            self.eval_data_collector.get_diagnostics(),
            prefix='evaluation/',
        )
        eval_paths = self.eval_data_collector.get_epoch_paths()
        logger.record_dict(
            get_custom_generic_path_information(eval_paths),
            prefix="evaluation/",
        )

        """
        Misc
        """
        gt.stamp('logging', unique=False)
        logger.record_dict(_get_epoch_timings())
        logger.record_tabular('Epoch', epoch)
        logger.dump_tabular(with_prefix=False, with_timestamp=False)

    def _train(self):
        if self.min_num_steps_before_training > 0:
            init_expl_paths = self.expl_data_collector.collect_new_paths(
                self.max_path_length,
                self.min_num_steps_before_training,
                discard_incomplete_paths=False,
            )
            self.replay_buffer.add_paths(init_expl_paths)
            self.expl_data_collector.end_epoch(-1)

        for epoch in range(self._start_epoch, self.num_epochs):
            self.eval_data_collector.collect_new_paths(
                self.max_path_length,
                self.num_eval_steps_per_epoch,
                discard_incomplete_paths=True,
            )
            gt.stamp('evaluation sampling', unique=False)

            for _ in range(self.num_train_loops_per_epoch):
                new_expl_paths = self.expl_data_collector.collect_new_paths(
                    self.max_path_length,
                    self.num_expl_steps_per_train_loop,
                    discard_incomplete_paths=False,
                )
                gt.stamp('exploration sampling', unique=False)

                self.replay_buffer.add_paths(new_expl_paths)
                gt.stamp('data storing', unique=False)

                self.training_mode(True)
                for _ in range(self.num_trains_per_train_loop):
                    train_data = self.replay_buffer.random_batch(
                        self.batch_size)
                    self.trainer.train(train_data)
                gt.stamp('training', unique=False)
                self.training_mode(False)

            self._end_epoch(epoch)

            # Stop the experiment early if necessary
            if hasattr(self.expl_data_collector, 'early_stopping') and \
                    self.expl_data_collector.early_stopping(epoch):
                return


def get_custom_generic_path_information(paths, stat_prefix=''):
    """
    Get an OrderedDict with a bunch of statistic names and values.
    """
    statistics = OrderedDict()

    # Add extra stats
    final_reward = [path["rewards"][-1] for path in paths]
    statistics.update(eval_util.create_stats_ordered_dict('FinalReward', final_reward,
                                                          stat_prefix=stat_prefix))

    statistics[stat_prefix + 'Average Returns'] = eval_util.get_average_returns(paths)

    for info_key in ['env_infos', 'agent_infos']:
        if info_key in paths[0]:
            all_env_infos = [
                ppp.list_of_dicts__to__dict_of_lists(p[info_key])
                for p in paths
            ]
            for k in all_env_infos[0].keys():
                final_ks = np.array([info[k][-1] for info in all_env_infos])
                statistics.update(eval_util.create_stats_ordered_dict(
                    stat_prefix + k,
                    final_ks,
                    stat_prefix='{}_final/'.format(info_key),
                    exclude_max_min=True,
                ))
    return statistics
