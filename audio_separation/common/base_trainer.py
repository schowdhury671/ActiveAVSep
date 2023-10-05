import os
import time
from typing import ClassVar, Dict, List

import torch

from habitat import Config, logger
from audio_separation.common.tensorboard_utils import TensorboardWriter
from audio_separation.common.utils import poll_checkpoint_folder


class BaseTrainer:
    r"""Generic trainer class that serves as a base template for more
    specific trainer classes like RL trainer, SLAM or imitation learner.
    Includes only the most basic functionality.
    """

    supported_tasks: ClassVar[List[str]]

    def train(self) -> None:
        raise NotImplementedError

    def eval(self) -> None:
        raise NotImplementedError

    def save_checkpoint(self, file_name) -> None:
        raise NotImplementedError

    def load_checkpoint(self, checkpoint_path, *args, **kwargs) -> Dict:
        raise NotImplementedError


class BaseRLTrainer(BaseTrainer):
    r"""Base trainer class for RL trainers. Future RL-specific
    methods should be hosted here.
    """
    device: torch.device
    config: Config
    val_config: Config
    video_option: List[str]
    _flush_secs: int

    def __init__(self, config: Config, val_config: Config):
        super().__init__()
        assert config is not None, "needs config file to initialize trainer"
        self.config = config
        if val_config is not None:
            self.val_config = val_config
        else:
            self.val_config = config
        self._flush_secs = 30

    @property
    def flush_secs(self):
        return self._flush_secs

    @flush_secs.setter
    def flush_secs(self, value: int):
        self._flush_secs = value

    def train(self) -> None:
        raise NotImplementedError

    def eval(self, eval_interval=1, prev_ckpt_ind=-1, separate_eval=False) -> None:
        r"""Main method of trainer evaluation. Calls _eval_checkpoint() that
        is specified in Trainer class that inherits from BaseRLTrainer

        Returns:
            None
        """
        self.device = (
            torch.device("cuda", self.val_config.TORCH_GPU_ID)
            if torch.cuda.is_available()
            else torch.device("cpu")
        )

        if "tensorboard" in self.val_config.VIDEO_OPTION:
            assert (
                len(self.val_config.TENSORBOARD_DIR) > 0
            ), "Must specify a tensorboard directory for video display"
        if "disk" in self.val_config.VIDEO_OPTION:
            assert (
                len(self.val_config.VIDEO_DIR) > 0
            ), "Must specify a directory for storing videos on disk"

        with TensorboardWriter(
            self.val_config.TENSORBOARD_DIR, flush_secs=self.flush_secs
        ) as writer:
            if not separate_eval:
                return self._eval_checkpoint(self.val_config.EVAL_CKPT_PATH_DIR, writer, separate_eval=separate_eval)
            
            if os.path.isfile(self.val_config.EVAL_CKPT_PATH_DIR):
                # evaluate singe checkpoint
                print("coming to single checkpoint loop ******************")
                agg_stats_mean = self._eval_checkpoint(self.val_config.EVAL_CKPT_PATH_DIR, writer, separate_eval=separate_eval)
            else:
                # evaluate multiple checkpoints in order
                print("coming to multi-checkpoint loop************************")
                while True:
                    current_ckpt = None
                    while current_ckpt is None:
                        current_ckpt = poll_checkpoint_folder(
                            self.val_config.EVAL_CKPT_PATH_DIR, prev_ckpt_ind, eval_interval
                        )
                        time.sleep(2)  # sleep for 2 secs before polling again
                    logger.info(f"=======current_ckpt: {current_ckpt}=======")
                    prev_ckpt_ind += eval_interval
                    
                    #print("@@@@@@@@@@@@@@@@@prev_ckpt_ind ", prev_ckpt_ind)
                    
                    print("after first line")
                    agg_stats_mean = self._eval_checkpoint(
                        checkpoint_path=current_ckpt,
                        writer=writer,
                        checkpoint_index=prev_ckpt_ind,
                        separate_eval=separate_eval
                    )
                    print("after second line")

        return agg_stats_mean

    def _setup_eval_config(self, checkpoint_config: Config) -> Config:
        r"""Sets up and returns a merged config for evaluation. Config
            object saved from checkpoint is merged into config file specified
            at evaluation time with the following overwrite priority:
                  eval_opts > ckpt_opts > eval_cfg > ckpt_cfg
            If the saved config is outdated, only the eval config is returned.

        Args:
            checkpoint_config: saved config from checkpoint.

        Returns:
            Config: merged config for eval.
        """

        config = self.val_config.clone()

        ckpt_cmd_opts = checkpoint_config.CMD_TRAILING_OPTS
        eval_cmd_opts = config.CMD_TRAILING_OPTS

        try:
            config.merge_from_other_cfg(checkpoint_config)
            config.merge_from_other_cfg(self.config)
            config.merge_from_list(ckpt_cmd_opts)
            config.merge_from_list(eval_cmd_opts)
            
        except KeyError:
            logger.info("Saved config is outdated, using solely eval config")
            config = self.val_config.clone()
            config.merge_from_list(eval_cmd_opts)
        if config.TASK_CONFIG.DATASET.SPLIT == "train":
            config.TASK_CONFIG.defrost()
            config.TASK_CONFIG.DATASET.SPLIT = "val"

        config.TASK_CONFIG.SIMULATOR.AGENT_0.defrost()
        config.TASK_CONFIG.SIMULATOR.AGENT_0.SENSORS = self.val_config.SENSORS
        config.freeze()

        return config

    def _eval_checkpoint(
        self,
        checkpoint_path: str,
        writer: TensorboardWriter,
        checkpoint_index: int = 0,
        separate_eval=False
    ) -> None:
        r"""Evaluates a single checkpoint. Trainer algorithms should
        implement this.

        Args:
            checkpoint_path: path of checkpoint
            writer: tensorboard writer object for logging to tensorboard
            checkpoint_index: index of cur checkpoint for logging

        Returns:
            None
        """
        raise NotImplementedError

    def save_checkpoint(self, file_name) -> None:
        raise NotImplementedError

    def load_checkpoint(self, checkpoint_path, *args, **kwargs) -> Dict:
        raise NotImplementedError

    @staticmethod
    def _pause_envs(
        envs_to_pause,
        envs,
        test_recurrent_hidden_states,
        not_done_masks,
        current_episode_reward,
        prev_actions,
        batch,
        rgb_frames,
    ):
        # pausing self.envs with no new episode
        if len(envs_to_pause) > 0:
            state_index = list(range(envs.num_envs))
            for idx in reversed(envs_to_pause):
                state_index.pop(idx)
                envs.pause_at(idx)

            # indexing along the batch dimensions
            test_recurrent_hidden_states = test_recurrent_hidden_states[
                :, state_index
            ]
            not_done_masks = not_done_masks[state_index]
            current_episode_reward = current_episode_reward[state_index]
            prev_actions = prev_actions[state_index]

            for k, v in batch.items():
                batch[k] = v[state_index]

            rgb_frames = [rgb_frames[i] for i in state_index]

        return (
            envs,
            test_recurrent_hidden_states,
            not_done_masks,
            current_episode_reward,
            prev_actions,
            batch,
            rgb_frames,
        )

