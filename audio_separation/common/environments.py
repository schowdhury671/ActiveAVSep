r"""
This file hosts task-specific or trainer-specific environments for trainers.
All environments here should be a (direct or indirect ) subclass of Env class
in habitat. Customized environments should be registered using
``@baseline_registry.register_env(name="myEnv")` for reusability
"""

import time
from typing import Optional, Type, Any, Dict, Iterator, List, Tuple, Union
import logging

import gym
# import numba
import numpy as np
# from gym import spaces
from gym.spaces.dict_space import Dict as SpaceDict

import habitat
from habitat import Config, Dataset

from habitat.config import Config
from habitat.core.dataset import Dataset, Episode, EpisodeIterator
from habitat.core.embodied_task import EmbodiedTask, Metrics
from habitat.core.simulator import Observations, Simulator
from habitat.datasets import make_dataset
from habitat.sims import make_sim
from habitat.tasks import make_task

from audio_separation.common.baseline_registry import baseline_registry
from habitat.sims.habitat_simulator.actions import HabitatSimActions


class EnvCustom:
    r"""Fundamental environment class for `habitat`.

    :data observation_space: ``SpaceDict`` object corresponding to sensor in
        sim and task.
    :data action_space: ``gym.space`` object corresponding to valid actions.

    All the information  needed for working on embodied tasks with simulator
    is abstracted inside `Env`. Acts as a base for other derived environment
    classes. `Env` consists of three major components: ``dataset`` (`episodes`), ``simulator`` (`sim`) and `task` and connects all the three components
    together.
    """

    observation_space: SpaceDict
    action_space: SpaceDict
    _config: Config
    _dataset: Optional[Dataset]
    _episodes: List[Type[Episode]]
    _current_episode_index: Optional[int]
    _current_episode: Optional[Type[Episode]]
    _episode_iterator: Optional[Iterator]
    _sim: Simulator
    _task: EmbodiedTask
    _max_episode_seconds: int
    _max_episode_steps: int
    _elapsed_steps: int
    _episode_start_time: Optional[float]
    _episode_over: bool

    def __init__(
        self, config: Config, dataset: Optional[Dataset] = None
    ) -> None:
        """Constructor

        :param config: config for the environment. Should contain id for
            simulator and ``task_name`` which are passed into ``make_sim`` and
            ``make_task``.
        :param dataset: reference to dataset for task instance level
            information. Can be defined as :py:`None` in which case
            ``_episodes`` should be populated from outside.
        """

        assert config.is_frozen(), (
            "Freeze the config before creating the "
            "environment, use config.freeze()."
        )
        self._config = config
        self._dataset = dataset
        self._current_episode_index = None
        if self._dataset is None and config.DATASET.TYPE:
            self._dataset = make_dataset(
                id_dataset=config.DATASET.TYPE, config=config.DATASET
            )
        self._episodes = self._dataset.episodes if self._dataset else []
        self._current_episode = None
        iter_option_dict = {
            k.lower(): v
            for k, v in config.ENVIRONMENT.ITERATOR_OPTIONS.items()
        }
        self._episode_iterator = self._dataset.get_episode_iterator(
            **iter_option_dict
        )

        # load the first scene if dataset is present
        if self._dataset:
            assert (
                len(self._dataset.episodes) > 0
            ), "dataset should have non-empty episodes list"
            self._config.defrost()
            self._config.SIMULATOR.SCENE = self._dataset.episodes[0].scene_id
            self._config.freeze()

        self._sim = make_sim(
            id_sim=self._config.SIMULATOR.TYPE, config=self._config.SIMULATOR
        )
        self._task = make_task(
            self._config.TASK.TYPE,
            config=self._config.TASK,
            sim=self._sim,
            dataset=self._dataset,
        )
        self.observation_space = SpaceDict(
            {
                **self._sim.sensor_suite.observation_spaces.spaces,
                **self._task.sensor_suite.observation_spaces.spaces,
            }
        )
        self.action_space = self._task.action_space
        self._max_episode_seconds = (
            self._config.ENVIRONMENT.MAX_EPISODE_SECONDS
        )
        self._max_episode_steps = self._config.ENVIRONMENT.MAX_EPISODE_STEPS
        self._elapsed_steps = 0
        self._episode_start_time: Optional[float] = None
        self._episode_over = False

    @property
    def current_episode(self) -> Type[Episode]:
        assert self._current_episode is not None
        return self._current_episode

    @current_episode.setter
    def current_episode(self, episode: Type[Episode]) -> None:
        self._current_episode = episode

    @property
    def episode_iterator(self) -> Iterator:
        return self._episode_iterator

    @episode_iterator.setter
    def episode_iterator(self, new_iter: Iterator) -> None:
        self._episode_iterator = new_iter

    @property
    def episodes(self) -> List[Type[Episode]]:
        return self._episodes

    @episodes.setter
    def episodes(self, episodes: List[Type[Episode]]) -> None:
        assert (
            len(episodes) > 0
        ), "Environment doesn't accept empty episodes list."
        self._episodes = episodes

    @property
    def sim(self) -> Simulator:
        return self._sim

    @property
    def episode_start_time(self) -> Optional[float]:
        return self._episode_start_time

    @property
    def episode_over(self) -> bool:
        return self._episode_over

    @property
    def task(self) -> EmbodiedTask:
        return self._task

    @property
    def _elapsed_seconds(self) -> float:
        assert (
            self._episode_start_time
        ), "Elapsed seconds requested before episode was started."
        return time.time() - self._episode_start_time

    def get_metrics(self) -> Metrics:
        return self._task.measurements.get_metrics()

    def _past_limit(self) -> bool:
        if (
            self._max_episode_steps != 0
            and self._max_episode_steps <= self._elapsed_steps
        ):
            return True
        elif (
            self._max_episode_seconds != 0
            and self._max_episode_seconds <= self._elapsed_seconds
        ):
            return True
        return False

    def _reset_stats(self) -> None:
        self._episode_start_time = time.time()
        self._elapsed_steps = 0
        self._episode_over = False

    def reset(self, is_train=True,) -> Observations:
        r"""Resets the environments and returns the initial observations.

        :return: initial observations from the environment.
        """
        self._reset_stats()

        assert len(self.episodes) > 0, "Episodes list is empty"

        self._current_episode = next(self._episode_iterator)
        self.reconfigure(self._config, is_train=is_train,)

        observations = self.task.reset(episode=self.current_episode)
        self._task.measurements.reset_measures(
            episode=self.current_episode, task=self.task
        )

        return observations

    def _update_step_stats(self) -> None:
        self._elapsed_steps += 1
        self._episode_over = not self._task.is_episode_active
        if self._past_limit():
            self._episode_over = True

        if self.episode_iterator is not None and isinstance(
            self.episode_iterator, EpisodeIterator
        ):
            self.episode_iterator.step_taken()

    def step(
        self, action: Union[int, str, Dict[str, Any]], **kwargs
    ) -> Observations:
        r"""Perform an action in the environment and return observations.

        :param action: action (belonging to `action_space`) to be performed
            inside the environment. Action is a name or index of allowed
            task's action and action arguments (belonging to action's
            `action_space`) to support parametrized and continuous actions.
        :return: observations after taking action in environment.
        """

        assert (
            self._episode_start_time is not None
        ), "Cannot call step before calling reset"
        assert (
            self._episode_over is False
        ), "Episode over, call reset before calling step"

        # Support simpler interface as well
        if isinstance(action, str) or isinstance(action, (int, np.integer)):
            action = {"action": action}

        observations = self.task.step(
            action=action, episode=self.current_episode
        )

        self._task.measurements.update_measures(
            episode=self.current_episode, action=action, task=self.task
        )

        self._update_step_stats()

        return observations

    def seed(self, seed: int) -> None:
        self._sim.seed(seed)
        self._task.seed(seed)

    def reconfigure(self, config: Config, is_train: bool = True,) -> None:
        self._config = config

        self._config.defrost()
        self._config.SIMULATOR = self._task.overwrite_sim_config(
            self._config.SIMULATOR, 
            self.current_episode,
            is_train=is_train,
        )
        self._config.freeze()

        self._sim.reconfigure(self._config.SIMULATOR)

    def render(self, mode="rgb") -> np.ndarray:
        return self._sim.render(mode)

    def close(self) -> None:
        self._sim.close()


class RLEnvCustom(gym.Env):
    r"""Reinforcement Learning (RL) environment class which subclasses ``gym.Env``.

    This is a wrapper over `Env` for RL users. To create custom RL
    environments users should subclass `RLEnv` and define the following
    methods: `get_reward_range()`, `get_reward()`, `get_done()`, `get_info()`.

    As this is a subclass of ``gym.Env``, it implements `reset()` and
    `step()`.
    """

    _env: EnvCustom # Env, EnvCustom

    def __init__(
        self, config: Config, dataset: Optional[Dataset] = None
    ) -> None:
        """Constructor

        :param config: config to construct `Env`
        :param dataset: dataset to construct `Env`.
        """

        self._env = EnvCustom(config, dataset)
        self.observation_space = self._env.observation_space
        self.action_space = self._env.action_space
        self.reward_range = self.get_reward_range()

    @property
    def habitat_env(self) -> EnvCustom:
        return self._env

    @property
    def episodes(self) -> List[Type[Episode]]:
        return self._env.episodes

    @property
    def current_episode(self) -> Type[Episode]:
        return self._env.current_episode

    @episodes.setter
    def episodes(self, episodes: List[Type[Episode]]) -> None:
        self._env.episodes = episodes

    def reset(self, is_train=True) -> Observations:
        return self._env.reset(is_train=is_train)

    def get_reward_range(self):
        r"""Get min, max range of reward.

        :return: :py:`[min, max]` range of reward.
        """
        raise NotImplementedError

    def get_reward(self, observations: Observations) -> Any:
        r"""Returns reward after action has been performed.

        :param observations: observations from simulator and task.
        :return: reward after performing the last action.

        This method is called inside the `step()` method.
        """
        raise NotImplementedError

    def get_done(self, observations: Observations) -> bool:
        r"""Returns boolean indicating whether episode is done after performing
        the last action.

        :param observations: observations from simulator and task.
        :return: done boolean after performing the last action.

        This method is called inside the step method.
        """
        raise NotImplementedError

    def get_info(self, observations) -> Dict[Any, Any]:
        r"""..

        :param observations: observations from simulator and task.
        :return: info after performing the last action.
        """
        raise NotImplementedError

    def step(self, *args, **kwargs) -> Tuple[Observations, Any, bool, dict]:
        r"""Perform an action in the environment.

        :return: :py:`(observations, reward, done, info)`
        """

        observations = self._env.step(*args, **kwargs)
        reward = self.get_reward(observations)
        done = self.get_done(observations)
        info = self.get_info(observations)

        return observations, reward, done, info

    def seed(self, seed: Optional[int] = None) -> None:
        self._env.seed(seed)

    def render(self, mode: str = "rgb") -> np.ndarray:
        return self._env.render(mode)

    def close(self) -> None:
        self._env.close()


# def get_env_class(env_name: str) -> Type[habitat.RLEnv]:
def get_env_class(env_name: str) -> Type[RLEnvCustom]:
    r"""Return environment class based on name.

    Args:
        env_name: name of the environment.

    Returns:
        Type[habitat.RLEnv]: env class.
    """
    return baseline_registry.get_env(env_name)


@baseline_registry.register_env(name="AAViDSSEnv")
# class AAViDSSEnv(habitat.RLEnv):
class AAViDSSEnv(RLEnvCustom):
    def __init__(self, 
                 config: Config, 
                 dataset: Optional[Dataset] = None,
                 is_train: bool = True,):
        self._rl_config = config.RL
        self._config = config
        self._core_env_config = config.TASK_CONFIG

        self._is_train = is_train

        super().__init__(self._core_env_config, dataset)

    def reset(self):
        self._env_step = 0
        observation = super().reset(is_train=self._is_train)
        logging.debug(super().current_episode)
        return observation

    def step(self, *args, **kwargs):
        observation, reward, done, info = super().step(*args, **kwargs)
        self._env_step += 1
        return observation, reward, done, info

    def get_reward_range(self):
        return (
            float('-inf'),
            0
        )

    def get_reward(self, observations):
        return 0

    def get_done(self, observations):
        done = False
        if self._env.episode_over:
            done = True
        return done

    def get_info(self, observations):
        return self.habitat_env.get_metrics()

    # for data collection
    def get_current_episode_id(self):
        return self.habitat_env.current_episode.episode_id
