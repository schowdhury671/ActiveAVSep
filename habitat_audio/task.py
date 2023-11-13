from typing import Any, Type, Union

import gzip
import json


import numpy as np
import math
from gym import spaces

from habitat.config import Config
from habitat.core.dataset import Episode
from habitat.tasks.nav.nav import NavigationTask, SimulatorTaskAction, Measure, EmbodiedTask, PointGoalSensor
from habitat.core.registry import registry
from habitat.core.simulator import (
    Sensor,
    SensorTypes,
    Simulator,
)
from habitat.tasks.utils import (
    cartesian_to_polar,
    quaternion_from_coeff,
    quaternion_rotate_vector,
)
from habitat.sims.habitat_simulator.actions import HabitatSimActions
from habitat.utils.visualizations import maps
# from habitat.tasks.nav.nav import NavigationTask, Measure, EmbodiedTask, PointGoalSensor, SimulatorTaskAction

IS_MOVING_SOURCE = True

def merge_sim_episode_config(
    sim_config: Config, episode: Type[Episode]
) -> Any:
    
    sim_config.defrost()
    # here's where the scene update happens, extract the scene name out of the path
    sim_config.SCENE = episode.scene_id
    sim_config.freeze()
    if (
        episode.start_position is not None
        and episode.start_rotation is not None
    ):
        agent_name = sim_config.AGENTS[sim_config.DEFAULT_AGENT_ID]
        agent_cfg = getattr(sim_config, agent_name)
        agent_cfg.defrost()
        agent_cfg.START_POSITION = episode.start_position
        agent_cfg.START_ROTATION = episode.start_rotation
        agent_cfg.TARGET_CLASS = episode.info[0]["target_label"]

        agent_cfg.AUDIO_SOURCE_POSITIONS = []
        
        if IS_MOVING_SOURCE:
            if 'val' in sim_config.AUDIO.DATASET_SPLIT:
                f_name = '/fs/nexus-projects/ego_data/active_avsep/active-AV-dynamic-separation/data/active_datasets/v1_old/val_100episodes/content/'+episode.scene_id.split('/')[3]+'_moving_source.json.gz'
            else:
                f_name = '/fs/nexus-projects/ego_data/active_avsep/active-AV-dynamic-separation/data/active_datasets/v1_old/train_731243episodes/content/'+episode.scene_id.split('/')[3]+'_moving_source.json.gz'
        
        
            with gzip.open(f_name, "rb") as f:
                mov_src = json.loads(f.read())
                mov_src_list = mov_src['episodes']
          
            mov_src_dict = {}
            for ll in mov_src_list:
                key_val = list(ll.keys())[0]
                # assert key_val == episode.episode_id
                mov_src_dict.update({key_val:ll[key_val]})
        
        for it, source in enumerate(episode.goals):
            agent_cfg.AUDIO_SOURCE_POSITIONS.append(source.position)

            if IS_MOVING_SOURCE:
                if it == 0:
                    agent_cfg.MOVING_SOURCE_POSITIONS = mov_src_dict[str(episode.episode_id)]  #[source.position]
              
        agent_cfg.SOUND_NAMES = []
        for source_info in episode.info:
            agent_cfg.SOUND_NAMES.append(source_info["sound"])

        agent_cfg.SOUND_STARTING_SAMPLING_IDXS = []
        for source_info_idx in range(len(episode.info)):
            source_info = (episode.info)[source_info_idx]
            agent_cfg.SOUND_STARTING_SAMPLING_IDXS.append(source_info["start_idx"] * sim_config.AUDIO.RIR_SAMPLING_RATE)

        agent_cfg.IS_SET_START_STATE = True
        agent_cfg.freeze()
    return sim_config


@registry.register_task(name="AAViDSS")
class AAViDSSTask(NavigationTask):
    def overwrite_sim_config(
        self, sim_config: Any, episode: Type[Episode]
    ) -> Any:
        return merge_sim_episode_config(sim_config, episode)

    def _check_episode_is_active(self, *args: Any, **kwargs: Any) -> bool:
        return self._sim._is_episode_active

@registry.register_sensor
class GroundTruthDeltaXDeltaYSensor(Sensor):
    r"""Mixed binaural spectrogram magnitude at the current step
    """

    def __init__(self, *args: Any, sim: Simulator, config: Config, **kwargs: Any):
        self._sim = sim
        super().__init__(config=config)

    def _get_uuid(self, *args: Any, **kwargs: Any):
        return "ground_truth_deltax_deltay"

    def _get_sensor_type(self, *args: Any, **kwargs: Any):
        return SensorTypes.PATH

    def _get_observation_space(self, *args: Any, **kwargs: Any):
        assert hasattr(self.config, 'FEATURE_SHAPE')
        sensor_shape = self.config.FEATURE_SHAPE

        return spaces.Box(
            low=np.finfo(np.float32).min,
            high=np.finfo(np.float32).max,
            shape=sensor_shape,
            dtype=np.float32,
        )

    def get_observation(self, *args: Any, observations, episode: Episode, **kwargs: Any):
        return self._sim.get_current_ground_truth_deltax_deltay()

## CHANGE DONE HERE    
@registry.register_sensor
class GroundTruthGeodesicDistanceSensor(Sensor):
    r"""Mixed binaural spectrogram magnitude at the current step
    """

    def __init__(self, *args: Any, sim: Simulator, config: Config, **kwargs: Any):
        self._sim = sim
        super().__init__(config=config)

    def _get_uuid(self, *args: Any, **kwargs: Any):
        return "ground_truth_geodesic_distance"

    def _get_sensor_type(self, *args: Any, **kwargs: Any):
        return SensorTypes.PATH

    def _get_observation_space(self, *args: Any, **kwargs: Any):
        assert hasattr(self.config, 'FEATURE_SHAPE')
        sensor_shape = self.config.FEATURE_SHAPE

        return spaces.Box(
            low=np.finfo(np.float32).min,
            high=np.finfo(np.float32).max,
            shape=sensor_shape,
            dtype=np.float32,
        )

    def get_observation(self, *args: Any, observations, episode: Episode, **kwargs: Any):
        return self._sim.get_geo_dist_to_target_audio_source()
    

@registry.register_sensor
class MixedBinAudioMagSensor(Sensor):
    r"""Mixed binaural spectrogram magnitude at the current step
    """

    def __init__(self, *args: Any, sim: Simulator, config: Config, **kwargs: Any):
        self._sim = sim
        super().__init__(config=config)

    def _get_uuid(self, *args: Any, **kwargs: Any):
        return "mixed_bin_audio_mag"

    def _get_sensor_type(self, *args: Any, **kwargs: Any):
        return SensorTypes.PATH

    def _get_observation_space(self, *args: Any, **kwargs: Any):
        assert hasattr(self.config, 'FEATURE_SHAPE')
        sensor_shape = self.config.FEATURE_SHAPE

        return spaces.Box(
            low=np.finfo(np.float32).min,
            high=np.finfo(np.float32).max,
            shape=sensor_shape,
            dtype=np.float32,
        )

    def get_observation(self, *args: Any, observations, episode: Episode, **kwargs: Any):
        return self._sim.get_current_mixed_bin_audio_mag_spec()


@registry.register_sensor
class MixedBinAudioPhaseSensor(Sensor):
    r"""Mixed binaural spectrogram phase at the current step
    """

    def __init__(self, *args: Any, sim: Simulator, config: Config, **kwargs: Any):
        self._sim = sim
        super().__init__(config=config)

    def _get_uuid(self, *args: Any, **kwargs: Any):
        return "mixed_bin_audio_phase"

    def _get_sensor_type(self, *args: Any, **kwargs: Any):
        return SensorTypes.PATH

    def _get_observation_space(self, *args: Any, **kwargs: Any):
        assert hasattr(self.config, 'FEATURE_SHAPE')
        sensor_shape = self.config.FEATURE_SHAPE

        return spaces.Box(
            low=np.finfo(np.float32).min,
            high=np.finfo(np.float32).max,
            shape=sensor_shape,
            dtype=np.float32,
        )

    def get_observation(self, *args: Any, observations, episode: Episode, **kwargs: Any):
        return self._sim.get_current_mixed_bin_audio_phase_spec()


@registry.register_sensor
class GtMonoComponentsSensor(Sensor):
    r"""Ground truth monaural components at the current step
    """

    def __init__(self, *args: Any, sim: Simulator, config: Config, **kwargs: Any):
        self._sim = sim
        super().__init__(config=config)

    def _get_uuid(self, *args: Any, **kwargs: Any):
        return "gt_mono_comps"

    def _get_sensor_type(self, *args: Any, **kwargs: Any):
        return SensorTypes.PATH

    def _get_observation_space(self, *args: Any, **kwargs: Any):
        assert hasattr(self.config, 'FEATURE_SHAPE')
        sensor_shape = self.config.FEATURE_SHAPE

        return spaces.Box(
            low=np.finfo(np.float32).min,
            high=np.finfo(np.float32).max,
            shape=sensor_shape,
            dtype=np.float32,
        )

    def get_observation(self, *args: Any, observations, episode: Episode, **kwargs: Any):
        return self._sim.get_current_gt_mono_audio_components()


@registry.register_sensor
class GtBinComponentsSensor(Sensor):
    r"""Ground truth binaural components at the current step
    """

    def __init__(self, *args: Any, sim: Simulator, config: Config, **kwargs: Any):
        self._sim = sim
        super().__init__(config=config)

    def _get_uuid(self, *args: Any, **kwargs: Any):
        return "gt_bin_comps"

    def _get_sensor_type(self, *args: Any, **kwargs: Any):
        return SensorTypes.PATH

    def _get_observation_space(self, *args: Any, **kwargs: Any):
        assert hasattr(self.config, 'FEATURE_SHAPE')
        sensor_shape = self.config.FEATURE_SHAPE

        return spaces.Box(
            low=np.finfo(np.float32).min,
            high=np.finfo(np.float32).max,
            shape=sensor_shape,
            dtype=np.float32,
        )

    def get_observation(self, *args: Any, observations, episode: Episode, **kwargs: Any):
        return self._sim.get_current_gt_bin_audio_components()


@registry.register_sensor(name="TargetClassSensor")
class TargetClassSensor(Sensor):
    r"""Target class for the current episode
    """

    def __init__(
        self, sim: Union[Simulator, Config], config: Config, *args: Any, **kwargs: Any
    ):
        super().__init__(config=config)
        self._sim = sim

    def _get_uuid(self, *args: Any, **kwargs: Any):
        return "target_class"

    def _get_sensor_type(self, *args: Any, **kwargs: Any):
        return SensorTypes.COLOR

    def _get_observation_space(self, *args: Any, **kwargs: Any):
        return spaces.Box(
            low=0,
            high=1,
            shape=(1,),
            dtype=bool
        )

    def get_observation(
        self, *args: Any, observations, episode: Episode, **kwargs: Any
    ) -> object:
        return [self._sim.target_class]


@registry.register_task_action
class PauseAction(SimulatorTaskAction):
    name: str = "PAUSE"

    def step(self, *args: Any, **kwargs: Any):
        r"""Update ``_metric``, this method is called from ``Env`` on each
        ``step``.
        """
        return self._sim.step(HabitatSimActions.PAUSE)


@registry.register_sensor(name="PoseSensor")
class PoseSensor(Sensor):
    r"""The agents current location and heading in the coordinate frame defined by the
    episode, i.e. the axis it faces along and the origin is defined by its state at
    t=0. Additionally contains the time-step of the episode.
    Args:
        sim: reference to the simulator for calculating task observations.
        config: Contains the DIMENSIONALITY field for the number of dimensions to express the agents position
    Attributes:
        _dimensionality: number of dimensions used to specify the agents position
    """
    cls_uuid: str = "pose"

    def __init__(
        self, sim: Simulator, config: Config, *args: Any, **kwargs: Any
    ):
        self._sim = sim
        self._episode_time = 0
        self._current_episode_id = None
        super().__init__(config=config)

    def _get_uuid(self, *args: Any, **kwargs: Any) -> str:
        return self.cls_uuid

    def _get_sensor_type(self, *args: Any, **kwargs: Any):
        return SensorTypes.POSITION

    def _get_observation_space(self, *args: Any, **kwargs: Any):
        return spaces.Box(
            low=np.finfo(np.float32).min,
            high=np.finfo(np.float32).max,
            shape=(4,),
            dtype=np.float32,
        )

    def _quat_to_xy_heading(self, quat):
        direction_vector = np.array([0, 0, -1])

        heading_vector = quaternion_rotate_vector(quat, direction_vector)

        phi = cartesian_to_polar(-heading_vector[2], heading_vector[0])[1]
        return np.array([phi], dtype=np.float32)

    def get_observation(
        self, observations, episode, *args: Any, **kwargs: Any
    ):
        episode_uniq_id = f"{episode.scene_id} {episode.episode_id}"
        if episode_uniq_id != self._current_episode_id:
            self._episode_time = 0.0
            self._current_episode_id = episode_uniq_id

        agent_state = self._sim.get_agent_state()

        origin = np.array(episode.start_position, dtype=np.float32)
        rotation_world_start = quaternion_from_coeff(episode.start_rotation)

        agent_position_xyz = agent_state.position
        rotation_world_agent = agent_state.rotation

        agent_position_xyz = quaternion_rotate_vector(
            rotation_world_start.inverse(), agent_position_xyz - origin
        )

        agent_heading = self._quat_to_xy_heading(
            rotation_world_agent.inverse() * rotation_world_start
        )

        ep_time = self._episode_time
        self._episode_time += 1.0

        return np.array(
            [-agent_position_xyz[2], agent_position_xyz[0], agent_heading, ep_time],
            dtype=np.float32
        )
    

@registry.register_measure
class NoveltyReward(Measure):
    r"""
    Assigns rewards based on the novelty of states visited. The environment is divided
    into uniform grids of size GRID_SIZE. Each valid grid location is considered to be a
    unique state. When an agent visits any location within a grid cell, the count for
    that state is incremented. The novetly reward is given by:
            r_t = 1/sqrt(n_s)
    where n_s is the visitation count for state s_t.
    """
    def __init__(self, *args: Any, sim: Simulator, config: Config, **kwargs: Any):
        self._sim = sim
        self._config = config
        self.current_episode_id = None
        self._state_map = None
        self.L_min = None
        self.L_max = None
        self._metric = 0.0
        super().__init__()
    def _get_uuid(self, *args: Any, **kwargs: Any):
        return "novelty_reward"
    def reset_metric(self, *args: Any, **kwargs: Any):
        self._metric = 0.0
        self.L_min = maps.COORDINATE_MIN
        self.L_max = maps.COORDINATE_MAX
        map_size = int((self.L_max - self.L_min) / self._config.GRID_SIZE) + 1
        self._state_map = np.zeros((map_size, map_size))
    def _convert_to_grid(self, position):
        """position - (x, y, z) in real-world coordinates """
        grid_x = (position[0] - self.L_min) / self._config.GRID_SIZE
        grid_y = (position[2] - self.L_min) / self._config.GRID_SIZE
        grid_x = int(grid_x)
        grid_y = int(grid_y)
        return (grid_x, grid_y)
    def update_metric(self, *args: Any, episode, action, task: EmbodiedTask, **kwargs: Any):
        episode_id = (episode.episode_id, episode.scene_id)
        if episode_id != self.current_episode_id:
            self.current_episode_id = episode_id
            self.reset_metric(args, episode, kwargs)
        agent_position = self._sim.get_agent_state().position
        grid_x, grid_y = self._convert_to_grid(agent_position)
        self._state_map[grid_y, grid_x] += 1.0
        novelty_reward = 1/math.sqrt(self._state_map[grid_y, grid_x])
        self._metric = novelty_reward
