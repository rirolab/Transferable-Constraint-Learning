import io
import pathlib
from typing import Generator, NamedTuple, Optional, Union, Tuple
from typing import Any, Dict, List

import warnings
import numpy as np
import torch as th
import math

from torch.nn import functional as F
import gym

from stable_baselines3.common.save_util import load_from_pkl, save_to_pkl
from gym import spaces
from stable_baselines3.common.buffers import ReplayBuffer, BaseBuffer
from stable_baselines3.common.type_aliases import TensorDict
from stable_baselines3.common.vec_env import VecNormalize
from stable_baselines3.common.noise import ActionNoise, VectorizedActionNoise

from stable_baselines3.common.type_aliases import TrainFreq, TrainFrequencyUnit
from stable_baselines3.common.utils import safe_mean, should_collect_more_steps

from copy import deepcopy
import time
try:
    # Check memory used by replay buffer when possible
    import psutil
except ImportError:
    psutil = None

class InitialStatesBufferSamples(NamedTuple):
    observations: th.Tensor
    

class ConstReplayBufferSamples(NamedTuple):
    observations: th.Tensor
    actions: th.Tensor
    next_observations: th.Tensor
    dones: th.Tensor
    rewards: th.Tensor

    ## ver1
    # New entry
    constraints: th.Tensor

class ConstRolloutBufferSamples(NamedTuple):
    observations: th.Tensor
    actions: th.Tensor
    old_values: th.Tensor
    old_log_prob: th.Tensor
    advantages: th.Tensor
    returns: th.Tensor


class DictConstRolloutBufferSamples(ConstRolloutBufferSamples):
    observations: TensorDict
    actions: th.Tensor
    old_values: th.Tensor
    old_log_prob: th.Tensor
    advantages: th.Tensor
    returns: th.Tensor


class RolloutBufferSamples(NamedTuple):
    observations: th.Tensor
    actions: th.Tensor
    rewards: th.Tensor
    new_observations: th.Tensor
    old_values: th.Tensor
    old_log_prob: th.Tensor
    advantages: th.Tensor
    returns: th.Tensor

class oriRolloutBuffer(BaseBuffer):
    """
    Rollout buffer used in on-policy algorithms like A2C/PPO.
    It corresponds to ``buffer_size`` transitions collected
    using the current policy.
    This experience will be discarded after the policy update.
    In order to use PPO objective, we also store the current value of each state
    and the log probability of each taken action.

    The term rollout here refers to the model-free notion and should not
    be used with the concept of rollout used in model-based RL or planning.
    Hence, it is only involved in policy and value function training but not action selection.

    :param buffer_size: Max number of element in the buffer
    :param observation_space: Observation space
    :param action_space: Action space
    :param device:
    :param gae_lambda: Factor for trade-off of bias vs variance for Generalized Advantage Estimator
        Equivalent to classic advantage when set to 1.
    :param gamma: Discount factor
    :param n_envs: Number of parallel environments
    """

    def __init__(
        self,
        buffer_size: int,
        observation_space: spaces.Space,
        action_space: spaces.Space,
        device: Union[th.device, str] = "cpu",
        gae_lambda: float = 1,
        gamma: float = 0.99,
        n_envs: int = 1,
    ):

        super().__init__(buffer_size, observation_space, action_space, device, n_envs=n_envs)
        self.gae_lambda = gae_lambda
        self.gamma = gamma
        self.observations, self.actions, self.rewards, self.advantages = None, None, None, None
        self.returns, self.episode_starts, self.values, self.log_probs = None, None, None, None
        self.generator_ready = False
        self.new_observations = None
        self.reset()

    def reset(self) -> None:

        self.observations = np.zeros((self.buffer_size, self.n_envs) + self.obs_shape, dtype=np.float32)
        self.new_observations = np.zeros((self.buffer_size, self.n_envs) + self.obs_shape, dtype=np.float32)
        self.actions = np.zeros((self.buffer_size, self.n_envs, self.action_dim), dtype=np.float32)
        self.rewards = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)
        self.returns = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)
        self.episode_starts = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)
        self.values = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)
        self.log_probs = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)
        self.advantages = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)
        self.generator_ready = False
        super().reset()

    def compute_returns_and_advantage(self, last_values: th.Tensor, dones: np.ndarray) -> None:
        """
        Post-processing step: compute the lambda-return (TD(lambda) estimate)
        and GAE(lambda) advantage.

        Uses Generalized Advantage Estimation (https://arxiv.org/abs/1506.02438)
        to compute the advantage. To obtain Monte-Carlo advantage estimate (A(s) = R - V(S))
        where R is the sum of discounted reward with value bootstrap
        (because we don't always have full episode), set ``gae_lambda=1.0`` during initialization.

        The TD(lambda) estimator has also two special cases:
        - TD(1) is Monte-Carlo estimate (sum of discounted rewards)
        - TD(0) is one-step estimate with bootstrapping (r_t + gamma * v(s_{t+1}))

        For more information, see discussion in https://github.com/DLR-RM/stable-baselines3/pull/375.

        :param last_values: state value estimation for the last step (one for each env)
        :param dones: if the last step was a terminal step (one bool for each env).
        """
        # Convert to numpy
        last_values = last_values.clone().cpu().numpy().flatten()

        last_gae_lam = 0
        for step in reversed(range(self.buffer_size)):
            if step == self.buffer_size - 1:
                next_non_terminal = 1.0 - dones
                next_values = last_values
            else:
                next_non_terminal = 1.0 - self.episode_starts[step + 1]
                next_values = self.values[step + 1]
            delta = self.rewards[step] + self.gamma * next_values * next_non_terminal - self.values[step]
            last_gae_lam = delta + self.gamma * self.gae_lambda * next_non_terminal * last_gae_lam
            self.advantages[step] = last_gae_lam
        # TD(lambda) estimator, see Github PR #375 or "Telescoping in TD(lambda)"
        # in David Silver Lecture 4: https://www.youtube.com/watch?v=PnHCvfgC_ZA
        self.returns = self.advantages + self.values

    def add(
        self,
        obs: np.ndarray,
        action: np.ndarray,
        reward: np.ndarray,
        next_obs: np.ndarray,
        episode_start: np.ndarray,
        value: th.Tensor,
        log_prob: th.Tensor,
    ) -> None:
        """
        :param obs: Observation
        :param action: Action
        :param reward:
        :param episode_start: Start of episode signal.
        :param value: estimated value of the current state
            following the current policy.
        :param log_prob: log probability of the action
            following the current policy.
        """
        if len(log_prob.shape) == 0:
            # Reshape 0-d tensor to avoid error
            log_prob = log_prob.reshape(-1, 1)

        # Reshape needed when using multiple envs with discrete observations
        # as numpy cannot broadcast (n_discrete,) to (n_discrete, 1)
        if isinstance(self.observation_space, spaces.Discrete):
            obs = obs.reshape((self.n_envs,) + self.obs_shape)

        self.observations[self.pos] = np.array(obs).copy()
        self.new_observations[self.pos] = np.array(next_obs).copy()
        self.actions[self.pos] = np.array(action).copy()
        self.rewards[self.pos] = np.array(reward).copy()
        self.episode_starts[self.pos] = np.array(episode_start).copy()
        self.values[self.pos] = value.clone().cpu().numpy().flatten()
        self.log_probs[self.pos] = log_prob.clone().cpu().numpy()
        self.pos += 1
        if self.pos == self.buffer_size:
            self.full = True

    def get(self, batch_size: Optional[int] = None) -> Generator[RolloutBufferSamples, None, None]:
        assert self.full, ""
        indices = np.random.permutation(self.buffer_size * self.n_envs)
        # Prepare the data
        if not self.generator_ready:

            _tensor_names = [
                "observations",
                "actions",
                "rewards",
                "new_observations",
                "values",
                "log_probs",
                "advantages",
                "returns",
            ]

            for tensor in _tensor_names:
                self.__dict__[tensor] = self.swap_and_flatten(self.__dict__[tensor])
            self.generator_ready = True

        # Return everything, don't create minibatches
        if batch_size is None:
            batch_size = self.buffer_size * self.n_envs

        start_idx = 0
        while start_idx < self.buffer_size * self.n_envs:
            yield self._get_samples(indices[start_idx : start_idx + batch_size])
            start_idx += batch_size

    def _get_samples(self, batch_inds: np.ndarray, env: Optional[VecNormalize] = None) -> RolloutBufferSamples:
        data = (
            self.observations[batch_inds],
            self.actions[batch_inds],
            self.rewards[batch_inds],
            self.new_observations[batch_inds],
            self.values[batch_inds].flatten(),
            self.log_probs[batch_inds].flatten(),
            self.advantages[batch_inds].flatten(),
            self.returns[batch_inds].flatten(),
        )
        return RolloutBufferSamples(*tuple(map(self.to_torch, data)))
    
class InitialStatesBuffer(BaseBuffer):
    """
    Replay buffer that stores initial states used in valueDICE and Constraint estimation
    """

    def __init__(
        self,
        buffer_size: int,
        observation_space: spaces.Space,
        action_space: spaces.Space,
        device: Union[th.device, str] = "cpu",
        n_envs: int = 1,
        optimize_memory_usage: bool = False,
        #handle_timeout_termination: bool = True,
    ):
        super().__init__(buffer_size, observation_space, action_space, device, n_envs=n_envs)

        # Adjust buffer size
        self.buffer_size = max(buffer_size // n_envs, 1)

        # Check that the replay buffer can fit into the memory
        if psutil is not None:
            mem_available = psutil.virtual_memory().available

        self.optimize_memory_usage = optimize_memory_usage

        self.observations = np.zeros((self.buffer_size, self.n_envs) + self.obs_shape, dtype=observation_space.dtype)



        if psutil is not None:
            total_memory_usage = self.observations.nbytes #+ self.actions.nbytes + self.rewards.nbytes + self.dones.nbytes 
            
            if total_memory_usage > mem_available:
                # Convert to GB
                total_memory_usage /= 1e9
                mem_available /= 1e9
                warnings.warn(
                    "This system does not have apparently enough memory to store the complete "
                    f"replay buffer {total_memory_usage:.2f}GB > {mem_available:.2f}GB"
                )

    def add(
        self,
        obs: np.ndarray,
    ) -> None:

        # Reshape needed when using multiple envs with discrete observations
        # as numpy cannot broadcast (n_discrete,) to (n_discrete, 1)
        if isinstance(self.observation_space, spaces.Discrete):
            obs = obs.reshape((self.n_envs,) + self.obs_shape)

        # Copy to avoid modification by reference
        self.observations[self.pos] = np.array(obs).copy()

        self.pos += 1
        if self.pos == self.buffer_size:
            self.full = True
            self.pos = 0

    def sample(self, batch_size: int, env: Optional[VecNormalize] = None) -> InitialStatesBufferSamples:
        """
        Sample elements from the replay buffer.
        Custom sampling when using memory efficient variant,
        as we should not sample the element with index `self.pos`
        See https://github.com/DLR-RM/stable-baselines3/pull/28#issuecomment-637559274

        :param batch_size: Number of element to sample
        :param env: associated gym VecEnv
            to normalize the observations/rewards when sampling
        :return:
        """
        if not self.optimize_memory_usage:
            return super().sample(batch_size=batch_size, env=env)
        # Do not sample the element with index `self.pos` as the transitions is invalid
        # (we use only one array to store `obs` and `next_obs`)
        if self.full:
            batch_inds = (np.random.randint(1, self.buffer_size, size=batch_size) + self.pos) % self.buffer_size
        else:
            batch_inds = np.random.randint(0, self.pos, size=batch_size)
        return self._get_samples(batch_inds, env=env)

    def _get_samples(self, batch_inds: np.ndarray, env: Optional[VecNormalize] = None) -> InitialStatesBufferSamples:
        # Sample randomly the env idx
        env_indices = np.random.randint(0, high=self.n_envs, size=(len(batch_inds),))

        data = (
            self._normalize_obs(self.observations[batch_inds, env_indices, :], env),
        )
        return InitialStatesBufferSamples(*tuple(map(self.to_torch, data)))


class RolloutBuffer(BaseBuffer):
    """
    Rollout buffer used in on-policy algorithms like A2C/PPO.
    It corresponds to ``buffer_size`` transitions collected
    using the current policy.
    This experience will be discarded after the policy update.
    In order to use PPO objective, we also store the current value of each state
    and the log probability of each taken action.

    The term rollout here refers to the model-free notion and should not
    be used with the concept of rollout used in model-based RL or planning.
    Hence, it is only involved in policy and value function training but not action selection.

    :param buffer_size: Max number of element in the buffer
    :param observation_space: Observation space
    :param action_space: Action space
    :param device:
    :param gae_lambda: Factor for trade-off of bias vs variance for Generalized Advantage Estimator
        Equivalent to classic advantage when set to 1.
    :param gamma: Discount factor
    :param n_envs: Number of parallel environments
    """

    def __init__(
        self,
        buffer_size: int,
        observation_space: spaces.Space,
        action_space: spaces.Space,
        device: Union[th.device, str] = "cpu",
        gae_lambda: float = 1,
        gamma: float = 0.99,
        n_envs: int = 1,
    ):

        super().__init__(buffer_size, observation_space, action_space, device, n_envs=n_envs)
        self.gae_lambda = gae_lambda
        self.gamma = gamma
        self.observations, self.actions, self.rewards, self.advantages = None, None, None, None
        self.returns, self.episode_starts, self.values, self.log_probs = None, None, None, None
        self.generator_ready = False
        self.reset()

    def reset(self) -> None:

        self.observations = np.zeros((self.buffer_size, self.n_envs) + self.obs_shape, dtype=np.float32)
        self.actions = np.zeros((self.buffer_size, self.n_envs, self.action_dim), dtype=np.float32)
        self.rewards = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)
        self.returns = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)
        self.episode_starts = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)
        self.values = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)
        self.log_probs = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)
        self.advantages = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)
        self.generator_ready = False
        super().reset()

    def compute_returns_and_advantage(self, last_values: th.Tensor, dones: np.ndarray) -> None:
        """
        Post-processing step: compute the lambda-return (TD(lambda) estimate)
        and GAE(lambda) advantage.

        Uses Generalized Advantage Estimation (https://arxiv.org/abs/1506.02438)
        to compute the advantage. To obtain Monte-Carlo advantage estimate (A(s) = R - V(S))
        where R is the sum of discounted reward with value bootstrap
        (because we don't always have full episode), set ``gae_lambda=1.0`` during initialization.

        The TD(lambda) estimator has also two special cases:
        - TD(1) is Monte-Carlo estimate (sum of discounted rewards)
        - TD(0) is one-step estimate with bootstrapping (r_t + gamma * v(s_{t+1}))

        For more information, see discussion in https://github.com/DLR-RM/stable-baselines3/pull/375.

        :param last_values: state value estimation for the last step (one for each env)
        :param dones: if the last step was a terminal step (one bool for each env).
        """
        # Convert to numpy
        last_values = last_values.clone().cpu().numpy().flatten()

        last_gae_lam = 0
        for step in reversed(range(self.buffer_size)):
            if step == self.buffer_size - 1:
                next_non_terminal = 1.0 - dones
                next_values = last_values
            else:
                next_non_terminal = 1.0 - self.episode_starts[step + 1]
                next_values = self.values[step + 1]
            delta = self.rewards[step] + self.gamma * next_values * next_non_terminal - self.values[step]
            last_gae_lam = delta + self.gamma * self.gae_lambda * next_non_terminal * last_gae_lam
            self.advantages[step] = last_gae_lam
        # TD(lambda) estimator, see Github PR #375 or "Telescoping in TD(lambda)"
        # in David Silver Lecture 4: https://www.youtube.com/watch?v=PnHCvfgC_ZA
        self.returns = self.advantages + self.values

    def add(
        self,
        obs: np.ndarray,
        action: np.ndarray,
        reward: np.ndarray,
        episode_start: np.ndarray,
        value: th.Tensor,
        log_prob: th.Tensor,
    ) -> None:
        """
        :param obs: Observation
        :param action: Action
        :param reward:
        :param episode_start: Start of episode signal.
        :param value: estimated value of the current state
            following the current policy.
        :param log_prob: log probability of the action
            following the current policy.
        """
        if len(log_prob.shape) == 0:
            # Reshape 0-d tensor to avoid error
            log_prob = log_prob.reshape(-1, 1)

        # Reshape needed when using multiple envs with discrete observations
        # as numpy cannot broadcast (n_discrete,) to (n_discrete, 1)
        if isinstance(self.observation_space, spaces.Discrete):
            obs = obs.reshape((self.n_envs,) + self.obs_shape)

        self.observations[self.pos] = np.array(obs).copy()
        self.actions[self.pos] = np.array(action).copy()
        self.rewards[self.pos] = np.array(reward).copy()
        self.episode_starts[self.pos] = np.array(episode_start).copy()
        self.values[self.pos] = value.clone().cpu().numpy().flatten()
        self.log_probs[self.pos] = log_prob.clone().cpu().numpy()
        self.pos += 1
        if self.pos == self.buffer_size:
            self.full = True

    def get(self, batch_size: Optional[int] = None, indices=None) -> Generator[ConstRolloutBufferSamples, None, None]:
        assert self.full, ""
        if indices is not None:
            indices = indices.copy()
        else:
            indices = np.random.permutation(self.buffer_size * self.n_envs)
        # Prepare the data
        if not self.generator_ready:

            _tensor_names = [
                "observations",
                "actions",
                "values",
                "log_probs",
                "advantages",
                "returns",
            ]

            for tensor in _tensor_names:
                self.__dict__[tensor] = self.swap_and_flatten(self.__dict__[tensor])
            self.generator_ready = True

        # Return everything, don't create minibatches
        if batch_size is None:
            batch_size = self.buffer_size * self.n_envs

        start_idx = 0
        while start_idx < self.buffer_size * self.n_envs:
            yield self._get_samples(indices[start_idx : start_idx + batch_size])
            start_idx += batch_size

    def _get_samples(self, batch_inds: np.ndarray, env: Optional[VecNormalize] = None) -> ConstRolloutBufferSamples:
        data = (
            self.observations[batch_inds],
            self.actions[batch_inds],
            self.values[batch_inds].flatten(),
            self.log_probs[batch_inds].flatten(),
            self.advantages[batch_inds].flatten(),
            self.returns[batch_inds].flatten(),
        )
        return ConstRolloutBufferSamples(*tuple(map(self.to_torch, data)))

class ConstReplayBuffer(BaseBuffer):
    """
    Replay buffer used in off-policy algorithms like SAC/TD3.

    :param buffer_size: Max number of element in the buffer
    :param observation_space: Observation space
    :param action_space: Action space
    :param device:
    :param n_envs: Number of parallel environments
    :param optimize_memory_usage: Enable a memory efficient variant
        of the replay buffer which reduces by almost a factor two the memory used,
        at a cost of more complexity.
        See https://github.com/DLR-RM/stable-baselines3/issues/37#issuecomment-637501195
        and https://github.com/DLR-RM/stable-baselines3/pull/28#issuecomment-637559274
    :param handle_timeout_termination: Handle timeout termination (due to timelimit)
        separately and treat the task as infinite horizon task.
        https://github.com/DLR-RM/stable-baselines3/issues/284
    """

    def __init__(
        self,
        buffer_size: int,
        observation_space: spaces.Space,
        action_space: spaces.Space,
        device: Union[th.device, str] = "cpu",
        n_envs: int = 1,
        optimize_memory_usage: bool = False,
        handle_timeout_termination: bool = True,
    ):
        super().__init__(buffer_size, observation_space, action_space, device, n_envs=n_envs)

        # Adjust buffer size
        self.buffer_size = max(buffer_size // n_envs, 1)

        # Check that the replay buffer can fit into the memory
        if psutil is not None:
            mem_available = psutil.virtual_memory().available

        self.optimize_memory_usage = optimize_memory_usage

        self.observations = np.zeros((self.buffer_size, self.n_envs) + self.obs_shape, dtype=observation_space.dtype)

        if optimize_memory_usage:
            # `observations` contains also the next observation
            self.next_observations = None
        else:
            self.next_observations = np.zeros((self.buffer_size, self.n_envs) + self.obs_shape, dtype=observation_space.dtype)

        self.actions = np.zeros((self.buffer_size, self.n_envs, self.action_dim), dtype=action_space.dtype)

        self.rewards = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)

        ## ver1
        # New entry
        self.constraints = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)

        self.dones = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)
        # Handle timeouts termination properly if needed
        # see https://github.com/DLR-RM/stable-baselines3/issues/284
        self.handle_timeout_termination = handle_timeout_termination
        self.timeouts = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)

        if psutil is not None:
            total_memory_usage = self.observations.nbytes + self.actions.nbytes + self.rewards.nbytes + self.dones.nbytes 
            
            ## ver1
            # New entry
            total_memory_usage += self.constraints.nbytes

            if self.next_observations is not None:
                total_memory_usage += self.next_observations.nbytes

            if total_memory_usage > mem_available:
                # Convert to GB
                total_memory_usage /= 1e9
                mem_available /= 1e9
                warnings.warn(
                    "This system does not have apparently enough memory to store the complete "
                    f"replay buffer {total_memory_usage:.2f}GB > {mem_available:.2f}GB"
                )

    def add(
        self,
        obs: np.ndarray,
        next_obs: np.ndarray,
        action: np.ndarray,
        reward: np.ndarray,
        
        ## ver1
        # New entry
        constraint: np.ndarray,
        
        done: np.ndarray,
        infos: List[Dict[str, Any]],
    ) -> None:

        # Reshape needed when using multiple envs with discrete observations
        # as numpy cannot broadcast (n_discrete,) to (n_discrete, 1)
        if isinstance(self.observation_space, spaces.Discrete):
            obs = obs.reshape((self.n_envs,) + self.obs_shape)
            next_obs = next_obs.reshape((self.n_envs,) + self.obs_shape)

        # Same, for actions
        if isinstance(self.action_space, spaces.Discrete):
            action = action.reshape((self.n_envs, self.action_dim))

        # Copy to avoid modification by reference
        self.observations[self.pos] = np.array(obs).copy()

        if self.optimize_memory_usage:
            self.observations[(self.pos + 1) % self.buffer_size] = np.array(next_obs).copy()
        else:
            self.next_observations[self.pos] = np.array(next_obs).copy()

        self.actions[self.pos] = np.array(action).copy()

        ## ver1
        # New entry
        self.constraints[self.pos] = np.array(constraint).copy() 

        self.rewards[self.pos] = np.array(reward).copy()
        self.dones[self.pos] = np.array(done).copy()

        if self.handle_timeout_termination:
            self.timeouts[self.pos] = np.array([info.get("TimeLimit.truncated", False) for info in infos])

        self.pos += 1
        if self.pos == self.buffer_size:
            self.full = True
            self.pos = 0

    def sample(self, batch_size: int, env: Optional[VecNormalize] = None) -> ConstReplayBufferSamples:
        """
        Sample elements from the replay buffer.
        Custom sampling when using memory efficient variant,
        as we should not sample the element with index `self.pos`
        See https://github.com/DLR-RM/stable-baselines3/pull/28#issuecomment-637559274

        :param batch_size: Number of element to sample
        :param env: associated gym VecEnv
            to normalize the observations/rewards when sampling
        :return:
        """
        if not self.optimize_memory_usage:
            return super().sample(batch_size=batch_size, env=env)
        # Do not sample the element with index `self.pos` as the transitions is invalid
        # (we use only one array to store `obs` and `next_obs`)
        if self.full:
            batch_inds = (np.random.randint(1, self.buffer_size, size=batch_size) + self.pos) % self.buffer_size
        else:
            batch_inds = np.random.randint(0, self.pos, size=batch_size)
        return self._get_samples(batch_inds, env=env)

    def _get_samples(self, batch_inds: np.ndarray, env: Optional[VecNormalize] = None) -> ConstReplayBufferSamples:
        # Sample randomly the env idx
        env_indices = np.random.randint(0, high=self.n_envs, size=(len(batch_inds),))

        if self.optimize_memory_usage:
            next_obs = self._normalize_obs(self.observations[(batch_inds + 1) % self.buffer_size, env_indices, :], env)
        else:
            next_obs = self._normalize_obs(self.next_observations[batch_inds, env_indices, :], env)

        data = (
            self._normalize_obs(self.observations[batch_inds, env_indices, :], env),
            self.actions[batch_inds, env_indices, :],
            next_obs,
            # Only use dones that are not due to timeouts
            # deactivated by default (timeouts is initialized as an array of False)
            (self.dones[batch_inds, env_indices] * (1 - self.timeouts[batch_inds, env_indices])).reshape(-1, 1),
            self._normalize_reward(self.rewards[batch_inds, env_indices].reshape(-1, 1), env),
            self.constraints[batch_inds, env_indices].reshape(-1, 1),
        )
        return ConstReplayBufferSamples(*tuple(map(self.to_torch, data)))



def buffer_loader(path: Union[str, pathlib.Path, io.BufferedIOBase],
                    truncate_last_traj: bool = True,
                    sample_interval: int = 1,
                    n_envs: int = 1,
                    verbose = 0):


    replay_buffer = load_from_pkl(path, verbose)
    assert (isinstance(replay_buffer, ConstReplayBuffer) or isinstance(replay_buffer, ReplayBuffer)), "The replay buffer must inherit from ReplayBuffer class"

    # Backward compatibility with SB3 < 2.1.0 replay buffer
    # Keep old behavior: do not handle timeout termination separately
    if not hasattr(replay_buffer, "handle_timeout_termination"):  # pragma: no cover
        replay_buffer.handle_timeout_termination = False
        replay_buffer.timeouts = np.zeros_like(replay_buffer.dones)

    print("Handle termination: ",  hasattr(replay_buffer, "handle_timeout_termination"))

    if replay_buffer.full:
        batch_inds = np.arange(0, replay_buffer.buffer_size, sample_interval)
    else:
        batch_inds = np.arange(0, replay_buffer.pos, sample_interval)
    assert len(batch_inds) != 0

    
    new_buffer = ReplayBuffer(buffer_size=len(batch_inds) * (replay_buffer.n_envs // n_envs),
                              observation_space=replay_buffer.observation_space,
                              action_space=replay_buffer.action_space,
                              device=replay_buffer.device,
                              n_envs=replay_buffer.n_envs,
                              optimize_memory_usage=replay_buffer.optimize_memory_usage,
                              handle_timeout_termination=replay_buffer.handle_timeout_termination,)
    
    print(replay_buffer.observations.shape)
    print(replay_buffer.rewards.shape)
    obs_shape = replay_buffer.observations.shape[-1]
    act_shape = replay_buffer.actions.shape[-1]
    new_buffer.observations = np.reshape( replay_buffer.observations[batch_inds, :, :], (-1, n_envs, obs_shape))
    new_buffer.actions = np.reshape(replay_buffer.actions[batch_inds, :, :], (-1, n_envs, act_shape))
    if replay_buffer.next_observations is not None:
        new_buffer.next_observations = np.reshape( replay_buffer.next_observations[batch_inds, :, :], (-1, n_envs, obs_shape))
    new_buffer.rewards = np.reshape( replay_buffer.rewards[batch_inds, :], (-1, n_envs))
    new_buffer.dones = np.reshape( replay_buffer.dones[batch_inds, :], (-1, n_envs))
    new_buffer.timeouts = np.reshape( replay_buffer.timeouts[batch_inds, :], (-1, n_envs))
    new_buffer.full = True
    """
    new_buffer.observations = replay_buffer.observations[batch_inds, :, :]
    new_buffer.actions = replay_buffer.actions[batch_inds, :, :]
    if replay_buffer.next_observations is not None:
        new_buffer.next_observations = replay_buffer.next_observations[batch_inds, :, :]
    new_buffer.rewards = replay_buffer.rewards[batch_inds, :]
    new_buffer.dones = replay_buffer.dones[batch_inds, :]
    new_buffer.timeouts = replay_buffer.timeouts[batch_inds, :]
    new_buffer.full = True
    """
    del replay_buffer
    return new_buffer


def _sample_action(
    model,
    observation: np.ndarray,
    action_noise: Optional[ActionNoise] = None,
    n_envs: int = 1,
    deterministic: bool = True,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Sample an action according to the exploration policy.
    This is either done by sampling the probability distribution of the policy,
    or sampling a random action (from a uniform distribution over the action space)
    or by adding noise to the deterministic output.

    :param action_noise: Action noise that will be used for exploration
        Required for deterministic policy (e.g. TD3). This can also be used
        in addition to the stochastic policy for SAC.
    :param learning_starts: Number of steps before learning for the warm-up phase.
    :param n_envs:
    :return: action to take in the environment
        and scaled action that will be stored in the replay buffer.
        The two differs when the action space is not normalized (bounds are not [-1, 1]).
    """
    # Select action randomly or according to policy
    unscaled_action, _ = model.predict(observation, deterministic=deterministic)
    # model(observation)
    # Rescale the action from [low, high] to [-1, 1]
    if isinstance(model.action_space, gym.spaces.Box):
        scaled_action = model.policy.scale_action(unscaled_action)

        # Add noise to the action (improve exploration)
        if action_noise is not None:
            scaled_action = np.clip(scaled_action + action_noise(), -1, 1)

        # We store the scaled action in the buffer
        buffer_action = scaled_action
        action = model.policy.unscale_action(scaled_action)
    else:
        # Discrete case, no need to normalize or clip
        buffer_action = unscaled_action
        action = buffer_action
    return action, buffer_action

def collect_demos(
    model,
    env,
    train_freq: TrainFreq,
    replay_buffer: Union[ConstReplayBuffer, ReplayBuffer],
    action_noise: Optional[ActionNoise] = None,
    deterministic=True,
):
    # Switch to eval mode (this affects batch norm / dropout)
    model.policy.set_training_mode(False)

    num_collected_steps, num_collected_episodes = 0, 0

    assert train_freq.frequency > 0, "Should at least collect one step or episode."

    # Vectorize action noise if needed
    if action_noise is not None and env.num_envs > 1 and not isinstance(action_noise, VectorizedActionNoise):
        action_noise = VectorizedActionNoise(action_noise, env.num_envs)

    if model.use_sde:
        model.actor.reset_noise(env.num_envs)

    obs = env.reset()

    while should_collect_more_steps(train_freq, num_collected_steps, num_collected_episodes):
        if model.use_sde and model.sde_sample_freq > 0 and num_collected_steps % model.sde_sample_freq == 0:
            # Sample a new noise matrix
            model.actor.reset_noise(env.num_envs)

        # Select action randomly or according to policy
        actions, buffer_actions = _sample_action(model, obs, action_noise, deterministic=deterministic)
        # true_actions, _ = _sample_action(model, obs, action_noise, deterministic=True)
        # false_actions, _ = _sample_action(model, obs, action_noise, deterministic=False)
        # q_values

        # Rescale and perform action
        new_obs, rewards, dones, infos = env.step(actions)
        env.envs[0].render()
        num_collected_steps += 1

        # print(obs)
        # print(new_obs)
        # print(rewards,(np.abs(obs[:,2]) < (0.33 * 12 * 2 * math.pi / 360)))
        # # env.render()
        # time.sleep(0.3)
        # print ()

        # Retrieve reward and episode length if using Monitor wrapper
        model._update_info_buffer(infos, dones)

        def extract_constraint(info):
            if 'constraint' in info:
                return info['constraint']
            else:
                return 0
        constraints = list(map(extract_constraint, infos))

        # Store data in replay buffer (normalized action and unnormalized observation)
        next_obs = deepcopy(new_obs)

        for i, done in enumerate(dones):
            if done and infos[i].get("terminal_observation") is not None:
                if isinstance(next_obs, dict):
                    next_obs_ = infos[i]["terminal_observation"]
                    # VecNormalize normalizes the terminal observation
                    for key in next_obs.keys():
                        next_obs[key][i] = next_obs_[key]
                else:
                    next_obs[i] = infos[i]["terminal_observation"]
                    # VecNormalize normalizes the terminal observation

        if isinstance(replay_buffer, ConstReplayBuffer):
            replay_buffer.add(
                obs,
                next_obs,
                buffer_actions,
                rewards,
                constraints,
                dones,
                infos,
            )
        else:
            replay_buffer.add(
                obs,
                next_obs,
                buffer_actions,
                rewards,
                dones,
                infos,
            )
            
        obs = new_obs
        for idx, done in enumerate(dones):
            if done:
                # Update stats

                if action_noise is not None:
                    kwargs = dict(indices=[idx]) if env.num_envs > 1 else {}
                    action_noise.reset(**kwargs)

        # obs = next_obs
        # print(obs)
        # print(next_obs)
        # print(new_obs)
        # print()
        # print(actions, buffer_actions)
        # print(true_actions, false_actions)
        # print(F.softmax( cal_Q(model, th.tensor(obs).to(model.device))/model.alpha ))
        # print(model.alpha)
        # print(done)
        # print(rewards)
        # time.sleep(0.5)


def cal_Q(self, obs, target=False):
    if target:
        q_values = self.q_net_target(obs)
    else:
        q_values = self.q_net(obs)

    if hasattr(self, "qc_net"):
        soft_beta = self.soft_plus(self.beta)
        if target:
            qc_values = self.qc_net_target(obs)
        else:
            qc_values = self.qc_net(obs)
        q_values = q_values - soft_beta * qc_values

    return q_values

def cal_V(self, obs, target=False):
    if hasattr(self, "qc_net"):
        v_values, vc_values = calculate_VVC(self, obs, target)
        soft_beta = self.soft_plus(self.beta)
        v_values -= soft_beta * vc_values
    else:
        v_values = calculate_V(self, obs, target)
    return v_values

def cal_VC(self, obs, target=False):
    if hasattr(self, "qc_net"):
        v_values, vc_values = calculate_VVC(self, obs, target)
    else:
        assert False
        v_values = calculate_V(self, obs, target)
    return vc_values


def calculate_VVC(self, obs, target=False):
    soft_beta = self.soft_plus(self.beta)
    if target:
        next_q_values = self.q_net_target(obs)
        next_qc_values = self.qc_net_target(obs)
    else:
        next_q_values = self.q_net(obs)
        next_qc_values = self.qc_net(obs)
        

    next_lag_values = next_q_values - soft_beta * next_qc_values
    action_probs = F.softmax(next_lag_values/self.alpha, dim=1)#, keepdim=True)
    log_action_probs = th.log(action_probs + 1e-9)

    #next_v_values = self.alpha * th.logsumexp(next_q_values/self.alpha, dim=1, keepdim=True)
    next_v_values = ((next_q_values - self.alpha * log_action_probs) * action_probs).sum(-1, keepdim=True)
    next_vc_values = (next_qc_values * action_probs).sum(-1, keepdim=True)
    return next_v_values, next_vc_values

def calculate_V(self, obs, target=False):
    if target:
        next_q_values = self.q_net_target(obs)
    else:
        next_q_values = self.q_net(obs)

    next_lag_values = next_q_values
    action_probs = F.softmax(next_lag_values/self.alpha, dim=1)#, keepdim=True)
    log_action_probs = th.log(action_probs + 1e-9)

    #next_v_values = self.alpha * th.logsumexp(next_q_values/self.alpha, dim=1, keepdim=True)
    next_v_values = ((next_q_values - self.alpha * log_action_probs) * action_probs).sum(-1, keepdim=True)
    return next_v_values