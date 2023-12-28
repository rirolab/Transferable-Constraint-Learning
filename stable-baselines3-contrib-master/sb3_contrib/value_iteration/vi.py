from argparse import Action
import warnings
from typing import Any, Dict, List, Optional, Tuple, Type, Union

import gym
import numpy as np
import torch as th
from torch.nn import functional as F
import time

from sb3_contrib.common.constraint.buffers import ConstReplayBuffer, InitialStatesBuffer
from stable_baselines3.common.buffers import DictReplayBuffer, ReplayBuffer
from stable_baselines3.her.her_replay_buffer import HerReplayBuffer
from stable_baselines3.common.off_policy_algorithm import OffPolicyAlgorithm
from stable_baselines3.common.policies import BasePolicy
from stable_baselines3.common.preprocessing import maybe_transpose
from stable_baselines3.common.type_aliases import GymEnv, MaybeCallback, RolloutReturn, Schedule, TrainFreq, TrainFrequencyUnit
from stable_baselines3.common.utils import get_linear_fn, is_vectorized_observation, polyak_update
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.base_class import BaseAlgorithm
from sb3_contrib.value_iteration.policies import CnnPolicy, MlpPolicy, MultiInputPolicy, SoftDQNPolicy

EPS = 1e-9

class VI(OffPolicyAlgorithm):
    """
    Deep Soft-Q-Network (Soft-DQN)

    Based on Deep Q-Netowkr (DQN)
    Paper: https://arxiv.org/abs/1312.5602, https://www.nature.com/articles/nature14236
    Default hyperparameters are taken from the nature paper,
    except for the optimizer and learning rate that were taken from Stable Baselines defaults.

    :param policy: The policy model to use (MlpPolicy, CnnPolicy, ...)
    :param env: The environment to learn from (if registered in Gym, can be str)
    :param learning_rate: The learning rate, it can be a function
        of the current progress remaining (from 1 to 0)
    :param buffer_size: size of the replay buffer
    :param learning_starts: how many steps of the model to collect transitions for before learning starts
    :param batch_size: Minibatch size for each gradient update

    ## ver1
    # >>>>>
    :param alpha: Entropy coefficient
    # <<<<<

    :param tau: the soft update coefficient ("Polyak update", between 0 and 1) default 1 for hard update
    :param gamma: the discount factor
    :param train_freq: Update the model every ``train_freq`` steps. Alternatively pass a tuple of frequency and unit
        like ``(5, "step")`` or ``(2, "episode")``.
    :param gradient_steps: How many gradient steps to do after each rollout (see ``train_freq``)
        Set to ``-1`` means to do as many gradient steps as steps done in the environment
        during the rollout.
    :param replay_buffer_class: Replay buffer class to use (for instance ``HerReplayBuffer``).
        If ``None``, it will be automatically selected.
    :param replay_buffer_kwargs: Keyword arguments to pass to the replay buffer on creation.
    :param optimize_memory_usage: Enable a memory efficient variant of the replay buffer
        at a cost of more complexity.
        See https://github.com/DLR-RM/stable-baselines3/issues/37#issuecomment-637501195
    :param target_update_interval: update the target network every ``target_update_interval``
        environment steps.
    :param exploration_fraction: fraction of entire training period over which the exploration rate is reduced
    :param exploration_initial_eps: initial value of random action probability
    :param exploration_final_eps: final value of random action probability
    :param max_grad_norm: The maximum value for the gradient clipping
    :param tensorboard_log: the log location for tensorboard (if None, no logging)
    :param create_eval_env: Whether to create a second environment that will be
        used for evaluating the agent periodically. (Only available when passing string for the environment)
    :param policy_kwargs: additional arguments to be passed to the policy on creation
    :param verbose: the verbosity level: 0 no output, 1 info, 2 debug
    :param seed: Seed for the pseudo random generators
    :param device: Device (cpu, cuda, ...) on which the code should be run.
        Setting it to auto, the code will be run on the GPU if possible.
    :param _init_setup_model: Whether or not to build the network at the creation of the instance
    """

    policy_aliases: Dict[str, Type[BasePolicy]] = {
        "MlpPolicy": MlpPolicy,
        "CnnPolicy": CnnPolicy,
        "MultiInputPolicy": MultiInputPolicy,
    }

    def __init__(
        self,
        policy: Union[str, Type[SoftDQNPolicy]],
        env: Union[GymEnv, str],
        learning_rate: Union[float, Schedule] = 1e-4,
        buffer_size: int = 1_000_000,  # 1e6
        learning_starts: int = 50000,
        batch_size: int = 32,
        alpha: float = 0.1,
        tau: float = 1.0,
        gamma: float = 0.99,
        train_freq: Union[int, Tuple[int, str]] = 4,
        gradient_steps: int = 1,
        replay_buffer_class: Optional[ReplayBuffer] = None,
        replay_buffer_kwargs: Optional[Dict[str, Any]] = None,
        optimize_memory_usage: bool = False,
        target_update_interval: int = 10000,
        exploration_fraction: float = 0.1,
        exploration_initial_eps: float = 1.0,
        exploration_final_eps: float = 0.05,
        max_grad_norm: float = 10,
        tensorboard_log: Optional[str] = None,
        create_eval_env: bool = False,
        policy_kwargs: Optional[Dict[str, Any]] = None,
        verbose: int = 0,
        seed: Optional[int] = None,
        device: Union[th.device, str] = "auto",
        _init_setup_model: bool = True,
        const_lim: float = 0.1,
    ):

        if policy_kwargs is None:
            policy_kwargs = {"alpha": alpha}
        else:
            policy_kwargs = {}

        super().__init__(
            policy,
            env,
            learning_rate,
            buffer_size,
            learning_starts,
            batch_size,
            tau,
            gamma,
            train_freq,
            gradient_steps,
            action_noise=None,  # No action noise
            replay_buffer_class=replay_buffer_class,
            replay_buffer_kwargs=replay_buffer_kwargs,
            policy_kwargs=policy_kwargs,
            tensorboard_log=tensorboard_log,
            verbose=verbose,
            device=device,
            create_eval_env=create_eval_env,
            seed=seed,
            sde_support=False,
            optimize_memory_usage=optimize_memory_usage,
            supported_action_spaces=(gym.spaces.Discrete,),
            support_multi_env=True,
        )

        self.exploration_initial_eps = exploration_initial_eps
        self.exploration_final_eps = exploration_final_eps
        self.exploration_fraction = exploration_fraction
        self.target_update_interval = target_update_interval
        # For updating the target network with multiple envs:
        self._n_calls = 0
        self.max_grad_norm = max_grad_norm
        # "epsilon" for the epsilon-greedy exploration
        self.exploration_rate = 0.0
        # Linear schedule will be defined in `_setup_model()`
        self.exploration_schedule = None
        self.q_net, self.qc_net = None, None

        ## ver1
        # >>>>>
        self.alpha = alpha
        # <<<<<

        if _init_setup_model:
            self._setup_model()

    def _convert_train_freq(self) -> None:
        """
        Convert `train_freq` parameter (int or tuple)
        to a TrainFreq object.
        """
        if not isinstance(self.train_freq, TrainFreq):
            train_freq = self.train_freq

            # The value of the train frequency will be checked later
            if not isinstance(train_freq, tuple):
                train_freq = (train_freq, "step")

            try:
                train_freq = (train_freq[0], TrainFrequencyUnit(train_freq[1]))
            except ValueError:
                raise ValueError(f"The unit of the `train_freq` must be either 'step' or 'episode' not '{train_freq[1]}'!")

            if not isinstance(train_freq[0], int):
                raise ValueError(f"The frequency of `train_freq` must be an integer and not {train_freq[0]}")

            self.train_freq = TrainFreq(*train_freq)
            
    def _setup_model(self) -> None:
        super()._setup_model()
        self._create_aliases()
        """
        self.exploration_schedule = get_linear_fn(
            self.exploration_initial_eps,
            self.exploration_final_eps,
            self.exploration_fraction,
        )
        """
        # Account for multiple environments
        # each call to step() corresponds to n_envs transitions
        if self.n_envs > 1:
            if self.n_envs > self.target_update_interval:
                warnings.warn(
                    "The number of environments used is greater than the target network "
                    f"update interval ({self.n_envs} > {self.target_update_interval}), "
                    "therefore the target network will be updated after each call to env.step() "
                    f"which corresponds to {self.n_envs} steps."
                )

            self.target_update_interval = max(self.target_update_interval // self.n_envs, 1)

    def _create_aliases(self) -> None:
        self.q_net = self.policy.q_net
        self.qc_net = self.policy.qc_net
        self.soft_plus = self.policy.soft_plus
        self.beta = self.policy.beta
        self.beta_target = self.policy.beta_target
        # self.beta = 1.0
    def _on_step(self) -> None:
        """
        Update the exploration rate and target network if needed.
        This method is called in ``collect_rollouts()`` after each step in the environment.
        """
        self._n_calls += 1
        if self._n_calls % self.target_update_interval == 0:
            polyak_update(self.q_net.parameters(), self.q_net_target.parameters(), self.tau)

        self.exploration_rate = self.exploration_schedule(self._current_progress_remaining)
        self.logger.record("rollout/exploration_rate", self.exploration_rate)

    def _setup_learn(
        self,
        total_timesteps: int,
        eval_env: Optional[GymEnv],
        callback: MaybeCallback = None,
        eval_freq: int = 10000,
        n_eval_episodes: int = 5,
        log_path: Optional[str] = None,
        reset_num_timesteps: bool = True,
        tb_log_name: str = "run",
    ) -> Tuple[int, BaseCallback]:
        """
        cf `BaseAlgorithm`.
        """
        # Prevent continuity issue by truncating trajectory
        # when using memory efficient replay buffer
        # see https://github.com/DLR-RM/stable-baselines3/issues/46

        # Special case when using HerReplayBuffer,
        # the classic replay buffer is inside it when using offline sampling
        if isinstance(self.replay_buffer, HerReplayBuffer):
            replay_buffer = self.replay_buffer.replay_buffer
        else:
            replay_buffer = self.replay_buffer

        truncate_last_traj = (
            self.optimize_memory_usage
            and reset_num_timesteps
            and replay_buffer is not None
            and (replay_buffer.full or replay_buffer.pos > 0)
        )

        if truncate_last_traj:
            warnings.warn(
                "The last trajectory in the replay buffer will be truncated, "
                "see https://github.com/DLR-RM/stable-baselines3/issues/46."
                "You should use `reset_num_timesteps=False` or `optimize_memory_usage=False`"
                "to avoid that issue."
            )
            # Go to the previous index
            pos = (replay_buffer.pos - 1) % replay_buffer.buffer_size
            replay_buffer.dones[pos] = True

        return super()._setup_learn(
            total_timesteps,
            eval_env,
            callback,
            eval_freq,
            n_eval_episodes,
            log_path,
            reset_num_timesteps,
            tb_log_name,
        )

    def _dump_logs(self) -> None:
        """
        Wrifte log.
        """
        time_elapsed = time.time() - self.start_time
        fps = int((self.num_timesteps - self._num_timesteps_at_start) / (time_elapsed + 1e-8))
        self.logger.record("time/episodes", self._episode_num, exclude="tensorboard")
        # if len(self.ep_info_buffer) > 0 and len(self.ep_info_buffer[0]) > 0:
        #     self.logger.record("rollout/ep_rew_mean", safe_mean([ep_info["r"] for ep_info in self.ep_info_buffer]))
        #     self.logger.record("rollout/ep_len_mean", safe_mean([ep_info["l"] for ep_info in self.ep_info_buffer]))
        self.logger.record("time/fps", fps)
        self.logger.record("time/time_elapsed", int(time_elapsed), exclude="tensorboard")
        self.logger.record("time/total_timesteps", self.num_timesteps, exclude="tensorboard")
        if self.use_sde:
            self.logger.record("train/std", (self.actor.get_std()).mean().item())

        # if len(self.ep_success_buffer) > 0:
        #     self.logger.record("rollout/success_rate", safe_mean(self.ep_success_buffer))
        # Pass the number of timesteps for tensorboard
        self.logger.dump(step=self.num_timesteps)

    def train(self, gradient_steps: int, batch_size: int = 100) -> None:
        # Switch to train mode (this affects batch norm / dropout)
        self.policy.set_training_mode(True)
        # Update learning rate according to schedule
        self._update_learning_rate(self.policy.optimizer)

        losses = []
        q_losses = []
        qc_losses = []
        for _ in range(gradient_steps):
            # Sample replay buffer
            replay_data = self.replay_buffer.sample(batch_size, env=self._vec_normalize_env)

            with th.no_grad():

                next_v_values, next_vc_values = self.calculate_VVC(replay_data.observations)
                target_q_values = replay_data.rewards + (1 - replay_data.dones) * self.gamma * next_v_values
                target_qc_values = replay_data.constraints + (1 - replay_data.dones) * self.gamma * next_vc_values


            # Get current Q-values estimates
            current_q_values = self.q_net(replay_data.observations)
            current_qc_values = self.qc_net(replay_data.observations)

            # Retrieve the q-values for the actions from the replay buffer
            current_q_values = th.gather(current_q_values, dim=1, index=replay_data.actions.long())
            current_qc_values = th.gather(current_qc_values, dim=1, index=replay_data.actions.long())

            # Compute Huber loss (less sensitive to outliers)
            q_loss = F.smooth_l1_loss(current_q_values, target_q_values)
            qc_loss = F.smooth_l1_loss(current_qc_values, target_qc_values)
            loss = q_loss + qc_loss
            losses.append(loss.item())
            q_losses.append(q_loss.item())
            qc_losses.append(qc_loss.item())

            # Optimize the policy
            self.policy.optimizer.zero_grad()
            loss.backward()
            # Clip gradient norm
            th.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
            self.policy.optimizer.step()


        # Increase update counter
        self._n_updates += gradient_steps
        self.logger.record("train/n_updates", self._n_updates, exclude="tensorboard")
        self.logger.record("train/loss", np.mean(losses))
        self.logger.record("train/q_loss", np.mean(q_losses))
        self.logger.record("train/qc_loss", np.mean(qc_losses))

    def predict(
        self,
        observation: np.ndarray,
        state: Optional[Tuple[np.ndarray, ...]] = None,
        episode_start: Optional[np.ndarray] = None,
        deterministic: bool = False,
    ) -> Tuple[np.ndarray, Optional[Tuple[np.ndarray, ...]]]:
        """
        Overrides the base_class predict function to include epsilon-greedy exploration.

        :param observation: the input observation
        :param state: The last states (can be None, used in recurrent policies)
        :param episode_start: The last masks (can be None, used in recurrent policies)
        :param deterministic: Whether or not to return deterministic actions.
        :return: the model's action and the next state
            (used in recurrent policies)
        """
        if not deterministic and np.random.rand() < self.exploration_rate:
            if is_vectorized_observation(maybe_transpose(observation, self.observation_space), self.observation_space):
                if isinstance(self.observation_space, gym.spaces.Dict):
                    n_batch = observation[list(observation.keys())[0]].shape[0]
                else:
                    n_batch = observation.shape[0]
                action = np.array([self.action_space.sample() for _ in range(n_batch)])
            else:
                action = np.array(self.action_space.sample())
        else:
            action, state = self.policy.predict(observation, state, episode_start, deterministic)

        # action, state = self.policy.predict(observation, state, episode_start, deterministic, alpha=self.alpha)
        return action, state

    def collect_whole(self):
        env = self.env
        
        whole_states = np.expand_dims( np.arange(0, self.observation_space.n), axis=-1)
        #whole_actions = np.expand_dims( np.arange(0, self.action_space.n), axis=-1)
        whole_actions =np.arange(0, self.action_space.n)

        new_buffer = ConstReplayBuffer(buffer_size=len(whole_states) * len(whole_actions),
                              observation_space=self.observation_space,
                              action_space=self.action_space,
                              device=self.device,
                              n_envs=1,
                              optimize_memory_usage=self.replay_buffer.optimize_memory_usage,
                              handle_timeout_termination=self.replay_buffer.handle_timeout_termination,)
        def extract_constraint(info):
            if 'constraint' in info:
                return info['constraint']
            else:
                return 0
        
        for s in whole_states:
            for a in whole_actions:
                obs = []
                next_obs = []
                actions = []
                dones = []
                infos = []
                rewards = []

                res = env.env_method('help', s, a)
                print(res)
                for result in res:
                    ns, r, d, info = result
                    obs.append(s)
                    next_obs.append(ns)
                    a = np.array([a])
                    d = np.array([d])
                    r = np.array([r])

                    actions.append(a)
                    dones.append(d)
                    infos.append(info)
                    rewards.append(r)
                obs = np.concatenate(obs, axis=0)
                next_obs = np.concatenate(next_obs, axis=0)
                actions = np.concatenate(actions, axis=0)
                dones = np.concatenate(dones, axis=0)
                # print(obs)
                # print(next_obs)
                # print(actions)
                # print(dones)
                # print(infos)

                    
                # ns, r, d, info = res[0] 
                c = list(map(extract_constraint, infos))
                # print(info)
                new_buffer.add(obs, next_obs, actions, rewards, c, dones, infos)
        
        del self.replay_buffer

        self.replay_buffer = new_buffer

    def learn(
        self,
        total_timesteps: int,
        callback: MaybeCallback = None,
        log_interval: int = 1000,
        eval_env: Optional[GymEnv] = None,
        eval_freq: int = -1,
        n_eval_episodes: int = 5,
        tb_log_name: str = "VI",
        eval_log_path: Optional[str] = None,
        reset_num_timesteps: bool = True,
    ) -> OffPolicyAlgorithm:

        total_timesteps, callback = self._setup_learn(
            total_timesteps,
            eval_env,
            callback,
            eval_freq,
            n_eval_episodes,
            eval_log_path,
            reset_num_timesteps,
            tb_log_name,
        )

        self.collect_whole()

        while self._n_updates < total_timesteps:
            gradient_steps = self.gradient_steps if self.gradient_steps >= 0 else 1
            self.train(batch_size=self.batch_size, gradient_steps=gradient_steps)

            if log_interval is not None and self._n_updates % log_interval == 0:
                self._dump_logs()
        return self

    def _excluded_save_params(self) -> List[str]:
        return super()._excluded_save_params() + ["q_net", "q_net_target"]

    def _get_torch_save_params(self) -> Tuple[List[str], List[str]]:
        state_dicts = ["policy", "policy.optimizer"]

        return state_dicts, []

    def calculate_VVC(self, obs, target=False):
        soft_beta = self.soft_plus(self.beta)
        if target:
            next_q_values = self.q_net_target(obs)
            next_qc_values = self.qc_net_target(obs)
        else:
            next_q_values = self.q_net(obs)
            next_qc_values = self.qc_net(obs)
            

        next_lag_values = next_q_values - soft_beta * next_qc_values
        # print(next_q_values.shape, next_qc_values.shape, next_lag_values.shape)
        action_probs = F.softmax(next_lag_values/self.alpha, dim=1)#, keepdim=True)
        log_action_probs = th.log(action_probs + EPS)

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
    log_action_probs = th.log(action_probs + EPS)

    #next_v_values = self.alpha * th.logsumexp(next_q_values/self.alpha, dim=1, keepdim=True)
    next_v_values = ((next_q_values - self.alpha * log_action_probs) * action_probs).sum(-1, keepdim=True)
    return next_v_values
