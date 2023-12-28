"""Adversarial Inverse Reinforcement Learning (AIRL)."""

import abc
import collections
import dataclasses
import logging
import os
from typing import Callable, Mapping, Optional, Sequence, Tuple, Type

import numpy as np
import torch as th
import torch.utils.tensorboard as thboard
import tqdm
from stable_baselines3.common import base_class, policies, vec_env
from stable_baselines3.sac import policies as sac_policies
from torch.nn import functional as F

from imitation.algorithms import base
from imitation.data import buffer, rollout, types, wrappers
from imitation.rewards import reward_nets, reward_wrapper
from imitation.util import logger, networks, util

import torch as th
from stable_baselines3.common import base_class, policies, vec_env
from stable_baselines3.sac import policies as sac_policies

from imitation.algorithms import base
from imitation.algorithms.adversarial import common
from imitation.rewards import reward_nets

from stable_baselines3.common.utils import obs_as_tensor, safe_mean

STOCHASTIC_POLICIES = (sac_policies.SACPolicy, policies.ActorCriticPolicy)
from imitation.util.networks import RunningNorm

class TCL(base.DemonstrationAlgorithm[types.Transitions]):
    venv: vec_env.VecEnv
    """The original vectorized environment."""

    venv_train: vec_env.VecEnv
    """Like `self.venv`, but wrapped with train reward unless in debug mode.
    If `debug_use_ground_truth=True` was passed into the initializer then
    `self.venv_train` is the same as `self.venv`."""

    def __init__(
        self,
        *,
        demonstrations: base.AnyTransitions,
        demo_batch_size: int,
        venv: vec_env.VecEnv,
        gen_algo: base_class.BaseAlgorithm,
        reward_net: reward_nets.RewardNet,
        primary_net: reward_nets.RewardNet,
        constraint_net: reward_nets.RewardNet,
        custom_net: reward_nets.RewardNet, 
        n_disc_updates_per_round: int = 2,
        log_dir: str = "output/",
        disc_opt_cls: Type[th.optim.Optimizer] = th.optim.Adam,
        disc_opt_kwargs: Optional[Mapping] = None,
        primary_disc_opt_kwargs: Optional[Mapping] = None,
        const_disc_opt_kwargs: Optional[Mapping] = None,
        gen_train_timesteps: Optional[int] = None,
        gen_replay_buffer_capacity: Optional[int] = None,
        custom_logger: Optional[logger.HierarchicalLogger] = None,
        init_tensorboard: bool = False,
        init_tensorboard_graph: bool = False,
        debug_use_ground_truth: bool = False,
        allow_variable_horizon: bool = False,
        
    ):
        self.demo_batch_size = demo_batch_size
        self._demo_data_loader = None
        self._endless_expert_iterator = None
        super().__init__(
            demonstrations=demonstrations,
            custom_logger=custom_logger,
            allow_variable_horizon=allow_variable_horizon,
        )

        self._global_step = 0
        self._disc_step = 0
        self.n_disc_updates_per_round = n_disc_updates_per_round

        self.debug_use_ground_truth = debug_use_ground_truth
        self.venv = venv
        self.gen_algo = gen_algo

        self._primary_net = primary_net.to(gen_algo.device)
        self._constraint_net = constraint_net.to(gen_algo.device)
        self._custom_net = custom_net.to(gen_algo.device)
        self._reward_net =lambda *args: self._constraint_net(*args) + self._primary_net(*args).detach()
        self._running_norm = RunningNorm(1).to(gen_algo.device)
        
        self._log_dir = log_dir

        # Create graph for optimising/recording stats on discriminator
        self._disc_opt_cls = disc_opt_cls
        self._disc_opt_kwargs = disc_opt_kwargs or {}
        self._primary_disc_opt_kwargs = primary_disc_opt_kwargs or {}
        self._const_disc_opt_kwargs = const_disc_opt_kwargs or {}
        self._init_tensorboard = init_tensorboard
        self._init_tensorboard_graph = init_tensorboard_graph
        self._disc_opt = []
        # self._disc_opt.append(
        #     self._disc_opt_cls(
        #         self._reward_net.parameters(),
        #         **self._disc_opt_kwargs,
        #     )
        # )
        self._disc_opt.append(
            self._disc_opt_cls(
                self._primary_net.parameters(),
                **self._primary_disc_opt_kwargs,
            )
        )
        self._disc_opt.append(
            self._disc_opt_cls(
                self._constraint_net.parameters(),
                **self._const_disc_opt_kwargs,
            )
        )
       
        self._disc_opt.append(
            self._disc_opt_cls(
                self._custom_net.parameters(),
                **self._const_disc_opt_kwargs,
            )
        ) 
        if self._init_tensorboard:
            logging.info("building summary directory at " + self._log_dir)
            summary_dir = os.path.join(self._log_dir, "summary")
            os.makedirs(summary_dir, exist_ok=True)
            self._summary_writer = thboard.SummaryWriter(summary_dir)

        venv = self.venv_buffering = wrappers.BufferingWrapper(self.venv)

        if debug_use_ground_truth:
            # Would use an identity reward fn here, but RewardFns can't see rewards.
            self.venv_wrapped = venv
            self.gen_callback = None
        else:
            venv = self.venv_wrapped = reward_wrapper.PrimaryConstRewardVecEnvWrapper(
                venv,
                reward_fn= lambda *args:  self.update_stats( self.primary_train.predict_processed(*args) +self.constraint_train.predict_processed (*args) , update_stats=False),
                primary_fn= lambda *args: self.update_stats( self.primary_train.predict_processed(*args), update_stats=False),
                constraint_fn= lambda *args: self.update_stats( self.constraint_train.predict_processed(*args), update_stats=True),
            )
            self.gen_callback = self.venv_wrapped.make_log_callback()
        self.venv_train = self.venv_wrapped

        self.gen_algo.set_env(self.venv_train)
        self.gen_algo.set_logger(self.logger)

        if gen_train_timesteps is None:
            gen_algo_env = self.gen_algo.get_env()
            assert gen_algo_env is not None
            self.gen_train_timesteps = gen_algo_env.num_envs
            if hasattr(self.gen_algo, "n_steps"):  # on policy
                self.gen_train_timesteps *= self.gen_algo.n_steps
        else:
            self.gen_train_timesteps = gen_train_timesteps

        self.gen_replay_buffer_capacity = gen_replay_buffer_capacity
        if gen_replay_buffer_capacity is None:
            gen_replay_buffer_capacity = self.gen_train_timesteps
        self._gen_replay_buffer = buffer.ReplayBuffer(
            gen_replay_buffer_capacity,
            self.venv,
        )
        
    def update_stats(self, outputs, update_stats=False):
        
        rew_th = th.tensor(
            outputs,
            device=self.gen_algo.device,
        )
        rew = self._running_norm(rew_th).detach().cpu().numpy().flatten()
        # outputs = self._running_norm(outputs)
        
        if update_stats:
            with th.no_grad():
                self._running_norm.update_stats(rew_th)
        return rew

    @property
    def policy(self) -> policies.BasePolicy:
        return self.gen_algo.policy

    @property
    def primary_policy(self) -> policies.BasePolicy:
        return self.gen_algo.primary_policy

    @property
    def const_policy(self) -> policies.BasePolicy:
        return self.gen_algo.const_policy

    def logits_expert_is_high(
        self,
        state: th.Tensor,
        action: th.Tensor,
        next_state: th.Tensor,
        done: th.Tensor,
        log_policy_act_prob: th.Tensor,
    ) -> th.Tensor:
        r"""Compute the discriminator's logits for each state-action sample.
        In Fu's AIRL paper (https://arxiv.org/pdf/1710.11248.pdf), the
        discriminator output was given as
        .. math::
            D_{\theta}(s,a) =
            \frac{ \exp{r_{\theta}(s,a)} } { \exp{r_{\theta}(s,a)} + \pi(a|s) }
        with a high value corresponding to the expert and a low value corresponding to
        the generator.
        In other words, the discriminator output is the probability that the action is
        taken by the expert rather than the generator.
        The logit of the above is given as
        .. math::
            \operatorname{logit}(D_{\theta}(s,a)) = r_{\theta}(s,a) - \log{ \pi(a|s) }
        which is what is returned by this function.
        Args:
            state: The state of the environment at the time of the action.
            action: The action taken by the expert or generator.
            next_state: The state of the environment after the action.
            done: whether a `terminal state` (as defined under the MDP of the task) has
                been reached.
            log_policy_act_prob: The log probability of the action taken by the
                generator, :math:`\log{ \pi(a|s) }`.
        Returns:
            The logits of the discriminator for each state-action sample.
        Raises:
            TypeError: If `log_policy_act_prob` is None.
        """
        if log_policy_act_prob is None:
            raise TypeError(
                "Non-None `log_policy_act_prob` is required for this method.",
            )
        primary_output_train = self._reward_net(state, action, next_state, done)
        return primary_output_train - log_policy_act_prob

    def primary_logits_expert_is_high(
        self,
        state: th.Tensor,
        action: th.Tensor,
        next_state: th.Tensor,
        done: th.Tensor,
        primary_log_policy_act_prob: th.Tensor,
    ) -> th.Tensor:
        if primary_log_policy_act_prob is None:
            raise TypeError(
                "Non-None `log_policy_act_prob` is required for this method.",
            )
        const_output_train = self._primary_net(state, action, next_state, done)
        return const_output_train 

    @property
    def reward_train(self) -> reward_nets.RewardNet:
        return self._reward_net

    @property
    def reward_test(self) -> reward_nets.RewardNet:
        """Returns the unshaped version of reward network used for testing."""
        reward_net = self._reward_net
        # Recursively return the base network of the wrapped reward net
        while isinstance(reward_net, reward_nets.RewardNetWrapper):
            reward_net = reward_net.base
        return reward_net

    @property
    def custom_train(self) -> reward_nets.RewardNet:
        return self._custom_net

    @property
    def custom_test (self) -> reward_nets.RewardNet:
        """Returns the unshaped version of reward network used for testing."""
        reward_net = self._custom_net
        # Recursively return the base network of the wrapped reward net
        while isinstance(reward_net, reward_nets.RewardNetWrapper):
            reward_net = reward_net.base
        return reward_net
    @property
    def constraint_train(self) -> reward_nets.RewardNet:
        return self._constraint_net

    @property
    def constraint_test (self) -> reward_nets.RewardNet:
        """Returns the unshaped version of reward network used for testing."""
        reward_net = self._constraint_net
        # Recursively return the base network of the wrapped reward net
        while isinstance(reward_net, reward_nets.RewardNetWrapper):
            reward_net = reward_net.base
        return reward_net

    @property
    def primary_train(self) -> reward_nets.RewardNet:
        return self._primary_net

    @property
    def primary_test (self) -> reward_nets.RewardNet:
        """Returns the unshaped version of reward network used for testing."""
        reward_net = self._primary_net
        # Recursively return the base network of the wrapped reward net
        while isinstance(reward_net, reward_nets.RewardNetWrapper):
            reward_net = reward_net.base
        return reward_net

    def set_demonstrations(self, demonstrations: base.AnyTransitions) -> None:
        self._demo_data_loader = base.make_data_loader(
            demonstrations,
            self.demo_batch_size,
        )
        self._endless_expert_iterator = util.endless_iter(self._demo_data_loader)

    def _next_expert_batch(self) -> Mapping:
        return next(self._endless_expert_iterator)

    def rew_gen(
        self,
        state: th.Tensor,
        action: th.Tensor,
        next_state: th.Tensor,
        done: th.Tensor,
        is_expert: th.Tensor, 
    ):

        const_output_train = self._reward_net(state, action, next_state, done)[~is_expert]# + self._constraint_net(state, action, next_state, done)[~is_expert]
        return const_output_train.mean()

    def rew_expert(
        self,
        state: th.Tensor,
        action: th.Tensor,
        next_state: th.Tensor,
        done: th.Tensor, 
        is_expert: th.Tensor,

    ):

        const_output_train = self._reward_net(state, action, next_state, done)[is_expert] #+ self._constraint_net(state, action, next_state, done)[is_expert]
        reward = - const_output_train + 0.5 * const_output_train ** 2 
        # reward = -th.exp(-const_output_train) * const_output_train
        # reward = -const_output_train
        return reward.mean()

    def custom_gen(
        self,
        state: th.Tensor,
        action: th.Tensor,
        next_state: th.Tensor,
        done: th.Tensor,
        is_expert: th.Tensor, 
        log_policy_act_prob: th.Tensor,
        primary_log_policy_act_prob: th.Tensor,
        clip_range: float = 0.4,
        #clip_range: Union[float, Schedule] = 0.2,
    ):

        const_output_train = self._custom_net(state, action, next_state, done)[~is_expert]
        return const_output_train.mean()

    def custom_expert(
        self,
        state: th.Tensor,
        action: th.Tensor,
        next_state: th.Tensor,
        done: th.Tensor, 
        is_expert: th.Tensor,
    ):
        const_output_train = self._custom_net(state, action, next_state, done)[is_expert]
        reward = -th.exp(-const_output_train) * const_output_train
        return reward.mean()
    
    def primary_gen(
        self,
        state: th.Tensor,
        action: th.Tensor,
        next_state: th.Tensor,
        done: th.Tensor,
        is_expert: th.Tensor, 
        log_policy_act_prob: th.Tensor,
        primary_log_policy_act_prob: th.Tensor,
        clip_range: float = 0.4,
        #clip_range: Union[float, Schedule] = 0.2,
    ):

        const_output_train = self._primary_net(state, action, next_state, done)[~is_expert]
        # ratio = th.exp(primary_log_policy_act_prob - log_policy_act_prob)
        # const_output_train = const_output_train *  th.clamp(ratio, 1 - clip_range, 1 + clip_range)
        return const_output_train.mean()

    def primary_expert(
        self,
        state: th.Tensor,
        action: th.Tensor,
        next_state: th.Tensor,
        done: th.Tensor, 
        is_expert: th.Tensor,
    ):
        const_output_train = self._primary_net(state, action, next_state, done)[is_expert]
        # reward = - const_output_train + 0.5 * const_output_train ** 2 
        # reward = - const_output_train + 0.5 * const_output_train ** 2 
        reward = -th.exp(-const_output_train) * const_output_train
        # reward = - ( th.exp(-const_output_train) /(2 - th.exp(- const_output_train)) ) * const_output_train
        return reward.mean()


    def const_gen(
        self,
        state: th.Tensor,
        action: th.Tensor,
        next_state: th.Tensor,
        done: th.Tensor,
        is_expert: th.Tensor, 
        log_policy_act_prob: th.Tensor,
        constraint_log_policy_act_prob: th.Tensor,
        clip_range: float = 0.4,
        #clip_range: Union[float, Schedule] = 0.2,
    ):

        const_output_train = self._reward_net(state, action, next_state, done)[~is_expert].detach() -self._primary_net(state, action, next_state, done)[~is_expert]
        ratio = th.exp(constraint_log_policy_act_prob - log_policy_act_prob)
        const_output_train = const_output_train * th.clamp( th.clamp(ratio, 1 - clip_range, 1 + clip_range)
        , 1 - clip_range, 1 + clip_range)
        return const_output_train.mean()

    def const_expert(
        self,
        state: th.Tensor,
        action: th.Tensor,
        next_state: th.Tensor,
        done: th.Tensor, 
        is_expert: th.Tensor,
    ):

        const_output_train = self._reward_net(state, action, next_state, done)[is_expert].detach()-self._primary_net(state, action, next_state, done)[is_expert]
        reward = - const_output_train + 0.5 * const_output_train ** 2 
        return reward.mean()

    def reg(
        self,
        Q_net,
        r_net,
        state: th.Tensor,
        action: th.Tensor,
        next_state: th.Tensor,
        next_action: th.Tensor,
        done: th.Tensor, 
    ):
        """
        Q_cur = Q_net(state, action, next_state=None, done=done).detach()
        Q_next = Q_net(next_state, next_action, next_state=None, done=done).detach()
        r_net = r_net(state, action, next_state, done)  
        reg1 = ((self.gen_algo.gamma * Q_next - Q_cur - r_net)**2).mean()
        """

        Q_cur = Q_net(state, action, next_state=None, done=done).detach()
        r_net = r_net(state, action, next_state, done)  
        reg1 = ((Q_cur - r_net)**2).mean()
        return reg1 #+ reg2
    
    def train_disc(
        self,
        *,
        expert_samples: Optional[Mapping] = None,
        gen_samples: Optional[Mapping] = None,
    ) -> Optional[Mapping[str, float]]:
        """Perform a single discriminator update, optionally using provided samples.
        Args:
            expert_samples: Transition samples from the expert in dictionary form.
                If provided, must contain keys corresponding to every field of the
                `Transitions` dataclass except "infos". All corresponding values can be
                either NumPy arrays or Tensors. Extra keys are ignored. Must contain
                `self.demo_batch_size` samples. If this argument is not provided, then
                `self.demo_batch_size` expert samples from `self.demo_data_loader` are
                used by default.
            gen_samples: Transition samples from the generator policy in same dictionary
                form as `expert_samples`. If provided, must contain exactly
                `self.demo_batch_size` samples. If not provided, then take
                `len(expert_samples)` samples from the generator replay buffer.
        Returns:
            Statistics for discriminator (e.g. loss, accuracy).
        """
        # self._reward_net.train()
        self._primary_net.train()
        self._constraint_net.train()

        with self.logger.accumulate_means("disc"):
            # optionally write TB summaries for collected ops
            write_summaries = self._init_tensorboard and self._global_step % 20 == 0

            # compute loss
            batch = self._make_disc_train_batch(
                gen_samples=gen_samples,
                expert_samples=expert_samples,
            )
            disc_logits = self.logits_expert_is_high(
                batch["state"],
                batch["action"],
                batch["next_state"],
                batch["done"],
                batch["log_policy_act_prob"],
            )
            loss = F.binary_cross_entropy_with_logits(
                disc_logits,
                batch["labels_expert_is_one"].float(),
            )
            
            new_batch = self._make_disc_train_batch(
                gen_samples=gen_samples,
                expert_samples=gen_samples,
            ) 

            primary_disc_logits = self.primary_logits_expert_is_high(
                new_batch["state"],
                new_batch["primary_action"],
                new_batch["next_state"],
                new_batch["done"],
                new_batch["primary_log_policy_act_prob"],
            )
            primary_loss =F.binary_cross_entropy_with_logits(
                primary_disc_logits,
                new_batch["labels_expert_is_one"].float(),
            ) 
             
            loss += primary_loss 
            for op in self._disc_opt:
                op.zero_grad()
            loss.backward()
            for op in self._disc_opt:
                op.step()
            self._disc_step += 1

            reg_loss = self.train_project(
                q_net = self._primary_net,
                r_net = self._custom_net,
            ) 
            # compute/write stats and TensorBoard data
            with th.no_grad():
                train_stats = compute_train_stats(
                    disc_logits,
                    batch["labels_expert_is_one"],
                    loss,
                )
            self.logger.record("global_step", self._global_step)
            
            for k, v in train_stats.items():
                self.logger.record(k, v)
            self.logger.record("reg_loss", float(reg_loss.detach().cpu().numpy()))
            self.logger.dump(self._disc_step)
            if write_summaries:
                self._summary_writer.add_histogram("disc_logits", disc_logits.detach())
                # self._summary_writer.add_histogram("disc_logits(primary)", primary_disc_logits.detach())

        # self._reward_net.eval()
        self._primary_net.eval()
        self._constraint_net.eval()
        return train_stats

    def train_gen(
        self,
        total_timesteps: Optional[int] = None,
        learn_kwargs: Optional[Mapping] = None,
    ) -> None:
        """Trains the generator to maximize the discriminator loss.
        After the end of training populates the generator replay buffer (used in
        discriminator training) with `self.disc_batch_size` transitions.
        Args:
            total_timesteps: The number of transitions to sample from
                `self.venv_train` during training. By default,
                `self.gen_train_timesteps`.
            learn_kwargs: kwargs for the Stable Baselines `RLModel.learn()`
                method.
        """
        if total_timesteps is None:
            total_timesteps = self.gen_train_timesteps
        if learn_kwargs is None:
            learn_kwargs = {}

        with self.logger.accumulate_means("gen"):
            self.gen_algo.learn(
                total_timesteps=total_timesteps,
                reset_num_timesteps=False,
                callback=self.gen_callback,
                **learn_kwargs,
            )
            self._global_step += 1

        gen_trajs, ep_lens = self.venv_buffering.pop_trajectories()
        self._check_fixed_horizon(ep_lens)
        gen_samples = rollout.flatten_trajectories_with_rew(gen_trajs)
        self._gen_replay_buffer.store(gen_samples)

    def train(
        self,
        total_timesteps: int,
        callback: Optional[Callable[[int], None]] = None,
    ) -> None:
        """Alternates between training the generator and discriminator.
        Every "round" consists of a call to `train_gen(self.gen_train_timesteps)`,
        a call to `train_disc`, and finally a call to `callback(round)`.
        Training ends once an additional "round" would cause the number of transitions
        sampled from the environment to exceed `total_timesteps`.
        Args:
            total_timesteps: An upper bound on the number of transitions to sample
                from the environment during training.
            callback: A function called at the end of every round which takes in a
                single argument, the round number. Round numbers are in
                `range(total_timesteps // self.gen_train_timesteps)`.
        """
        n_rounds = total_timesteps // self.gen_train_timesteps
        assert n_rounds >= 1, (
            "No updates (need at least "
            f"{self.gen_train_timesteps} timesteps, have only "
            f"total_timesteps={total_timesteps})!"
        )
        for r in tqdm.tqdm(range(0, n_rounds), desc="round"):
            self.train_gen(self.gen_train_timesteps)
            for _ in range(self.n_disc_updates_per_round):
                # with networks.training(self.reward_train):
                    # switch to training mode (affects dropout, normalization)
                self.train_disc()
            if callback:
                callback(r)
            self.logger.dump(self._global_step)

    def _torchify_array(self, ndarray: Optional[np.ndarray]) -> Optional[th.Tensor]:
        if ndarray is not None:
            return th.as_tensor(ndarray, device=self.primary_train.device)

    def _get_log_policy_act_prob(
        self,
        obs_th: th.Tensor,
        acts_th: th.Tensor,
        policy = None,
    ) -> Optional[th.Tensor]:
        """Evaluates the given actions on the given observations.
        Args:
            obs_th: A batch of observations.
            acts_th: A batch of actions.
        Returns:
            A batch of log policy action probabilities.
        """
        if policy is None:
            policy = self.policy
        if isinstance(policy, policies.ActorCriticPolicy):
            # policies.ActorCriticPolicy has a concrete implementation of
            # evaluate_actions to generate log_policy_act_prob given obs and actions.
            _, log_policy_act_prob_th, _ = policy.evaluate_actions(
                obs_th,
                acts_th,
            )
        elif isinstance(policy, sac_policies.SACPolicy):
            gen_algo_actor = policy.actor
            assert gen_algo_actor is not None
            # generate log_policy_act_prob from SAC actor.
            mean_actions, log_std, _ = gen_algo_actor.get_action_dist_params(obs_th)
            distribution = gen_algo_actor.action_dist.proba_distribution(
                mean_actions,
                log_std,
            )
            # SAC applies a squashing function to bound the actions to a finite range
            # `acts_th` need to be scaled accordingly before computing log prob.
            # Scale actions only if the policy squashes outputs.
            assert policy.squash_output
            scaled_acts_th = policy.scale_action(acts_th)
            log_policy_act_prob_th = distribution.log_prob(scaled_acts_th)
        else:
            return None
        return log_policy_act_prob_th

    def train_project(
        self,
        q_net,
        r_net,
        num_itr = 10,
    ):
        total_loss = 0
        for itr in range(num_itr):
            expert_samples = self._next_expert_batch()
            gen_samples = self._gen_replay_buffer.sample(self.demo_batch_size)
            gen_samples = types.dataclass_quick_asdict(gen_samples)

            if gen_samples is None:
                if self._gen_replay_buffer.size() == 0:
                    raise RuntimeError(
                        "No generator samples for training. " "Call `train_gen()` first.",
                    )
                gen_samples = self._gen_replay_buffer.sample(self.demo_batch_size)
                gen_samples = types.dataclass_quick_asdict(gen_samples)

            n_gen = len(gen_samples["obs"])
            n_expert = len(expert_samples["obs"])
            # Guarantee that Mapping arguments are in mutable form.
            expert_samples = dict(expert_samples)
            gen_samples = dict(gen_samples)

            # Convert applicable Tensor values to NumPy.
            for field in dataclasses.fields(types.Transitions):
                k = field.name
                if k == "infos":
                    continue
                for d in [gen_samples, expert_samples]:
                    if isinstance(d[k], th.Tensor):
                        d[k] = d[k].detach().numpy()
            assert isinstance(gen_samples["obs"], np.ndarray)
            assert isinstance(expert_samples["obs"], np.ndarray)

            # Check dimensions.
            n_samples = n_expert + n_gen
            assert n_expert == len(expert_samples["acts"])
            assert n_expert == len(expert_samples["next_obs"])
            assert n_gen == len(gen_samples["acts"])
            assert n_gen == len(gen_samples["next_obs"])

            # Concatenate rollouts, and label each row as expert or generator.
            obs = np.concatenate([expert_samples["obs"], gen_samples["obs"]])
            acts = np.concatenate([expert_samples["acts"], gen_samples["acts"]])
            next_obs = np.concatenate([expert_samples["next_obs"], gen_samples["next_obs"]])
            dones = np.concatenate([expert_samples["dones"], gen_samples["dones"]])
            # notice that the labels use the convention that expert samples are
            # labelled with 1 and generator samples with 0.

            with th.no_grad():
                # Convert to pytorch tensor or to TensorDict
                _obs_tensor = obs_as_tensor(next_obs, self.gen_algo.device)
                _next_acts, values, log_probs = self.policy(_obs_tensor)
                _next_acts = _next_acts.detach().cpu().numpy()
                assert _next_acts.shape == acts.shape
            next_acts = _next_acts

            obs_th, acts_th, next_obs_th, dones_th = self.primary_train.preprocess(
                obs,
                acts,
                next_obs,
                dones,
            )

            _, next_acts_th, _, _ = self.primary_train.preprocess(
                obs,
                next_acts,
                next_obs,
                dones,
            )
            batch = {
                "state": obs_th,
                "action": acts_th,
                "next_action": next_acts_th,
                "next_state": next_obs_th,
                "done": dones_th,
            }

            loss = self.reg(
                q_net,
                r_net,
                batch["state"],
                batch["action"],
                batch["next_state"], 
                batch["next_action"], 
                batch["done"],
            )
            for op in self._disc_opt:
                op.zero_grad()
            loss.backward()
            for op in self._disc_opt:
                op.step()
            total_loss += loss
        return total_loss

    def _make_disc_train_batch(
        self,
        *,
        gen_samples: Optional[Mapping] = None,
        expert_samples: Optional[Mapping] = None,
    ) -> Mapping[str, th.Tensor]:
        """Build and return training batch for the next discriminator update.
        Args:
            gen_samples: Same as in `train_disc`.
            expert_samples: Same as in `train_disc`.
        Returns:
            The training batch: state, action, next state, dones, labels
            and policy log-probabilities.
        Raises:
            RuntimeError: Empty generator replay buffer.
            ValueError: `gen_samples` or `expert_samples` batch size is
                different from `self.demo_batch_size`.
        """
        if expert_samples is None:
            expert_samples = self._next_expert_batch()

        if gen_samples is None:
            if self._gen_replay_buffer.size() == 0:
                raise RuntimeError(
                    "No generator samples for training. " "Call `train_gen()` first.",
                )
            gen_samples = self._gen_replay_buffer.sample(self.demo_batch_size)
            gen_samples = types.dataclass_quick_asdict(gen_samples)

        n_gen = len(gen_samples["obs"])
        n_expert = len(expert_samples["obs"])
        if not (n_gen == n_expert == self.demo_batch_size):
            raise ValueError(
                "Need to have exactly self.demo_batch_size number of expert and "
                "generator samples, each. "
                f"(n_gen={n_gen} n_expert={n_expert} "
                f"demo_batch_size={self.demo_batch_size})",
            )

        # Guarantee that Mapping arguments are in mutable form.
        expert_samples = dict(expert_samples)
        gen_samples = dict(gen_samples)

        # Convert applicable Tensor values to NumPy.
        for field in dataclasses.fields(types.Transitions):
            k = field.name
            if k == "infos":
                continue
            for d in [gen_samples, expert_samples]:
                if isinstance(d[k], th.Tensor):
                    d[k] = d[k].detach().numpy()
        assert isinstance(gen_samples["obs"], np.ndarray)
        assert isinstance(expert_samples["obs"], np.ndarray)

        # Check dimensions.
        n_samples = n_expert + n_gen
        assert n_expert == len(expert_samples["acts"])
        assert n_expert == len(expert_samples["next_obs"])
        assert n_gen == len(gen_samples["acts"])
        assert n_gen == len(gen_samples["next_obs"])

        # Concatenate rollouts, and label each row as expert or generator.
        obs = np.concatenate([expert_samples["obs"], gen_samples["obs"]])
        acts = np.concatenate([expert_samples["acts"], gen_samples["acts"]])
        next_obs = np.concatenate([expert_samples["next_obs"], gen_samples["next_obs"]])
        dones = np.concatenate([expert_samples["dones"], gen_samples["dones"]])
        # notice that the labels use the convention that expert samples are
        # labelled with 1 and generator samples with 0.

        with th.no_grad():
            # Convert to pytorch tensor or to TensorDict
            obs_tensor = obs_as_tensor(gen_samples["obs"], self.gen_algo.device)
            actions, values, log_probs = self.const_policy(obs_tensor)
            actions = actions.detach().cpu().numpy()
            assert actions.shape == gen_samples["acts"].shape
        const_acts = np.concatenate([expert_samples["acts"], actions])


        with th.no_grad():
            # Convert to pytorch tensor or to TensorDict
            obs_tensor = obs_as_tensor(gen_samples["obs"], self.gen_algo.device)
            primary_actions, values, log_probs = self.primary_policy(obs_tensor)
            primary_actions = primary_actions.detach().cpu().numpy()
            assert primary_actions.shape == gen_samples["acts"].shape
        primary_acts = np.concatenate([expert_samples["acts"], primary_actions])


        with th.no_grad():
            # Convert to pytorch tensor or to TensorDict
            _obs_tensor = obs_as_tensor(obs, self.gen_algo.device)
            _next_acts, values, log_probs = self.policy(_obs_tensor)
            _next_acts = _next_acts.detach().cpu().numpy()
            assert _next_acts.shape == acts.shape
        next_acts = _next_acts

        labels_expert_is_one = np.concatenate(
            [np.ones(n_expert, dtype=int), np.zeros(n_gen, dtype=int)],
        )

        # Calculate generator-policy log probabilities.
        with th.no_grad():
            obs_th = th.as_tensor(obs, device=self.gen_algo.device)
            acts_th = th.as_tensor(acts, device=self.gen_algo.device)

            primary_acts_th = th.as_tensor(primary_acts, device=self.gen_algo.device)
            const_acts_th = th.as_tensor(const_acts, device=self.gen_algo.device)

            log_policy_act_prob = self._get_log_policy_act_prob(obs_th, acts_th)
            primary_log_policy_act_prob = self._get_log_policy_act_prob(obs_th, primary_acts_th, self.primary_policy)
            const_log_policy_act_prob = self._get_log_policy_act_prob(obs_th, acts_th, self.const_policy)
            if log_policy_act_prob is not None:
                assert len(log_policy_act_prob) == n_samples
                log_policy_act_prob = log_policy_act_prob.reshape((n_samples,))
            if primary_log_policy_act_prob is not None:
                assert len(primary_log_policy_act_prob) == n_samples
                primary_log_policy_act_prob = primary_log_policy_act_prob.reshape((n_samples,))
            if const_log_policy_act_prob is not None:
                assert len(const_log_policy_act_prob) == n_samples
                const_log_policy_act_prob = const_log_policy_act_prob.reshape((n_samples,))
            del obs_th, acts_th, primary_acts_th, const_acts_th  # unneeded

        obs_th, acts_th, next_obs_th, dones_th = self.primary_train.preprocess(
            obs,
            acts,
            next_obs,
            dones,
        )

        _, primary_acts_th, _, _ = self.primary_train.preprocess(
            obs,
            primary_acts,
            next_obs,
            dones,
        )
        _, const_acts_th, _, _ = self.primary_train.preprocess(
            obs,
            const_acts,
            next_obs,
            dones,
        )

        _, next_acts_th, _, _ = self.primary_train.preprocess(
            obs,
            next_acts,
            next_obs,
            dones,
        )
        batch_dict = {
            "state": obs_th,
            "action": acts_th,
            "primary_action": primary_acts_th,
            "const_action": const_acts_th,
            "next_action": next_acts_th,
            "next_state": next_obs_th,
            "done": dones_th,
            "labels_expert_is_one": self._torchify_array(labels_expert_is_one),
            "log_policy_act_prob": log_policy_act_prob,
            "primary_log_policy_act_prob": primary_log_policy_act_prob,
            "const_log_policy_act_prob": const_log_policy_act_prob,
        }

        return batch_dict
def compute_train_stats(
    disc_logits_expert_is_high: th.Tensor,
    labels_expert_is_one: th.Tensor,
    disc_loss: th.Tensor,
) -> Mapping[str, float]:
    """Train statistics for GAIL/AIRL discriminator.
    Args:
        disc_logits_expert_is_high: discriminator logits produced by
            `AdversarialTrainer.logits_expert_is_high`.
        labels_expert_is_one: integer labels describing whether logit was for an
            expert (0) or generator (1) sample.
        disc_loss: final discriminator loss.
    Returns:
        A mapping from statistic names to float values.
    """
    with th.no_grad():
        # Logits of the discriminator output; >0 for expert samples, <0 for generator.
        bin_is_generated_pred = disc_logits_expert_is_high < 0
        # Binary label, so 1 is for expert, 0 is for generator.
        bin_is_generated_true = labels_expert_is_one == 0
        bin_is_expert_true = th.logical_not(bin_is_generated_true)
        int_is_generated_pred = bin_is_generated_pred.long()
        int_is_generated_true = bin_is_generated_true.long()
        n_generated = float(th.sum(int_is_generated_true))
        n_labels = float(len(labels_expert_is_one))
        n_expert = n_labels - n_generated
        pct_expert = n_expert / float(n_labels) if n_labels > 0 else float("NaN")
        n_expert_pred = int(n_labels - th.sum(int_is_generated_pred))
        if n_labels > 0:
            pct_expert_pred = n_expert_pred / float(n_labels)
        else:
            pct_expert_pred = float("NaN")
        correct_vec = th.eq(bin_is_generated_pred, bin_is_generated_true)
        acc = th.mean(correct_vec.float())

        _n_pred_expert = th.sum(th.logical_and(bin_is_expert_true, correct_vec))
        if n_expert < 1:
            expert_acc = float("NaN")
        else:
            # float() is defensive, since we cannot divide Torch tensors by
            # Python ints
            expert_acc = _n_pred_expert / float(n_expert)

        _n_pred_gen = th.sum(th.logical_and(bin_is_generated_true, correct_vec))
        _n_gen_or_1 = max(1, n_generated)
        generated_acc = _n_pred_gen / float(_n_gen_or_1)

        label_dist = th.distributions.Bernoulli(logits=disc_logits_expert_is_high)
        entropy = th.mean(label_dist.entropy())

    pairs = [
        ("disc_loss", float(th.mean(disc_loss))),
        # accuracy, as well as accuracy on *just* expert examples and *just*
        # generated examples
        ("disc_acc", float(acc)),
        ("disc_acc_expert", float(expert_acc)),
        ("disc_acc_gen", float(generated_acc)),
        # entropy of the predicted label distribution, averaged equally across
        # both classes (if this drops then disc is very good or has given up)
        ("disc_entropy", float(entropy)),
        # true number of expert demos and predicted number of expert demos
        ("disc_proportion_expert_true", float(pct_expert)),
        ("disc_proportion_expert_pred", float(pct_expert_pred)),
        ("n_expert", float(n_expert)),
        ("n_generated", float(n_generated)),
    ]  # type: Sequence[Tuple[str, float]]
    return collections.OrderedDict(pairs)