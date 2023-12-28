"""Configuration settings for train_rl, training a policy with RL."""

import sacred

from imitation.scripts.common import common, rl, train
import sb3_contrib 
from stable_baselines3.common.utils import get_linear_fn

from imitation.rewards import reward_nets, serialize
from imitation.util import networks, util
from imitation.algorithms.adversarial.gail import RewardNetFromDiscriminatorLogit
train_rl_ex = sacred.Experiment(
    "train_rl",
    ingredients=[common.common_ingredient, train.train_ingredient, rl.rl_ingredient],
    save_git_info=False,
)


@train_rl_ex.config
def train_rl_defaults():
    total_timesteps = int(1e6)  # Number of training timesteps in model.learn()
    normalize_reward = True  # Use VecNormalize to normalize the reward
    normalize_kwargs = dict()  # kwargs for `VecNormalize`

    # If specified, overrides the ground-truth environment reward
    reward_type = None  # override reward type
    reward_path = None  # override reward path
    load_reward_kwargs = {}

    rollout_save_final = True  # If True, save after training is finished.
    rollout_save_n_timesteps = None  # Min timesteps saved per file, optional.
    rollout_save_n_episodes = None  # Num episodes saved per file, optional.

    policy_save_interval = 10000  # Num timesteps between saves (<=0 disables)
    policy_save_final = True  # If True, save after training is finished.

    agent_path = None  # Path to load agent from, optional.


@train_rl_ex.config
def default_end_cond(rollout_save_n_timesteps, rollout_save_n_episodes):
    # Only set default if both end cond options are None.
    # This way the Sacred CLI caller can set `rollout_save_n_episodes` only
    # without getting an error that `rollout_save_n_timesteps is not None`.
    if rollout_save_n_timesteps is None and rollout_save_n_episodes is None:
        rollout_save_n_timesteps = 2000  # Min timesteps saved per file, optional.


 
# Custom Gym env configs
@train_rl_ex.named_config
def peginhole_v1():
    normalize_reward=False
    # normalize = False  # Use VecNormalize
    common = dict(env_name="Gripper-v0",
    max_episode_steps = 30,
    num_vec = 16  # number of environments in VecEnv)
    )
    rl = dict(
        rl_cls=sb3_contrib.SAC,
        # batch_size=4096,
        batch_size = 2048,  # batch size for RL algorithm
        rl_kwargs=dict(
        #     gamma=0.99,
        learning_rate=3e-4,
        tau=0.01, gamma=0.99, gradient_steps=-1, ent_coef=0.01,
        target_update_interval=1,
        # device="cpu"

        exploration_schedule = get_linear_fn(
            0.3,
            0.00,
            0.3,
        )
        ),
    )
    seed = 0
    total_timesteps = int(5e5)

    # Custom Gym env configs
@train_rl_ex.named_config
def peginhole_v2_ppo():
    normalize_reward=False
    # normalize = False  # Use VecNormalize
    common = dict(env_name="Gripper-v1",)

    # Custom Gym env configs
@train_rl_ex.named_config
def peginhole_v2():
    normalize_reward=False
    # normalize = False  # Use VecNormalize
    common = dict(env_name="Gripper-v1",
    max_episode_steps = 40,
    num_vec = 16  # number of environments in VecEnv)
    )
    rl = dict(
        rl_cls=sb3_contrib.SAC,
        # batch_size=4096,
        batch_size = 1024,  # batch size for RL algorithm
        rl_kwargs=dict(
        #     gamma=0.99,
        learning_rate=3e-4,
        tau=0.005, gamma=0.99, gradient_steps=-1, ent_coef=0.001,
        target_update_interval=1,
        # device="cpu"

        exploration_schedule = get_linear_fn(
            0.2,
            0.00,
            0.2,
        )
        ),
    )
    seed = 0
    total_timesteps = int(5e5)

@train_rl_ex.named_config
def peginhole_v1_imit():
    normalize_reward=False
    # normalize = False  # Use VecNormalize
    common = dict(env_name="GripperPegInHole2DPyBulletEnv-v1",
    max_episode_steps = 100,
    num_vec = 8  # number of environments in VecEnv)
    )
    rl = dict(
        rl_cls=sb3_contrib.SAC,
        # batch_size=4096,
        batch_size = 1024,  # batch size for RL algorithm
        rl_kwargs=dict(
        #     gamma=0.99,
        learning_rate=3e-4,
        tau=0.005,gamma=0.99, gradient_steps=-1, ent_coef=0.1,
        target_update_interval=1,
        # device="cpu"

        exploration_schedule = get_linear_fn(
            0.0,
            0.0,
            0.1,
        )
        ),
    )
    seed = 0
    total_timesteps = int(2e5)
    # If specified, overrides the ground-truth environment reward
    reward_type = "RewardNet_shaped"  # override reward type
    reward_path = "/root/imitation/jjh_data/expert_models/peginhole_v1_imit/reward_train.pt"  # override reward path
    # load_reward_kwargs = {"normalize_output_layer": networks.RunningNorm,}
    # "normalize_output_layer": networks.RunningNorm}
    load_reward_kwargs = {"normalize_output_layer": None, 
                "net_kwargs": {"normalize_input_layer": None} }
# Standard Gym env configs

@train_rl_ex.named_config
def peginhole_v1_imit2():
    normalize_reward=False
    # normalize = False  # Use VecNormalize
    common = dict(env_name="GripperPegInHole2DPyBulletEnv-v1",
    max_episode_steps = 100,
    num_vec = 8  # number of environments in VecEnv)
    )
    rl = dict(
        rl_cls=sb3_contrib.SAC,
        # batch_size=4096,
        batch_size = 1024,  # batch size for RL algorithm
        rl_kwargs=dict(
        #     gamma=0.99,
        learning_rate=3e-4,
        tau=0.05, gamma=0.99, gradient_steps=-1, ent_coef=0.01,
        target_update_interval=1,
        # device="cpu"

        exploration_schedule = get_linear_fn(
            0.0,
            0.0,
            0.1,
        )
        ),
    )
    seed = 0
    total_timesteps = int(2e5)
    # If specified, overrides the ground-truth environment reward
    reward_type = "RewardNet_normalized"  # override reward type
    reward_path = "/root/imitation/jjh_data/expert_models/peginhole_v1_airl_norm/reward_train.pt"  # override reward path
    # load_reward_kwargs = {"normalize_output_layer": networks.RunningNorm,}
    # "normalize_output_layer": networks.RunningNorm}
    load_reward_kwargs = {"normalize_input_layer": None, "normalize_output_layer": None} 
# Standard Gym env configs


@train_rl_ex.named_config
def acrobot():
    common = dict(env_name="Acrobot-v1")


@train_rl_ex.named_config
def ant():
    common = dict(env_name="Ant-v2")
    rl = dict(batch_size=16384)
    total_timesteps = int(5e6)


@train_rl_ex.named_config
def cartpole():
    common = dict(env_name="CartPole-v1")
    total_timesteps = int(1e5)


@train_rl_ex.named_config
def seals_cartpole():
    common = dict(env_name="seals/CartPole-v0")
    total_timesteps = int(1e6)


@train_rl_ex.named_config
def half_cheetah():
    common = dict(env_name="HalfCheetah-v3")
    total_timesteps = int(5e6)  # does OK after 1e6, but continues improving


@train_rl_ex.named_config
def seals_hopper():
    common = dict(env_name="seals/Hopper-v0")


@train_rl_ex.named_config
def seals_humanoid():
    common = dict(env_name="seals/Humanoid-v0")
    rl = dict(batch_size=16384)
    total_timesteps = int(10e6)  # fairly discontinuous, needs at least 5e6


@train_rl_ex.named_config
def mountain_car():
    common = dict(env_name="MountainCar-v0")


@train_rl_ex.named_config
def seals_mountain_car():
    common = dict(env_name="seals/MountainCar-v0")


@train_rl_ex.named_config
def serving():
    common = dict(env_name="Serving-v0",
    max_episode_steps = 100,
    num_vec = 32  # number of environments in VecEnv)
    )
    rl = dict(
        # batch_size=4096,
        rl_kwargs=dict(
            gamma=0.99,
            ent_coef=0.01,
            learning_rate=3e-4,
            target_kl=0.2,
        ),
    )
    total_timesteps = int(2.0e6)


@train_rl_ex.named_config
def pendulum():
    common = dict(env_name="Pendulum-v1")
    rl = dict(
        batch_size=4096,
        rl_kwargs=dict(
            gamma=0.9,
            learning_rate=1e-3,
        ),
    )
    total_timesteps = int(2e5)


@train_rl_ex.named_config
def reacher():
    common = dict(env_name="Reacher-v2")


@train_rl_ex.named_config
def seals_ant():
    common = dict(env_name="seals/Ant-v0")


@train_rl_ex.named_config
def seals_swimmer():
    common = dict(env_name="seals/Swimmer-v0")


@train_rl_ex.named_config
def seals_walker():
    common = dict(env_name="seals/Walker2d-v0")


# Debug configs


@train_rl_ex.named_config
def fast():
    # Intended for testing purposes: small # of updates, ends quickly.
    total_timesteps = int(4)
    policy_save_interval = 2
