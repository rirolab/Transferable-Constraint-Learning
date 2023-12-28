"""Configuration settings for eval_policy, evaluating pre-trained policies."""

import sacred

from imitation.scripts.common import common

eval_policy_ex = sacred.Experiment(
    "eval_policy",
    ingredients=[common.common_ingredient],
)


@eval_policy_ex.config
def replay_defaults():
    eval_n_timesteps = int(1e4)  # Min timesteps to evaluate, optional.
    eval_n_episodes = None  # Num episodes to evaluate, optional.

    videos = False  # save video files
    video_kwargs = {}  # arguments to VideoWrapper
    render = False  # render to screen
    render_fps = 60  # -1 to render at full speed

    policy_type = None  # class to load policy, see imitation.policies.loader
    policy_path = None  # path to serialized policy

    reward_type = None  # Optional: override with reward of this type
    reward_path = None  # Path of serialized reward to load

    rollout_save_path = None  # where to save rollouts to -- if None, do not save


@eval_policy_ex.named_config
def render():
    common = dict(num_vec=1, parallel=False)
    render = True


@eval_policy_ex.named_config
def acrobot():
    common = dict(env_name="Acrobot-v1")


@eval_policy_ex.named_config
def ant():
    common = dict(env_name="Ant-v2")


@eval_policy_ex.named_config
def cartpole():
    common = dict(env_name="CartPole-v1")


@eval_policy_ex.named_config
def seals_cartpole():
    common = dict(env_name="seals/CartPole-v0")


@eval_policy_ex.named_config
def half_cheetah():
    common = dict(env_name="HalfCheetah-v2")


@eval_policy_ex.named_config
def seals_hopper():
    common = dict(env_name="seals/Hopper-v0")


@eval_policy_ex.named_config
def seals_humanoid():
    common = dict(env_name="seals/Humanoid-v0")


@eval_policy_ex.named_config
def mountain_car():
    common = dict(env_name="MountainCar-v0")


@eval_policy_ex.named_config
def seals_mountain_car():
    common = dict(env_name="seals/MountainCar-v0")


@eval_policy_ex.named_config
def pendulum():
    common = dict(env_name="Pendulum-v1")


@eval_policy_ex.named_config
def reacher():
    common = dict(env_name="Reacher-v2")


@eval_policy_ex.named_config
def seals_ant():
    common = dict(env_name="seals/Ant-v0")


@eval_policy_ex.named_config
def seals_swimmer():
    common = dict(env_name="seals/Swimmer-v0")


@eval_policy_ex.named_config
def seals_walker():
    common = dict(env_name="seals/Walker2d-v0")


@eval_policy_ex.named_config
def fast():
    common = dict(env_name="CartPole-v1", num_vec=1, parallel=False)
    render = True
    policy_path = "tests/testdata/expert_models/cartpole_0/policies/final/"
    eval_n_timesteps = 1
    eval_n_episodes = None

@eval_policy_ex.named_config
def gripper_v1():
    # normalize_reward=False
    # normalize = False  # Use VecNormalize
    common = dict(env_name="Gripper-v1",
    max_episode_steps = 40,
    num_vec = 16  # number of environments in VecEnv)
    )
    policy_type = "sac"
@eval_policy_ex.named_config
def peginhole_v1():
    # normalize_reward=False
    # normalize = False  # Use VecNormalize
    common = dict(env_name="Gripper-v0",
    max_episode_steps = 40,
    num_vec = 16  # number of environments in VecEnv)
    )
    policy_type = "sac"


@eval_policy_ex.named_config
def serving_oneway():
    eval_n_timesteps = int(200e2)  # Min timesteps to evaluate, optional.
    common = dict(env_name="Serving-v3",
    max_episode_steps = 100,
    num_vec = 8  # number of environments in VecEnv)
    )
   
@eval_policy_ex.named_config
def serving():
    eval_n_timesteps = int(100e2)  # Min timesteps to evaluate, optional.
    common = dict(env_name="Serving-v0",
    max_episode_steps = 100,
    num_vec = 8  # number of environments in VecEnv)
    )
 
  
@eval_policy_ex.named_config
def twodconst_v0():
    eval_n_timesteps = int(10*100)  # Min timesteps to evaluate, optional.
    common = dict(env_name="TwoDConstraint-v0",
    max_episode_steps = 100,
    num_vec = 32 # number of environments in VecEnv)
    )


@eval_policy_ex.named_config
def twodconst_real():
    eval_n_timesteps = int(40*30)  # Min timesteps to evaluate, optional.
    common = dict(env_name="TwoDConstraintReal-v0",
    max_episode_steps = 40,
    num_vec = 32 # number of environments in VecEnv)
    )

@eval_policy_ex.named_config
def twodconst_v2():
    eval_n_timesteps = int(30*50)  # Min timesteps to evaluate, optional.
    common = dict(env_name="TwoDConstraint-v2",
    max_episode_steps = 30,
    num_vec = 32 # number of environments in VecEnv)
    )

@eval_policy_ex.named_config
def twodconst_v1():
    eval_n_timesteps = int(10*200)  # Min timesteps to evaluate, optional.
    common = dict(env_name="TwoDConstraint-v1",
    max_episode_steps = 200,
    num_vec = 32 # number of environments in VecEnv)
    )

@eval_policy_ex.named_config
def twodcontact():
    eval_n_timesteps = int(20*100)  # Min timesteps to evaluate, optional.
    common = dict(env_name="TwoDContact-v0",
    max_episode_steps = 100,
    num_vec = 32 # number of environments in VecEnv)
    )
    
@eval_policy_ex.named_config
def twodmaze():
    eval_n_timesteps = int(1000*20)  # Min timesteps to evaluate, optional.
    common = dict(env_name="TwoDMaze-v0",
    max_episode_steps = 20,
    num_vec = 32 # number of environments in VecEnv)
    )

@eval_policy_ex.named_config
def cartpole_const():
    common = dict(env_name="CartPole-Const-v0",
    
    max_episode_steps = 100,
    num_vec = 4  # number of environments in VecEnv))
    )

    policy_type = "sac"

    # policy_type = "sac"