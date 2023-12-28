from imitation.rewards.reward_nets import BasicRewardNet, FixedRewardNet, PredefinedShapedRewardNet, BasicShapedRewardNet, NormalizedRewardNet, ScaledRewardNet, ShapedScaledRewardNet, PredefinedRewardNet, DropoutRewardNet
from imitation.util.networks import RunningNorm
from sb3_contrib import TCL_PPO 
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.ppo import MlpPolicy 
import gym
from imitation.data import types
import wandb
from imitation.util import logger as imit_logger
import os
import sys
from imitation.util import util
import torch as th
import time
import numpy as np

from imitation.algorithms.adversarial.tcl import TCL

import hydra
from hydra.utils import get_original_cwd, to_absolute_path
from omegaconf import DictConfig, OmegaConf
from util import save

from torch.nn import functional as F
from imitation.rewards import reward_nets
    
def load_rollouts(dir):
    with open(dir, 'rb') as f:
        rollouts = types.load(f)
    return rollouts

def make_env(env_id, rank, seed=0):
    def _init():
        env = gym.make(env_id)
        env.seed(seed + rank)
        env = Monitor(env)
        return env
    return _init

def reward_fn(s, a, ns, d):
    return  ns[...,[0]]
combined_size  = 1

def gt_net(s, a, ns, d):
    return  th.zeros_like(s[...,0])
@hydra.main(config_path="config", config_name="common")
def main(cfg: DictConfig):
    
    normalize_layer = {"None":None, "RunningNorm":RunningNorm}
    opt_cls = {"None":None, "Adam":th.optim.Adam, "AdamW": th.optim.AdamW}
    
    n_envs = int(cfg.n_envs)
    total_steps = int(cfg.total_steps)
    is_wandb = bool(cfg.is_wandb)
    device = cfg.device
    render = bool(cfg.render)
    
    env_id = cfg.env.env_id
    r_gamma = float(cfg.env.r_gamma)
    
    gen_lr = float(cfg.gen.lr)
    ent_coef = float(cfg.gen.ent_coef)
    target_kl = float(cfg.gen.target_kl)
    batch_size = int(cfg.gen.batch_size)
    n_epochs = int(cfg.gen.n_epochs)
    n_steps = int(cfg.gen.n_steps)

    reg_coeff = [float(coef) for coef in cfg.disc.reg_coef]

    rew_opt = cfg.disc.reward_net_opt
    primary_opt = cfg.disc.primary_net_opt
    constraint_opt = cfg.disc.constraint_net_opt

    
    disc_lr = float(cfg.disc.lr)
    demo_batch_size = int(cfg.disc.demo_batch_size)
    gen_replay_buffer_capacity = int(cfg.disc.gen_replay_buffer_capacity)
    n_disc_updates_per_round = int(cfg.disc.n_disc_updates_per_round)
    hid_size = int(cfg.disc.hid_size)
    normalize = cfg.disc.normalize
    rollouts = load_rollouts(os.path.join(to_absolute_path('.'), "../data/expert_models/","wallfollowing","rollout.pkl"))
    
    tensorboard_log = os.path.join(to_absolute_path('logs'), f"{cfg.gen.model}_{cfg.env.env_id}")

    log_format_strs = ["stdout"]
    if is_wandb:
        log_format_strs.append("wandb")
        
    log_dir = os.path.join(
                "output",
                sys.argv[0].split(".")[0],
                util.make_unique_timestamp(),
            )
    os.makedirs(log_dir, exist_ok=True)
    
    if cfg.comment == "None":
        comment = ""
    else:
        comment = f"_{str(cfg.comment)}"
    name = 'ird' + comment
    wandb.init(project='test_bench-real', sync_tensorboard=True, dir=log_dir, config=cfg, name=name)
    # if "wandb" in log_format_strs:
    #     wb.wandb_init(log_dir=log_dir)
    custom_logger = imit_logger.configure(
        folder=os.path.join(log_dir, "log"),
        format_strs=log_format_strs,
    )
    #venv = DummyVecEnv([lambda: gym.make("Gripper-v0")] * 4)
    venv = SubprocVecEnv( [make_env(env_id, i) for i in range(n_envs)])
    learner = TCL_PPO(
        env=venv,
        policy=MlpPolicy,
        batch_size=batch_size,
        ent_coef=ent_coef,
        learning_rate=gen_lr,
        target_kl=target_kl,
        n_epochs=n_epochs,
        n_steps=n_steps,
        policy_kwargs={'optimizer_class':th.optim.Adam},
        tensorboard_log='./logs/',
        device=device,
    )
    reward_net = BasicRewardNet(
        venv.observation_space, venv.action_space, normalize_input_layer=normalize_layer[normalize],#RunningNorm,
        hid_sizes=[hid_size, hid_size],
    )
    constraint_net = BasicShapedRewardNet(
        venv.observation_space, venv.action_space, normalize_input_layer=normalize_layer[normalize],#RunningNorm,
        reward_hid_sizes=[32,32],
        potential_hid_sizes=[64,64],
        use_state=True,
        use_action=False,
        use_next_state=False,
    )

    primary_net = BasicRewardNet(
        venv.observation_space, venv.action_space, normalize_input_layer=normalize_layer[normalize],#RunningNorm,
        hid_sizes=[hid_size, hid_size],
    )
    custom_net = PredefinedShapedRewardNet(
            venv.observation_space, venv.action_space, reward_fn=reward_fn, combined_size=combined_size, use_action=False, normalize_input_layer=normalize_layer[normalize], #RunningNorm, #RunningNorm,
        discount_factor=0.99,
        reward_hid_sizes=[32,32],
        potential_hid_sizes=[64,64],
    )
    
    custom_net = PredefinedShapedRewardNet(
            venv.observation_space, venv.action_space, reward_fn=reward_fn, combined_size=combined_size, use_action=False, normalize_input_layer=normalize_layer[normalize], #RunningNorm, #RunningNorm,
        discount_factor=0.99,
        reward_hid_sizes=[32,32],
        potential_hid_sizes=[64,64],
    )
     
    gt_nets = FixedRewardNet(
            venv.observation_space, venv.action_space, reward_fn=gt_net, combined_size=combined_size, use_action=True, #normalize_input_layer=normalize_layer[normalize], #RunningNorm, #RunningNorm,
    )
    tcl_trainer = TCL(
        demonstrations=rollouts,
        demo_batch_size=demo_batch_size,
        gen_replay_buffer_capacity=gen_replay_buffer_capacity,
        n_disc_updates_per_round=n_disc_updates_per_round,
        venv=venv,
        gen_algo=learner,
        reward_net=reward_net,
        disc_opt_kwargs={"lr":disc_lr},
        log_dir=log_dir,
        primary_net=primary_net,
        constraint_net=constraint_net,
        custom_net=custom_net,
        disc_opt_cls=opt_cls[rew_opt],
        custom_logger=custom_logger
    )

    eval_env = DummyVecEnv([lambda: gym.make(env_id)] * 1)
    test_env = gym.make("WallFollowing-v1")
    if render:
        eval_env.render(mode='human')
    
    test_env.reset()
    states_m = []
    gt_r = []
    gt_c = []
    cnt_x = 0
    for x in np.linspace(-0.5, 0.4, 40):
        cnt_x += 1
        cnt_y = 0
        
        for y in np.linspace(-0.5, 0.5, 40):
            cnt_y += 1

            s, r, d, info = test_env.set_states(car=(x, y))
            states_m.append(np.expand_dims(s , axis=0))
            gt_r.append(np.expand_dims(r , axis=0))
            gt_c.append(np.expand_dims(info['constraint'] , axis=0))
    states_m = np.concatenate(states_m, axis=0)
    gt_r = np.concatenate(gt_r, axis=0)
    gt_c = np.concatenate(gt_c, axis=0)
    observations = []
    next_observations = []
    actions = []
    done = []
    for rollout in rollouts:
        observations.append(rollout.obs[:-1, :])
        next_observations.append(rollout.obs[1:, :])
        actions.append(rollout.acts[:,:])
    observations = np.concatenate(observations, axis=0)
    next_observations = np.concatenate(next_observations, axis=0)
    actions = np.concatenate(actions, axis=0)
    dones = np.zeros(actions.shape[:-1])
    constraint_fn = lambda *args: -1*tcl_trainer.update_stats( 1.0 * tcl_trainer.constraint_test.predict_processed(*args), update_stats=False )
    
    checkpoint_interval=10
    def cb(round_num):
        if checkpoint_interval > 0 and round_num % checkpoint_interval == 0:
            save(tcl_trainer, os.path.join(log_dir, "checkpoints", f"{round_num:05d}"))
            obs = eval_env.reset()
            for i in range(300):
                action, _states = tcl_trainer.gen_algo.predict(obs, deterministic=False)
                obs, _, _, _= eval_env.step(action)
                if render:
                    eval_env.render(mode='human')
                    time.sleep(0.005)
            consts = constraint_fn(observations, actions, next_observations, dones)


    tcl_trainer.train(int(total_steps), callback=cb)  
    

if __name__ == '__main__':
    main()