"""Configuration settings for train_rl, training a policy with RL."""

import sacred

from imitation.scripts.common import common, rl, train
import stable_baselines3 as sb3


@train_ingredient.capture
def customized_sac(
    _seed,
    env_name: str,
    num_vec: int,
    parallel: bool,
    log_dir: str,
    max_episode_steps: int,
    env_make_kwargs: Mapping[str, Any],
    **kwargs,
) -> vec_env.VecEnv:
    """Builds the vector environment.

    Args:
        env_name: The environment to train in.
        num_vec: Number of `gym.Env` instances to combine into a vector environment.
        parallel: Whether to use "true" parallelism. If True, then use `SubProcVecEnv`.
            Otherwise, use `DummyVecEnv` which steps through environments serially.
        max_episode_steps: If not None, then a TimeLimit wrapper is applied to each
            environment to artificially limit the maximum number of timesteps in an
            episode.
        log_dir: Logs episode return statistics to a subdirectory 'monitor`.
        env_make_kwargs: The kwargs passed to `spec.make` of a gym environment.
        kwargs: Passed through to `util.make_vec_env`.

    Returns:
        The constructed vector environment.
    """
    return util.make_vec_env(
        env_name,
        num_vec,
        seed=_seed,
        parallel=parallel,
        max_episode_steps=max_episode_steps,
        log_dir=log_dir,
        env_make_kwargs=env_make_kwargs,
        **kwargs,
    )

# Standard Gym env configs
@train_rl_ex.named_config
def peginhole_v1():
    normalize_reward=False
    # normalize = False  # Use VecNormalize
    common = dict(env_name="GripperPegInHole2DPyBulletEnv-v1",
    max_episode_steps = 100,
    num_vec = 8  # number of environments in VecEnv)
    )
    rl = dict(
        rl_cls=sb3.SAC,
        # batch_size=4096,
        batch_size = 1024,  # batch size for RL algorithm
        rl_kwargs=dict(
        #     gamma=0.99,
        learning_rate=1e-3,
        tau=0.05, gamma=0.99, gradient_steps=-1, ent_coef=0.01,
        target_update_interval=1,
        device="cpu",
        ),
    )
    seed = 0
    total_timesteps = int(2e5)

