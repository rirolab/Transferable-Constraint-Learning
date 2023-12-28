from gym.envs.registration import registry, register, make, spec

register(
    id='WallFollowing-v1',
    entry_point='gym.envs.box2d:WallFollowingTest',
    max_episode_steps=80,
)
register(
    id='WallFollowing-v0',
    entry_point='gym.envs.box2d:WallFollowingDemo',
    max_episode_steps=40,
)

