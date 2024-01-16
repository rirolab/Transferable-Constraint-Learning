try:
    import Box2D
    from gym.envs.box2d.wallfollowing_demo import WallFollowingDemo
    from gym.envs.box2d.wallfollowing_test import WallFollowingTest
except ImportError:
    Box2D = None
