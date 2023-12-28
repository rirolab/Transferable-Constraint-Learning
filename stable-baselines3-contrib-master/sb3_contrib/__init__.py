import os

from sb3_contrib.sac_custom import SAC
from sb3_contrib.ppo_custom import PPO
from sb3_contrib.tcl_ppo import TCL_PPO
from sb3_contrib.sac_lagrangian import SACLagrangian
from sb3_contrib.ppo_lagrangian import PPOLag
# Read version from file
version_file = os.path.join(os.path.dirname(__file__), "version.txt")
with open(version_file) as file_handler:
    __version__ = file_handler.read().strip()
