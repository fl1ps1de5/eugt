from isaacgym import gymapi, gymutil, gymtorch
import numpy as np
import torch

class cartpoleEnv:
    def __init__(self, gym, sim):
        self.gym = gym
        self.sim = sim
        self.env = self._create_env()
        self.viewer = self.gym.create_viewer(self.sim, gymapi.CameraProperties())
        if self.viewer is None:
            raise Exception("Failed to create viewer")
        self._setup_env()
        
    def _create_env(self):
        num_envs = 16 # can be changed 
        env_lower = gymapi.Vec3(0.5 * -env_spacing, -env_spacing, 0.0)
        env_upper = gymapi.Vec3(0.5 * env_spacing, env_spacing, env_spacing)
        
        asset_options = gymapi.AssetOptions()
        asset_options.fix_base_link = True
        cartpole_asset = self.gym.load_asset(self.sim, asset_root, asset_file, asset_options)

        asset_root = "../assets"
        asset_file = "urdf/cartpole.urdf"