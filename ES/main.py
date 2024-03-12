from isaacgym import gymapi
from isaacgym import gymutil
from isaacgym import gymtorch

import sys

import numpy as np
import torch

from policies import simpleMLP

def gen_pop(num_params, population_size):
    """Generate random population"""
    base_params = torch.randn(num_params)
    population = torch.empty((population_size, num_params))
    for i in range(population_size):
        # adding noise to generate rest of population (0.1 = sigma)
        population[i] = base_params + torch.randn(num_params) * 0.1
    return population

gym = gymapi.acquire_gym()

# parse arguments
args = gymutil.parse_arguments()

# configure sim
sim_params = gymapi.SimParams()
sim_params.up_axis = gymapi.UP_AXIS_Z
sim_params.gravity = gymapi.Vec3(0.0, 0.0, -9.81)
if args.physics_engine == gymapi.SIM_PHYSX:
    sim_params.substeps = 1 
    sim_params.physx.solver_type = 1
    sim_params.physx.num_position_iterations = 4
    sim_params.physx.num_velocity_iterations = 1
    sim_params.physx.num_threads = args.num_threads
    sim_params.physx.use_gpu = args.use_gpu
elif args.physics_engine == gymapi.SIM_FLEX and not args.use_gpu_pipeline:
    sim_params.flex.shape_collision_margin = 0.25
    sim_params.flex.num_outer_iterations = 4
    sim_params.flex.num_inner_iterations = 10
else:
    raise Exception("GPU pipeline is only available with PhysX")

sim_params.use_gpu_pipeline = args.use_gpu_pipeline
device = args.sim_device if args.use_gpu_pipeline else 'cpu'

sim = gym.create_sim(args.compute_device_id, args.graphics_device_id, args.physics_engine, sim_params)
if sim is None:
    raise Exception("Failed to create sim")
    
# add ground plane
plane_params = gymapi.PlaneParams()
plane_params.normal = gymapi.Vec3(0, 0, 1)
gym.add_ground(sim, plane_params)

# create viewer
viewer = gym.create_viewer(sim, gymapi.CameraProperties())
if viewer is None:
    raise Exception("Failed to create viewer")

# load ball asset
asset_root = "../assets"
asset_file = "urdf/cartpole.urdf"

asset_options = gymapi.AssetOptions()
asset_options.fix_base_link = True

asset = gym.load_asset(sim, asset_root, asset_file, asset_options)    


num_DOFs = gym.get_asset_dof_count(asset)
print('dof_count', num_DOFs)

pose = gymapi.Transform()
pose.p.z = 2.0
pose.r = gymapi.Quat(0.0, 0.0, 0.0, 1.0)

# set up the env grid
num_envs = 20
num_per_row = int(np.sqrt(num_envs))
env_spacing = 2.0
env_lower = gymapi.Vec3(0.5 * -env_spacing, -env_spacing, 0.0)
env_upper = gymapi.Vec3(0.5 * env_spacing, env_spacing, env_spacing)
# set random seed
np.random.seed(17)

envs = []
handles = []

for i in range(num_envs):
    # create env
    env = gym.create_env(sim, env_lower, env_upper, num_per_row)
    envs.append(env)

    # create handle    
    ahandle = gym.create_actor(env, asset, pose, "actor", i, 1)
    
    dof_props = gym.get_actor_dof_properties(env, ahandle)
    dof_props['driveMode'][0] = gymapi.DOF_MODE_EFFORT
    dof_props['driveMode'][1] = gymapi.DOF_MODE_NONE
    dof_props['stiffness'][:] = 0.0
    dof_props['damping'][:] = 0.0
    gym.set_actor_dof_properties(env, ahandle, dof_props)

    handles.append(ahandle)

gym.viewer_camera_look_at(viewer, None, gymapi.Vec3(20, 20, 5), gymapi.Vec3(0, 0, 1))

gym.prepare_sim(sim)

step = 0

_dof_states = gym.acquire_dof_state_tensor(sim)
dof_states = gymtorch.wrap_tensor(_dof_states)

# loop de doop
while not gym.query_viewer_has_closed(viewer):
    
    num_dofs = gym.get_sim_dof_count(sim)
    
    actions =  torch.rand(num_dofs, dtype=torch.float32, device="cuda:0")

    scaled_actions = actions * 2 - 1

    # must unwrap before we feed it into the next function
    forces = gymtorch.unwrap_tensor(scaled_actions)
    
    # step the physics
    gym.simulate(sim)
    gym.fetch_results(sim, True)
    
    step += 1

    print(actions)
    print()
    
    # if step % 100 == 0:
    gym.set_dof_actuation_force_tensor(sim, forces)

    # update the viewer
    gym.step_graphics(sim)
    gym.draw_viewer(viewer, sim, True)

    # Wait for dt to elapse in real time.
    # This synchronizes the physics simulation with the rendering rate.
    gym.sync_frame_time(sim)

gym.destroy_viewer(viewer)
gym.destroy_sim(sim)


