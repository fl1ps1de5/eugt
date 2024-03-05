from isaacgym import gymapi
import math
import numpy as np
import random

# clamp function
def clamp(x, min_value, max_value):
    return max(min(x, max_value), min_value)

# load gym, sim and ground and viewer
gym = gymapi.acquire_gym()

# default simParams
sim_params = gymapi.SimParams() 

sim_params.dt = dt = 1.0/60.0

sim_params.use_gpu_pipeline = False

sim = gym.create_sim(0, 0, gymapi.SIM_PHYSX, sim_params)

plane_params = gymapi.PlaneParams()

gym.add_ground(sim, plane_params)

# create viewer and set it up
viewer = gym.create_viewer(sim, gymapi.CameraProperties())
if viewer is None:
    raise Exception("Failed to create viewer")

# loading assets
asset_root = "./assets" 
asset_file = "urdf/cartpole.urdf"
#asset_file = "urdf/franka_description/robots/franka_panda.urdf" 

asset_options = gymapi.AssetOptions()

asset_options.fix_base_link = True
asset_options.default_dof_drive_mode = gymapi.DOF_MODE_POS

cartpole_asset = gym.load_asset(sim, asset_root, asset_file, asset_options)


# get asset descriptors
dof_names = gym.get_asset_dof_names(cartpole_asset) # array of DOF names

dof_props = gym.get_asset_dof_properties(cartpole_asset) # array of DOF properties

num_dofs = gym.get_asset_dof_count(cartpole_asset) 
dof_states = np.zeros(num_dofs, dtype=gymapi.DofState.dtype) # array of DOF states to update actors

dof_types = [gym.get_asset_dof_type(cartpole_asset, i) for i in range(num_dofs)] # get list of DOF types

# get position slices of DOF state array
dof_positions = dof_states['pos'] 

# get the limit-related slices of the DOF properties array
stiffnesses = dof_props['stiffness']
dampings = dof_props['damping']
armatures = dof_props['armature']
has_limits = dof_props['hasLimits']
lower_limits = dof_props['lower']
upper_limits = dof_props['upper']

# initialize default positions, limits, and speeds (make sure they are in reasonable ranges)
defaults = np.zeros(num_dofs)
speeds = np.zeros(num_dofs)

for i in range(num_dofs):
    if has_limits[i]:
        if dof_types[i] == gymapi.DOF_ROTATION: # for rotation type, the limits are basically a circle
            lower_limits[i] = clamp(lower_limits[i], -math.pi, math.pi)
            upper_limits[i] = clamp(lower_limits[i], -math.pi, math.pi)
        if lower_limits[i] > 0.0:
            defaults[i] = lower_limits[i]
        elif upper_limits[i] < 0.0:
            defaults[i] = upper_limits[i]
    else: # we do not have limits
        # we will pick reasonable animation limits for unlimited joints (so stuff doesnt look wacky)
        if dof_types[i] == gymapi.DOF_ROTATION:
            lower_limits[i] = -math.pi
            upper_limits[i] = math.pi
        elif dof_types[i] == gymapi.DOF_TRANSLATION:
            lower_limits[i] = -1.0
            upper_limits[i] = 1.0

    dof_positions[i] = defaults[i] # set DOF position to default 
    
    if dof_types[i] == gymapi.DOF_ROTATION:
        speeds[i] = clamp(2 * (upper_limits[i] - lower_limits[i]), 0.25 * math.pi, 3.0 * math.pi)
    else:
        speeds[i] = clamp(2 * (upper_limits[i] - lower_limits[i]), 0.1, 7.0)
        

# set up the env grid
num_envs = 1
envs_per_row = 8
env_spacing = 2.0
env_lower = gymapi.Vec3(-env_spacing, 0.0, -env_spacing)
env_upper = gymapi.Vec3(env_spacing, env_spacing, env_spacing)

# cache some common handles for later use
envs = []
actor_handles = []

# create and populate the environments
for i in range(num_envs):
    env = gym.create_env(sim, env_lower, env_upper, envs_per_row)
    envs.append(env)
    
    # add an actor
    pose = gymapi.Transform()
    pose.p = gymapi.Vec3(0, 1.5, 0.0)
    pose.r = gymapi.Quat(-0.707107, 0.0, 0.0, 0.707107)

    actor_handle = gym.create_actor(env, cartpole_asset, pose, "actor", i, 1)
    actor_handles.append(actor_handle)
    
    # set default DOF positions
    gym.set_actor_dof_states(env, actor_handle, dof_states, gymapi.STATE_ALL)
   
   
  
# create some animiation states
ANIM_MOVE_LOWER = 1     # move joint toward upper limit
ANIM_MOVE_UPPER = 2     # move joint toward lower limit 
ANIM_MOVE_DEFAULT = 3   # returns joint to "default" position
ANIM_FINISH = 4    # inidcates animiation cycle is complete for current joint

anim_state = ANIM_MOVE_LOWER
current_dof = 0

while not gym.query_viewer_has_closed(viewer):

    # step the physics
    gym.simulate(sim)
    gym.fetch_results(sim, True)
    
    speed = speeds[current_dof]
    
#    #animate DOF's
#    if anim_state == ANIM_MOVE_LOWER:
#        dof_postitions[current_dof] -= speed * dt
#        
#        
    # animate the dofs
    if anim_state == ANIM_MOVE_LOWER:
        dof_positions[current_dof] -= speed * dt
        if dof_positions[current_dof] <= lower_limits[current_dof]:
            dof_positions[current_dof] = lower_limits[current_dof]
            anim_state = ANIM_MOVE_UPPER
    elif anim_state == ANIM_MOVE_UPPER:
        dof_positions[current_dof] += speed * dt
        if dof_positions[current_dof] >= upper_limits[current_dof]:
            dof_positions[current_dof] = upper_limits[current_dof]
            anim_state = ANIM_MOVE_DEFAULT
    if anim_state == ANIM_MOVE_DEFAULT:
        dof_positions[current_dof] -= speed * dt
        if dof_positions[current_dof] <= defaults[current_dof]:
            dof_positions[current_dof] = defaults[current_dof]
            anim_state = ANIM_FINISH
    elif anim_state == ANIM_FINISH:
        dof_positions[current_dof] = defaults[current_dof]
        current_dof = (current_dof + 1) % num_dofs
        anim_state = ANIM_MOVE_LOWER
        print("Animating DOF %d ('%s')" % (current_dof, dof_names[current_dof]))
        
    for i in range(num_envs):
        gym.set_actor_dof_states(envs[i], actor_handles[i], dof_states, gymapi.STATE_POS)

    # update the viewer
    gym.step_graphics(sim)
    gym.draw_viewer(viewer, sim, True)

    # Wait for dt to elapse in real time.
    # This synchronizes the physics simulation with the rendering rate.
    gym.sync_frame_time(sim)
    
gym.destroy_viewer(viewer)
gym.destroy_sim(sim)
    
