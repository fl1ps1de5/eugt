import isaacgym
import isaacgymenvs
import torch

num_envs = 1

envs = isaacgymenvs.make(
	seed=0, 
	task="Cartpole", 
	num_envs=num_envs, 
	sim_device="cuda:0",
	rl_device="cuda:0",
)
print("Observation space is", envs.observation_space)
print("Action space is", envs.action_space)
obs = envs.reset()
print(obs)
for _ in range(20):
	random_actions = 2.0 * torch.rand((num_envs,) + envs.action_space.shape, device = 'cuda:0') - 1.0
	n, r, d, _ = envs.step(random_actions)
	print(d)