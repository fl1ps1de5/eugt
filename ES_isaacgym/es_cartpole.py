import numpy as np
import time
import scipy.stats as ss

import isaacgym
import isaacgymenvs

import torch
import torch.multiprocessing as mp
from torch import optim

from model import simpleMLP


NOISE_STD = 0.05
POPULATIONSIZE = 100
LEARNINGRATE = 0.01
MAXITERATIONS = 100
WORKERS = 20

NUM_ENVS = 500

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def generate_noise(neural_net):
    nn_noise = []
    for n in neural_net.parameters():
        noise = np.random.normal(size=n.data.numpy().shape)
        nn_noise.append(noise)
    return np.array(nn_noise)


def evaluate_NN(nn, envs):
    obs = envs.reset()

    game_rewards = torch.zeros(NUM_ENVS).to(device)

    while True:
        # convert observations to tensor
        obs_tensor = obs['obs'].to(device)
        net_output = nn(obs_tensor)

        actions = net_output.argmax(axis=1)

        new_obs, rewards, dones, _ = envs.step(actions)
        
        game_rewards += rewards  # Add rewards for each environment

        print(dones)

        if dones.all():
            break 
        
        obs = new_obs
    
    return game_rewards.mean()




env_config = {
    "seed": 0,
    "task": "Cartpole",
    "num_envs": NUM_ENVS,  
    "sim_device": "cuda:0",
    "rl_device": "cuda:0",
}

envs = isaacgymenvs.make(**env_config)

actor = simpleMLP(envs.observation_space.shape[0], envs.action_space.shape[0]).to(device)

ev = evaluate_NN(actor, envs)



