import numpy as np
import torch
import torch.multiprocessing as mp
from torch import optim
import time
import scipy.stats as ss

import isaacgym
import isaacgymenvs

from model import simpleMLP


NOISE_STD = 0.05
POPULATIONSIZE = 100
LEARNINGRATE = 0.01
MAXITERATIONS = 100
WORKERS = 20

NUM_ENVS = 20

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def generate_noise(neural_net):
    nn_noise = []
    for n in neural_net.parameters():
        noise = np.random.normal(size=n.data.numpy().shape)
        nn_noise.append(noise)
    return np.array(nn_noise)


def evaluate_NN(nn, envs):
    obs = envs.reset()
    game_rewards = np.zeros(NUM_ENVS)

    while True:
        # convert observations to tensor
        obs_tensor = torch.tensor(obs, dtype=torch.float32).to(device)
        net_output = nn(obs_tensor)

def evaluate_noisy_NN(noise, nn, env):
    
    # save noise free params
    old_dict = nn.state_dict()

    # add noise to params
    for n, p in zip(noise, nn.parameters()):
        p.data += torch.FloatTensor(n * NOISE_STD)

    reward = evaluate_NN(nn, env)

    # load previous parameters (with no noise)
    nn.load_state_dict(old_dict)

    return reward

def worker(param_queue, output_queue):

    # env = TicTacToeEnv()
    actor = simpleMLP(env.observation_space, env.action_space)
    
    while True:
        actor_params = param_queue.get()
        if actor_params != None:

            # load actor params from queue
            actor.load_state_dict(actor_params)

            # get random seed
            seed = np.random.randint(1e5)
            
            # set new seed
            np.random.seed(seed)

            noise = generate_noise(actor)

            reward = evaluate_noisy_NN(noise, actor, env)

            output_queue.put([reward, seed])

        else:
            break

def normalized_rank(rewards):
    ranked = ss.rankdata(rewards)
    norm = (ranked - 1) / (len(ranked) - 1)
    norm -= 0.5
    return norm

def main():

    print()

if __name__ == '__main__':
    main()