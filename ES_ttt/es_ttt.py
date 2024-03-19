import numpy as np
import torch
import torch.multiprocessing as mp
from torch import optim
import time
import scipy.stats as ss

from envs import TicTacToeEnv
from model import simpleMLP

NOISE_STD = 0.05
POPULATIONSIZE = 100
LEARNINGRATE = 0.01
MAXITERATIONS = 100

WORKERS = 20

def generate_noise(neural_net):
    nn_noise = []
    for n in neural_net.parameters():
        noise = np.random.normal(size=n.data.numpy().shape)
        nn_noise.append(noise)
    return np.array(nn_noise)


def evaluate_NN(nn, env):
    obs = env.reset(1)
    game_reward = 0

    while True:
        # neural net output
        net_output = nn(torch.tensor(obs))
        # get action with max liklihood
        action = net_output.data.cpu().numpy().argmax()
        new_obs, reward, done = env.step(action)
        obs = new_obs

        game_reward += reward

        if done:
            break
    
    return game_reward

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

    env = TicTacToeEnv()
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

    env = TicTacToeEnv()

    actor = simpleMLP(env.observation_space, env.action_space)
    optimizer = optim.Adam(actor.parameters(), lr = LEARNINGRATE)

    # create queues so I can pass variables between processes
    output_queue = mp.Queue(maxsize=POPULATIONSIZE)
    param_queue = mp.Queue(maxsize=POPULATIONSIZE)

    processes = []

    for _ in range(WORKERS):
        p = mp.Process(target=worker, args=(param_queue, output_queue))
        p.start()
        processes.append(p)

    for iteration in range(MAXITERATIONS):
        it_time = time.time()

        batch_noise = []
        batch_reward = []

        for _ in range(POPULATIONSIZE):
            param_queue.put(actor.state_dict())


        # get results from each worker
        for _ in range(POPULATIONSIZE):
            
            r, s = output_queue.get()

            np.random.seed(s)
            noise = generate_noise(actor)
            batch_noise.append(noise)

            batch_reward.append(r)
        

        avg_reward = np.round(np.mean(batch_reward), 2)
        print(iteration, 'Mean:',avg_reward, 'Max:', np.round(np.max(batch_reward), 2), 'Time:', np.round(time.time()-it_time, 2)) 

        batch_reward = normalized_rank(batch_reward)

        th_update = []
        optimizer.zero_grad()
        # for each actor's parameter, and for each noise in the batch, update it by the reward * the noise value
        for idx, p in enumerate(actor.parameters()):
            upd_weights = np.zeros(p.data.shape)

            for n,r in zip(batch_noise, batch_reward):
                upd_weights += r*n[idx]

            upd_weights = upd_weights / (POPULATIONSIZE*NOISE_STD)
            # put the updated weight on the gradient variable so that afterwards the optimizer will use it
            p.grad = torch.FloatTensor( -upd_weights)
            th_update.append(np.mean(upd_weights))

        # Optimize the actor's NN
        optimizer.step()

    # quit processes

    for _ in range(WORKERS):
        param_queue.put(None)

    for p in processes:
        p.join()

if __name__ == '__main__':
    main()