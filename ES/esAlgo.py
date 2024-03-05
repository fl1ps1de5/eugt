import numpy as np

def mutate(policy, sigma=0.1):
    # Apply Gaussian noise to the policy parameters
    return policy + np.random.normal(0, sigma, policy.shape)

def evaluate_policy(env, policy_params):
    total_reward = 0
    env.reset_idx(torch.arange(env.num_envs, device=env.rl_device))  # Reset the environment
    done = False
    while not done:
        obs = env.compute_observations()  # Get observations
        action = policy(obs, policy_params)  # Determine action based on current policy
        env.pre_physics_step(action)  # Apply action
        env.gym.simulate(env.sim)  # Simulate one step
        env.gym.fetch_results(env.sim, True)  # Fetch results
        env.post_physics_step()  # Update environment state
        reward, reset = env.compute_reward()  # Compute reward
        total_reward += reward.sum().item()  # Aggregate reward
        done = reset.any().item()  # Check if done
    return total_reward / env.num_envs  # Average reward across environments

def es_algorithm(env, policy_model, population_size=50, generations=100, sigma=0.1):
    # Initialize your population with copies of the policy_model's parameters
    population = [policy_model.parameters() + np.random.normal(0, sigma, policy_model.parameters().shape) for _ in range(population_size)]
    for generation in range(generations):
        fitness_scores = [evaluate_policy(env, policy) for policy in population]
        # Selection and mutation logic here
        # ...
    return select_best_policy(population)