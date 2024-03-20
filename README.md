# Ethan's Undergraduate Thesis (EUGT)

## INFO4001

This repo currently contains 3 directories:

1. VariousGymTesting

This dir contains files that represent my first exploration into the usage and functionality of isaac gym. 

* cartpole_tensor_DOF.py - explores how forces can be applied to a DOF using Isaac Gym Tensor API, adds random forces and then simulates the movement of a Cartpole

* cartpole_manual_DOF.py - explores how the user can manually apply forces to a DOF. Allows the user to use the keypad to move the Cartpole, and spacebar to change the DOF being controlled.

(the other two files are irrelevant and should be removed)

2. ES_gym

* Contains an implementation of the Evolution Strategies algorithm to train the gradients of a policy, which is solving the 'Cartpole-v1' task, from Open AI gym
* Contains a policy definition (model.py), and a main script which defines the ES functions + performs the training loop (es_gym.py). Also contains a script to produce a plot.
* This approach was the roots of implementing the same technique in Isaac Gym, which would alternatively involve multiple environments

3. ES_isaacgym

* Non-complete implementation of Evolution Strategies within the Isaac Gym framework
* Currently borrows most code from ES_gym, as the final modifications have not been pushed yet
