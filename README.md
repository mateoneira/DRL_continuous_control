# Continuous Control Using Deep Reinforcement Learning

Implementation of DDPG algorithm to train an agent to solve Unity's [Reacher environment](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Learning-Environment-Examples.md). The implementation uses a 20 agent environment, the environment is considered solved if the mean reward across the 20 agents is >30 for 100 consecutive episodes. The implementation of DDPG solved the environment in 120 episodes. 

__Project contains__:

* __report.pdf__: model architecture details and training choices made to solve the environment.
* __continuous_control.ipynd__: jupyter notebook for training the agent, and watching the agent interact with the environment with pre-trained weights. 
* __agent__: folder containing the implementation of DDPG in an agent class, and actor-critic network classes.
* __actor.pth__: actor network pretrained weights.
* __critic.pth__: critic network pretrained weights.

### Project Details

* __Goal__: Agent must move a double-jointed arm to move and keep its hand at moving target's location. 
* __Agent Reward Function__: +0.1 each step an (individual) agent is at its target location.
* __Behaviour Parameters__:
    - Observation space: vector of 33 continuous variables corresponding to position, location, velocity, and angular velocities of the two arm rigid bodies.
    - Action space: vector of 4 continuous variables corresponding to torque applied to two joints. 

### Getting Started

* __Dependencies__: Project uses [PyTorch](https://pytorch.org/) and [Unity ML-Agents Toolkit](https://github.com/Unity-Technologies/ml-agents). To set up your python environment follow instruction on following link:
https://github.com/udacity/deep-reinforcement-learning

* __Environment__: This project uses a 20 agent environent. Download the environment from one of the links below. You need only select the environment that matches your operating system:
    - Linux: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/Reacher_Linux.zip)
    - Mac OSX: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/Reacher.app.zip)
    - Windows (32-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/Reacher_Windows_x86.zip)
    - Windows (64-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/Reacher_Windows_x86_64.zip)  

### Instructions

To train the agent yourself, run __continuous_control.ipynb__. In the same notebook you can also load the pre-trained weights and watch the agents solve the task. 

