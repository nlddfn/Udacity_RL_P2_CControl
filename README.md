[//]: # (Image References)

[image1]: https://user-images.githubusercontent.com/10624937/43851024-320ba930-9aff-11e8-8493-ee547c6af349.gif "Trained Agent"


# Train a Unity Environment (Reacher) using Deep Deterministic Policy Gradient

## Introduction

For this project, 20 double-jointed arms are trained to move to target locations.

![Trained Agent][image1]

A reward of +0.1 is provided for each step if the agent's hand is on the target location. Thus, the agent goal is to maintain its position at the target location for as many time steps as possible. The environment is considered solved when the trained agents achieves an average score of +30 over 100 consecutive episodes (where the average is over all 20 agents).

The observation space consists of 33 variables corresponding to position, rotation, velocity, and angular velocities of the arm. Each action is a vector with four numbers, corresponding to torque applicable to two joints. Every entry in the action vector should be a number between -1 and 1. More details can be found [here](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Learning-Environment-Examples.md#reacher).

### Distributed Training

Each agent acts independently on the environment. This concept is useful for algorithms like [PPO](https://arxiv.org/pdf/1707.06347.pdf), [A3C](https://arxiv.org/pdf/1602.01783.pdf), and [D4PG](https://openreview.net/pdf?id=SyZipzbCb) that use multiple (non-interacting, parallel) copies of the same agent to distribute the task of gathering experience.

### Getting Started

1. Download the environment from one of the links below.  You need only select the environment that matches your operating system:

     - Linux: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/Reacher_Linux.zip)
     - Mac OSX: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/Reacher.app.zip)
     - Windows (32-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/Reacher_Windows_x86.zip)
     - Windows (64-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/Reacher_Windows_x86_64.zip)
    
    (_For Windows users_) Check out [this link](https://support.microsoft.com/en-us/help/827218/how-to-determine-whether-a-computer-is-running-a-32-bit-version-or-64) if you need help with determining if your computer is running a 32-bit version or 64-bit version of the Windows operating system.

    (_For AWS_) If you'd like to train the agent on AWS (and have not [enabled a virtual screen](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Training-on-Amazon-Web-Service.md)), then please use [this link](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/Reacher_Linux_NoVis.zip) to obtain the "headless" version of the environment.  You will **not** be able to watch the agent without enabling a virtual screen, but you will be able to train the agent.  (_To watch the agent, you should follow the instructions to [enable a virtual screen](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Training-on-Amazon-Web-Service.md), and then download the environment for the **Linux** operating system above._)

2. Clone the `Udacity_RL_P2_CControl` GitHub repository, place the file in the folder and decompress it.

3. Create a virtual environment and install the required libraries. For OSX users, you can use the MakeFile included in the repo. The option `make all` will create a new venv called `Udacity_RL_P2` and install the relevant dependencies to execute the notebook.

4. Activate the virtual environment using `source ./Udacity_RL_P2/bin/activate`

5. Type `jupyter lab` and select `Udacity_RL_P2` kernel.

## Train and execute the model

Within the virtual environment you can train and evaluate the model using `python main.py`. If you only want to evaluate the solved model included in the repo, set `TRAIN = False` in `main.py` and then run the script.

You can also use the notebook `Continuous_Control.ipynb` to (re)train and evaluate the model. Set the flag train to `TRUE` to retrain the model. Further details can be found [here](Report.md)