
# coding: utf-8

# # Teach a smartcab to drive
# 
# 
# ## Setup
# 
# You need Python 2.7 and pygame for this project: https://www.pygame.org/wiki/GettingStarted
# For help with installation, it is best to reach out to the pygame community [help page, Google group, reddit].
# 
# 

# In[2]:

# Import what we need, and setup the basic function to run from later.

import math
import string
import sys
import os
import random

import numpy as np
import pandas as pd
from IPython.display import display # Allows the use of display() for DataFrames

# Show matplotlib plots inline (nicely formatted in the notebook)
get_ipython().magic(u'matplotlib inline')

sys.path.append("./smartcab/")
from environment import Agent, Environment
from planner import RoutePlanner
from simulator import Simulator

def run(agentType,trials=10, gui=False, deadline=False, delay=0):
    """Run the agent for a finite number of trials."""

    # Set up environment and agent
    e = Environment()  # create environment (also adds some dummy traffic)
    a = e.create_agent(agentType)  # create agent
    e.set_primary_agent(a, enforce_deadline=deadline)  # specify agent to track
    # NOTE: You can set enforce_deadline=False while debugging to allow longer trials

    # Now simulate it
    sim = Simulator(e, update_delay=delay, display=gui)  # create simulator (uses pygame when display=True, if available)
    # NOTE: To speed up simulation, reduce update_delay and/or set display=False

    sim.run(n_trials=trials)  # run for a specified number of trials
    # NOTE: To quit midway, press Esc or close pygame window, or hit Ctrl+C on the command-line
    print "Successfull runs = {}".format(a.goal)
    print "----------------------------------------------------------"
    features= []
    for i in range(len(a.features)):
        features.append(pd.DataFrame(a.features[i]).T)
        
    return features
    
print "Environment ready"


# ## Implement a basic driving agent
# 
# Implement the basic driving agent, which processes the following inputs at each time step:
# 
# Next waypoint location, relative to its current location and heading,
# Intersection state (traffic light and presence of cars), and,
# Current deadline value (time steps remaining),
# And produces some random move/action (None, 'forward', 'left', 'right'). Don’t try to implement the correct strategy! That’s exactly what your agent is supposed to learn.
# 
# Run this agent within the simulation environment with enforce_deadline set to False (see run function in agent.py), and observe how it performs. In this mode, the agent is given unlimited time to reach the destination. The current state, action taken by your agent and reward/penalty earned are shown in the simulator.
# 
# In your report, mention what you see in the agent’s behavior. Does it eventually make it to the target location?
# 

# In[3]:

class RandomAgent(Agent):
    """An agent that learns to drive in the smartcab world."""

    def __init__(self, env):
        super(RandomAgent, self).__init__(env)  # sets self.env = env, state = None, next_waypoint = None, and a default color
        self.color = 'red'  # override color
        self.planner = RoutePlanner(self.env, self)  # simple route planner to get next_waypoint
        # TODO: Initialize any additional variables here
        self.availableAction = [None, 'forward', 'left', 'right']   
        self.goal=0
        self.steps=0
        self.features=[]

    def reset(self, destination=None):
        self.planner.route_to(destination)
        # TODO: Prepare for a new trip; reset any variables here, if required
        #print"RESET, Final state:\n", self.state
        self.features.append({})
        self.steps=0
        try:
            if self.state['deadline']>0:
                print "PASS! {} steps to goal,Goal reached {} times out of {}!".format(self.state['deadline'],self.goal,len(self.features))
                self.goal+=1
            else:
                print "FAIL! {} steps to goal,Goal reached {} times out of {}!".format(self.state['deadline'],self.goal,len(self.features))
                pass
        except:
            print "Trial 0 - Goal reached {} times out of {}!".format(self.goal,len(self.features))
            pass
        print "----------------------------------------------------------"

    def update(self, t):
        # Gather inputs
        self.steps+=1
        self.next_waypoint = self.planner.next_waypoint()  # from route planner, also displayed by simulator
        inputs = self.env.sense(self)
        deadline = self.env.get_deadline(self)
        inputs['deadline']=deadline
        self.state=inputs
    
        # TODO: Select action according to your policy
        action = self.availableAction[random.randint(0,3)]    
        
        # Execute action and get reward
        reward = self.env.act(self, action)
        self.lastReward=reward
        # TODO: Learn policy based on state, action, reward
        
        #print "LearningAgent.update():deadline{}, inputs{}, action = {}, reward = {}, next_waypoint = {}".format(
        #                                            deadline, inputs, action, reward,self.next_waypoint, )  # [debug]
print "RandomAgent ready"


# In[4]:

out=run(agentType=RandomAgent,trials=2, deadline=False) #Example of a random run, with no deadline 


# In[5]:

out=run(agentType=RandomAgent,trials=2, deadline=True) #Example of a random run


# ### Random Agent Answer:
# 
# When we run an agent with a random action policy, we see that it will move about the board with no pattern, and will eventually reach the destination. If we allow the use of deadlines, we see that the agent rarely reaches the destination in time, although it may still occur.

# 
# ## Identify and update state
# 
# Identify a set of states that you think are appropriate for modeling the driving agent. The main source of state variables are current inputs, but not all of them may be worth representing. Also, you can choose to explicitly define states, or use some combination (vector) of inputs as an implicit state.
# 
# At each time step, process the inputs and update the current state. Run it again (and as often as you need) to observe how the reported state changes through the run.

# In[6]:

class StateAgent(RandomAgent):
    """An agent that learns to drive in the smartcab world."""

    def __init__(self, env):
        super(StateAgent, self).__init__(env)  # sets self.env = env, state = None, next_waypoint = None, and a default color
        self.color = 'red'  # override color
        self.planner = RoutePlanner(self.env, self)  # simple route planner to get next_waypoint
        # TODO: Initialize any additional variables here
        self.availableAction = [None, 'forward', 'left', 'right']   
        self.next_waypoint   = None
        self.goal=0
        self.steps=0
        self.features=[]
        
    def update(self, t):
        # Gather inputs
        self.steps+=1
        
        self.lastWaypoint = self.next_waypoint
        self.next_waypoint = self.planner.next_waypoint()  # from route planner, also displayed by simulator
        inputs = self.env.sense(self)
        
        deadline = self.env.get_deadline(self)
        
        # TODO: Update state
        inputs['deadline']=deadline
        inputs['next_waypoint']=self.next_waypoint
        self.state= inputs    
        self.features[len(self.features)-1][self.steps]=self.state
        # TODO: Select action according to your policy

        action = self.availableAction[random.randint(0,3)]    
    
        # Execute action and get reward
        reward = self.env.act(self, action)
        # TODO: Learn policy based on state, action, reward

        #print "LearningAgent.update(): self.state{}, action = {}, reward = {}, next_waypoint = {}".format(
        #                                        self.state, action, reward,self.next_waypoint, )  # [debug]
print "StateAgent Ready"


# In[18]:

stateFeatures=run(agentType=StateAgent,trials=5)

#stateFeatures.next_waypoint

import matplotlib.pyplot as plt

for f in stateFeatures:
    print "features:{}\n".format(len(f))
    #print f.head(5)
    fig, axes = plt.subplots(nrows=2, ncols=3,figsize=(14,6))

    try:
        pd.value_counts(f.left.ravel()).plot(kind='bar', title="Left",ax=axes[0,0])
    except:
        pass
    try:
        pd.value_counts(f.light.ravel()).plot(kind='bar', title="Light",ax=axes[0,1])
    except:
        pass
    pd.value_counts(f.next_waypoint.ravel()).plot(kind='bar', title="next_waypoint",ax=axes[0,2])
    try:
        pd.value_counts(f.oncoming.ravel()).plot(kind='bar', title="oncoming",ax=axes[1,0])
    except:
        pass
    try:
        pd.value_counts(f.right.ravel()).plot(kind='bar', title="right",ax=axes[1,2])
    except:
        pass

    print f.deadline.ravel().min()

    fig.title= "test"
    fig.show()


# ## Implement Q-Learning
# 
# Implement the Q-Learning algorithm by initializing and updating a table/mapping of Q-values at each time step. Now, instead of randomly selecting an action, pick the best action available from the current state based on Q-values, and return that.
# 
# Each action generates a corresponding numeric reward or penalty (which may be zero). Your agent should take this into account when updating Q-values. Run it again, and observe the behavior.
# 
# What changes do you notice in the agent’s behavior?
# 
# 

# ## Enhance the driving agent
# 
# Apply the reinforcement learning techniques you have learnt, and tweak the parameters (e.g. learning rate, discount factor, action selection method, etc.), to improve the performance of your agent. Your goal is to get it to a point so that within 100 trials, the agent is able to learn a feasible policy - i.e. reach the destination within the allotted time, with net reward remaining positive.
# 
# Report what changes you made to your basic implementation of Q-Learning to achieve the final version of the agent. How well does it perform?
# 
# Does your agent get close to finding an optimal policy, i.e. reach the destination in the minimum possible time, and not incur any penalties?
# 
#  PREVIOUS

# In[7]:

if __name__ == '__main__':
    print  "running...."
    run(agentType=RandomAgent,trials=2, gui=True, delay=.3)


# #EOF
