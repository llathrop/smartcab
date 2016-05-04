
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

# In[156]:

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

# In[210]:

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

        print len(self.features), self.steps
        try:
            if self.features[len(self.features)-1][self.steps] >0: #deadline less than zero
                print "PASS! {} steps to goal,Goal reached {} times out of {}!".format(self.features[len(self.features)-1][self.steps],self.goal,len(self.features))
                self.goal+=1 #FIXME - order
            else:
                print "FAIL! {} steps to goal,Goal reached {} times out of {}!".format(self.features[len(self.features)-1][self.steps],self.goal,len(self.features))
                pass
        except:
            print "Trial 0 - Goal reached {} times out of {}!".format(self.goal,len(self.features))
            pass
        print "----------------------------------------------------------"
        self.features.append({})
        self.steps=0

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


# In[211]:

out=run(agentType=RandomAgent,trials=2, deadline=False) #Example of a random run, with no deadline 


# In[212]:

out=run(agentType=RandomAgent,trials=2, deadline=True) #Example of a random run


# ### Random Agent - Discussion:
# 
# When we run an agent with a random action policy, we see that it will move about the board with no pattern, and will eventually reach the destination. If we allow the use of deadlines, we see that the agent rarely reaches the destination in time, although it may still occur.

# 
# ## Identify and update state
# 
# Identify a set of states that you think are appropriate for modeling the driving agent. The main source of state variables are current inputs, but not all of them may be worth representing. Also, you can choose to explicitly define states, or use some combination (vector) of inputs as an implicit state.
# 
# At each time step, process the inputs and update the current state. Run it again (and as often as you need) to observe how the reported state changes through the run.

# In[213]:

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
        
        inputs['next_waypoint']=self.next_waypoint
        self.state= inputs    
        inputs['deadline']=deadline
        self.features[len(self.features)-1][self.steps]=inputs
        # TODO: Select action according to your policy

        action = self.availableAction[random.randint(0,3)]    
    
        # Execute action and get reward
        reward = self.env.act(self, action)
        # TODO: Learn policy based on state, action, reward

        #print "LearningAgent.update(): self.state{}, action = {}, reward = {}, next_waypoint = {}".format(
        #                                        self.state, action, reward,self.next_waypoint, )  # [debug]
print "StateAgent Ready"


# In[214]:

# run the trials for the state
stateFeatures=run(agentType=StateAgent,trials=25)


# In[215]:

# display the feedback from the prior run
import matplotlib.pyplot as plt

all_deadlines=[]
for f in stateFeatures:
    fig, axes = plt.subplots(nrows=2, ncols=3,figsize=(14,6))
    fig.suptitle( "States:{}".format(len(f)))
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

    all_deadlines.append(f.deadline.ravel().min())

    fig.show()


# In[216]:

plt.plot (all_deadlines)
plt.ylabel('Deadline')
plt.xlabel('Run')
plt.title("Deadline per Run")
plt.show()


# ### Identify and update state - discussion.
# When we sense our environment, we perceive 4 variables, with several possible states These include: left, light, next_waypoint, oncoming, and right. We can see right away that light and next_waypoint contains new information at every poll, while the others usually have no value. 
# 
# It's not readily apparent that the direction of travel information of the other cars (described by left/right/oncoming) is relevant to our agent. A case could be made to remove the direction information, and only retain information about another car being present at the light. This would have the benefit of reducing the number of possible states, increasing the speed of the agent. This may be a valuable approach in resource constrained environments. 
# 
# The downside is that the agent may pick an action that causes a longer trip. Early in the learning phase, it could also pick an action incorrectly. For instance, by proceeding through a light when the opposite car is turning left. In this case, it may have previously seen a positive reward for moving through the light, because the opposite car was not turning. This time through, it will recieve a negative reward, and in the future when a car is at the oncoming light, it will always wait till the intersection is clear.
# 
# In the interest of correctness, we will choose to use the state as returned from the sensor, with the addition of the next_waypoint.
# 
# While I have tracked the deadline, it is not apparent that it will provide useful information to the agent. It is useful to see note that the agent does not see any usefull increase in the deadline value yet. We may expect this to adapt as we implement learning.

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
