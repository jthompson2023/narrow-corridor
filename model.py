from __future__ import annotations

import numpy as np
import numpy.typing as npt
import matplotlib.pyplot as plt
from random import normalvariate as normal
from collections.abc import Sequence
from typing import Union
from heapq import nlargest, nsmallest

# INTRODUCTION
# 
# Imagine that your nation exists on the narrow corridor graph somewhere. The actions of major
# agents in the game (politicians, unions, lobbying groups, businesses, public opinion, etc.)
# can *move* your nation around on that graph—nudging it towards (or away from) state (or society)
# power. Where the nation is in the corridor determines the benefits those agents receive.
# 
# It's best for everyone if both state and society power are maximized, but in balance. However,
# agents always have the temptation to grab for state power for themselves, since it can allow
# them to extract a greater cut of the wealth… even if the general welfare would be better off if
# they didn't. It's a tragedy of the commons.
#
# I designed this simulation to explore that conflict. A pool of agents, each with their own strategy,
# exist in a nation with some pre-existing spot on the graph. Each one then must ask, "given the state
# of the nation, and what my strategy is, in what direction should I nudge the nation this round?"
#
# The ultimate position of the nation affects each agent's score. Agents with the highest scores
# "reproduce," creating others like them—but with a slightly *mutated* strategy, like evolution, allowing
# new traits to emerge over time. Meanwhile, those with the lowest scores "die," not having the power
# to keep going. Thus, the whole simulation can evolve over time.


def shrink(vector: npt.ArrayLike):
    return vector / abs(max(vector, key=lambda x: abs(x)) or 1)
    
# A class to determine the behavior of agents in the simulation
class Agent:
    def __init__(self, weights=None, bias=None):
        # As in machine learning, each agent calculates what it's going to do with a series of
        # weights (which react to the current state of society) and a bias (which inclines them
        # in a certain direction regardless of what's going on). 
        # 
        # The exact math isn't important to get the gist—what matters is that these are numbers
        # which can be manipulated and *changed* by a machine. It's a way of representing a
        # strategy in a nice way for computers to deal with.


        # initialize weights
        if weights == 'random':
            self.weights = np.array(np.random.rand(2) * 2 - 1)
        elif weights is not None:
            self.weights = np.array(weights)
        else:
            self.weights = np.zeros(2)
        
        # initialize bias
        if bias == 'random':
            self.bias = np.array(np.random.rand(2) * 2 - 1)
        elif bias is not None:
            self.bias = np.array(bias)
        else:
            self.bias = np.zeros(2)
        
        # initialize externally set vars
        self.score = 0
        self.state_builder = False
    
    def decide(self, vector: npt.ArrayLike):
        # get the product of the input and the weights, plus the bias, rescaled so the magnitude of
        # the largest component vector is 1.
        product = shrink(self.weights * vector + self.bias)
        magnitudes = abs(product)

        # return a decision vector in one of the 4 cardinal directions: [1, 0]; [-1, 0]; [0, 1]; or [0, -1],
        # based on whichever component of the product has the greatest magnitude. return [0, 0] if they're equal.
        if magnitudes[0] > magnitudes[1]:
            return np.array([product[0], 0])
        elif magnitudes[1] > magnitudes[0]:
            return np.array([0, product[1]])
        if abs(product[0]) == abs(product[1]):
            return np.array([0, 0])
        
        # That's a bunch of math jargon to say, "the agent looks at the state of the simulated society, 
        # and according to its weights and bias, decides which direction to give that society a little push."

    def reproduce(self, std_dev=0) -> Agent:
        # return a new agent with bias and weights chosen by a normal distribution centered around
        # current agent's bias and weights. This means strategies *mutate* when these agents reproduce,
        # allowing the simulation to select for the optimal self-interested strategy.

        spawn_weights = [normal(x, std_dev) for x in self.weights]
        spawn_bias = [normal(x, std_dev) for x in self.bias]
        return Agent(weights=spawn_weights, bias=spawn_bias)

# A class to code for the actual operation of the simulation
class Simulation:
    def __init__(self, nation_vec: Union[str, npt.ArrayLike]=None, corridor_boost: int=1, 
                 agents: Sequence[Agent]=None, population: int=0, **agent_kwargs):
        self.round = 0
        self.corridor_boost = 1

        # initialize the nation vector, which represents its position on the Narrow Corridor 2-axis plot.
        # So for instance, a nation vector (2, 5) would be 2 units of power for society and 5 units for
        # state. What are those units? They're made-up numbers; what matters is how high they are, and how
        # they compare to each other.

        if nation_vec == 'random':
            self.nation = np.random.rand(2) * 5
        elif nation_vec is not None:
            self.nation = np.array(nation_vec)
        else:
            self.nation = np.zeros(2)
        
        # initialize agents by randomly generating them or allowing the programmer to specify their parameters
        if agents:
            self.agents = agents
            if population:
                raise ValueError('cannot have agents parameter and population parameter specified; choose 1')
        else:
            self.agents = [Agent(**agent_kwargs) for __ in range(population)]
    
    def display(self):
        # plot the nation vector on a graph
        plt.title(f'national state, round {self.round}')
        plt.xlabel('society power')
        plt.ylabel('state power')
        plt.scatter(*self.nation)
        plt.show()
    
    def get_decisions(self):
        # Give each agent the status of the nation, and collect each one's decision.
        output = []
        for agent in self.agents:
            decision = agent.decide(self.nation)
            if decision == np.array([0, 1]) or decision == np.array([-1, 0]):
                agent.state_builder = True
            else:
                agent.state_builder = False
            output.append(decision)

        return output

    def move_nation(self, decisions: Sequence[np.ArrayLike]):
        # Add up all the decisions of the agent on the nation
        new_pos = self.nation + sum(decisions)
        # clamp the new position so that it can never have negative coordinates.
        # (0 state power makes sense; -1 state power does not.)
        for i, item in new_pos:
            new_pos[i] = max(item, 0)

        # move to new position
        self.nation = new_pos
    
    def give_payouts(self):
        # Now each agent gets to reap the benefits of the nation's position on the state-society
        # power graph! This is calculated by a mathematical formula, which is highest in the
        # corridor, and advantages high state & society power as long as they're kept in balance.
        #
        # However! Those who make the state stronger get an extra cut of the nation's payout,
        # proportional to how strong the state is. Building the state can benefit everyone,
        # but agents are tempted to keep boosting state power to extract more and more of the
        # nation's payout, even if that harms the nation itself… as despots do.

        x, y = self.nation
        payout = self.corridor_boost * x * y / (x + y)
        state_builder_cut = payout * y / (x + y)
        shared_cut = payout - state_builder_cut

        for agent in self.agents:
            agent.score += shared_cut / len(self.agents)

        state_builders = self.get_state_builders()
        for state_builder in state_builders:
            state_builder.score += state_builder_cut / len(state_builders)
    
    def reproduce(self, std_dev: float=0.1, cull: int=5, spawn: int=5):
        # Make the highest-performing agents reproduce, and kill off the lowest-performing agents.
        # This allows the simulation to evolve towards whatever the optimal, selfish, evolutionary
        # strategy or strategies are.

        new_agents = [agent.reproduce(std_dev) for agent in nlargest(spawn, self.agents, key=lambda x: x.score)]
        for agent in nsmallest(cull, self.agents, key=lambda x: x.score):
            self.agents.remove(agent)
        self.agents.extend(new_agents)
    
    def advance(self):
        self.move_nation(self.get_decisions())
        self.give_payouts()
        self.reproduce()

    def get_state_builders(self):
        return [agent for agent in self.agents if agent.state_builder]

sim = Simulation()
sim.display()
