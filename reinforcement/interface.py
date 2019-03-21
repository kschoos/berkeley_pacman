import os
from threading import Thread

import numpy as np

from pacman import runGames, readCommand
import time


class ActionSpace:
    def __init__(self):
        self.n = 5

    def sample(self):
        return np.random.randint(0, self.n)

class ObservationSpace:
    def __init__(self, observation_shape):
        self.shape = observation_shape

class Env:
    def __init__(self, layout, numGames, numGhosts, numTraining):
        argv = []
        os.chdir("/home/skusku/Documents/Uni/Pacman/reinforcement")
        argv.append("-pInterfaceAgent")
        argv.append("-l{}".format(layout))
        argv.append("-n{}".format(numGames))
        argv.append("-x{}".format(numTraining))
        argv.append("-k{}".format(numGhosts))
        argv.append("-q")

        self.argv = argv

        args = readCommand(argv)

        self.layout = args['layout']

        self.agent = None

        observation_shape = (self.layout.width, self.layout.height)
        self.action_space = ActionSpace()
        self.observation_space = ObservationSpace(observation_shape)

    def reset(self):
        args = readCommand(self.argv)
        thread = Thread(target=runGames, kwargs=args)
        thread.start()
        self.agent = args['pacman']

        self.agent.first_observation_barrier.wait(2)
        return self.agent.last_observation


    def step(self, action):
        self.agent.action_to_take = action
        self.agent.getAction_barrier.wait(2)
        self.agent.update_barrier.wait(2)
        next_state = self.agent.last_next_observation
        reward =  self.agent.last_reward
        done = abs(reward) > 400
        info = dict()

        return next_state, reward, done, info

if __name__ == "__main__":
    # EXAMPLE CREATION OF ENVIRONMENT
    env = Env(layout="smallClassic", numGames=100, numGhosts=1, numTraining=0)

    # EXAMPLARY USE
    # -----------------------------------------------------------
    print("nb Actions: {}".format(env.action_space.n))
    print("Observation space shape: {}".format(env.observation_space.shape))

    for i in range(200):
        next_state, reward, done, _ = env.step(2)
    env.agent.stop()
    # ----------------------------------------------------------
