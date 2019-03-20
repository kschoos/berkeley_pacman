from threading import Thread

from pacman import runGames, readCommand
import time


class Space:
    def __init__(self, observation_shape):
        self.n = 5
        self.shape = observation_shape

class Env:
    def __init__(self, observation_shape, layout, numGames, numGhosts, numTraining):
        self.action_space = Space(observation_shape)

        argv = []
        argv.append("-pInterfaceAgent")
        argv.append("-l{}".format(layout))
        argv.append("-n{}".format(numGames))
        argv.append("-x{}".format(numTraining))
        argv.append("-k{}".format(numGhosts))

        args = readCommand(argv)

        thread = Thread(target=runGames, kwargs=args)
        thread.start()

        self.agent = args['pacman']
        self.getAction_CV = self.agent.getAction_CV
        self.update_CV = self.agent.update_CV

        for i in range(200):
            next_state, reward, done, _ = self.step(2)
        self.agent.stop()

    def step(self, action):
        self.agent.getAction_CV.acquire()
        self.agent.action_to_take = action
        self.agent.new_action = True
        self.agent.getAction_CV.notify()
        self.agent.getAction_CV.release()


        self.agent.update_CV.acquire()
        while not self.agent.new_update_data:
            self.agent.update_CV.wait()
        self.agent.new_update_data = False
        self.agent.update_CV.release()


        next_state = self.agent.last_next_observation
        reward =  self.agent.last_reward
        done = abs(reward) > 400
        info = None

        return next_state, reward, done, info

if __name__ == "__main__":
    Env((1, 2), "smallClassic", 100, 1, 0)
