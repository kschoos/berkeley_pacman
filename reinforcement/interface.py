from threading import Thread

from pacman import runGames, readCommand
import time


class ActionSpace:
    def __init__(self):
        self.n = 5

class ObservationSpace:
    def __init__(self, observation_shape):
        self.shape = observation_shape

class Env:
    def __init__(self, layout, numGames, numGhosts, numTraining):
        argv = []
        argv.append("-pInterfaceAgent")
        argv.append("-l{}".format(layout))
        argv.append("-n{}".format(numGames))
        argv.append("-x{}".format(numTraining))
        argv.append("-k{}".format(numGhosts))

        args = readCommand(argv)

        observation_shape = (6, args['layout'].width, args['layout'].height)
        self.action_space = ActionSpace()
        self.observation_space = ObservationSpace(observation_shape)

        thread = Thread(target=runGames, kwargs=args)
        thread.start()

        self.agent = args['pacman']
        self.getAction_CV = self.agent.getAction_CV
        self.update_CV = self.agent.update_CV


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
