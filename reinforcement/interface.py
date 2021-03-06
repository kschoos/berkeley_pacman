import os
import socket
import subprocess
import sys
from threading import Thread

import gym
import numpy as np

path = "/home/skusku/Documents/Repos/berkeley_pacman"

class ActionSpace:
    def __init__(self):
        self.n = 5

    def sample(self):
        return np.random.randint(0, self.n)

class ObservationSpace:
    def __init__(self, observation_shape):
        self.shape = observation_shape

class Env(gym.Env):
    def initlogging(self):
        if(self.logging):
            self.error_log = open("{}/error.log".format(path), "w+")

    def log(self, str):
        if(self.logging):
            self.error_log.write(str)

    def flushlog(self):
        if(self.logging):
            self.error_log.flush()

    def closelog(self):
        if(self.logging):
            self.error_log.close()

    def __init__(self, layout, numGames, numGhosts, numTraining, layoutWidth, layoutHeight, logging):
        self.logging = logging
        self.error_log = None
        self.initlogging()

        self.pacman_cwd = "{}/reinforcement/".format(path)

        argv = []
        argv.append("python2")
        argv.append(self.pacman_cwd + "pacman.py")
        argv.append("-pInterfaceAgent")
        argv.append("-l{}".format(layout))
        argv.append("-n{}".format(numGames))
        argv.append("-x{}".format(numTraining))
        argv.append("-k{}".format(numGhosts))
        argv.append("--udp")

        self.layoutWidth = layoutWidth
        self.layoutHeight = layoutHeight
        self.process = None

        self.argv = argv
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.server_addr = ('localhost', 10000)
        self.client_addr = None
        self.socket.bind(self.server_addr)
        self.socket.listen(1)
        self.connection = None

        observation_shape = (self.layoutWidth, self.layoutHeight)

        self.observation_length = (self.layoutWidth + 1) * self.layoutHeight
        self.reward_length = 4
        self.done_length = 1
        self.packet_length = self.observation_length + self.reward_length + self.done_length

        self.action_space = gym.spaces.Discrete(5)
        self.observation_space = gym.spaces.Box(32, 111, [self.layoutWidth, self.layoutHeight, 1])

    def render(self, state):
        pass

    def close(self):
        try:
            # self.socket.shutdown(socket.SHUT_RDWR)
            self.connection.close()
            self.socket.close()
            self.process.kill()
            self.closelog()
        except Exception as e:
            print(e)
            pass

    def process_observation_string(self, str):
        width, height = self.layoutWidth, self.layoutHeight
        map = np.zeros(shape=(width, height))

        agentChrs = ['v', '^', '>', '<']
        ghostChrs = ['M', 'W', '3', 'E']

        for y in range(height):
            for x in range(width):
                chr = str[(width + 1) * y + x]

                if chr in agentChrs:
                    chr = 'P'
                if chr in ghostChrs:
                    chr = 'G'


                map[x][y] = ord(chr)

        map = map.reshape((self.layoutWidth, self.layoutHeight, 1))

        return map

    def reset(self):
        self.log("Resetting environment\n")
        if self.process:
            self.process.kill()
        # if self.connection:
        #    self.connection.close()

        self.process = subprocess.Popen(self.argv, cwd=self.pacman_cwd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        self.log("Started subprocess\n")

        self.connection, self.client_addr = self.socket.accept()
        self.log("Accepted connection\n")

        data = self.connection.recv(self.observation_length)
        data = data.decode("ASCII")

        self.log("Got first observation: ---------------------------------\n")
        self.log(data)
        self.log("-------------------------------------------------------\n")

        self.flushlog()

        return self.process_observation_string(data)


    def step(self, action):
        self.connection.send(bytes("{}".format(action), "ASCII"))

        self.log("Sending: {}\n".format(action))
        data = self.connection.recv(self.packet_length)
        data = data.decode("ASCII")

        done_str = data[:self.done_length]
        reward_str = data[self.done_length:self.done_length + self.reward_length]
        state_str = data[self.done_length + self.reward_length:]

        self.log("Received: --------------------------------------\n")
        self.log(data)
        self.log("------------------------------------------------\n")

        try:
            next_state = self.process_observation_string(state_str)
        except Exception as e:
            self.closelog()
            raise e

        reward = int(reward_str)
        done = abs(reward) > 450

        info = dict()

        return next_state, reward, done, info

# class ThreadEnv(gym.Env):
#     def __init__(self, layout, numGames, numGhosts, numTraining):
#         argv = []
#         os.chdir("/home/skusku/Documents/Uni/Pacman/reinforcement")
#         argv.append("-pInterfaceAgent")
#         argv.append("-l{}".format(layout))
#         argv.append("-n{}".format(numGames))
#         argv.append("-x{}".format(numTraining))
#         argv.append("-k{}".format(numGhosts))
#         argv.append("--udp")
#         # argv.append("-q")
#
#         self.argv = argv
#         self.server_addr = ('localhost', 10000)
#         self.socket = None
#
#         args = readCommand(argv)
#
#         self.layout = args['layout']
#
#         self.agent = None
#
#         observation_shape = (self.layout.width, self.layout.height)
#
#         self.observation_length = (self.layout.width + 1) * self.layout.height
#         self.reward_length = 4
#         self.done_length = 1
#         self.packet_length = self.observation_length + self.reward_length + self.done_length
#
#         print(self.layout.width, self.layout.height)
#
#         self.action_space = ActionSpace()
#         self.observation_space = ObservationSpace(observation_shape)
#
#     def process_observation_string(self, str):
#         width, height = self.layout.width, self.layout.height
#         map = np.zeros(shape=(width, height))
#
#         agentChrs = ['v', '^', '>', '<']
#         ghostChrs = ['M', 'W', '3', 'E']
#
#         for y in range(self.layout.height):
#             for x in range(self.layout.width):
#                 chr = str[(width + 1) * y + x]
#
#                 if chr in agentChrs:
#                     chr = 'P'
#                 if chr in ghostChrs:
#                     chr = 'G'
#
#
#                 map[x][y] = ord(chr)
#
#         return map
#
#     def reset(self):
#         args = readCommand(self.argv)
#         thread = Thread(target=runGames, kwargs=args)
#         thread.start()
#
#         self.socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
#
#         self.socket.sendto("start", self.server_addr)
#         data, addr = self.socket.recvfrom(self.observation_length)
#
#         return self.process_observation_string(data)
#
#
#     def step(self, action):
#         sent = self.socket.sendto("{}".format(action), self.server_addr)
#         data, addr = self.socket.recvfrom(self.packet_length)
#
#         done_str = data[:self.done_length]
#         reward_str = data[self.done_length:self.done_length + self.reward_length]
#         state_str = data[self.done_length + self.reward_length:]
#
#         next_state = self.process_observation_string(state_str)
#         reward = int(reward_str)
#         done = abs(reward) > 200
#         info = dict()
#
#         return next_state, reward, done, info

# if __name__ == "__main__":
#     # EXAMPLE CREATION OF ENVIRONMENT
#     env = Env(layout="smallClassic", numGames=100, numGhosts=1, numTraining=0)
#
#     # EXAMPLARY USE
#     # -----------------------------------------------------------
#     print("nb Actions: {}".format(env.action_space.n))
#     print("Observation space shape: {}".format(env.observation_space.shape))
#
#     for i in range(200):
#         next_state, reward, done, _ = env.step(2)
#     env.agent.stop()
#     # ----------------------------------------------------------


if __name__ == "__main__":
    lastMove = 4
    keys = []

    print("Starting env")
    env = Env(layout="smallClassic", numGames=1, numGhosts=0, numTraining=0, layoutWidth=20, layoutHeight=7)

    actions = [1, 2, 1, 1, 2, 4, 3, 2, 0, 0, 2]
    for i in range(10):
        observation = env.reset()
        for action in actions:
            try:
                # action = int(sys.stdin.read(1))
                _, _, done, _ = env.step(action)
            except Exception as e:
                print(e)
                pass

    env.close()

    # done = False
    # while(not done):

    print("Done!")
