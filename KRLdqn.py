
from __future__ import division
import argparse

from PIL import Image
import numpy as np
import gym
import os

from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten, Convolution2D, Permute
from keras.optimizers import Adam
from keras.callbacks import TensorBoard
import keras.backend as K

from rl.agents.dqn import DQNAgent
from rl.policy import LinearAnnealedPolicy, BoltzmannQPolicy, EpsGreedyQPolicy
from rl.memory import SequentialMemory
from rl.core import Processor
from rl.callbacks import FileLogger, ModelIntervalCheckpoint

from keras.callbacks import Callback as KerasCallback, CallbackList as KerasCallbackList
from keras.callbacks import TensorBoard
from keras.optimizers import Adam, RMSprop
import keras.backend as K

class SubTensorBoard(TensorBoard):
    """Subclassing of tensorboard to log and visualize custom metrics and others
  
    Note that for this to work, you will have to define a way how to handle `on_episode_end`
    calls. 
  
    Check https://github.com/martinholub/demos-blogs-examples/blob/master/rl-gym/atari/callbacks.py
    for working implementation.
    """
    def __init__(self, *args, **kwargs):
        super(SubTensorBoard, self).__init__(*args, **kwargs)

    def on_step_end(self, episode_step, logs={}):
        interesting_logs = {'reward': logs['reward']}
        for i, name in enumerate(self.model.metrics_names):
            interesting_logs.update({name: logs['metrics'][i] if logs['metrics'][i] != np.nan else 0})
        super(SubTensorBoard, self)._write_logs(interesting_logs, self.model.step)


WINDOW_LENGTH=1

test = 5
path = "KDQN/smallClassic/TEST{}".format(test)

exploration_fraction=0.5
exploration_final_eps=0.1
learning_rate=0.000025
gamma=0.95
target_network_update_freq=10000
num_steps = 2000000
memory_size = 1000000
num_runs = 3
num_ghosts = 2

os.makedirs(path)

setup = """ TEST {}:
DQN:
num_steps: {}
memory_size: {}
window_length: {}
exploration_fraction: {}
exploration_final_eps: {}
learning_rate: {}
gamma: {}
target network update freq: {}
num runs: {}
num ghosts: {}
""".format(test, num_steps, memory_size, WINDOW_LENGTH, exploration_fraction, exploration_final_eps, learning_rate, gamma, target_network_update_freq, num_runs, num_ghosts)

with open("{}/setup.txt".format(path), "w+") as f:
  f.write(setup)


class AtariProcessor(Processor):
    def process_observation(self, observation):
        return observation

    def process_state_batch(self, batch):
        return batch

    def process_reward(self, reward):
        reward = reward/400
        return np.clip(reward, -1., 1.)

parser = argparse.ArgumentParser()
parser.add_argument('--mode', choices=['train', 'test'], default='train')
parser.add_argument('--env-name', type=str, default='BerkeleyPacman-training-v0')
parser.add_argument('--weights', type=str, default=None)
args = parser.parse_args()

for i in range(num_runs):
  run_path = "{}/RUN{}".format(path, i)
  os.makedirs(run_path)
  # Get the environment and extract the number of actions.
  env = gym.make(args.env_name)
  np.random.seed(123)
  # env.seed(123)
  nb_actions = env.action_space.n
  
  # Next, we build our model. We use the same model that was described by Mnih et al. (2015).
  model = Sequential()
  model.add(Flatten(input_shape=(WINDOW_LENGTH,) + env.observation_space.shape))
  model.add(Dense(512))
  model.add(Activation('relu'))
  model.add(Dense(512))
  model.add(Activation('relu'))
  model.add(Dense(512))
  model.add(Activation('relu'))
  model.add(Dense(nb_actions))
  model.add(Activation('linear'))
  print(model.summary())
  
  # Finally, we configure and compile our agent. You can use every built-in Keras optimizer and
  # even the metrics!
  memory = SequentialMemory(limit=memory_size, window_length=WINDOW_LENGTH)
  processor = AtariProcessor()
  
  # Select a policy. We use eps-greedy action selection, which means that a random action is selected
  # with probability eps. We anneal eps from 1.0 to 0.1 over the course of 1M steps. This is done so that
  # the agent initially explores the environment (high eps) and then gradually sticks to what it knows
  # (low eps). We also set a dedicated eps value that is used during testing. Note that we set it to 0.05
  # so that the agent still performs some random actions. This ensures that the agent cannot get stuck.
  policy = LinearAnnealedPolicy(EpsGreedyQPolicy(), attr='eps', value_max=1., value_min=exploration_final_eps, value_test=.05,
                                nb_steps=int(exploration_fraction * num_steps))
  
  # The trade-off between exploration and exploitation is difficult and an on-going research topic.
  # If you want, you can experiment with the parameters or use a different policy. Another popular one
  # is Boltzmann-style exploration:
  # policy = BoltzmannQPolicy(tau=1.)
  # Feel free to give it a try!
  
  dqn = DQNAgent(model=model, nb_actions=nb_actions, policy=policy, memory=memory,
                 processor=processor, nb_steps_warmup=50000, gamma=gamma, target_model_update=target_network_update_freq,
                 train_interval=1, delta_clip=1.)
  dqn.compile(Adam(lr=learning_rate), metrics=['mae'])
  
  if args.mode == 'train':
      # Okay, now it's time to learn something! We capture the interrupt exception so that training
      # can be prematurely aborted. Notice that now you can use the built-in Keras callbacks!
      weights_filename = '{}/dqn_{}_weights.h5f'.format(run_path, args.env_name)
      checkpoint_weights_filename = '{}/dqn_'.format(run_path) + args.env_name + '_weights_{step}.h5f'
      log_filename = '{}/dqn_{}_log.json'.format(run_path, args.env_name)
      callbacks = [ModelIntervalCheckpoint(checkpoint_weights_filename, interval=250000)]
      callbacks += [SubTensorBoard(log_dir='{}/'.format(run_path))]
      callbacks += [FileLogger(log_filename, interval=100)]
      dqn.fit(env, callbacks=callbacks, nb_steps=num_steps, log_interval=10000)
  
      # After training is done, we save the final weights one more time.
      dqn.save_weights(weights_filename, overwrite=True)
  elif args.mode == 'test':
      weights_filename = '{}/dqn_{}_weights.h5f'.format(run_path, args.env_name)
      if args.weights:
          weights_filename = args.weights
      dqn.load_weights(weights_filename)
      dqn.test(env, nb_episodes=10, visualize=True)

  env.close()
