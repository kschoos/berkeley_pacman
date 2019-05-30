import gym
import random

from stable_baselines.deepq.policies import FeedForwardPolicy
from stable_baselines import DQN
from stable_baselines.common.vec_env import DummyVecEnv

# Custom MLP policy of two layers of size 32 each
class CustomPolicy(FeedForwardPolicy):
    def __init__(self, *args, **kwargs):
        super(CustomPolicy, self).__init__(*args, **kwargs,
                                           layers=[256, 256],
                                           layer_norm=False,
                                           feature_extraction="mlp")

# env = gym.make('BerkeleyPacman-training-v0')
# env = DummyVecEnv([lambda: env])
# model = DQN(CustomPolicy, exploration_fraction=0.9, learning_rate=0.00005, env=env, verbose=1, tensorboard_log="./logdir/dqn_smallClassic/", full_tensorboard_log=True)
# model.learn(total_timesteps=30000)
# model.save("dqn_smallClassic")
# 
# env.close()

env = gym.make('BerkeleyPacman-testing-v0')
env = DummyVecEnv([lambda: env])

model = DQN.load("dqn_smallClassic")

obs = env.reset()
dones = [False]
while not dones[0]:
  action, _states = model.predict(obs)
  obs, rewards, dones, info = env.step(action)
