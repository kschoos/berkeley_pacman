import gym

from stable_baselines.deepq.policies import MlpPolicy
from stable_baselines import DQN
from stable_baselines.common.vec_env import DummyVecEnv

env = gym.make('BerkeleyPacman-training-v0')
env = DummyVecEnv([lambda: env])
model = DQN(MlpPolicy, env, verbose=1, prioritized_replay=True, param_noise=True, tensorboard_log="./logdir/dqn_smallClassic/")
model.learn(total_timesteps=20000000)
model.save("dqn_smallClassic")

env.close()

env = gym.make('BerkeleyPacman-testing-v0')
env = DummyVecEnv([lambda: env])

obs = env.reset()
dones = [False]
while not dones[0]:
  action, _states = model.predict(obs)
  obs, rewards, dones, info = env.step(action)
