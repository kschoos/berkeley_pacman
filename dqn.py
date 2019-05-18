import gym

from stable_baselines.deepq.policies import MlpPolicy
from stable_baselines import DQN

env = gym.make('BerkeleyPacman-training-v0')
model = DQN(MlpPolicy, env, verbose=1, prioritized_replay=True, param_noise=True, tensorboard_log="./logdir")
model.learn(total_timesteps=1000000)
model.save("dqn_origClassic")

env.close()

env = gym.make('BerkeleyPacman-testing-v0')
env = DummyVecEnv([lambda: env])

obs = env.reset()
dones = [False]
while not dones[0]:
  action, _states = model.predict(obs)
  obs, rewards, dones, info = env.step(action)
