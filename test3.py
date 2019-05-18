import gym
from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines import A2C

model = A2C.load("a2c_pacman")
env = gym.make('BerkeleyPacman-testing-v0')
env = DummyVecEnv([lambda: env])

obs = env.reset()
dones = [False]
while not dones[0]:
  action, _states = model.predict(obs)
  obs, rewards, dones, info = env.step(action)
