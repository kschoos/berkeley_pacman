import gym

from stable_baselines.common.policies import MlpPolicy
from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines import A2C


n_cpu = 1
env = gym.make('BerkeleyPacman-training-v0')
env = DummyVecEnv([lambda: env])
model = A2C(MlpPolicy, env, verbose=1, tensorboard_log="./logdir")
model.learn(total_timesteps=50000000)
model.save("a2c_pacman_origClass_with_RT_default_lr")

env.close()

env = gym.make('BerkeleyPacman-testing-v0')
env = DummyVecEnv([lambda: env])

obs = env.reset()
dones = [False]
while not dones[0]:
  action, _states = model.predict(obs)
  obs, rewards, dones, info = env.step(action)
