import minari
import gymnasium as gym
from PIL import Image
import os


dataset = minari.load_dataset('kitchen-complete-v0')

env = dataset.recover_environment()
env = gym.make(env.spec, render_mode = 'rgb_array')

frames = []

for i, eps in enumerate(dataset.iterate_episodes()):
    if i == 1:
        break
    env.reset(seed=eps.seed.item())
    for act in eps.actions:
        env.step(act)
        frames.append(Image.fromarray(env.render()))
env.close()

videos_path = os.path.join('videos', 'kitchen-complete-v0.gif')

frames[0].save(videos_path, save_all=True, append_images=frames[1:], duration=20,loop=0)
