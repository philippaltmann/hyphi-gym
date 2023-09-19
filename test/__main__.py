import argparse
from hyphi_gym import named, Monitor
import gymnasium as gym
from PIL import Image
import numpy as np; import math
from gymnasium.wrappers.autoreset import AutoResetWrapper

parser = argparse.ArgumentParser()
parser.add_argument('envs', nargs='+', help='A list of environments to test')
parser.add_argument('--demo', nargs='+', default=[], type=int, help='A list of actions to execute')
parser.add_argument('--runs', type=int, default=9, help='Number of random configurations to be generated')
parser.add_argument('--square', action='store_true', help='Save suare images instead of 3:2')
parser.add_argument('--grid', action='store_true', help='Save grid of layouts instead of gif')

args = parser.parse_args()

def image_grid(imgs, rows, cols):
  assert len(imgs) == rows*cols; w, h = imgs[0].size
  grid = Image.new('RGB', size=(cols*w, rows*h))
  for i, img in enumerate(imgs):
    grid.paste(img, box=(i%cols*w, i//cols*h))
  return grid

for name in args.envs:
  if "Point" in name or "Plane" in name:
    render,lookup,n = '2D', [[0,1],[1,0],[0,-1],[-1,0]], 20
    demo = [lookup[a] for a in args.demo for _ in range(n)]
  else: demo, render = args.demo, 'blender'

  env = AutoResetWrapper(Monitor(
      gym.make(**named(name), render_mode=render, seed=42)
    , record_video=True)) #len(args.demo)>0)
  layouts = []
  for i in range(args.runs if len(env.random) else 1): 
    env.reset(); render = env.render(); 
    if isinstance(render, np.ndarray): render = Image.fromarray(render)
    if not args.square: w = 720; h = w/3*2; c = (w-h)/2; render = render.crop((0,c,w,h+c))
    layouts.append(render) 
    R = []
    for i,a in enumerate(demo): 
      s,r,tm,tr,info = env.step(a); R.append(r)
      if tm or tr: break
    print(sum(R))
  if args.grid:
    s = int(math.sqrt(len(layouts))); image_grid(layouts,s,s).save(f"test/render/{name}.png")
  else:
    layouts[0].save(
      f"test/render/{name}.{'gif' if len(layouts) > 1 else 'png'}",
      save_all=True, append_images=layouts[1:], optimize=False, 
      duration=1000/env.metadata['render_fps'], loop=0
    )
  # if len(layouts) > 1: image_grid(layouts, 10, 10).save(f"test/render/{name}-grid.png")

  if len(args.demo): env.save_video(f'test/demo/{name}.mp4')
  print("done")

