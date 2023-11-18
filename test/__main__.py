import argparse; import numpy as np; import math; from PIL import Image
import gymnasium as gym; from gymnasium.wrappers.autoreset import AutoResetWrapper
from hyphi_gym import named, Monitor

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

disc = lambda d, lo=-1, hi=1: [[*([0]*n), v, *([0]*(d-n-1))] for v in (hi,lo) for n in range(d)[::-1]]

for name in args.envs:
  if "Point" in name or "Plane" in name:
    render,lookup,n = '3D', disc(2), 20
    demo = [lookup[a] for a in args.demo for _ in range(n)]
    # args.square = True

  elif "Fetch" in name:
    # 0       1  2     3     4       5    6    7
    # Gripper Up Right Front Gripper Down Left Back
    render, lookup, n = '3D', disc(4), 1
    args.square = True
    demo = [lookup[a] for a in args.demo for _ in range(n)]
  else: demo, render = args.demo, '3D'#'blender'

  env = AutoResetWrapper(Monitor(
      gym.make(**named(name), render_mode=render, seed=42)
    , record_video=True)) #len(args.demo)>0)
  layouts = []; env.reset(); tm, tr = False, False
  for i in range(args.runs if len(env.unwrapped.random) else 1): 
    if not (tm or tr): env.reset(seed=i)
    render = env.render(); 
    if isinstance(render, np.ndarray): render = Image.fromarray(render)
    if not args.square: w = 720; h = w/3*2; c = (w-h)/2; render = render.crop((0,c,w,h+c))
    layouts.append(render) 
    R = []
    for i,a in enumerate(demo): 
      s,r,tm,tr,info = env.step(a); R.append(r)
      if tm or tr: break
    # print(sum(R))
  if args.grid:
    s = int(math.sqrt(len(layouts))); image_grid(layouts,s,s).save(f"test/render/{name}.png")
  else:
    layouts[0].save(
      f"test/render/{name}.{'gif' if len(layouts) > 1 else 'png'}",
      save_all=True, append_images=layouts[1:], optimize=False, 
      loop=0, duration=200 #100/env.metadata['render_fps']
    )
  # if len(layouts) > 1: image_grid(layouts, 10, 10).save(f"test/render/{name}-grid.png")

  if len(args.demo): env.get_wrapper_attr('save_video')(f'test/demo/{name}.mp4')
  print("done")

