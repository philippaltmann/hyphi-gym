from hyphi_gym.envs.common import Grid, Holes

class HoleyGrid(Holes, Grid):
  def __init__(self, level='Train', render_mode=None, **kwargs):
    Holes.__init__(self, level, max_episode_steps=100, **kwargs)
    Grid.__init__(self, render_mode=render_mode)

if __name__ == "__main__":  
  import sys
  from hyphi_gym import named, Monitor
  import gymnasium as gym

  demo = '--demo' in sys.argv[1:]; path = {'Holes': [2,1,1,1,1,1,0,1], 'HolesShift': [2,1,1]}
  if demo: sys.argv.pop(sys.argv.index('--demo'))
  envs = sys.argv[1:] if len(sys.argv[1:]) else ['Holes', 'HolesShift']
  if '-h' in envs: print('Provide names of Maze environments to be rendered'); sys.exit(0)
  for name in envs:
    env = Monitor(gym.make(**named(name), render_mode='3D', seed=42), record_video=demo)
    env.reset(); env.save_image(f'hyphi_gym/assets/render/{name}.png')
    if demo: 
      for a in path[name]: env.step(a)
      env.save_video(f'hyphi_gym/assets/render/{name}.gif')
