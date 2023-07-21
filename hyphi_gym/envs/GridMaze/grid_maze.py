from hyphi_gym.envs.common import Grid, Maze

class GridMaze(Maze,Grid):
  def __init__(self, render_mode=None, **kwargs):
    Maze.__init__(self, max_episode_steps=100, **kwargs)
    Grid.__init__(self, render_mode=render_mode)

# Create Renderings
if __name__ == "__main__":
  import sys
  from hyphi_gym import named, Monitor
  import gymnasium as gym
  envs = sys.argv[1:] if len(sys.argv[1:]) else ['Maze7', 'Maze11', 'Maze13', 'Maze15'] 
  #['Mazes7', 'Mazes11', 'Mazes13', 'Mazes15']
  if '-h' in envs[0]: print('Provide names of Maze environments to be rendered'); sys.exit(0)
  for name in envs:
    env = Monitor(gym.make(**named(name), render_mode='blender', seed=42), record_video=True)
    if len(env.random): 
      for _ in range(10): env.reset()
      env.save_video(f'hyphi_gym/assets/render/{name}.gif')
    else: env.reset(); env.save_image(f'hyphi_gym/assets/render/{name}.png')
