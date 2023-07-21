from hyphi_gym.envs.common import Maze, Point

class PointMaze(Point, Maze): # Point(Mujoco) | Maze(Board(Base))
  def __init__(self, render_mode=None, **kwargs):
    Maze.__init__(self, max_episode_steps=800, **kwargs) # Init Maze
    Point.__init__(self, render_mode=render_mode)


if __name__ == "__main__":
  from hyphi_gym import named, Monitor
  import gymnasium as gym
  render = True; name = "PointMazes15"
  env = Monitor(gym.make(**named(name), seed=42, render_mode='3D'), record_video=render)
  for i in range(4):
    print(i); obs, _ = env.reset(); R = []
    for a in [env.action_space.sample() for _ in range(100)]: 
      observation, reward, terminated, truncated, info = env.step(a); R.append(reward)
      if terminated or truncated: break
  if render: env.save_video(f'test/video/{name}.mp4')

