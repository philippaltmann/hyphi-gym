""" Script for systematically testing all hyphi envs w.r.t.
- sparse and dense reward calcualtions
- training and exploration setups
- 7-, 9-, 11-, 13-, 15-sized mazes with kown solutions
- Train and shifted holey grids
- random layouts, agent-, and target- positions """
import sys
from typing import Union
from PIL import Image
import hyphi_gym
import gymnasium as gym
from hyphi_gym import Monitor
from hyphi_gym.utils import stdout_redirected

render = '--render' in sys.argv 

TEST = {
  'PointMaze7': { 'desc': 'Continuous MAze with size', 'args': {
      'sparse': False, 'explore': False, 'size': 7, 
      'random': [], 'render_mode': '3D', 'seed':42
    }, 'path': [2,1,1,1,1,1,1,0], 'return': 42,
  },
  # 'HolesSparse': { 'desc': 'Sparse Train Success', 'args': {
  #     'sparse': True, 'explore': False, 'level': 'Train', 
  #     'random': [], 'render_mode': '3D', 'seed':42
  #   }, 'path': [2,1,1,1,1,1,1,0], 'return': 42,
  # },
  # 'HolesShift': { 'desc': 'Dense Test Fail', 'args': {
  #     'sparse': False, 'explore': False, 'level': 'Shift', 
  #     'random': [], 'render_mode': '3D', 'seed':42
  #   }, 'path': [2,1,1,1,1,1,1,0], 'return': -53,
  # },
  # 'AgentHoles': { 'desc': 'Initial Random Agent Dense Train Success',
  #   'args': { 'sparse': False, 'explore': False, 'level': 'Train', 
  #     'random': ['Agent'], 'render_mode': '2D', 'seed':42
  #   }, 'path': [1,1,1,0,0,0], 'return': 44,
  # },
  # 'SparseHolesTargets': { 'desc': 'Random Tragets Sparse Train Fail',
  #   'args': { 'sparse': True, 'explore': False, 'level': 'Train', 
  #     'random': ['Targets'], 'render_mode': '2D', 'seed':42
  #   }, 'path': [2,2,2,2,1,1], 'return': -56,
  # },
  # 'ShiftExploreHoles': { 'desc': 'Deterministic Shift Explore',
  #   'args': { 'sparse': False, 'explore': True, 'level': 'Shift', 
  #     'random': [], 'render_mode': '3D', 'seed':42
  #   }, 'path': [2,2,1,1,1,1,1,1,1,0,0], 'return': 0,
  # },
  # 'ShiftTargetsHolesAgent': { 'desc': 'Initial Random Agent Random Target Dense Test Fail',
  #   'args': { 'sparse': False, 'explore': False, 'level': 'Shift', 
  #     'random': ['Agent','Targets'], 'render_mode': '3D', 'seed':42
  #   }, 'path': [2,2,1,1,1,1,1,1,1,2,2], 'return': -11,
  # },
  # 'Maze7': { 'desc': 'Single Dense Maze 7 Success',
  #   'args': {'sparse': False, 'explore': False, 'size': 7, 'random': [], 'seed': None,
  #     'render_mode': '3D'}, 'path': [1,1,0,0,1,1,0,0], 'return': 42,
  # },
  # 'Agents7Mazes': { 'desc': 'Random Dense Maze 7 Incomplete',
  #   'args': {'sparse': False, 'explore': False, 'size': 7, 
  #     'random': ['Agents','Layouts'], 'render_mode': '2D', 'seed':42
  #   }, 'path': [1,1,0,0,1,1,0,0], 'return': -8,
  # },
  # 'ExploreMaze7Agent': { 'desc': 'Explorative Single Maze 7 Incomplete',
  #   'args': {'sparse': False, 'explore': True, 'size': 7, 
  #     'random': ['Agent'], 'render_mode': '3D', 'seed':42
  #   }, 'path': [1,1,0,0,1,1,0,0], 'return': 0,
  # },
  # 'MazeSparse9': { 'desc': 'Single Sparse Maze 9 Success',
  #   'args': {'sparse': True, 'explore': False, 'size': 9, 
  #     'random': [], 'render_mode': '3D', 'seed':42
  #   }, 'path': [1,1,1,1,0,0,3,3,0,0,1,1,1,1,0,0], 'return': 34,
  # }, 
  # 'AgentsSparse9Maze': { 'desc': 'Single Sparse Maze 7 Random Agent Success',
  #   'args': {'sparse': True, 'explore': False, 'size': 9, 
  #     'random': ['Agents'], 'render_mode': '2D', 'seed':42
  #   }, 'path': [3,3,3,1,3,1,1,1,1], 'return': 41,
  # }, 
  # 'SparseMazes9ExploreAgents': { 'desc': 'Explorative Random Maze 9 Incomplete',
  #   'args': {'sparse': True, 'explore': True, 'size': 9, 
  #     'random': ['Agents','Layouts'], 'render_mode': '2D', 'seed':42
  #   }, 'path': [1,1,1,1,0,0,3,3,0,0,1,1,1,1,0,0], 'return': 0,
  # }, 
  # '11Maze': { 'desc': 'Single Dense Maze 11 Success', 'args': { 'seed': None,
  #   'sparse': False, 'explore': False, 'size': 11, 'random': [], 'render_mode': '3D'
  #   }, 'path': [1,1,0,0,1,1,1,1,0,0,1,1,0,0,3,3,0,0,1,1], 'return': 30,#26, 
  # },
  # '11TargetsMaze': { 'desc': 'Single Dense Maze 11 Random Target Fail',
  #   'args': {'sparse': False, 'explore': False, 'size': 11,
  #     'random': ['Targets'], 'render_mode': '2D', 'seed':40
  #   }, 'path': None, 'return': -100,
  # },
  # 'TargetMaze11ExploreAgents': { 'desc': 'Explorative Single Maze 11 Random Target Incomplete ',
  #   'args': {'sparse': False, 'explore': True, 'size': 11, 
  #     'random': ['Agents','Target'], 'render_mode': '3D', 'seed':42
  #   }, 'path': [1,1,0,0,1,1,1,1,0,0,1,1,0,0,3,3,0,0,1,1], 'return': 0, 
  # },
  # 'Sparse13Maze': { 'desc': 'Single Sparse Maze 13 Success', 'args': { 'seed': None,
  #   'sparse': True, 'explore': False, 'size': 13, 'random': [], 'render_mode': '3D'},
  #   'path': [1,1,1,1,0,0,3,3,0,0,1,1,0,0,1,1,2,2,2,2,1,1,1,1,0,0,3,3,0,0,0,0,0,0,1,1], 'return': 14, 
  # }, 
  # '13AgentMaze': { 'desc': 'Single Dense Maze 13 Random Agent Success',
  #   'args': {'sparse': False, 'explore': False, 'size': 13, 
  #     'random': ['Agent'], 'render_mode': '2D', 'seed':42
  #   }, 'path': [1,1,1,1,0,0,3,3,0,0,1,1,0,0,1,1], 'return': 34, 
  # }, 
  # 'MazeTargetSparse13Explore': { 'desc': 'Explorative Single Maze 13 Complete',
  #   'args': {'sparse': True, 'explore': True, 'size': 13, 
  #     'random': ['Target'], 'render_mode': '2D', 'seed':42
  #   }, 'path': None, 'return': -100, 
  # }, 
  # 'MazeSparse15': { 'desc': 'Single Sparse Maze 15 Success', 'args': { 'seed': None,
  #   'sparse': True, 'explore': False, 'size': 15, 'random': [], 'render_mode': '3D'},
  #   'path': [1,1,0,0,3,3,0,0,1,1,1,1,1,1,1,1,2,2,1,1,2,2,1,1,0,0,0,0,3,3,0,0,3,3,0,0,0,0,1,1,1,1,0,0,], 'return': 6,
  # }, 
  # 'Sparse15MazesAgents': { 'desc': 'Random Sparse Maze 15 Incomplete',
  #   'args': {'sparse': True, 'explore': False, 'size': 15, 
  #     'random': ['Agents','Layouts'], 'render_mode': '2D', 'seed':42
  #   }, 'path': [1,1,0,0,3,3,0,0,1,1,1,1,1,1,1,1,2,2,1,1,2,2,1,1,0,0,0,0,3,3,0,0,3,3,0,0,0,0,1,1,1,1,0,0,], 'return': 0,
  # }, 
  # 'AgentsExploreMaze15Target': { 'desc': 'Explorative Single Maze 15 Random Agent  Incomplete',
  #   'args': {'sparse': False, 'explore': True, 'size': 15, 
  #     'random': ['Agents','Target'], 'render_mode': '3D', 'seed':42
  #   }, 'path': [1,1,0,0,3,3,0,0,1,1,1,1,1,1,1,1,2,2,1,1,2,2,1,1,0,0,0,0,3,3,0,0,3,3,0,0,0,0,1,1,1,1,0,0,], 'return': 0,
  # }, 
}

print('Running tests...')
for name, test in TEST.items():
  print(f"\t• {test['desc']}")
  print(hyphi_gym.named(name))
  env = Monitor(gym.make(**hyphi_gym.named(name), render_mode='rgb_array'), record_video=render)
  obs, _ = env.reset(); R = []
  for a in [env.action_space.sample() for _ in range(env.spec.max_episode_steps or 100)]:
    observation, reward, terminated, truncated, info = env.step(a); R.append(reward)
    if terminated or truncated: break; #  print(info)       
  if render: env.save_video(f'test/video/{name}-{test["args"]["render_mode"]}.mp4')

  assert False
  env:Union[Monitor, gym.Env] = Monitor(gym.make(**hyphi_gym.named(name), seed=test['args']['seed'], render_mode=test['args']['render_mode']), record_video=render)
  assert False
  assert env.spec is not None; 
  for k,v in env.spec.kwargs.items(): assert test['args'][k]==v, f"Value at {k} shold be {test['args'][k]}, not {v}"
  obs, _ = env.reset(); R, frame_buffer = [], []  
  with stdout_redirected(f'test/text/{env.name}.txt'): [print(f"{l}") for l in env.to_ascii(env.board)]
  assert all(obs == env.reset()[0]) is not any(['s' in r for r in env.random]), 'Random Reset Failed'
  # assert all(obs == env.layout.flat) is not any(['s' in r for r in env.random]), 'Random Reset Failed'
  Image.fromarray(env.render()).save(f"test/image/{env.name}.png")
  for a in test['path'] or [env.action_space.sample() for _ in range(env.spec.max_episode_steps or 100)]:
    observation, reward, terminated, truncated, info = env.step(a); R.append(reward)
    if terminated or truncated: break; #  print(info)       
  if env.spec.kwargs['sparse']: assert R[-1] == test['return'], f"Sparse reward was {R[-1]} not {test['return']}"
  else: assert sum(R) == test['return'], f"Max reward was {sum(R)} not {test['return']}"
  env.heatmap(lambda e,s,a,r: r, (-50, 50)).savefig(f'test/heatmap/{name}.png')
  if render: env.save_video(f'test/video/{env.name}-{test["args"]["render_mode"]}.mp4')

ERROR = ['Maze10', 'HolesDense', 'Mazes19']; fails = 0
for name in ERROR:
  try:
    env = gym.make(**hyphi_gym.named(name))
  except(AssertionError): fails += 1
assert len(ERROR) == fails

print("Passed all tests")
