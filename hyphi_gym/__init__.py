from hyphi_gym.envs import *; from hyphi_gym.wrappers import Monitor
import re; from functools import reduce
from gymnasium.envs.registration import register

def register_envs():
  register(id="HoleyGrid", entry_point="hyphi_gym.envs:HoleyGrid")
  register(id="HoleyPlane", entry_point="hyphi_gym.envs:HoleyPlane")
  register(id="GridMaze", entry_point="hyphi_gym.envs:GridMaze") 
  register(id="PointMaze", entry_point="hyphi_gym.envs:PointMaze")
  register(id="FlatGrid", entry_point="hyphi_gym.envs:FlatGrid")
  register(id="Fetch", entry_point="hyphi_gym.envs:Fetch")

def named(name):
  """Enviroment creation helper, trasforms string name to make arguments.
  Usage: `gym.make(hyphi_gym.named(name))`
  Supported Envs: Any Sized Grid Mazes and and Holey Grids
  Supported Options: Sparse, Explore, Random-layout, -target, and -agent placement"""      
  level = {}; random = []
  if 'Maze' in name: 
    if 'Mazes' in name: random.append('Layouts')
    id = 'PointMaze' if 'Point' in name else 'GridMaze'
    size = int(re.findall(r'\d+', name)[0]); level= {'id':id, 'size': size}
    name = reduce(lambda n,r: n.replace(r,''), [str(size),'Points','Point','Mazes','Maze'], name)
  if 'Holey' in name:
    if 'Grids' in name or 'Planes' in name: random.append('Layouts')
    id = 'HoleyGrid' if 'Grid' in name else 'HoleyPlane'
    if len(s:=re.findall(r'\d+', name)): level = int(s[0])
    else: level ='Shift' if 'Shift' in name else 'Train'
    level = {'id': id, 'level': level}
    name = reduce(lambda n,r: n.replace(r,''), ['Holey','Shift','Planes','Plane','Grids','Grid'], name)
  if 'FlatGrid' in name:
    id = 'FlatGrid'; size = int(re.findall(r'\d+', name)[0]); level= {'id':id, 'size': size}
    name = reduce(lambda n,r: n.replace(r,''), ['Flat','Grid'], name)
  if 'Fetch' in name:
    tasks = ['Reach']
    level = {'id': 'Fetch', 'task': ''.join([t for t in tasks if t in name])}
    name = reduce(lambda n,r: n.replace(r,''), ['Fetch', *tasks], name)
  args = {'sparse': 'Sparse' in name, 'detailed': 'Detailed' in name, 'explore': 'Explore' in name}
  name = name.replace('Sparse','').replace('Explore','').replace('Detailed','')
  random = [*random, *re.findall('[A-Z][^A-Z]*', name)]
  return {**level, **args, 'random': random, }
