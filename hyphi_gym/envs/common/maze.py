from hyphi_gym.envs.common.board import *
import numpy as np; import math

LEVELS = {
  'Maze7': ['#######', # Maze 7x7 
            '#    T#', 
            '# ### #', 
            '# #   #', 
            '### # #', 
            '#A  # #', 
            '#######'],
  'Maze9': ['#########',  # Maze 9x9 (Seed 1)
            '#      T#', 
            '# ##### #', 
            '# #     #', 
            '# # ### #', 
            '# #   # #', 
            '##### # #', 
            '#A    # #',
            '#########'],
  'Maze11':  ['###########', #Maze11x11 (Seed 0)
              '#        T#', 
              '# ##### ###', 
              '#     #   #', 
              '##### ### #', 
              '#     #   #', 
              '# ##### # #', 
              '# #     # #', 
              '### ##### #', 
              '#A  #     #',
              '###########'],
  'Maze13':  ['#############',  #Maze13x13 (seed 4)
              '#     #    T#', 
              '# ##### # # #', 
              '# #     # # #', 
              '# # ##### # #', 
              '#   #   # # #', 
              '# ### # # ###', 
              '# #   # #   #', 
              '# # ### ### #', 
              '# #   #     #', 
              '##### ##### #', 
              '#A    #     #', 
              '#############'], 
  'Maze15':  ['###############', 
              '#     #      T#', 
              '# ### # ##### #', 
              '#   # # #     #', 
              '### # # # ### #', 
              '# # # # # #   #', 
              '# # # # # ### #', 
              '#   #   #   # #', 
              '########### ###', 
              '#         #   #', 
              '# ####### ### #', 
              '#   #   #   # #', 
              '### # ##### # #', 
              '#A  #         #', 
              '###############']
}

class Maze(Board):
  # Board
  """Gridworld Maze Environment based on hyphi Grid.
  :param size: (≤15, guarantee solvability within 100 steps (worst for 15: 96))
  :param random: optional list of features to be stochastic
    supporting layout, agent-, and target-placement"""
  def __init__(self, size, random=[], **kwargs): 
    assert size % 2 == 1 and 15 >= size >= 3; self.size = size; self._name = f'Maze{size}' 
    layout = None if 'Layouts' in random else LEVELS[f"Maze{size}"] 
    max_path = (self.size-1)**2/2-2; max_steps = math.ceil(max_path * self.step_scale * 1.2 / 100) * 100
    Board.__init__(self, size=(size,size), layout=layout, random=random, RADD=['Layouts'], max_episode_steps=max_steps, **kwargs)

  def _generate(self):
    """Generate random mazes of `self.size` using `Kruskal's algorithm. 
    Generated mazes are forced to difer the static configurations above."""
    APOS, GPOS = (self.size[0]-2,1), (1,self.size[1]-2)
    maze, visited = np.full(self.size, CELLS[WALL]), []
    def visit(position, d=2): # Carve out" empty spaces in the maze 
      maze[position] = CELLS[FIELD]; visited.append(position)
      while True: # recursively move to neighboring unvisited spaces
        actions = [a for a, p in self.iterate_actions(
            position, d, lambda act, pos, n: self.action_possible(act, pos, n)
          ).items() if p not in visited]
        if not len(actions): return
        action = self.np_random.choice(actions); intermediate = self.newpos(position,action)
        maze[intermediate] = CELLS[FIELD]; t = visit(self.newpos(intermediate,action),d)
    visit(APOS); maze[APOS],maze[GPOS] = CELLS[AGENT], CELLS[TARGET];
    if self.ascii(maze) in LEVELS[f'Maze{self.size[0]}']: return self._generate()
    return maze
