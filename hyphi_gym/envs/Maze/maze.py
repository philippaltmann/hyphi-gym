from hyphi_gym.envs.common.grid import *

LEVELS = {
  'Maze7': ( (-100,42), # Maze 7x7 
    ['#######', 
     '#    T#', 
     '# ### #', 
     '# #   #', 
     '### # #', 
     '#A  # #', 
     '#######']),
  'Maze9': ( (-100,34), # Maze 9x9 (Seed 1)
    ['#########', 
     '#      T#', 
     '# ##### #', 
     '# #     #', 
     '# # ### #', 
     '# #   # #', 
     '##### # #', 
     '#A    # #',
     '#########']),
  'Maze11': ( (-100,30), #Maze11x11 (Seed 0)
    ['###########', 
     '#        T#', 
     '# ##### ###', 
     '#     #   #', 
     '##### ### #', 
     '#     #   #', 
     '# ##### # #', 
     '# #     # #', 
     '### ##### #', 
     '#A  #     #',
     '###########']),
  'Maze13': ( (-100,14), #Maze13x13 (seed 4)
    ['#############', 
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
     '#############']), 
  'Maze15': ( (-100,6), # Maze 15x15
    ['###############', 
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
     '###############']) 
}

class Maze(Grid):
  """Gridworld Maze Environment based on hyphi Grid.
  :param size: (≤15, guarantee solvability within 100 steps (worst for 15: 96))
  :param random: optional list of features to be stochastic
    supporting layout, agent-, and target-placement"""
  def __init__(self, size, random=[], **kwargs): # Could add Mazes for multiple layouts
    assert size % 2 == 1 and 15 >= size >= 3; self.size = size; self._name = f'Maze{size}' 
    if 'Layouts' not in random: self.reward_range, layout = LEVELS[f"Maze{size}"] 
    else: self.reward_range, layout = (-100,50-(size-1)**2/2-2), None
    # Worst case / Maximum steps: assuming 1/2 fields of size square are walls, 
    # remaining fields need to be visited, excluding agent and target itself
    super(Maze, self).__init__((size,size), layout, random=random, **kwargs)

  def _generate(self):
    """Mazes are generated by Kruskal's algorithm and range in size from 3x3 """
    APOS, GPOS = (self.size[0]-2,1), (1,self.size[1]-2)
    maze, visited = np.full(self.size, CELL_LOOKUP.index(WALL)), []
    #Carve out" empty spaces in the maze at x, y and then recursively move to neighboring unvisited spaces
    def visit(pos, d=2):
      maze[pos] = CELL_LOOKUP.index(FIELD); visited.append(pos)
      while True:
        actions = [a for a, p in self.iterate_actions(
            pos, d, lambda a, p, n: self.action_possible(a, p, n)
          ).items() if p not in visited]
        if not len(actions): return
        action = self.np_random.choice(actions); intermediate = self.newpos(pos,action)
        maze[intermediate] = CELL_LOOKUP.index(FIELD); t = visit(self.newpos(intermediate,action),d)
    visit(APOS); maze[APOS],maze[GPOS] = CELL_LOOKUP.index(AGENT), CELL_LOOKUP.index(TARGET);
    if self._to_ascii(maze) in LEVELS[f'Maze{self.size[0]}']: return self._generate()
    return maze
    