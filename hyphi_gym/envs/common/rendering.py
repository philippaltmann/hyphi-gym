import os; import math; import numpy as np
from mathutils import Vector; from PIL import Image
from hyphi_gym.utils import stdout_redirected
from hyphi_gym.envs.common.board import *
with stdout_redirected(): import bpy;
import tempfile; import time 

SCENE = f'{os.path.dirname(__file__)}/../../assets/env.blend'

class Rendering(Board):
  def __init__(self, grid):
    """Init blender rendering using `grid` mode by default. If `self.layout` 
    is not set upon init, use `setup3D(layout)` once available. """
    with tempfile.TemporaryDirectory() as tmp: self.metadata['tmp'] = tmp
    ox, oy = (s/20+.25 for s in self.size)
    self._pos = lambda x,y,t: Vector((x*.1-ox,y*.1-oy, -.2 if t == ' ' else -.1))
    # self._pos = lambda x,y,t: Vector((x*.1-.05,y*.1-.05, -.01 if t == ' ' else .09))
    if self.layout is not None: self.setup3D(self.layout)

  def setup3D(self, layout:np.ndarray): 
    with stdout_redirected(): bpy.ops.wm.open_mainfile(filepath=SCENE)                    # type: ignore
    self.scene = bpy.context.scene                                                        # type: ignore
    
    def _place_3D(x, y, t):
      if t == HOLE: return
      o = bpy.data.objects[t]                                                             # type: ignore
      if t in [AGENT, TARGET]: _place_3D(x, y, FIELD)
      else: o = o.copy(); bpy.context.collection.objects.link(o)                          # type: ignore
      o.location = self._pos(x,y,t)

    [_place_3D(x, y, CHARS[cell]) for x, row in enumerate(layout) for y, cell in enumerate(row)]
    for proto in [WALL,FIELD]: bpy.data.objects[proto].hide_render = True                 # type: ignore
    self.scene.render.resolution_x, self.scene.render.resolution_y = self.metadata['render_resolution']
    camera = bpy.data.objects['Camera']                                                   # type: ignore
    cam = { 7: Vector((0.58,0.59,1.18)),  8: Vector((0.67,0.70,1.35)),  9: Vector((0.82,0.83,1.52)),
           11: Vector((1.05,1.07,1.82)), 13: Vector((1.28,1.30,2.15)), 15: Vector((1.52,1.54,2.50))}
    camera.location = cam[int(sum(self.size)/len(self.size))]

  def reset_world(self):
    """Reset simulation and reposition agent and target to respective `i_pos`"""
    if self.layout is None: self.setup3D(self.board) #; self.load_world()
    bpy.data.objects['A'].location = self._pos(*[ *self.getpos(), 'A'])                    # type: ignore
    bpy.data.objects['A'].rotation_euler = (0,0,0)      # Rotate towards action            # type: ignore
    bpy.data.objects['T'].hide_render = self.explore    # Unhide target                    # type: ignore
    bpy.data.objects['A'].hide_render =  False          # Unhide agent                     # type: ignore
  
  def update_world(self, action, mPos, Cell):
    bpy.data.objects['A'].rotation_euler = (0,0,[0.5,0,1.5,1][action] * math.pi)           # type: ignore
    if Cell in [FIELD, TARGET]: bpy.data.objects['A'].location = self._pos(*[ *mPos, 'A']) # type: ignore
    if Cell == TARGET: bpy.data.objects['T'].hide_render = True  # Hide Overlap Components # type: ignore 
    if Cell == HOLE: bpy.data.objects['A'].hide_render = True    # Hide Overlap Components # type: ignore 
    
  def render3D(self):
    start = time.time(); bpy.ops.render.render(); print(f"Render took {time.time()-start}") # type: ignore
    bpy.data.images['Render Result'].save_render(self.metadata['tmp'])                      # type: ignore
    img = np.asarray(Image.open(self.metadata['tmp']))[:,:,:3]
    return Image.open(self.metadata['tmp']).copy() #img
