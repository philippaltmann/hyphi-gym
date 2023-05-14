# type: ignore
import os; import math; import numpy as np
import bpy; from mathutils import Vector
from hyphi_gym.envs.common.base import *
from hyphi_gym.envs.common.grid import CELL_LOOKUP
from hyphi_gym.utils import stdout_redirected
from PIL import Image

class Env3D:
  def setup3D(self, layout):
    bpy.ops.wm.read_homefile(use_empty=True)
    with stdout_redirected(): bpy.ops.import_scene.fbx(filepath=f'{os.path.dirname(__file__)}/env.fbx') 
    self.scene = bpy.context.scene # Ref scene after import for sandboxed look

    def _place_3D(x, y, t):
      if t == HOLE: return
      o = bpy.data.objects[t]
      if t in [AGENT, TARGET]: _place_3D(x, y, FIELD)
      else: o = o.copy(); bpy.context.collection.objects.link(o)
      o.location = Vector((y*.1,x*.1,-0.1 if t == ' ' else 0))

    [_place_3D(x, y, CELL_LOOKUP[cell]) for y, row in enumerate(layout) for x, cell in enumerate(row)]
    for proto in [WALL,FIELD]: bpy.data.objects[proto].hide_render = True 
    self.scene.render.engine = 'BLENDER_WORKBENCH'
    self.scene.render.resolution_x, self.scene.render.resolution_y = self.metadata['render_resolution']
    self.scene.display.shading.studio_light = 'studio.sl' # slightly darker 
    camera = bpy.data.objects.new('Camera', bpy.data.cameras.new(name='Camera'))
    bpy.context.scene.collection.objects.link(camera); self.scene.camera = camera 
    zoom = 3 * sum(layout.shape)/len(layout.shape)
    camera.location = Vector((np.array((*layout.shape, zoom)) / 4 + min(layout.shape)) / 10)
    camera.rotation_euler = math.pi * np.array((30,0,135))/180

  def update3D(self, action, mPos, Cell):
    bpy.data.objects['A'].rotation_euler = (0,0,[0.5,0,1.5,1][action] * math.pi)
    if Cell in [FIELD, TARGET]: bpy.data.objects['A'].location = Vector((mPos[0]*.1,mPos[1]*.1,0))
    if Cell == TARGET: bpy.data.objects['T'].hide_render = True  # Hide Overlap Components 
    if Cell == HOLE: bpy.data.objects['A'].hide_render = True    # Hide Overlap Components 
    
  def render3D(self, write_file=None):
    bpy.ops.render.render(); bpy.data.images['Render Result'].save_render(self.metadata['tmp'])
    if write_file is not None: bpy.data.images['Render Result'].save_render(write_file)
    return np.asarray(Image.open(self.metadata['tmp']))[:,:,:3]
  
  def reset3D(self):
    if len(self.random): self.setup3D(self.board)
    aPos = self.getpos()
    bpy.data.objects['A'].location = Vector((aPos[0]*.1,aPos[1]*.1,0))
    bpy.data.objects['A'].rotation_euler = (0,0,0)
    bpy.data.objects['T'].hide_render, bpy.data.objects['A'].hide_render = False, False
    bpy.data.objects['T'].hide_render = self.explore
