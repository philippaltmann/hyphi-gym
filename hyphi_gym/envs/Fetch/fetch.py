from os import path; import numpy as np
from hyphi_gym.envs.common import Robot, get_xml

TASKS = ['Reach'] 

class Fetch(Robot):
  def __init__(self, task:str='Reach', render_mode=None, **kwargs):
    assert task in TASKS; self.task = task; self.base_xml = get_xml(task)
    layout = {'agent': np.array([0,0,0]), 'target': np.array([0,0,0])}
    if len(kwargs['random']): kwargs = {**kwargs, 'continue_task': False}
    Robot.__init__(self, render_mode=render_mode, **kwargs)
