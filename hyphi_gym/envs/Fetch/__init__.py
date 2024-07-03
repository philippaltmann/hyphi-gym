from hyphi_gym.common.robot import Robot
from hyphi_gym.common.simulation import get_xml
TASKS = ['Reach']

class Fetch(Robot):
  """Control the 7-DoF [Fetch Mobile Manipulator](https://fetchrobotics.com) robot arm to reach the target by moving the gripper, sliding the object, or picking and placing the object within 100, 200 and 400 steps respectively."""
  def __init__(self, task:str='Reach', render_mode=None, **kwargs):
    """:property task: The environment """
    assert task in TASKS; self.task = task; self.base_xml = get_xml(task)
    self._name = f'Fetch{task}' 
    if len(kwargs['random']): kwargs = {**kwargs, 'continue_task': False}
    Robot.__init__(self, render_mode=render_mode, **kwargs)
