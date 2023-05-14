from pathlib import Path
from setuptools import find_packages, setup

setup(
  name='hyphi-gym',
  version='1.0',
  description='Gymnasium benchmark suite for evaluating robustness and multi-task performance of reinforcement learning algorithms in various discrete and continuous environments.',
  author='Philipp Altmann',
  author_email='philipp@hyphi.co',
  url='https://github.com/philippaltmann/hyphi-gym',
  license='MIT',
  long_description=(Path(__file__).parent / "README.md").read_text(),
  long_description_content_type="text/markdown",
  packages=find_packages(exclude=["test"]),
  install_requires=[
    'gymnasium==0.28.1',
    'bpy==3.4.0', # Render 3D
    'pygame', # For Drawing 2D
    'mathutils', 
    'Pillow', 
    'matplotlib', # For heatmaps
    'moviepy',
  ]
)
