from setuptools import setup, find_packages

setup(
  name='hyphi-gym',
  version='1.0',
  description='Python Distribution Utilities',
  author='Philipp Altmann',
  author_email='philipp@hyphi.co',
  url='https://gym.hyphi.co',
  packages=find_packages(include=['hyphi_gym', 'hyphi_gym.*']),
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
