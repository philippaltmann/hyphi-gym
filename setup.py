from pathlib import Path
from setuptools import setup, find_packages

rendering = ["bpy==3.6.0"]
simulation = ["mujoco==2.3.3"]
plotting = ["matplotlib>=3.5"]

setup(
  name="hyphi_gym", version="0.8",
  description="Gymnasium benchmark suite for evaluating robustness and multi-task performance of reinforcement learning algorithms in various discrete and continuous environments.",
  url="https://github.com/philippaltmann/hyphi-gym", author_email="philipp@hyphi.co", license="MIT",
  keywords="reinforcement-learning robustness benchmark gymnasium gridworld maze mujoco openai gym",
  long_description=(Path(__file__).parent / "README.md").read_text(), long_description_content_type="text/markdown",
  packages=[package for package in find_packages() if package.startswith("hyphi_gym")],
  package_data={"hyphi_gym": ["assets/*", "assets/Agent/*", "assets/fetch/**", "assets/fetch/mesh/*", "assets/fetch/textures/*"]},
  install_requires=[
    "gymnasium>=0.29",  # Bump to 1.0 once available 
    'numpy>=1.20',      # Utils
    "Pillow>=9.5",      # Save images
    "moviepy>=1.0",     # Save videos
  ]+rendering+simulation,
  extras_require={
    "rendering": rendering,
    "simulation": simulation,
    "plotting": plotting,
    "all": rendering + simulation + plotting,
    "docs": ['sphinx', 'furo', 'sphinx_autoapi', 'myst_parser'],
    "tests": [ "pytest", "black"],
  },
  python_requires=">=3.8",
  entry_points={
      "gymnasium.envs": ["__root__ = hyphi_gym.__init__:register_envs"]
  }
)
