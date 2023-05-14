from pathlib import Path
from setuptools import setup

setup(
  name='hyphi-gym',
  long_description=(Path(__file__).parent / "README.md").read_text(),
  long_description_content_type="text/markdown",
)
