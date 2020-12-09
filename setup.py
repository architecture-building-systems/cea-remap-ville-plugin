from setuptools import setup, find_packages

__author__ = "Daren Thomas"
__copyright__ = "Copyright 2020, Architecture and Building Systems - ETH Zurich"
__credits__ = ["Daren Thomas"]
__license__ = "MIT"
__version__ = "1.0.0"
__maintainer__ = "Daren Thomas"
__email__ = "cea@arch.ethz.ch"
__status__ = "Production"

setup(name='cea-remap-ville-plugin',
      version=__version__,
      description="A CEA plugin for the ReMaP / VILLE project",
      license='MIT',
      author='Architecture and Building Systems',
      author_email='cea@arch.ethz.ch',
      url='https://github.com/architecture-building-systems/cea-remap-ville-plugin',
      long_description="A CEA plugin for the ReMaP / VILLE project.",
      py_modules=[''],
      packages=find_packages(),
      package_data={},
      include_package_data=True)
