"""
Runs the three ReMaP scripts (remap-copy-archetypes, remap-use-types, remap-construction-standards).
"""
import cea.api
import cea.config

__author__ = "Daren Thomas"
__copyright__ = "Copyright 2021, Architecture and Building Systems - ETH Zurich"
__credits__ = ["Daren Thomas"]
__license__ = "MIT"
__version__ = "0.1"
__maintainer__ = "Daren Thomas"
__email__ = "cea@arch.ethz.ch"
__status__ = "Production"


def main(config):
    cea.api.remap_copy_archetypes(config=config)
    cea.api.remap_use_types(config=config)
    cea.api.remap_construction_standards(config=config)


if __name__ == "__main__":
    main(cea.config.Configuration())
