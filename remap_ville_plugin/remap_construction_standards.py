"""
Map the field ``typology.df:STANDARD`` to the new value, based on

- urban-development-scenario
- year
- district-archetype

This can be done on a new scenario, _before_ running archetypes-mapper.
"""
import os
import pandas as pd
import cea.config
import cea.inputlocator
import cea.utilities.dbf

__author__ = "Daren Thomas"
__copyright__ = "Copyright 2021, Architecture and Building Systems - ETH Zurich"
__credits__ = ["Daren Thomas"]
__license__ = "MIT"
__version__ = "0.1"
__maintainer__ = "Daren Thomas"
__email__ = "cea@arch.ethz.ch"
__status__ = "Production"


def main(config):
    locator = cea.inputlocator.InputLocator(scenario=config.scenario, plugins=config.plugins)
    mapping = pd.read_excel(os.path.join(os.path.dirname(__file__), "mapping_CONSTRUCTION_STANDARD.xlsx"))
    typology = cea.utilities.dbf.dbf_to_dataframe(locator.get_building_typology())

    print(mapping)
    print(typology)


if __name__ == "__main__":
    main(cea.config.Configuration())