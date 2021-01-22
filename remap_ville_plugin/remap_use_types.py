"""
Map the fields in ``typology.df`` to the new values, based on

- district-archetype
- year
- urban-development-scenario

This updates the 1ST_USE, 2ND_USE and 3RD_USE fields as well as the ratios of those uses.

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
    district_archetype = config.remap_ville_scenarios.district_archetype
    year = config.remap_ville_scenarios.year
    urban_development_scenario = config.remap_ville_scenarios.urban_development_scenario

    locator = cea.inputlocator.InputLocator(scenario=config.scenario, plugins=config.plugins)

    mapping = read_mapping(district_archetype, year, urban_development_scenario)
    typology = cea.utilities.dbf.dbf_to_dataframe(locator.get_building_typology())

    column_names = ("1ST_USE", "1ST_USE_R", "2ND_USE", "2ND_USE_R", "3RD_USE", "3RD_USE_R")

    for index, row in typology.iterrows():
        building = row.Name

        old_values = [row[cn] for cn in column_names]
        new_values = do_mapping(mapping, old_values)
        print(f"Updating {building}, {old_values} => {new_values}")

        for cn, v in zip(column_names, new_values):
            typology.loc[index, cn] = v

    cea.utilities.dbf.dataframe_to_dbf(typology, locator.get_building_typology())


def do_mapping(mapping, old_values):
    for pattern, rule in mapping:
        if match_pattern(pattern, old_values):
            return rule
    # none of the patterns matched...
    return old_values


def match_pattern(pattern, values):
    """
    pattern is a six-tuple: (use1, ratio1, use2, ratio2, use3, ratio3)
    usex can either be a valid use type, NONE or ANY
    ratiox can either be a real number between 0.0 and 1.0 or ANY
    :param pattern:
    :param values:
    :return:
    """
    pattern_uses = pattern[0:6:2]
    pattern_ratios = pattern[1:6:2]
    value_uses = values[0:6:2]
    value_ratios = values[1:6:2]
    for p_use, p_ratio, v_use, v_ratio in zip(pattern_uses, pattern_ratios, value_uses, value_ratios):
        if p_use != "ANY" and p_use != v_use:
            # use doesn't match
            return False
        if p_ratio != "ANY" and float(p_ratio) > float(v_ratio):
            return False
    return True


def read_mapping(district_archetype, year, urban_development_scenario):
    """
    Create a list version of the mapping because it's easier to use...
    :return:
    """
    worksheet = f"{district_archetype}_{year}_{urban_development_scenario}"
    print(f"Reading mappings from worksheet {worksheet}")
    mapping_df = pd.read_excel(os.path.join(os.path.dirname(__file__), "mapping_USE_TYPE.xlsx"), sheet_name=worksheet)
    mapping = []
    for _, row in mapping_df.iterrows():
        pattern = (row["1ST_USE"], row["1ST_USE_RATIO"], row["2ND_USE"], row["2ND_USE_RATIO"], row["3RD_USE"],
                   row["3RD_USE_RATIO"])
        rule = (row["UPDATED_1ST_USE"], row["UPDATED_1ST_USE_RATIO"], row["UPDATED_2ND_USE"],
                row["UPDATED_2ND_USE_RATIO"], row["UPDATED_3RD_USE"], row["UPDATED_3RD_USE_RATIO"])
        mapping.append((pattern, rule))
    return mapping


if __name__ == "__main__":
    main(cea.config.Configuration())