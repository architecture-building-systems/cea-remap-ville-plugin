"""
Initialize a new scenario based on a old scenario.
"""
import os
import shutil

import cea.config
import cea.inputlocator
from cea.utilities.dbf import dbf_to_dataframe, dataframe_to_dbf
from remap_ville_plugin.create_technology_database import copy_file
import remap_ville_plugin.urban_transformation as urban_transformation

__author__ = "Shanshan Hsieh"
__copyright__ = "Copyright 2021, Architecture and Building Systems - ETH Zurich"
__credits__ = ["Shanshan Hsieh"]
__license__ = "MIT"
__version__ = "0.1"
__maintainer__ = "Shanshan Hsieh"
__email__ = "cea@arch.ethz.ch"
__status__ = "Production"


def main(config):
    district_archetype = config.remap_ville_scenarios.district_archetype
    year = config.remap_ville_scenarios.year
    urban_development_scenario = config.remap_ville_scenarios.urban_development_scenario

    old_scenario_name = config.scenario_name
    old_locator = cea.inputlocator.InputLocator(scenario=config.scenario, plugins=config.plugins)
    new_scenario_name = f"{year}_{urban_development_scenario}"
    config.scenario_name = new_scenario_name
    new_locator = cea.inputlocator.InputLocator(scenario=config.scenario, plugins=config.plugins)
    initialize_input_folder(config, new_locator)
    print(f"Initializing new scenario: {new_scenario_name} base on {old_scenario_name}")
    # copy files
    copy_folder(old_locator.get_building_geometry_folder(), new_locator.get_building_geometry_folder())
    copy_folder(old_locator.get_terrain_folder(), new_locator.get_terrain_folder())
    os.mkdir(new_locator.get_building_properties_folder())
    copy_file(old_locator.get_building_typology(), new_locator.get_building_typology())
    copy_file(old_locator.get_building_architecture(), new_locator.get_building_architecture())

    # Urban Transformation
    print(f"Transforming scenario according to urban development scenario... {urban_development_scenario}")
    urban_transformation.main(config)

    # modify dbf
    print(f"Modifying building properties in... {year}")
    # copy air conditioning and supply system
    copy_file(old_locator.get_building_air_conditioning(), new_locator.get_building_air_conditioning())
    copy_file(old_locator.get_building_supply(), new_locator.get_building_supply())
    typology_df = dbf_to_dataframe(new_locator.get_building_typology()).set_index('Name')
    MULTI_RES_2040, ORIG_RES, OTHER_USES = get_building_lists_by_use(typology_df)
    # update air-conditioning according to use
    airconditioning_df = dbf_to_dataframe(new_locator.get_building_air_conditioning()).set_index('Name')
    airconditioning_df = update_air_conditioning_dbf(MULTI_RES_2040, ORIG_RES, OTHER_USES, airconditioning_df)
    # update supply system according to use
    supply_df = dbf_to_dataframe(new_locator.get_building_supply()).set_index('Name')
    supply_df.loc[:, 'type_cs'] = 'SUPPLY_COOLING_AS1'
    # update architecture for MULTI_RES_2040
    architecture_df = dbf_to_dataframe(new_locator.get_building_architecture()).set_index('Name')
    architecture_df = update_architecture_dbf(MULTI_RES_2040, architecture_df)
    save_dbfs(airconditioning_df, new_locator.get_building_air_conditioning())
    save_dbfs(supply_df, new_locator.get_building_supply())
    save_dbfs(architecture_df, new_locator.get_building_architecture())

    # TODO: update use_types in the technology folder!!!

    return

def save_dbfs(dbf, path_to_save_dbf):
    print('saving...', path_to_save_dbf)
    dbf_df = dbf.copy()
    dbf_df.insert(0, 'Name', dbf_df.index)
    dataframe_to_dbf(dbf_df, path_to_save_dbf)

def update_architecture_dbf(MULTI_RES_2040, architecture_df):
    MULTI_RES_2040_architecture = [0.82, 0, 0.82, 0.82, 0, 0.4, 0.4, 0.4, 0.4,
                                   'CONSTRUCTION_AS3', 'TIGHTNESS_AS2', 'FLOOR_AS3', 'WALL_AS7', 'FLOOR_AS6',
                                   'ROOF_AS4', 'WALL_AS5', 'WINDOW_AS4', 'SHADING_AS1']
    architecture_df.loc[MULTI_RES_2040] = MULTI_RES_2040_architecture
    return architecture_df

def update_air_conditioning_dbf(MULTI_RES_2040, ORIG_RES, OTHER_USES, airconditioning_df):
    # MULTI_RES_2040
    airconditioning_df.loc[MULTI_RES_2040, 'type_cs'] = 'HVAC_COOLING_AS5'
    airconditioning_df.loc[MULTI_RES_2040, 'type_ctrl'] = 'HVAC_CONTROLLER_AS2'
    airconditioning_df.loc[MULTI_RES_2040, 'type_vent'] = 'HVAC_VENTILATION_AS0'
    # ORIG_RES
    airconditioning_df.loc[ORIG_RES, 'type_cs'] = 'HVAC_COOLING_AS2'
    airconditioning_df.loc[ORIG_RES, 'type_ctrl'] = 'HVAC_CONTROLLER_AS2'
    airconditioning_df.loc[ORIG_RES, 'type_vent'] = 'HVAC_VENTILATION_AS0'
    # OTHER_USES
    airconditioning_df.loc[OTHER_USES, 'type_cs'] = 'HVAC_COOLING_AS3'
    airconditioning_df.loc[OTHER_USES, 'type_ctrl'] = 'HVAC_CONTROLLER_AS2'
    airconditioning_df.loc[OTHER_USES, 'type_vent'] = 'HVAC_VENTILATION_AS2'
    return airconditioning_df

def get_building_lists_by_use(typology_df):
    _RES = list(typology_df[typology_df['1ST_USE'].str.contains("_RES")].index)
    print('_RES', len(_RES))
    MULTI_RES_2040 = list(typology_df[typology_df['1ST_USE'] == 'MULTI_RES_2040'].index)
    print('MULTI_RES_2040', len(MULTI_RES_2040))
    ORIG_RES = list(set(_RES) - set(MULTI_RES_2040))
    print('ORIG_RES', len(ORIG_RES))
    OTHER_USES = list(set(typology_df.index) - set(_RES))
    print('OTHER_USES', len(OTHER_USES))
    return MULTI_RES_2040, ORIG_RES, OTHER_USES


def copy_folder(src, dst):
    print(f" - {dst}")
    shutil.copytree(src, dst)


def initialize_input_folder(config, new_locator):
    if os.path.exists(config.scenario):
        raise ValueError((f"{config.scenario} exists, please remove the folder before proceeding."))
        # shutil.rmtree(config.scenario)
    os.mkdir(config.scenario)
    os.mkdir(new_locator.get_input_folder())
    return


if __name__ == "__main__":
    main(cea.config.Configuration())