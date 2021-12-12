"""
Initialize a new scenario based on a old scenario.
"""
import os
import pandas as pd
from pathlib import Path

import cea.config
import cea.inputlocator
from cea.utilities.dbf import dbf_to_dataframe, dataframe_to_dbf
from remap_ville_plugin.create_technology_database import copy_file
import remap_ville_plugin.urban_transformation as urban_transformation
import remap_ville_plugin.urban_transformation_sequential as sequential_urban_transformation
from remap_ville_plugin.utilities import copy_folder
from remap_ville_plugin.create_technology_database import create_input_technology_folder, update_indoor_comfort

__author__ = "Shanshan Hsieh"
__copyright__ = "Copyright 2021, Architecture and Building Systems - ETH Zurich"
__credits__ = ["Shanshan Hsieh"]
__license__ = "MIT"
__version__ = "0.1"
__maintainer__ = "Shanshan Hsieh"
__email__ = "cea@arch.ethz.ch"
__status__ = "Production"


def main(config):
    ## 0. Get case study inputs
    path_to_case_study_inputs = os.path.join(config.scenario, "case_study_inputs.xlsx")
    check_case_study_inputs(config, path_to_case_study_inputs)
    worksheet = f"{config.remap_ville_scenarios.district_archetype}_{config.remap_ville_scenarios.urban_development_scenario}"
    case_study_inputs_df = pd.read_excel(path_to_case_study_inputs, sheet_name=worksheet).set_index('year')

    scenario_locator_sequences = {}
    initial_scenario_name = config.scenario_name
    scenario_locator_sequences[initial_scenario_name] = cea.inputlocator.InputLocator(scenario=config.scenario, plugins=config.plugins)
    ## 1. Initialize new scenario
    # decide year order, the year with larger population is the endstate
    [year_endstate, year_intermidiate] = case_study_inputs_df['additional_population'].sort_values(ascending=False).index
    config.remap_ville_scenarios.year = year_endstate
    config, new_locator, old_locator, urban_development_scenario, year = initialize_new_scenario(config)
    endstate_scenario_name = config.scenario_name
    scenario_locator_sequences[endstate_scenario_name] = new_locator

    ## 2. Urban Transformation (end-state)
    print(f"Transforming scenario according to urban development scenario... {urban_development_scenario}")
    urban_transformation.main(config, case_study_inputs_df)

    # modify dbf
    modify_building_properties_after_urban_transformation(new_locator, year)
    create_technology_folder(config, new_locator, urban_development_scenario, year)
    update_indoor_comfort('SQ', new_locator)

    ## 3. Urban Transformation (2040)
    config.scenario_name = str(year_endstate)+'_'+urban_development_scenario
    old_locator = cea.inputlocator.InputLocator(scenario=config.scenario, plugins=config.plugins)
    new_scenario_name = str(year_intermidiate)+'_'+urban_development_scenario
    config.scenario_name = new_scenario_name
    config.remap_ville_scenarios.year = year_intermidiate
    new_locator = cea.inputlocator.InputLocator(scenario=config.scenario, plugins=config.plugins)
    initialize_input_folder(config, new_locator)
    copy_inputs_folder_content(old_locator, new_locator)
    case_study_inputs = case_study_inputs_df.loc[int(year_intermidiate)]
    sequential_urban_transformation.main(config, new_locator, scenario_locator_sequences, case_study_inputs, int(year_intermidiate), year_endstate)

    # TODO: update use_types in the technology folder!!!


def create_technology_folder(config, new_locator, urban_development_scenario, year):
    district_archetype = config.remap_ville_scenarios.district_archetype
    folder_name = f"{district_archetype}_{year}_{urban_development_scenario}"
    print(f'Creating technology folder from database: {folder_name}')
    create_input_technology_folder(folder_name, new_locator)


def modify_building_properties_after_urban_transformation(new_locator, year):
    print(f"Modifying building properties in... {year}")
    typology_df = dbf_to_dataframe(new_locator.get_building_typology()).set_index('Name')
    # update air-conditioning according to use
    MULTI_RES_2040, ORIG_RES, OTHER_USES = get_building_lists_by_use(typology_df)
    airconditioning_df = dbf_to_dataframe(new_locator.get_building_air_conditioning()).set_index('Name')
    airconditioning_df = update_air_conditioning_dbf(MULTI_RES_2040, ORIG_RES, OTHER_USES, airconditioning_df)
    save_dbfs(airconditioning_df, new_locator.get_building_air_conditioning())
    # update supply system according to use
    supply_df = dbf_to_dataframe(new_locator.get_building_supply()).set_index('Name')
    supply_df.loc[:, 'type_cs'] = 'SUPPLY_COOLING_AS1'
    save_dbfs(supply_df, new_locator.get_building_supply())
    # update architecture for MULTI_RES_2040
    architecture_df = dbf_to_dataframe(new_locator.get_building_architecture()).set_index('Name')
    architecture_df = update_architecture_dbf(MULTI_RES_2040, architecture_df)
    save_dbfs(architecture_df, new_locator.get_building_architecture())


def check_case_study_inputs(config, path_to_case_study_inputs):
    if not os.path.exists(path_to_case_study_inputs):
        path_to_template = Path(os.path.join(os.path.dirname(__file__), "case_study_inputs_template.xlsx"))
        output = os.path.join(config.scenario, "case_study_inputs_template.xlsx")
        copy_file(path_to_template, output)
        raise ValueError(
            'Please provide `case_study_inputs.xlsx` by modifying `case_study_inputs_template.xlsx` in {}'.format(
                config.scenario))


def initialize_new_scenario(config):
    old_scenario_name = config.scenario_name
    old_locator = cea.inputlocator.InputLocator(scenario=config.scenario, plugins=config.plugins)
    config, new_scenario_name, urban_development_scenario, year = update_config(config)
    new_locator = cea.inputlocator.InputLocator(scenario=config.scenario, plugins=config.plugins)
    initialize_input_folder(config, new_locator)
    print(f"Initializing new scenario: {new_scenario_name} base on {old_scenario_name}")
    copy_inputs_folder_content(old_locator, new_locator)
    return config, new_locator, old_locator, urban_development_scenario, year


def copy_inputs_folder_content(old_locator, new_locator):
    print(f'copying from...{old_locator.get_input_folder()}')
    # copy files from old_locator to new_locator
    copy_folder(old_locator.get_building_geometry_folder(), new_locator.get_building_geometry_folder())
    copy_folder(old_locator.get_terrain_folder(), new_locator.get_terrain_folder())
    os.mkdir(new_locator.get_building_properties_folder())
    copy_file(old_locator.get_building_typology(), new_locator.get_building_typology())
    copy_file(old_locator.get_building_architecture(), new_locator.get_building_architecture())
    copy_file(old_locator.get_building_air_conditioning(), new_locator.get_building_air_conditioning())
    copy_file(old_locator.get_building_supply(), new_locator.get_building_supply())


def update_config(config):
    year = config.remap_ville_scenarios.year
    urban_development_scenario = config.remap_ville_scenarios.urban_development_scenario
    new_scenario_name = f"{year}_{urban_development_scenario}"
    config.scenario_name = new_scenario_name
    return config, new_scenario_name, urban_development_scenario, year


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
    MULTI_RES_2040 = list(typology_df[typology_df['1ST_USE'] == 'MULTI_RES_2040'].index)
    ORIG_RES = list(set(_RES) - set(MULTI_RES_2040))
    OTHER_USES = list(set(typology_df.index) - set(_RES))
    print(f'RES: {len(_RES)}; MULTI_RES_2040: {len(MULTI_RES_2040)}; ORIG_RES: {len(ORIG_RES)}')
    print(f'OTHER_USES: {len(OTHER_USES)}')
    return MULTI_RES_2040, ORIG_RES, OTHER_USES


def initialize_input_folder(config, new_locator):
    print(f'\nInitializing {config.scenario}')
    if os.path.exists(config.scenario):
        raise ValueError(f"{config.scenario} exists, please remove the folder before proceeding.")
        # shutil.rmtree(config.scenario)
    os.mkdir(config.scenario)
    os.mkdir(new_locator.get_input_folder())
    return


if __name__ == "__main__":
    main(cea.config.Configuration())
