import os
import shutil
from pathlib import Path
import pandas as pd
import geopandas as gpd

import cea.config
import cea.inputlocator
from cea.utilities.dbf import dbf_to_dataframe, dataframe_to_dbf

def main():
    PARAMS = {
        'path_to_initial_scenario':'C:\\Users\\shsieh\\Nextcloud\\VILLE\\Case studies\\Altstetten\\CEA_inputs\\23082021\\2020',
        'path_to_area_mapper': 'C:\\Users\\shsieh\\Nextcloud\\VILLE\\Case studies\\Altstetten\\CEA_inputs\\27082021\\area_mapper',
        'path_to_new_scenario': 'C:\\Users\\shsieh\\Nextcloud\\VILLE\\Case studies\\Altstetten\\CEA_inputs\\27082021',
        'path_to_database': 'C:\\Users\\shsieh\\Nextcloud\\VILLE\\Case studies\\Altstetten\\CEA_inputs\\27082021\\inputs_technology_2040',
        'new_scenario':'2040_BAL',
    }
    ## Initialize input folders
    path_to_initial_input_folder = os.path.join(PARAMS['path_to_initial_scenario'],'inputs')
    path_to_initial_building_prop_folder = os.path.join(path_to_initial_input_folder, 'building-properties')
    path_to_new_input_folder = os.path.join('',*[PARAMS['path_to_new_scenario'], PARAMS['new_scenario'], 'inputs'])
    path_to_new_building_prop_folder = os.path.join(path_to_new_input_folder, 'building-properties')
    if os.path.exists(path_to_new_input_folder):
        shutil.rmtree(path_to_new_input_folder)
    shutil.copytree(path_to_initial_input_folder, path_to_new_input_folder, ignore=ignore_files)
    # copy surroundings.shp and terrain.tif
    path_initial_topography = os.path.join('', *[path_to_initial_input_folder, 'topography', 'terrain.tif'])
    shutil.copy2(path_initial_topography, os.path.join(path_to_new_input_folder, 'topography'))
    path_initial_surroundings = os.path.join('', *[path_to_initial_input_folder, 'building-geometry', 'surroundings.shp'])
    surroundings_df = gpd.read_file(path_initial_surroundings)
    surroundings_df.to_file(os.path.join('',*[path_to_new_input_folder, 'building-geometry', 'surroundings.shp']))
    # copy dbfs
    DBF_TO_COPY = ['architecture.dbf', 'air_conditioning.dbf', 'supply_systems.dbf']
    for dbf_file in DBF_TO_COPY:
        path_init_dbf = os.path.join(path_to_initial_building_prop_folder, dbf_file)
        shutil.copy2(path_init_dbf, os.path.join(path_to_new_input_folder, 'building-properties'))

    ## Get outputs from area_mapper
    path_zone_shp = os.path.join('',*[PARAMS['path_to_area_mapper'], PARAMS['new_scenario'], 'zone.shp'])
    zone_df = gpd.read_file(path_zone_shp)
    zone_df.to_file(os.path.join(path_to_new_input_folder, 'building-geometry', 'zone.shp'))
    path_typology_dbf = os.path.join('', *[PARAMS['path_to_area_mapper'], PARAMS['new_scenario'], 'typology.dbf'])
    shutil.copy2(path_typology_dbf, os.path.join(path_to_new_input_folder, 'building-properties'))

    ## Modify dbfs
    DBF_FILES = ['typology.dbf', 'architecture.dbf', 'air_conditioning.dbf', 'supply_systems.dbf']
    dbfs = get_dbfs(DBF_FILES, path_to_new_building_prop_folder)
    # get uses
    MULTI_RES_2040, ORIG_RES, OTHER_USES = get_building_lists_by_use(dbfs)
    # update air_conditioning
    dbfs = update_air_conditioning_dbf(MULTI_RES_2040, ORIG_RES, OTHER_USES, dbfs)
    # update supply_systems
    dbfs = update_supply_systems_dbf(dbfs)
    # update architecture properties for MULTI_RES_2040
    dbfs = update_architecture_dbf(MULTI_RES_2040, dbfs)
    # save dbfs
    save_dbfs(dbfs, path_to_new_building_prop_folder)

    ## Get technology
    path_to_new_tech_folder = os.path.join(path_to_new_input_folder, 'technology')
    shutil.rmtree(path_to_new_tech_folder)
    shutil.copytree(PARAMS['path_to_database'], path_to_new_tech_folder)

def get_dbfs(DBF_FILES, path_to_new_building_prop_folder):
    dbfs = {}
    for dbf_file in DBF_FILES:
        dbfs[dbf_file] = read_dbf(dbf_file, path_to_new_building_prop_folder)
    return dbfs


def get_building_lists_by_use(dbfs):
    _RES = list(dbfs['typology.dbf'][dbfs['typology.dbf']['1ST_USE'].str.contains("_RES")].index)
    print('_RES', len(_RES))
    MULTI_RES_2040 = list(dbfs['typology.dbf'][dbfs['typology.dbf']['1ST_USE'] == 'MULTI_RES_2040'].index)
    print('MULTI_RES_2040', len(MULTI_RES_2040))
    ORIG_RES = list(set(_RES) - set(MULTI_RES_2040))
    print('ORIG_RES', len(ORIG_RES))
    OTHER_USES = list(set(dbfs['typology.dbf'].index) - set(_RES))
    print('OTHER_USES', len(OTHER_USES))
    return MULTI_RES_2040, ORIG_RES, OTHER_USES


def save_dbfs(dbfs, path_to_new_building_prop_folder):
    for dbf_name in dbfs.keys():
        path_to_save_dbf = os.path.join(path_to_new_building_prop_folder, dbf_name)
        print('saving...', path_to_save_dbf)
        dbf_df = dbfs[dbf_name].copy()
        dbf_df.insert(0, 'Name', dbf_df.index)
        dataframe_to_dbf(dbf_df, path_to_save_dbf)


def update_architecture_dbf(MULTI_RES_2040, dbfs):
    MULTI_RES_2040_architecture = [0.82, 0, 0.82, 0.82, 0, 0.4, 0.4, 0.4, 0.4,
                                   'CONSTRUCTION_AS3', 'TIGHTNESS_AS2', 'FLOOR_AS3', 'WALL_AS7', 'FLOOR_AS6',
                                   'ROOF_AS4', 'WALL_AS5', 'WINDOW_AS4', 'SHADING_AS1']
    dbfs['architecture.dbf'].loc[MULTI_RES_2040] = MULTI_RES_2040_architecture
    return dbfs


def update_supply_systems_dbf(dbfs):
    dbfs['supply_systems.dbf'].loc[:, 'type_cs'] = 'SUPPLY_COOLING_AS1'
    return dbfs


def update_air_conditioning_dbf(MULTI_RES_2040, ORIG_RES, OTHER_USES, dbfs):
    # MULTI_RES_2040
    dbfs['air_conditioning.dbf'].loc[MULTI_RES_2040, 'type_cs'] = 'HVAC_COOLING_AS5'
    dbfs['air_conditioning.dbf'].loc[MULTI_RES_2040, 'type_ctrl'] = 'HVAC_CONTROLLER_AS2'
    dbfs['air_conditioning.dbf'].loc[MULTI_RES_2040, 'type_vent'] = 'HVAC_VENTILATION_AS0'
    # ORIG_RES
    dbfs['air_conditioning.dbf'].loc[ORIG_RES, 'type_cs'] = 'HVAC_COOLING_AS2'
    dbfs['air_conditioning.dbf'].loc[ORIG_RES, 'type_ctrl'] = 'HVAC_CONTROLLER_AS2'
    dbfs['air_conditioning.dbf'].loc[ORIG_RES, 'type_vent'] = 'HVAC_VENTILATION_AS0'
    # OTHER_USES
    dbfs['air_conditioning.dbf'].loc[OTHER_USES, 'type_cs'] = 'HVAC_COOLING_AS3'
    dbfs['air_conditioning.dbf'].loc[OTHER_USES, 'type_ctrl'] = 'HVAC_CONTROLLER_AS2'
    dbfs['air_conditioning.dbf'].loc[OTHER_USES, 'type_vent'] = 'HVAC_VENTILATION_AS2'
    return dbfs


def ignore_files(dir, files):
    return [f for f in files if os.path.isfile(os.path.join(dir, f))]

def read_dbf(filename, path_to_folder):
    path_to_filename = os.path.join(path_to_folder, filename)
    dbf_df = dbf_to_dataframe(path_to_filename).set_index('Name')
    print(filename, 'imported', dbf_df.shape)
    return dbf_df


if __name__ == "__main__":
    main()