"""
Map the field ``typology.df:STANDARD`` to the new value, based on:
- construction (BAU / NEP)
- year (2040 / 2060)
- district-archetype (URB, SURB, RRL)

This can be done on a new scenario, _before_ running archetypes-mapper.
"""
import os
import win32com.client
import pandas as pd

import cea.config
import cea.inputlocator
import cea.utilities.dbf
import cea.datamanagement.archetypes_mapper
from remap_ville_plugin.create_technology_database import copy_file, copy_assemblies_folder, copy_use_types

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
    construction = config.remap_ville_scenarios.construction

    locator = cea.inputlocator.InputLocator(scenario=config.scenario, plugins=config.plugins)

    # FIXME: work around to copy construction standard, should be deleted before merging
    # print(f"Creating construction standards...")
    # database_root = os.path.join(os.path.dirname(__file__), "CH_ReMaP")
    # copy_file(os.path.join(database_root, "archetypes", "CONSTRUCTION_STANDARD_SUMMARY.xlsx"),
    #           locator.get_database_construction_standards())
    # print(f"Creating assemblies...")
    # copy_assemblies_folder(database_root, locator)
    #
    # print(f"Creating use types...")
    # urban_development_scenario = config.remap_ville_scenarios.urban_development_scenario
    # folder_name = f"{district_archetype}_{year}_{urban_development_scenario}"
    # print(folder_name)
    # copy_use_types(database_root, folder_name, locator)
    # # FIXME: END OF HACK

    # update INDOOR_COMFORT
    RF_scenario = f'RF_{construction}'
    update_indoor_comfort(RF_scenario, locator)

    mapping = read_mapping()
    print('\n modifying typology in...', locator.get_building_typology())
    typology = cea.utilities.dbf.dbf_to_dataframe(locator.get_building_typology())

    construction = 'NEP' # FIXME: hard-coded since 'NEP' is always applied
    for index, row in typology.iterrows():
        building = row.Name
        old_standard = row.STANDARD
        use_type = row["1ST_USE"]
        new_standard = do_mapping(mapping, old_standard, district_archetype, use_type, year, construction)
        # print(f"Updating {building}, {old_standard} => {new_standard}")
        typology.loc[index, "STANDARD"] = new_standard # FIXME: investigate Key error!

    cea.utilities.dbf.dataframe_to_dbf(typology, locator.get_building_typology())

    buildings = locator.get_zone_building_names()
    cea.datamanagement.archetypes_mapper.archetypes_mapper(
        locator=locator,
        update_architecture_dbf=True,
        update_air_conditioning_systems_dbf=True,
        update_indoor_comfort_dbf=True,
        update_internal_loads_dbf=False,
        update_supply_systems_dbf=True,
        update_schedule_operation_cea=False,
        buildings=buildings)
    print('\n Building properties are updated!')


def update_indoor_comfort(RF_scenario, locator):
    # call excel application
    o = win32com.client.Dispatch("Excel.Application")
    o.Visible = False
    o.DisplayAlerts = False
    # read INDOOR_COMFORT_SUMMARY
    path_summary = os.path.join(os.path.dirname(__file__), "CH_ReMaP", "archetypes", "INDOOR_COMFORT_SUMMARY.xlsx")
    wb_summary = o.Workbooks.Open(path_summary)
    ws_summary = wb_summary.Sheets(RF_scenario)
    # modify USE_TYPE_PROPERTIES
    path_orig = locator.get_database_use_types_properties()
    wb = o.Workbooks.Open(path_orig)
    print([sheet.Name for sheet in wb.Sheets])
    # delete INDOOR_COMFORT
    for sheet in wb.Worksheets:
        if sheet.Name == 'INDOOR_COMFORT':
            sheet.Delete()
    print([sheet.Name for sheet in wb.Sheets])
    ws_comfort = wb.Worksheets.Add()
    ws_comfort.Name = 'INDOOR_COMFORT'
    ws_summary.Range("A1:AF100").Copy(ws_comfort.Range("A1:AF100"))
    wb.Close(True)  # close and save changes
    o.Quit()
    del o


def do_mapping(mapping, old_standard, district_archetype, use_type, year, construction):
    try:
        new_standard = mapping[(old_standard, district_archetype, use_type, year, construction)]
    except KeyError:
        print("Archetype not specificed in the mapping table (mapping_CONSTRUCTION_STANDARD.xlsx)", (old_standard, district_archetype, use_type, year, construction))
        new_standard = old_standard
    return new_standard


def read_mapping():
    """
    Read construction standards according to district, use_type, year, retrofit scenarios
    TODO: at the moment, all retrofit scenarios have the same construction standard
    :return:
    """
    mapping_df = pd.read_excel(os.path.join(os.path.dirname(__file__), "mapping_CONSTRUCTION_STANDARD.xlsx"))
    mapping = {}
    for _, row in mapping_df.iterrows():
        status_quo = row.STATUS_QUO
        district = row.DISTRICT
        use_type = row.USE_TYPE
        year = str(row.YEAR)
        mapping[(status_quo, district, use_type, year, "BAU")] = row.BAU
        mapping[(status_quo, district, use_type, year, "NEP")] = row.NEP
    return mapping


if __name__ == "__main__":
    main(cea.config.Configuration())
