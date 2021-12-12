"""
Copy the archetypes for the selection (district-archetype, year, urban-development-scenario) into the
scenario.
"""
import os
import shutil
import win32com.client
import glob

import cea.config
import cea.inputlocator

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
    folder_name = f"{district_archetype}_{year}_{urban_development_scenario}"
    print(f"Creating technology folder for {folder_name}")

    create_input_technology_folder(folder_name, locator)


def create_input_technology_folder(folder_name, locator):
    database_root = os.path.join(os.path.dirname(__file__), "CH_ReMaP")
    print(f"Creating assemblies...")
    copy_assemblies_folder(database_root, locator)
    # .xlsx
    print("saving .xlsx")
    o = win32com.client.Dispatch("Excel.Application")
    o.Visible = False
    o.DisplayAlerts = False
    input_dir = os.path.join(locator.get_databases_assemblies_folder())
    output_dir = input_dir
    files = glob.glob(input_dir + "/*.xls")
    for filename in files:
        print(filename)
        file = os.path.basename(filename)
        output = output_dir + '/' + file.replace('.xls', '.xlsx')
        wb = o.Workbooks.Open(filename)
        wb.ActiveSheet.SaveAs(output, 51)
        wb.Close(True)
    o.Quit()
    del o
    print(f"Creating components...")
    copy_components_folder(database_root, locator)
    print(f"Creating archetypes/construction standards...")
    copy_file(os.path.join(database_root, "archetypes", "CONSTRUCTION_STANDARD_SUMMARY.xlsx"),
              locator.get_database_construction_standards())
    print(f"Creating archetypes/use_types...")
    copy_use_types(database_root, folder_name, locator)


def copy_use_types(database_root, folder_name, locator):
    use_types_folder = os.path.join(database_root, "archetypes", folder_name, "use_types")
    for fname in os.listdir(use_types_folder):
        copy_file(os.path.join(use_types_folder, fname),
                  os.path.join(locator.get_database_use_types_folder(), fname))


def copy_components_folder(database_root, locator):
    copy_file(os.path.join(database_root, "components", "CONVERSION.xls"),
              locator.get_database_conversion_systems())
    copy_file(os.path.join(database_root, "components", "DISTRIBUTION.xls"),
              locator.get_database_distribution_systems())
    copy_file(os.path.join(database_root, "components", "FEEDSTOCKS.xls"),
              locator.get_database_feedstocks())


def copy_assemblies_folder(database_root, locator):
    copy_file(os.path.join(database_root, "assemblies", "ENVELOPE.xls"),
              os.path.join(locator.get_databases_assemblies_folder(), 'ENVELOPE.xls')) #FIXME: workaround to be competible with CEA master, the assemblies folder should be updated when the docker image is updated with the latest master.
    copy_file(os.path.join(database_root, "assemblies", "HVAC.xls"),
              os.path.join(locator.get_databases_assemblies_folder(), 'HVAC.xls'))
    copy_file(os.path.join(database_root, "assemblies", "SUPPLY.xls"),
              os.path.join(locator.get_databases_assemblies_folder(), 'SUPPLY.xls'))


def copy_file(src, dst):
    if not os.path.exists(os.path.dirname(dst)):
        os.makedirs(os.path.dirname(dst))
        print(f" - {os.path.dirname(dst)} created")
    shutil.copyfile(src, dst)
    # print(f" - {dst}")


def update_indoor_comfort(RF_scenario, locator):
    print('Updating INDOOR_COMFORT according to scenario')
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
    # print([sheet.Name for sheet in wb.Sheets])
    # delete INDOOR_COMFORT
    for sheet in wb.Worksheets:
        if sheet.Name == 'INDOOR_COMFORT':
            sheet.Delete()
    # print([sheet.Name for sheet in wb.Sheets])
    ws_comfort = wb.Worksheets.Add()
    ws_comfort.Name = 'INDOOR_COMFORT'
    ws_summary.Range("A1:AF100").Copy(ws_comfort.Range("A1:AF100"))
    wb.Close(True)  # close and save changes
    o.Quit()
    del o


if __name__ == "__main__":
    main(cea.config.Configuration())
