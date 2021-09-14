"""
Copy the archetypes for the selection (district-archetype, year, urban-development-scenario) into the
scenario.
"""
import os
import shutil

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
    database_root = os.path.join(os.path.dirname(__file__), "CH_ReMaP")

    folder_name = f"{district_archetype}_{year}_{urban_development_scenario}"
    print(f"Creating technology folder for {folder_name}")

    print(f"Creating assemblies...")
    copy_assemblies_folder(database_root, locator)

    print(f"Creating components...")
    copy_components_folder(database_root, locator)

    print(f"Creating construction standards...")
    copy_file(os.path.join(database_root, "archetypes", folder_name, "CONSTRUCTION_STANDARD.xlsx"),
              locator.get_database_construction_standards())

    print(f"Creating use types...")
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
              locator.get_database_envelope_systems())
    copy_file(os.path.join(database_root, "assemblies", "HVAC.xls"),
              locator.get_database_air_conditioning_systems())
    copy_file(os.path.join(database_root, "assemblies", "SUPPLY.xls"),
              locator.get_database_supply_assemblies())


def copy_file(src, dst):
    if not os.path.exists(os.path.dirname(dst)):
        os.makedirs(os.path.dirname(dst))
        print(f" - {os.path.dirname(dst)} created")
    shutil.copyfile(src, dst)
    print(f" - {dst}")


if __name__ == "__main__":
    main(cea.config.Configuration())
