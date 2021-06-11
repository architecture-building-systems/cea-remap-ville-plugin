import pandas as pd
from pathlib import Path

from cea.utilities.dbf import dbf_to_dataframe
from area_mapper_sample import calc_gfa_per_use, get_prop_geometry


def main():
    path = Path(
        r"C:\Users\shsieh\Nextcloud\VILLE\Case studies\Echallens\04062021_test_run_input_files\future_2040\area_mapper\sample_data")
    path_to_architecture = path / "architecture.dbf"

    files_dict = {
        'current': {'zone': 'zone.shp', 'typology': 'typology.dbf'},
        'updated': {'zone': 'zone_updated.shp', 'typology': 'typology_updated.dbf'},
    }

    gfa_per_use_dict = {}
    for scenario in files_dict.keys():
        print(scenario)
        path_to_zone_shp = path / files_dict[scenario]["zone"]
        path_to_typology = path / files_dict[scenario]["typology"]
        prop_geometry = get_prop_geometry(path_to_zone_shp, path_to_architecture)
        typology = dbf_to_dataframe(path_to_typology).set_index('Name', drop=False)
        typology_merged = typology.merge(prop_geometry, left_index=True, right_on='Name')
        gfa_per_use_type = calc_gfa_per_use(typology_merged)
        gfa_per_use_dict[scenario] = gfa_per_use_type

    print(pd.DataFrame(gfa_per_use_dict))

    return

if __name__ == "__main__":
    main()
