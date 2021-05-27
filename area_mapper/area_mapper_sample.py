import area_mapper as amap
import pandas as pd

from typing import Dict, Set, List
from collections import defaultdict
from pathlib import Path
from geopandas import GeoDataFrame

from cea.utilities.dbf import dbf_to_dataframe
from cea.demand.building_properties import calc_useful_areas


def data_dir() -> Path:
    p = Path().absolute() / "sample_data"
    if not p.exists():
        raise IOError("data_dir not found [%s]" %p)
    return p


def sample_architecture_data(path: Path=data_dir()) -> pd.DataFrame:
    path_to_architecture = path / "architecture.dbf"
    if not path_to_architecture.exists():
        raise IOError("architecture file not found [%s]" %path_to_architecture)
    architecture = dbf_to_dataframe(path_to_architecture).set_index('Name')

    path_to_zone_shp = path / "zone.shp"
    if not path_to_zone_shp.exists():
        raise IOError("shape file not found [%s]" %path_to_zone_shp)
    prop_geometry = GeoDataFrame.from_file(str(path_to_zone_shp.absolute()))

    prop_geometry['footprint'] = prop_geometry.area
    prop_geometry['GFA_m2'] = prop_geometry['footprint'] * (prop_geometry['floors_ag'] + prop_geometry['floors_bg'])
    prop_geometry['GFA_ag_m2'] = prop_geometry['footprint'] * prop_geometry['floors_ag']
    prop_geometry['GFA_bg_m2'] = prop_geometry['footprint'] * prop_geometry['floors_bg']
    prop_geometry = prop_geometry.merge(architecture, on='Name').set_index('Name')
    prop_geometry = calc_useful_areas(prop_geometry)

    return prop_geometry


def sample_typology_data(path: Path=data_dir()) -> pd.DataFrame:
    path_to_typology = path / "typology.dbf"
    typology = dbf_to_dataframe(path_to_typology).set_index('Name', drop=False)
    return typology


def sample_data() -> pd.DataFrame:
    typology_merged = sample_typology_data().merge(sample_architecture_data(), left_index=True, right_on='Name')

    # columns updated
    typology_merged.floors_ag = typology_merged.floors_ag.astype(int)
    typology_merged["1ST_USE"].replace({"MULTI_RES": "RESIDENTIAL", "SINGLE_RES": "RESIDENTIAL"}, inplace=True)
    typology_merged["2ND_USE"].replace({"MULTI_RES": "RESIDENTIAL", "SINGLE_RES": "RESIDENTIAL"}, inplace=True)
    typology_merged["3RD_USE"].replace({"MULTI_RES": "RESIDENTIAL", "SINGLE_RES": "RESIDENTIAL"}, inplace=True)

    # new columns added
    typology_merged["city_zone"] = 1
    typology_merged["additional_floors"] = 0
    typology_merged["floors_ag_updated"] = typology_merged.floors_ag.astype(int)
    typology_merged["height_ag_updated"] = typology_merged.height_ag.astype(int)

    return typology_merged


def sample_use_columns() -> List[str]:
    return ["1ST_USE", "2ND_USE", "3RD_USE"]


def sample_mapping() -> Dict[int, Set[str]]:
    return {
        1: {"RESIDENTIAL", "RETAIL", "NONE", "HOSPITAL", "INDUSTRIAL", "GYM", "SCHOOL", "PARKING", "LIBRARY"},
    }


def calculate_per_use_gfa(typology_merged: pd.DataFrame):
    """calculate status quo GFA per use type based on the 1st use, 2nd use and 3rd use [m2]"""
    typology_merged["1ST_USE"].replace({"MULTI_RES": "RESIDENTIAL", "SINGLE_RES": "RESIDENTIAL"}, inplace=True)
    typology_merged["2ND_USE"].replace({"MULTI_RES": "RESIDENTIAL", "SINGLE_RES": "RESIDENTIAL"}, inplace=True)
    typology_merged["3RD_USE"].replace({"MULTI_RES": "RESIDENTIAL", "SINGLE_RES": "RESIDENTIAL"}, inplace=True)

    typology_merged["GFA_1ST_USE"] = typology_merged["1ST_USE_R"] * typology_merged["GFA_m2"]
    typology_merged["GFA_2ND_USE"] = typology_merged["2ND_USE_R"] * typology_merged["GFA_m2"]
    typology_merged["GFA_3RD_USE"] = typology_merged["3RD_USE_R"] * typology_merged["GFA_m2"]

    gfa_series_1st_use = typology_merged.groupby("1ST_USE").sum().loc[:, "GFA_1ST_USE"]
    gfa_series_2nd_use = typology_merged.groupby("2ND_USE").sum().loc[:, "GFA_2ND_USE"]
    gfa_series_3rd_use = typology_merged.groupby("3RD_USE").sum().loc[:, "GFA_3RD_USE"]

    gfa_per_use_type = defaultdict(float)
    for use_series in [gfa_series_1st_use, gfa_series_2nd_use, gfa_series_3rd_use]:
        for use, val in use_series.iteritems():
            gfa_per_use_type[use] += val

    # get rid of the unwanted "NONE" use-type
    del gfa_per_use_type["NONE"]

    gfa_per_use_type = pd.Series(gfa_per_use_type)
    gfa_ratio_per_use_type = gfa_per_use_type / gfa_per_use_type.sum()

    return gfa_per_use_type, gfa_ratio_per_use_type


def main():
    clean = True
    if clean:
        import os
        data_files = os.listdir(data_dir())
        for file in data_files:
            if file.find("_update") >= 0:
                print("cleaning: [%s]" %(data_dir() / file))
                Path(data_dir() / file).unlink()

    typology_merged = sample_data()
    all_known_use_types = set([leaf
                               for tree in typology_merged[sample_use_columns()].values
                               for leaf in tree])
    assert all([use in all_known_use_types for _, zone in sample_mapping().items() for use in zone])
    gfa_per_use_type, gfa_ratio_per_use_type = calculate_per_use_gfa(typology_merged)
    relative_gfa_ratio_to_res = gfa_ratio_per_use_type / gfa_ratio_per_use_type.RESIDENTIAL

    # calculate future required area per use type
    additional_population = 7900 - 5725  # people
    future_occupant_density = 100  # m2/occupants
    future_required_additional_res_gfa = additional_population * future_occupant_density / 0.82
    future_required_res_gfa = future_required_additional_res_gfa + gfa_per_use_type["RESIDENTIAL"]
    future_required_gfa_series = pd.Series(
        {
            use_type: future_required_res_gfa * relative_gfa_ratio_to_res[use_type]
            for use_type in relative_gfa_ratio_to_res.index
        }
    )
    future_required_gfa_series = future_required_gfa_series.astype(int)
    total_future_required_gfa = future_required_gfa_series.sum()
    print("future_required_gfa_series:\n", future_required_gfa_series)
    print("total_future_required_gfa:\n", total_future_required_gfa)

    # calculate future use ratio based on GFA
    future_required_gfa_ratio = future_required_gfa_series / total_future_required_gfa
    assert future_required_gfa_ratio.sum() == 1.0

    # calculate additional required GFA as (future required area - existing area)
    additional_required_gfa = future_required_gfa_series - gfa_per_use_type
    target_per_use_gfa = additional_required_gfa.astype(int).to_dict()

    # upper bound = maximum_allowed_building_height / room_height
    room_height = 3
    city_zones = {1: (0, 24 // room_height)}
    max_allowed_floors = defaultdict(int)

    # calculate maximum allowed number of additional floors for each building
    for name, building in typology_merged.iterrows():
        min_floors, max_floors = city_zones[building.city_zone]
        max_allowed_floors[name] = [0, max(0, max_floors - building.floors_ag)]
    building_zones = {building: max_allowed_floors[building] for building in typology_merged.index}

    # input parameters
    min_additional_floors = 0
    max_additional_floors = 10
    scenarios = amap.randomize_scenarios(
        typology_merged=typology_merged,
        mapping=sample_mapping(),
        use_columns=sample_use_columns(),
        scenario_count=10,
    )

    optimizations = dict()
    metrics = dict()
    target = additional_required_gfa.sum()
    for scenario in scenarios:
        scenario_typology_merged = scenarios[scenario]

        partitions = [[(n, k) for k in m if k != "NONE"]
                      for n, m in scenario_typology_merged[sample_use_columns()].iterrows()]

        sub_buildings = [leaf for tree in partitions
                         for leaf in tree if leaf != "NONE"]

        sub_building_idx = ["%s.%i" % (b[0], i)
                            for i, b in enumerate(sub_buildings)]

        sub_building_use = {"%s.%i" % (b[0], i): b[1]
                            for i, b in enumerate(sub_buildings)}

        print("scenario [%i], target [%.4f]" %(scenario, target))

        footprint_area = scenario_typology_merged.footprint.to_dict()
        sub_footprint_area = {sb: footprint_area[sb.split(".")[0]]
                              for sb in sub_building_idx}

        building_to_sub_building = defaultdict(list)
        for sb in sub_building_idx:
            building, num = sb.split('.')
            building_to_sub_building[building].append(sb)
        print("len of problem [%i]" %len(sub_building_idx))

        solution = amap.optimize(
            target,
            sub_building_idx,
            min_additional_floors,
            max_additional_floors,
            sub_footprint_area,
            building_to_sub_building,
            building_zones,
            target_per_use_gfa,
            sub_building_use
        )

        optimizations[scenario] = {
            "solution": solution,
            "sub_footprint_area": sub_footprint_area,
            "building_to_sub_building": building_to_sub_building
        }
        print("is-success [%i]" %solution.sol_status)

        detailed_metrics = amap.detailed_result_metrics(
            solution=solution,
            sub_building_use=sub_building_use,
            sub_footprint_area=sub_footprint_area,
            per_use_gfa=target_per_use_gfa,
            target=target
        )
        metrics[scenario] = detailed_metrics
        print("absolute error [%.4f]" %detailed_metrics["absolute_error"])
        print("relative error [%.4f]" %detailed_metrics["relative_error"])

    best_scenario, scenario_errors = amap.find_optimum_scenario(
        optimizations=optimizations,
        target=target
    )
    print("best scenario: [%i] with absolute error [%.4f]"
          %(best_scenario, scenario_errors[best_scenario]))

    amap.update_zone_shp_file(
        solution=optimizations[best_scenario]["solution"],
        typology_merged=scenarios[best_scenario],
        building_to_sub_building=optimizations[best_scenario]["building_to_sub_building"],
        path_to_input_zone_shape_file=data_dir() / "zone.shp",
        path_to_output_zone_shape_file=data_dir() / "zone_updated.shp"
    )
    amap.update_typology_file(
        solution=optimizations[best_scenario]["solution"],
        status_quo_typology=typology_merged,
        optimized_typology=scenarios[best_scenario],
        building_to_sub_building=optimizations[best_scenario]["building_to_sub_building"],
        path_to_output_typology_file=data_dir() / "typology_updated.dbf",
        columns_to_keep=[("Name", str), ("YEAR", int), ("STANDARD", str), ("1ST_USE", str), ("1ST_USE_R", float),
                         ("2ND_USE", str), ("2ND_USE_R", float), ("3RD_USE", str), ("3RD_USE_R", float),
                         ("REFERENCE", str)]
    )


if __name__ == "__main__":
    main()
