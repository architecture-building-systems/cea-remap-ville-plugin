__author__ = "Anastasiya Popova"
__copyright__ = "Copyright 2021, Architecture and Building Systems - ETH Zurich"
__credits__ = ["Anastasiya Popova"]
__license__ = "MIT"
__version__ = "0.1"
__maintainer__ = "Anastasiya Popova"
__email__ = "cea@arch.ethz.ch"
__status__ = "Production"

import os
import pandas as pd
import numpy as np
import operator
import random

from typing import Dict, Set, List
from collections import defaultdict
from pathlib import Path
from geopandas import GeoDataFrame
import json
import area_mapper as amap

from cea.utilities.dbf import dbf_to_dataframe
from cea.demand.building_properties import calc_useful_areas


def sample_data_dir(PARAMS) -> Path:
    p = Path(os.path.join(PARAMS['path'], 'area_mapper\sample_data'))
    if not p.exists():
        raise IOError("data_dir not found [%s]" % p)
    return p


def sample_architecture_data(path_to_zone_shp, path_to_architecture) -> pd.DataFrame:
    """
    combines zone.shp and architecture.dbf and calculate GFA
    :param path_to_zone_shp:
    :param path_to_architecture:
    :return:
    """
    architecture = dbf_to_dataframe(path_to_architecture).set_index('Name')
    prop_geometry = GeoDataFrame.from_file(str(path_to_zone_shp.absolute()))
    prop_geometry['footprint'] = prop_geometry.area
    prop_geometry['GFA_m2'] = prop_geometry['footprint'] * (prop_geometry['floors_ag'] + prop_geometry['floors_bg'])
    prop_geometry['GFA_ag_m2'] = prop_geometry['footprint'] * prop_geometry['floors_ag']
    prop_geometry['GFA_bg_m2'] = prop_geometry['footprint'] * prop_geometry['floors_bg']
    prop_geometry = prop_geometry.merge(architecture, on='Name').set_index('Name')
    prop_geometry = calc_useful_areas(prop_geometry)

    return prop_geometry


def sample_typology_data(PARAMS) -> pd.DataFrame:
    path_to_typology = sample_data_dir(PARAMS) / "typology.dbf"
    typology = dbf_to_dataframe(path_to_typology).set_index('Name', drop=False)
    return typology


def filter_buildings_by_year_sample_data(typology_merged: pd.DataFrame, year: int, less_than: bool = True):
    if "YEAR" not in typology_merged:
        raise KeyError("provided data frame is missing the column 'YEAR'")
    if less_than:
        op = operator.lt
    else:
        op = operator.gt
    return typology_merged[op(typology_merged.YEAR, year)]


def filter_buildings_by_use_sample_data(typology_merged: pd.DataFrame, use_to_be_filtered: str):
    typology_copy = typology_merged.copy()
    building_names = typology_merged.index
    # drop MULTI_RES_2040 buildings
    typology_merged = typology_merged.drop(typology_merged[typology_merged["1ST_USE"] == use_to_be_filtered].index)
    building_names_remained = typology_merged.index
    dropped_buildings_by_use = typology_copy.loc[list(set(building_names) - set(building_names_remained))]
    return typology_merged, dropped_buildings_by_use


def sample_data(PARAMS) -> pd.DataFrame:
    """
    merges topology.dbf and architecture.dbf, calculate GFA, and initiates empty columns
    :return:
    """
    # READ
    path_to_architecture = sample_data_dir(PARAMS) / "architecture.dbf"
    if not path_to_architecture.exists():
        raise IOError("architecture file not found [%s]" % path_to_architecture)
    path_to_zone_shp = sample_data_dir(PARAMS) / "zone.shp"
    if not path_to_zone_shp.exists():
        raise IOError("shape file not found [%s]" % path_to_zone_shp)
    prop_geometries = sample_architecture_data(path_to_zone_shp, path_to_architecture)
    typology = sample_typology_data(PARAMS)
    typology_merged = typology.merge(prop_geometries, left_index=True, right_on='Name')
    typology_merged.floors_ag = typology_merged.floors_ag.astype(int)

    typology_merged["city_zone"] = 1
    typology_merged["additional_floors"] = 0
    typology_merged["floors_ag_updated"] = typology_merged.floors_ag.astype(int)
    typology_merged["height_ag_updated"] = typology_merged.height_ag.astype(int)

    return typology_merged


def typology_use_columns() -> List[str]:
    return ["1ST_USE", "2ND_USE", "3RD_USE"]


def sample_mapping() -> Dict[int, Set[str]]:
    """
    possible building uses per city zone
    :return:
    """
    return {
        1: {"SINGLE_RES", "MULTI_RES", "RETAIL", "NONE", "HOSPITAL", "INDUSTRIAL",
            "GYM", "SCHOOL", "PARKING", "LIBRARY", "FOODSTORE", "RESTAURANT", "HOTEL"},
    }


def calculate_per_use_gfa(typology_merged: pd.DataFrame):
    """
    calculates GFA per use type based on the 1st use, 2nd use and 3rd use [m2]
    :param typology_merged:
    :return:
    """
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
    PARAMS = {
        # scenario-specific
        'path': r"C:\Users\shsieh\Nextcloud\VILLE\Case studies\Echallens\04062021_test_run_input_files\future_2040",
        'additional_population': 2175,  # people, 7900 - 5725 # TODO: add population model to store the data
        'future_occupant_density': 50,  # living space m2/occupants
        'building_height_limit': 24,  # m
        'MULTI_RES_USE_TYPE': 'MULTI_RES_2040',
        'preserve_buildings_built_before': 1920,
        'SINGLE_to_MULTI_RES_ratio': 0.0,
        'max_additional_floors': 10,
        # constants
        'ratio_living_space_to_GFA': 0.82,
        'floor_height': 3,  # m
        'min_additional_floors': 0,
    }

    with open(os.path.join(PARAMS['path'], 'PARAMS.json'), 'w') as fp:
        json.dump(PARAMS, fp)

    remove_updated_files(PARAMS, clean=True)  # if True, cleans the output files in sample_data folder

    typology_merged = sample_data(PARAMS)

    all_known_use_types = set([leaf
                               for tree in typology_merged[typology_use_columns()].values
                               for leaf in tree])
    # check if all use types are known # FIXME: is it true?
    assert all([use in all_known_use_types for _, zone in sample_mapping().items() for use in zone])
    current_gfa_per_use, gfa_ratio_per_use_type = calculate_per_use_gfa(typology_merged)
    current_gfa_per_use.to_csv(os.path.join(sample_data_dir(PARAMS), "current_gfa_per_use_type.csv"))
    # FIXME: below is a work-around to remove MULTI_RES_2040
    if PARAMS['MULTI_RES_USE_TYPE'] in current_gfa_per_use.index:
        future_planned_res_gfa = current_gfa_per_use["MULTI_RES_2040"]
        current_gfa_per_use.drop("MULTI_RES_2040", inplace=True)
        gfa_ratio_per_use_type.drop("MULTI_RES_2040", inplace=True)
    else:
        future_planned_res_gfa = 0.0

    # set target gfa ratios # FIXME: get from files
    relative_gfa_ratio_to_res = (gfa_ratio_per_use_type
                                 / (gfa_ratio_per_use_type.SINGLE_RES + gfa_ratio_per_use_type.MULTI_RES))  # 2020

    # calculate future required area per use type
    future_required_additional_res_gfa, \
    future_required_res_gfa = calc_future_required_residential_gfa(current_gfa_per_use, PARAMS)
    future_required_gfa_dict = dict()
    for use_type in relative_gfa_ratio_to_res.index:
        if use_type == "SINGLE_RES":
            future_required_gfa_dict[use_type] = current_gfa_per_use[use_type]  # TODO: update if convert SFH to MFH
        elif use_type == "MULTI_RES":
            future_required_gfa_dict[use_type] = future_required_additional_res_gfa + current_gfa_per_use[use_type]
        else:
            future_required_gfa_dict.update({use_type: future_required_res_gfa * relative_gfa_ratio_to_res[use_type]})
    future_required_gfa_per_use = pd.Series(future_required_gfa_dict)
    future_required_gfa_per_use = future_required_gfa_per_use.astype(int)
    future_required_gfa_per_use.to_csv(os.path.join(sample_data_dir(PARAMS), "future_required_gfa_per_use.csv"))
    total_future_required_gfa = future_required_gfa_per_use.sum()
    check_ratio(future_required_gfa_per_use, total_future_required_gfa)

    ## calculate additional required GFA as (future required area - current existing area)
    additional_required_gfa_per_use = future_required_gfa_per_use - current_gfa_per_use
    # remove MULTI_RES_2040 from additional required area
    additional_required_gfa_per_use["MULTI_RES"] = additional_required_gfa_per_use["MULTI_RES"] - future_planned_res_gfa
    target_per_use_gfa = additional_required_gfa_per_use.astype(int).to_dict()

    ## calculate possible range of additional floors for each building
    height_limit_per_city_zone = {1: (0, PARAMS['building_height_limit'] // PARAMS['floor_height'])}
    range_additional_floors_per_building = dict()
    for name, building in typology_merged.iterrows():
        min_floors, max_floors = height_limit_per_city_zone[building.city_zone]
        range_additional_floors_per_building[name] = [0, max(0, max_floors - building.floors_ag)]

    # filter out buildings by use
    buildings_filtered_out_by_use, dropped_buildings_by_use = filter_buildings_by_use_sample_data(
        typology_merged,
        use_to_be_filtered=PARAMS["MULTI_RES_USE_TYPE"],
    )

    # keep old buildings unchanged
    buildings_filtered_out_by_age = filter_buildings_by_year_sample_data(
        buildings_filtered_out_by_use,
        year=PARAMS["preserve_buildings_built_before"] + 1,
        less_than=True
    ).copy()

    buildings_kept = filter_buildings_by_year_sample_data(
        buildings_filtered_out_by_use,
        year=PARAMS["preserve_buildings_built_before"],
        less_than=False
    ).copy()

    # transform parts of SINGLE_RES to MULTI_RES # FIXME: maybe this should be done earlier?
    buildings_SINGLE_RES = list(buildings_kept.loc[buildings_kept['1ST_USE'] == 'SINGLE_RES'].index)
    num_buildings_to_MULTI_RES = int(len(buildings_SINGLE_RES) * PARAMS['SINGLE_to_MULTI_RES_ratio'])
    buildings_to_MULTI_RES = random.sample(buildings_SINGLE_RES, num_buildings_to_MULTI_RES)
    for building in buildings_to_MULTI_RES:
        buildings_kept.loc[building, :] = buildings_kept.loc[building].replace({"SINGLE_RES": "MULTI_RES"})
    # FIXME: add the area of replaced SFH to equivalent target_per_use_gfa["MULTI_RES"]

    # create random scenarios
    scenarios = amap.randomize_scenarios(
        typology_merged=buildings_kept,
        mapping=sample_mapping(),
        use_columns=typology_use_columns(),
        scenario_count=10,
    )

    # built area allocation for all scenarios
    optimizations = dict()
    metrics = dict()
    target = additional_required_gfa_per_use.sum()
    for scenario in scenarios:
        scenario_typology_merged = scenarios[scenario]

        partitions = [[(n, k) for k in m if k != "NONE"]
                      for n, m in scenario_typology_merged[typology_use_columns()].iterrows()]

        sub_buildings = [leaf
                         for tree in partitions
                         for leaf in tree if leaf != "NONE"]

        sub_building_idx = ["%s.%i" % (b[0], i)
                            for i, b in enumerate(sub_buildings)]

        sub_building_use = {"%s.%i" % (b[0], i): b[1]
                            for i, b in enumerate(sub_buildings)}

        print("scenario [%i], target [%.4f]" % (scenario, target))

        footprint_area = scenario_typology_merged.footprint.to_dict()
        sub_footprint_area = {sb: footprint_area[sb.split(".")[0]]
                              for sb in sub_building_idx}

        building_to_sub_building = defaultdict(list)
        for sb in sub_building_idx:
            building, num = sb.split('.')
            building_to_sub_building[building].append(sb)
        print("len of problem [%i]" % len(sub_building_idx))

        solution = amap.optimize(
            target,
            sub_building_idx,
            PARAMS['min_additional_floors'],
            PARAMS['max_additional_floors'],
            sub_footprint_area,
            building_to_sub_building,
            range_additional_floors_per_building,
            target_per_use_gfa,
            sub_building_use
        )

        optimizations[scenario] = {
            "solution": solution,
            "sub_footprint_area": sub_footprint_area,
            "building_to_sub_building": building_to_sub_building
        }
        print("is-success [%i]" % solution.sol_status)

        detailed_metrics = amap.detailed_result_metrics(
            solution=solution,
            sub_building_use=sub_building_use,
            sub_footprint_area=sub_footprint_area,
            per_use_gfa=target_per_use_gfa,
            target=target
        )
        metrics[scenario] = detailed_metrics
        print("absolute error [%.4f]" % detailed_metrics["absolute_error"])
        print("relative error [%.4f]" % detailed_metrics["relative_error"])

    # find the best scenario
    best_scenario, scenario_errors = amap.find_optimum_scenario(
        optimizations=optimizations,
        target=target
    )
    print("best scenario: [%i] with absolute error [%.4f]"
          % (best_scenario, scenario_errors[best_scenario]))

    # write typology.dbf and zone.shp with the best scenario
    typology_df = scenarios[best_scenario].copy()
    # add back those buildings initially filtered out
    typology_df = typology_df.append(buildings_filtered_out_by_age, sort=True)
    typology_df = typology_df.append(dropped_buildings_by_use, sort=True)
    amap.update_zone_shp_file(
        solution=optimizations[best_scenario]["solution"],
        typology_merged=typology_df,
        building_to_sub_building=optimizations[best_scenario]["building_to_sub_building"],
        path_to_input_zone_shape_file=sample_data_dir(PARAMS) / "zone.shp",
        path_to_output_zone_shape_file=sample_data_dir(PARAMS) / "zone_updated.shp"
    )
    amap.update_typology_file(
        solution=optimizations[best_scenario]["solution"],
        status_quo_typology=typology_merged,
        optimized_typology=typology_df,
        building_to_sub_building=optimizations[best_scenario]["building_to_sub_building"],
        path_to_output_typology_file=sample_data_dir(PARAMS) / "typology_updated.dbf",
        columns_to_keep=[("Name", str), ("YEAR", int), ("STANDARD", str), ("1ST_USE", str), ("1ST_USE_R", float),
                         ("2ND_USE", str), ("2ND_USE_R", float), ("3RD_USE", str), ("3RD_USE_R", float),
                         ("REFERENCE", str)],
        PARAMS=PARAMS
    )


def check_ratio(future_required_gfa_series, total_future_required_gfa):
    # calculate future use ratio based on GFA
    future_required_gfa_ratio = future_required_gfa_series / total_future_required_gfa
    if not np.isclose(future_required_gfa_ratio.sum(), 1.0):
        raise ValueError("sum of use-ratios does not equal 1.0, instead [%f]" % future_required_gfa_ratio.sum())


def calc_future_required_residential_gfa(gfa_per_use_type, PARAMS):
    future_required_additional_res_living_space = PARAMS['additional_population'] * PARAMS['future_occupant_density']
    future_required_additional_res_gfa = future_required_additional_res_living_space / PARAMS[
        'ratio_living_space_to_GFA']
    future_required_res_gfa = (future_required_additional_res_gfa +
                               gfa_per_use_type.SINGLE_RES + gfa_per_use_type.MULTI_RES)
    return future_required_additional_res_gfa, future_required_res_gfa


def remove_updated_files(PARAMS, clean=True):
    if clean:
        data_files = os.listdir(sample_data_dir(PARAMS))
        for file in data_files:
            if file.find("_update") >= 0:
                print("cleaning: [%s]" % (sample_data_dir(PARAMS) / file))
                Path(sample_data_dir(PARAMS) / file).unlink()
    return


if __name__ == "__main__":
    main()