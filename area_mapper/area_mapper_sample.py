__author__ = "Anastasiya Popova"
__copyright__ = "Copyright 2021, Architecture and Building Systems - ETH Zurich"
__credits__ = ["Anastasiya Popova", "Shanshan Hsieh"]
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

from cea.utilities.dbf import dbf_to_dataframe, dataframe_to_dbf
from cea.demand.building_properties import calc_useful_areas


def main():
    PARAMS = {
        # scenario-specific
        'path': r"C:\Users\shsieh\Nextcloud\VILLE\Case studies\Echallens\04062021_test_run_input_files\future_2040",
        'additional_population': 2175,  # people
        'future_occupant_density': 50,  # living space m2/occupants
        'current_SFH_occupant_density': 120,  # living space m2/occupants
        'UD_scenario': 'BAL',  # SQ, DCL, BAL
        'building_height_limit': 24,  # m
        'MULTI_RES_PLANNED': 'MULTI_RES_2040',
        'USES_UNTOUCH': ['SINGLE_RES'],
        'preserve_buildings_built_before': 1920,
        'SINGLE_to_MULTI_RES_ratio': 0.0,
        'max_additional_floors': 10,
        # constants
        'ratio_living_space_to_GFA': 0.82,
        'floor_height': 3,  # m
        'min_additional_floors': 0,
        'scenario_count': 2
    }

    with open(os.path.join(sample_data_dir(PARAMS), 'PARAMS.json'), 'w') as fp:
        json.dump(PARAMS, fp)

    remove_updated_files(PARAMS, clean=True)  # if True, cleans the output files in sample_data folder
    overview = {}
    ## Read typology_merged
    typology_merged = get_sample_data(PARAMS)
    existing_uses = set([leaf for tree in typology_merged[typology_use_columns()].values for leaf in tree])
    valid_use_types = ["SINGLE_RES", "MULTI_RES", "RETAIL", "NONE", "HOSPITAL", "INDUSTRIAL", "GYM", "SCHOOL",
                       "PARKING", "LIBRARY", "FOODSTORE", "RESTAURANT", "HOTEL", "OFFICE", "MUSEUM", "SERVERROOM",
                       "SWIMMING", "UNIVERSITY", "COOLROOM", "MULTI_RES_2040"]
    assert all([True if use in valid_use_types else False for use in existing_uses])  # check valid uses

    ## get typology_statusquo and typology_planned
    typology_statusquo = typology_merged.copy()
    typology_statusquo, typology_planned = remove_buildings_by_uses(typology_statusquo,
                                                                    uses_to_remove=[PARAMS['MULTI_RES_PLANNED']])
    gfa_per_use_statusquo = calc_gfa_per_use(typology_statusquo, "GFA_m2")
    gfa_per_use_planned = calc_gfa_per_use(typology_planned, "GFA_m2")
    gfa_res_planned = gfa_per_use_planned[PARAMS['MULTI_RES_PLANNED']]
    overview["gfa_per_use_statusquo"] = gfa_per_use_statusquo.astype(int)
    overview["gfa_ratio_per_use_statusquo"] = round(gfa_per_use_statusquo / gfa_per_use_statusquo.sum(), 5)
    overview["gfa_per_use_planned"] = gfa_per_use_planned.astype(int)

    # TODO: KEEP "FUTURE RESERVED AREA" (ONLY FOOTPRINTS BUT NO HEIGHT) TO BUILD MULTI_RES

    ## SET TARGET GFA RATIOS
    # get relative ratio total_additional_gfa_target
    gfa_per_use_statusquo = combine_statusquo_MULTI_RES_gfa(gfa_per_use_statusquo)
    gfa_per_use_statusquo, rel_ratio_to_res_gfa_target = calc_rel_ratio_to_res_gfa_target(gfa_per_use_statusquo,
                                                                                          PARAMS['UD_scenario'])
    # get future required residential gfa
    future_required_additional_res_gfa = calc_additional_requied_residential_gfa(PARAMS)
    # calculate future required gfa per us
    gfa_per_use_future_required_target = calc_future_required_gfa_per_use(future_required_additional_res_gfa,
                                                                          gfa_per_use_statusquo,
                                                                          rel_ratio_to_res_gfa_target)

    ## calculate target_add_gfa_per_use
    # additional gfa per use
    gfa_per_use_additional_reuqired = gfa_per_use_future_required_target - gfa_per_use_statusquo
    gfa_per_use_additional_reuqired["MULTI_RES"] = gfa_per_use_additional_reuqired["MULTI_RES"] - gfa_res_planned
    gfa_per_use_additional_reuqired[gfa_per_use_additional_reuqired < 0] = 0.0

    # transform parts of SINGLE_RES to MULTI_RES # FIXME: maybe this should be done earlier?
    buildings_SINGLE_RES = list(typology_statusquo.loc[typology_statusquo['1ST_USE'] == 'SINGLE_RES'].index)
    num_buildings_to_MULTI_RES = int(len(buildings_SINGLE_RES) * PARAMS['SINGLE_to_MULTI_RES_ratio'])
    if num_buildings_to_MULTI_RES > 0.0:
        buildings_to_MULTI_RES = random.sample(buildings_SINGLE_RES, num_buildings_to_MULTI_RES)
        extra_gfa_from_SINGLE_RES_conversion, gfa_to_MULTI_RES = 0.0, 0.0
        for b in buildings_to_MULTI_RES:
            building_gfa = typology_statusquo.loc[b]['GFA_m2']
            gfa_to_MULTI_RES += building_gfa
            num_occupants = round(
                building_gfa * PARAMS["ratio_living_space_to_GFA"] / PARAMS["current_SFH_occupant_density"])
            extra_gfa = building_gfa - num_occupants * PARAMS["future_occupant_density"]
            extra_gfa_from_SINGLE_RES_conversion += extra_gfa
            typology_statusquo.loc[b, :] = typology_statusquo.loc[b].replace({"SINGLE_RES": "MULTI_RES"})
        gfa_per_use_future_required_target["SINGLE_RES"] = gfa_per_use_future_required_target[
                                                               "SINGLE_RES"] - gfa_to_MULTI_RES
        gfa_per_use_additional_reuqired["MULTI_RES"] = gfa_per_use_additional_reuqired[
                                                           "MULTI_RES"] - extra_gfa_from_SINGLE_RES_conversion
        assert gfa_per_use_additional_reuqired["MULTI_RES"] > 0.0

    # update target_add_gfa_per_use
    target_add_gfa_per_use = gfa_per_use_additional_reuqired.astype(int).to_dict()
    total_additional_gfa_target = gfa_per_use_additional_reuqired.sum()
    overview["gfa_per_use_future_required_target"] = gfa_per_use_future_required_target.astype(int)
    overview["target_add_gfa_per_use"] = gfa_per_use_additional_reuqired.astype(int)

    ## FINALIZE INPUTS
    # filter out buildings by use
    buildings_filtered_out_by_use, typology_untouched_uses = remove_buildings_by_uses(typology_statusquo,
                                                                                      uses_to_remove=PARAMS[
                                                                                          'USES_UNTOUCH'])

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

    # set constraints
    range_additional_floors_per_building = calc_range_additional_floors_per_building(PARAMS, buildings_kept)
    possible_uses_per_cityzone = update_possible_uses_per_cityzone(rel_ratio_to_res_gfa_target)

    # create random scenarios
    scenarios = amap.randomize_scenarios(
        typology_merged=buildings_kept,
        mapping=possible_uses_per_cityzone,
        use_columns=typology_use_columns(),
        scenario_count=PARAMS['scenario_count'],
    )

    # built area allocation for all scenarios
    optimizations = dict()
    metrics = dict()
    for scenario in scenarios:
        scenario_typology_merged = scenarios[scenario]

        partitions = [[(n, k) for k in m if k != "NONE"]
                      for n, m in scenario_typology_merged[typology_use_columns()].iterrows()]

        sb = [leaf
                         for tree in partitions
                         for leaf in tree if leaf != "NONE"]

        sub_building_idx = ["%s.%i" % (b[0], i)
                            for i, b in enumerate(sb)]

        sub_building_use = {"%s.%i" % (b[0], i): b[1]
                            for i, b in enumerate(sb)}

        print("scenario [%i], total_additional_gfa_target [%.4f]" % (scenario, total_additional_gfa_target))

        footprint_area = scenario_typology_merged.footprint.to_dict()
        sub_footprint_area = {sb: footprint_area[sb.split(".")[0]]
                              for sb in sub_building_idx} # FIXME: check

        building_to_sub_building = defaultdict(list)
        for sb in sub_building_idx:
            b, num = sb.split('.')
            building_to_sub_building[b].append(sb)
        print("len of problem [%i]" % len(sub_building_idx))

        solution = amap.optimize(
            total_additional_gfa_target,
            sub_building_idx,
            PARAMS['min_additional_floors'],
            PARAMS['max_additional_floors'],
            sub_footprint_area,
            building_to_sub_building,
            range_additional_floors_per_building,
            target_add_gfa_per_use,
            sub_building_use
        )

        optimizations[scenario] = {
            "solution": solution,
            "sub_footprint_area": sub_footprint_area,
            "building_to_sub_building": building_to_sub_building
        }
        print("scenario [%i] is-success (if 1): [%i]" % (scenario, solution.sol_status))

        detailed_metrics = amap.detailed_result_metrics(
            solution=solution,
            sub_building_use=sub_building_use,
            sub_footprint_area=sub_footprint_area,
            target_add_gfa_per_use=target_add_gfa_per_use,
            target=total_additional_gfa_target
        )
        metrics[scenario] = detailed_metrics

    # find the best scenario
    best_scenario, scenario_errors = amap.find_optimum_scenario(
        optimizations=optimizations,
        target=total_additional_gfa_target
    )
    overview['result_add_gfa_per_use'] = metrics[best_scenario]['gfa_per_use'].loc['result']

    ## write typology.dbf and zone.shp with the best scenario
    best_typology_df = scenarios[best_scenario].copy()
    # add back those buildings initially filtered out
    best_typology_df = best_typology_df.append(buildings_filtered_out_by_age, sort=True)
    best_typology_df = best_typology_df.append(typology_planned, sort=True)
    best_typology_df = best_typology_df.append(typology_untouched_uses, sort=True)
    result_add_floors = amap.parse_milp_solution(optimizations[best_scenario]["solution"])
    path_to_output_zone_shape_file = sample_data_dir(PARAMS)/ "updated/zone.shp"
    path_to_output_typology_file = sample_data_dir(PARAMS) / "updated/typology.dbf"
    save_best_scenraio(best_typology_df, result_add_floors, building_to_sub_building, typology_statusquo,
                       path_to_output_zone_shape_file, path_to_output_typology_file, PARAMS)

    # get updated data
    prop_geometry = get_prop_geometry(path_to_output_zone_shape_file,
                                      sample_data_dir(PARAMS) / "architecture.dbf")
    typology_updated = dbf_to_dataframe(str(path_to_output_typology_file.absolute())).set_index('Name', drop=False)
    typology_updated = typology_updated.merge(prop_geometry, left_index=True, right_on='Name')
    gfa_per_use_type_updated = calc_gfa_per_use(typology_updated, "GFA_m2")
    overview['gfa_per_use_updated'] = gfa_per_use_type_updated.astype(int)
    overview['gfa_per_ratio_updated'] = round(gfa_per_use_type_updated / gfa_per_use_type_updated.sum(), 5)
    overview['actual_add_gfa_per_use'] = overview['gfa_per_use_updated'] - overview['gfa_per_use_statusquo']
    overview_df = pd.DataFrame(overview)
    overview_df.to_csv(os.path.join(sample_data_dir(PARAMS), PARAMS["UD_scenario"] + "_overview.csv"))

    return


def update_possible_uses_per_cityzone(rel_ratio_to_res_gfa_target):
    possible_uses_per_cityzone = get_possible_uses_per_cityzone()
    for cityzone, uses in possible_uses_per_cityzone.items():
        possible_uses_per_cityzone[cityzone] = uses.intersection(rel_ratio_to_res_gfa_target.index)
    return possible_uses_per_cityzone


def calc_future_required_gfa_per_use(future_required_additional_res_gfa, gfa_per_use_statusquo,
                                     rel_ratio_to_res_gfa_target):
    future_required_res_gfa = (future_required_additional_res_gfa + gfa_per_use_statusquo.filter(like="_RES").sum())
    future_required_gfa_dict = dict()
    for use_type in rel_ratio_to_res_gfa_target.index:
        if use_type == "SINGLE_RES":
            future_required_gfa_dict[use_type] = gfa_per_use_statusquo[use_type]  # unchanged
        elif use_type == "MULTI_RES":
            future_required_gfa_dict[use_type] = future_required_additional_res_gfa + gfa_per_use_statusquo[use_type]
        else:
            future_required_gfa_dict.update(
                {use_type: future_required_res_gfa * rel_ratio_to_res_gfa_target[use_type]})
    future_required_gfa_per_use = pd.Series(future_required_gfa_dict)
    return future_required_gfa_per_use


def combine_statusquo_MULTI_RES_gfa(gfa_per_use_statusquo):
    MULTI_RES_gfa_statusquo = gfa_per_use_statusquo.filter(like='MULTI_RES').sum()
    gfa_per_use_statusquo = gfa_per_use_statusquo.drop(gfa_per_use_statusquo.filter(like="MULTI_RES").index)
    gfa_per_use_statusquo["MULTI_RES"] = MULTI_RES_gfa_statusquo
    return gfa_per_use_statusquo


def calc_rel_ratio_to_res_gfa_target(gfa_per_use_statusquo, scenario):
    gfa_per_use_statusquo_in = gfa_per_use_statusquo.copy()
    if scenario == 'SQ':
        rel_ratio_to_res_gfa_target = (gfa_per_use_statusquo / gfa_per_use_statusquo.filter(like='_RES').sum())
    else:
        target_ratios = read_mapping('SURB', '2040', scenario)
        # unify columns in target_ratios and gfa_per_use_statusquo
        zero_series = pd.DataFrame(0.0, index=range(1), columns=target_ratios.index).loc[0]
        gfa_per_use_statusquo = zero_series.combine(gfa_per_use_statusquo, max)
        # get target rel_ratios
        rel_ratio_to_res_gfa_statusquo = (gfa_per_use_statusquo / gfa_per_use_statusquo.filter(like='_RES').sum())
        rel_ratio_to_res_gfa_target = zero_series.copy()
        for use, target_val in target_ratios.items():
            if np.isclose(target_val, 0.0) and rel_ratio_to_res_gfa_statusquo[use] > 0:
                # if not required in target, keep existing uses in status quo
                rel_ratio_to_res_gfa_target[use] = rel_ratio_to_res_gfa_statusquo[use]
            else:
                rel_ratio_to_res_gfa_target[use] = target_val
    # drop uses with zero ratio
    rel_ratio_to_res_gfa_target = rel_ratio_to_res_gfa_target[rel_ratio_to_res_gfa_target > 0.0]
    gfa_per_use_statusquo = gfa_per_use_statusquo.loc[rel_ratio_to_res_gfa_target.index]
    assert np.isclose(gfa_per_use_statusquo_in.sum(), gfa_per_use_statusquo.sum()) # make sure total gfa is unchanged
    return gfa_per_use_statusquo, rel_ratio_to_res_gfa_target


def calc_range_additional_floors_per_building(PARAMS, typology_status_quo):
    height_limit_per_city_zone = {1: (0, PARAMS['building_height_limit'] // PARAMS['floor_height'])}
    range_additional_floors_per_building = dict()
    for name, building in typology_status_quo.iterrows():
        min_floors, max_floors = height_limit_per_city_zone[building.city_zone]
        range_additional_floors_per_building[name] = [0, max(0, max_floors - building.floors_ag)]
    return range_additional_floors_per_building


def get_planned_res_gfa(PARAMS, gfa_per_use_statusquo):
    future_planned_res_gfa = 0.0
    if PARAMS['MULTI_RES_PLANNED'] in gfa_per_use_statusquo.index:
        future_planned_res_gfa = gfa_per_use_statusquo[PARAMS['MULTI_RES_PLANNED']]
        # gfa_per_use_statusquo.drop(PARAMS['MULTI_RES_USE_TYPE'], inplace=True)
    return future_planned_res_gfa


def sample_data_dir(PARAMS) -> Path:
    p = Path(os.path.join(PARAMS['path'], 'area_mapper\sample_data'))
    if not p.exists():
        raise IOError("data_dir not found [%s]" % p)
    return p


def get_prop_geometry(path_to_zone_shp, path_to_architecture) -> pd.DataFrame:
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


def remove_buildings_by_uses(typology_df: pd.DataFrame, uses_to_remove: list):
    typology_remained_df = typology_df.copy()
    orig_building_names = typology_remained_df.index
    for use in uses_to_remove:
        typology_remained_df = typology_remained_df.drop(
            typology_remained_df[typology_remained_df["1ST_USE"] == use].index)
    building_names_remained = typology_remained_df.index
    typology_removed_df = typology_df.loc[list(set(orig_building_names) - set(building_names_remained))]
    return typology_remained_df, typology_removed_df


def get_sample_data(PARAMS) -> pd.DataFrame:
    """
    merges topology.dbf and architecture.dbf, calculate GFA, and initiates empty columns
    :return:
    """
    path_to_architecture = sample_data_dir(PARAMS) / "architecture.dbf"
    if not path_to_architecture.exists():
        raise IOError("architecture file not found [%s]" % path_to_architecture)
    path_to_zone_shp = sample_data_dir(PARAMS) / "zone.shp"
    if not path_to_zone_shp.exists():
        raise IOError("shape file not found [%s]" % path_to_zone_shp)
    prop_geometries = get_prop_geometry(path_to_zone_shp, path_to_architecture)
    typology = sample_typology_data(PARAMS)
    # write typology_merged
    typology_merged = typology.merge(prop_geometries, left_index=True, right_on='Name')
    typology_merged.floors_ag = typology_merged.floors_ag.astype(int)
    typology_merged["city_zone"] = 1
    typology_merged["additional_floors"] = 0
    typology_merged["floors_ag_updated"] = typology_merged.floors_ag.astype(int)
    typology_merged["height_ag_updated"] = typology_merged.height_ag.astype(int)
    return typology_merged


def typology_use_columns() -> List[str]:
    return ["1ST_USE", "2ND_USE", "3RD_USE"]


def get_possible_uses_per_cityzone() -> Dict[int, Set[str]]:
    """
    possible building uses per city zone
    :return:
    """
    return {
        1: {"SINGLE_RES", "MULTI_RES", "RETAIL", "NONE", "HOSPITAL", "INDUSTRIAL",
            "GYM", "SCHOOL", "PARKING", "LIBRARY", "FOODSTORE", "RESTAURANT", "HOTEL",
            "OFFICE", "MUSEUM", "SERVERROOM", "SWIMMING", "UNIVERSITY", "COOLROOM"},
    }


def calc_gfa_per_use(typology_merged: pd.DataFrame, GFA_type="GFA_m2"):
    """
    calculates GFA per use type based on the 1st use, 2nd use and 3rd use [m2]
    :param typology_merged:
    :return:
    """
    typology_merged["GFA_1ST_USE"] = typology_merged["1ST_USE_R"] * typology_merged[GFA_type]
    typology_merged["GFA_2ND_USE"] = typology_merged["2ND_USE_R"] * typology_merged[GFA_type]
    typology_merged["GFA_3RD_USE"] = typology_merged["3RD_USE_R"] * typology_merged[GFA_type]

    gfa_series_1st_use = typology_merged.groupby("1ST_USE").sum().loc[:, "GFA_1ST_USE"]
    gfa_series_2nd_use = typology_merged.groupby("2ND_USE").sum().loc[:, "GFA_2ND_USE"]
    gfa_series_3rd_use = typology_merged.groupby("3RD_USE").sum().loc[:, "GFA_3RD_USE"]

    gfa_per_use_type = defaultdict(float)
    for use_series in [gfa_series_1st_use, gfa_series_2nd_use, gfa_series_3rd_use]:
        for use, val in use_series.iteritems():
            gfa_per_use_type[use] += val

    # get rid of the unwanted "NONE" use-type
    if "NONE" in gfa_per_use_type.keys():
        del gfa_per_use_type["NONE"]

    return pd.Series(gfa_per_use_type)


def calc_additional_requied_residential_gfa(PARAMS):
    future_required_additional_res_living_space = PARAMS['additional_population'] * PARAMS['future_occupant_density']
    future_required_additional_res_gfa = future_required_additional_res_living_space / PARAMS[
        'ratio_living_space_to_GFA']
    return future_required_additional_res_gfa


def remove_updated_files(PARAMS, clean=True):
    updated_folder = sample_data_dir(PARAMS)/"updated"
    if clean and updated_folder.exists():
        data_files = os.listdir(updated_folder)
        for file in data_files:
            # if file.find("_update") >= 0:
            #     print("cleaning: [%s]" % (sample_data_dir(PARAMS) / file))
            #     Path(sample_data_dir(PARAMS) / file).unlink()
            print("cleaning: [%s]" % (updated_folder / file))
            Path(updated_folder / file).unlink()
    else:
        os.mkdir(updated_folder)
    return


def read_mapping(district_archetype, year, urban_development_scenario):
    """
    read numbers from table
    :return:
    """
    worksheet = f"{district_archetype}_{year}"
    print(f"Reading mappings from worksheet {worksheet}")
    path = Path.cwd().parent
    path_to_mapping_table = os.path.join('', *[path, "remap_ville_plugin", "mapping_BUILDING_USE_RATIO.xlsx"])
    mapping_df = pd.read_excel(path_to_mapping_table, sheet_name=worksheet).set_index("Scenario")
    rel_ratio_to_res_gfa_per_use = mapping_df.loc[urban_development_scenario].drop("Reference")
    return rel_ratio_to_res_gfa_per_use


def save_best_scenraio(best_typology_df, result_add_floors, building_to_sub_building, typology_statusquo,
                       path_to_output_zone_shape_file, path_to_output_typology_file, PARAMS):
    # update zone.shp
    path_to_input_zone_shape_file = sample_data_dir(PARAMS) / "zone.shp"
    floors_ag_updated, zone_shp_updated = amap.update_zone_shp(best_typology_df,
                                                          result_add_floors,
                                                          building_to_sub_building,
                                                          path_to_input_zone_shape_file,
                                                          path_to_output_zone_shape_file)
    # update typology.dbf
    amap.update_typology_dbf(best_typology_df, result_add_floors, building_to_sub_building, typology_statusquo,
                        zone_shp_updated, floors_ag_updated, path_to_output_typology_file, PARAMS)

if __name__ == "__main__":
    main()
