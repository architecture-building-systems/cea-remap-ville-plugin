"""
Initialize a new scenario based on a old scenario.
"""
import pandas as pd
import numpy as np
import os
import json
import operator
import geopandas as gpd
from pathlib import Path
from collections import defaultdict

import cea.config
import cea.inputlocator
from cea.demand.building_properties import calc_useful_areas
from cea.utilities.dbf import dbf_to_dataframe
from remap_ville_plugin.utilities import calc_gfa_per_use, typology_use_columns, count_usetype, save_updated_typology
import remap_ville_plugin.area_optimization_mapper as amap
from remap_ville_plugin.remap_setup_scenario import update_config
import remap_ville_plugin.urban_transformation_preprocessing as preprocessing

__author__ = "Anastasiya Popova, Shanshan Hsieh"
__copyright__ = "Copyright 2021, Architecture and Building Systems - ETH Zurich"
__credits__ = ["Anastasiya Popova, Shanshan Hsieh"]
__license__ = "MIT"
__version__ = "0.1"
__maintainer__ = "Shanshan Hsieh"
__email__ = "cea@arch.ethz.ch"
__status__ = "Production"

from urban_transformation_preprocessing import remove_buildings_by_uses

PARAMS = {
    'MULTI_RES_PLANNED': 'MULTI_RES_2040',
    'additional_population': 1140,  # people
    'current_SFH_occupant_density': 150,  # living space m2/occupants # FIXME: read from input scenario
    'future_occupant_density': 80,  # living space m2/occupants # FIXME: get from CH_ReMaP
    'USES_UNTOUCH': ['SINGLE_RES'],
    'SINGLE_to_MULTI_RES_ratio': 0.0,
    'preserve_buildings_built_before': 1920,
    'building_height_limit': 42,  # m
    # constants
    'ratio_living_space_to_GFA': 0.82,
    'floor_height': 3,  # m
    'min_additional_floors': 0,
    'max_additional_floors': 50,
    'scenario_count': 10  # FIXME: advanced config parameter
}

PARAMS = {
    'scenario_count': 10,
    'lower_bound_floors': 0,
    'upper_bound_floors': 50,
    'floor_height': 3,
    'ratio_living_space_to_GFA': 0.82,
}


def main(config, case_study_inputs_df):
    config, new_scenario_name, urban_development_scenario, year = update_config(config)
    ## Save PARAMS
    with open(os.path.join(config.scenario, str(new_scenario_name) + "_PARAMS.json"), "w") as fp:
        json.dump(PARAMS, fp)
    case_study_inputs = case_study_inputs_df.loc[float(year)]
    ## 1. Gather input data
    new_locator = cea.inputlocator.InputLocator(scenario=config.scenario, plugins=config.plugins)
    typology_merged = get_sample_data(new_locator)
    existing_uses = read_existing_uses(typology_merged)
    # exclude planned buildings
    typology_statusquo = typology_merged.copy()
    gfa_per_use_future_target, gfa_per_use_additional_target, gfa_total_additional_target, \
    overview, rel_ratio_to_res_gfa_target, \
    typology_planned, typology_statusquo = preprocessing.main(config, typology_statusquo, case_study_inputs)

    ## 3. Finalize Inputs
    # filter out buildings by use
    typology_kept_uses, typology_untouched_uses = remove_buildings_by_uses(typology_statusquo,
                                                                           uses_to_remove=case_study_inputs[
                                                                               'USES_UNTOUCH'])
    # filter out buildings by year
    typology_preserved_year, typology_after_year = filter_buildings_by_year(typology_kept_uses, year=case_study_inputs[
        "preserve_buildings_built_before"])
    # final typology_input to be optimized
    typology_input = typology_after_year.copy()
    # set constraints
    range_additional_floors_per_building = calc_range_additional_floors_per_building(typology_input, case_study_inputs)
    possible_uses_per_cityzone = update_possible_uses_per_cityzone(rel_ratio_to_res_gfa_target)
    # create random scenarios
    scenarios = amap.randomize_scenarios(typology_merged=typology_input, usetype_constraints=possible_uses_per_cityzone,
                                         use_columns=typology_use_columns(), scenario_count=PARAMS['scenario_count'])
    ## 4. Optimize Urban Transformation
    metrics, op_solutions = optimize_all_scenarios(range_additional_floors_per_building, scenarios,
                                                   gfa_per_use_additional_target, gfa_total_additional_target,
                                                   case_study_inputs)
    ## 5. Reconstruct District with the Best Scenario
    best_scenario, scenario_errors = amap.find_optimum_scenario(op_solutions=op_solutions,
                                                                target=gfa_total_additional_target)
    overview['result_add_gfa_per_use'] = metrics[best_scenario]['gfa_per_use'].loc['result']
    best_typology_df = scenarios[best_scenario].copy()
    # add back those buildings that got filtered out
    best_typology_df = best_typology_df.append(typology_preserved_year, sort=True)
    best_typology_df = best_typology_df.append(typology_planned, sort=True)
    best_typology_df = best_typology_df.append(typology_untouched_uses, sort=True)
    # get floors added per building
    result_add_floors = amap.parse_milp_solution(op_solutions[best_scenario]["solution"])
    save_best_scenario(best_typology_df, result_add_floors, op_solutions[best_scenario]['building_to_sub_building'],
                       typology_statusquo, new_locator, year)

    ## 6. Check results and save overview_df
    save_updated_typology_to_overview(new_locator, new_scenario_name, overview)

    return


def save_updated_typology_to_overview(new_locator, new_scenario_name, overview):
    prop_geometry = get_prop_geometry(new_locator)
    typology_updated = dbf_to_dataframe(new_locator.get_building_typology()).set_index('Name', drop=False)
    typology_updated = typology_updated.merge(prop_geometry, left_index=True, right_on='Name')
    use_count_df = count_usetype(typology_updated)
    gfa_per_use_updated = calc_gfa_per_use(typology_updated, "GFA_m2")
    overview['gfa_per_use_updated'] = gfa_per_use_updated.astype(int)
    overview['gfa_per_use_ratio_updated'] = round(gfa_per_use_updated / gfa_per_use_updated.sum(), 5)
    overview['actual_add_gfa_per_use'] = overview['gfa_per_use_updated'] - overview['gfa_per_use_statusquo']
    overview['diff_add_gfa_per_use'] = overview['gfa_per_use_additional_target'] - overview['actual_add_gfa_per_use']
    overview_df = pd.DataFrame(overview)
    overview_df = pd.concat([overview_df, use_count_df], axis=1)
    overview_df.fillna(0.0, inplace=True)
    # TODO: fill na with 0
    overview_df.to_csv(os.path.join(new_locator.get_input_folder(), new_scenario_name + "_overview.csv"))
    return


def update_zone_shp(best_typology_df, result_add_floors, building_to_sub_building, path_to_input_zone_shape_file,
                    path_to_output_zone_shape_file):
    # update floors_ag and height_ag
    floors_ag_updated, height_ag_updated = defaultdict(int), defaultdict(int)
    for b, sb in building_to_sub_building.items():
        floors_ag_updated[b] = sum([result_add_floors[_sb] for _sb in sb])
        height_ag_updated[b] = floors_ag_updated[b] * 3
        best_typology_df.loc[b, "additional_floors"] = floors_ag_updated[b]
        best_typology_df.loc[b, "floors_ag_updated"] = best_typology_df.floors_ag[b] + floors_ag_updated[b]
        best_typology_df.loc[b, "height_ag_updated"] = best_typology_df.height_ag[b] + height_ag_updated[b]
    # save zone_shp_updated
    zone_shp_updated = gpd.read_file(str(path_to_input_zone_shape_file))
    zone_shp_updated = zone_shp_updated.set_index("Name")
    zone_shp_updated["floors_ag"] = best_typology_df["floors_ag_updated"]
    zone_shp_updated["height_ag"] = best_typology_df["height_ag_updated"]
    zone_shp_updated["REFERENCE"] = "after-optimization"
    zone_shp_updated.to_file(path_to_output_zone_shape_file)
    return floors_ag_updated, zone_shp_updated


def save_best_scenario(best_typology_df, result_add_floors, building_to_sub_building, typology_statusquo, new_locator,
                       year):
    # update zone.shp
    path_to_output_zone_shp = Path(new_locator.get_zone_geometry())
    path_to_output_typology_dbf = Path(new_locator.get_building_typology())

    floors_ag_updated, zone_shp_updated = update_zone_shp(best_typology_df,
                                                          result_add_floors,
                                                          building_to_sub_building,
                                                          path_to_output_zone_shp,  # FIXME: redundant
                                                          path_to_output_zone_shp)
    # update typology.dbf
    update_typology_dbf(best_typology_df, result_add_floors, building_to_sub_building, typology_statusquo,
                        zone_shp_updated, floors_ag_updated, path_to_output_typology_dbf, year)


def update_typology_dbf(best_typology_df, result_add_floors, building_to_sub_building, typology_statusquo,
                        zone_shp_updated, floors_ag_updated, path_to_output_typology_file, year):
    status_quo_typology = typology_statusquo.copy()
    simulated_typology = best_typology_df.copy()

    zone_updated_gfa_per_building = zone_shp_updated.area * (
            zone_shp_updated['floors_ag'] + zone_shp_updated['floors_bg'])

    # update usetype ratios
    simulated_typology["1ST_USE_R"] = simulated_typology["1ST_USE_R"].astype(float)
    simulated_typology["2ND_USE_R"] = simulated_typology["2ND_USE_R"].astype(float)
    simulated_typology["3RD_USE_R"] = simulated_typology["3RD_USE_R"].astype(float)
    simulated_typology["REFERENCE"] = status_quo_typology["REFERENCE_x"]
    use_col_dict = {i: column for i, column in enumerate(["1ST_USE", "2ND_USE", "3RD_USE"])}
    for b, sb in building_to_sub_building.items():
        updated_floor_per_use_col = dict()
        current_floors = status_quo_typology.loc[b, "floors_ag"] + status_quo_typology.loc[b, "floors_bg"]
        total_additional_floors = sum([result_add_floors[y] for y in sb])
        assert np.isclose(total_additional_floors, floors_ag_updated[b])
        updated_floors = current_floors + total_additional_floors
        total_gfa = updated_floors * status_quo_typology.footprint[b]
        assert np.isclose(zone_updated_gfa_per_building.loc[b], total_gfa)
        # get updated_floor_per_use_col
        for i, sub_building in enumerate(sb):
            sub_building_additional_floors = result_add_floors[sub_building]
            current_ratio = status_quo_typology.loc[b, use_col_dict[i] + '_R']
            updated_floor_per_use_col[use_col_dict[i]] = (
                    sub_building_additional_floors + (current_ratio * current_floors))
        if not np.isclose(updated_floors, sum(updated_floor_per_use_col.values())):
            raise ValueError("total number of floors mis-match excpeted number of floors")
        # write updated usetype ratio
        for use_col in updated_floor_per_use_col:
            use_statusquo = typology_statusquo.loc[b, use_col]
            r_statusquo = typology_statusquo.loc[b, use_col+'_R']
            use_updated = simulated_typology.loc[b, use_col]
            r_updated = updated_floor_per_use_col[use_col] / updated_floors
            if np.isclose(r_updated, 0.0):
                use_updated = "NONE"
                simulated_typology.loc[b, use_col] = use_updated
            if r_updated > 0 and (use_statusquo, r_statusquo, current_floors) != (use_updated, r_updated, updated_floors):
                # building that changes usetype or ratios
                simulated_typology.loc[b, use_col + '_R'] = r_updated
                simulated_typology.loc[b, 'YEAR'] = year
                # if simulated_typology.loc[b, use_col] == 'MULTI_RES':
                #     # update MULTI_RES use-type properties
                #     if updated_floor_per_use_col[use_col] > 0 or status_quo_typology.loc[b, use_col] == 'SINGLE_RES':
                #         simulated_typology.loc[b, use_col] = 'MULTI_RES_2040'  # FIXME: hard-coded, MAYBE REDUNDANT

    save_updated_typology(path_to_output_typology_file, simulated_typology)


def optimize_all_scenarios(range_additional_floors_per_building, scenarios, target_add_gfa_per_use,
                           total_additional_gfa_target, case_study_inputs):
    op_solutions = dict()
    metrics = dict()
    for scenario in scenarios:
        print("scenario [%i], total_additional_gfa_target [%.4f]" % (scenario, total_additional_gfa_target))
        scenario_typology = scenarios[scenario]
        # get sub-buildings
        partitions = [[(n, k) for k in m if k != "NONE"]
                      for n, m in scenario_typology[typology_use_columns()].iterrows()]
        sb = [leaf
              for tree in partitions
              for leaf in tree if leaf != "NONE"]
        sub_building_idx = ["%s.%i" % (b[0], i)
                            for i, b in enumerate(sb)]
        sub_building_use = {"%s.%i" % (b[0], i): b[1]
                            for i, b in enumerate(sb)}
        footprint_area = scenario_typology.footprint.to_dict()
        sub_building_footprint_area = {sb: footprint_area[sb.split(".")[0]] for sb in sub_building_idx}
        building_to_sub_building = defaultdict(list)
        for sb in sub_building_idx:
            b, num = sb.split('.')
            building_to_sub_building[b].append(sb)
        print("len of problem [%i]" % len(sub_building_idx))
        # floor area optimization
        solution = amap.optimize(
            total_additional_gfa_target,
            sub_building_idx,
            PARAMS['lower_bound_floors'],
            PARAMS['upper_bound_floors'],
            sub_building_footprint_area,
            building_to_sub_building,
            range_additional_floors_per_building,
            target_add_gfa_per_use,
            sub_building_use
        )
        print("\n scenario [%i] is-success (if 1): [%i]" % (scenario, solution.sol_status))
        # save solutions
        op_solutions[scenario] = {
            "solution": solution,
            "sub_building_footprint_area": sub_building_footprint_area,
            "building_to_sub_building": building_to_sub_building
        }
        detailed_metrics = amap.detailed_result_metrics(
            solution=solution,
            sub_building_use=sub_building_use,
            sub_building_footprint_area=sub_building_footprint_area,
            target_add_gfa_per_use=target_add_gfa_per_use,
            target=total_additional_gfa_target
        )
        metrics[scenario] = detailed_metrics
    return metrics, op_solutions


def update_possible_uses_per_cityzone(rel_ratio_to_res_gfa_target):
    possible_uses_per_cityzone = get_possible_uses_per_cityzone()
    for cityzone, uses in possible_uses_per_cityzone.items():
        possible_uses_per_cityzone[cityzone] = uses.intersection(rel_ratio_to_res_gfa_target.index)
    return possible_uses_per_cityzone


def get_possible_uses_per_cityzone():
    """
    possible building uses per city zone
    :return:
    """
    return {
        0: {"SINGLE_RES", "MULTI_RES", "RETAIL", "NONE", "HOSPITAL", "INDUSTRIAL",
            "GYM", "SCHOOL", "PARKING", "LIBRARY", "FOODSTORE", "RESTAURANT", "HOTEL",
            "MUSEUM", "SWIMMING", "UNIVERSITY"},
        1: {"SINGLE_RES", "MULTI_RES", "RETAIL", "NONE", "HOSPITAL", "INDUSTRIAL",
            "GYM", "SCHOOL", "PARKING", "LIBRARY", "FOODSTORE", "RESTAURANT", "HOTEL",
            "OFFICE", "MUSEUM", "SERVERROOM", "SWIMMING", "UNIVERSITY", "COOLROOM"},
        2: {"SINGLE_RES", "MULTI_RES", "RETAIL", "NONE", "HOSPITAL", "INDUSTRIAL",
            "GYM", "SCHOOL", "PARKING", "LIBRARY", "FOODSTORE", "RESTAURANT", "HOTEL",
            "OFFICE", "MUSEUM", "SERVERROOM", "SWIMMING", "UNIVERSITY", "COOLROOM"},
        3: {"SINGLE_RES", "MULTI_RES", "RETAIL", "NONE", "HOSPITAL", "INDUSTRIAL",
            "GYM", "SCHOOL", "PARKING", "LIBRARY", "FOODSTORE", "RESTAURANT", "HOTEL",
            "OFFICE", "MUSEUM", "SERVERROOM", "SWIMMING", "UNIVERSITY", "COOLROOM"},
    }


def calc_range_additional_floors_per_building(typology_status_quo, case_study_inputs):
    height_limit_per_city_zone = {
        0: (0, case_study_inputs['building_height_limit'] // PARAMS['floor_height']),
        1: (0, case_study_inputs['building_height_limit'] // PARAMS['floor_height']),
        2: (0, case_study_inputs['building_height_limit'] // PARAMS['floor_height']),
        3: (0, case_study_inputs['building_height_limit'] // PARAMS['floor_height'])}  # FIXME
    # height_limit_per_city_zone = {0: (0, 8), 1: (0, 26), 2: (0, 26),
    #                               3: (0, 13)}  # FIXME: this only applies to Altstetten
    range_additional_floors_per_building = dict()
    for name, building in typology_status_quo.iterrows():
        min_floors, max_floors = height_limit_per_city_zone[building.city_zone]
        range_additional_floors_per_building[name] = [0, max(0, max_floors - building.floors_ag)]
    return range_additional_floors_per_building


def filter_buildings_by_year(typology_df: pd.DataFrame, year: int):
    if "YEAR" not in typology_df:
        raise KeyError("provided data frame is missing the column 'YEAR'")
    typology_before_year = typology_df[operator.lt(typology_df.YEAR, year + 1)]
    typology_after_year = typology_df[operator.gt(typology_df.YEAR, year)]
    return typology_before_year, typology_after_year


def read_existing_uses(typology_merged):
    existing_uses = set([leaf for tree in typology_merged[typology_use_columns()].values for leaf in tree])
    valid_use_types = ["SINGLE_RES", "MULTI_RES", "SECONDARY_RES", "RETAIL", "NONE", "HOSPITAL", "INDUSTRIAL", "GYM",
                       "SCHOOL", "PARKING", "LIBRARY", "FOODSTORE", "RESTAURANT", "HOTEL", "OFFICE", "MUSEUM",
                       "SERVERROOM", "SWIMMING", "UNIVERSITY", "COOLROOM", "MULTI_RES_2040"]  # TODO: config?
    assert all([True if use in valid_use_types else False for use in existing_uses])  # check valid uses
    return existing_uses


def get_sample_data(new_locator):
    """
    merges topology.dbf and architecture.dbf, calculate GFA, and initiates empty columns
    :return:
    """
    prop_geometries_df = get_prop_geometry(new_locator)
    typology_df = dbf_to_dataframe(new_locator.get_building_typology()).set_index('Name', drop=False)
    # write typology_merged
    typology_merged = typology_df.merge(prop_geometries_df, left_index=True, right_on='Name')
    typology_merged.floors_ag = typology_merged.floors_ag.astype(int)
    # initialize columns
    typology_merged["additional_floors"] = 0
    typology_merged["floors_ag_updated"] = typology_merged.floors_ag.astype(int)
    typology_merged["height_ag_updated"] = typology_merged.height_ag.astype(int)
    return typology_merged


def get_prop_geometry(locator):
    """
    combines zone.shp and architecture.dbf and calculate GFA
    :return:
    """
    architecture_dbf = dbf_to_dataframe(locator.get_building_architecture()).set_index('Name')
    zone_gdf = gpd.read_file(locator.get_zone_geometry())
    if not 'city_zone' in zone_gdf.columns:
        zone_gdf['city_zone'] = 1  # TODO: temp fix
    prop_geometry = zone_gdf.copy()
    prop_geometry['footprint'] = zone_gdf.area
    prop_geometry['GFA_m2'] = prop_geometry['footprint'] * (prop_geometry['floors_ag'] + prop_geometry['floors_bg'])
    prop_geometry['GFA_ag_m2'] = prop_geometry['footprint'] * prop_geometry['floors_ag']
    prop_geometry['GFA_bg_m2'] = prop_geometry['footprint'] * prop_geometry['floors_bg']
    prop_geometry = prop_geometry.merge(architecture_dbf, on='Name').set_index('Name')
    prop_geometry = calc_useful_areas(prop_geometry)

    return prop_geometry
