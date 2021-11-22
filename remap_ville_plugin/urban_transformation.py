"""
Initialize a new scenario based on a old scenario.
"""
import pandas as pd
import numpy as np
import os
import json
import random
import operator
import geopandas as gpd
from pathlib import Path
from collections import defaultdict

import cea.config
import cea.inputlocator
from cea.demand.building_properties import calc_useful_areas
from cea.utilities.dbf import dbf_to_dataframe, dataframe_to_dbf
from remap_ville_plugin.utilities import calc_gfa_per_use, typology_use_columns, count_usetype
import remap_ville_plugin.area_optimization_mapper as amap

__author__ = "Anastasiya Popova, Shanshan Hsieh"
__copyright__ = "Copyright 2021, Architecture and Building Systems - ETH Zurich"
__credits__ = ["Anastasiya Popova, Shanshan Hsieh"]
__license__ = "MIT"
__version__ = "0.1"
__maintainer__ = "Shanshan Hsieh"
__email__ = "cea@arch.ethz.ch"
__status__ = "Production"

PARAMS = {
    'MULTI_RES_PLANNED': 'MULTI_RES_2040',
    'additional_population': 114,  # people
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


def main(config):
    district_archetype = config.remap_ville_scenarios.district_archetype
    year = config.remap_ville_scenarios.year
    urban_development_scenario = config.remap_ville_scenarios.urban_development_scenario

    new_scenario_name = f"{year}_{urban_development_scenario}"
    config.scenario_name = new_scenario_name
    with open(os.path.join(config.scenario, str(new_scenario_name) + "_PARAMS.json"), "w") as fp:
        json.dump(PARAMS, fp)
    new_locator = cea.inputlocator.InputLocator(scenario=config.scenario, plugins=config.plugins)
    ## gather input data
    typology_merged = get_sample_data(new_locator)
    existing_uses = read_existing_uses(typology_merged)
    ## get typology_statusquo and typology_planned (dont touch)
    typology_statusquo = typology_merged.copy()
    typology_statusquo, typology_planned = remove_buildings_by_uses(typology_statusquo,
                                                                    uses_to_remove=[PARAMS['MULTI_RES_PLANNED']])
    ## get overview
    gfa_per_use_statusquo = calc_gfa_per_use(typology_statusquo, "GFA_m2")
    gfa_per_use_planned = calc_gfa_per_use(typology_planned, "GFA_m2") if typology_planned else None
    gfa_res_planned = gfa_per_use_planned[PARAMS['MULTI_RES_PLANNED']] if gfa_per_use_planned else 0.0
    overview = {}
    overview["gfa_per_use_statusquo"] = gfa_per_use_statusquo.astype(int)
    overview["gfa_ratio_per_use_statusquo"] = round(gfa_per_use_statusquo / gfa_per_use_statusquo.sum(), 5)
    overview["gfa_per_use_planned"] = gfa_per_use_planned.astype(int) if gfa_per_use_planned else 0.0

    # TODO: KEEP "FUTURE RESERVED AREA" (ONLY FOOTPRINTS BUT NO HEIGHT) TO BUILD MULTI_RES

    ## SET TARGET GFA RATIOS
    # get relative ratio
    # total_additional_gfa_target
    gfa_per_use_statusquo = combine_MULTI_RES_gfa(gfa_per_use_statusquo)  # FIXME: redundant?
    gfa_per_use_statusquo, rel_ratio_to_res_gfa_target = calc_rel_ratio_to_res_gfa_target(gfa_per_use_statusquo, config)
    # get future required residential gfa
    future_required_additional_res_gfa = calc_additional_requied_residential_gfa(PARAMS)
    # calculate future required gfa per use
    gfa_per_use_future_required_target = calc_future_required_gfa_per_use(future_required_additional_res_gfa,
                                                                          gfa_per_use_statusquo,
                                                                          rel_ratio_to_res_gfa_target)

    ## calculate target_add_gfa_per_use
    # additional gfa per use
    gfa_per_use_additional_reuqired = gfa_per_use_future_required_target - gfa_per_use_statusquo
    gfa_per_use_additional_reuqired["MULTI_RES"] = gfa_per_use_additional_reuqired["MULTI_RES"] - gfa_res_planned
    gfa_per_use_additional_reuqired[gfa_per_use_additional_reuqired < 0] = 0.0

    # transform part of SECONDARY_RES to MULTI_RES
    gfa_per_use_additional_reuqired, \
    gfa_per_use_future_required_target, \
    typology_statusquo = convert_SECONDARY_RES(gfa_per_use_additional_reuqired, gfa_per_use_future_required_target,
                                               typology_statusquo, PARAMS)

    # transform parts of SINGLE_RES to MULTI_RES # FIXME: maybe this should be done earlier?
    gfa_per_use_additional_reuqired, \
    gfa_per_use_future_required_target, \
    typology_statusquo = convert_SINGLE_RES(gfa_per_use_additional_reuqired, gfa_per_use_future_required_target,
                                            typology_statusquo)

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
    buildings_filtered_out_by_age = filter_buildings_by_year(
        buildings_filtered_out_by_use,
        year=PARAMS["preserve_buildings_built_before"] + 1,
        less_than=True
    ).copy()

    buildings_kept = filter_buildings_by_year(
        buildings_filtered_out_by_use,
        year=PARAMS["preserve_buildings_built_before"],
        less_than=False
    ).copy()
    # set constraints
    range_additional_floors_per_building = calc_range_additional_floors_per_building(buildings_kept)
    possible_uses_per_cityzone = update_possible_uses_per_cityzone(rel_ratio_to_res_gfa_target)
    # create random scenarios
    scenarios = amap.randomize_scenarios(typology_merged=buildings_kept, usetype_constraints=possible_uses_per_cityzone,
                                         use_columns=typology_use_columns(), scenario_count=PARAMS['scenario_count'])

    ## OPTIMIZE ALL SCENARIOS
    metrics, optimizations = optimize_all_scenarios(range_additional_floors_per_building, scenarios,
                                                    target_add_gfa_per_use, total_additional_gfa_target)

    # find the best scenario
    print("getting the best scenario...")
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
    save_best_scenario(best_typology_df, result_add_floors, optimizations[best_scenario]['building_to_sub_building'],
                       typology_statusquo, new_locator)

    # get updated data
    prop_geometry = get_prop_geometry(new_locator)
    typology_updated = dbf_to_dataframe(new_locator.get_building_typology()).set_index('Name', drop=False)
    typology_updated = typology_updated.merge(prop_geometry, left_index=True, right_on='Name')
    use_count_df = count_usetype(typology_updated)
    gfa_per_use_type_updated = calc_gfa_per_use(typology_updated, "GFA_m2")
    overview['gfa_per_use_updated'] = gfa_per_use_type_updated.astype(int)
    overview['gfa_per_ratio_updated'] = round(gfa_per_use_type_updated / gfa_per_use_type_updated.sum(), 5)
    overview['actual_add_gfa_per_use'] = overview['gfa_per_use_updated'] - overview['gfa_per_use_statusquo']
    overview['diff_add_gfa_per_use'] = overview['target_add_gfa_per_use'] - overview['actual_add_gfa_per_use']
    overview_df = pd.DataFrame(overview)
    overview_df = pd.concat([overview_df, use_count_df], axis=1)
    overview_df.to_csv(os.path.join(new_locator.get_input_folder(), new_scenario_name + "_overview.csv"))

    return


def convert_SINGLE_RES(gfa_per_use_additional_reuqired, gfa_per_use_future_required_target, typology_statusquo):
    buildings_SINGLE_RES = list(typology_statusquo.loc[typology_statusquo['1ST_USE'] == 'SINGLE_RES'].index)
    num_buildings_to_MULTI_RES = int(len(buildings_SINGLE_RES) * PARAMS['SINGLE_to_MULTI_RES_ratio'])
    if num_buildings_to_MULTI_RES > 0.0:
        print('Converting...', num_buildings_to_MULTI_RES, 'SINGLE_RES to MULTI_RES')
        buildings_to_MULTI_RES = random.sample(buildings_SINGLE_RES, num_buildings_to_MULTI_RES)
        extra_gfa_from_SINGLE_RES_conversion, gfa_to_MULTI_RES = 0.0, 0.0
        for b in buildings_to_MULTI_RES:
            building_gfa = typology_statusquo.loc[b]['GFA_m2']
            gfa_to_MULTI_RES += building_gfa
            num_occupants = round(
                building_gfa * PARAMS["ratio_living_space_to_GFA"] / PARAMS["current_SFH_occupant_density"])
            extra_gfa = building_gfa - num_occupants * (
                    PARAMS["future_occupant_density"] / PARAMS["ratio_living_space_to_GFA"])
            extra_gfa_from_SINGLE_RES_conversion += extra_gfa
            typology_statusquo.loc[b, :] = typology_statusquo.loc[b].replace({"SINGLE_RES": "MULTI_RES"})
        gfa_per_use_future_required_target["SINGLE_RES"] = gfa_per_use_future_required_target[
                                                               "SINGLE_RES"] - gfa_to_MULTI_RES
        gfa_per_use_additional_reuqired["MULTI_RES"] = gfa_per_use_additional_reuqired[
                                                           "MULTI_RES"] - extra_gfa_from_SINGLE_RES_conversion
        assert gfa_per_use_additional_reuqired["MULTI_RES"] > 0.0

    return gfa_per_use_additional_reuqired, gfa_per_use_future_required_target, typology_statusquo


def convert_SECONDARY_RES(gfa_per_use_additional_reuqired, gfa_per_use_future_required_target,
                          typology_statusquo, PARAMS):
    buildings_SECONDARY_RES = list(typology_statusquo.loc[typology_statusquo['1ST_USE'] == 'SECONDARY_RES'].index)
    SECONDARY_gfa = typology_statusquo.loc[buildings_SECONDARY_RES]['GFA_m2'].sum()
    additional_required_MULTI_RES_gfa = gfa_per_use_additional_reuqired['MULTI_RES']
    if SECONDARY_gfa > additional_required_MULTI_RES_gfa * 2:
        results_dict = {}
        for i in range(1000):
            num_sampled_buildings = random.randrange(0, 50)
            sampled_buildings = random.sample(set(buildings_SECONDARY_RES), num_sampled_buildings)
            sampled_gfa = typology_statusquo.loc[sampled_buildings]['GFA_m2'].sum()
            delta_gfa = round(sampled_gfa - additional_required_MULTI_RES_gfa, 2)
            buffer = PARAMS['future_occupant_density'] * 2
            if delta_gfa < buffer and delta_gfa > 0.0:
                results_dict[delta_gfa] = sampled_buildings
        buildings_to_MULTI_RES = results_dict[min(results_dict.keys())]
        print('Converting...', len(buildings_to_MULTI_RES), 'SECONDARY_RES to MULTI_RES')
        typology_statusquo.loc[buildings_to_MULTI_RES, '1ST_USE'] = 'MULTI_RES'
        SECONDARY_to_MULTI_RES_gfa = typology_statusquo.loc[buildings_to_MULTI_RES]['GFA_m2'].sum()
        gfa_per_use_additional_reuqired['MULTI_RES'] = max(
            additional_required_MULTI_RES_gfa - SECONDARY_to_MULTI_RES_gfa, 0)
    else:
        SECONDARY_to_MULTI_RES_gfa = 0.0
    # update targets
    gfa_per_use_future_required_target["SECONDARY_RES"] = gfa_per_use_future_required_target["SECONDARY_RES"] \
                                                          - SECONDARY_to_MULTI_RES_gfa
    return gfa_per_use_additional_reuqired, gfa_per_use_future_required_target, typology_statusquo


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


def save_best_scenario(best_typology_df, result_add_floors, building_to_sub_building, typology_statusquo, new_locator):
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
                        zone_shp_updated, floors_ag_updated, path_to_output_typology_dbf)


def update_typology_dbf(best_typology_df, result_add_floors, building_to_sub_building, typology_statusquo,
                        zone_shp_updated, floors_ag_updated, path_to_output_typology_file):
    status_quo_typology = typology_statusquo.copy()
    simulated_typology = best_typology_df.copy()

    zone_updated_gfa_per_building = zone_shp_updated.area * (
            zone_shp_updated['floors_ag'] + zone_shp_updated['floors_bg'])

    simulated_typology["1ST_USE_R"] = simulated_typology["1ST_USE_R"].astype(float)
    simulated_typology["2ND_USE_R"] = simulated_typology["2ND_USE_R"].astype(float)
    simulated_typology["3RD_USE_R"] = simulated_typology["3RD_USE_R"].astype(float)
    simulated_typology["REFERENCE"] = "after-optimization"  # FIXME: change year as well
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
        # write update ratio
        for use_col in updated_floor_per_use_col:
            r = updated_floor_per_use_col[use_col] / updated_floors
            simulated_typology.loc[b, use_col + '_R'] = r
            if np.isclose(r, 0.0):
                simulated_typology.loc[b, use_col] = "NONE"
            if simulated_typology.loc[b, use_col] == 'MULTI_RES':
                # update MULTI_RES use-type properties
                if updated_floor_per_use_col[use_col] > 0 or status_quo_typology.loc[b, use_col] == 'SINGLE_RES':
                    simulated_typology.loc[b, use_col] = PARAMS['MULTI_RES_PLANNED']  # FIXME: hard-coded
                    # simulated_typology.loc[b, "STANDARD"] = "STANDARD5" # TODO: get from input
    save_updated_typology(path_to_output_typology_file, simulated_typology)


def save_updated_typology(path_to_output_typology_file, simulated_typology):
    output = simulated_typology.copy()
    keep = list()
    columns_to_keep = [("Name", str), ("YEAR", int), ("STANDARD", str), ("1ST_USE", str), ("1ST_USE_R", float),
                       ("2ND_USE", str), ("2ND_USE_R", float), ("3RD_USE", str), ("3RD_USE_R", float),
                       ("REFERENCE", str)]
    for column, column_type in columns_to_keep:
        keep.append(column)
        output[column] = output[column].astype(column_type)
    dataframe_to_dbf(output[keep], str(path_to_output_typology_file.absolute()))


def optimize_all_scenarios(range_additional_floors_per_building, scenarios, target_add_gfa_per_use,
                           total_additional_gfa_target):
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
                              for sb in sub_building_idx}  # FIXME: check

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
        print("\n scenario [%i] is-success (if 1): [%i]" % (scenario, solution.sol_status))

        detailed_metrics = amap.detailed_result_metrics(
            solution=solution,
            sub_building_use=sub_building_use,
            sub_footprint_area=sub_footprint_area,
            target_add_gfa_per_use=target_add_gfa_per_use,
            target=total_additional_gfa_target
        )
        metrics[scenario] = detailed_metrics
    return metrics, optimizations


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


def calc_range_additional_floors_per_building(typology_status_quo):
    height_limit_per_city_zone = {
        0: (0, PARAMS['building_height_limit'] // PARAMS['floor_height']),
        1: (0, PARAMS['building_height_limit'] // PARAMS['floor_height']),
        2: (0, PARAMS['building_height_limit'] // PARAMS['floor_height']),
        3: (0, PARAMS['building_height_limit'] // PARAMS['floor_height'])}  # FIXME
    # height_limit_per_city_zone = {0: (0, 8), 1: (0, 26), 2: (0, 26),
    #                               3: (0, 13)}  # FIXME: this only applies to Altstetten
    range_additional_floors_per_building = dict()
    for name, building in typology_status_quo.iterrows():
        min_floors, max_floors = height_limit_per_city_zone[building.city_zone]
        range_additional_floors_per_building[name] = [0, max(0, max_floors - building.floors_ag)]
    return range_additional_floors_per_building


def filter_buildings_by_year(typology_merged: pd.DataFrame, year: int, less_than: bool = True):
    if "YEAR" not in typology_merged:
        raise KeyError("provided data frame is missing the column 'YEAR'")
    if less_than:
        op = operator.lt
    else:
        op = operator.gt
    return typology_merged[op(typology_merged.YEAR, year)]


def calc_future_required_gfa_per_use(future_required_additional_res_gfa,
                                     gfa_per_use_statusquo, rel_ratio_to_res_gfa_target):
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


def calc_additional_requied_residential_gfa(PARAMS):
    future_required_additional_res_living_space = PARAMS['additional_population'] * PARAMS['future_occupant_density']
    future_required_additional_res_gfa = future_required_additional_res_living_space / PARAMS[
        'ratio_living_space_to_GFA']
    return future_required_additional_res_gfa


def calc_rel_ratio_to_res_gfa_target(gfa_per_use_statusquo, config):
    """
    1. read target ratio
    2. update rel_ratio_to_res_gfa for use that exists but not specified in target
    :param gfa_per_use_statusquo: in m2
    :return:
    """
    gfa_per_use_statusquo_in = gfa_per_use_statusquo.copy()
    # read target ratio
    rel_ratio_to_res_gfa_target = read_mapping_use_ratio(config)
    # unify columns in target_rel_ratios and gfa_per_use_statusquo
    zero_series = pd.DataFrame(0.0, index=range(1), columns=rel_ratio_to_res_gfa_target.index).loc[0]
    gfa_per_use_statusquo = zero_series.combine(gfa_per_use_statusquo, max)  # TODO: redundant?
    rel_ratio_to_res_gfa_statusquo = (gfa_per_use_statusquo / gfa_per_use_statusquo.filter(like='_RES').sum())
    # update rel_ratio_to_res_gfa for use that exists but not specified in target
    for use, target_val in rel_ratio_to_res_gfa_target.items():
        if np.isclose(target_val, 0.0) and rel_ratio_to_res_gfa_statusquo[use] > 0:
            # if not required in target, keep existing uses in status quo
            rel_ratio_to_res_gfa_target[use] = rel_ratio_to_res_gfa_statusquo[use]
        else:
            rel_ratio_to_res_gfa_target[use] = target_val
    # drop uses with zero ratio
    rel_ratio_to_res_gfa_target = rel_ratio_to_res_gfa_target[rel_ratio_to_res_gfa_target > 0.0]
    gfa_per_use_statusquo = gfa_per_use_statusquo.loc[rel_ratio_to_res_gfa_target.index]  # get relevant uses
    assert np.isclose(gfa_per_use_statusquo_in.sum(), gfa_per_use_statusquo.sum())  # make sure total gfa is unchanged
    return gfa_per_use_statusquo, rel_ratio_to_res_gfa_target


def read_mapping_use_ratio(config):
    """
    read numbers from table
    :return:
    """
    district_archetype = config.remap_ville_scenarios.district_archetype
    year = config.remap_ville_scenarios.year
    urban_development_scenario = config.remap_ville_scenarios.urban_development_scenario
    path_to_current_directory = os.path.join(os.path.dirname(__file__))
    path_to_mapping_table = os.path.join('', *[path_to_current_directory, "mapping_BUILDING_USE_RATIO.xlsx"])
    worksheet = f"{district_archetype}_{year}"
    print(f"Reading use type ratio mappings from worksheet {worksheet}")
    mapping_df = pd.read_excel(path_to_mapping_table, sheet_name=worksheet).set_index("Scenario")
    rel_ratio_to_res_gfa_per_use = mapping_df.loc[urban_development_scenario].drop("Reference")
    return rel_ratio_to_res_gfa_per_use


def combine_MULTI_RES_gfa(gfa_per_use):
    MULTI_RES_gfa_sum = gfa_per_use.filter(like='MULTI_RES').sum()
    gfa_per_use = gfa_per_use.drop(gfa_per_use.filter(like="MULTI_RES").index)
    gfa_per_use["MULTI_RES"] = MULTI_RES_gfa_sum
    return gfa_per_use


def remove_buildings_by_uses(typology_df: pd.DataFrame, uses_to_remove: list):
    typology_remained_df = typology_df.copy()
    orig_building_names = typology_remained_df.index
    for use in uses_to_remove:
        typology_remained_df = typology_remained_df.drop(
            typology_remained_df[typology_remained_df["1ST_USE"] == use].index)
    building_names_remained = typology_remained_df.index
    typology_removed_df = typology_df.loc[list(set(orig_building_names) - set(building_names_remained))]
    if len(typology_removed_df) < 1:
        typology_removed_df = None
    return typology_remained_df, typology_removed_df


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
