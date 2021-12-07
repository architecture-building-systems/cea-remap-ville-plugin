import os
import random

import numpy as np
import pandas as pd
from remap_ville_plugin.utilities import calc_gfa_per_use

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

def main(config, typology_statusquo):
    typology_statusquo, typology_planned = remove_buildings_by_uses(typology_statusquo,
                                                                    uses_to_remove=[PARAMS['MULTI_RES_PLANNED']])
    gfa_per_use_statusquo = calc_gfa_per_use(typology_statusquo, "GFA_m2")
    gfa_per_use_planned = calc_gfa_per_use(typology_planned, "GFA_m2") if typology_planned else None
    gfa_res_planned = gfa_per_use_planned[PARAMS['MULTI_RES_PLANNED']] if gfa_per_use_planned else 0.0
    # get overview
    overview = {}
    overview["gfa_per_use_statusquo"] = gfa_per_use_statusquo.astype(int)
    overview["gfa_ratio_per_use_statusquo"] = round(gfa_per_use_statusquo / gfa_per_use_statusquo.sum(), 5)
    overview["gfa_per_use_planned"] = gfa_per_use_planned.astype(int) if gfa_per_use_planned else 0.0
    # TODO: KEEP "FUTURE RESERVED AREA" (ONLY FOOTPRINTS BUT NO HEIGHT) TO BUILD MULTI_RES (Altstetten)
    ## 2. Set Targets
    # get rel_ratio_to_res_gfa_target
    gfa_per_use_statusquo = combine_MULTI_RES_gfa(
        gfa_per_use_statusquo)  # FIXME: redundant since remove_buildings_by_uses
    gfa_per_use_statusquo, rel_ratio_to_res_gfa_target = calc_rel_ratio_to_res_gfa_target(gfa_per_use_statusquo, config)
    # get future required residential gfa
    future_required_additional_res_gfa = calc_future_required_additional_res_gfa(PARAMS)
    # calculate future required gfa per use
    gfa_per_use_future_target = calc_gfa_per_use_future_target(future_required_additional_res_gfa,
                                                               gfa_per_use_statusquo,
                                                               rel_ratio_to_res_gfa_target)
    # get additional required gfa per use
    gfa_per_use_additional = gfa_per_use_future_target - gfa_per_use_statusquo
    gfa_per_use_additional["MULTI_RES"] = gfa_per_use_additional["MULTI_RES"] - gfa_res_planned
    gfa_per_use_additional[gfa_per_use_additional < 0] = 0.0  # FIXME: TRANSFORM THE USETYPE THAT IS DIMINISHING
    # transform part of SECONDARY_RES to MULTI_RES
    gfa_per_use_additional, \
    gfa_per_use_future_target, \
    typology_statusquo = convert_SECONDARY_to_MULTI_RES(gfa_per_use_additional, gfa_per_use_future_target,
                                                        typology_statusquo)
    # transform parts of SINGLE_RES to MULTI_RES
    gfa_per_use_additional, \
    gfa_per_use_future_target, \
    typology_statusquo = convert_SINGLE_TO_MULTI_RES(gfa_per_use_additional, gfa_per_use_future_target,
                                                     typology_statusquo)
    # get gfa_per_use_additional_target
    gfa_per_use_additional_target = gfa_per_use_additional.astype(int).to_dict()
    gfa_total_additional_target = gfa_per_use_additional.sum()
    overview["gfa_per_use_future_target"] = gfa_per_use_future_target.astype(int)
    overview["gfa_per_use_additional_target"] = gfa_per_use_additional.astype(int)
    return gfa_per_use_future_target, gfa_per_use_additional_target, gfa_total_additional_target, overview, rel_ratio_to_res_gfa_target, typology_planned, typology_statusquo


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


def combine_MULTI_RES_gfa(gfa_per_use):
    """
    combine MULTI_RES and MULTI_RES_2040
    :param gfa_per_use:
    :return:
    """
    MULTI_RES_gfa_sum = gfa_per_use.filter(like='MULTI_RES').sum()
    gfa_per_use = gfa_per_use.drop(gfa_per_use.filter(like="MULTI_RES").index)
    gfa_per_use["MULTI_RES"] = MULTI_RES_gfa_sum
    return gfa_per_use


def calc_rel_ratio_to_res_gfa_target(gfa_per_use_statusquo, config):
    """
    1. read target ratio
    2. update rel_ratio_to_res_gfa for use that exists but not specified in target
    3. expand use types in gfa_per_use_statusquo according to rel_ratio_to_res_gfa
    :param gfa_per_use_statusquo: in m2
    :return:
    """
    gfa_per_use_statusquo_in = gfa_per_use_statusquo.copy()
    # read target ratio
    rel_ratio_to_res_gfa_target = read_mapping_use_ratio(config)
    # calculate rel_ratio_to_res_gfa_statusquo
    zero_series = pd.DataFrame(0.0, index=range(1), columns=rel_ratio_to_res_gfa_target.index).loc[0]
    gfa_per_use_statusquo = zero_series.combine(gfa_per_use_statusquo, max)  # TODO: redundant?
    rel_ratio_to_res_gfa_statusquo = (gfa_per_use_statusquo / gfa_per_use_statusquo.filter(like='_RES').sum())
    ## Update rel_ratio_to_res_gfa_target
    for use, target_val in rel_ratio_to_res_gfa_target.items():
        if np.isclose(target_val, 0.0) and rel_ratio_to_res_gfa_statusquo[use] > 0:
            # if not specified in target, keep rel_ratio in status quo
            rel_ratio_to_res_gfa_target[use] = rel_ratio_to_res_gfa_statusquo[use]
        else:
            rel_ratio_to_res_gfa_target[use] = target_val
    # drop uses with zero ratio
    rel_ratio_to_res_gfa_target = rel_ratio_to_res_gfa_target[rel_ratio_to_res_gfa_target > 0.0]
    # expand gfa_per_use_statusquo with all use types
    gfa_per_use_statusquo = gfa_per_use_statusquo.loc[rel_ratio_to_res_gfa_target.index]
    assert np.isclose(gfa_per_use_statusquo_in.sum(), gfa_per_use_statusquo.sum())  # make sure total gfa is unchanged
    return gfa_per_use_statusquo, rel_ratio_to_res_gfa_target


def calc_future_required_additional_res_gfa(PARAMS):
    future_required_additional_living_space = PARAMS['additional_population'] * PARAMS['future_occupant_density']
    future_required_additional_res_gfa = future_required_additional_living_space / PARAMS[
        'ratio_living_space_to_GFA']
    return future_required_additional_res_gfa


def calc_gfa_per_use_future_target(future_required_additional_res_gfa,
                                   gfa_per_use_statusquo, rel_ratio_to_res_gfa_target):
    future_required_res_gfa = (future_required_additional_res_gfa + gfa_per_use_statusquo.filter(like="_RES").sum())
    # write future required gfa for all use types
    gfa_per_use_future_target_dict = dict()
    for use_type in rel_ratio_to_res_gfa_target.index:
        gfa_future_required = future_required_res_gfa * rel_ratio_to_res_gfa_target[use_type]
        gfa_statusquo = gfa_per_use_statusquo[use_type]
        if use_type == "SINGLE_RES":
            gfa_per_use_future_target_dict[use_type] = gfa_statusquo  # unchanged
        elif use_type == "SECONDARY_RES":
            rel_ratio = rel_ratio_to_res_gfa_target[use_type]
            if gfa_per_use_statusquo['SECONDARY_RES'] > gfa_future_required or rel_ratio > 0.2:
                gfa_per_use_future_target_dict[use_type] = gfa_statusquo
            else:
                gfa_per_use_future_target_dict[use_type] = gfa_future_required
        elif use_type == "MULTI_RES":
            gfa_per_use_future_target_dict[use_type] = gfa_statusquo + future_required_additional_res_gfa
        else:
            gfa_per_use_future_target_dict.update({use_type: gfa_future_required})
    gfa_per_use_future_target = pd.Series(gfa_per_use_future_target_dict)
    return gfa_per_use_future_target


def convert_SINGLE_TO_MULTI_RES(gfa_per_use_additional_required, gfa_per_use_future_target, typology_statusquo):
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
            extra_gfa_from_SINGLE_RES_conversion += extra_gfa # extra gfa to host additional population
            typology_statusquo.loc[b, :] = typology_statusquo.loc[b].replace({"SINGLE_RES": "MULTI_RES"})
        gfa_per_use_future_target["SINGLE_RES"] = gfa_per_use_future_target[
                                                               "SINGLE_RES"] - gfa_to_MULTI_RES
        gfa_per_use_additional_required["MULTI_RES"] = gfa_per_use_additional_required[
                                                           "MULTI_RES"] - extra_gfa_from_SINGLE_RES_conversion
        assert gfa_per_use_additional_required["MULTI_RES"] > 0.0

    return gfa_per_use_additional_required, gfa_per_use_future_target, typology_statusquo


def convert_SECONDARY_to_MULTI_RES(gfa_per_use_additional_required, gfa_per_use_future_target, typology_statusquo):
    """
    Sample Secondary Residential buildings to convert to Multi-Residential buildings.
    :param gfa_per_use_additional_required:
    :param gfa_per_use_future_target:
    :param typology_statusquo:
    :return:
    """
    SECONDARY_RES_buildings = list(typology_statusquo.loc[typology_statusquo['1ST_USE'] == 'SECONDARY_RES'].index)
    SECONDARY_RES_gfa = typology_statusquo.loc[SECONDARY_RES_buildings]['GFA_m2'].sum()
    print(len(SECONDARY_RES_buildings), 'SECONDARY_RES buildings in the district.')
    required_SECONDARY_RES_gfa = 0.0
    if 'SECONDARY_RES' in gfa_per_use_additional_required.index:
        required_SECONDARY_RES_gfa = gfa_per_use_additional_required['SECONDARY_RES']
    required_MULTI_RES_gfa = gfa_per_use_additional_required['MULTI_RES']
    if SECONDARY_RES_gfa > 0 and np.isclose(required_SECONDARY_RES_gfa, 0.0) and SECONDARY_RES_gfa / required_MULTI_RES_gfa > 10:
        delta_gfa_dict = {}
        buffer = required_MULTI_RES_gfa * 0.1
        for i in range(1000):
            num_sampled_buildings = random.randrange(0, len(SECONDARY_RES_buildings))
            sampled_buildings = random.sample(set(SECONDARY_RES_buildings), num_sampled_buildings)
            sampled_gfa = typology_statusquo.loc[sampled_buildings]['GFA_m2'].sum()
            delta_gfa = round(required_MULTI_RES_gfa - sampled_gfa, 2)
            if abs(delta_gfa) < buffer:
                delta_gfa_dict[delta_gfa] = sampled_buildings
        buildings_to_MULTI_RES = delta_gfa_dict[min(delta_gfa_dict.keys())]
        print('Converting...', len(buildings_to_MULTI_RES), 'SECONDARY_RES to MULTI_RES')
        typology_statusquo.loc[buildings_to_MULTI_RES, '1ST_USE'] = 'MULTI_RES'
        SECONDARY_to_MULTI_RES_gfa = typology_statusquo.loc[buildings_to_MULTI_RES]['GFA_m2'].sum()
        gfa_per_use_additional_required['MULTI_RES'] = max(
            required_MULTI_RES_gfa - SECONDARY_to_MULTI_RES_gfa, 0)
        # update targets
        gfa_per_use_future_target["SECONDARY_RES"] = gfa_per_use_future_target[
                                                         "SECONDARY_RES"] - SECONDARY_to_MULTI_RES_gfa
    else:
        SECONDARY_to_MULTI_RES_gfa = 0.0
        gfa_per_use_future_target["SECONDARY_RES"] = 0.0

    return gfa_per_use_additional_required, gfa_per_use_future_target, typology_statusquo


def read_mapping_use_ratio(config):
    """
    read numbers from mapping_BUILDING_USE_RATIO.xlsx
    :return:
    """
    district_archetype = config.remap_ville_scenarios.district_archetype
    year = config.remap_ville_scenarios.year
    urban_development_scenario = config.remap_ville_scenarios.urban_development_scenario
    path_to_current_directory = os.path.join(os.path.dirname(__file__))
    path_to_mapping_table = os.path.join('', *[path_to_current_directory, "mapping_BUILDING_USE_RATIO.xlsx"])
    worksheet = f"{district_archetype}_{year}"
    print(f"Reading use type ratio mappings from worksheet {worksheet} for {urban_development_scenario}")
    mapping_df = pd.read_excel(path_to_mapping_table, sheet_name=worksheet).set_index("Scenario")
    rel_ratio_to_res_gfa_per_use = mapping_df.loc[urban_development_scenario].drop("Reference")
    return rel_ratio_to_res_gfa_per_use