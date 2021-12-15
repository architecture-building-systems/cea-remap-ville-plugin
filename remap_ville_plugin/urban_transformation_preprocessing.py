import os
import random

import numpy as np
import pandas as pd

import cea.config
import cea.inputlocator
from remap_ville_plugin.utilities import calc_gfa_per_use
from utilities import select_buildings_from_candidates, get_building_candidates, convert_uses

USE_TYPE_CONVERSION = {
    'RETAIL': ['OFFICE', 'HOTEL', 'RESTAURANT'],
    'MULTI_RES': ['HOTEL', 'OFFICE', 'SECONDARY_RES'],
    'SCHOOL':['OFFICE', 'HOTEL', 'HOSPITAL', 'MULTI_RES'],
    'OFFICE': ['RETAIL', 'LIBRARY', 'MULTI_RES'],
    'INDUSTRIAL':['OFFICE', 'RESTAURANT', 'HOTEL', 'HOSPITAL', 'LIBRARY'],
    'PARKING': ['RESTAURANT', 'GYM', 'MULTI_RES']
}

PARAMS = {
    'scenario_count': 10,
    'lower_bound_floors': 0,
    'upper_bound_floors': 50,
    'floor_height': 3,
    'ratio_living_space_to_GFA': 0.82,
}

def main(config, typology_statusquo, case_inputs, type):
    typology_statusquo, typology_planned = remove_buildings_by_uses(typology_statusquo,
                                                                    uses_to_remove=[case_inputs['MULTI_RES_PLANNED']])
    gfa_per_use_statusquo = calc_gfa_per_use(typology_statusquo, "GFA_m2")
    gfa_per_use_planned = calc_gfa_per_use(typology_planned, "GFA_m2") if typology_planned is not None else None
    gfa_res_planned = gfa_per_use_planned[case_inputs['MULTI_RES_PLANNED']] if gfa_per_use_planned is not None else 0.0
    # get overview
    overview = {}
    overview["gfa_per_use_statusquo"] = gfa_per_use_statusquo.astype(int)
    overview["gfa_ratio_per_use_statusquo"] = round(gfa_per_use_statusquo / gfa_per_use_statusquo.sum(), 5)
    overview["gfa_per_use_planned"] = gfa_per_use_planned.astype(int) if gfa_per_use_planned is not None else 0.0
    # TODO: KEEP "FUTURE RESERVED AREA" (ONLY FOOTPRINTS BUT NO HEIGHT) TO BUILD MULTI_RES (Altstetten)
    ## 2. Set Targets
    # get rel_ratio_to_res_gfa_target
    gfa_per_use_statusquo = combine_MULTI_RES_gfa(
        gfa_per_use_statusquo)  # FIXME: redundant since remove_buildings_by_uses
    gfa_per_use_statusquo, rel_ratio_to_res_gfa_target = calc_rel_ratio_to_res_gfa_target(gfa_per_use_statusquo, config)
    # get future required residential gfa
    future_required_additional_res_gfa = calc_future_required_additional_res_gfa(case_inputs)
    # calculate future required gfa per use
    gfa_per_use_future_target = calc_gfa_per_use_future_target(future_required_additional_res_gfa,
                                                               gfa_per_use_statusquo,
                                                               rel_ratio_to_res_gfa_target)
    # get additional required gfa per use
    gfa_per_use_additional = gfa_per_use_future_target - gfa_per_use_statusquo
    gfa_per_use_additional["MULTI_RES"] = gfa_per_use_additional["MULTI_RES"] - gfa_res_planned
    district_archetype = config.remap_ville_scenarios.district_archetype
    if district_archetype=='RRL':
        if config.remap_ville_scenarios.urban_development_scenario == 'BAU':
            # convert MULTI_RES to SECONDARY_RES
            gfa_to_convert = abs(gfa_per_use_additional['MULTI_RES'])
            gfa_converted, buildings_converted, typology_statusquo = convert_uses(gfa_to_convert, typology_statusquo, 'MULTI_RES', 'SECONDARY_RES')
            gfa_per_use_additional = pd.Series(data=np.zeros(len(gfa_per_use_additional)), index=list(gfa_per_use_additional.index))
            gfa_per_use_future_target = gfa_per_use_statusquo.copy()
            gfa_per_use_future_target['MULTI_RES'] = gfa_per_use_future_target['MULTI_RES'] - gfa_converted
            gfa_per_use_future_target['SECONDARY_RES'] = gfa_per_use_future_target['SECONDARY_RES'] + gfa_converted
        elif config.remap_ville_scenarios.urban_development_scenario=='DGT' and type=='end':
            # convert diminishing uses
            gfa_per_use_additional, typology_statusquo = convert_diminishing_uses(gfa_per_use_additional, typology_statusquo)
            gfa_per_use_additional[gfa_per_use_additional < 50] = 0.0 # remove additional GFA < 50
    elif district_archetype=='URB' or district_archetype=='SURB':
        gfa_per_use_additional[gfa_per_use_additional < 0] = 0.0 # FIXME: TEMP FIX
    else:
        gfa_per_use_additional = gfa_per_use_additional
    # transform part of SECONDARY_RES to MULTI_RES
    gfa_per_use_additional, \
    gfa_per_use_future_target, \
    typology_statusquo = convert_SECONDARY_to_MULTI_RES(gfa_per_use_additional, gfa_per_use_future_target,
                                                        typology_statusquo)
    # transform parts of SINGLE_RES to MULTI_RES
    gfa_per_use_additional, \
    gfa_per_use_future_target, \
    typology_statusquo = convert_SINGLE_TO_MULTI_RES(gfa_per_use_additional, gfa_per_use_future_target,
                                                     typology_statusquo, case_inputs)
    # transform parts of OFFICE to MULTI_RES
    gfa_per_use_additional, \
    gfa_per_use_future_target, \
    typology_statusquo = convert_OFFICE_TO_MULTI_RES(gfa_per_use_additional, gfa_per_use_future_target,
                                                     typology_statusquo, case_inputs)
    # get gfa_per_use_additional_target
    gfa_per_use_additional_target = gfa_per_use_additional.astype(int).to_dict()
    gfa_total_additional_target = gfa_per_use_additional.sum()
    overview["gfa_per_use_future_target"] = gfa_per_use_future_target.astype(int)
    overview["gfa_per_use_additional_target"] = gfa_per_use_additional.astype(int)
    return gfa_per_use_future_target, gfa_per_use_additional_target, gfa_total_additional_target, overview, \
           rel_ratio_to_res_gfa_target, typology_planned, typology_statusquo


def convert_diminishing_uses(gfa_per_use_additional, typology_statusquo):
    print(f'Diminishing uses: {gfa_per_use_additional[gfa_per_use_additional < 0].index.values}')
    diminishing_uses = gfa_per_use_additional[gfa_per_use_additional < 0]
    for use_to_reduce in diminishing_uses.index:
        gfa_to_reduce = abs(diminishing_uses[use_to_reduce])
        print('\nStarting to reduce', use_to_reduce, 'with GFA:', int(gfa_to_reduce))
        gfa_to_add_list = [gfa_per_use_additional[use_to_add] for use_to_add in USE_TYPE_CONVERSION[use_to_reduce]]
        # gfa_to_add_list.sort(reverse=True) # no need to order by GFA
        for gfa_to_add in gfa_to_add_list:
            use_to_add = gfa_per_use_additional[np.isclose(gfa_per_use_additional, gfa_to_add)].index[0]
            print(f'...by converting to {use_to_add} (additional required: {int(gfa_per_use_additional[use_to_add])} m2)')
            # select possible buildings to remove from statusquo
            floors_usetype_sq, footprint_usetype_sq = get_building_candidates(use_to_reduce, typology_statusquo)
            if gfa_to_add > gfa_to_reduce:  # convert all gfa_to_reduce
                selected_floors_to_convert = select_buildings_from_candidates(gfa_to_reduce,
                                                                              floors_usetype_sq,
                                                                              footprint_usetype_sq)  # convert all
            else:  # convert required gfa_to_add
                selected_floors_to_convert = select_buildings_from_candidates(gfa_to_add,
                                                                              floors_usetype_sq, footprint_usetype_sq)
            if selected_floors_to_convert is not None:
                updated_floors_reduced_use = floors_usetype_sq[selected_floors_to_convert.index] - selected_floors_to_convert
                print(f'...converting {len(selected_floors_to_convert)} buildings from {use_to_reduce} to {use_to_add}')
                gfa_converted = 0.0
                for b in selected_floors_to_convert.index:
                    building_usetypes = typology_statusquo.loc[b, ['1ST_USE', '2ND_USE', '3RD_USE']]
                    use_to_reduce_order = building_usetypes[building_usetypes == use_to_reduce].index[0]
                    if np.isclose(updated_floors_reduced_use[b], 0.0):
                        typology_statusquo.loc[b, :] = typology_statusquo.loc[b].replace(
                            {str(use_to_reduce): str(use_to_add)})
                    else:
                        typology_statusquo.loc[b, use_to_reduce_order + '_F'] = updated_floors_reduced_use[b]
                        # add use
                        if use_to_add in building_usetypes:
                            use_to_add_order = building_usetypes[building_usetypes == use_to_add].index[0]
                        else:
                            use_to_add_order = building_usetypes[building_usetypes == "NONE"].index[0]
                            typology_statusquo.loc[b, use_to_add_order] = use_to_add
                        typology_statusquo.loc[b, use_to_add_order + '_F'] = typology_statusquo.loc[b, use_to_add_order + '_F'] + selected_floors_to_convert[b]
                        typology_statusquo = update_typology_R_GFA_from_F(b, typology_statusquo)
                    typology_statusquo.loc[b, 'orig_uses'].append(use_to_reduce)
                    typology_statusquo.loc[b, 'new_uses'].append(use_to_add)
                    gfa_converted_b = typology_statusquo.loc[b, 'footprint'] * selected_floors_to_convert[b]
                    gfa_converted += gfa_converted_b
                print(f'\tgfa convereted to {use_to_add}:', int(gfa_converted))
                gfa_additional = gfa_per_use_additional[use_to_add] - gfa_converted
                gfa_per_use_additional[use_to_add] = gfa_additional if gfa_additional > 1 else 0.0
                print(f'\t{use_to_add} additional required:', int(gfa_per_use_additional[use_to_add]))
                gfa_to_reduce = gfa_to_reduce - gfa_converted
            print(f'remaining {use_to_reduce} gfa_to_reduce', int(gfa_to_reduce), 'm2')
            gfa_per_use_additional[use_to_reduce] = gfa_to_reduce * (-1) # update
            if gfa_to_reduce < 0.0:
                gfa_per_use_additional[use_to_reduce] = 0.0
                break
        # gfa_per_use_additional[use_to_reduce] = 0.0
        print(f'{use_to_reduce} gfa_per_use_additional', int(gfa_per_use_additional[use_to_reduce]), '\n')
    # removing uses from typology_statusquo
    print('Directly remove GFA:',gfa_per_use_additional[diminishing_uses.index])
    for use_to_remove in diminishing_uses.index:
        floors_usetype_sq, footprint_usetype_sq = get_building_candidates(use_to_remove, typology_statusquo)
        gfa_to_remove = abs(gfa_per_use_additional[use_to_remove])
        if gfa_to_remove > min(footprint_usetype_sq):
            selected_floors_to_remove = select_buildings_from_candidates(gfa_to_remove,
                                                                         floors_usetype_sq, footprint_usetype_sq)
            for b in selected_floors_to_remove.index:
                updated_floors_remove_use = floors_usetype_sq[b] - selected_floors_to_remove[b]
                building_usetypes = typology_statusquo.loc[b, ['1ST_USE', '2ND_USE', '3RD_USE']]
                use_to_remove_order = building_usetypes[building_usetypes == use_to_remove].index[0]
                typology_statusquo.loc[b, use_to_remove_order + '_F'] = updated_floors_remove_use
                # update floors_ag
                updated_total_floors = typology_statusquo.loc[b].filter(like='_F').sum()
                if np.isclose(updated_total_floors, 0):
                    typology_statusquo = typology_statusquo.drop(b)
                elif np.isclose(updated_total_floors, 1):
                    typology_statusquo.loc[b,'floors_ag'] = 1
                    typology_statusquo.loc[b,'floors_bg'] = 0
                    typology_statusquo = update_typology_R_GFA_from_F(b, typology_statusquo)
                else:
                    typology_statusquo.loc[b,'floors_ag'] = updated_total_floors - typology_statusquo.loc[b,'floors_bg']
                    assert typology_statusquo.loc[b].filter(like='_F').sum() > 0
                    typology_statusquo = update_typology_R_GFA_from_F(b, typology_statusquo)
        gfa_per_use_additional[use_to_remove] = 0.0
    return gfa_per_use_additional, typology_statusquo


def update_typology_R_GFA_from_F(b, typology_statusquo):
    total_floors_statusquo = typology_statusquo.loc[b, ['floors_ag', 'floors_bg']].sum()
    # update ratios and GFA per use
    total_floors = typology_statusquo.loc[b].filter(like='_F').sum()
    assert total_floors == total_floors_statusquo
    typology_statusquo.loc[b, ['1ST_USE_R', '2ND_USE_R', '3RD_USE_R']] = [
        typology_statusquo.loc[b, use_order + '_F'] / total_floors for use_order in
        ['1ST_USE', '2ND_USE', '3RD_USE']]
    typology_statusquo.loc[b, ['GFA_1ST_USE', 'GFA_2ND_USE', 'GFA_3RD_USE']] = [
        typology_statusquo.loc[b, use_order + '_R'] * typology_statusquo.loc[b, 'GFA_m2'] for
        use_order in
        ['1ST_USE', '2ND_USE', '3RD_USE']]
    return typology_statusquo


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


def calc_future_required_additional_res_gfa(case_study_inputs):
    additional_population = case_study_inputs['additional_population']
    occupant_density = case_study_inputs['future_MFH_density']
    future_required_additional_living_space = additional_population * occupant_density
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


def convert_OFFICE_TO_MULTI_RES(gfa_per_use_additional_required, gfa_per_use_future_target, typology_statusquo, case_inputs):
    typology_usetype = typology_statusquo[typology_statusquo['1ST_USE'] == 'OFFICE'][typology_statusquo['1ST_USE_R'] >= 1.0]
    buildings_usetype = list(typology_usetype.index)
    print(f'\t{len(buildings_usetype)} OFFICE in the district.')
    num_buildings_to_MULTI_RES = int(len(buildings_usetype) * case_inputs['OFFICE_to_MULTI_RES_ratio'])
    if num_buildings_to_MULTI_RES > 0.0:
        buildings_to_MULTI_RES = random.sample(buildings_usetype, num_buildings_to_MULTI_RES)
        gfa_to_MULTI_RES = 0.0
        b_count = 0
        for b in buildings_to_MULTI_RES:
            building_gfa = typology_statusquo.loc[b]['GFA_m2']
            gfa_to_MULTI_RES += building_gfa
            typology_statusquo.loc[b, :] = typology_statusquo.loc[b].replace({"OFFICE": "MULTI_RES"})
            typology_statusquo.loc[b, 'orig_uses'].append('OFFICE')
            typology_statusquo.loc[b, 'new_uses'].append('MULTI_RES')
            typology_statusquo.loc[b, 'REFERENCE_x'] = 'from OFFICE'
            b_count += 1
            if gfa_to_MULTI_RES > gfa_per_use_additional_required["MULTI_RES"]:
                break
        print('Converting...', b_count, 'OFFICE to MULTI_RES')
        gfa_per_use_future_target["OFFICE"] = gfa_per_use_future_target["OFFICE"] - gfa_to_MULTI_RES
        gfa_per_use_additional_required["OFFICE"] = 0.0 # no additional OFFICE
        gfa_per_use_additional_required["MULTI_RES"] = max(gfa_per_use_additional_required["MULTI_RES"] - gfa_to_MULTI_RES, 0.0)
        assert gfa_per_use_additional_required["MULTI_RES"] >= 0.0
    return gfa_per_use_additional_required, gfa_per_use_future_target, typology_statusquo


def convert_SINGLE_TO_MULTI_RES(gfa_per_use_additional_required, gfa_per_use_future_target, typology_statusquo, case_inputs):
    buildings_SINGLE_RES = list(typology_statusquo.loc[typology_statusquo['1ST_USE'] == 'SINGLE_RES'].index)
    print(f'\t{len(buildings_SINGLE_RES)} SINGLE_RES in the district.')
    num_buildings_to_MULTI_RES = int(len(buildings_SINGLE_RES) * case_inputs['SINGLE_to_MULTI_RES_ratio'])
    if num_buildings_to_MULTI_RES > 0.0:
        print('Converting...', num_buildings_to_MULTI_RES, 'SINGLE_RES to MULTI_RES')
        print('\tMULTI_RES additional required:', gfa_per_use_additional_required["MULTI_RES"])
        buildings_to_MULTI_RES = random.sample(buildings_SINGLE_RES, num_buildings_to_MULTI_RES)
        extra_gfa_from_SINGLE_RES_conversion, gfa_to_MULTI_RES = 0.0, 0.0
        for b in buildings_to_MULTI_RES:
            building_gfa = typology_statusquo.loc[b]['GFA_m2']
            gfa_to_MULTI_RES += building_gfa
            num_occupants = round(
                building_gfa * PARAMS["ratio_living_space_to_GFA"] / case_inputs["current_SFH_density"])
            extra_gfa = building_gfa - num_occupants * (
                    case_inputs["future_MFH_density"] / PARAMS["ratio_living_space_to_GFA"])
            extra_gfa_from_SINGLE_RES_conversion += extra_gfa # extra gfa to host additional population
            typology_statusquo.loc[b, :] = typology_statusquo.loc[b].replace({"SINGLE_RES": "MULTI_RES"})
            typology_statusquo.loc[b, 'orig_uses'].append('SINGLE_RES')
            typology_statusquo.loc[b, 'new_uses'].append('MULTI_RES')
            typology_statusquo.loc[b, 'REFERENCE_x'] = 'from SINGLE_RES'
        gfa_per_use_future_target["SINGLE_RES"] = gfa_per_use_future_target[
                                                               "SINGLE_RES"] - gfa_to_MULTI_RES
        gfa_per_use_future_target["MULTI_RES"] = gfa_per_use_future_target["MULTI_RES"] + gfa_to_MULTI_RES
        if gfa_per_use_additional_required["MULTI_RES"] > 0.0:
            gfa_per_use_additional_required["MULTI_RES"] = max(gfa_per_use_additional_required[
                                                               "MULTI_RES"] - extra_gfa_from_SINGLE_RES_conversion, 0.0)
        assert gfa_per_use_additional_required["MULTI_RES"] >= 0.0

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
    print('\t', len(SECONDARY_RES_buildings), 'SECONDARY_RES buildings in the district.')
    required_SECONDARY_RES_gfa = 0.0
    if 'SECONDARY_RES' in gfa_per_use_additional_required.index:
        required_SECONDARY_RES_gfa = gfa_per_use_additional_required['SECONDARY_RES']
    required_MULTI_RES_gfa = gfa_per_use_additional_required['MULTI_RES']
    if SECONDARY_RES_gfa > 0 and np.isclose(required_SECONDARY_RES_gfa, 0.0) and SECONDARY_RES_gfa / required_MULTI_RES_gfa > 10 and required_MULTI_RES_gfa > 0:
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
        typology_statusquo.loc[buildings_to_MULTI_RES, 'orig_uses'] = pd.Series([['SECONDARY_RES'] for _ in buildings_to_MULTI_RES], index=buildings_to_MULTI_RES)
        typology_statusquo.loc[buildings_to_MULTI_RES, 'new_uses'] = pd.Series([['MULTI_RES'] for _ in buildings_to_MULTI_RES], index=buildings_to_MULTI_RES)
        typology_statusquo.loc[buildings_to_MULTI_RES, 'REFERENCE_x'] = 'from SECONDARY_RES' # TODO: remove
        # update targets
        SECONDARY_to_MULTI_RES_gfa = typology_statusquo.loc[buildings_to_MULTI_RES]['GFA_m2'].sum()
        gfa_per_use_additional_required['MULTI_RES'] = max(required_MULTI_RES_gfa - SECONDARY_to_MULTI_RES_gfa, 0)
        gfa_per_use_future_target["SECONDARY_RES"] = gfa_per_use_future_target["SECONDARY_RES"] - SECONDARY_to_MULTI_RES_gfa
        gfa_per_use_future_target["MULTI_RES"] = gfa_per_use_future_target["MULTI_RES"] + SECONDARY_to_MULTI_RES_gfa
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


if __name__ == "__main__":
    config = cea.config.Configuration()
    # status-quo
    config.project = r'C:\Users\shsieh\Desktop\TEST_UT_REDUCE\Echallens'
    config.scenario_name = 2020
    locator = cea.inputlocator.InputLocator(scenario=config.scenario, plugins=config.plugins)
    from remap_ville_plugin.urban_transformation_sequential import get_district_typology_merged
    typology_statusquo = get_district_typology_merged(locator.get_input_folder())
    print(locator.get_input_folder())
    path_to_case_study_inputs = os.path.join(config.scenario, "case_study_inputs.xlsx")
    worksheet = f"{config.remap_ville_scenarios.district_archetype}_{config.remap_ville_scenarios.urban_development_scenario}"
    case_study_inputs_df = pd.read_excel(path_to_case_study_inputs, sheet_name=worksheet).set_index('year')
    # future scenario
    config.remap_ville_scenarios.year = 2060
    config.remap_ville_scenarios.urban_development_scenario = 'BAU'
    s_name = f'{config.remap_ville_scenarios.year}_{config.remap_ville_scenarios.urban_development_scenario}'
    case_study_inputs = case_study_inputs_df.loc[int(config.remap_ville_scenarios.year)]
    _, _, _, overview, _, _, _ = main(config, typology_statusquo, case_study_inputs, type='intermediate')
    pd.DataFrame(overview).to_csv(os.path.join(config.scenario, s_name+'_gfa_targets.csv'))

