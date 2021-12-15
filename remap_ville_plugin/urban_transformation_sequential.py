import pandas as pd
import numpy as np
import os
import geopandas as gpd
from collections import defaultdict
from pathlib import Path
import random

import cea.config
import cea.inputlocator
from cea.utilities.dbf import dbf_to_dataframe

import utilities
from remap_ville_plugin.utilities import save_updated_typology, filter_buildings_by_year, order_uses_in_typology
from remap_ville_plugin.create_technology_database import create_input_technology_folder, update_indoor_comfort
import remap_ville_plugin.urban_transformation_preprocessing as preprocessing
from utilities import select_buildings_from_candidates, get_building_candidates, calc_gfa_per_use

path_to_folder = r'C:\Users\shsieh\Desktop\TEST_UT_REDUCE\Echallens'
use_cols = ['MULTI_RES', 'SINGLE_RES', 'SECONDARY_RES', 'HOTEL', 'OFFICE', 'RETAIL', 'FOODSTORE',
            'RESTAURANT', 'INDUSTRIAL', 'SCHOOL', 'HOSPITAL', 'GYM', 'SWIMMING',
            'SERVERROOM', 'PARKING', 'COOLROOM', 'LAB', 'MUSEUM', 'LIBRARY',
            'UNIVERSITY']


def main(config, new_locator, scenario_locator_sequences, case_study_inputs, scenario_year, year_endstate):
    print('\nStarting sequential urban transformation for', config.scenario_name)
    scenario_statusquo = list(scenario_locator_sequences.keys())[0]
    scenario_endstate = list(scenario_locator_sequences.keys())[-1]
    scenario_intermediate = config.scenario_name
    print(f'...according to status-quo scenario: {scenario_statusquo} and end-state scenarios {scenario_endstate}\n')
    # 1. get projection
    typology_dict = {}
    for scenario_name, scenario_locator in scenario_locator_sequences.items():
        typology_dict[scenario_name] = get_district_typology_merged(scenario_locator.get_input_folder())
    gfa_per_use_years_df = pd.concat([get_gfa_per_usetype(typology_dict[key], key) for key in typology_dict.keys()])
    # calculate 2040 according to 2020
    typology_statusquo = typology_dict[scenario_statusquo].copy()
    gfa_per_use_future_target, _, _, _, _, _, _ = preprocessing.main(config, typology_statusquo, case_study_inputs, type='intermediate')
    # missing_usetypes = set(use_cols) - set(gfa_per_use_future_target.index)
    # gfa_per_use_intermediate = gfa_per_use_future_target.add(pd.Series(index=missing_usetypes), fill_value=0.0)
    # gfa_per_use_intermediate = gfa_per_use_future_target.fillna(0.0)
    # gfa_per_use_intermediate = gfa_per_use_intermediate[use_cols]
    gfa_per_use_years_df.loc[scenario_intermediate] = gfa_per_use_future_target
    gfa_per_use_years_df.fillna(0.0, inplace=True)
    # 2. get diff_gfa
    diff_gfa = gfa_per_use_years_df.loc[scenario_endstate] - gfa_per_use_years_df.loc[scenario_intermediate]
    # remove 'MULTI_RES_PLANNED' from target
    if case_study_inputs['MULTI_RES_PLANNED'] in diff_gfa.index:
        diff_gfa['MULTI_RES'] = diff_gfa['MULTI_RES'] - diff_gfa[case_study_inputs['MULTI_RES_PLANNED']]
        diff_gfa = diff_gfa.drop(labels=case_study_inputs['MULTI_RES_PLANNED'])
    uses_to_add_back = []
    for use in diff_gfa.index:
        if np.isclose(gfa_per_use_years_df.loc[scenario_endstate,use], gfa_per_use_years_df.loc[scenario_statusquo,use]):
            diff_gfa[use] = 0.0
            print(f'diff_gfa for {use} is set to 0.0')

    # 3. modify buildings
    typology_endstate = typology_dict[scenario_endstate]
    # remove preserved buildings
    typology_preserved_year, typology_after_year = filter_buildings_by_year(typology_endstate, year=case_study_inputs[
        "preserve_buildings_built_before"])
    typology_endstate = typology_after_year.copy()
    typology_updated = typology_endstate.copy()
    # revert MULTI_RES to SINGLE_RES
    typology_updated, typology_endstate, diff_gfa = revert_MULTI_RES_to_SINGLE_RES(diff_gfa, scenario_statusquo,
                                                                                   typology_dict,
                                                                                   typology_updated, typology_endstate,
                                                                                   year_endstate)
    # revert MULTI_RES to OFFICE
    typology_updated, typology_endstate, diff_gfa = revert_MULTI_RES_to_OFFICE(diff_gfa, scenario_statusquo,
                                                                               typology_dict,
                                                                               typology_updated, typology_endstate,
                                                                               year_endstate)
    # intermediate typology (RRL-DGT)
    if config.remap_ville_scenarios.district_archetype=='RRL':
        if config.remap_ville_scenarios.urban_development_scenario=='DGT':
            typology_updated, typology_endstate, diff_gfa = calc_intermediate_state_for_diminishing_uses(diff_gfa,
                                                                                                         gfa_per_use_years_df,
                                                                                                         scenario_endstate,
                                                                                                         scenario_statusquo,
                                                                                                         scenario_year,
                                                                                                         typology_dict,
                                                                                                         typology_endstate,
                                                                                                         typology_updated,
                                                                                                         uses_to_add_back)
        elif config.remap_ville_scenarios.urban_development_scenario=='BAU':
            # convert MULTI_RES to SINGLE_RES
            gfa_to_convert = diff_gfa['MULTI_RES']
            gfa_converted, buildings_converted, typology_endstate = utilities.convert_uses(gfa_to_convert, typology_endstate, 'MULTI_RES',
                                                             'SECONDARY_RES')
            typology_updated.loc[buildings_converted] = typology_endstate.loc[buildings_converted]
            diff_gfa = pd.Series(data=np.zeros(len(diff_gfa)), index=list(diff_gfa.index))


    buildings_modified = set()
    for usetype in diff_gfa.index:
        if round(diff_gfa[usetype]) > 0:
            print('\n', usetype, 'GFA to reduce', round(diff_gfa[usetype], 1))
            typology_updated, buildings_modified_usetype = modify_typology_per_building_usetype(usetype,
                                                                                                typology_updated,
                                                                                                typology_endstate,
                                                                                                diff_gfa,
                                                                                                scenario_year)
            buildings_modified = set.union(buildings_modified, buildings_modified_usetype)
            # get errors
            if len(buildings_modified_usetype) > 0:
                gfa_updated = get_gfa_per_usetype(typology_updated, scenario_intermediate).loc[
                    scenario_intermediate, usetype]
                gfa_projected = gfa_per_use_years_df.loc[scenario_intermediate, usetype]
                print('\tGFA(updated):', round(gfa_updated), ' GFA(projected):', round(gfa_projected))
    # 4. save typology_updated, zone_updated
    buildings_not_yet_built = set(list(typology_endstate.index)) - set(list(typology_updated.index))
    buildings_modified = list(buildings_modified - buildings_not_yet_built)
    # typology
    typology_save = pd.concat([typology_updated, typology_preserved_year])
    typology_save['REFERENCE'] = typology_save['REFERENCE_x']
    typology_save.loc[buildings_modified, 'REFERENCE'] = 'sequential-transformation'
    # TODO: add back the buildings untouched
    typology_save = order_uses_in_typology(typology_save)
    save_updated_typology(Path(new_locator.get_building_typology()), typology_save)
    # zone
    zone_endstate = gpd.read_file(scenario_locator_sequences[scenario_endstate].get_zone_geometry()).set_index('Name')
    zone_updated = zone_endstate.copy()
    zone_updated = zone_updated.fillna('-')
    zone_updated = zone_updated.drop(buildings_not_yet_built)  # remove buildings
    # move floor_bg to floor_ag
    zone_updated["floors_ag"] = typology_save['floors_all'] - typology_save['floors_bg']
    building_no_floors_ag = zone_updated[zone_updated['floors_ag'] == 0].index
    zone_updated.loc[building_no_floors_ag, 'floors_ag'] = zone_updated.loc[building_no_floors_ag, 'floors_bg']
    zone_updated.loc[building_no_floors_ag, 'floors_bg'] = 0
    zone_updated["height_ag"] = zone_updated['floors_ag'] * 3
    # save zone_shp_updated
    zone_updated.loc[buildings_modified, 'REFERENCE'] = 'sequential-transformation'
    if zone_updated.isnull().sum().sum() > 0:
        raise ValueError('nan values in zone_updated')
    zone_updated.to_file(new_locator.get_zone_geometry())
    # TODO: add back the buildings untouched
    print(f'zone.shp updated...{new_locator.get_zone_geometry()}')

    # create technology folder
    district_archetype = config.remap_ville_scenarios.district_archetype
    year = 2040
    urban_development_scenario = config.remap_ville_scenarios.urban_development_scenario
    folder_name = f"{district_archetype}_{year}_{urban_development_scenario}"
    create_input_technology_folder(folder_name, new_locator)
    update_indoor_comfort('SQ', new_locator)


def calc_intermediate_state_for_diminishing_uses(diff_gfa, gfa_per_use_years_df, scenario_endstate,
                                                 scenario_statusquo, scenario_year, typology_dict, typology_endstate,
                                                 typology_updated, uses_to_add_back):
    # add uses to typology_updated from typology_endstate
    for use in diff_gfa.index:
        if gfa_per_use_years_df.loc[scenario_endstate, use] < gfa_per_use_years_df.loc[scenario_statusquo, use]:
            diff_gfa_use = gfa_per_use_years_df.loc[scenario_statusquo, use] - gfa_per_use_years_df.loc[
                scenario_endstate, use]
            diff_gfa[use] = round(diff_gfa_use / 2) * (-1)
            uses_to_add_back.append(use)
            print(f'add back {use} from typology_endstate: {diff_gfa[use]}')
    buildings_removed_in_endstate = set(typology_dict[scenario_statusquo].index) - set(
        typology_dict[scenario_endstate].index)
    typology_removed_in_endstate = typology_dict[scenario_statusquo].loc[buildings_removed_in_endstate]
    for use in uses_to_add_back:
        gfa_to_add_intermediate = abs(diff_gfa[use])
        print(f'\nAdding {use} to typology_updated: {gfa_to_add_intermediate} m2')
        buildings_avail_to_revert = typology_removed_in_endstate[typology_removed_in_endstate['1ST_USE'] == use].index
        gfa_avail_to_add_sum = typology_removed_in_endstate.loc[buildings_avail_to_revert, 'GFA_m2'].sum()
        if gfa_avail_to_add_sum > gfa_to_add_intermediate:
            buildings_to_add_back = sample_buildings_to_match_gfa(buildings_avail_to_revert, gfa_to_add_intermediate,
                                                                  typology_removed_in_endstate)
        else:  # revert all
            buildings_to_add_back = buildings_avail_to_revert
        typology_updated = typology_updated.append(typology_removed_in_endstate.loc[buildings_to_add_back], sort=True)
        gfa_added_back = typology_removed_in_endstate.loc[buildings_to_add_back, 'GFA_m2'].sum()
        print(f'\tadd back {len(buildings_to_add_back)} to typology_updated: {round(gfa_added_back)}')
        diff_gfa[use] += gfa_added_back
        # convert some buildings
        remaining_diff_gfa_use = abs(diff_gfa[use])
        if remaining_diff_gfa_use > 0:
            typology_avail_use = typology_endstate[typology_endstate['orig_uses'].str.contains(use)]
            buildings_avail_to_convert = typology_avail_use.index
            typology_avail_to_convert = typology_dict[scenario_statusquo].loc[buildings_avail_to_convert]
            buildings_to_convert = sample_buildings_to_match_gfa(buildings_avail_to_convert, remaining_diff_gfa_use,
                                                                 typology_avail_to_convert)
            print(f'\tconverting {len(buildings_to_convert)} from {scenario_statusquo}')
            # update diff_gfa
            gfa_per_use_statusquo = calc_gfa_per_use(typology_dict[scenario_statusquo].loc[buildings_to_convert])
            gfa_per_use_endstate = calc_gfa_per_use(typology_dict[scenario_endstate].loc[buildings_to_convert])
            for idx in set(gfa_per_use_statusquo.index).union(set(gfa_per_use_endstate.index)):
                end = gfa_per_use_endstate[idx] if idx in gfa_per_use_endstate.index else 0
                sq = gfa_per_use_statusquo[idx] if idx in gfa_per_use_statusquo.index else 0
                diff_gfa[idx] = diff_gfa[idx] - sq + end if diff_gfa[idx] > 0 else diff_gfa[idx] + sq - end
            # update typology_statusquo
            typology_updated = typology_updated.drop(buildings_to_convert)
            typology_updated = typology_updated.append(typology_dict[scenario_statusquo].loc[buildings_to_convert],
                                                       sort=True)
            typology_updated.loc[buildings_to_convert, 'YEAR'] = scenario_year
            typology_endstate = typology_endstate.drop(buildings_to_convert)
        print(f'\tmismatched diff_gfa: {round(diff_gfa[use])}')
    return typology_updated, typology_endstate, diff_gfa


def sample_buildings_to_match_gfa(buildings_avail, gfa_target, typology_buildings_avail):
    delta_gfa_dict = {}
    for i in range(1000):
        num_sampled_buildings = random.randrange(0, len(buildings_avail))
        sampled_buildings = random.sample(list(buildings_avail), num_sampled_buildings)
        total_gfa_sampled = 0.0
        for b in sampled_buildings:
            gfa_building = typology_buildings_avail.loc[b, 'GFA_m2']
            total_gfa_sampled += gfa_building
        delta_gfa = abs(round(gfa_target - total_gfa_sampled, 2))
        delta_gfa_dict[delta_gfa] = sampled_buildings
    buildings_sampled = delta_gfa_dict[min(delta_gfa_dict.keys())]
    return buildings_sampled


def revert_MULTI_RES_to_OFFICE(diff_gfa, scenario_statusquo, typology_dict, typology_updated, typology_endstate, year_endstate):
    if diff_gfa['OFFICE'] < 0:
        avail_MULTI_RES_buildings = \
            typology_updated[typology_updated['REFERENCE_x'] == 'from OFFICE'][typology_updated['1ST_USE_R'] >= 1][
                typology_updated['YEAR'] < year_endstate].index # single-use MULTI_RES that is not modified in year_endstate
        total_avail_gfa_to_OFFICE = typology_dict[scenario_statusquo].loc[avail_MULTI_RES_buildings]['GFA_m2'].sum()
        if total_avail_gfa_to_OFFICE > abs(diff_gfa['MULTI_RES']):
            delta_gfa_dict = {}
            for i in range(20):
                num_sampled_buildings = random.randrange(0, len(avail_MULTI_RES_buildings))
                sampled_buildings = random.sample(list(avail_MULTI_RES_buildings), num_sampled_buildings)
                gfa_to_SINGLE_RES = 0.0
                for b in sampled_buildings:
                    building_gfa = typology_dict[scenario_statusquo].loc[b]['GFA_m2']
                    gfa_to_SINGLE_RES += building_gfa
                delta_gfa = abs(round(abs(diff_gfa['SINGLE_RES']) - gfa_to_SINGLE_RES, 2))
                delta_gfa_dict[delta_gfa] = sampled_buildings
            buildings_to_SINGLE_RES = delta_gfa_dict[min(delta_gfa_dict.keys())]
        else:
            buildings_to_SINGLE_RES = avail_MULTI_RES_buildings
        print('Reverting...', len(buildings_to_SINGLE_RES), 'MULTI_RES to SINGLE_RES')
        MULTI_to_SINGLE_RES_gfa = typology_updated.loc[buildings_to_SINGLE_RES]['GFA_m2'].sum()
        # write typology
        typology_updated.loc[buildings_to_SINGLE_RES, :] = typology_dict[scenario_statusquo].loc[
                                                           buildings_to_SINGLE_RES, :]
        diff_gfa['SINGLE_RES'] = 0.0
        diff_gfa['MULTI_RES'] = diff_gfa['MULTI_RES'] - MULTI_to_SINGLE_RES_gfa
        typology_endstate = typology_endstate.drop(buildings_to_SINGLE_RES)
    return typology_updated, typology_endstate, diff_gfa


def revert_MULTI_RES_to_SINGLE_RES(diff_gfa, scenario_statusquo, typology_dict, typology_updated, typology_endstate, year_endstate):
    if diff_gfa['SINGLE_RES'] < 0:
        avail_MULTI_RES_buildings = \
            typology_updated[typology_updated['REFERENCE_x'] == 'from SINGLE_RES'][typology_updated['1ST_USE_R'] >= 1][
                typology_updated['YEAR'] < year_endstate].index # single-use MULTI_RES that is not modified in year_endstate
        total_avail_gfa_to_SINGLE_RES = typology_dict[scenario_statusquo].loc[avail_MULTI_RES_buildings]['GFA_m2'].sum()
        if total_avail_gfa_to_SINGLE_RES > abs(diff_gfa['SINGLE_RES']):
            delta_gfa_dict = {}
            for i in range(100):
                num_sampled_buildings = random.randrange(0, len(avail_MULTI_RES_buildings))
                sampled_buildings = random.sample(list(avail_MULTI_RES_buildings), num_sampled_buildings)
                gfa_to_SINGLE_RES = 0.0
                for b in sampled_buildings:
                    building_gfa = typology_dict[scenario_statusquo].loc[b]['GFA_m2']
                    gfa_to_SINGLE_RES += building_gfa
                delta_gfa = abs(round(abs(diff_gfa['SINGLE_RES']) - gfa_to_SINGLE_RES, 2))
                delta_gfa_dict[delta_gfa] = sampled_buildings
            buildings_to_SINGLE_RES = delta_gfa_dict[min(delta_gfa_dict.keys())]
        else:
            buildings_to_SINGLE_RES = avail_MULTI_RES_buildings
        print('Reverting...', len(buildings_to_SINGLE_RES), 'MULTI_RES to SINGLE_RES')
        MULTI_to_SINGLE_RES_gfa = typology_updated.loc[buildings_to_SINGLE_RES]['GFA_m2'].sum()
        # write typology
        typology_updated.loc[buildings_to_SINGLE_RES, :] = typology_dict[scenario_statusquo].loc[
                                                           buildings_to_SINGLE_RES, :]
        diff_gfa['SINGLE_RES'] = 0.0
        diff_gfa['MULTI_RES'] = diff_gfa['MULTI_RES'] - MULTI_to_SINGLE_RES_gfa
        typology_endstate = typology_endstate.drop(buildings_to_SINGLE_RES)
    return typology_updated, typology_endstate, diff_gfa


def modify_typology_per_building_usetype(usetype, typology_updated, typology_endstate, diff_gfa, scenario_year):
    # select possible buildings to remove from endstate
    floors_usetype_endstate, footprint_usetype_endstate = get_building_candidates(usetype, typology_endstate)
    selected_floors_to_reduce_usetype = None
    # print(diff_gfa[usetype], min(footprint_usetype_endstate))
    if diff_gfa[usetype] > min(footprint_usetype_endstate):
        selected_floors_to_reduce_usetype = select_buildings_from_candidates(diff_gfa[usetype], floors_usetype_endstate,
                                                                             footprint_usetype_endstate)
    # update typology
    buildings_modified = []
    if not selected_floors_to_reduce_usetype is None:
        typology_updated = write_selected_buildings_in_typology(usetype, selected_floors_to_reduce_usetype,
                                                                floors_usetype_endstate, typology_endstate,
                                                                typology_updated, scenario_year)
        buildings_modified = list(selected_floors_to_reduce_usetype.index)
    return typology_updated, buildings_modified


def write_selected_buildings_in_typology(building_usetype, selected_floors_to_reduce, floors_of_usetype_end,
                                         typology_endstate, typology_updated, scenario_year):
    selected_buildings = selected_floors_to_reduce.index
    floors_of_usetype_updated = (floors_of_usetype_end[selected_buildings] - selected_floors_to_reduce).astype(int)
    # original floors
    original_total_floors = typology_updated.loc[selected_buildings, 'floors_all']
    expected_total_floors = original_total_floors - selected_floors_to_reduce
    # update floors
    for building in floors_of_usetype_updated.index:
        # find use_order
        building_usetypes = typology_updated.loc[building, ['1ST_USE', '2ND_USE', '3RD_USE']]
        # print(building_usetype, building_usetypes[building_usetypes == building_usetype])
        use_order = building_usetypes[building_usetypes == building_usetype].index[0]
        # update floors
        typology_updated.loc[building, use_order + '_F'] = floors_of_usetype_updated[building]
        updated_total_floors = typology_updated.loc[building, ['1ST_USE_F', '2ND_USE_F', '3RD_USE_F']].sum()
        typology_updated.loc[building, 'floors_all'] = updated_total_floors
        #     if not np.isclose(updated_total_floors, expected_total_floors[building]):
        if not abs(updated_total_floors - expected_total_floors[building]) < 2:
            print(building, use_order, 'updated:', updated_total_floors, 'expected:', expected_total_floors[building])
            a_cols = list(typology_endstate.filter(like='USE').columns)
            a_cols.extend(['floors_all'])
            print(typology_updated.loc[building, a_cols])
            raise ValueError(building, "Total number of floors mis-match excpeted number of floors")
        if np.isclose(typology_updated.loc[building, 'floors_all'], 0.0):
            # drop building if no floors_all in 2040
            typology_updated = typology_updated.drop(building)
            # print('dropping...', building)
        else:
            # recalculate ratios
            for use_order in ['1ST_USE', '2ND_USE', '3RD_USE']:
                usetype_ratio = typology_updated.loc[building, use_order + '_F'] / typology_updated.loc[
                    building, 'floors_all']
                typology_updated.loc[building, use_order + '_R'] = usetype_ratio
                typology_updated.loc[building, 'GFA_' + use_order] = typology_updated.loc[building, 'footprint'] * \
                                                                     typology_updated.loc[building, use_order + '_F']
            typology_updated.loc[building, 'floors_ag'] = typology_updated.loc[building, 'floors_ag'] - \
                                                          selected_floors_to_reduce[building]

            # find out if building changed from 2020
            # if yes, change year
            # if not, do nothing
            typology_updated.loc[building, 'YEAR'] = scenario_year  # TODO: compare to 2020 or REDUNDANT?
    # calculate errors
    # gfa of building_usetype in typology_updated
    # buildings_not_yet_built = typology_updated[typology_updated['YEAR'] == 2060].index
    # typology_updated = typology_updated.drop(buildings_not_yet_built)
    if typology_updated.isnull().sum().sum() > 0:
        raise ValueError('nan value in typology_updated')
    return typology_updated


def get_district_typology_merged(path_to_input):
    zone_gdf = gpd.read_file(os.path.join(path_to_input, 'building-geometry\\zone.shp')).set_index('Name')
    zone_gdf['footprint'] = zone_gdf.area
    zone_gdf['GFA_m2'] = zone_gdf['footprint'] * (zone_gdf['floors_ag'] + zone_gdf['floors_bg'])
    zone_gdf['GFA_ag_m2'] = zone_gdf['footprint'] * zone_gdf['floors_ag']
    zone_gdf['GFA_bg_m2'] = zone_gdf['footprint'] * zone_gdf['floors_bg']
    typology_df = dbf_to_dataframe(os.path.join(path_to_input, 'building-properties\\typology.dbf')).set_index('Name')
    # merge
    typology_merged = typology_df.merge(zone_gdf, left_on='Name', right_on='Name')
    typology_merged = typology_merged.fillna('-')
    # calculate other values
    typology_merged['floors_all'] = typology_merged['floors_ag'] + typology_merged['floors_bg']
    for use_order in ['1ST_USE', '2ND_USE', '3RD_USE']:
        typology_merged["GFA_" + use_order] = typology_merged[use_order + "_R"] * typology_merged["GFA_m2"]
        typology_merged[use_order + '_F'] = (round(typology_merged['floors_all'] * typology_merged[use_order + '_R'])).astype(int)
    # initialize columns
    typology_merged["additional_floors"] = 0
    typology_merged["floors_ag_updated"] = typology_merged.floors_ag.astype(int)
    typology_merged["height_ag_updated"] = typology_merged.height_ag.astype(int)
    if "orig_uses" not in typology_merged.columns:
        typology_merged["orig_uses"] = [[] for _ in range(len(typology_merged))]
    else:
        typology_merged["orig_uses"] = typology_merged["orig_uses"].apply(lambda x: x.strip('[\'\']').split())
    if "new_uses" not in typology_merged.columns:
        typology_merged["new_uses"] = [[] for _ in range(len(typology_merged))]
    else:
        typology_merged["new_uses"] = typology_merged["new_uses"].apply(lambda x: x.strip('[\'\']').split())
    typology_merged.fillna('-', inplace=True)
    return typology_merged


def get_gfa_per_usetype(typology_merged, key):
    # GFA per use whole district # TODO: import from cea utilities
    gfa_series_1st_use = typology_merged.groupby("1ST_USE").sum().loc[:, "GFA_1ST_USE"]
    gfa_series_2nd_use = typology_merged.groupby("2ND_USE").sum().loc[:, "GFA_2ND_USE"]
    gfa_series_3rd_use = typology_merged.groupby("3RD_USE").sum().loc[:, "GFA_3RD_USE"]
    gfa_per_use_type = defaultdict(float)
    for use_series in [gfa_series_1st_use, gfa_series_2nd_use, gfa_series_3rd_use]:
        for use, val in use_series.iteritems():
            gfa_per_use_type[use] += val
    # use_not_in_district = set(use_cols) - set(list(gfa_per_use_type.keys()))
    # for use in use_not_in_district:
    #     gfa_per_use_type[use] = 0.0
    gfa_per_use_type = dict(gfa_per_use_type)
    gfa_per_use_type_df = pd.DataFrame.from_dict(gfa_per_use_type, orient="index").T
    # gfa_per_use_type_df = gfa_per_use_type_df[use_cols]
    gfa_per_use_type_df.index = [key]
    return gfa_per_use_type_df


if __name__ == "__main__":
    config = cea.config.Configuration()
    config.project = r'C:\Users\shsieh\Desktop\TEST_AI'

    scenario_locator_sequences = {}
    s_name = '2020'
    config.scenario_name = s_name
    scenario_locator_sequences[s_name] = cea.inputlocator.InputLocator(scenario=config.scenario, plugins=config.plugins)
    path_to_case_study_inputs = os.path.join(config.scenario, "case_study_inputs.xlsx")
    worksheet = f"{config.remap_ville_scenarios.district_archetype}_{config.remap_ville_scenarios.urban_development_scenario}"
    case_study_inputs_df = pd.read_excel(path_to_case_study_inputs, sheet_name=worksheet).set_index('year')
    s_name = '2040_BAU'
    year_endstate = 2040
    config.scenario_name = s_name
    scenario_locator_sequences[s_name] = cea.inputlocator.InputLocator(scenario=config.scenario, plugins=config.plugins)

    config.remap_ville_scenarios.year = 2060
    config.remap_ville_scenarios.urban_development_scenario = 'BAU'
    s_name = f'{config.remap_ville_scenarios.year}_{config.remap_ville_scenarios.urban_development_scenario}_test'
    config.scenario_name = s_name
    new_locator = cea.inputlocator.InputLocator(scenario=config.scenario, plugins=config.plugins)
    year_intermediate = 2060
    case_study_inputs = case_study_inputs_df.loc[int(config.remap_ville_scenarios.year)]
    os.mkdir(config.scenario)
    os.mkdir(new_locator.get_input_folder())
    os.mkdir(new_locator.get_building_geometry_folder())
    os.mkdir(new_locator.get_building_properties_folder())

    main(config, new_locator, scenario_locator_sequences, case_study_inputs, year_intermediate, year_endstate)
