import pandas as pd
import numpy as np
import os
import geopandas as gpd
import pulp
from collections import defaultdict
from pathlib import Path
import random

import cea.config
import cea.inputlocator
from cea.utilities.dbf import dbf_to_dataframe
from remap_ville_plugin.utilities import save_updated_typology, filter_buildings_by_year, order_uses_in_typology
from remap_ville_plugin.create_technology_database import create_input_technology_folder
import remap_ville_plugin.urban_transformation_preprocessing as preprocessing

path_to_folder = r'C:\Users\shsieh\Desktop\TEST_UT_REDUCE\Echallens'
use_cols = ['MULTI_RES', 'SINGLE_RES', 'SECONDARY_RES', 'HOTEL', 'OFFICE', 'RETAIL', 'FOODSTORE',
            'RESTAURANT', 'INDUSTRIAL', 'SCHOOL', 'HOSPITAL', 'GYM', 'SWIMMING',
            'SERVERROOM', 'PARKING', 'COOLROOM', 'LAB', 'MUSEUM', 'LIBRARY',
            'UNIVERSITY']


def main(config, new_locator, scenario_locator_sequences, case_study_inputs, scenario_year):
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
    typology_statusquo = typology_dict[scenario_statusquo]
    gfa_per_use_future_target, gfa_per_use_additional_target, gfa_total_additional_target, \
    overview, rel_ratio_to_res_gfa_target, \
    typology_planned, typology_statusquo = preprocessing.main(config, typology_statusquo, case_study_inputs)
    # missing_usetypes = set(use_cols) - set(gfa_per_use_future_target.index)
    # gfa_per_use_intermediate = gfa_per_use_future_target.add(pd.Series(index=missing_usetypes), fill_value=0.0)
    # gfa_per_use_intermediate = gfa_per_use_future_target.fillna(0.0)
    # gfa_per_use_intermediate = gfa_per_use_intermediate[use_cols]
    gfa_per_use_years_df.loc[scenario_intermediate] = gfa_per_use_future_target
    gfa_per_use_years_df.fillna(0.0, inplace=True)
    # 2. get diff_gfa
    diff_gfa = gfa_per_use_years_df.loc[scenario_endstate] - gfa_per_use_years_df.loc[scenario_intermediate]
    # remove 'MULTI_RES_PLANNED' from target
    diff_gfa['MULTI_RES'] = diff_gfa['MULTI_RES'] - diff_gfa[case_study_inputs['MULTI_RES_PLANNED']]
    diff_gfa = diff_gfa.drop(labels=case_study_inputs['MULTI_RES_PLANNED'])
    # 3. modify buildings
    typology_endstate = typology_dict[scenario_endstate]
    # remove preserved buildings
    typology_preserved_year, typology_after_year = filter_buildings_by_year(typology_endstate, year=case_study_inputs[
        "preserve_buildings_built_before"])
    typology_endstate = typology_after_year.copy()
    typology_updated = typology_endstate.copy()
    # revert MULTI_RES to SINGLE_RES
    typology_updated, diff_gfa = revert_MULTI_RES_to_SINGLE_RES(diff_gfa, scenario_statusquo, typology_dict,
                                                                typology_updated)
    buildings_modified = set()
    for usetype in diff_gfa.index:
        print('\n', usetype, 'GFA to reduce', round(diff_gfa[usetype], 1))
        if round(diff_gfa[usetype]) > 0:  # FIXME: quick solve to only move when m2 > 100
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
                print('GFA(updated):', round(gfa_updated), ' GFA(projected):', round(gfa_projected))
    # 4. save typology_updated, zone_updated
    buildings_not_yet_built = set(list(typology_endstate.index)) - set(list(typology_updated.index))
    buildings_modified = list(buildings_modified - buildings_not_yet_built)
    # typology
    typology_save = pd.concat([typology_updated, typology_preserved_year])
    typology_save['REFERENCE'] = typology_save['REFERENCE_x']
    typology_save.loc[buildings_modified, 'REFERENCE'] = 'sequential-transformation'
    # TODO: add back the buildings untouched
    typology_save_reindex = typology_save.reset_index()
    typology_save_reindex = order_uses_in_typology(typology_save_reindex)
    save_updated_typology(Path(new_locator.get_building_typology()), typology_save_reindex)
    print(new_locator.get_building_typology())
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
    print(f'zone.shp updates...{new_locator.get_zone_geometry()}')

    # create technology folder
    district_archetype = config.remap_ville_scenarios.district_archetype
    year = 2040
    urban_development_scenario = config.remap_ville_scenarios.urban_development_scenario
    folder_name = f"{district_archetype}_{year}_{urban_development_scenario}"
    create_input_technology_folder(folder_name, new_locator)


def revert_MULTI_RES_to_SINGLE_RES(diff_gfa, scenario_statusquo, typology_dict, typology_updated):
    if diff_gfa['SINGLE_RES'] < 0:
        avail_MULTI_RES_buildings = \
            typology_updated[typology_updated['REFERENCE_x'] == 'from SINGLE_RES'][typology_updated['1ST_USE_R'] >= 1][
                typology_updated['YEAR'] == 2060].index
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
        print('Reverting...', len(buildings_to_SINGLE_RES), ' to SINGLE_RES')
        MULTI_to_SINGLE_RES_gfa = typology_updated.loc[buildings_to_SINGLE_RES]['GFA_m2'].sum()
        # write typology
        typology_updated.loc[buildings_to_SINGLE_RES, :] = typology_dict[scenario_statusquo].loc[
                                                           buildings_to_SINGLE_RES, :]
        diff_gfa['SINGLE_RES'] = 0.0
        diff_gfa['MULTI_RES'] = diff_gfa['MULTI_RES'] - MULTI_to_SINGLE_RES_gfa
    return typology_updated, diff_gfa


def modify_typology_per_building_usetype(usetype, typology_updated, typology_endstate, diff_gfa, scenario_year):
    # select possible buildings to remove from endstate
    floors_usetype_endstate, footprint_usetype_endstate = get_building_candidates(usetype, typology_endstate)
    selected_floors_to_reduce_usetype = None
    # print(diff_gfa[usetype], min(footprint_usetype_endstate))
    if diff_gfa[usetype] > min(footprint_usetype_endstate):
        selected_floors_to_reduce_usetype = select_buildings_from_candidates(usetype, diff_gfa,
                                                                             floors_usetype_endstate,
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


def select_buildings_from_candidates(usetype, diff_gfa, floors_usetype, footprint_usetype):
    if len(floors_usetype) > 0:
        x_floors = optimization_problem(usetype, floors_usetype, footprint_usetype, diff_gfa)
        selected_floors_to_reduce_usetype = pd.Series(dtype=np.float)
        for key in x_floors.keys():
            if x_floors[key].varValue > 0:
                selected_floors_to_reduce_usetype[key] = x_floors[key].varValue
        print(len(selected_floors_to_reduce_usetype), 'buildings selected:', selected_floors_to_reduce_usetype.index)
    else:
        selected_floors_to_reduce_usetype = None
    return selected_floors_to_reduce_usetype


def get_building_candidates(building_usetype, typology_endstate):
    floors_of_usetype, footprint_of_usetype = pd.Series(dtype=np.int), pd.Series(dtype=np.float)
    for use_order in ['1ST_USE', '2ND_USE', '3RD_USE']:
        buildings = list(typology_endstate.loc[typology_endstate[use_order] == building_usetype].index)
        floors_of_use_order = typology_endstate[use_order + '_F'].loc[buildings]
        floors_of_usetype = floors_of_usetype.append(floors_of_use_order)
        footprint_of_usetype = footprint_of_usetype.append(typology_endstate['footprint'].loc[buildings])
    # print('GFA status-quo:', round((footprint_of_usetype * floors_of_usetype).sum(), 1))
    floors_of_usetype = floors_of_usetype[~np.isclose(floors_of_usetype, 0.0)]
    footprint_of_usetype = footprint_of_usetype[floors_of_usetype.index]
    print(len(footprint_of_usetype), building_usetype, 'buildings are in district.')
    return floors_of_usetype, footprint_of_usetype


def optimization_problem(building_usetype, floors_of_usetype, footprint_of_usetype, diff_gfa):
    # Initialize Class
    opt_problem = pulp.LpProblem("Maximize", pulp.LpMaximize)

    # Define Decision Variables
    target_variables = floors_of_usetype.index  # buildings
    target = diff_gfa[building_usetype]
    target_variable_min = 0
    target_variable_max = max(floors_of_usetype)

    x_floors = pulp.LpVariable.dict(
        '',
        target_variables,
        target_variable_min,
        target_variable_max,
        pulp.LpInteger
    )

    # Define Objective Function
    sub_building_footprint_area = footprint_of_usetype
    opt_problem += pulp.lpSum([x_floors[i] * sub_building_footprint_area[i]
                               for i in target_variables])  # objective

    # Define Constraints
    opt_problem += pulp.lpSum([x_floors[i] * sub_building_footprint_area[i]
                               for i in target_variables]) <= target
    for i in target_variables:
        opt_problem += x_floors[i] <= floors_of_usetype[i]

    # Solve Model
    # print(opt_problem)
    opt_problem.solve(pulp.GLPK(options=['--mipgap', '0.01'], msg=False))
    return x_floors


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
        typology_merged[use_order + '_F'] = (
            round(typology_merged['floors_all'] * typology_merged[use_order + '_R'])).astype(
            int)
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
    config.project = r'Z:\03-Research\01-Projects\30_VILLE\CEA_Projects\_CaseStudies\Echallens\CEA_inputs'

    scenario_locator_sequences = {}
    s_name = '2020_planned'
    config.scenario_name = s_name
    scenario_locator_sequences[s_name] = cea.inputlocator.InputLocator(scenario=config.scenario, plugins=config.plugins)
    path_to_case_study_inputs = os.path.join(config.scenario, "case_study_inputs.xlsx")
    worksheet = f"{config.remap_ville_scenarios.district_archetype}_{config.remap_ville_scenarios.urban_development_scenario}"
    case_study_inputs_df = pd.read_excel(path_to_case_study_inputs, sheet_name=worksheet).set_index('year')
    s_name = '2060_BAU'
    config.scenario_name = s_name
    scenario_locator_sequences[s_name] = cea.inputlocator.InputLocator(scenario=config.scenario, plugins=config.plugins)

    config.remap_ville_scenarios.year = 2040
    config.remap_ville_scenarios.urban_development_scenario = 'BAU'
    s_name = f'{config.remap_ville_scenarios.year}_{config.remap_ville_scenarios.urban_development_scenario}_test'
    config.scenario_name = s_name
    new_locator = cea.inputlocator.InputLocator(scenario=config.scenario, plugins=config.plugins)
    scenario_year = int(config.remap_ville_scenarios.year)
    case_study_inputs = case_study_inputs_df.loc[int(config.remap_ville_scenarios.year)]
    os.mkdir(config.scenario)
    os.mkdir(new_locator.get_input_folder())
    os.mkdir(new_locator.get_building_geometry_folder())
    os.mkdir(new_locator.get_building_properties_folder())

    main(config, new_locator, scenario_locator_sequences, case_study_inputs, scenario_year)