import pandas as pd
import numpy as np
import os
import geopandas as gpd
import pulp
from collections import defaultdict

import cea.config
import cea.inputlocator
from cea.utilities.dbf import dbf_to_dataframe

path_to_folder = r'C:\Users\shsieh\Desktop\TEST_UT_REDUCE\Echallens'
use_cols = ['MULTI_RES', 'SINGLE_RES', 'SECONDARY_RES', 'HOTEL', 'OFFICE', 'RETAIL', 'FOODSTORE',
            'RESTAURANT', 'INDUSTRIAL', 'SCHOOL', 'HOSPITAL', 'GYM', 'SWIMMING',
            'SERVERROOM', 'PARKING', 'COOLROOM', 'LAB', 'MUSEUM', 'LIBRARY',
            'UNIVERSITY']


def main(config):
    # 1. get projection
    typology_dict, gfa_per_use_dict = {}, {}
    path_to_input = os.path.join(path_to_folder, '2020/inputs')
    typology_dict['2020'] = get_district_typology_merged(path_to_input)
    path_to_input = os.path.join(path_to_folder, '2040_PUN/inputs')
    typology_dict['2040'] = get_district_typology_merged(
        path_to_input)  # FIXME: replace get_gfa_per_usetype by urban_transformation
    path_to_input = os.path.join(path_to_folder, '2060_PUN/inputs')
    typology_dict['2060'] = get_district_typology_merged(path_to_input)
    gfa_per_use_years_df = pd.concat(
        [get_gfa_per_usetype(typology_dict[key], key, use_cols) for key in typology_dict.keys()])
    # 2. get diff_gfa
    final_gfa = gfa_per_use_years_df.loc['2060']
    intermediate_gfa = gfa_per_use_years_df.loc['2040']
    diff_gfa = final_gfa - intermediate_gfa
    # 3. modify buildings
    typology_endstate = typology_dict['2060']
    typology_updated = typology_endstate.copy()
    # TODO: drop buildings that are kept as in 2020 state
    for usetype in use_cols:
        print('\n', usetype, 'GFA to reduce', round(diff_gfa[usetype], 1))
        if diff_gfa[usetype] > 0:
            typology_updated = modify_typology_per_building_usetype(usetype, typology_updated, typology_endstate,
                                                                    diff_gfa)
            # get errors
            gfa_updated = get_gfa_per_usetype(typology_updated, '2040', use_cols).loc['2040', usetype]
            gfa_projected = gfa_per_use_years_df.loc['2040', usetype]
            print('GFA(updated):', round(gfa_updated), ' GFA(projected):', round(gfa_projected))
    # 4. save typology_updated, zone_updated


def modify_typology_per_building_usetype(usetype, typology_updated, typology_endstate, diff_gfa):
    # select possible buildings to remove from endstate
    floors_usetype_endstate, footprint_usetype_endstate = get_building_candidates(usetype, typology_endstate)
    selected_floors_to_reduce_usetype = select_buildings_from_candidates(usetype, diff_gfa,
                                                                         floors_usetype_endstate,
                                                                         footprint_usetype_endstate)
    # update typology
    typology_updated = write_selected_buildings_in_typology(usetype, selected_floors_to_reduce_usetype,
                                                            floors_usetype_endstate, typology_endstate,
                                                            typology_updated)
    return typology_updated


def write_selected_buildings_in_typology(building_usetype, selected_floors_to_reduce, floors_of_usetype_end,
                                         typology_endstate, typology_updated):
    selected_buildings = selected_floors_to_reduce.index
    floors_of_usetype_updated = (floors_of_usetype_end[selected_buildings] - selected_floors_to_reduce).astype(int)
    # original floors
    original_total_floors = typology_updated.loc[selected_buildings, 'floors_all']
    expected_total_floors = original_total_floors - selected_floors_to_reduce
    # update floors
    for building in floors_of_usetype_updated.index:
        # find use_order
        building_usetypes = typology_updated.loc[building, ['1ST_USE', '2ND_USE', '3RD_USE']]
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
            b_updated = typology_updated.loc[building, a_cols]
            print(b_updated)
            raise ValueError(building, "Total number of floors mis-match excpeted number of floors")
        if np.isclose(typology_updated.loc[building, 'floors_all'], 0.0):
            # drop building if no floors_all in 2040
            typology_updated.loc[building, 'YEAR'] = 2060
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
            typology_updated.loc[building, 'YEAR'] = 2040  # TODO: compare to 2020
    # calculate errors
    # gfa of building_usetype in typology_updated
    buildings_not_yet_built = typology_updated[typology_updated['YEAR'] == 2060].index
    typology_updated = typology_updated.drop(buildings_not_yet_built)
    return typology_updated


def select_buildings_from_candidates(usetype, diff_gfa, floors_usetype, footprint_usetype):
    x_floors = optimization_problem(usetype, floors_usetype, footprint_usetype, diff_gfa)
    selected_floors_to_reduce_usetype = pd.Series(dtype=np.float)
    for key in x_floors.keys():
        if x_floors[key].varValue > 0:
            selected_floors_to_reduce_usetype[key] = x_floors[key].varValue
    print(len(selected_floors_to_reduce_usetype), 'buildings selected:', selected_floors_to_reduce_usetype.index)
    return selected_floors_to_reduce_usetype


def get_building_candidates(building_usetype, typology_endstate):
    floors_of_usetype, footprint_of_usetype = pd.Series(dtype=np.int), pd.Series(dtype=np.float)
    for use_order in ['1ST_USE', '2ND_USE', '3RD_USE']:
        buildings = list(typology_endstate.loc[typology_endstate[use_order] == building_usetype].index)
        floors_of_use_order = typology_endstate[use_order + '_F'].loc[buildings]
        floors_of_usetype = floors_of_usetype.append(floors_of_use_order)
        footprint_of_usetype = footprint_of_usetype.append(typology_endstate['footprint'].loc[buildings])
    # print('GFA status-quo:', round((footprint_of_usetype * floors_of_usetype).sum(), 1))
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
    # calculate other values
    typology_merged['floors_all'] = typology_merged['floors_ag'] + typology_merged['floors_bg']
    for use_order in ['1ST_USE', '2ND_USE', '3RD_USE']:
        typology_merged["GFA_" + use_order] = typology_merged[use_order + "_R"] * typology_merged["GFA_m2"]
        typology_merged[use_order + '_F'] = (typology_merged['floors_all'] * typology_merged[use_order + '_R']).astype(
            int)
    return typology_merged


def get_gfa_per_usetype(typology_merged, key, use_cols):
    # GFA per use whole district # TODO: import from cea utilities
    gfa_series_1st_use = typology_merged.groupby("1ST_USE").sum().loc[:, "GFA_1ST_USE"]
    gfa_series_2nd_use = typology_merged.groupby("2ND_USE").sum().loc[:, "GFA_2ND_USE"]
    gfa_series_3rd_use = typology_merged.groupby("3RD_USE").sum().loc[:, "GFA_3RD_USE"]
    gfa_per_use_type = defaultdict(float)
    for use_series in [gfa_series_1st_use, gfa_series_2nd_use, gfa_series_3rd_use]:
        for use, val in use_series.iteritems():
            gfa_per_use_type[use] += val
    use_not_in_district = set(use_cols) - set(list(gfa_per_use_type.keys()))
    for use in use_not_in_district:
        gfa_per_use_type[use] = 0.0
    gfa_per_use_type = dict(gfa_per_use_type)
    gfa_per_use_type_df = pd.DataFrame.from_dict(gfa_per_use_type, orient="index").T
    gfa_per_use_type_df = gfa_per_use_type_df[use_cols]
    gfa_per_use_type_df.index = [key]
    return gfa_per_use_type_df


if __name__ == "__main__":
    main(cea.config.Configuration())