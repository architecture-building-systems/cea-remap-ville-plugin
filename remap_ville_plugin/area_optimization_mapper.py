"""
optimization functions to map area increase per building for urban_transformation.py
"""

from typing import List, Dict, Set, Tuple, Union, NoReturn
from numpy.random import Generator, PCG64
from itertools import compress
from geopandas import GeoDataFrame
from collections import defaultdict
from pathlib import Path

from cea.utilities.dbf import dataframe_to_dbf

import pulp
import pandas as pd
import numpy as np

__author__ = "Anastasiya Popova, Shanshan Hsieh"
__copyright__ = "Copyright 2021, Architecture and Building Systems - ETH Zurich"
__credits__ = ["Daren Thomas"]
__license__ = "MIT"
__version__ = "0.1"
__maintainer__ = "Anastasiya Popova"
__email__ = "cea@arch.ethz.ch"
__status__ = "Production"

incompetibale_use_types = {
    "COOLROOM": ["SINGLE_RES"],
    "FOODSTORE": ["SINGLE_RES"],
    "GYM": ["SINGLE_RES"],
    "HOTEL": ["SCHOOL", "SINGLE_RES"],
    "HOSPITAL": ["MULTI_RES", "INDUSTRIAL", "SCHOOL", "SINGLE_RES"],
    "INDUSTRIAL": ["MULTI_RES", "HOSPITAL", "SCHOOL", "LIBRARY", "MUSEUM", "UNIVERSITY", "SINGLE_RES"],
    "LIBRARY": ["INDUSTRIAL", "SINGLE_RES"],
    "MULTI_RES": ["INDUSTRIAL", "SERVERROOM", "SWIMMING", "UNIVERSITY", "COOLROOM", "MUSEUM", "SCHOOL",
                  "HOSPITAL", "SINGLE_RES"],
    "MUSEUM": ["INDUSTRIAL", "SINGLE_RES"],
    "OFFICE": ["SINGLE_RES"],
    "RETAIL": ["SINGLE_RES"],
    "RESTAURANT": ["SINGLE_RES"],
    "SCHOOL": ["INDUSTRIAL", "HOSPITAL", "HOTEL", "FOODSTORE", "RETAIL", "RESTAURANT", "MULTI_RES", "SINGLE_RES"],
    "SERVERROOM": ["MULTI_RES", "SINGLE_RES"],
    "SWIMMING": ["SINGLE_RES"],
    "UNIVERSITY": ["SINGLE_RES"],
}


def randomize(
        data_frame: pd.DataFrame,
        generator: Generator,
        mapping: Dict[int, Set],
        use_columns: List[str],
        missing_use_name: str = "NONE",
        city_zone_name: str = "city_zone"
) -> pd.DataFrame:
    """
    generates additional building uses through randomization
    given rules by city_zone in 'mapping'
    """
    data_frame_copy = data_frame.copy()
    free_uses = data_frame_copy[use_columns] == missing_use_name
    for i, row in free_uses.iterrows():
        how_many_free_uses = row.sum()
        current_uses = data_frame_copy.loc[i, use_columns][~row].values
        # get available uses
        available_uses_cityzone = set(mapping[data_frame_copy.at[i, city_zone_name]])
        not_available_uses_building = []
        for use in current_uses:
            if use in incompetibale_use_types.keys():
                not_available_uses_building.extend(incompetibale_use_types[use])
        available_uses_building = available_uses_cityzone.difference(set(not_available_uses_building))
        choices = list(available_uses_building.difference(set(current_uses)))
        additional = generator.choice(choices, size=how_many_free_uses, replace=False)
        mask = additional == "NONE"
        if mask.sum() == 2:  # both empty uses
            pass
        elif mask.sum() == 0:  # no empty uses
            pass
        else:  # one empty use
            temp = additional[~mask]
            additional = np.append(temp, "NONE")
        mask = list(compress(use_columns, row))
        assert len(mask) == how_many_free_uses
        data_frame_copy.loc[i, mask] = additional

    return data_frame_copy


def optimize(
        target: float,
        target_variables: List[str],
        target_variable_min: int,
        target_variable_max: int,
        sub_building_footprint_area: Dict[str, int],
        building_to_sub_building: Dict[str, List[str]],
        range_additional_floors_per_building: Dict[str, List[int]],
        per_use_gfa: Dict[str, int],
        sub_building_use: Dict[str, str]
):
    """TODO: docstring"""
    x_additional_floors = pulp.LpVariable.dict(
        '',
        target_variables,
        target_variable_min,
        target_variable_max,
        pulp.LpInteger
    )

    opt_problem = pulp.LpProblem("maximize_gfa", pulp.LpMaximize)
    opt_problem += pulp.lpSum([x_additional_floors[i] * sub_building_footprint_area[i]
                               for i in target_variables])  # objective
    opt_problem += pulp.lpSum([x_additional_floors[i] * sub_building_footprint_area[i]
                               for i in target_variables]) <= target
    for building, sub_buildings in building_to_sub_building.items():
        lower, upper = range_additional_floors_per_building[building]
        opt_problem += pulp.lpSum([x_additional_floors[i] for i in sub_buildings]) >= lower
        opt_problem += pulp.lpSum([x_additional_floors[i] for i in sub_buildings]) <= upper
    for use, sub_target in per_use_gfa.items():
        cond = [
            x_additional_floors[i] * sub_building_footprint_area[i]
            for i in target_variables
            if sub_building_use[i] == use
        ]
        opt_problem += pulp.lpSum(cond) <= sub_target

    opt_problem.solve(pulp.GLPK(options=['--mipgap', '0.01']))
    return opt_problem


def _var_name(variable: pulp.LpVariable) -> str:
    return variable.name.split("_")[1]


def detailed_result_metrics(
        solution: pulp.LpProblem,
        sub_building_use: Dict,
        sub_footprint_area: Dict,
        target_add_gfa_per_use: Dict,
        target: float
) -> NoReturn:
    result_add_floors = parse_milp_solution(solution)
    result_add_gfa_per_use = {k: 0 for k in set(sub_building_use.values())}
    metrics = dict()
    if result_add_floors:
        # additional floors per use
        for v in result_add_floors:
            result_add_gfa_per_use[sub_building_use[v]] += sub_footprint_area[v] * result_add_floors[v]
        # calculate errors
        abs_error, rel_error = calculate_result_metrics(solution, target, sub_footprint_area)
        for use in result_add_gfa_per_use:
            print("use [%s] actual [%.1f] vs. target [%.1f]" % (
            use, result_add_gfa_per_use[use], target_add_gfa_per_use[use]))
            use_abs_error = abs(target_add_gfa_per_use[use] - result_add_gfa_per_use[use])
            try:
                use_rel_error = use_abs_error / target_add_gfa_per_use[use]
            except ZeroDivisionError as e:
                use_rel_error = 0.0
            print("    abs. error [%.1f]\n    rel. error [%.4f]" % (use_abs_error, use_rel_error))
            metrics[use] = {"result": int(result_add_gfa_per_use[use]),
                            "target": int(target_add_gfa_per_use[use])}
    else:
        abs_error = 1e20
        rel_error = 1e20
    return {"gfa_per_use": pd.DataFrame(metrics), "absolute_error": abs_error, "relative_error": rel_error}


def calculate_result_metrics(
        solution: pulp.LpProblem,
        total_addtitional_gfa_target: float,
        sub_footprint_area: Dict[str, float]
) -> Tuple[float, float]:
    result_additional_floors = parse_milp_solution(solution)
    if result_additional_floors:
        result_additioanl_gfa = sum(
            sub_footprint_area[v] * result_additional_floors[v] for v in result_additional_floors)
        abs_error = abs(total_addtitional_gfa_target - result_additioanl_gfa)
        rel_error = abs_error / total_addtitional_gfa_target
        print("compare total target [%.1f] vs. actual [%.1f]" % (total_addtitional_gfa_target, result_additioanl_gfa))
        print("total absolute error [%.1f], relative error [%.4f]" % (abs_error, rel_error))
    else:
        abs_error = 1e20
        rel_error = 1e20
    return abs_error, rel_error


def find_optimum_scenario(
        optimizations: Dict[str, Dict],
        target: float,
):
    errors = dict()
    minimum = None
    for i, optimization in enumerate(optimizations):
        print('scenario %i' % i)
        if minimum is None:
            minimum = i
        abs_error, rel_error = calculate_result_metrics(
            optimizations[optimization]["solution"],
            target,
            optimizations[optimization]["sub_footprint_area"]
        )
        errors[i] = abs_error
        if errors[minimum] > abs_error:
            minimum = i
    print("best scenario: [%i] with absolute error [%.1f]" % (minimum, errors[minimum]))
    return minimum, errors


def randomize_scenarios(
        typology_merged: pd.DataFrame,
        mapping: Dict[int, Set[str]],
        use_columns: List[str],
        scenario_count: int = 1000,
        seed: int = 12357
) -> Dict[int, pd.DataFrame]:
    scenarios = dict()
    generator = Generator(PCG64(seed))
    for scenario in range(scenario_count):
        df = typology_merged.copy()
        scenarios[scenario] = randomize(df, generator, mapping, use_columns)

    return scenarios


def update_zone_shp_file(
        solution: pulp.LpProblem,
        typology_merged: pd.DataFrame,
        building_to_sub_building: Dict[str, List[str]],
        path_to_input_zone_shape_file: Path,
        path_to_output_zone_shape_file: Path,
        sub_footprint_area,
        sub_building_use
):
    """updates floors_ag and height_ag in zone.shp"""

    floors_ag_updated = defaultdict(int)
    height_ag_updated = defaultdict(int)
    result = parse_milp_solution(solution)
    building_additional_gfa = defaultdict(int)
    for b, sb in building_to_sub_building.items():
        floors_ag_updated[b] = sum([result[_sb] for _sb in sb])
        building_additional_gfa[b] = [typology_merged.footprint[b] * floors_ag_updated[b]]
        height_ag_updated[b] = floors_ag_updated[b] * 3
        typology_merged.loc[b, "additional_floors"] = floors_ag_updated[b]
        typology_merged.loc[b, "floors_ag_updated"] = typology_merged.floors_ag[b] + floors_ag_updated[b]
        typology_merged.loc[b, "height_ag_updated"] = typology_merged.height_ag[b] + height_ag_updated[b]

    if path_to_input_zone_shape_file.exists():
        zone_shp_updated = GeoDataFrame.from_file(str(path_to_input_zone_shape_file.absolute()))
    else:
        raise IOError("input zone.shp file [%s] does not exist" % path_to_input_zone_shape_file)

    zone_shp_updated = zone_shp_updated.set_index("Name")
    zone_shp_updated["floors_ag"] = typology_merged["floors_ag_updated"]
    zone_shp_updated["height_ag"] = typology_merged["height_ag_updated"]
    zone_shp_updated["REFERENCE"] = "after-optimization"

    # check
    result_add_floors = parse_milp_solution(solution)
    result_add_gfa_per_use = {k: 0 for k in set(sub_building_use.values())}
    total_add_area = 0.0
    for v in result_add_floors:
        result_add_gfa_per_use[sub_building_use[v]] += sub_footprint_area[v] * result_add_floors[v]
        total_add_area += sub_footprint_area[v] * result_add_floors[v]

    calculated_additional_gfa = (zone_shp_updated['floors_ag'] - typology_merged['floors_ag']) * zone_shp_updated.area

    if path_to_output_zone_shape_file.exists():
        raise IOError("output zone.shp file [%s] already exists" % path_to_output_zone_shape_file)
    else:
        zone_shp_updated.to_file(path_to_output_zone_shape_file)


def parse_milp_solution(solution: pulp.LpProblem) -> Dict[str, int]:
    results = None
    none_list = 0
    for v in solution.variables():  # check if all outputs are valid
        if v.varValue is None:
            none_list += 1
    if none_list <= 0:
        results = {v.name.split("_")[1]: int(v.varValue) for v in solution.variables()}
    else:
        print("no result\n")
    return results


def update_typology_dbf(best_typology_df, result_add_floors, building_to_sub_building, typology_statusquo,
                        zone_shp_updated, floors_ag_updated, path_to_output_typology_file, PARAMS):
    status_quo_typology = typology_statusquo.copy()
    simulated_typology = best_typology_df.copy()

    zone_updated_gfa_per_building = zone_shp_updated.area * (
            zone_shp_updated['floors_ag'] + zone_shp_updated['floors_bg'])

    simulated_typology["1ST_USE_R"] = simulated_typology["1ST_USE_R"].astype(float)
    simulated_typology["2ND_USE_R"] = simulated_typology["2ND_USE_R"].astype(float)
    simulated_typology["3RD_USE_R"] = simulated_typology["3RD_USE_R"].astype(float)
    simulated_typology["REFERENCE"] = "after-optimization"
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
                if updated_floor_per_use_col[use_col] > 0 or status_quo_typology.loc[b, use_col] == 'SINGLE_RES':
                    simulated_typology.loc[b, use_col] = PARAMS['MULTI_RES_PLANNED']
                    # simulated_typology.loc[b, "STANDARD"] = "STANDARD5" # TODO: get from input
    save_updated_typology(path_to_output_typology_file, simulated_typology)


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
    zone_shp_updated = save_updated_zone_shp(best_typology_df, path_to_input_zone_shape_file,
                                             path_to_output_zone_shape_file)
    return floors_ag_updated, zone_shp_updated


def save_updated_typology(path_to_output_typology_file, simulated_typology):
    if path_to_output_typology_file.exists():
        raise IOError("output typology_updated.dbf file [%s] already exists" % path_to_output_typology_file)
    else:
        output = simulated_typology.copy()
        keep = list()
        columns_to_keep = [("Name", str), ("YEAR", int), ("STANDARD", str), ("1ST_USE", str), ("1ST_USE_R", float),
                           ("2ND_USE", str), ("2ND_USE_R", float), ("3RD_USE", str), ("3RD_USE_R", float),
                           ("REFERENCE", str)]
        for column, column_type in columns_to_keep:
            keep.append(column)
            output[column] = output[column].astype(column_type)
        dataframe_to_dbf(output[keep], str(path_to_output_typology_file.absolute()))


def save_updated_zone_shp(best_typology_df, path_to_input_zone_shape_file, path_to_output_zone_shape_file):
    if path_to_input_zone_shape_file.exists():
        zone_shp_updated = GeoDataFrame.from_file(str(path_to_input_zone_shape_file))
    else:
        raise IOError("input zone.shp file [%s] does not exist" % path_to_input_zone_shape_file)
    zone_shp_updated = zone_shp_updated.set_index("Name")
    zone_shp_updated["floors_ag"] = best_typology_df["floors_ag_updated"]
    zone_shp_updated["height_ag"] = best_typology_df["height_ag_updated"]
    zone_shp_updated["REFERENCE"] = "after-optimization"
    if path_to_output_zone_shape_file.exists():
        raise IOError("output zone.shp file [%s] already exists" % path_to_output_zone_shape_file)
    else:
        zone_shp_updated.to_file(path_to_output_zone_shape_file)
    return zone_shp_updated
