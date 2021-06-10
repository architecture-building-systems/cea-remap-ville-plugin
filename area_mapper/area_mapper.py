__author__ = "Anastasiya Popova"
__copyright__ = "Copyright 2021, Architecture and Building Systems - ETH Zurich"
__credits__ = ["Daren Thomas"]
__license__ = "MIT"
__version__ = "0.1"
__maintainer__ = "Anastasiya Popova"
__email__ = "cea@arch.ethz.ch"
__status__ = "Production"

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
        choices = list(set(mapping[data_frame_copy.at[i, city_zone_name]]).difference(set(current_uses)))
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
        per_use_gfa: Dict,
        target: float
) -> NoReturn:
    result = parse_milp_solution(solution)
    actual_use = {k: 0 for k in set(sub_building_use.values())}
    for v in result:
        actual_use[sub_building_use[v]] += sub_footprint_area[v] * result[v]

    # checks
    abs_error, rel_error = calculate_result_metrics(solution, target, sub_footprint_area)
    print("compare target [%.1f] vs. actual [%.1f]" % (target, sum(sub_footprint_area[v] * result[v] for v in result)))
    print("absolute error [%.4f]" % abs_error)
    print("relative error [%.8f]" % rel_error)

    metrics = dict()
    for use in actual_use:
        print("use [%s] actual [%.2f] vs. target [%.2f]" % (use, actual_use[use], per_use_gfa[use]))
        abs_error = abs(per_use_gfa[use] - actual_use[use])
        try:
            rel_error = abs_error / per_use_gfa[use]
        except ZeroDivisionError as e:
            rel_error = 0.0
        print("abs. error [%.4f]" % abs_error)
        print("rel. error [%.8f]" % rel_error)
        metrics[use] = {"actual_use": actual_use[use], "per_use_gfa": per_use_gfa[use]}

    return {"detailed_metrics": metrics, "absolute_error": abs_error, "relative_error": rel_error}


def calculate_result_metrics(
        solution: pulp.LpProblem,
        target: float,
        sub_footprint_area: Dict[str, float]
) -> Tuple[float, float]:
    res = parse_milp_solution(solution)
    abs_error = abs(target - sum(sub_footprint_area[v] * res[v] for v in res))
    rel_error = abs_error / target
    return abs_error, rel_error


def find_optimum_scenario(
        optimizations: Dict[str, Dict],
        target: float,
):
    errors = dict()
    minimum = None
    for i, optimization in enumerate(optimizations):
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
):
    """updates floors_ag and height_ag in zone.shp"""
    floors_ag_updated = defaultdict(int)
    height_ag_updated = defaultdict(int)
    result = parse_milp_solution(solution)
    for b, sb in building_to_sub_building.items():
        for _sb in sb:
            floors_ag_updated[b] += result[_sb]
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

    if path_to_output_zone_shape_file.exists():
        raise IOError("output zone.shp file [%s] already exists" % path_to_output_zone_shape_file)
    else:
        zone_shp_updated.to_file(path_to_output_zone_shape_file)


def parse_milp_solution(solution: pulp.LpProblem) -> Dict[str, int]:
    return {v.name.split("_")[1]: int(v.varValue) for v in solution.variables()}


def update_typology_file(
        solution: pulp.LpProblem,
        status_quo_typology: pd.DataFrame,
        optimized_typology: pd.DataFrame,
        building_to_sub_building: Dict[str, List[str]],
        path_to_output_typology_file: Path,
        columns_to_keep: List[Tuple],
        PARAMS
) -> NoReturn:
    """
    updates use ratios: updated_use_ratio = updated_number_of_floors_use_type_X/updated_total_number_of_floors, where:
    number_of_floors_use_type_X = additional_floors_use_type_X + initial_use_ratio_X * initial_total_number_of_floors
    """
    status_quo_typology = status_quo_typology.copy()
    simulated_typology = optimized_typology.copy()
    simulated_typology["1ST_USE_R"] = simulated_typology["1ST_USE_R"].astype(float)
    simulated_typology["2ND_USE_R"] = simulated_typology["2ND_USE_R"].astype(float)
    simulated_typology["3RD_USE_R"] = simulated_typology["3RD_USE_R"].astype(float)
    simulated_typology["REFERENCE"] = "after-optimization"

    use_col_dict = {i: column for i, column in enumerate(["1ST_USE", "2ND_USE", "3RD_USE"])}
    result = parse_milp_solution(solution)
    for building, sub_buildings in building_to_sub_building.items():
        updated_floors = dict()
        current_floors = status_quo_typology.loc[building, "floors_ag"]
        total_additional_floors = sum([result[y] for y in sub_buildings])
        total_floors = current_floors + total_additional_floors
        for i, sub_building in enumerate(sub_buildings):
            sub_building_additional_floors = result[sub_building]
            current_ratio = status_quo_typology.loc[building, use_col_dict[i] + '_R']
            updated_floors[use_col_dict[i]] = (sub_building_additional_floors + (current_ratio * current_floors))
            for use_col in updated_floors:
                r = updated_floors[use_col] / total_floors
                simulated_typology.loc[building, use_col + '_R'] = r
                if np.isclose(r, 0.0):
                    simulated_typology.loc[building, use_col] = "NONE"
                if simulated_typology.loc[building, use_col] == 'MULTI_RES':
                    if sub_building_additional_floors > 0 or status_quo_typology.loc[building, use_col] == 'SINGLE_RES':
                        simulated_typology.loc[building, use_col] = PARAMS[
                            'MULTI_RES_USE_TYPE']  # FIXME: TAKE FROM INPUT
        if not np.isclose(total_floors, sum(updated_floors.values())):
            raise ValueError("total number of floors mis-match excpeted number of floors")
    if path_to_output_typology_file.exists():
        raise IOError("output typology_updated.dbf file [%s] already exists" % path_to_output_typology_file)
    else:
        output = simulated_typology.copy()
        keep = list()
        for column, column_type in columns_to_keep:
            keep.append(column)
            output[column] = output[column].astype(column_type)
        dataframe_to_dbf(output[keep], str(path_to_output_typology_file.absolute()))
