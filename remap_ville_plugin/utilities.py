import pandas as pd
import numpy as np
import shutil
import operator
from collections import defaultdict
from cea.utilities.dbf import dbf_to_dataframe, dataframe_to_dbf

def save_updated_typology(path_to_output_typology_file, simulated_typology):
    output = simulated_typology.copy()
    keep = list()
    columns_to_keep = [("Name", str), ("YEAR", int), ("STANDARD", str), ("1ST_USE", str), ("1ST_USE_R", float),
                       ("2ND_USE", str), ("2ND_USE_R", float), ("3RD_USE", str), ("3RD_USE_R", float),
                       ("REFERENCE", str)]
    for column, column_type in columns_to_keep:
        keep.append(column)
        output[column] = output[column].astype(column_type)
    if output.isnull().sum().sum() > 0:
        raise ValueError('nan values in output')
    dataframe_to_dbf(output[keep], str(path_to_output_typology_file.absolute()))
    print(f'typology.dbf updated: {str(path_to_output_typology_file.absolute())}')
    return

def copy_folder(src, dst):
    print(f" - {dst}")
    shutil.copytree(src, dst)


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

def filter_buildings_by_year(typology_df: pd.DataFrame, year: int):
    if "YEAR" not in typology_df:
        raise KeyError("provided data frame is missing the column 'YEAR'")
    typology_before_year = typology_df[operator.lt(typology_df.YEAR, year + 1)]
    typology_after_year = typology_df[operator.gt(typology_df.YEAR, year)]
    return typology_before_year, typology_after_year

def typology_use_columns():
    return ["1ST_USE", "2ND_USE", "3RD_USE"]

def count_usetype(typology_updated):
    use_count_df = typology_updated[['1ST_USE', '2ND_USE', '3RD_USE']]
    use_count_df = use_count_df.apply(pd.value_counts)
    return use_count_df


def order_uses_in_typology(typology_df):
    ordered_typology_df = typology_df.copy()
    mixed_use_buildings = typology_df[typology_df['1ST_USE_R'] < 1.0].index
    for building in mixed_use_buildings:
        ratios = typology_df.loc[building, ['1ST_USE_R', '2ND_USE_R', '3RD_USE_R']].values
        uses = typology_df.loc[building, ['1ST_USE', '2ND_USE', '3RD_USE']].values
        order = np.argsort(ratios)
        ordered_ratios = [ratios[order[-1]], ratios[order[-2]], ratios[order[-3]]]
        ordered_uses = [uses[order[-1]], uses[order[-2]], uses[order[-3]]]
        # set use to NONE
        for i, ratio in enumerate(ordered_ratios):
            if ratio == 0.0:
                ordered_uses[i] = 'NONE'
        # move parking back
        if ordered_uses[0] == 'PARKING' and ordered_uses[1] != 'NONE':
            ordered_uses = [ordered_uses[1], ordered_uses[0], ordered_uses[2]]
            ordered_ratios = [ordered_ratios[1], ordered_ratios[0], ordered_ratios[2]]
        # write back to dbf_df
        for i, use_order in enumerate(['1ST_USE', '2ND_USE', '3RD_USE']):
            ordered_typology_df.loc[building, use_order] = ordered_uses[i]
            ordered_typology_df.loc[building, use_order + '_R'] = ordered_ratios[i]
    return ordered_typology_df

