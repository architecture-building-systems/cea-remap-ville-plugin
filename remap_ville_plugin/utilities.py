import pandas as pd
from collections import defaultdict


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


def typology_use_columns():
    return ["1ST_USE", "2ND_USE", "3RD_USE"]

def count_usetype(typology_updated):
    use_count_df = typology_updated[['1ST_USE', '2ND_USE', '3RD_USE']]
    use_count_df = use_count_df.apply(pd.value_counts)
    return use_count_df