import os
from cea.utilities.dbf import dbf_to_dataframe, dataframe_to_dbf

path_to_project = r"C:\Users\shsieh\Documents\CEA_Porjects\echallens_0606"
scenario = '2040_density65'
path_to_architecture = os.path.join(path_to_project, scenario, "inputs\\building-properties\\air_conditioning.dbf")
air_conditioning_df = dbf_to_dataframe(path_to_architecture)
print(air_conditioning_df.columns)
air_conditioning_df['type_cs'] = 'HVAC_COOLING_AS2'
print(air_conditioning_df[['cool_starts', 'cool_ends']])

dataframe_to_dbf(air_conditioning_df, os.path.join(path_to_project, scenario, "inputs\\building-properties\\air_conditioning_new.dbf"))