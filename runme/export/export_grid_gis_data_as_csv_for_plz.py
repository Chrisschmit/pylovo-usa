# select single or multiple regional_identifier
# the geodata of the grids of the regional_identifier selected below is exported as two csv-files to be used for visualisation in QGIS
# one file contains the line and the other the bus data
import os
import sys

import pandas as pd

from plotting.export_net import save_geodata_as_csv
from src.config_loader import *

# enter the regional_identifier for which the geodata is exported
regional_identifier_list = ['91720', '80639']
df_regional_identifier = pd.DataFrame(
    regional_identifier_list,
    columns=['regional_identifier'])

# define the datapaths you want to export the grids to
line_datapath = os.path.abspath(
    os.path.join(
        PROJECT_ROOT,
        "QGIS",
        "lines_multiple_grids.csv"))
sys.path.append(line_datapath)
bus_datapath = os.path.abspath(
    os.path.join(
        PROJECT_ROOT,
        "QGIS",
        "bus_multiple_grids.csv"))
sys.path.append(bus_datapath)

save_geodata_as_csv(
    df_regional_identifier=df_regional_identifier,
    data_path_lines=line_datapath,
    data_path_bus=bus_datapath)
