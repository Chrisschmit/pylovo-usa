from src.database_modules.databaseRead import ReadMixin
from src.database_modules.databaseWrite import WriteMixin
from src.database_modules.databaseClustering import ClusteringMixin
from src.database_modules.databaseGrid import GridMixin
from src.database_modules.databaseAnalysis import AnalysisMixin
from src.database_modules.databaseUtils import UtilsMixin

import json
import math
import time
from typing import *
from decimal import *

import geopandas as gpd
import pandapower as pp
import pandapower.topology as top
import psycopg2 as psy
import numpy as np
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform
from shapely.geometry import LineString
from sqlalchemy import create_engine, text

from src import utils
from src.config_table_structure import *
from src.config_loader import *

import warnings
warnings.simplefilter(action='ignore', category=UserWarning)


class DatabaseClient(ReadMixin, WriteMixin, ClusteringMixin, GridMixin, AnalysisMixin, UtilsMixin):
    """Main database client handling connections."""

    def __init__(self, dbname=DBNAME, user=USER, pw=PASSWORD, host=HOST, port=PORT, **kwargs):
        self.logger = utils.create_logger(
            "DatabaseClient", log_file=kwargs.get("log_file", "log.txt"), log_level=LOG_LEVEL
        )
        try:
            self.conn = psy.connect(
                database=dbname,
                user=user,
                password=pw,
                host=host,
                port=port,
                options=f"-c search_path={TARGET_SCHEMA},public",
            )
            self.cur = self.conn.cursor()
            self.db_path = f"postgresql+psycopg2://{user}:{pw}@{host}:{port}/{dbname}"
            self.sqla_engine = create_engine(
                self.db_path,
                connect_args={"options": f"-c search_path={TARGET_SCHEMA},public"},
            )
        except psy.OperationalError as err:
            self.logger.warning(
                f"Connecting to {dbname} was not successful. Make sure, that you have established the SSH connection with correct port mapping."
            )
            raise err

        self.logger.debug(f"DatabaseClient is constructed and connected to {self.db_path}.")

    def __del__(self):
        self.cur.close()
        self.conn.close()
