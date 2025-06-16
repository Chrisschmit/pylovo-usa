import time
import warnings

import numpy as np
import psycopg2 as psy

from config.config_table_structure import *
from src import utils
from src.config_loader import *

warnings.simplefilter(action='ignore', category=UserWarning)


class UtilsMixin:
    def __del__(self):
        self.cur.close()
        self.conn.close()

    def create_temp_tables(self) -> None:
        for query in TEMP_CREATE_QUERIES.values():
            self.cur.execute(query)

    def drop_temp_tables(self) -> None:
        for table_name in TEMP_CREATE_QUERIES.keys():
            self.cur.execute(f"DROP TABLE IF EXISTS {table_name}")
        self.cur.execute("DROP TABLE IF EXISTS ways_tem_vertices_pgr")

    def commit_changes(self):
        self.conn.commit()
