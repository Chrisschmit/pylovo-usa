import time
import warnings

import numpy as np

from config.config_table_structure import *
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

    def calculate_cost_arr_dist_matrix(self, costmatrix_query: str, params: dict) -> tuple[dict, np.ndarray, dict]:
        """
        Helper function for calculating cost array and distance matrix from given parameters
        """
        st = time.time()
        cost_df = pd.read_sql_query(costmatrix_query, con=self.conn, params=params,
                                    dtype={"start_vid": np.int32, "end_vid": np.int32, "agg_cost": np.int32}, )
        cost_arr = cost_df.to_numpy()
        et = time.time()
        self.logger.debug(f"Elapsed time for SQL to cost_arr: {et - st}")
        # Speichere die echte vertices_ids mit neuen Indexen
        # 0 5346
        # 1 3263
        # 2 3653
        # ...
        localid2vid = dict(enumerate(cost_df["start_vid"].unique()))
        vid2localid = {y: x for x, y in localid2vid.items()}

        # Square distance matrix
        dist_matrix = np.zeros([len(localid2vid), len(localid2vid)])
        st = time.time()
        for i in range(len(cost_df)):
            start_id = vid2localid[cost_arr[i, 0]]
            end_id = vid2localid[cost_arr[i, 1]]
            dist_matrix[start_id][end_id] = cost_arr[i, 2]
        et = time.time()
        self.logger.debug(f"Elapsed time for dist_matrix creation: {et - st}")
        return localid2vid, dist_matrix, vid2localid

    def get_list_from_plz(self, plz: int) -> list:
        query = """SELECT DISTINCT kcid, bcid
                   FROM grid_result
                   WHERE version_id = %(v)s
                     AND plz = %(p)s
                   ORDER BY kcid, bcid;"""
        self.cur.execute(query, {"p": plz, "v": VERSION_ID})
        cluster_list = self.cur.fetchall()

        return cluster_list

    def delete_transformers_from_buildings_tem(self, vertices: list) -> None:
        """
        Deletes selected transformers from buildings_tem
        :param vertices:
        :return:
        """
        query = """
                DELETE
                FROM buildings_tem
                WHERE vertice_id IN %(v)s;"""
        self.cur.execute(query, {"v": tuple(map(int, vertices))})

    def get_consumer_categories(self):
        """
        Returns: A dataframe with self-defined consumer categories and typical values
        """
        query = """SELECT *
                   FROM consumer_categories"""
        cc_df = pd.read_sql_query(query, self.conn)
        cc_df.set_index("definition", drop=False, inplace=True)
        cc_df.sort_index(inplace=True)
        self.logger.debug("Consumer categories fetched.")
        return cc_df

    def get_settlement_type_from_plz(self, plz) -> int:
        """
        Args:
            plz:
        Returns: Settlement type: 1=City, 2=Village, 3=Rural
        """
        settlement_query = """SELECT settlement_type
                              FROM postcode_result
                              WHERE postcode_result_plz = %(p)s
                              LIMIT 1; """
        self.cur.execute(settlement_query, {"p": plz})
        settlement_type = self.cur.fetchone()[0]

        return settlement_type
