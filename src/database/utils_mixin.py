import warnings
from abc import ABC

from config.config_table_structure import *
from src.config_loader import *
from src.database.base_mixin import BaseMixin

warnings.simplefilter(action='ignore', category=UserWarning)


class UtilsMixin(BaseMixin, ABC):
    def __init__(self):
        super().__init__()

    def __del__(self):
        self.cur.close()
        self.conn.close()

    def create_temp_tables(self) -> None:
        for query in TEMP_CREATE_QUERIES.values():
            self.cur.execute(query, {"epsg": EPSG})

    def drop_temp_tables(self) -> None:
        for table_name in TEMP_CREATE_QUERIES.keys():
            self.cur.execute(f"DROP TABLE IF EXISTS {table_name}")
        self.cur.execute("DROP TABLE IF EXISTS ways_tem_vertices_pgr")

    def commit_changes(self):
        self.conn.commit()

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
