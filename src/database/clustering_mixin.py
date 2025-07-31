import math
import time
import warnings
from abc import ABC
from decimal import *
from typing import *

import numpy as np
from scipy.cluster.hierarchy import fcluster

from src import utils
from src.config_loader import *
from src.database.base_mixin import BaseMixin

warnings.simplefilter(action='ignore', category=UserWarning)


class ClusteringMixin(BaseMixin, ABC):
    def __init__(self):
        super().__init__()

    def get_connected_component(self) -> tuple[np.ndarray, np.ndarray]:
        """
        Reads from ways_tem
        :return:
        """
        component_query = """SELECT component, node
                             FROM pgr_connectedComponents(
                                     'SELECT way_id as id, source, target, cost, reverse_cost FROM ways_tem');"""
        self.cur.execute(component_query)
        data = self.cur.fetchall()
        component = np.asarray([i[0] for i in data])
        node = np.asarray([i[1] for i in data])

        return component, node

    def count_no_kmean_buildings(self):
        """
        Counts relative buildings in buildings_tem, which could not be clustered via k-means
        :return: count
        """
        query = """SELECT COUNT(*)
                   FROM buildings_tem
                   WHERE peak_load_in_kw != 0
                     AND kcid ISNULL;"""
        self.cur.execute(query)
        count = self.cur.fetchone()[0]

        return count

    def count_connected_buildings(self, vertices: Union[list, tuple]) -> int:
        """
        Get count from buildings_tem where type is not transformer
        :param vertices: np.array
        :return: count of buildings with given vertice_id s from buildings_tem
        """
        query = """SELECT COUNT(*)
                   FROM buildings_tem
                   WHERE vertice_id IN %(v)s
                     AND type != 'Transformer';"""
        self.cur.execute(query, {"v": tuple(map(int, vertices))})
        count = self.cur.fetchone()[0]

        return count

    def delete_ways(self, vertices: list) -> None:
        """
        Deletes selected ways from ways_tem and ways_tem_vertices_pgr
        :param vertices:
        :return:
        """
        query = """DELETE
                   FROM ways_tem
                   WHERE target IN %(v)s;
        DELETE
        FROM ways_tem_vertices_pgr
        WHERE id IN %(v)s;"""
        self.cur.execute(query, {"v": tuple(map(int, vertices))})

    def update_large_kmeans_cluster(
            self, vertices: Union[list, tuple], cluster_count: int):
        """
        Applies k-means clustering to large components and updated values in buildings_tem
        :param vertices:
        :param cluster_count:
        :return:
        """
        query = """
                WITH kmean AS (SELECT osm_id,
                                      ST_ClusterKMeans(center, %(ca)s)
                                      OVER () AS cid
                               FROM buildings_tem
                               WHERE vertice_id IN %(v)s),
                     maxk AS (SELECT MAX(kcid) AS max_k FROM buildings_tem)
                UPDATE buildings_tem b
                SET kcid = (CASE
                                WHEN m.max_k ISNULL THEN k.cid + 1
                                ELSE m.max_k + k.cid + 1
                    END)
                FROM kmean AS k,
                     maxk AS m
                WHERE b.osm_id = k.osm_id;"""
        self.cur.execute(query, {"ca": cluster_count,
                         "v": tuple(map(int, vertices))})

    def update_kmeans_cluster(self, vertices: list) -> None:
        """
        Groups connected components into a k-means id withouth applying clustering
        :param vertices:
        :return:
        """
        query = """
                WITH maxk AS (SELECT MAX(kcid) AS max_k FROM buildings_tem)
                UPDATE buildings_tem
                SET kcid = (CASE
                                WHEN m.max_k ISNULL THEN 1
                                ELSE m.max_k + 1
                    END)
                FROM maxk AS m
                WHERE vertice_id IN %(v)s;"""
        self.cur.execute(query, {"v": tuple(map(int, vertices))})

    def get_distance_matrix_from_kcid(
            self, kcid: int) -> tuple[dict, np.ndarray, dict]:
        """
        Creates a distance matrix from the buildings in the kcid
        Args:
            kcid: k-means cluster id
        Returns: The distance matrix of the buildings in the k-means cluster as np.array and the mapping between vertice_id and local ID as dict
        """

        costmatrix_query = """SELECT * \
                              FROM pgr_dijkstraCostMatrix( \
                                      'SELECT way_id as id, source, target, cost, reverse_cost FROM ways_tem', \
                                      (SELECT array_agg(DISTINCT b.connection_point) \
                                       FROM (SELECT * \
                                             FROM buildings_tem \
                                             WHERE kcid = %(k)s \
                                               AND bcid ISNULL \
                                             ORDER BY connection_point) AS b), \
                                      false);"""
        params = {"k": kcid}
        localid2vid, dist_mat, _ = self.calculate_cost_arr_dist_matrix(
            costmatrix_query, params)

        return localid2vid, dist_mat, _

    def calculate_cost_arr_dist_matrix(
            self, costmatrix_query: str, params: dict) -> tuple[dict, np.ndarray, dict]:
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

    def generate_load_vector(self, kcid: int, bcid: int) -> np.ndarray:
        query = """SELECT SUM(peak_load_in_kw)::float
                   FROM buildings_tem
                   WHERE kcid = %(k)s
                     AND bcid = %(b)s
                   GROUP BY connection_point
                   ORDER BY connection_point;"""
        self.cur.execute(query, {"k": kcid, "b": bcid})
        load = np.asarray([i[0] for i in self.cur.fetchall()])

        return load

    def try_clustering(self, Z: np.ndarray, cluster_amount: int, localid2vid: dict, buildings: pd.DataFrame,
                       consumer_cat_df: pd.DataFrame, transformer_capacities: np.ndarray, double_trans: np.ndarray, ) -> tuple[
            dict, dict, int]:
        # Clusters into maximum cluster amount -- 2 is the maximum
        flat_groups = fcluster(Z, t=cluster_amount, criterion="maxclust")
        cluster_ids = np.unique(flat_groups)
        cluster_count = len(cluster_ids)
        # Check if simultaneous load can be satisfied with possible
        # transformers
        cluster_dict = {}
        invalid_cluster_dict = {}
        # For each cluster, check if the load can be satisfied with possible
        # transformers
        for cluster_id in range(1, cluster_count + 1):
            # Python list of vertex ids (e.g. [142, 3891, 557 ...]) that belong
            # to the current hierarchical-cluster being inspected.
            vid_list = [localid2vid[lid[0]]
                        for lid in np.argwhere(flat_groups == cluster_id)]
            total_sim_load = utils.simultaneousPeakLoad(
                buildings, consumer_cat_df, vid_list)
            # In US first step we need to check for only small transformers
            # pole mounted
            if (total_sim_load >= max(transformer_capacities)
                    and len(vid_list) >= 5):  # the cluster is too big
                invalid_cluster_dict[cluster_id] = vid_list
            elif total_sim_load < max(transformer_capacities):
                # find the smallest transformer, that satisfies the load
                opt_transformer = transformer_capacities[transformer_capacities >
                                                         total_sim_load][0]
                opt_double_transformer = double_trans[double_trans >
                                                      total_sim_load * 1.15][0]
                if (opt_double_transformer -
                        total_sim_load) > (opt_transformer - total_sim_load):
                    cluster_dict[cluster_id] = (vid_list, opt_transformer)
                else:
                    cluster_dict[cluster_id] = (
                        vid_list, opt_double_transformer)
            else:
                # "Over-sized load but tiny cluster"
                opt_transformer = math.ceil(total_sim_load)
                cluster_dict[cluster_id] = (vid_list, opt_transformer)
        return invalid_cluster_dict, cluster_dict, cluster_count

    def get_kcid_length(self) -> int:
        query = """SELECT COUNT(DISTINCT kcid)
                   FROM buildings_tem
                   WHERE kcid IS NOT NULL; """
        self.cur.execute(query)
        kcid_length = self.cur.fetchone()[0]
        return kcid_length

    def get_next_unfinished_kcid(self, plz: int) -> int:
        """
        :return: one unmodeled k mean cluster ID - plz
        """
        query = """SELECT kcid
                   FROM buildings_tem
                   WHERE kcid NOT IN (SELECT DISTINCT kcid
                                      FROM grid_result
                                      WHERE version_id = %(v)s
                                        AND grid_result.plz = %(plz)s)
                     AND kcid IS NOT NULL
                   ORDER BY kcid
                   LIMIT 1;"""
        self.cur.execute(query, {"v": VERSION_ID, "plz": plz})
        kcid = self.cur.fetchone()[0]
        return kcid

    def get_included_transformers(self, kcid: int) -> list:
        """
        Reads the vertice ids of transformers from a given kcid
        :param kcid:
        :return: list
        """
        query = """SELECT vertice_id
                   FROM buildings_tem
                   WHERE kcid = %(k)s
                     AND type = 'Transformer';"""
        self.cur.execute(query, {"k": kcid})
        transformers_list = ([t[0] for t in data] if (
            data := self.cur.fetchall()) else [])
        return transformers_list

    def clear_grid_result_in_kmean_cluster(self, plz: int, kcid: int):
        # Remove old clustering at same postcode cluster
        clear_query = """DELETE
                         FROM grid_result
                         WHERE version_id = %(v)s
                           AND plz = %(pc)s
                           AND kcid = %(kc)s
                           AND bcid >= 0; """

        params = {"v": VERSION_ID, "pc": plz, "kc": kcid}
        self.cur.execute(clear_query, params)
        self.logger.debug(
            f"Building clusters with plz = {plz}, k_mean cluster = {kcid} area cleared.")

    def upsert_bcid(self, plz: int, kcid: int, bcid: int,
                    vertices: list, transformer_rated_power: int):
        """
        Assign buildings in buildings_tem the bcid and stores the cluster in grid_result
        Args:
            plz: postcode cluster ID - plz
            kcid: kmeans cluster ID
            bcid: building cluster ID
            vertices: List of vertice_id of selected buildings
            transformer_rated_power: Apparent power of the selected transformer
        """
        # Insert references to building elements in which cluster they are.
        building_query = """UPDATE buildings_tem
                            SET bcid = %(bc)s
                            WHERE plz = %(pc)s
                              AND kcid = %(kc)s
                              AND bcid ISNULL
                              AND connection_point IN %(vid)s
                              AND type != 'Transformer'; """

        params = {"v": VERSION_ID, "pc": plz, "bc": bcid,
                  "kc": kcid, "vid": tuple(map(int, vertices)), }
        self.cur.execute(building_query, params)

        # Insert new clustering
        cluster_query = """INSERT INTO grid_result (version_id, plz, kcid, bcid, transformer_rated_power)
                           VALUES (%(v)s, %(pc)s, %(kc)s, %(bc)s, %(s)s); """

        params = {"v": VERSION_ID, "pc": plz, "bc": bcid,
                  "kc": kcid, "s": int(transformer_rated_power)}
        self.cur.execute(cluster_query, params)

    def get_consumer_to_transformer_df(
            self, kcid: int, transformer_list: list) -> pd.DataFrame:
        consumer_query = """SELECT DISTINCT connection_point
                            FROM buildings_tem
                            WHERE kcid = %(k)s
                              AND type != 'Transformer';"""
        self.cur.execute(consumer_query, {"k": kcid})
        consumer_list = [t[0] for t in self.cur.fetchall()]

        cost_query = """SELECT *
                        FROM pgr_dijkstraCost(
                                'SELECT way_id as id, source, target, cost, reverse_cost FROM ways_tem',
                                %(cl)s, %(tl)s,
                                false);"""
        cost_df = pd.read_sql_query(cost_query, con=self.conn, params={"cl": consumer_list, "tl": transformer_list},
                                    dtype={"start_vid": np.int32, "end_vid": np.int32, "agg_cost": np.int32}, )

        return cost_df

    def count_kmean_cluster_consumers(self, kcid: int) -> int:
        query = """SELECT COUNT(DISTINCT vertice_id)
                   FROM buildings_tem
                   WHERE kcid = %(k)s
                     AND type != 'Transformer'
                     AND bcid ISNULL;"""
        self.cur.execute(query, {"k": kcid})
        count = self.cur.fetchone()[0]

        return count

    def delete_isolated_building(self, plz: int, kcid):
        query = """DELETE
                   FROM buildings_tem
                   WHERE plz = %(p)s
                     AND kcid = %(k)s
                     AND bcid ISNULL;"""
        self.cur.execute(query, {"p": plz, "k": kcid})

    def get_greenfield_bcids(self, plz: int, kcid: int) -> list:
        """
        Args:
            plz: loadarea cluster ID
            kcid: kmeans cluster ID
        Returns: A list of greenfield building clusters for a given plz
        """
        query = """SELECT DISTINCT bcid
                   FROM grid_result
                   WHERE version_id = %(v)s
                     AND kcid = %(kc)s
                     AND plz = %(pc)s
                     AND model_status ISNULL
                   ORDER BY bcid; """
        params = {"v": VERSION_ID, "pc": plz, "kc": kcid}
        self.cur.execute(query, params)
        bcid_list = [t[0] for t in data] if (
            data := self.cur.fetchall()) else []
        return bcid_list

    def get_buildings_from_kcid(self, kcid: int, ) -> pd.DataFrame:
        """
        Args:
            kcid: kmeans_cluster ID
        Returns: A dataframe with all building information
        """
        buildings_query = """SELECT *
                             FROM buildings_tem
                             WHERE connection_point IS NOT NULL
                               AND kcid = %(k)s
                               AND bcid ISNULL;"""
        params = {"k": kcid}

        buildings_df = pd.read_sql_query(
            buildings_query, con=self.conn, params=params)
        buildings_df.set_index("vertice_id", drop=False, inplace=True)
        buildings_df.sort_index(inplace=True)

        self.logger.debug(
            f"Building data fetched. {len(buildings_df)} buildings from kc={kcid} ...")

        return buildings_df

    def get_buildings_from_bcid(
            self, plz: int, kcid: int, bcid: int) -> pd.DataFrame:

        buildings_query = """SELECT *
                             FROM buildings_tem
                             WHERE type != 'Transformer'
                               AND plz = %(p)s
                               AND bcid = %(b)s
                               AND kcid = %(k)s;"""
        params = {"p": plz, "b": bcid, "k": kcid}

        buildings_df = pd.read_sql_query(
            buildings_query, con=self.conn, params=params)
        buildings_df.set_index("vertice_id", drop=False, inplace=True)
        buildings_df.sort_index(inplace=True)
        # dropping duplicate indices
        # buildings_df = buildings_df[~buildings_df.index.duplicated(keep='first')]

        self.logger.debug(f"{len(buildings_df)} building data fetched.")

        return buildings_df

    def update_transformer_rated_power(
            self, plz: int, kcid: int, bcid: int, note: int):
        """
        Updates transformer_rated_power in grid_result
        :param plz:
        :param kcid:
        :param bcid:
        :param note:
        :return:
        """
        sdl = self.get_settlement_type_from_plz(plz)
        transformer_capacities, _ = self.get_transformer_data(sdl)

        if note == 0:
            old_query = """SELECT transformer_rated_power
                           FROM grid_result
                           WHERE version_id = %(v)s
                             AND plz = %(p)s
                             AND kcid = %(k)s
                             AND bcid = %(b)s;"""
            self.cur.execute(
                old_query, {
                    "v": VERSION_ID, "p": plz, "k": kcid, "b": bcid})
            transformer_rated_power = self.cur.fetchone()[0]

            new_transformer_rated_power = transformer_capacities[transformer_capacities > transformer_rated_power][
                0].item()
            update_query = """UPDATE grid_result
                              SET transformer_rated_power = %(n)s
                              WHERE version_id = %(v)s
                                AND plz = %(p)s
                                AND kcid = %(k)s
                                AND bcid = %(b)s;"""
            self.cur.execute(update_query,
                             {"v": VERSION_ID, "p": plz, "k": kcid, "b": bcid, "n": new_transformer_rated_power}, )
        else:
            double_trans = np.multiply(transformer_capacities[2:4], 2)
            combined = np.concatenate(
                (transformer_capacities, double_trans), axis=None)
            np.sort(combined, axis=None)
            old_query = """SELECT transformer_rated_power
                           FROM grid_result
                           WHERE version_id = %(v)s
                             AND plz = %(p)s
                             AND kcid = %(k)s
                             AND bcid = %(b)s;"""
            self.cur.execute(
                old_query, {
                    "v": VERSION_ID, "p": plz, "k": kcid, "b": bcid})
            transformer_rated_power = self.cur.fetchone()[0]
            if transformer_rated_power in combined.tolist():
                return None
            new_transformer_rated_power = np.ceil(
                transformer_rated_power / 630) * 630
            update_query = """UPDATE grid_result
                              SET transformer_rated_power = %(n)s
                              WHERE version_id = %(v)s
                                AND plz = %(p)s
                                AND kcid = %(k)s
                                AND bcid = %(b)s;"""
            self.cur.execute(update_query,
                             {"v": VERSION_ID, "p": plz, "k": kcid, "b": bcid, "n": new_transformer_rated_power}, )
            self.logger.debug(
                "double or multiple transformer group transformer_rated_power assigned")

    def get_transformer_data(
            self, settlement_type: int = None) -> tuple[np.array, dict]:
        """
        Args:
            Settlement type: 1=City, 2=Village, 3=Rural
        Returns: Typical transformer capacities and costs depending on the settlement type
        """
        if settlement_type == 1:
            application_area_tuple = (1, 2, 3)
        elif settlement_type == 2:
            application_area_tuple = (2, 3, 4)
        elif settlement_type == 3:
            application_area_tuple = (3, 4, 5)
        else:
            self.logger.debug("Incorrect settlement type number specified.")
            return

        query = """SELECT equipment_data.s_max_kva, cost_eur
                   FROM equipment_data
                   WHERE typ = 'Transformer' \
                     AND application_area IN %(tuple)s
                   ORDER BY s_max_kva;"""

        self.cur.execute(query, {"tuple": application_area_tuple})
        data = self.cur.fetchall()
        capacities = [i[0] for i in data]
        transformer2cost = {i[0]: i[1] for i in data}

        self.logger.debug("Transformer data fetched.")
        return np.array(capacities), transformer2cost

    def update_building_cluster(self, transformer_id: int, conn_id_list: Union[list, tuple], count: int, kcid: int,
                                plz: int, transformer_rated_power: int) -> None:
        """
        Update building cluster information by performing multiple operations:
          - Update the 'bcid' in 'buildings_tem' where 'vertice_id' matches the transformer_id.
          - Update the 'bcid' in 'buildings_tem' for rows where 'connection_point' is in the provided list and type is not 'Transformer'.
          - Insert a new record into 'grid_result'.
          - Insert a new record into 'transformer_positions' using subqueries for geometry and OGC ID.
        Args:
            transformer_id (int): The ID of the transformer.
            conn_id_list (Union[list, tuple]): A list or tuple of connection point IDs.
            count (int): The new building cluster identifier.
            kcid (int): The KCID value.
            plz (int): The postcode value.
            transformer_rated_power (int): The selected transformer size for the building cluster.
        """
        query = """
                UPDATE buildings_tem
                SET bcid = %(count)s
                WHERE vertice_id = %(t)s;

                UPDATE buildings_tem
                SET bcid = %(count)s
                WHERE connection_point IN %(c)s
                  AND type != 'Transformer';

                INSERT INTO grid_result (version_id, plz, kcid, bcid, transformer_vertice_id, transformer_rated_power)
                VALUES (%(v)s, %(pc)s, %(k)s, %(count)s, %(t)s, %(l)s);

                INSERT INTO transformer_positions (version_id, grid_result_id, geom, osm_id, comment)
                VALUES (
                        %(v)s,
                        (SELECT grid_result_id
                         FROM grid_result
                         WHERE version_id = %(v)s AND plz = %(pc)s AND kcid = %(k)s AND bcid = %(count)s),
                        (SELECT center FROM buildings_tem WHERE vertice_id = %(t)s),
                        (SELECT osm_id FROM buildings_tem WHERE vertice_id = %(t)s),
                        'Normal'); \
                """
        params = {"v": VERSION_ID, "count": count, "c": tuple(conn_id_list), "t": transformer_id, "k": kcid, "pc": plz,
                  "l": transformer_rated_power, }
        self.cur.execute(query, params)

    def calculate_sim_load(self, conn_list: Union[tuple, list]) -> Decimal:
        residential = """WITH residential AS
                                  (SELECT b.peak_load_in_kw AS load, b.houses_per_building AS count, c.sim_factor
                                   FROM buildings_tem AS b
                                            LEFT JOIN consumer_categories AS c
                                                      ON b.type = c.definition
                                   WHERE b.connection_point IN %(c)s
                                     AND b.type IN ('SFH', 'MFH', 'AB', 'TH'))
                         SELECT SUM(load), SUM(count), sim_factor
                         FROM residential
                         GROUP BY sim_factor; \
                      """
        self.cur.execute(residential, {"c": tuple(conn_list)})

        data = self.cur.fetchone()
        if data:
            residential_load = Decimal(data[0])
            residential_count = Decimal(data[1])
            residential_factor = Decimal(data[2])
            residential_sim_load = residential_load * (
                residential_factor + (1 - residential_factor) * (residential_count ** Decimal(-3 / 4)))
        else:
            residential_sim_load = 0
        # TODO can the following 4 repetitions simplified with a general
        # function?
        commercial = """WITH commercial AS
                                 (SELECT b.peak_load_in_kw AS load, b.houses_per_building AS count, c.sim_factor
                                  FROM buildings_tem AS b
                                           LEFT JOIN consumer_categories AS c
                                                     ON c.definition = b.type
                                  WHERE b.connection_point IN %(c)s
                                    AND b.type = 'Commercial')
                        SELECT SUM(load), SUM(count), sim_factor
                        FROM commercial
                        GROUP BY sim_factor; \
                     """
        self.cur.execute(commercial, {"c": tuple(conn_list)})
        data = self.cur.fetchone()
        if data:
            commercial_load = Decimal(data[0])
            commercial_count = Decimal(data[1])
            commercial_factor = Decimal(data[2])
            commercial_sim_load = commercial_load * (
                commercial_factor + (1 - commercial_factor) * (commercial_count ** Decimal(-3 / 4)))
        else:
            commercial_sim_load = 0

        public = """WITH public AS
                             (SELECT b.peak_load_in_kw AS load, b.houses_per_building AS count, c.sim_factor
                              FROM buildings_tem AS b
                                       LEFT JOIN consumer_categories AS c
                                                 ON c.definition = b.type
                              WHERE b.connection_point IN %(c)s
                                AND b.type = 'Public')
                    SELECT SUM(load), SUM(count), sim_factor
                    FROM public
                    GROUP BY sim_factor; \
                 """
        self.cur.execute(public, {"c": tuple(conn_list)})
        data = self.cur.fetchone()
        if data:
            public_load = Decimal(data[0])
            public_count = Decimal(data[1])
            public_factor = Decimal(data[2])
            public_sim_load = public_load * \
                (public_factor + (1 - public_factor)
                 * (public_count ** Decimal(-3 / 4)))
        else:
            public_sim_load = 0

        industrial = """WITH industrial AS
                                 (SELECT b.peak_load_in_kw AS load, b.houses_per_building AS count, c.sim_factor
                                  FROM buildings_tem AS b
                                           LEFT JOIN consumer_categories AS c
                                                     ON c.definition = b.type
                                  WHERE b.connection_point IN %(c)s
                                    AND b.type = 'Industrial')
                        SELECT SUM(load), SUM(count), sim_factor
                        FROM industrial
                        GROUP BY sim_factor; \
                     """
        self.cur.execute(industrial, {"c": tuple(conn_list)})
        data = self.cur.fetchone()
        if data:
            industrial_load = Decimal(data[0])
            industrial_count = Decimal(data[1])
            industrial_factor = Decimal(data[2])
            industrial_sim_load = industrial_load * (
                industrial_factor + (1 - industrial_factor) * (industrial_count ** Decimal(-3 / 4)))
        else:
            industrial_sim_load = 0

        total_sim_load = (
            residential_sim_load +
            commercial_sim_load +
            industrial_sim_load +
            public_sim_load)

        return total_sim_load

    def get_building_connection_points_from_bc(
            self, kcid: int, bcid: int) -> list:
        """
        Args:
            kcid: kmeans_cluster ID
            bcid: building cluster ID
        Returns: A dataframe with all building information
        """
        count_query = """SELECT DISTINCT connection_point
                         FROM buildings_tem
                         WHERE vertice_id IS NOT NULL
                           AND bcid = %(b)s
                           AND kcid = %(k)s;"""
        params = {"b": bcid, "k": kcid}
        self.cur.execute(count_query, params)
        try:
            cp = [t[0] for t in self.cur.fetchall()]
        except BaseException:
            cp = []

        return cp

    def upsert_transformer_selection(
            self, plz: int, kcid: int, bcid: int, connection_id: int):
        """Writes the vertice_id of chosen building as Transformer location in the grid_result table"""

        query = """UPDATE grid_result
                   SET transformer_vertice_id = %(c)s
                   WHERE version_id = %(v)s
                     AND plz = %(p)s
                     AND kcid = %(k)s
                     AND bcid = %(b)s;

        UPDATE grid_result
        SET model_status = 1
        WHERE version_id = %(v)s
          AND plz = %(p)s
          AND kcid = %(k)s
          AND bcid = %(b)s;

        INSERT INTO transformer_positions (version_id, grid_result_id, geom, comment)
        VALUES(
                %(v)s,
                (SELECT grid_result_id
                 FROM grid_result
                 WHERE version_id = %(v)s \
                   AND plz = %(p)s \
                   AND kcid = %(k)s \
                   AND bcid = %(b)s),
                (SELECT the_geom FROM ways_tem_vertices_pgr WHERE id = %(c)s),
                'on_way');"""
        params = {
            "v": VERSION_ID,
            "c": connection_id,
            "b": bcid,
            "k": kcid,
            "p": plz}

        self.cur.execute(query, params)

    def get_distance_matrix_from_bcid(
            self, kcid: int, bcid: int) -> tuple[dict, np.ndarray, dict]:
        """
        Args:
            kcid: k mean cluster ID
            bcid: building cluster ID
        Returns: The distance matrix of the buildings in the building cluster as np.array and the mapping between vertice_id and local ID as dict
        """

        costmatrix_query = """SELECT *
                              FROM pgr_dijkstraCostMatrix(
                                      'SELECT way_id as id, source, target, cost, reverse_cost FROM ways_tem',
                                      (SELECT array_agg(DISTINCT b.connection_point)
                                       FROM (SELECT *
                                             FROM buildings_tem
                                             WHERE kcid = %(k)s
                                               AND bcid = %(b)s
                                             ORDER BY connection_point) AS b),
                                      false);"""
        params = {"b": bcid, "k": kcid}
        localid2vid, dist_mat, _ = self.calculate_cost_arr_dist_matrix(
            costmatrix_query, params)

        return localid2vid, dist_mat, _

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
