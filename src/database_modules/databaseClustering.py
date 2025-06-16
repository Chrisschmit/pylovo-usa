import math
from typing import *

import numpy as np
from scipy.cluster.hierarchy import fcluster
from scipy.cluster.hierarchy import linkage
from scipy.spatial.distance import squareform

from src import utils
from src.config_loader import *

import warnings
warnings.simplefilter(action='ignore', category=UserWarning)

class ClusteringMixin:
        def get_distance_matrix_from_bcid(self, kcid: int, bcid: int) -> tuple[dict, np.ndarray, dict]:
            """
            Args:
                kcid: k mean cluster ID
                bcid: building cluster ID
            Returns: The distance matrix of the buildings in the building cluster as np.array and the mapping between vertice_id and local ID as dict
            """
    
            costmatrix_query = """SELECT * FROM pgr_dijkstraCostMatrix(
                                'SELECT way_id as id, source, target, cost, reverse_cost FROM ways_tem',
                                (SELECT array_agg(DISTINCT b.connection_point) FROM (SELECT * FROM buildings_tem 
                                    WHERE kcid = %(k)s
                                    AND bcid = %(b)s
                                    ORDER BY connection_point) AS b),
                                false);"""
            params = {"b": bcid, "k": kcid}
    
            return self._calculate_cost_arr_dist_matrix(costmatrix_query, params)

        def get_distance_matrix_from_kcid(self, kcid: int) -> tuple[dict, np.ndarray, dict]:
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

            return self._calculate_cost_arr_dist_matrix(costmatrix_query, params)

        def _calculate_cost_arr_dist_matrix(self, costmatrix_query: str, params: dict) -> tuple[dict, np.ndarray, dict]:
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

        def upsert_building_cluster(self, plz: int, kcid: int, bcid: int, vertices: list, transformer_rated_power: int):
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
    
            params = {
                "v": VERSION_ID,
                "pc": plz,
                "bc": bcid,
                "kc": kcid,
                "vid": tuple(map(int, vertices)),
            }
            self.cur.execute(building_query, params)
    
            # Insert new clustering
            cluster_query = """INSERT INTO grid_result (version_id, plz, kcid, bcid, transformer_rated_power) 
                    VALUES(%(v)s, %(pc)s, %(kc)s, %(bc)s, %(s)s); """
    
            params = {"v": VERSION_ID, "pc": plz, "bc": bcid, "kc": kcid, "s": int(transformer_rated_power)}
            self.cur.execute(cluster_query, params)
        def clear_grid_result_in_kmean_cluster(self, plz: int, kcid: int):
            # Remove old clustering at same postcode cluster
            clear_query = """DELETE FROM grid_result
                    WHERE  version_id = %(v)s 
                    AND plz = %(pc)s
                    AND kcid = %(kc)s
                    AND bcid >= 0; """
    
            params = {"v": VERSION_ID, "pc": plz, "kc": kcid}
            self.cur.execute(clear_query, params)
            self.logger.debug(
                f"Building clusters with plz = {plz}, k_mean cluster = {kcid} area cleared."
            )
        def count_kmean_cluster_consumers(self, kcid: int) -> int:
            query = """SELECT COUNT(DISTINCT vertice_id) FROM buildings_tem WHERE kcid = %(k)s AND type != 'Transformer' AND bcid ISNULL;"""
            self.cur.execute(query, {"k": kcid})
            count = self.cur.fetchone()[0]
    
            return count
        def count_no_kmean_buildings(self):
            """
            Counts relative buildings in buildings_tem, which could not be clustered via k-means
            :return: count
            """
            query = """SELECT COUNT(*) FROM buildings_tem WHERE peak_load_in_kw != 0 AND kcid ISNULL;"""
            self.cur.execute(query)
            count = self.cur.fetchone()[0]
    
            return count
        def update_building_cluster(
                self,
                transformer_id: int,
                conn_id_list: Union[list, tuple],
                count: int,
                kcid: int,
                plz: int,
                transformer_rated_power: int
        ) -> None:
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
    
                INSERT INTO grid_result (version_id, plz, kcid, bcid, ont_vertice_id, transformer_rated_power)
                VALUES (%(v)s, %(pc)s, %(k)s, %(count)s, %(t)s, %(l)s);
    
                INSERT INTO transformer_positions (grid_result_id, geom, osm_id, comment)
                VALUES (
                    (SELECT grid_result_id FROM grid_result WHERE version_id = %(v)s AND plz = %(pc)s AND kcid = %(k)s AND bcid = %(count)s),
                    (SELECT center FROM buildings_tem WHERE vertice_id = %(t)s),
                    (SELECT osm_id FROM buildings_tem WHERE vertice_id = %(t)s),
                    'Normal'
                );
            """
            params = {
                "v": VERSION_ID,
                "count": count,
                "c": tuple(conn_id_list),
                "t": transformer_id,
                "k": kcid,
                "pc": plz,
                "l": transformer_rated_power,
            }
            self.cur.execute(query, params)
        def count_clustering_parameters(self, plz: int) -> int:
            """
            :param plz:
            :return:
            """
            query = """SELECT COUNT(cp.grid_result_id) FROM clustering_parameters cp
                       JOIN grid_result gr ON gr.grid_result_id = cp.grid_result_id
                       WHERE version_id = %(v)s
                       AND plz = %(p)s"""
            self.cur.execute(query, {"v": VERSION_ID, "p": plz})
            return int(self.cur.fetchone()[0])
        def try_clustering(
                self,
                Z: np.ndarray,
                cluster_amount: int,
                localid2vid: dict,
                buildings: pd.DataFrame,
                consumer_cat_df: pd.DataFrame,
                transformer_capacities: np.ndarray,
                double_trans: np.ndarray,
        ) -> tuple[dict, dict, int]:
            flat_groups = fcluster(Z, t=cluster_amount, criterion="maxclust")
            cluster_ids = np.unique(flat_groups)
            cluster_count = len(cluster_ids)
            # Check if simultaneous load can be satisfied with possible transformers
            cluster_dict = {}
            invalid_cluster_dict = {}
            for cluster_id in range(1, cluster_count + 1):
                vid_list = [
                    localid2vid[lid[0]] for lid in np.argwhere(flat_groups == cluster_id)
                ]
                total_sim_load = utils.simultaneousPeakLoad(
                    buildings, consumer_cat_df, vid_list
                )
                if (
                        total_sim_load >= max(transformer_capacities) and len(vid_list) >= 5
                ):  # the cluster is too big
                    invalid_cluster_dict[cluster_id] = vid_list
                elif total_sim_load < max(transformer_capacities):
                    # find the smallest transformer, that satisfies the load
                    opt_transformer = transformer_capacities[
                        transformer_capacities > total_sim_load
                        ][0]
                    opt_double_transformer = double_trans[
                        double_trans > total_sim_load * 1.15
                        ][0]
                    if (opt_double_transformer - total_sim_load) > (
                            opt_transformer - total_sim_load
                    ):
                        cluster_dict[cluster_id] = (vid_list, opt_transformer)
                    else:
                        cluster_dict[cluster_id] = (vid_list, opt_double_transformer)
                else:
                    opt_transformer = math.ceil(total_sim_load)
                    cluster_dict[cluster_id] = (vid_list, opt_transformer)
            return invalid_cluster_dict, cluster_dict, cluster_count
        def count_one_building_cluster(self) -> int:
            query = """SELECT COUNT(*) FROM grid_result gr 
                WHERE (SELECT COUNT(*) FROM buildings_tem b WHERE b.kcid = gr.kcid AND b.bcid = gr.bcid) = 1;"""
            self.cur.execute(query)
            try:
                count = self.cur.fetchone()[0]
            except:
                count = 0
    
            return count
        def update_large_kmeans_cluster(self, vertices: Union[list, tuple], cluster_count:int):
            """
            Applies k-means clustering to large components and updated values in buildings_tem
            :param vertices:
            :param cluster_count:
            :return:
            """
            query = """
                    WITH kmean AS (SELECT osm_id, ST_ClusterKMeans(center, %(ca)s)
                    OVER() AS cid FROM buildings_tem WHERE vertice_id IN %(v)s),
                    maxk AS (SELECT MAX(kcid) AS max_k FROM buildings_tem)
                UPDATE buildings_tem b SET kcid = (CASE 
                WHEN m.max_k ISNULL THEN k.cid + 1 
                ELSE m.max_k + k.cid + 1
                END)
                FROM kmean AS k, maxk AS m
                WHERE b.osm_id = k.osm_id;"""
            self.cur.execute(query, {"ca": cluster_count, "v": tuple(map(int, vertices))})
        def update_kmeans_cluster(self, vertices: list) -> None:
            """
            Groups connected components into a k-means id withouth applying clustering
            :param vertices:
            :return:
            """
            query = """
                    WITH maxk AS (SELECT MAX(kcid) AS max_k FROM buildings_tem)
                UPDATE buildings_tem SET kcid = (CASE 
                WHEN m.max_k ISNULL THEN 1 
                ELSE m.max_k + 1
                END)
                FROM maxk AS m
                WHERE vertice_id IN %(v)s;"""
            self.cur.execute(query, {"v": tuple(map(int, vertices))})
        def get_clustering_parameters_for_plz_list(self, plz_tuple: tuple) -> pd.DataFrame:
            """get clustering parameter for multiple plz"""
            query = """
                    WITH plz_table(plz) AS (
                        VALUES (%(p)s)
                    ),
                    clustering AS (
                        SELECT version_id, plz, kcid, bcid, cp.*
                        FROM clustering_parameters cp 
                        JOIN grid_result gr ON cp.grid_result_id = gr.grid_result_id
                        WHERE version_id = %(v)s
                    )
                    SELECT c.* 
                    FROM clustering c
                    FULL JOIN plz_table p
                    ON p.plz = c.plz;"""
            params = {"v": VERSION_ID, "p": plz_tuple}
            df_query = pd.read_sql_query(query, con=self.conn, params=params, )
            columns = CLUSTERING_PARAMETERS
            df_parameter = pd.DataFrame(df_query, columns=columns)
            return df_parameter

        def create_bcid_for_kcid(self, plz: int, kcid: int) -> None:
            """
            Create building clusters (bcids) with average linkage method for a given kcid.
            :param plz: Postal code
            :param kcid: K-means cluster ID
            :return: None
            """
            # Get data needed for clustering
            buildings = self.get_buildings_from_kcid(kcid)
            consumer_cat_df = self.get_consumer_categories()
            settlement_type = self.get_settlement_type_from_plz(plz)
            transformer_capacities, _ = self.get_transformer_data(settlement_type)
            double_trans = np.multiply(transformer_capacities[2:4], 2)

            # Get distance matrix and prepare for hierarchical clustering
            localid2vid, dist_mat, vid2localid = self.get_distance_matrix_from_kcid(kcid)
            dist_vector = squareform(dist_mat)

            if len(dist_vector) == 0:
                return

            # Initialize hierarchical clustering
            Z = linkage(dist_vector, method="average")
            valid_cluster_dict = {}
            invalid_trans_cluster_dict = {}
            cluster_amount = 2
            new_localid2vid = localid2vid

            # Iterative clustering process
            while True:
                # Try clustering with current parameters
                invalid_cluster_dict, cluster_dict, _ = self.try_clustering(Z, cluster_amount, new_localid2vid,
                                                                            buildings, consumer_cat_df,
                                                                            transformer_capacities, double_trans)

                # Process valid clusters
                if cluster_dict:
                    current_valid_amount = len(valid_cluster_dict)
                    valid_cluster_dict.update({x + current_valid_amount: y for x, y in cluster_dict.items()})
                    valid_cluster_dict = dict(
                        enumerate(valid_cluster_dict.values()))  # reindexing the dict with enumerate

                # Process invalid clusters
                if invalid_cluster_dict:
                    current_invalid_amount = len(invalid_trans_cluster_dict)
                    invalid_trans_cluster_dict.update(
                        {x + current_invalid_amount: y for x, y in invalid_cluster_dict.items()})
                    invalid_trans_cluster_dict = dict(enumerate(invalid_trans_cluster_dict.values()))

                # Check if clustering is complete
                if not invalid_trans_cluster_dict:
                    self.logger.info(
                        f"Found {len(valid_cluster_dict)} single transformer clusters for PLZ: {plz}, KCID: {kcid}")
                    break
                else:
                    # Process too large clusters by re-clustering them
                    self.logger.info(
                        f"Found {len(invalid_trans_cluster_dict)} too_large clusters for PLZ: {plz}, KCID: {kcid}")

                    # Get buildings from the first too-large cluster for re-clustering
                    invalid_vertice_ids = list(invalid_trans_cluster_dict[0])
                    invalid_local_ids = [vid2localid[v] for v in invalid_vertice_ids]

                    # Create new mappings and distance matrix for the subclustering
                    new_localid2vid = {k: v for k, v in localid2vid.items() if k in invalid_local_ids}
                    new_localid2vid = dict(enumerate(new_localid2vid.values()))
                    new_dist_mat = dist_mat[invalid_local_ids][:, invalid_local_ids]
                    new_dist_vector = squareform(new_dist_mat)

                    # Prepare for next iteration
                    Z = linkage(new_dist_vector, method="average")
                    cluster_amount = 2
                    del invalid_trans_cluster_dict[0]
                    invalid_trans_cluster_dict = dict(enumerate(invalid_trans_cluster_dict.values()))

            # At this point, we've successfully found a valid electrical clustering solution with the minimum
            # number of clusters. Each cluster:
            #   1. Contains buildings that can be served by a single transformer
            #   2. Has an appropriately sized transformer assigned
            # The valid_cluster_dict maps building cluster IDs to tuples of (building_vertices_list, optimal_transformer_size)
            # We could calculate the total transformer cost by summing the costs of all selected transformers:
            # total_transformer_cost = sum([transformer2cost[v[1]] for v in valid_cluster_dict.values()])

            # Save results to database
            self.clear_grid_result_in_kmean_cluster(plz, kcid)
            for bcid, cluster_data in valid_cluster_dict.items():
                self.upsert_building_cluster(plz, kcid, bcid, vertices=cluster_data[0],
                                             transformer_rated_power=cluster_data[1])
            self.logger.debug(f"bcids for plz {plz} kcid {kcid} found...")

        def connect_unconnected_ways(self) -> None:
                """
                Updates ways_tem
                :return:
                """
                query = """SELECT draw_way_connections();"""
                self.cur.execute(query)

        def assign_close_buildings(self) -> None:
            """
            * Set peak load to zero, if a building is too near or touching to a too large customer?
            :return:
            """
            while True:
                remove_query = """WITH close (un) AS (SELECT ST_Union(geom)
                                                      FROM buildings_tem
                                                      WHERE peak_load_in_kw = 0)
                                  UPDATE buildings_tem b
                                  SET peak_load_in_kw = 0
                                  FROM close AS c
                                  WHERE ST_Touches(b.geom, c.un)
                                    AND b.type IN ('Commercial', 'Public', 'Industrial')
                                    AND b.peak_load_in_kw != 0;"""
                self.cur.execute(remove_query)

                count_query = """WITH close (un) AS (SELECT ST_Union(geom)
                                                     FROM buildings_tem
                                                     WHERE peak_load_in_kw = 0)
                                 SELECT COUNT(*)
                                 FROM buildings_tem AS b,
                                      close AS c
                                 WHERE ST_Touches(b.geom, c.un)
                                   AND b.type IN ('Commercial', 'Public', 'Industrial')
                                   AND b.peak_load_in_kw != 0;"""
                self.cur.execute(count_query)
                count = self.cur.fetchone()[0]
                if count == 0 or count is None:
                    break

            return None