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

from pylovo import utils
from pylovo.config_table_structure import *
from pylovo.config_loader import *

import warnings
warnings.simplefilter(action='ignore', category=UserWarning)

class ClusteringMixin:
        def get_distance_matrix_from_building_cluster(self, kcid: int, bcid: int) -> tuple[dict, np.ndarray, dict]:
            """
            Args:
                kcid: k mean cluster ID
                bcid: building cluster ID
            Returns: Die Distanzmatrix der Gebäuden als np.array und das Mapping zwischen vertice_id und lokale ID als dict
            """
            # Creates a distance matrix from the buildings in the postcode cluster or smaller in the building cluster
    
            costmatrix_query = """SELECT * FROM pgr_dijkstraCostMatrix(
                                'SELECT way_id as id, source, target, cost, reverse_cost FROM ways_tem',
                                (SELECT array_agg(DISTINCT b.connection_point) FROM (SELECT * FROM buildings_tem 
                                    WHERE kcid = %(k)s
                                    AND bcid = %(b)s
                                    ORDER BY connection_point) AS b),
                                false);"""
            params = {"b": bcid, "k": kcid}
    
            return self._calculate_cost_arr_dist_matrix(costmatrix_query, params)
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
