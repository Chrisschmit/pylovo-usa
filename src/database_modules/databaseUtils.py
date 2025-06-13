import time

import psycopg2 as psy
import numpy as np
from scipy.cluster.hierarchy import linkage
from scipy.spatial.distance import squareform
from sqlalchemy import create_engine

from src import utils
from config.config_table_structure import *
from src.config_loader import *

import warnings
warnings.simplefilter(action='ignore', category=UserWarning)

class UtilsMixin:
        def __init__(
                self, dbname=DBNAME, user=USER, pw=PASSWORD, host=HOST, port=PORT, **kwargs
            ):
            self.logger = utils.create_logger(
                "PgReaderWriter", log_file=kwargs.get("log_file", "log.txt"), log_level=LOG_LEVEL
            )
            try:
                self.conn = psy.connect(
                    database=dbname, user=user, password=pw, host=host, port=port, options=f"-c search_path={TARGET_SCHEMA},public"
                )
                self.cur = self.conn.cursor()
                self.db_path = f"postgresql+psycopg2://{user}:{pw}@{host}:{port}/{dbname}"
                self.sqla_engine = create_engine(
                    self.db_path,
                    connect_args={"options": f"-c search_path={TARGET_SCHEMA},public"}
                    )
            except psy.OperationalError as err:
                self.logger.warning(
                    f"Connecting to {dbname} was not successful. Make sure, that you have established the SSH "
                    f"connection with correct port mapping."
                )
                raise err
    
    
            self.logger.debug(f"PgReaderWriter is constructed and connected to {self.db_path}.")
        def __del__(self):
            self.cur.close()
            self.conn.close()
        def zero_too_large_consumers(self) -> int:
            """
            Sets the load to zero if the peak load is too large (> 100)
            :return: number of the large customers
            """
            query = """
                UPDATE buildings_tem SET peak_load_in_kw = 0 
                WHERE peak_load_in_kw > 100 AND type IN ('Commercial', 'Public');
                SELECT COUNT(*) FROM buildings_tem WHERE peak_load_in_kw = 0;"""
            self.cur.execute(query)
            too_large = self.cur.fetchone()[0]
    
            return too_large
        def connect_unconnected_ways(self) -> None:
            """
            Updates ways_tem
            :return:
            """
            query = """SELECT draw_way_connections();"""
            self.cur.execute(query)
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
                        invalid_cluster_dict, cluster_dict, _ = self.try_clustering(
                            Z, cluster_amount, new_localid2vid, buildings,
                            consumer_cat_df, transformer_capacities, double_trans
                        )
    
                        # Process valid clusters
                        if cluster_dict:
                            current_valid_amount = len(valid_cluster_dict)
                            valid_cluster_dict.update({x + current_valid_amount: y for x, y in cluster_dict.items()})
                            valid_cluster_dict = dict(enumerate(valid_cluster_dict.values())) # reindexing the dict with enumerate
    
                        # Process invalid clusters
                        if invalid_cluster_dict:
                            current_invalid_amount = len(invalid_trans_cluster_dict)
                            invalid_trans_cluster_dict.update({x + current_invalid_amount: y for x, y in invalid_cluster_dict.items()})
                            invalid_trans_cluster_dict = dict(enumerate(invalid_trans_cluster_dict.values()))
    
                        # Check if clustering is complete
                        if not invalid_trans_cluster_dict:
                            self.logger.info(f"Found {len(valid_cluster_dict)} single transformer clusters for PLZ: {plz}, KCID: {kcid}")
                            break
                        else:
                            # Process too large clusters by re-clustering them
                            self.logger.info(f"Found {len(invalid_trans_cluster_dict)} too_large clusters for PLZ: {plz}, KCID: {kcid}")
    
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
                        self.upsert_building_cluster(
                            plz, kcid, bcid,
                            vertices=cluster_data[0],
                            transformer_rated_power=cluster_data[1]
                        )
                    self.logger.debug(f"bcids for plz {plz} kcid {kcid} found...")
        def assign_close_buildings(self) -> None:
            """
            * Set peak load to zero, if a building is too near or touching to a too large customer?
            :return:
            """
            while True:
                remove_query = """WITH close (un) AS (
                        SELECT ST_Union(geom) FROM buildings_tem WHERE peak_load_in_kw = 0)
                        UPDATE buildings_tem b SET peak_load_in_kw = 0 FROM close AS c WHERE ST_Touches(b.geom, c.un) 
                            AND b.type IN ('Commercial', 'Public', 'Industrial')
                            AND b.peak_load_in_kw != 0;"""
                self.cur.execute(remove_query)
    
                count_query = """WITH close (un) AS (
                        SELECT ST_Union(geom) FROM buildings_tem WHERE peak_load_in_kw = 0)
                        SELECT COUNT(*) FROM buildings_tem AS b, close AS c WHERE ST_Touches(b.geom, c.un) 
                            AND b.type IN ('Commercial', 'Public', 'Industrial')
                            AND b.peak_load_in_kw != 0;"""
                self.cur.execute(count_query)
                count = self.cur.fetchone()[0]
                if count == 0 or count is None:
                    break
    
            return None
        def remove_duplicate_buildings(self):
            """
            * Remove buildings without geometry or osm_id
            * Remove buildings which are duplicates of other buildings and have a copied id
            :return:
            """
            remove_query = """DELETE FROM buildings_tem WHERE geom ISNULL;"""
            self.cur.execute(remove_query)
    
            remove_noid_building = """DELETE FROM buildings_tem WHERE osm_id ISNULL;"""
            self.cur.execute(remove_noid_building)
    
            query = """DELETE FROM buildings_tem WHERE geom IN 
                        (SELECT geom FROM buildings_tem GROUP BY geom HAVING count(*) > 1) 
                        AND osm_id LIKE '%copy%';"""
            self.cur.execute(query)
        def _calculate_cost_arr_dist_matrix(self, costmatrix_query: str, params: dict) -> tuple[dict, np.ndarray, dict]:
            """
            Helper function for calculating cost array and distance matrix from given parameters
            """
            st = time.time()
            cost_df = pd.read_sql_query(
                costmatrix_query,
                con=self.conn,
                params=params,
                dtype={"start_vid": np.int32, "end_vid": np.int32, "agg_cost": np.int32},
            )
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
        def create_temp_tables(self) -> None:
            for query in TEMP_CREATE_QUERIES.values():
                self.cur.execute(query)
        def drop_temp_tables(self) -> None:
            for table_name in TEMP_CREATE_QUERIES.keys():
                self.cur.execute(f"DROP TABLE IF EXISTS {table_name}")
            self.cur.execute("DROP TABLE IF EXISTS ways_tem_vertices_pgr")
        def commit_changes(self):
            self.conn.commit()
