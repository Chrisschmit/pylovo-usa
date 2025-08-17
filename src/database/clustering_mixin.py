"""
Enhanced Clustering Mixin that merges functionality from clustering_mixin.py and infrastructure_placement_mixin.py.

This unified mixin provides hierarchical clustering capabilities for both LV and MV infrastructure
placement, reusing existing clustering logic while extending it to support both grid levels.
"""

import math
import time
import warnings
from abc import ABC
from decimal import *
from typing import *

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import squareform

from src import utils
from src.config_loader import *
from src.database.base_mixin import BaseMixin
from src.equipment_data_schema import (InfrastructureCluster, LVLoadAggregator,
                                       MVLoadAggregator, TransformerEquipment)

warnings.simplefilter(action='ignore', category=UserWarning)


class ClusteringMixin(BaseMixin, ABC):
    """
    Enhanced clustering mixin that unifies LV and MV infrastructure placement.

    This mixin merges the existing clustering functionality with the new infrastructure
    placement engine, providing a unified interface for both LV transformer placement
    and MV substation placement.
    """

    def __init__(self):
        super().__init__()

    # ==== UNIFIED INFRASTRUCTURE PLACEMENT INTERFACE ====

    def perform_infrastructure_placement(
        self,
        kcid: int,
        regional_identifier: int,
        grid_level: str,
        settlement_type: int,
    ) -> List[InfrastructureCluster]:
        """
        Main entry point for infrastructure placement using traditional clustering.

        This method implements capacity-only clustering to create infrastructure clusters
        using hierarchical clustering with maxclust criterion.

        Args:
            kcid: K-means cluster ID to process
            regional_identifier: Regional identifier for the area
            grid_level: "LV" or "MV" grid level
            settlement_type: Settlement type (1=rural, 2=suburban, 3=urban)
        Returns:
            List of infrastructure clusters with equipment and positioning
        """
        try:
            self.logger.info(
                f"Starting {grid_level} infrastructure placement for "
                f"kcid {kcid}, regional_identifier {regional_identifier}"
            )

            # Load equipment catalog
            equipment_catalog = self._load_equipment_catalog(
                settlement_type, grid_level
            )

            if not equipment_catalog:
                self.logger.warning(
                    f"No equipment available for {grid_level} placement"
                )
                return []

            # Load building and consumer data
            buildings_df = self._get_buildings(
                kcid, regional_identifier, grid_level)
            consumer_df = self._get_consumer_categories()

            # Get distance matrix and create vertex mapping using unified
            # approach
            localid2vid, dist_mat, _ = self.get_distance_matrix(
                kcid, regional_identifier, grid_level)

            if dist_mat.size == 0 or len(localid2vid) <= 1:
                self.logger.warning(
                    f"Insufficient data for clustering in kcid {kcid}"
                )
                return []

            # Create initial linkage matrix
            dist_vector = squareform(dist_mat)
            Z = linkage(dist_vector, method="average")

            # Use traditional clustering approach (similar to old
            # try_clustering method)
            infrastructure_clusters = self._clustering(
                Z=Z,
                localid2vid=localid2vid,
                buildings_df=buildings_df,
                grid_level=grid_level,
                consumer_df=consumer_df,
                equipment_catalog=equipment_catalog,
                kcid=kcid,
                dist_mat=dist_mat
            )

            self.logger.info(
                f"Successfully completed {grid_level} infrastructure placement "
                f"for kcid {kcid}: {
                    len(infrastructure_clusters)} clusters created"
            )

            return infrastructure_clusters

        except Exception as e:
            self.logger.error(
                f"Error in {grid_level} infrastructure placement for "
                f"kcid {kcid}: {str(e)}"
            )
            raise

    def _clustering(
        self,
        Z: np.ndarray,
        localid2vid: dict,
        buildings_df: pd.DataFrame,
        grid_level: str,
        consumer_df: pd.DataFrame,
        equipment_catalog: List[TransformerEquipment],
        kcid: int,
        dist_mat: np.ndarray
    ) -> List[InfrastructureCluster]:
        """
        Clustering approach based on capacity limits only.
        Uses maxclust criterion and iterative approach.
        """
        # Convert equipment catalog to old format for compatibility
        equipment_capacities = np.array(
            [eq.s_max_kva for eq in equipment_catalog])
        equipment_capacities = np.sort(equipment_capacities)

        # Create double transformer options (similar to old approach)
        if len(equipment_capacities) >= 4:
            double_trans = np.multiply(equipment_capacities[2:4], 2)
        else:
            double_trans = np.array([])

        valid_cluster_dict = {}
        invalid_cluster_dict = {}
        # Explore if we create multipel clusters first
        cluster_amount = 2  # Start with 2 clusters
        new_localid2vid = localid2vid

        while True:
            # Try clustering with current cluster amount
            invalid_cluster_dict, cluster_dict, _ = self._try_clustering(
                Z=Z,
                cluster_amount=cluster_amount,
                localid2vid=new_localid2vid,
                buildings=buildings_df,
                grid_level=grid_level,
                consumer_cat_df=consumer_df,
                transformer_capacities=equipment_capacities,
                double_trans=double_trans
            )

            # Process valid clusters
            if cluster_dict:
                current_valid_amount = len(valid_cluster_dict)
                valid_cluster_dict.update(
                    {x + current_valid_amount: y for x, y in cluster_dict.items()})
                # reindexing the dict with enumerate
                valid_cluster_dict = dict(
                    enumerate(valid_cluster_dict.values()))

            # Process invalid clusters
            if invalid_cluster_dict:
                current_invalid_amount = len(invalid_cluster_dict)
                invalid_cluster_dict_temp = {}
                invalid_cluster_dict_temp.update(
                    {x + current_invalid_amount: y for x, y in invalid_cluster_dict.items()})
                invalid_cluster_dict = dict(
                    enumerate(invalid_cluster_dict_temp.values()))

            # Check if clustering is complete
            if not invalid_cluster_dict:
                self.logger.info(
                    f"Found {len(valid_cluster_dict)} single transformer clusters for kcid: {kcid}")
                break
            else:
                # Process first invalid cluster by increasing cluster amount
                self.logger.info(
                    f"Found {len(invalid_cluster_dict)} too_large clusters for kcid: {kcid}")

                # Get first invalid cluster for re-clustering
                invalid_vertice_ids = list(invalid_cluster_dict[0])
                vid2localid = {v: k for k, v in localid2vid.items()}
                invalid_local_ids = [vid2localid[v]
                                     for v in invalid_vertice_ids if v in vid2localid]

                # Create new mappings and distance matrix for the subclustering
                new_localid2vid = {
                    k: v for k, v in localid2vid.items() if k in invalid_local_ids}
                new_localid2vid = dict(enumerate(new_localid2vid.values()))
                new_dist_mat = dist_mat[np.ix_(
                    invalid_local_ids, invalid_local_ids)]
                new_dist_vector = squareform(new_dist_mat)

                # Prepare for next iteration
                Z = linkage(new_dist_vector, method="average")
                cluster_amount = 2
                del invalid_cluster_dict[0]
                invalid_cluster_dict = dict(
                    enumerate(invalid_cluster_dict.values()))

        # Convert valid clusters to InfrastructureCluster objects
        infrastructure_clusters = []
        for cluster_id, (cluster_nodes,
                         transformer_size) in valid_cluster_dict.items():
            # Find equipment that matches the selected transformer size
            selected_equipment = None
            for eq in equipment_catalog:
                if eq.s_max_kva == transformer_size:
                    selected_equipment = eq
                    break

            # If no exact match, find closest larger equipment
            if selected_equipment is None:
                suitable_equipment = [
                    eq for eq in equipment_catalog if eq.s_max_kva >= transformer_size]
                if suitable_equipment:
                    selected_equipment = min(
                        suitable_equipment, key=lambda eq: eq.s_max_kva)
                else:
                    # Fallback to largest available equipment
                    selected_equipment = max(
                        equipment_catalog, key=lambda eq: eq.s_max_kva)

            infrastructure_cluster = self._create_infrastructure_cluster(
                cluster_id=cluster_id,
                cluster_nodes=cluster_nodes,
                selected_equipment=selected_equipment,
                buildings_df=buildings_df,
                grid_level=grid_level,
                consumer_df=consumer_df,
                localid2vid=localid2vid,
                dist_mat=dist_mat,
                kcid=kcid
            )
            infrastructure_clusters.append(infrastructure_cluster)

        return infrastructure_clusters

    def _try_clustering(
        self,
        Z: np.ndarray,
        cluster_amount: int,
        localid2vid: dict,
        buildings: pd.DataFrame,
        grid_level: str,
        consumer_cat_df: pd.DataFrame,
        transformer_capacities: np.ndarray,
        double_trans: np.ndarray
    ) -> tuple[dict, dict, int]:
        """
        Traditional clustering method copied from old try_clustering.
        Uses maxclust criterion instead of distance criterion.
        """
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
            # Python list of vertex ids that belong to the current
            # hierarchical-cluster
            vid_list = [localid2vid[lid[0]]
                        for lid in np.argwhere(flat_groups == cluster_id)]
            total_sim_load = utils.simultaneousPeakLoad(
                buildings, consumer_cat_df, vid_list)

            # Too large load and buildings count >5 --> invalid cluster
            if (total_sim_load >= max(transformer_capacities)
                    and len(vid_list) >= 5):
                invalid_cluster_dict[cluster_id] = vid_list

            # Load can be served by a given transformer --> valid cluster
            elif total_sim_load < max(transformer_capacities):
                # find the smallest transformer that satisfies the load
                optimal_transformer = transformer_capacities[transformer_capacities >
                                                             total_sim_load][0]

                if len(double_trans) > 0:
                    # Check if any double transformer can handle the load
                    suitable_double_trans = double_trans[double_trans >
                                                         total_sim_load * 1.15]
                    if len(suitable_double_trans) > 0:
                        optimal_double_transformer = suitable_double_trans[0]
                        if (optimal_double_transformer -
                                total_sim_load) > (optimal_transformer - total_sim_load):
                            cluster_dict[cluster_id] = (
                                vid_list, optimal_transformer)
                        else:
                            cluster_dict[cluster_id] = (
                                vid_list, optimal_double_transformer)
                    else:
                        cluster_dict[cluster_id] = (
                            vid_list, optimal_transformer)
                else:
                    cluster_dict[cluster_id] = (vid_list, optimal_transformer)
            else:
                # FALLBACK: If the load can be served by a given transformer,
                # but the number of buildings in the cluster is less than 5
                optimal_transformer = math.ceil(total_sim_load)
                cluster_dict[cluster_id] = (vid_list, optimal_transformer)

        return invalid_cluster_dict, cluster_dict, cluster_count

    def _create_infrastructure_cluster(
        self,
        cluster_id: int,
        cluster_nodes: List[int],
        selected_equipment: TransformerEquipment,
        buildings_df: pd.DataFrame,
        grid_level: str,
        consumer_df: pd.DataFrame,
        localid2vid: dict,
        dist_mat: np.ndarray,
        kcid: int
    ) -> InfrastructureCluster:
        """Create an infrastructure cluster result object."""
        # Find optimal position for equipment
        optimal_vertex = self.find_optimal_position(
            cluster_nodes=cluster_nodes,
            buildings_df=buildings_df,
            distance_matrix=dist_mat,
            localid2vid=localid2vid
        )

        if grid_level == "LV":
            load_aggregator = LVLoadAggregator()
        else:
            load_aggregator = MVLoadAggregator(self)

        aggregate_load = load_aggregator.calculate_aggregate_load(
            cluster_nodes, buildings_df, consumer_df,
            kcid=kcid, power_factor=POWER_FACTOR
        )

        return InfrastructureCluster(
            cluster_id=cluster_id,
            node_vertices=cluster_nodes,
            equipment=selected_equipment,
            optimal_vertex=optimal_vertex,
            aggregate_load=aggregate_load,
            total_cost=selected_equipment.cost
        )

    # ==== GRID LEVEL DISPATCHING ====

    def _get_candidate_nodes(
        self,
        kcid: int,
        regional_identifier: int,
        grid_level: str,
    ) -> List[int]:
        """
        Get candidate nodes for infrastructure placement based on grid level.

        For LV: Returns building vertices that need LV transformer connection
        For MV: Returns mix of LV transformer vertices and MV building vertices
        """
        if grid_level == "LV":
            return self._get_lv_candidate_nodes(
                kcid, regional_identifier)
        elif grid_level == "MV":
            return self._get_mv_candidate_nodes(
                kcid, regional_identifier)
        else:
            raise ValueError(f"Unsupported grid level: {grid_level}")

    def _get_lv_candidate_nodes(
        self,
        kcid: int,
        regional_identifier: int,
    ) -> List[int]:
        """Get building vertices that need LV transformer connection."""
        query = """
        SELECT DISTINCT connection_point
        FROM buildings_tem bt
        WHERE bt.kcid = %s
          AND bt.regional_identifier = %s
          AND bt.grid_level_connection = 'LV'
          AND bt.connection_point IS NOT NULL
        ORDER BY connection_point
        """

        self.cur.execute(query, (kcid, regional_identifier))
        results = self.cur.fetchall()

        return [row[0] for row in results]

    def _get_mv_candidate_nodes(
        self,
        kcid: int,
        regional_identifier: int,
    ) -> List[int]:
        """
        Get candidate nodes for MV substation placement.
        Returns mix of LV transformer vertices and MV building vertices.
        """
        # Get LV transformer positions from previous LV placement
        lv_transformer_query = """
        SELECT DISTINCT dist_transformer_vertice_id as connection_point
        FROM lv_grid_result
        WHERE kcid = %s
          AND regional_identifier = %s
          AND version_id = %s
          AND dist_transformer_vertice_id IS NOT NULL
        """

        # Get MV buildings (high load buildings that connect directly to MV)
        mv_building_query = """
        SELECT DISTINCT connection_point
        FROM buildings_tem bt
        WHERE bt.kcid = %s
          AND bt.regional_identifier = %s
          AND bt.grid_level_connection = 'MV'
          AND bt.connection_point IS NOT NULL
          ORDER BY connection_point
        """

        candidate_nodes = []

        # Add LV transformer vertices
        self.cur.execute(
            lv_transformer_query,
            (kcid,
             regional_identifier,
             VERSION_ID))
        lv_transformers = self.cur.fetchall()
        candidate_nodes.extend([row[0] for row in lv_transformers])

        # Add MV building vertices
        self.cur.execute(mv_building_query, (kcid, regional_identifier))
        mv_buildings = self.cur.fetchall()
        candidate_nodes.extend([row[0] for row in mv_buildings])

        return list(set(candidate_nodes))  # Remove duplicates

    # ==== EQUIPMENT LOADING METHODS ====

    def _load_equipment_catalog(
        self,
        settlement_type: int,
        grid_level: str
    ) -> List[TransformerEquipment]:
        """Load transformer/substation equipment from database."""
        voltage_level_map = {
            "LV": "MV-LV",
            "MV": "HV-MV"
        }

        voltage_level = voltage_level_map.get(grid_level, "MV-LV")

        if settlement_type == 1:
            application_area_tuple = (1, 2, 3)
        elif settlement_type == 2:
            application_area_tuple = (2, 3, 4)
        elif settlement_type == 3:
            application_area_tuple = (3, 4, 5)

        query = """
        SELECT *
        FROM equipment_data
        WHERE type IN ('Transformer', 'Substation')
          AND application_area IN %s
          AND voltage_level = %s
          AND s_max_kva IS NOT NULL
        ORDER BY s_max_kva ASC
        """

        self.cur.execute(query, (application_area_tuple, voltage_level))
        results = self.cur.fetchall()

        # Convert to list of dictionaries
        columns = [desc[0] for desc in self.cur.description]
        equipment_data = [dict(zip(columns, row)) for row in results]

        return [TransformerEquipment.from_database_row(
            row) for row in equipment_data]

    def _get_buildings(
        self,
        kcid: int,
        regional_identifier: int,
        grid_level: str = None
    ) -> pd.DataFrame:
        """Load buildings data for the specified cluster."""
        base_query = """
        SELECT *
        FROM buildings_tem
        WHERE kcid = %s AND regional_identifier = %s
        """

        params = [kcid, regional_identifier]

        # Add grid level filter if specified
        if grid_level:
            base_query += " AND grid_level_connection = %s"
            params.append(grid_level)

        return pd.read_sql_query(
            base_query,
            self.sqla_engine,
            # Convert list to tuple for pandas compatibility
            params=tuple(params)
        )

    def _get_consumer_categories(self) -> pd.DataFrame:
        """Load consumer categories from database."""
        query = """
        SELECT consumer_category_id, definition, peak_load, yearly_consumption,
               peak_load_per_m2, yearly_consumption_per_m2, sim_factor
        FROM consumer_categories
        """

        return pd.read_sql_query(query, self.sqla_engine)

    def find_optimal_position(
        self,
        cluster_nodes: List[int],
        buildings_df: pd.DataFrame,
        distance_matrix: np.ndarray,
        localid2vid: Dict[int, int]
    ) -> int:
        """
        Find optimal position for infrastructure using weighted distance minimization.

        Args:
            cluster_nodes: List of vertex IDs in the cluster
            buildings_df: Buildings dataframe for load weights
            distance_matrix: Distance matrix between nodes
            localid2vid: Mapping from local indices to vertex IDs

        Returns:
            Optimal vertex ID for infrastructure placement
        """

        if len(cluster_nodes) == 1:
            return cluster_nodes[0]

        self.logger.debug(
            f"Lenght of cluster_nodes: {len(cluster_nodes)}")

        # Create reverse mapping
        vid2localid = {v: k for k, v in localid2vid.items()}

        # Get local indices for cluster nodes
        cluster_local_ids = [vid2localid[vid]
                             for vid in cluster_nodes if vid in vid2localid]

        if not cluster_local_ids:
            return cluster_nodes[0]

        # Get load weights for each node
        load_weights = []
        for local_id in cluster_local_ids:
            # This is a connection_point, NOT a vertice_id!
            connection_point = localid2vid[local_id]
            # Match by connection_point, not vertice_id
            building = buildings_df[buildings_df['connection_point']
                                    == connection_point]
            if not building.empty:
                # Sum loads if multiple buildings share same connection_point
                total_load = building['peak_load_in_kw'].sum()
                load_weights.append(total_load)
            else:
                load_weights.append(1.0)  # Default weight

        load_weights = np.array(load_weights)

        # Extract sub-distance matrix for cluster
        cluster_dist_mat = distance_matrix[np.ix_(
            cluster_local_ids, cluster_local_ids)]

        # Calculate weighted distances
        weighted_distances = cluster_dist_mat.dot(load_weights)

        # Select node with minimum weighted distance
        min_local_idx = np.argmin(weighted_distances)
        optimal_local_id = cluster_local_ids[min_local_idx]

        return localid2vid[optimal_local_id]

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

    def get_distance_matrix(
            self, kcid: int, regional_identifier: int, grid_level: str) -> tuple[dict, np.ndarray, dict]:
        """
        Unified distance matrix generation that uses _get_candidate_nodes() for both LV and MV.

        Args:
            kcid: k-means cluster id
            regional_identifier: Regional identifier
            grid_level: Grid level ("LV" or "MV")
        Returns:
            - localid2vid: dict mapping local indices to vertex IDs
            - dist_mat: distance matrix as np.array
            - vid2localid: dict mapping vertex IDs to local indices
        """
        # Get candidate connection_points nodes using the efunction
        candidate_nodes = self._get_candidate_nodes(
            kcid, regional_identifier, grid_level)

        if not candidate_nodes:
            self.logger.warning(
                f"No candidate nodes found for {grid_level} in kcid {kcid}")
            return {}, np.array([]), {}

        # Generate distance matrix for these specific nodes
        costmatrix_query = """SELECT *
                              FROM pgr_dijkstraCostMatrix(
                                      'SELECT way_id as id, source, target, cost, reverse_cost FROM ways_tem',
                                      %(nodes)s,
                                      false);"""
        params = {"nodes": candidate_nodes}

        localid2vid, dist_mat, vid2localid = self.calculate_cost_arr_dist_matrix(
            costmatrix_query, params)

        return localid2vid, dist_mat, vid2localid

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

    def get_kcid_length(self) -> int:
        query = """SELECT COUNT(DISTINCT kcid)
                   FROM buildings_tem
                   WHERE kcid IS NOT NULL; """
        self.cur.execute(query)
        kcid_length = self.cur.fetchone()[0]
        return kcid_length

    def get_next_unfinished_kcid(
            self, regional_identifier: int, target_table: str = "grid_result") -> int:
        """
        :return: one unmodeled k mean cluster ID - regional_identifier
        """
        query = f"""SELECT kcid
                   FROM buildings_tem
                   WHERE kcid NOT IN (SELECT DISTINCT kcid
                                      FROM {target_table}
                                      WHERE version_id = %(v)s
                                        AND {target_table}.regional_identifier = %(regional_identifier)s)
                     AND kcid IS NOT NULL
                   ORDER BY kcid
                   LIMIT 1;"""
        self.cur.execute(
            query, {
                "v": VERSION_ID, "regional_identifier": regional_identifier})
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

    def clear_lv_grid_result_in_kmean_cluster(
            self, regional_identifier: int, kcid: int):
        # Remove old clustering at same postcode cluster
        clear_query = """DELETE
                         FROM lv_grid_result
                         WHERE version_id = %(v)s
                           AND regional_identifier = %(pc)s
                           AND kcid = %(kc)s
                           AND bcid >= 0; """

        params = {"v": VERSION_ID, "pc": regional_identifier, "kc": kcid}
        self.cur.execute(clear_query, params)
        self.logger.debug(
            f"Building clusters with regional_identifier = {regional_identifier}, k_mean cluster = {kcid} area cleared.")

    def upsert_bcid(self, regional_identifier: int, kcid: int, bcid: int,
                    vertices: list, transformer_rated_power: int):
        """
        Assign buildings in buildings_tem the bcid and stores the cluster in lv_grid_result
        Args:
            regional_identifier: postcode cluster ID - regional_identifier
            kcid: kmeans cluster ID
            bcid: building cluster ID
            vertices: List of vertice_id of selected buildings
            transformer_rated_power: Apparent power of the selected transformer
        """
        # Insert references to building elements in which cluster they are.
        building_query = """UPDATE buildings_tem
                            SET bcid = %(bc)s
                            WHERE regional_identifier = %(pc)s
                              AND kcid = %(kc)s
                              AND bcid ISNULL
                              AND connection_point IN %(vid)s
                              AND type != 'Transformer'; """

        params = {"v": VERSION_ID, "pc": regional_identifier, "bc": bcid,
                  "kc": kcid, "vid": tuple(map(int, vertices)), }
        self.cur.execute(building_query, params)

        # Insert new clustering
        cluster_query = """INSERT INTO lv_grid_result (version_id, regional_identifier, kcid, bcid, dist_transformer_rated_power)
                           VALUES (%(v)s, %(pc)s, %(kc)s, %(bc)s, %(s)s); """

        params = {"v": VERSION_ID, "pc": regional_identifier, "bc": bcid,
                  "kc": kcid, "s": int(transformer_rated_power)}
        self.cur.execute(cluster_query, params)

    def count_kmean_cluster_consumers(self, kcid: int) -> int:
        query = """SELECT COUNT(DISTINCT vertice_id)
                   FROM buildings_tem
                   WHERE kcid = %(k)s
                     AND type != 'Transformer'
                     AND bcid ISNULL;"""
        self.cur.execute(query, {"k": kcid})
        count = self.cur.fetchone()[0]

        return count

    def delete_isolated_building(self, regional_identifier: int, kcid):
        query = """DELETE
                   FROM buildings_tem
                   WHERE regional_identifier = %(p)s
                     AND kcid = %(k)s
                     AND bcid ISNULL;"""
        self.cur.execute(query, {"p": regional_identifier, "k": kcid})

    def get_greenfield_bcids(
            self, regional_identifier: int, kcid: int) -> list:
        """
        Args:
            regional_identifier: loadarea cluster ID
            kcid: kmeans cluster ID
        Returns: A list of greenfield building clusters for a given regional_identifier
        """
        query = """SELECT DISTINCT bcid
                   FROM lv_grid_result
                   WHERE version_id = %(v)s
                     AND kcid = %(kc)s
                     AND regional_identifier = %(pc)s
                     AND lv_model_status ISNULL
                   ORDER BY bcid; """
        params = {"v": VERSION_ID, "pc": regional_identifier, "kc": kcid}
        self.cur.execute(query, params)
        bcid_list = [t[0] for t in data] if (
            data := self.cur.fetchall()) else []
        return bcid_list

    def get_buildings_from_bcid(
            self, regional_identifier: int, kcid: int, bcid: int) -> pd.DataFrame:

        buildings_query = """SELECT *
                             FROM buildings_tem
                             WHERE type != 'Transformer'
                               AND regional_identifier = %(p)s
                               AND bcid = %(b)s
                               AND kcid = %(k)s;"""
        params = {"p": regional_identifier, "b": bcid, "k": kcid}

        buildings_df = pd.read_sql_query(
            buildings_query, con=self.conn, params=params)
        buildings_df.set_index("vertice_id", drop=False, inplace=True)
        buildings_df.sort_index(inplace=True)
        # dropping duplicate indices
        # buildings_df = buildings_df[~buildings_df.index.duplicated(keep='first')]

        self.logger.debug(f"{len(buildings_df)} building data fetched.")

        return buildings_df

    def update_dist_transformer_rated_power(
            self, regional_identifier: int, kcid: int, bcid: int, note: int):
        """
        Update the transformer_rated_power (kVA) for a specific building cluster in lv_grid_result
        according to the allowed catalog for the postcode's settlement type.

        Behavior
        - Determines the settlement type for the provided regional_identifier and loads the allowed
          single-transformer sizes (ascending) via get_transformer_data.
        - If note == 0:
            Bump the currently stored transformer_rated_power to the next larger single size
            from the catalog (no double-transformer options considered).
        - If note != 0:
            Consider both the single sizes and the double-transformer options (parallel units).
            If the currently stored value already matches any allowed size, do nothing.
            Otherwise normalize it by rounding up to the nearest 630 multiple and update.

        Parameters
        - regional_identifier: Postcode/area identifier of the cluster.
        - kcid: K-means component identifier the cluster belongs to.
        - bcid: Building cluster identifier to update.
        - note: Mode flag controlling the update strategy.
                0  -> Only single-transformer catalog bump to next larger size.
                !=0 -> Include double-transformer combinations and normalize fallback sizes.

        Returns
        - None. Updates are written directly to lv_grid_result.

        Side effects
        - Updates lv_grid_result.dist_transformer_rated_power for the (version_id, regional_identifier, kcid, bcid) row.
        - Emits a debug log when a double/multiple transformer group assignment occurs.
        """
        sdl = self.get_settlement_type_from_regional_identifier(
            regional_identifier)
        transformer_capacities, _ = self.get_transformer_data(
            sdl, grid_level="LV")

        if note == 0:
            old_query = """SELECT dist_transformer_rated_power
                           FROM lv_grid_result
                           WHERE version_id = %(v)s
                             AND regional_identifier = %(p)s
                             AND kcid = %(k)s
                             AND bcid = %(b)s;"""
            self.cur.execute(
                old_query, {
                    "v": VERSION_ID, "p": regional_identifier, "k": kcid, "b": bcid})
            transformer_rated_power = self.cur.fetchone()[0]

            new_transformer_rated_power = transformer_capacities[transformer_capacities > transformer_rated_power][
                0].item()
            update_query = """UPDATE lv_grid_result
                              SET dist_transformer_rated_power = %(n)s
                              WHERE version_id = %(v)s
                                AND regional_identifier = %(p)s
                                AND kcid = %(k)s
                                AND bcid = %(b)s;"""
            self.cur.execute(update_query,
                             {"v": VERSION_ID, "p": regional_identifier, "k": kcid, "b": bcid, "n": new_transformer_rated_power}, )
        else:
            double_trans = np.multiply(transformer_capacities[2:4], 2)
            combined = np.concatenate(
                (transformer_capacities, double_trans), axis=None)
            np.sort(combined, axis=None)
            old_query = """SELECT dist_transformer_rated_power
                           FROM lv_grid_result
                           WHERE version_id = %(v)s
                             AND regional_identifier = %(p)s
                             AND kcid = %(k)s
                             AND bcid = %(b)s;"""
            self.cur.execute(
                old_query, {
                    "v": VERSION_ID, "p": regional_identifier, "k": kcid, "b": bcid})
            transformer_rated_power = self.cur.fetchone()[0]
            if transformer_rated_power in combined.tolist():
                return None
            new_transformer_rated_power = np.ceil(
                transformer_rated_power / 630) * 630
            update_query = """UPDATE lv_grid_result
                              SET dist_transformer_rated_power = %(n)s
                              WHERE version_id = %(v)s
                                AND regional_identifier = %(p)s
                                AND kcid = %(k)s
                                AND bcid = %(b)s;"""
            self.cur.execute(update_query,
                             {"v": VERSION_ID, "p": regional_identifier, "k": kcid, "b": bcid, "n": new_transformer_rated_power}, )
            self.logger.debug(
                "double or multiple transformer group transformer_rated_power assigned")

    def get_transformer_data(
            self, settlement_type: int = None, grid_level: str = "LV") -> tuple[np.array, dict]:
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

        if grid_level == 'LV':
            transformer_type = 'Transformer'
        elif grid_level == 'MV':
            transformer_type = 'Substation'

        query = """SELECT equipment_data.s_max_kva, cost
                   FROM equipment_data
                   WHERE type = %(transformer_type)s \
                     AND application_area IN %(tuple)s
                   ORDER BY s_max_kva;"""

        self.cur.execute(query,
                         {"tuple": application_area_tuple,
                          "transformer_type": transformer_type})
        data = self.cur.fetchall()
        capacities = [i[0] for i in data]
        transformer2cost = {i[0]: i[1] for i in data}

        self.logger.debug("Transformer data fetched.")
        return np.array(capacities), transformer2cost

    def get_settlement_type_from_regional_identifier(
            self, regional_identifier) -> int:
        """
        Args:
            regional_identifier:
        Returns: Settlement type: 1=City, 2=Village, 3=Rural
        """
        settlement_query = """SELECT settlement_type
                              FROM postcode_result
                              WHERE postcode_result_regional_identifier = %(p)s
                              LIMIT 1; """
        self.cur.execute(settlement_query, {"p": regional_identifier})
        settlement_type = self.cur.fetchone()[0]

        return settlement_type

    def get_lv_transformer_power_at_vertex(
        self,
        vertex_id: int,
        kcid: int
    ) -> Optional[float]:
        """
        Get LV transformer power rating at specified vertex.
        Used by MVLoadAggregator to calculate aggregate loads.
        """
        query = """
        SELECT lgr.dist_transformer_rated_power
        FROM lv_grid_result lgr
        WHERE lgr.dist_transformer_vertice_id = %s
          AND lgr.kcid = %s
          AND lgr.version_id = %s
        LIMIT 1
        """

        self.cur.execute(query, (int(vertex_id), int(kcid), VERSION_ID))
        result = self.cur.fetchone()

        return float(result[0]) if result and result[0] else None

    def _update_buildings_bcid(
        self,
        regional_identifier: int,
        kcid: int,
        bcid: int,
        node_vertices: List[int]
    ) -> None:
        """
        Update bcid in buildings_tem for all buildings in this cluster.

        Args:
            regional_identifier: Regional identifier
            kcid: K-means cluster ID
            bcid: Building cluster ID (from InfrastructureCluster.cluster_id)
            node_vertices: List of connection_point vertices in this cluster
        """
        # Update buildings_tem with bcid for all buildings in this cluster
        update_query = """
        UPDATE buildings_tem
        SET bcid = %s
        WHERE regional_identifier = %s
          AND kcid = %s
          AND bcid IS NULL
          AND connection_point IN %s
          AND type != 'Transformer'
        """

        self.cur.execute(update_query, (
            bcid,
            regional_identifier,
            kcid,
            tuple(node_vertices)
        ))

        self.logger.debug(
            f"Updated bcid={bcid} for {
                self.cur.rowcount} buildings in kcid={kcid}"
        )

    def save_infrastructure_placement_results(
        self,
        infrastructure_clusters: List[InfrastructureCluster],
        kcid: int,
        regional_identifier: int,
        grid_level: str,
    ) -> List[int]:
        """
        Save infrastructure placement results to database.

        Args:
            infrastructure_clusters: List of infrastructure placement results
            kcid: K-means cluster ID
            regional_identifier: Regional identifier
            grid_level: "LV" or "MV"
            parent_grid_result_id: Parent grid result ID for LV grids

        Returns:
            List of created grid result IDs
        """
        created_ids = []

        try:

            for cluster in infrastructure_clusters:
                if grid_level == "LV":

                    # Insert into lv_grid_result
                    insert_query = """
                    INSERT INTO lv_grid_result (
                        version_id, regional_identifier, kcid, bcid,
                        parent_grid_result_id, dist_transformer_rated_power,
                        dist_transformer_vertice_id, lv_model_status
                    ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                    RETURNING lv_grid_result_id
                    """

                    self.cur.execute(insert_query, (
                        VERSION_ID,
                        regional_identifier,
                        kcid,
                        int(cluster.cluster_id),
                        None,
                        int(cluster.equipment.s_max_kva),
                        int(cluster.optimal_vertex),
                        0  # Initial model status
                    ))

                    self._update_buildings_bcid(
                        regional_identifier=regional_identifier,
                        kcid=kcid,
                        bcid=int(cluster.cluster_id),
                        node_vertices=list(int(v)
                                           for v in cluster.node_vertices)
                    )

                elif grid_level == "MV":
                    # Insert into grid_result
                    insert_query = """
                    INSERT INTO grid_result (
                        version_id, regional_identifier, kcid, scid,
                        substation_rated_power, substation_vertice_id,
                        model_status, grid
                    ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                    RETURNING grid_result_id
                    """

                    self.cur.execute(insert_query, (
                        VERSION_ID,
                        regional_identifier,
                        kcid,
                        int(cluster.cluster_id),
                        int(cluster.equipment.s_max_kva),
                        int(cluster.optimal_vertex),
                        0,  # Initial model status
                        None  # Grid JSON will be populated later
                    ))

                    result_id = self.cur.fetchone()[0]
                    created_ids.append(result_id)

                self.logger.debug(
                    f"Created {grid_level} infrastructure cluster {
                        cluster.cluster_id} "
                    f"with {len(cluster.node_vertices)} nodes, "
                    f"equipment {cluster.equipment.name}, "
                    f"power {cluster.equipment.s_max_kva} kVA"
                )

            self.logger.info(
                f"Saved {
                    len(created_ids)} {grid_level} infrastructure clusters "
                f"for kcid {kcid}"
            )

            return created_ids

        except Exception as e:
            self.conn.rollback()
            self.logger.error(
                f"Error saving {grid_level} infrastructure placement results: {
                    str(e)}"
            )
            raise

    def create_transformer_positions(
        self,
        infrastructure_clusters: List[InfrastructureCluster],
        grid_level: str,
        grid_result_ids: List[int]
    ) -> None:
        """
        Create transformer position records in database.

        Args:
            infrastructure_clusters: Infrastructure placement results
            grid_level: "LV" or "MV"
            grid_result_ids: Corresponding grid result IDs
        """
        try:
            for cluster, grid_result_id in zip(
                    infrastructure_clusters, grid_result_ids):
                # Get geometry for the optimal vertex (which is a
                # connection_point on the road network)
                geom_query = """
                SELECT ST_Transform(the_geom, %s) as geom
                FROM ways_tem_vertices_pgr
                WHERE id = %s
                """

                self.cur.execute(
                    geom_query, (EPSG, int(
                        cluster.optimal_vertex)))
                geom_result = self.cur.fetchone()

                if not geom_result:
                    self.logger.warning(
                        f"No geometry found for vertex {
                            cluster.optimal_vertex}"
                    )
                    continue

                # Insert transformer position
                insert_query = """
                INSERT INTO transformer_positions (
                    version_id, grid_result_id, lv_grid_result_id, grid_level,
                    comment, geom
                ) VALUES (%s, %s, %s, %s, %s, %s)
                """

                lv_grid_result_id = int(
                    grid_result_id) if grid_level == "LV" else None
                mv_grid_result_id = int(
                    grid_result_id) if grid_level == "MV" else None

                comment = (
                    f"{grid_level} {cluster.equipment.type} "
                    f"{cluster.equipment.s_max_kva}kVA"
                )

                self.cur.execute(insert_query, (
                    VERSION_ID,
                    mv_grid_result_id,
                    lv_grid_result_id,
                    grid_level,
                    comment,
                    geom_result[0]
                ))

            # If this is MV level, update existing LV transformer positions
            # to reference the MV grid_result_id
            if grid_level == "MV":
                self._update_lv_transformer_positions_with_mv_reference(
                    infrastructure_clusters, grid_result_ids)

            self.logger.info(
                f"Created {len(infrastructure_clusters)} {grid_level} "
                f"transformer positions"
            )

        except Exception as e:
            self.conn.rollback()
            self.logger.error(
                f"Error creating {grid_level} transformer positions: {str(e)}"
            )
            raise

    def _update_lv_transformer_positions_with_mv_reference(
        self,
        infrastructure_clusters: List[InfrastructureCluster],
        grid_result_ids: List[int]
    ) -> None:
        """
        Update existing LV transformer positions to reference their parent MV grid_result_id.

        Args:
            infrastructure_clusters: MV infrastructure clusters
            grid_result_ids: Corresponding MV grid_result_ids
        """
        try:
            for cluster, mv_grid_result_id in zip(
                    infrastructure_clusters, grid_result_ids):
                # Find LV transformer vertices in this MV cluster
                lv_transformer_vertices = []

                for vertex in cluster.node_vertices:
                    # Check if this vertex is an LV transformer by looking for
                    # corresponding lv_grid_result
                    lv_check_query = """
                    SELECT lv_grid_result_id
                    FROM lv_grid_result
                    WHERE dist_transformer_vertice_id = %s
                      AND version_id = %s
                    """
                    self.cur.execute(lv_check_query, (int(vertex), VERSION_ID))
                    lv_result = self.cur.fetchone()

                    if lv_result:
                        lv_transformer_vertices.append(int(vertex))

                # Update transformer_positions for these LV transformers
                if lv_transformer_vertices:
                    # Update transformer_positions by matching lv_grid_result_id
                    # This is more reliable than geometry matching
                    update_query = """
                    UPDATE transformer_positions tp
                    SET grid_result_id = %s
                    FROM lv_grid_result lgr
                    WHERE tp.lv_grid_result_id = lgr.lv_grid_result_id
                      AND tp.version_id = %s
                      AND tp.grid_level = 'LV'
                      AND lgr.dist_transformer_vertice_id IN %s
                      AND lgr.version_id = %s
                    """
                    self.cur.execute(update_query, (
                        int(mv_grid_result_id),
                        VERSION_ID,
                        tuple(lv_transformer_vertices),
                        VERSION_ID
                    ))

                    self.logger.debug(
                        f"Updated {
                            len(lv_transformer_vertices)} LV transformer positions "
                        f"with MV grid_result_id {mv_grid_result_id}"
                    )

        except Exception as e:
            self.logger.error(
                f"Error updating LV transformer positions with MV references: {
                    str(e)}"
            )
            raise

    def update_lv_mv_links(
        self,
        infrastructure_clusters: List[InfrastructureCluster],
        grid_result_ids: List[int],
        kcid: int,
        regional_identifier: int
    ) -> None:
        """
        Update LV transformers and buildings with scid and parent grid_result_id references.

        This method implements the final linking step for MV substation placement:
        1. Updates lv_grid_result records to reference parent grid_result_id
        2. Assigns scid to LV transformers based on which MV cluster they belong to
        3. Propagates scid to all buildings (LV and MV) in the cluster
        """
        try:
            for cluster, grid_result_id in zip(
                    infrastructure_clusters, grid_result_ids):
                scid = cluster.cluster_id

                # Get LV transformer vertices and MV building vertices in this
                # cluster
                lv_transformer_vertices = []
                mv_building_vertices = []

                for vertex in cluster.node_vertices:
                    # Check if this vertex is an LV transformer
                    lv_check_query = """
                    SELECT lv_grid_result_id
                    FROM lv_grid_result
                    WHERE dist_transformer_vertice_id = %s
                      AND kcid = %s
                      AND regional_identifier = %s
                      AND version_id = %s
                    """
                    self.cur.execute(
                        lv_check_query, (int(vertex), kcid, regional_identifier, VERSION_ID))
                    lv_result = self.cur.fetchone()

                    if lv_result:
                        lv_transformer_vertices.append(int(vertex))
                    else:
                        mv_building_vertices.append(int(vertex))

                # Update LV transformers with parent reference and scid
                if lv_transformer_vertices:
                    update_lv_query = """
                    UPDATE lv_grid_result
                    SET parent_grid_result_id = %s,
                        scid = %s
                    WHERE dist_transformer_vertice_id IN %s
                      AND kcid = %s
                      AND regional_identifier = %s
                      AND version_id = %s
                    """
                    self.cur.execute(update_lv_query, (
                        int(grid_result_id), int(scid), tuple(
                            lv_transformer_vertices),
                        kcid, regional_identifier, VERSION_ID
                    ))

                # Propagate scid to all buildings connected to these LV
                # transformers
                if lv_transformer_vertices:
                    # Get building cluster IDs (bcids) for these LV
                    # transformers
                    bcid_query = """
                    SELECT DISTINCT bcid
                    FROM lv_grid_result
                    WHERE dist_transformer_vertice_id IN %s
                      AND kcid = %s
                      AND regional_identifier = %s
                      AND version_id = %s
                    """
                    self.cur.execute(bcid_query, (
                        tuple(
                            lv_transformer_vertices), kcid, regional_identifier, VERSION_ID
                    ))
                    bcids = [row[0] for row in self.cur.fetchall()]

                    if bcids:
                        update_lv_buildings_query = """
                        UPDATE buildings_tem
                        SET scid = %s
                        WHERE kcid = %s
                          AND regional_identifier = %s
                          AND bcid IN %s
                        """
                        self.cur.execute(update_lv_buildings_query, (
                            int(scid), kcid, regional_identifier, tuple(bcids)
                        ))

                # Propagate scid to MV buildings directly connected to this
                # substation
                if mv_building_vertices:
                    update_mv_buildings_query = """
                    UPDATE buildings_tem
                    SET scid = %s
                    WHERE connection_point IN %s
                      AND kcid = %s
                      AND regional_identifier = %s
                      AND grid_level_connection = 'MV'
                    """
                    self.cur.execute(update_mv_buildings_query, (
                        int(scid), tuple(
                            mv_building_vertices), kcid, regional_identifier
                    ))

                self.logger.debug(
                    f"Updated scid {scid} for {
                        len(lv_transformer_vertices)} LV transformers "
                    f"and {
                        len(mv_building_vertices)} MV buildings in cluster {
                        cluster.cluster_id}"
                )

            self.conn.commit()
            self.logger.info(
                f"Successfully updated LV-MV links for kcid {kcid}")

        except Exception as e:
            self.conn.rollback()
            self.logger.error(
                f"Error updating LV-MV links for kcid {kcid}: {str(e)}")
            raise

    def finalize_mv_substation_placement(
            self, grid_result_ids: List[int]) -> None:
        """
        Set model_status = 1 for completed MV substation placements.
        """
        try:
            update_query = """
            UPDATE grid_result
            SET model_status = 1
            WHERE grid_result_id IN %s
              AND version_id = %s
            """
            self.cur.execute(
                update_query, (tuple(grid_result_ids), VERSION_ID))

            self.logger.info(
                f"Finalized {
                    len(grid_result_ids)} MV substation placements")

        except Exception as e:
            self.conn.rollback()
            self.logger.error(
                f"Error finalizing MV substation placements: {
                    str(e)}")
            raise


# Legacy placement methods:

    # TODO: Remove this method

    # def get_consumer_to_transformer_df(
    #         self, kcid: int, transformer_list: list) -> pd.DataFrame:
    #     consumer_query = """SELECT DISTINCT connection_point
    #                             FROM buildings_tem
    #                             WHERE kcid = %(k)s
    #                             AND type != 'Transformer';"""
    #     self.cur.execute(consumer_query, {"k": kcid})
    #     consumer_list = [t[0] for t in self.cur.fetchall()]

    #     cost_query = """SELECT *
    #                         FROM pgr_dijkstraCost(
    #                                 'SELECT way_id as id, source, target, cost, reverse_cost FROM ways_tem',
    #                                 %(cl)s, %(tl)s,
    #                                 false);"""
    #     cost_df = pd.read_sql_query(cost_query, con=self.conn, params={"cl": consumer_list, "tl": transformer_list},
    # dtype={"start_vid": np.int32, "end_vid": np.int32, "agg_cost":
    # np.int32}, )

    #     return cost_df

    # def calculate_sim_load(self, conn_list: Union[tuple, list]) -> Decimal:
    #     residential = """WITH residential AS
    #                               (SELECT b.peak_load_in_kw AS load, b.houses_per_building AS count, c.sim_factor
    #                                FROM buildings_tem AS b
    #                                         LEFT JOIN consumer_categories AS c
    #                                                   ON b.type = c.definition
    #                                WHERE b.connection_point IN %(c)s
    #                                  AND b.type IN ('SFH', 'MFH', 'AB', 'TH'))
    #                      SELECT SUM(load), SUM(count), sim_factor
    #                      FROM residential
    #                      GROUP BY sim_factor; \
    #                   """
    #     self.cur.execute(residential, {"c": tuple(conn_list)})

    #     data = self.cur.fetchone()
    #     if data:
    #         residential_load = Decimal(data[0])
    #         residential_count = Decimal(data[1])
    #         residential_factor = Decimal(data[2])
    #         residential_sim_load = residential_load * (
    #             residential_factor + (1 - residential_factor) * (residential_count ** Decimal(-3 / 4)))
    #     else:
    #         residential_sim_load = 0
    #     # TODO can the following 4 repetitions simplified with a general
    #     # function?
    #     commercial = """WITH commercial AS
    #                              (SELECT b.peak_load_in_kw AS load, b.houses_per_building AS count, c.sim_factor
    #                               FROM buildings_tem AS b
    #                                        LEFT JOIN consumer_categories AS c
    #                                                  ON c.definition = b.type
    #                               WHERE b.connection_point IN %(c)s
    #                                 AND b.type = 'Commercial')
    #                     SELECT SUM(load), SUM(count), sim_factor
    #                     FROM commercial
    #                     GROUP BY sim_factor; \
    #                  """
    #     self.cur.execute(commercial, {"c": tuple(conn_list)})
    #     data = self.cur.fetchone()
    #     if data:
    #         commercial_load = Decimal(data[0])
    #         commercial_count = Decimal(data[1])
    #         commercial_factor = Decimal(data[2])
    #         commercial_sim_load = commercial_load * (
    #             commercial_factor + (1 - commercial_factor) * (commercial_count ** Decimal(-3 / 4)))
    #     else:
    #         commercial_sim_load = 0

    #     public = """WITH public AS
    #                          (SELECT b.peak_load_in_kw AS load, b.houses_per_building AS count, c.sim_factor
    #                           FROM buildings_tem AS b
    #                                    LEFT JOIN consumer_categories AS c
    #                                              ON c.definition = b.type
    #                           WHERE b.connection_point IN %(c)s
    #                             AND b.type = 'Public')
    #                 SELECT SUM(load), SUM(count), sim_factor
    #                 FROM public
    #                 GROUP BY sim_factor; \
    #              """
    #     self.cur.execute(public, {"c": tuple(conn_list)})
    #     data = self.cur.fetchone()
    #     if data:
    #         public_load = Decimal(data[0])
    #         public_count = Decimal(data[1])
    #         public_factor = Decimal(data[2])
    #         public_sim_load = public_load * \
    #             (public_factor + (1 - public_factor)
    #              * (public_count ** Decimal(-3 / 4)))
    #     else:
    #         public_sim_load = 0

    #     industrial = """WITH industrial AS
    #                              (SELECT b.peak_load_in_kw AS load, b.houses_per_building AS count, c.sim_factor
    #                               FROM buildings_tem AS b
    #                                        LEFT JOIN consumer_categories AS c
    #                                                  ON c.definition = b.type
    #                               WHERE b.connection_point IN %(c)s
    #                                 AND b.type = 'Industrial')
    #                     SELECT SUM(load), SUM(count), sim_factor
    #                     FROM industrial
    #                     GROUP BY sim_factor; \
    #                  """
    #     self.cur.execute(industrial, {"c": tuple(conn_list)})
    #     data = self.cur.fetchone()
    #     if data:
    #         industrial_load = Decimal(data[0])
    #         industrial_count = Decimal(data[1])
    #         industrial_factor = Decimal(data[2])
    #         industrial_sim_load = industrial_load * (
    #             industrial_factor + (1 - industrial_factor) * (industrial_count ** Decimal(-3 / 4)))
    #     else:
    #         industrial_sim_load = 0

    #     total_sim_load = (
    #         residential_sim_load +
    #         commercial_sim_load +
    #         industrial_sim_load +
    #         public_sim_load)

    #     return total_sim_load

    # def update_building_cluster(self, transformer_id: int, conn_id_list: Union[list, tuple], count: int, kcid: int,
    #                             regional_identifier: int, transformer_rated_power: int) -> None:
    #     """
    #     Update building cluster information by performing multiple operations:
    #       - Update the 'bcid' in 'buildings_tem' where 'vertice_id' matches the transformer_id.
    #       - Update the 'bcid' in 'buildings_tem' for rows where 'connection_point' is in the provided list and type is not 'Transformer'.
    #       - Insert a new record into 'grid_result'.
    #       - Insert a new record into 'transformer_positions' using subqueries for geometry and OGC ID.
    #     Args:
    #         transformer_id (int): The ID of the transformer.
    #         conn_id_list (Union[list, tuple]): A list or tuple of connection point IDs.
    #         count (int): The new building cluster identifier.
    #         kcid (int): The KCID value.
    #         regional_identifier (int): The postcode value.
    #         transformer_rated_power (int): The selected transformer size for the building cluster.
    #     """
    #     query = """
    #             UPDATE buildings_tem
    #             SET bcid = %(count)s
    #             WHERE vertice_id = %(t)s;

    #             UPDATE buildings_tem
    #             SET bcid = %(count)s
    #             WHERE connection_point IN %(c)s
    #               AND type != 'Transformer';

    #             INSERT INTO lv_grid_result (version_id, regional_identifier, kcid, bcid, dist_transformer_rated_power)
    #             VALUES (%(v)s, %(pc)s, %(k)s, %(count)s, %(t)s, %(l)s);

    #             INSERT INTO transformer_positions (version_id, lv_grid_result_id, grid_level, osm_id, comment, geom)
    #             VALUES (
    #                     %(v)s,
    #                     (SELECT lv_grid_result_id
    #                      FROM lv_grid_result
    #                      WHERE version_id = %(v)s AND regional_identifier = %(pc)s AND kcid = %(k)s AND bcid = %(count)s),
    #                     'LV',
    #                     (SELECT osm_id FROM buildings_tem WHERE vertice_id = %(t)s),
    #                     'Normal',
    #                     (SELECT center FROM buildings_tem WHERE vertice_id = %(t)s)); \
    #             """
    #     params = {"v": VERSION_ID, "count": count, "c": tuple(conn_id_list), "t": transformer_id, "k": kcid, "pc": regional_identifier,
    #               "l": transformer_rated_power, }
    #     self.cur.execute(query, params)

    # def get_building_connection_points_from_bc(
    #         self, kcid: int, bcid: int) -> list:
    #     """
    #     Args:
    #         kcid: kmeans_cluster ID
    #         bcid: building cluster ID
    #     Returns: A dataframe with all building information
    #     """
    #     count_query = """SELECT DISTINCT connection_point
    #                      FROM buildings_tem
    #                      WHERE vertice_id IS NOT NULL
    #                        AND bcid = %(b)s
    #                        AND kcid = %(k)s;"""
    #     params = {"b": bcid, "k": kcid}
    #     self.cur.execute(count_query, params)
    #     try:
    #         cp = [t[0] for t in self.cur.fetchall()]
    #     except BaseException:
    #         cp = []

    #     return cp

    # def get_distance_matrix_from_bcid(
    #         self, kcid: int, bcid: int) -> tuple[dict, np.ndarray, dict]:
    #     """
    #     Args:
    #         kcid: k mean cluster ID
    #         bcid: building cluster ID
    #     Returns: The distance matrix of the buildings in the building cluster as np.array and the mapping between vertice_id and local ID as dict
    #     """

    #     costmatrix_query = """SELECT *
    #                           FROM pgr_dijkstraCostMatrix(
    #                                   'SELECT way_id as id, source, target, cost, reverse_cost FROM ways_tem',
    #                                   (SELECT array_agg(DISTINCT b.connection_point)
    #                                    FROM (SELECT *
    #                                          FROM buildings_tem
    #                                          WHERE kcid = %(k)s
    #                                            AND bcid = %(b)s
    #                                          ORDER BY connection_point) AS b),
    #                                   false);"""
    #     params = {"b": bcid, "k": kcid}
    #     localid2vid, dist_mat, _ = self.calculate_cost_arr_dist_matrix(
    #         costmatrix_query, params)

    #     return localid2vid, dist_mat, _

    # def generate_load_vector(self, kcid: int, bcid: int) -> np.ndarray:
    #     query = """SELECT SUM(peak_load_in_kw)::float
    #                FROM buildings_tem
    #                WHERE kcid = %(k)s
    #                  AND bcid = %(b)s
    #                GROUP BY connection_point
    #                ORDER BY connection_point;"""
    #     self.cur.execute(query, {"k": kcid, "b": bcid})
    #     load = np.asarray([i[0] for i in self.cur.fetchall()])

    #     return load

    # def upsert_transformer_selection(
    #         self, regional_identifier: int, kcid: int, bcid: int, connection_id: int):
    #     """
    # Persist the user's transformer selection for a building cluster in three
    # steps.

    #     Steps
    #     1) Update `grid_result.transformer_vertice_id` with the selected road-graph vertex (`connection_id`).
    #     2) Set `grid_result.model_status = 1` to mark the cluster as modeled/confirmed.
    #     3) Insert a row into `transformer_positions` linking the corresponding `grid_result_id` and storing the
    # geometry of the selected vertex (`ways_tem_vertices_pgr.id =
    # connection_id`) with comment "on_way".

    #     Args:
    #         regional_identifier: Postcode/area identifier of the cluster.
    #         kcid: K-means component identifier.
    #         bcid: Building-cluster identifier.
    # connection_id: Selected road-graph vertex id
    # (`ways_tem_vertices_pgr.id`) as transformer location.

    #     Returns:
    #         None
    #     """

    #     query = """UPDATE lv_grid_result
    #                SET dist_transformer_vertice_id = %(c)s
    #                WHERE version_id = %(v)s
    #                  AND regional_identifier = %(p)s
    #                  AND kcid = %(k)s
    #                  AND bcid = %(b)s;

    #     UPDATE lv_grid_result
    #     SET lv_model_status = 1
    #     WHERE version_id = %(v)s
    #       AND regional_identifier = %(p)s
    #       AND kcid = %(k)s
    #       AND bcid = %(b)s;

    #     INSERT INTO transformer_positions (version_id, lv_grid_result_id, grid_level, osm_id, comment, geom)
    #     VALUES(
    #             %(v)s,
    #             (SELECT lv_grid_result_id
    #              FROM lv_grid_result
    #              WHERE version_id = %(v)s \
    #                AND regional_identifier = %(p)s \
    #                AND kcid = %(k)s \
    #                AND bcid = %(b)s),
    #             'LV',
    #             (SELECT osm_id FROM buildings_tem WHERE vertice_id = %(c)s),
    #             'on_way',
    #             (SELECT the_geom FROM ways_tem_vertices_pgr WHERE id = %(c)s));"""
    #     params = {
    #         "v": VERSION_ID,
    #         "c": connection_id,
    #         "b": bcid,
    #         "k": kcid,
    #         "p": regional_identifier}

    #     self.cur.execute(query, params)
