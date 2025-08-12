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
                                               AND grid_level_connection = 'LV'
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
        """
        Partition a k-means component into at most `cluster_amount` electrical sub-clusters and classify them
        as valid (feasible with available transformer sizing) or invalid (too large and must be split again).

        Inputs
        - Z: Linkage matrix from hierarchical clustering (e.g., scipy.cluster.hierarchy.linkage) built on a
             condensed distance vector for the connection points of this k-means component.
        - cluster_amount: Target maximum number of clusters to cut from the hierarchy (criterion="maxclust").
        - localid2vid: Mapping from local index (0..N-1 used in the distance matrix) to graph vertex id
                       (vertice_id used in the database). Example: {0: 5346, 1: 3263, ...}.
        - buildings: DataFrame of buildings (for this kcid) used by the simultaneity-load calculations.
        - consumer_cat_df: Consumer categories table (peak loads, sim factors) used in load aggregation.
        - transformer_capacities: 1D array of available single-transformer sizes (ascending, typically in kVA).
        - double_trans: 1D array of available “double transformer” sizes (ascending), e.g., two units in parallel.

        Algorithm
        1) Cut the dendrogram into up to `cluster_amount` flat groups with fcluster(..., criterion="maxclust").
        2) For each resulting flat group (cluster_id in [1..cluster_count]):
           - Translate local indices to global vertex ids (vid_list) via `localid2vid`.
           - Compute simultaneous peak load for the buildings mapped to `vid_list`.
           - Decision logic:
             a) If total_sim_load ≥ max(transformer_capacities) and the cluster has ≥ 5 buildings → mark as
                invalid (needs further splitting): invalid_cluster_dict[cluster_id] = vid_list.
             b) If total_sim_load < max(transformer_capacities) → valid. Pick the “closest-fitting” size between
                the minimal single transformer that exceeds the load and the minimal double-transformer group
                that exceeds 1.15 × load. Store as cluster_dict[cluster_id] = (vid_list, chosen_size).
             c) Otherwise (oversized load but the cluster is tiny, < 5 buildings) → treat as valid fallback and
                assign a custom size rounded up to the next integer: cluster_dict[cluster_id] = (vid_list, ceil(load)).

        Returns
        - invalid_cluster_dict: dict[int, list[int]]
            Maps flat-group ids → list of vertex ids that form an “invalid” cluster (too large and must be split
            again at a later iteration). Example: {2: [142, 3891, 557], 3: [101, 77, 902, ...]}.
        - cluster_dict: dict[int, tuple[list[int], int]]
            Maps flat-group ids → (vid_list, transformer_size). The transformer_size is the selected rating for
            this cluster, chosen as described above. Example: {1: ([12, 45, 78], 250), 4: ([9, 10], 400)}.
        - cluster_count: int
            The number of flat groups created by the current cut (len(unique(fcluster(...)))).

        Notes
        - The keys (cluster_id) refer to the current flat cut of the dendrogram and are not stable across
          re-clustering iterations. Callers typically reindex/normalize these after aggregating results.
        - Feasible clusters are returned in `cluster_dict`; infeasible ones in `invalid_cluster_dict` for further
          recursive splitting by the caller.
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
            # Python list of vertex ids (e.g. [142, 3891, 557 ...]) that belong
            # to the current hierarchical-cluster being inspected.
            vid_list = [localid2vid[lid[0]]
                        for lid in np.argwhere(flat_groups == cluster_id)]
            total_sim_load = utils.simultaneousPeakLoad(
                buildings, consumer_cat_df, vid_list)
            # In US first step we need to check for only small transformers
            # pole mounted
            # Too large load and buildings count >5 --> invalid cluster
            if (total_sim_load >= max(transformer_capacities)
                    and len(vid_list) >= 5):  # the cluster is too big
                invalid_cluster_dict[cluster_id] = vid_list

            # Load can be served by a given transformer --> valid cluster
            elif total_sim_load < max(transformer_capacities):

                # find the smallest transformer, that satisfies the load
                optimal_transformer = transformer_capacities[transformer_capacities >
                                                             total_sim_load][0]
                optimal_double_transformer = double_trans[double_trans >
                                                          total_sim_load * 1.15][0]
                if (optimal_double_transformer -
                        total_sim_load) > (optimal_transformer - total_sim_load):
                    cluster_dict[cluster_id] = (vid_list, optimal_transformer)
                else:
                    cluster_dict[cluster_id] = (
                        vid_list, optimal_double_transformer)
            else:
                # FALLBACK: If the load can be served by a given transformer, but the number of buildings in the cluster is less than 5, then we use the smallest transformer
                # total_sim_load ≥ max(transformer_capacities) and number of
                # buildings in the cluster is less than 5. --> valid cluster
                optimal_transformer = math.ceil(total_sim_load)
                cluster_dict[cluster_id] = (vid_list, optimal_transformer)

        return invalid_cluster_dict, cluster_dict, cluster_count

    def get_kcid_length(self) -> int:
        query = """SELECT COUNT(DISTINCT kcid)
                   FROM buildings_tem
                   WHERE kcid IS NOT NULL; """
        self.cur.execute(query)
        kcid_length = self.cur.fetchone()[0]
        return kcid_length

    def get_next_unfinished_kcid_for_lv(self, regional_identifier: int) -> int:
        """
        :return: one unmodeled k mean cluster ID - regional_identifier
        """
        query = """SELECT kcid
                   FROM buildings_tem
                   WHERE kcid NOT IN (SELECT DISTINCT kcid
                                      FROM lv_grid_result
                                      WHERE version_id = %(v)s
                                        AND lv_grid_result.regional_identifier = %(regional_identifier)s)
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

    def get_buildings_from_kcid(
            self, kcid: int, grid_level_connection: str = "LV") -> pd.DataFrame:
        """
        Args:
            kcid: kmeans_cluster ID
            grid_level_connection: grid_level_connection of the buildings
        Returns: A dataframe with all building information
        """
        buildings_query = """SELECT *
                             FROM buildings_tem
                             WHERE connection_point IS NOT NULL
                               AND kcid = %(k)s
                               AND bcid ISNULL
                               AND grid_level_connection = %(g)s;"""
        params = {"k": kcid, "g": grid_level_connection}

        buildings_df = pd.read_sql_query(
            buildings_query, con=self.conn, params=params)
        buildings_df.set_index("vertice_id", drop=False, inplace=True)
        buildings_df.sort_index(inplace=True)

        self.logger.debug(
            f"Building data fetched. {len(buildings_df)} buildings from kc={kcid} ...")

        return buildings_df

    def try_clustering_new(
        self,
        Z: np.ndarray,
        localid2vid: dict,
        buildings: pd.DataFrame,
        consumer_cat_df: pd.DataFrame,
        transformer_capacities: np.ndarray,
        dist_cap_m: float,
        homes_cap: int,
        pf_planning: float = 0.90,
    ) -> tuple[dict, dict, int]:
        """
        NEW (distance-limited, capacity-constrained) clustering evaluation.

        Differences vs. old try_clustering:
        - Uses a *distance* cut on the dendrogram (criterion="distance") rather than a fixed cluster count.
        - Enforces both a max-homes constraint and a transformer loading constraint during validation.

        Inputs
        - Z: Linkage matrix built from the condensed road-graph distance vector for the current subproblem.
        - localid2vid: {local_index -> vertice_id} mapping for the current subproblem.
        - buildings: DataFrame of candidate buildings (for this kcid) used by simultaneity-load calculations.
        - consumer_cat_df: Consumer categories (used by utils.simultaneousPeakLoad).
        - transformer_capacities: ascending single-transformer sizes (kVA) allowed for the current settlement.
        - dist_cap_m: maximum allowed inter-merge distance (meters) for fcluster(..., criterion="distance").
        - homes_cap: maximum number of service points per cluster for this settlement type.
        - pf_planning: power factor used to convert kW→kVA during planning.

        Returns
        - invalid_cluster_dict: dict[int, list[int]] -> group_id -> list of vertex ids needing further split.
        - cluster_dict: dict[int, tuple[list[int], int]] -> group_id -> (vid_list, selected_transformer_kVA).
                         Here we pick the smallest single transformer >= cluster kVA (no doubles).
        - cluster_count: number of groups from the current flat cut.
        """
        # 1) Flat cut by distance
        flat_groups = fcluster(Z, t=dist_cap_m, criterion="distance")
        cluster_ids = np.unique(flat_groups)
        cluster_count = len(cluster_ids)

        cluster_dict: dict[int, tuple[list[int], int]] = {}
        invalid_cluster_dict: dict[int, list[int]] = {}

        # Pre-calc max feasible kVA
        if transformer_capacities.size == 0:
            max_kva_allowed = 0.0
        else:
            max_kva_allowed = float(transformer_capacities.max())

        for cluster_id in range(1, cluster_count + 1):
            self.logger.debug(f"Cluster ID: {cluster_id}")
            # Map local ids in this group back to global graph vertex ids
            lid_idx = np.argwhere(flat_groups == cluster_id)
            vid_list = [localid2vid[int(lid[0])] for lid in lid_idx]

            # Compute simultaneous peak load in kW
            total_sim_kw = utils.simultaneousPeakLoad(
                buildings, consumer_cat_df, vid_list)
            total_sim_kva = float(
                total_sim_kw) / pf_planning if pf_planning > 0 else float(total_sim_kw)

            homes_ok = (len(vid_list) <= homes_cap)
            kva_ok = (total_sim_kva <= max_kva_allowed)

            if homes_ok and kva_ok:
                self.logger.debug(f"Cluster vertex_ids: {vid_list}")
                self.logger.debug(f"Cluster ID: {cluster_id} is valid")
                # Choose the smallest single transformer that exceeds kVA demand
                # (so that planned loading stays within limit)
                needed_kva = total_sim_kva
                feasible = transformer_capacities[transformer_capacities >= needed_kva]
                if feasible.size == 0:
                    self.logger.debug(
                        f"Cluster ID: {cluster_id} is invalid because no single transformer large enough")
                    # No single transformer large enough -> mark invalid
                    # (caller will split further)
                    invalid_cluster_dict[cluster_id] = vid_list
                else:
                    self.logger.debug(f"Cluster ID: {cluster_id} is valid")
                    # Choose the smallest single transformer that exceeds kVA
                    # demand
                    chosen_size = int(feasible[0])
                    cluster_dict[cluster_id] = (vid_list, chosen_size)
            else:
                self.logger.debug(
                    f"Cluster ID: {cluster_id} is invalid because of not homes_ok and kva_ok. homes_ok = {homes_ok} or kva_ok = {kva_ok}")
                # If the cluster is not valid, mark it as invalid
                invalid_cluster_dict[cluster_id] = vid_list

        return invalid_cluster_dict, cluster_dict, cluster_count

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
        according to the allowed catalog for the postcode’s settlement type.

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
        transformer_capacities, _ = self.get_transformer_data(sdl)

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

        query = """SELECT equipment_data.s_max_kva, cost
                   FROM equipment_data
                   WHERE type = 'Transformer' \
                     AND application_area IN %(tuple)s
                   ORDER BY s_max_kva;"""

        self.cur.execute(query, {"tuple": application_area_tuple})
        data = self.cur.fetchall()
        capacities = [i[0] for i in data]
        transformer2cost = {i[0]: i[1] for i in data}

        self.logger.debug("Transformer data fetched.")
        return np.array(capacities), transformer2cost

    def update_building_cluster(self, transformer_id: int, conn_id_list: Union[list, tuple], count: int, kcid: int,
                                regional_identifier: int, transformer_rated_power: int) -> None:
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
            regional_identifier (int): The postcode value.
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

                INSERT INTO lv_grid_result (version_id, regional_identifier, kcid, bcid, dist_transformer_rated_power)
                VALUES (%(v)s, %(pc)s, %(k)s, %(count)s, %(t)s, %(l)s);

                INSERT INTO transformer_positions (version_id, lv_grid_result_id, grid_level, osm_id, comment, geom)
                VALUES (
                        %(v)s,
                        (SELECT lv_grid_result_id
                         FROM lv_grid_result
                         WHERE version_id = %(v)s AND regional_identifier = %(pc)s AND kcid = %(k)s AND bcid = %(count)s),
                        'LV',
                        (SELECT osm_id FROM buildings_tem WHERE vertice_id = %(t)s),
                        'Normal',
                        (SELECT center FROM buildings_tem WHERE vertice_id = %(t)s)); \
                """
        params = {"v": VERSION_ID, "count": count, "c": tuple(conn_id_list), "t": transformer_id, "k": kcid, "pc": regional_identifier,
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
            self, regional_identifier: int, kcid: int, bcid: int, connection_id: int):
        """
        Persist the user's transformer selection for a building cluster in three steps.

        Steps
        1) Update `grid_result.transformer_vertice_id` with the selected road-graph vertex (`connection_id`).
        2) Set `grid_result.model_status = 1` to mark the cluster as modeled/confirmed.
        3) Insert a row into `transformer_positions` linking the corresponding `grid_result_id` and storing the
           geometry of the selected vertex (`ways_tem_vertices_pgr.id = connection_id`) with comment "on_way".

        Args:
            regional_identifier: Postcode/area identifier of the cluster.
            kcid: K-means component identifier.
            bcid: Building-cluster identifier.
            connection_id: Selected road-graph vertex id (`ways_tem_vertices_pgr.id`) as transformer location.

        Returns:
            None
        """

        query = """UPDATE lv_grid_result
                   SET dist_transformer_vertice_id = %(c)s
                   WHERE version_id = %(v)s
                     AND regional_identifier = %(p)s
                     AND kcid = %(k)s
                     AND bcid = %(b)s;

        UPDATE lv_grid_result
        SET lv_model_status = 1
        WHERE version_id = %(v)s
          AND regional_identifier = %(p)s
          AND kcid = %(k)s
          AND bcid = %(b)s;

        INSERT INTO transformer_positions (version_id, lv_grid_result_id, grid_level, osm_id, comment, geom)
        VALUES(
                %(v)s,
                (SELECT lv_grid_result_id
                 FROM lv_grid_result
                 WHERE version_id = %(v)s \
                   AND regional_identifier = %(p)s \
                   AND kcid = %(k)s \
                   AND bcid = %(b)s),
                'LV',
                (SELECT osm_id FROM buildings_tem WHERE vertice_id = %(c)s),
                'on_way',
                (SELECT the_geom FROM ways_tem_vertices_pgr WHERE id = %(c)s));"""
        params = {
            "v": VERSION_ID,
            "c": connection_id,
            "b": bcid,
            "k": kcid,
            "p": regional_identifier}

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
