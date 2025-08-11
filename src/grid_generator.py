import traceback
import warnings
from pathlib import Path

import numpy as np
import pandapower as pp
import pandas as pd
from scipy.cluster.hierarchy import linkage
from scipy.spatial.distance import squareform

import src.database.database_client as dbc
from src import utils
from src.config_loader import *
from src.parameter_calculator import ParameterCalculator


class ResultExistsError(Exception):
    "Raised when the regional_identifier has already been created."


class GridGenerator:
    """
    Generates the grid for the given regional_identifier area
    """

    def __init__(self, regional_identifier=999999, **kwargs):
        self.regional_identifier = regional_identifier
        self.dbc = dbc.DatabaseClient()
        self.dbc.insert_version_if_not_exists()
        self.dbc.insert_parameter_tables(
            consumer_categories=CONSUMER_CATEGORIES)
        self.logger = utils.create_logger(
            name="GridGenerator",
            log_level=LOG_LEVEL,
            log_file=LOG_FILE
        )

    def __del__(self):
        self.dbc.__del__()

    def generate_grid_for_single_regional_identifier(
            self, regional_identifier: str, analyze_grids: bool = False) -> None:
        """
        Generates the grid for a single regional_identifier.

        :param regional_identifier: Postal code for which the grid should be generated.
        :type regional_identifier: str
        :param analyze_grids: Option to analyze the results after grid generation, defaults to False.
        :type analyze_grids: bool
        """
        self.regional_identifier = regional_identifier
        print(
            '-------------------- start',
            self.regional_identifier,
            '---------------------------')

        self.dbc.create_temp_tables()  # create temp tables for the grid generation

        try:
            self.generate_grid()
            # Save data from temporary tables to result tables
            self.dbc.save_tables(regional_identifier=self.regional_identifier)
            if analyze_grids:
                pc = ParameterCalculator()
                pc.calc_parameters_per_regional_identifier(
                    regional_identifier=self.regional_identifier)
        except ResultExistsError:
            self.dbc.logger.info(
                f"Grid for the postcode area {regional_identifier} has already been generated.")
        except Exception as e:
            self.logger.error(
                f"Error during grid generation for regional_identifier {self.regional_identifier}: {e}")
            self.logger.info(
                f"Skipped regional_identifier {self.regional_identifier} due to generation error.")
            self.dbc.conn.rollback()  # rollback the transaction
            traceback.print_exc()
            return

        self.dbc.drop_temp_tables()  # drop temp tables
        self.dbc.commit_changes()  # commit the changes to the database

        print(
            '-------------------- end',
            self.regional_identifier,
            '-----------------------------')

    def generate_grid_for_multiple_regional_identifier(
            self, df_regional_identifier: pd.DataFrame, analyze_grids: bool = False) -> None:
        """generates grid for all regional_identifier contained in the column 'regional_identifier' of df_samples

        :param df_regional_identifier: table that contains regional_identifier for grid generation
        :type df_regional_identifier: pd.DataFrame
        :param analyze_grids: option to analyse the results after grid generation, defaults to False
        :type analyze_grids: bool
        """
        self.dbc.create_temp_tables()  # create temp tables for the grid generation

        for _, row in df_regional_identifier.iterrows():
            self.regional_identifier = str(row['regional_identifier'])
            print(
                '-------------------- start',
                self.regional_identifier,
                '---------------------------')
            try:
                self.generate_grid()
                # Save data from temporary tables to result tables
                self.dbc.save_tables(
                    regional_identifier=self.regional_identifier)
                self.dbc.reset_tables()  # Reset temporary tables
                if analyze_grids:
                    pc = ParameterCalculator()
                    pc.calc_parameters_per_regional_identifier(
                        regional_identifier=self.regional_identifier)
            except ResultExistsError:
                self.dbc.logger.info(
                    f"Grid for the postcode area {self.regional_identifier} has already been generated.")
            except Exception as e:
                self.logger.error(
                    f"Error during grid generation for regional_identifier {self.regional_identifier}: {e}")
                self.logger.info(
                    f"Skipped regional_identifier {self.regional_identifier} due to generation error.")
                self.dbc.conn.rollback()  # rollback the transaction
                continue
            print(
                '-------------------- end',
                self.regional_identifier,
                '-----------------------------')

        self.dbc.drop_temp_tables()  # drop temp tables
        self.dbc.commit_changes()  # commit the changes to the database

    def generate_grid(self):
        if self.dbc.is_grid_generated(self.regional_identifier):
            raise ResultExistsError(
                f"The grids for the postcode area {
                    self.regional_identifier} is already generated "
                f"for the version {VERSION_ID}."
            )
        self.prepare_postcodes()
        self.prepare_buildings()
        self.prepare_transformers()
        self.prepare_ways()

        self.apply_kmeans_clustering()
        # We are going to generate MV clusters for every kmeans cluster, currently each kmeans clsuter contains around ~1000 buildings.
        # Prep buildings for the clustering: Above >100kw will be directly
        # connected to MV, below will be clustered.
        self.position_all_transformers()
        self.install_cables_parallel(max_workers=4)

    def prepare_postcodes(self):
        """
        Caches postcode from raw data tables and stores in temporary tables.
        FROM: postcode
        INTO: postcode_result
        """
        self.dbc.copy_postcode_result_table(self.regional_identifier)
        self.logger.info(
            f"Working on regional_identifier {
                self.regional_identifier}")

    def prepare_buildings(self):
        """
        Caches buildings from raw data tables and stores in temporary tables.
        FROM: res, oth
        INTO: buildings_tem
        """
        self.dbc.set_residential_buildings_table(self.regional_identifier)
        self.dbc.set_other_buildings_table(self.regional_identifier)
        self.logger.info("Buildings_tem table prepared")
        self.dbc.remove_duplicate_buildings()
        self.logger.info("Duplicate buildings removed from buildings_tem")

        unloadcount = self.dbc.set_building_peak_load()
        self.logger.info(
            f"Building peakload calculated in buildings_tem, {unloadcount} unloaded buildings are removed from buildings_tem")
        # Update all buildings with peak load > TRESHHOLD to MV level
        self.dbc.assign_grid_level_connection_by_peak_load()

        # Remove all buildings from buildings_tem with peak load = 0
        self.dbc.remove_zero_peak_load_buildings()

        self.dbc.set_regional_identifier_settlement_type(
            self.regional_identifier)
        self.logger.info("Load density and settlement_type in postcode_result")

        self.dbc.assign_close_buildings()

    def prepare_transformers(self):
        """
        Cache transformers from raw data tables and stores in temporary tables.
        FROM: transformers
        INTO: buildings_tem
        """
        self.dbc.insert_transformers(self.regional_identifier)
        self.logger.info("Transformers inserted in to the buildings_tem table")
        self.dbc.count_indoor_transformers()
        self.dbc.drop_indoor_transformers()
        self.logger.info(
            "Indoor transformers dropped from the buildings_tem table")

    def prepare_ways(self):
        """
        Cache ways, create network, connect buildings to the ways network
        FROM: ways, buildings_tem
        INTO: ways_tem, buildings_tem, ways_tem_vertices_pgr, ways_tem_
        """
        ways_count = self.dbc.set_ways_tem_table(self.regional_identifier)
        self.logger.info(f"The ways_tem table filled with {ways_count} ways")
        # self.dbc.connect_unconnected_ways()
        self.logger.info("Ways connection finished in ways_tem")
        self.dbc.draw_building_connection()
        self.logger.info("Building connection finished in ways_tem")

        self.dbc.update_ways_cost()
        unconn = self.dbc.set_vertice_id()
        self.logger.debug(
            f"vertice id set, {unconn} buildings with no vertice id")

    def apply_kmeans_clustering(self):
        """
        Find connected components (subgraphs) of an undirected street-graph applying the Depth-First Search algorithm
        to edges and vertices from ways_tem and (if necessary due to their size) apply k-means clustering to these
        street network components.

        FROM: ways_tem, buildings_tem
        INTO: ways_tem, vertices_pgr, buildings_tem
        """

        # Get connected components from the street network
        component, vertices = self.dbc.get_connected_component()
        component_ids = np.unique(component)

        if len(component_ids) > 0:
            # Handle components based on number
            if len(component_ids) > 1:
                # Process multiple connected components
                for i, component_id in enumerate(component_ids):
                    related_vertices = vertices[np.argwhere(
                        component == component_id)]
                    self._process_component_to_kcid(related_vertices, i)
            else:
                # Process single connected component
                self._process_component_to_kcid(vertices)
        else:
            # No components found - issue warning
            warnings.warn("No connected components found in ways_tem table")

        # Verify clustering was successful for all buildings
        no_kmean_count = self.dbc.count_no_kmean_buildings()
        if no_kmean_count not in [0, None]:
            warnings.warn(
                f"K-means clustering issue: {no_kmean_count} buildings not assigned to clusters")

    def _process_component_to_kcid(self, vertices, component_index=None):
        """Helper method to process components to kcid groups"""
        conn_building_count = self.dbc.count_connected_buildings(vertices)

        if conn_building_count <= 1 or conn_building_count is None:
            # Remove isolated or empty components
            self.dbc.delete_ways(vertices)
            self.dbc.delete_transformers_from_buildings_tem(vertices)
            self.logger.debug(
                "Empty/isolated component removed. Ways and transformers deleted from temporary tables.")
        elif conn_building_count >= LARGE_COMPONENT_LOWER_BOUND:
            # K-means applied to large component to define subgroups with
            # cluster ids
            cluster_count = int(conn_building_count / LARGE_COMPONENT_DIVIDER)
            self.dbc.update_large_kmeans_cluster(vertices, cluster_count)
            log_msg = f"Large component {component_index} clustered into {cluster_count} groups" if component_index is not None else f"Large component clustered into {cluster_count} groups"
            self.logger.debug(log_msg)
        else:
            # Allocate cluster id for connected component smaller than the
            # building threshold
            self.dbc.update_kmeans_cluster(vertices)

    def position_all_transformers(self):
        """
        Positions all transformers for each bcid cluster (brownfield with existing transformers and greenfield)
        FROM: buildings_tem, grid_result
        INTO: buildings_tem, grid_result
        """
        kcid_length = self.dbc.get_kcid_length()

        for _ in range(kcid_length):
            kcid = self.dbc.get_next_unfinished_kcid(self.regional_identifier)
            self.logger.debug(f"working on kcid {kcid}")
            # Building clustering
            # 0. Check for existing transformers from OSM
            transformers = self.dbc.get_included_transformers(kcid)

            # Case 1: No transformers present
            if not transformers:
                self.logger.debug(f"kcid{kcid} has no included transformer")
                # Create greenfield building clusters
                self.create_bcid_for_kcid(self.regional_identifier, kcid)
                self.logger.debug(f"kcid{kcid} building clusters finished")

            # Case 2: Transformers present
            else:
                self.logger.debug(
                    f"kcid{kcid} has {len(transformers)} transformers")
                # Create brownfield building clusters with existing
                # transformers
                self.position_brownfield_transformers(
                    self.regional_identifier, kcid, transformers)

                # Check buildings and manage clusters
                if self.dbc.count_kmean_cluster_consumers(kcid) > 1:
                    # TODO: name should include transformer_size allocation
                    self.create_bcid_for_kcid(self.regional_identifier, kcid)
                else:
                    # TODO: check approach with isolated buildings
                    self.dbc.delete_isolated_building(
                        self.regional_identifier, kcid)
                self.logger.debug("rest building cluster finished")

            # Process unfinished clusters
            for bcid in self.dbc.get_greenfield_bcids(
                    self.regional_identifier, kcid):
                # Transformer positioning for greenfield clusters
                if bcid >= 0:
                    self.position_greenfield_transformers(
                        self.regional_identifier, kcid, bcid)
                    self.logger.debug(
                        f"Transformer positioning for kcid{kcid}, bcid{bcid} finished")
                    self.dbc.update_transformer_rated_power(
                        self.regional_identifier, kcid, bcid, 1)
                    self.logger.debug(
                        "transformer_rated_power in grid_result is updated.")

    def create_bcid_for_kcid(
            self, regional_identifier: int, kcid: int) -> None:
        """
        Create building clusters (bcids) with average linkage method for a given kcid.
        :param regional_identifier: Postal code
        :param kcid: K-means cluster ID
        :return: None
        """
        # Get data needed for clustering
        buildings = self.dbc.get_buildings_from_kcid(kcid)
        consumer_cat_df = self.dbc.get_consumer_categories()
        settlement_type = self.dbc.get_settlement_type_from_regional_identifier(
            regional_identifier)
        transformer_capacities, _ = self.dbc.get_transformer_data(
            settlement_type)
        double_trans = np.multiply(transformer_capacities[2:4], 2)

        # Get distance matrix and prepare for hierarchical clustering
        localid2vid, dist_mat, vid2localid = self.dbc.get_distance_matrix_from_kcid(
            kcid)
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
            invalid_cluster_dict, cluster_dict, _ = self.dbc.try_clustering(Z, cluster_amount, new_localid2vid, buildings,
                                                                            consumer_cat_df, transformer_capacities,
                                                                            double_trans)

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
                current_invalid_amount = len(invalid_trans_cluster_dict)
                invalid_trans_cluster_dict.update(
                    {x + current_invalid_amount: y for x, y in invalid_cluster_dict.items()})
                invalid_trans_cluster_dict = dict(
                    enumerate(invalid_trans_cluster_dict.values()))

            # Check if clustering is complete
            if not invalid_trans_cluster_dict:
                self.logger.info(
                    f"Found {len(valid_cluster_dict)} single transformer clusters for regional_identifier: {regional_identifier}, KCID: {kcid}")
                break
            else:
                # Process too large clusters by re-clustering them
                self.logger.info(
                    f"Found {len(invalid_trans_cluster_dict)} too_large clusters for regional_identifier: {regional_identifier}, KCID: {kcid}")

                # Get buildings from the first too-large cluster for
                # re-clustering
                invalid_vertice_ids = list(invalid_trans_cluster_dict[0])
                invalid_local_ids = [vid2localid[v]
                                     for v in invalid_vertice_ids]

                # Create new mappings and distance matrix for the subclustering
                new_localid2vid = {
                    k: v for k, v in localid2vid.items() if k in invalid_local_ids}
                new_localid2vid = dict(enumerate(new_localid2vid.values()))
                new_dist_mat = dist_mat[invalid_local_ids][:,
                                                           invalid_local_ids]
                new_dist_vector = squareform(new_dist_mat)

                # Prepare for next iteration
                Z = linkage(new_dist_vector, method="average")
                cluster_amount = 2
                del invalid_trans_cluster_dict[0]
                invalid_trans_cluster_dict = dict(
                    enumerate(invalid_trans_cluster_dict.values()))

        # At this point, we've successfully found a valid electrical clustering solution with the minimum
        # number of clusters. Each cluster:
        #   1. Contains buildings that can be served by a single transformer
        #   2. Has an appropriately sized transformer assigned
        # The valid_cluster_dict maps building cluster IDs to tuples of (building_vertices_list, optimal_transformer_size)
        # We could calculate the total transformer cost by summing the costs of all selected transformers:
        # total_transformer_cost = sum([transformer2cost[v[1]] for v in valid_cluster_dict.values()])

        # Save results to database
        self.dbc.clear_grid_result_in_kmean_cluster(regional_identifier, kcid)
        for bcid, cluster_data in valid_cluster_dict.items():
            self.dbc.upsert_bcid(regional_identifier, kcid, bcid, vertices=cluster_data[0],
                                 transformer_rated_power=cluster_data[1])
        self.logger.debug(
            f"bcids for regional_identifier {regional_identifier} kcid {kcid} found...")

    def position_brownfield_transformers(
            self, regional_identifier: int, kcid: int, transformer_list: list) -> None:
        """
        Assign buildings to the existing transformers and store them as bcid in buildings_tem.
        Args:
            regional_identifier: Postal code
            kcid: K-means cluster ID
            transformer_list: List of transformer IDs
        """
        self.logger.debug(f"{len(transformer_list)} transformers found")

        # Get cost dataframe between consumers and transformers
        cost_df = self.dbc.get_consumer_to_transformer_df(
            kcid, transformer_list)

        # Filter out connections with distance >= 300 why?
        cost_df = cost_df[cost_df["agg_cost"]
                          < 300].sort_values(by=["agg_cost"])

        # Initialize tracking variables
        pre_result_dict = {transformer_id: []
                           for transformer_id in transformer_list}
        full_transformer_list = []
        assigned_consumer_list = []

        # Assign consumers to closest transformer
        for _, row in cost_df.iterrows():
            start_consumer_id = row["start_vid"]
            end_transformer_id = row["end_vid"]

            # Skip if consumer already assigned or transformer full
            if start_consumer_id in assigned_consumer_list or end_transformer_id in full_transformer_list:
                continue

            # Try to assign consumer to transformer
            pre_result_dict[end_transformer_id].append(int(start_consumer_id))
            sim_load = self.dbc.calculate_sim_load(
                pre_result_dict[end_transformer_id])

            # Check if transformer capacity exceeded
            #  TODO: check with settlement_type approach 630 is standard here?
            if float(sim_load) >= 630:
                # Remove consumer and mark transformer as full
                pre_result_dict[end_transformer_id].pop()
                full_transformer_list.append(end_transformer_id)

                # Exit if all transformers are full
                if len(full_transformer_list) == len(transformer_list):
                    self.logger.info("All transformers full")
                    break
            else:
                # Mark consumer as assigned
                assigned_consumer_list.append(start_consumer_id)

        self.logger.debug("Transformer selection finished")

        # Create building clusters for each transformer
        building_cluster_count = 0
        for transformer_id in transformer_list:
            # Skip empty transformers
            if not pre_result_dict[transformer_id]:
                self.logger.debug(
                    f"Transformer {transformer_id} has no assigned consumer, deleted")
                self.dbc.delete_transformers_from_buildings_tem(
                    [transformer_id])
                continue

            # Create building cluster with sequential negative ID
            building_cluster_count -= 1

            # Calculate the simulated load for all loads assigned to this
            # transformer
            sim_load = self.dbc.calculate_sim_load(
                pre_result_dict[transformer_id])

            # Define the available standard transformer sizes in kVA
            # TODO: check with settlement_type approach 630 is standard here?
            possible_transformers = np.array([100, 160, 250, 400, 630])

            # Select the smallest transformer that is larger than the simulated
            # load
            transformer_rated_power = possible_transformers[possible_transformers > float(
                sim_load)][0].item()

            # Update database with new building cluster
            self.dbc.update_building_cluster(transformer_id, pre_result_dict[transformer_id], building_cluster_count, kcid,
                                             regional_identifier, transformer_rated_power)

        self.logger.debug("Brownfield clusters completed")

    def position_greenfield_transformers(
            self, regional_identifier, kcid, bcid):
        """
        Positions a transformer at the optimal location for a greenfield building cluster.

        The optimal location minimizes the sum of distance*load from each vertex to others.

        Args:
            regional_identifier: Postcode
            kcid: Kmeans cluster ID
            bcid: Building cluster ID
        """
        # Get all connection points in the building cluster
        connection_points = self.dbc.get_building_connection_points_from_bc(
            kcid, bcid)

        # If there's only one connection point, use it
        if len(connection_points) == 1:
            self.dbc.upsert_transformer_selection(
                regional_identifier, kcid, bcid, connection_points[0])
            return

        # Get distance matrix between all connection points
        localid2vid, dist_mat, _ = self.dbc.get_distance_matrix_from_bcid(
            kcid, bcid)

        # Get load vector for each connection point
        loads = self.dbc.generate_load_vector(kcid, bcid)

        # Calculate weighted distance (distance * load) for each potential
        # location
        #  Relevant for the transformer placement
        total_load_per_vertice = dist_mat.dot(loads)

        # Select the point with minimum weighted distance as transformer
        # location
        min_localid = np.argmin(total_load_per_vertice)
        transformer_connection_id = int(localid2vid[min_localid])

        # Update the database with the selected transformer position
        self.dbc.upsert_transformer_selection(
            regional_identifier, kcid, bcid, transformer_connection_id)

    def prepare_vertices_list(self, regional_identifier: int, kcid: int, bcid: int) -> tuple[
            dict, int, list, pd.DataFrame, pd.DataFrame, list, list]:
        vertices_dict, transformer_vertice = self.dbc.get_vertices_from_bcid(
            regional_identifier, kcid, bcid)
        #  All vertices needed for this buidling cluster to reach from
        # transformer to the consumer
        vertices_list = list(vertices_dict.keys())

        buildings_df = self.dbc.get_buildings_from_bcid(
            regional_identifier, kcid, bcid)
        consumer_df = self.dbc.get_consumer_categories()
        # Vertices of the buildingcentroids in the building cluster
        consumer_list = buildings_df.vertice_id.to_list()
        consumer_list = list(dict.fromkeys(consumer_list)
                             )  # removing duplicates
        #  Vertices of the connection points in the building cluster
        connection_nodes = [i for i in vertices_list if i not in consumer_list]

        return (vertices_dict, transformer_vertice, vertices_list, buildings_df,
                consumer_df, consumer_list, connection_nodes,)

    def get_consumer_simultaneous_load_dict(self, consumer_list: list, buildings_df: pd.DataFrame) -> tuple[
            dict, dict, dict]:
        # dict of all vertices in bc, 0 as default
        #
        Pd = {consumer: 0 for consumer in consumer_list}
        load_units = {consumer: 0 for consumer in consumer_list}
        load_type = {consumer: "SFH" for consumer in consumer_list}

        for row in buildings_df.itertuples():
            load_units[row.vertice_id] = row.houses_per_building
            load_type[row.vertice_id] = row.type
            # lOOKS UP THE SIMULATENEITY FACTOR
            # todo REFACTOR THIS TO sim_factor
            gzf = CONSUMER_CATEGORIES.loc[CONSUMER_CATEGORIES.definition ==
                                          row.type, "sim_factor"].item()

            # Determine simultaneous load of each building in MW --> is this
            # correct in MW?
            Pd[row.vertice_id] = utils.oneSimultaneousLoad(
                row.peak_load_in_kw * 1e-3, row.houses_per_building, gzf)

        return Pd, load_units, load_type

    def create_lvmv_bus(self, regional_identifier: int, kcid: int, bcid: int,
                        net: pp.pandapowerNet) -> None:

        #  Get the geometry of the transformer
        geodata = self.dbc.get_transformer_geom_from_bcid(
            regional_identifier, kcid, bcid)
        # VN = 400V defined  config_version , min_vm_pu and max_vm_pu defined
        # in represent acceptable VOLTAE levels 1.05 and 0.95
        pp.create_bus(net, name="LVbus 1", vn_kv=VN * 1e-3, geodata=geodata, max_vm_pu=V_BAND_HIGH,
                      min_vm_pu=V_BAND_LOW, type="n", )

        # medium voltage external network and mvbus
        mv_data = (float(geodata[0]), float(geodata[1]) + 1.5 * 1e-4)
        # MVBUS 20kv --> vn_kv
        mv_bus = pp.create_bus(net, name="MVbus 1", vn_kv=20, geodata=mv_data, max_vm_pu=V_BAND_HIGH,
                               min_vm_pu=V_BAND_LOW, type="n", )
        pp.create_ext_grid(net, bus=mv_bus, vm_pu=1, name="External grid")

        return None

    def create_transformer(self, regional_identifier: int, kcid: int,
                           bcid: int, net: pp.pandapowerNet) -> None:

        # transformer_rated_power is the rated power of the transformer in kVA
        transformer_rated_power = self.dbc.get_transformer_rated_power_from_bcid(
            regional_identifier, kcid, bcid)
        # TODO remove hardcoded transformer sizes
        if transformer_rated_power in (250, 400, 630):
            trafo_name = f"{str(transformer_rated_power)} transformer"
            trafo_std = f"{str(transformer_rated_power * 1e-3)} MVA 20/0.4 kV"
            parallel = 1
        elif transformer_rated_power in (100, 160):
            trafo_name = f"{str(transformer_rated_power)} transformer"
            trafo_std = "0.25 MVA 20/0.4 kV"
            parallel = 1
        elif transformer_rated_power in (500, 800):
            trafo_name = f"{str(transformer_rated_power * 0.5)} transformer"
            trafo_std = f"{str(transformer_rated_power *
                               1e-3 *
                               0.5)} MVA 20/0.4 kV"
            parallel = 2
        else:
            trafo_name = "630 transformer"
            trafo_std = "0.63 MVA 20/0.4 kV"
            parallel = transformer_rated_power / 630

        #  Creates the transformer between MV,LV
        trafo_index = pp.create_transformer(net, pp.get_element_index(net, "bus", "MVbus 1"),
                                            pp.get_element_index(net, "bus", "LVbus 1"), name=trafo_name, std_type=trafo_std, tap_pos=0,
                                            parallel=parallel, )
        #  Set the rated power of the transformer in MVA
        net.trafo.at[trafo_index, "sn_mva"] = transformer_rated_power * 1e-3
        return None

    def create_connection_bus(
            self, connection_nodes: list, net: pp.pandapowerNet):
        for i in range(len(connection_nodes)):
            #  Get the geometry of the connection node
            node_geodata = self.dbc.get_node_geom(connection_nodes[i])
            #  Create the connection node bus
            pp.create_bus(net, name=f"Connection Nodebus {connection_nodes[i]}", vn_kv=VN * 1e-3, geodata=node_geodata,
                          max_vm_pu=V_BAND_HIGH, min_vm_pu=V_BAND_LOW, type="n", )

    def create_consumer_bus_and_load(self, consumer_list: list, load_units: dict, net: pp.pandapowerNet,
                                     load_type: dict, building_df: pd.DataFrame) -> None:

        for i in range(len(consumer_list)):
            #  Get the geometry of the consumer node i.e. building centroid:
            node_geodata = self.dbc.get_node_geom(consumer_list[i])

            #  Get the type of the consumer node i.e. building centroid: SFH,
            # MFH, AB, TH, Commercial, Public
            ltype = load_type[consumer_list[i]]

            #  Get the peak load of the consumer node i.e. building centroid:
            if ltype in ["SFH", "MFH", "AB", "TH"]:
                peak_load = CONSUMER_CATEGORIES.loc[CONSUMER_CATEGORIES["definition"]
                                                    == ltype, "peak_load"].values[0]
            else:
                peak_load = building_df[building_df["vertice_id"] ==
                                        consumer_list[i]]["peak_load_in_kw"].tolist()[0]

            pp.create_bus(net=net, name=f"Consumer Nodebus {consumer_list[i]}", vn_kv=VN * 1e-3, geodata=node_geodata,
                          max_vm_pu=V_BAND_HIGH, min_vm_pu=V_BAND_LOW, type="n", zone=ltype, )

            # Create individual loads for each household in the building
            for j in range(1, load_units[consumer_list[i]] + 1):
                pp.create_load(net=net, bus=pp.get_element_index(net, "bus", f"Consumer Nodebus {consumer_list[i]}"),
                               p_mw=0, name=f"Load {consumer_list[i]} household {j}", max_p_mw=peak_load * 1e-3, )

    def install_consumer_cables(self, regional_identifier: int, bcid: int, kcid: int, branch_deviation: float, branch_node_list: list,
                                transformer_vertice: int, vertices_dict: dict, Pd: dict, net: pp.pandapowerNet,
                                connection_available_cables: list[str], local_length_dict: dict, ) -> dict:
        # lines
        # first draw house connections from consumer node to corresponding
        # connection node
        consumer_list = self.dbc.get_vertices_from_connection_points(
            branch_node_list)
        #  List of vertices that are connection points and also in the building
        # cluster
        branch_consumer_list = [
            n for n in consumer_list if n in vertices_dict.keys()]
        #  Loop through all the connection points in the current branch
        for vertice in branch_consumer_list:  # TODO: looping for duplicate vertices
            #  Get the path from the connection point to the transformer
            path_list = self.dbc.get_path_to_bus(vertice, transformer_vertice)
            start_vid = path_list[1]  # Connection point
            end_vid = path_list[0]  # Consumer building centroid

            geodata = self.dbc.get_node_geom(start_vid)
            #  Vor visualization purposes
            start_node_geodata = (float(geodata[0]) + 5 * 1e-6 * branch_deviation,
                                  float(geodata[1]) + 5 * 1e-6 * branch_deviation,)

            end_node_geodata = self.dbc.get_node_geom(end_vid)

            line_geodata = [start_node_geodata, end_node_geodata]

            #  Distance between connection point and consumer building centroid
            cost_km = (vertices_dict[end_vid] -
                       vertices_dict[start_vid]) * 1e-3

            #  CABLE SIZING
            count = 1
            sim_load = Pd[end_vid]  # power in Watt
            Imax = sim_load * 1e-3 / \
                (VN * V_BAND_LOW * np.sqrt(3))  # current in kA
            voltage_available_cables_df = None
            while True:
                line_df = pd.DataFrame.from_dict(
                    net.std_types["line"], orient="index")
                #  Filter out cables that are too small for the current
                current_available_cables_df = line_df[
                    (line_df["max_i_ka"] >= Imax / count) & (line_df.index.isin(connection_available_cables))]

                if len(current_available_cables_df) == 0:
                    count += 1
                    continue

                current_available_cables_df["cable_impedence"] = np.sqrt(
                    current_available_cables_df["r_ohm_per_km"] ** 2 + current_available_cables_df[
                        "x_ohm_per_km"] ** 2)  # impedence in ohm / km
                #  Filter out cables that are too small for the current
                if sim_load <= 100:  # Small loads has 2V drop
                    voltage_available_cables_df = current_available_cables_df[
                        current_available_cables_df["cable_impedence"] <= 2 * 1e-3 / (Imax * cost_km / count)]
                else:  #  Large loads has 4V drop
                    voltage_available_cables_df = current_available_cables_df[
                        current_available_cables_df["cable_impedence"] <= 4 * 1e-3 / (Imax * cost_km / count)]

                if len(voltage_available_cables_df) == 0:
                    count += 1
                    continue
                else:
                    break
            #  Sort the cables by the cross-sectional area and select the
            # smallest one
            cable = voltage_available_cables_df.sort_values(
                by=["q_mm2"]).index.tolist()[0]
            #  Count cost
            local_length_dict[cable] += count * cost_km

            #  With count we allow for parallel lines:
            pp.create_line(net, from_bus=pp.get_element_index(net, "bus", f"Connection Nodebus {start_vid}"),
                           to_bus=pp.get_element_index(net, "bus", f"Consumer Nodebus {end_vid}"), length_km=cost_km,
                           std_type=cable, name=f"Line to {end_vid}", geodata=line_geodata, parallel=count, )

            self.dbc.insert_lines(geom=line_geodata, regional_identifier=regional_identifier, bcid=bcid, kcid=kcid, line_name=f"Line to {end_vid}",
                                  std_type=cable,
                                  from_bus=pp.get_element_index(
                                      net, "bus", f"Connection Nodebus {start_vid}"),
                                  to_bus=pp.get_element_index(net, "bus", f"Consumer Nodebus {end_vid}"), length_km=cost_km)

        return local_length_dict

    def find_minimal_available_cable(self, Imax: float, net: pp.pandapowerNet, cables_list: list, distance: int = 0) -> \
            tuple[str, int]:
        count = 1
        cable = None
        while 1:
            line_df = pd.DataFrame.from_dict(
                net.std_types["line"], orient="index")
            current_available_cables = line_df[
                (line_df.index.isin(cables_list)) & (line_df["max_i_ka"] >= Imax / count)]
            if len(current_available_cables) == 0:
                count += 1
                continue

            if distance != 0:
                current_available_cables["cable_impedence"] = np.sqrt(
                    current_available_cables["r_ohm_per_km"] ** 2 + current_available_cables[
                        "x_ohm_per_km"] ** 2)  # impedence in ohm / km
                voltage_available_cables = current_available_cables[
                    current_available_cables["cable_impedence"] <= 400 * 0.045 / (Imax * distance / count)]
                if len(voltage_available_cables) == 0:
                    count += 1
                    continue
                else:
                    cable = voltage_available_cables.sort_values(
                        by=["q_mm2"]).index.tolist()[0]
                    break
            else:
                cable = current_available_cables.sort_values(
                    by=["q_mm2"]).index.tolist()[0]
                break

        return cable, count

    def create_line_transformer_to_lv_bus(self, regional_identifier: int, bcid: int, kcid: int, branch_start_node: int, branch_deviation: float,
                                          net: pp.pandapowerNet, cable: str, count: int):  # TODO: check if this line is required
        end_vid = branch_start_node
        node_geodata = self.dbc.get_node_geom(end_vid)
        node_geodata = (float(node_geodata[0]) + 5 * 1e-6 * branch_deviation,
                        float(node_geodata[1]) + 5 * 1e-6 * branch_deviation,)
        lvbus_geodata = (
            net.bus_geodata.loc[pp.get_element_index(
                net, "bus", "LVbus 1"), "x"] + 5 * 1e-6 * branch_deviation,
            net.bus_geodata.loc[pp.get_element_index(net, "bus", "LVbus 1"), "y"],)
        line_geodata = [lvbus_geodata, node_geodata]

        cost_km = 0
        pp.create_line(net, from_bus=pp.get_element_index(net, "bus", "LVbus 1"),
                       to_bus=pp.get_element_index(net, "bus", f"Connection Nodebus {end_vid}"), length_km=cost_km, std_type=cable,
                       name=f"Line to {end_vid}", geodata=line_geodata, parallel=count, )

        self.dbc.insert_lines(geom=line_geodata, regional_identifier=regional_identifier, bcid=bcid, kcid=kcid, line_name=f"Line to {end_vid}",
                              std_type=cable, from_bus=pp.get_element_index(
                                  net, "bus", "LVbus 1"),
                              to_bus=pp.get_element_index(net, "bus", f"Connection Nodebus {end_vid}"), length_km=cost_km)

    def create_line_start_to_lv_bus(self, regional_identifier: int, bcid: int, kcid: int, branch_start_node: int,
                                    branch_deviation: float, net: pp.pandapowerNet, vertices_dict: dict, cable: str, count: int,
                                    transformer_vertice: int, ) -> int:

        node_path_list = self.dbc.get_path_to_bus(
            branch_start_node, transformer_vertice)

        line_geodata = []
        for p in node_path_list:
            node_geodata = self.dbc.get_node_geom(p)
            node_geodata = (float(node_geodata[0]) + 5 * 1e-6 * branch_deviation,
                            float(node_geodata[1]) + 5 * 1e-6 * branch_deviation,)
            line_geodata.append(node_geodata)
        lvbus_geodata = (
            net.bus_geodata.loc[pp.get_element_index(
                net, "bus", "LVbus 1"), "x"] + 5 * 1e-6 * branch_deviation,
            net.bus_geodata.loc[pp.get_element_index(net, "bus", "LVbus 1"), "y"],)
        line_geodata.append(lvbus_geodata)
        line_geodata.reverse()

        cost_km = vertices_dict[branch_start_node] * 1e-3
        length = count * cost_km  # distance in m
        pp.create_line(net, from_bus=pp.get_element_index(net, "bus", "LVbus 1"),
                       to_bus=pp.get_element_index(net, "bus", f"Connection Nodebus {branch_start_node}"), length_km=cost_km,
                       std_type=cable, name=f"Line to {branch_start_node}", geodata=line_geodata, parallel=count, )

        self.dbc.insert_lines(geom=line_geodata, regional_identifier=regional_identifier, bcid=bcid, kcid=kcid, line_name=f"Line to {branch_start_node}",
                              std_type=cable, from_bus=pp.get_element_index(
                                  net, "bus", "LVbus 1"),
                              to_bus=pp.get_element_index(
                                  net, "bus", f"Connection Nodebus {branch_start_node}"),
                              length_km=cost_km)

        return length

    def deviate_bus_geodata(self, branch_node_list: list,
                            branch_deviation: float, net: pp.pandapowerNet):
        for node in branch_node_list:
            net.bus_geodata.at[pp.get_element_index(net, "bus", f"Connection Nodebus {node}"), "x"] += (
                5 * 1e-6 * branch_deviation)
            net.bus_geodata.at[pp.get_element_index(net, "bus", f"Connection Nodebus {node}"), "y"] += (
                5 * 1e-6 * branch_deviation)

    def find_furthest_node_path_list(
            self, connection_node_list: list, vertices_dict: dict, transformer_vertice: int) -> list:
        # Dictionary mapping connection node to its distance from the transformer
        #  vertice_dict;: # t[0] = vertex ID, t[1] = routing cost
        connection_node_dict = {
            n: vertices_dict[n] for n in connection_node_list}
        furthest_node = max(connection_node_dict, key=connection_node_dict.get)
        # all the connection nodes in the path from transformer to furthest
        # node are considered as potential branch loads
        #  Dijkstra algorithm to find the shortest path from the transformer to
        # the furthest node , returns list of nodes
        furthest_node_path_list = self.dbc.get_path_to_bus(
            furthest_node, transformer_vertice)
        #  Filter out connection nodes that are not in the path from transformer to furthest node
        # What is the main trunk of ccable installation from which we branch
        # off.

        furthest_node_path = [
            p for p in furthest_node_path_list if p in connection_node_list]

        return furthest_node_path

    def determine_maximum_load_branch(self, furthest_node_path_list: list, buildings_df: pd.DataFrame,
                                      consumer_df: pd.DataFrame) -> tuple[list, float]:
        # This function essentially answers: "How much of the main trunk can we
        # build with standard heavy-duty cables before we need special
        # solutions?"
        branch_node_list = []
        #  Startong form the transpofrmer we do incremental load calculation
        for node in furthest_node_path_list:
            branch_node_list.append(node)
            #  sim_peak load in kW
            sim_load = utils.simultaneousPeakLoad(
                buildings_df, consumer_df, branch_node_list)
            #  current in kA
            # current in kA, worstcase scenario
            Imax = sim_load / (VN * V_BAND_LOW * np.sqrt(3))
            #  Hard coded put into function, set of available cables --> MAX
            # CURRENT LIMIT OF CABLES
            if Imax >= 0.313 and len(
                    branch_node_list) > 1:  # 0.313 is the current limit of the largest allowed cable 4x185SE
                branch_node_list.remove(node)
                break
            elif Imax >= 0.313 and len(branch_node_list) == 1:
                break
        sim_load = utils.simultaneousPeakLoad(
            buildings_df, consumer_df, branch_node_list)
        Imax = sim_load / (VN * V_BAND_LOW * np.sqrt(3))

        return branch_node_list, Imax

    def create_line_node_to_node(self, regional_identifier: int, kcid: int, bcid: int, branch_node_list: list, branch_deviation: float,
                                 vertices_dict: dict, local_length_dict: dict, cable: str, transformer_vertice: int, count: float,
                                 net: pp.pandapowerNet) -> dict:
        """creates the lines / cables from one Connection Nodebus to the next. Adds them to the pandapower network
        and lines result table"""
        for i in range(len(branch_node_list) - 1):
            # to get the line geodata, we now need to consider all the nodes in
            # database, not only connection points
            node_path_list = self.dbc.get_path_to_bus(
                # gets the path along ways_result
                branch_node_list[i], transformer_vertice)
            # end at next connection point
            # if next node of branch node list not in node path list
            if branch_node_list[i + 1] not in node_path_list:
                self.logger.debug(
                    f"creating line to node i + 1: {i + 1} node: {branch_node_list[i + 1]}")
                node_path_list = self.dbc.get_path_to_bus(branch_node_list[i], branch_node_list[
                    # node_path_list = [branch_node_list[i], branch_node_list[i
                    # + 1]]  # intermediate nodes up to next connection nodebus
                    # are neglected  # the cable will directly connect to next
                    # connection nodebus
                    i + 1])

            node_path_list = node_path_list[: node_path_list.index(
                # the node path list goes up to the index (branch_node_list[i +
                # 1]) +1
                branch_node_list[i + 1]) + 1]
            node_path_list.reverse()  # to keep the correct direction

            start_vid = node_path_list[0]
            end_vid = node_path_list[-1]

            line_geodata = []
            for p in node_path_list:
                node_geodata = self.dbc.get_node_geom(p)
                node_geodata = (float(node_geodata[0]) + 5 * 1e-6 * branch_deviation,
                                float(node_geodata[1]) + 5 * 1e-6 * branch_deviation,)
                line_geodata.append(node_geodata)

            cost_km = (vertices_dict[end_vid] -
                       vertices_dict[start_vid]) * 1e-3

            local_length_dict[cable] += count * cost_km
            pp.create_line(net, from_bus=pp.get_element_index(net, "bus", f"Connection Nodebus {start_vid}"),
                           to_bus=pp.get_element_index(net, "bus", f"Connection Nodebus {end_vid}"), length_km=cost_km,
                           std_type=cable, name=f"Line to {end_vid}", geodata=line_geodata, parallel=count, )

            self.dbc.insert_lines(geom=line_geodata, regional_identifier=regional_identifier, bcid=bcid, kcid=kcid, line_name=f"Line to {end_vid}",
                                  std_type=cable, from_bus=pp.get_element_index(
                                      net, "bus", f"Connection Nodebus {start_vid}"),
                                  to_bus=pp.get_element_index(net, "bus", f"Connection Nodebus {end_vid}"), length_km=cost_km)
        return local_length_dict

    def save_net(self, net, kcid, bcid):
        """
        Save one grid to file and to database
        """
        if SAVE_GRID_FOLDER:
            savepath_folder = Path(
                RESULT_DIR,
                "grids",
                f"version_{VERSION_ID}",
                self.regional_identifier)
            savepath_folder.mkdir(parents=True, exist_ok=True)
            filename = f"kcid{kcid}bcid{bcid}.json"
            savepath_file = Path(savepath_folder, filename)
            pp.to_json(net, filename=savepath_file)

        json_string = pp.to_json(net, filename=None)

        self.dbc.save_pp_net_with_json(
            self.regional_identifier, kcid, bcid, json_string)

        self.logger.info(f"Grid with kcid:{kcid} bcid:{bcid} is stored. ")

    def install_cables_parallel(self, max_workers: int = 4):
        """
        Parallelized version of install_cables using multiprocessing.
        Installs electrical cables to connect buildings and transformers in power grid clusters.

        This method creates a pandapower network for each building cluster (kcid, bcid) in the
        postal code area and connects the buildings with appropriate electrical cables. It follows
        a branch-by-branch approach, starting from the furthest nodes and working inward toward
        the transformer.

        The algorithm works as follows:
        1. Retrieves all clusters (kcid, bcid) for the postal code area
        2. For each cluster:
           a. Prepares building and connection data
           b. Creates an electrical network with pandapower
           c. Adds buses, transformers, and loads to the network
           d. Installs cables using a greedy algorithm that:
              - Starts from the furthest nodes from the transformer
              - Creates branches with maximum possible load
              - Selects minimum size cables that can handle the current
              - Connects branches back to transformer
        3. Tracks progress and saves the network configurations

        The cable installation prioritizes cost efficiency while ensuring the electrical
        requirements are met for each branch of the distribution network.

        Returns:
            None

        Args:
            max_workers: Maximum number of worker processes. If None, uses CPU count.
        """
        from concurrent.futures import ProcessPoolExecutor, as_completed

        cluster_list = self.dbc.get_list_from_regional_identifier(
            self.regional_identifier)
        if not cluster_list:
            self.logger.warning(
                f"No clusters to process for regional_identifier {
                    self.regional_identifier}")
            return

        self.logger.info(
            f"Starting parallel cable installation for {len(cluster_list)} clusters using {max_workers} workers.")

        # Create batches of clusters to process
        def create_batches(items, batch_size):
            for i in range(0, len(items), batch_size):
                yield items[i:i + batch_size]

        # Calculate batch size to distribute work evenly
        batch_size = max(1, len(cluster_list) // max_workers)
        cluster_batches = list(create_batches(cluster_list, batch_size))

        with ProcessPoolExecutor(max_workers=max_workers, initializer=GridGenerator._init_worker, initargs=(self.regional_identifier,)) as executor:
            future_to_batch = {
                executor.submit(GridGenerator._process_cluster_batch, batch): batch
                for batch in cluster_batches
            }

            for future in as_completed(future_to_batch):
                batch = future_to_batch[future]
                try:
                    future.result()
                    self.logger.debug(
                        f"Successfully processed batch with {len(batch)} clusters")
                except Exception as e:
                    self.logger.error(
                        f"Failed to process batch with {
                            len(batch)} clusters: {e}",
                        exc_info=True)

        self.logger.info(
            f"Parallel cable installation completed for regional_identifier {self.regional_identifier}")

    @staticmethod
    def _init_worker(regional_identifier):
        """Initialize worker process with one GridGenerator per worker."""
        global _worker_grid_generator
        _worker_grid_generator = GridGenerator(
            regional_identifier=regional_identifier)

    @staticmethod
    def _process_cluster_batch(cluster_batch):
        """Process a batch of clusters using the worker's GridGenerator."""
        global _worker_grid_generator
        try:
            for kcid, bcid in cluster_batch:
                _worker_grid_generator._install_cables_for_cluster(kcid, bcid)
        except Exception as e:
            print(f"Error in worker batch processing: {e}")
            raise

    def _install_cables_for_cluster(self, kcid, bcid):
        """
        Installs electrical cables for a single building cluster (kcid, bcid).
        """
        self.logger.debug(f"working on kcid {kcid}, bcid {bcid}")

        # Get data for this cluster
        vertices_dict, transformer_vertice, vertices_list, buildings_df, consumer_df, consumer_list, connection_nodes = (
            self.prepare_vertices_list(self.regional_identifier, kcid, bcid)
        )
        # PD is the simultaneous load of the consumer list
        # load_units is the number of units of the consumer list: units per building
        # load_type is the type of the consumer list [SFH, MFH, AB, TH,
        # Commercial, Public]
        Pd, load_units, load_type = self.get_consumer_simultaneous_load_dict(
            consumer_list, buildings_df)
        local_length_dict = {c: 0 for c in CABLE_COST_DICT.keys()}

        # Create network and add components
        net = pp.create_empty_network()
        # Creates the standard cable types from the equipment_data table as
        # pandapoewer objects
        self.dbc.create_cable_std_type(net)
        # Creates two buses LV and MV  and also defines external grid
        self.create_lvmv_bus(self.regional_identifier, kcid, bcid, net)

        # Defines the transformer between MV,LV
        self.create_transformer(self.regional_identifier, kcid, bcid, net)

        # Creates buses for all the conenction nodes
        self.create_connection_bus(connection_nodes, net)
        # Creates buses for all the consumers and also one load per unit in the
        # building
        self.create_consumer_bus_and_load(
            consumer_list, load_units, net, load_type, buildings_df)

        # Install cables branch by branch
        branch_deviation = 0
        connection_node_list = connection_nodes
        main_street_available_cables = CABLE_COST_DICT.keys()

        while connection_node_list:
            # Handle single remaining node case
            if len(connection_node_list) == 1:
                sim_load = utils.simultaneousPeakLoad(
                    buildings_df, consumer_df, connection_node_list)

                #  Calculate the maximum current for the connection node
                Imax = sim_load / (VN * V_BAND_LOW * np.sqrt(3))

                # Install consumer cables
                local_length_dict = self.install_consumer_cables(
                    self.regional_identifier, bcid, kcid, branch_deviation, connection_node_list,
                    transformer_vertice, vertices_dict, Pd, net, CONNECTION_AVAILABLE_CABLES, local_length_dict,
                )

                # Connect to transformer
                if connection_node_list[0] == transformer_vertice:
                    cable, count = self.find_minimal_available_cable(
                        Imax, net, main_street_available_cables)
                    self.create_line_transformer_to_lv_bus(
                        self.regional_identifier, bcid, kcid, connection_node_list[
                            0], branch_deviation, net, cable, count
                    )
                else:
                    cable, count = self.find_minimal_available_cable(
                        Imax, net, main_street_available_cables, vertices_dict[connection_nodes[0]]
                    )
                    length = self.create_line_start_to_lv_bus(
                        self.regional_identifier, bcid, kcid, connection_node_list[
                            0], branch_deviation,
                        net, vertices_dict, cable, count, transformer_vertice
                    )
                    local_length_dict[cable] += length

                self.deviate_bus_geodata(
                    connection_node_list, branch_deviation, net)
                self.logger.debug(
                    "main street cable installation finished")
                break

            # List of nodes until the furthest node from the transformer
            furthest_node_path_list = self.find_furthest_node_path_list(
                connection_node_list, vertices_dict, transformer_vertice
            )
            # Returns the list of nodes that can be connected with largest
            # heavy-duty cables, given curren limitations
            branch_node_list, Imax = self.determine_maximum_load_branch(
                furthest_node_path_list, buildings_df, consumer_df
            )

            # Install cables for this branch --> this is the main trunk of the
            # cable installation
            local_length_dict = self.install_consumer_cables(
                self.regional_identifier, bcid, kcid, branch_deviation, branch_node_list,
                transformer_vertice, vertices_dict, Pd, net, CONNECTION_AVAILABLE_CABLES, local_length_dict
            )

            branch_distance = vertices_dict[branch_node_list[0]]
            cable, count = self.find_minimal_available_cable(
                Imax, net, main_street_available_cables, branch_distance
            )

            if len(branch_node_list) >= 2:
                local_length_dict = self.create_line_node_to_node(
                    self.regional_identifier, kcid, bcid, branch_node_list, branch_deviation,
                    vertices_dict, local_length_dict, cable, transformer_vertice, count, net
                )

            # Connect branch to transformer
            branch_start_node = branch_node_list[-1]
            if branch_start_node == transformer_vertice:
                self.create_line_transformer_to_lv_bus(
                    self.regional_identifier, bcid, kcid, branch_start_node, branch_deviation, net, cable, count
                )
            else:
                length = self.create_line_start_to_lv_bus(
                    self.regional_identifier, bcid, kcid, branch_start_node, branch_deviation,
                    net, vertices_dict, cable, count, transformer_vertice
                )
                local_length_dict[cable] += length

            # Update processed nodes and visualization
            for vertice in branch_node_list:
                connection_node_list.remove(vertice)

            self.deviate_bus_geodata(
                branch_node_list, branch_deviation, net)
            branch_deviation += 1

        self.save_net(net, kcid, bcid)
