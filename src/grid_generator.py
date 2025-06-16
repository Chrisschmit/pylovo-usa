import warnings
from pathlib import Path

import pandapower as pp
import numpy as np
import src.database_client as dbc
from src import utils
from src.config_loader import *


class ResultExistsError(Exception):
    "Raised when the PLZ has already been created."
    pass


class GridGenerator:
    """
    Generates the grid for the given plz area
    """

    def __init__(self, plz=999999, **kwargs):
        self.plz = str(plz)
        self.dbc = dbc.DatabaseClient()
        self.dbc.insert_version_if_not_exists()
        self.dbc.insert_parameter_tables(consumer_categories=CONSUMER_CATEGORIES)
        self.logger = utils.create_logger(
            name="GridGenerator", log_file=kwargs.get("log_file", "log.txt"), log_level=LOG_LEVEL
        )

    def __del__(self):
        self.dbc.__del__()

    def generate_grid(self):
        self.check_if_results_exist()
        self.prepare_postcodes()
        self.prepare_buildings()
        self.prepare_transformers()
        self.prepare_ways()
        self.apply_kmeans_clustering()
        self.position_all_transformers()
        self.install_cables()
        

    def check_if_results_exist(self):
        postcode_count = self.dbc.count_postcode_result(self.plz)
        if postcode_count:
            raise ResultExistsError(
                f"The grids for the postcode area {self.plz} is already generated "
                f"for the version {VERSION_ID}."
            )

    def prepare_postcodes(self):
        """
        Caches postcode from raw data tables and stores in temporary tables.
        FROM: postcode
        INTO: postcode_result
        """
        self.dbc.copy_postcode_result_table(self.plz)
        self.logger.info(f"Working on plz {self.plz}")

    def prepare_buildings(self):
        """
        Caches buildings from raw data tables and stores in temporary tables.
        FROM: res, oth
        INTO: buildings_tem
        """
        self.dbc.set_residential_buildings_table(self.plz)
        self.dbc.set_other_buildings_table(self.plz)
        self.logger.info("Buildings_tem table prepared")
        self.dbc.remove_duplicate_buildings()
        self.logger.info("Duplicate buildings removed from buildings_tem")

        self.dbc.set_plz_settlement_type(self.plz)
        self.logger.info("House_distance and settlement_type in postcode_result")

        unloadcount = self.dbc.set_building_peak_load()
        self.logger.info(
            f"Building peakload calculated in buildings_tem, {unloadcount} unloaded buildings are removed from "
            f"buildings_tem"
        )
        too_large_consumers = self.dbc.update_too_large_consumers_to_zero()
        self.logger.info(f"{too_large_consumers} too large consumers removed from buildings_tem")

        self.dbc.assign_close_buildings()
        self.logger.info("All close buildings assigned and removed from buildings_tem")

    def prepare_transformers(self):
        """
        Cache transformers from raw data tables and stores in temporary tables.
        FROM: transformers
        INTO: buildings_tem
        """
        self.dbc.insert_transformers(self.plz)
        self.logger.info("Transformers inserted in to the buildings_tem table")
        self.dbc.count_indoor_transformers()
        self.dbc.drop_indoor_transformers()
        self.logger.info("Indoor transformers dropped from the buildings_tem table")

    def prepare_ways(self):
        """
        Cache ways, create network, connect buildings to the ways network
        FROM: ways, buildings_tem
        INTO: ways_tem, buildings_tem, ways_tem_vertices_pgr, ways_tem_
        """
        ways_count = self.dbc.set_ways_tem_table(self.plz)
        self.logger.info(f"The ways_tem table filled with {ways_count} ways")
        self.dbc.connect_unconnected_ways()
        self.logger.info("Ways connection finished in ways_tem")
        self.dbc.draw_building_connection()
        self.logger.info("Building connection finished in ways_tem")

        self.dbc.update_ways_cost()
        unconn = self.dbc.set_vertice_id()
        self.logger.debug(f"vertice id set, {unconn} buildings with no vertice id")

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
                    related_vertices = vertices[np.argwhere(component == component_id)]
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
            warnings.warn(f"K-means clustering issue: {no_kmean_count} buildings not assigned to clusters")

    def _process_component_to_kcid(self, vertices, component_index=None):
        """Helper method to process components to kcid groups"""
        conn_building_count = self.dbc.count_connected_buildings(vertices)

        if conn_building_count <= 1 or conn_building_count is None:
            # Remove isolated or empty components
            self.dbc.delete_ways(vertices)
            self.dbc.delete_transformers_from_buildings_tem(vertices)
            self.logger.debug("Empty/isolated component removed. Ways and transformers deleted from temporary tables.")
        elif conn_building_count >= LARGE_COMPONENT_LOWER_BOUND:
            # K-means applied to large component to define subgroups with cluster ids
            cluster_count = int(conn_building_count / LARGE_COMPONENT_DIVIDER)
            self.dbc.update_large_kmeans_cluster(vertices, cluster_count)
            log_msg = f"Large component {component_index} clustered into {cluster_count} groups" if component_index is not None else f"Large component clustered into {cluster_count} groups"
            self.logger.debug(log_msg)
        else:
            # Allocate cluster id for connected component smaller than the building threshold
            self.dbc.update_kmeans_cluster(vertices)

    def position_all_transformers(self):
        """
        Positions all transformers for each bcid cluster (brownfield with existing transformers and greenfield)
        FROM: buildings_tem, grid_result
        INTO: buildings_tem, grid_result
        """
        kcid_length = self.dbc.get_kcid_length()

        for _ in range(kcid_length):
            kcid = self.dbc.get_next_unfinished_kcid(self.plz)
            self.logger.debug(f"working on kcid {kcid}")
            # Building clustering
            # 0. Check for existing transformers from OSM
            transformers = self.dbc.get_included_transformers(kcid)

            # Case 1: No transformers present
            if not transformers:
                self.logger.debug(f"kcid{kcid} has no included transformer")
                # Create greenfield building clusters
                self.dbc.create_bcid_for_kcid(self.plz, kcid)
                self.logger.debug(f"kcid{kcid} building clusters finished")

            # Case 2: Transformers present
            else:
                self.logger.debug(f"kcid{kcid} has {len(transformers)} transformers")
                # Create brownfield building clusters with existing transformers
                self.dbc.position_brownfield_transformers(self.plz, kcid, transformers)

                # Check buildings and manage clusters
                if self.dbc.count_kmean_cluster_consumers(kcid) > 1:
                    self.dbc.create_bcid_for_kcid(self.plz, kcid) #TODO: name should include transformer_size allocation
                else:
                    self.dbc.delete_isolated_building(self.plz, kcid) #TODO: check approach with isolated buildings
                self.logger.debug("rest building cluster finished")

            # Process unfinished clusters
            for bcid in self.dbc.get_greenfield_bcids(self.plz, kcid):
                # Transformer positioning for greenfield clusters
                if bcid >= 0:
                    self.dbc.position_greenfield_transformers(self.plz, kcid, bcid)
                    self.logger.debug(f"Transformer positioning for kcid{kcid}, bcid{bcid} finished")
                    self.dbc.update_transformer_rated_power(self.plz, kcid, bcid, 1)
                    self.logger.debug("transformer_rated_power in grid_result is updated.")

    def install_cables(self):
        """
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
        """
        # Get all clusters for the postal code area
        cluster_list = self.dbc.get_list_from_plz(self.plz)
        ci_count = 0
        ci_process = 0
        main_street_available_cables = CABLE_COST_DICT.keys()

        for id in cluster_list:
            kcid, bcid = id
            self.logger.debug(f"working on kcid {kcid}, bcid {bcid}")

            # Get data for this cluster
            vertices_dict, ont_vertice, vertices_list, buildings_df, consumer_df, consumer_list, connection_nodes = (
                self.dbc.prepare_vertices_list(self.plz, kcid, bcid)
            )
            Pd, load_units, load_type = self.dbc.get_consumer_simultaneous_load_dict(consumer_list, buildings_df)
            local_length_dict = {c: 0 for c in CABLE_COST_DICT.keys()}

            # Create network and add components
            net = pp.create_empty_network()
            self.dbc.create_cable_std_type(net)
            self.dbc.create_lvmv_bus(self.plz, kcid, bcid, net)
            self.dbc.create_transformer(self.plz, kcid, bcid, net)
            self.dbc.create_connection_bus(connection_nodes, net)
            self.dbc.create_consumer_bus_and_load(consumer_list, load_units, net, load_type, buildings_df)

            # Install cables branch by branch
            branch_deviation = 0
            connection_node_list = connection_nodes

            while connection_node_list:
                # Handle single remaining node case
                if len(connection_node_list) == 1:
                    sim_load = utils.simultaneousPeakLoad(buildings_df, consumer_df, connection_node_list)
                    Imax = sim_load / (VN * V_BAND_LOW * np.sqrt(3))

                    # Install consumer cables
                    local_length_dict = self.dbc.install_consumer_cables(
                        self.plz, bcid, kcid, branch_deviation, connection_node_list,
                        ont_vertice, vertices_dict, Pd, net, CONNECTION_AVAILABLE_CABLES, local_length_dict,
                    )

                    # Connect to transformer
                    if connection_node_list[0] == ont_vertice:
                        cable, count = self.dbc.find_minimal_available_cable(Imax, net, main_street_available_cables)
                        self.dbc.create_line_ont_to_lv_bus(
                            self.plz, bcid, kcid, connection_node_list[0], branch_deviation, net, cable, count
                        )
                    else:
                        cable, count = self.dbc.find_minimal_available_cable(
                            Imax, net, main_street_available_cables, vertices_dict[connection_nodes[0]]
                        )
                        length = self.dbc.create_line_start_to_lv_bus(
                            self.plz, bcid, kcid, connection_node_list[0], branch_deviation,
                            net, vertices_dict, cable, count, ont_vertice
                        )
                        local_length_dict[cable] += length

                    self.dbc.deviate_bus_geodata(connection_node_list, branch_deviation, net)
                    self.logger.debug("main street cable installation finished")
                    break

                # Process multiple nodes as branches
                furthest_node_path_list = self.dbc.find_furthest_node_path_list(
                    connection_node_list, vertices_dict, ont_vertice
                )
                branch_node_list, Imax = self.dbc.determine_maximum_load_branch(
                    furthest_node_path_list, buildings_df, consumer_df
                )

                # Install cables for this branch
                local_length_dict = self.dbc.install_consumer_cables(
                    self.plz, bcid, kcid, branch_deviation, branch_node_list,
                    ont_vertice, vertices_dict, Pd, net, CONNECTION_AVAILABLE_CABLES, local_length_dict
                )

                # Select appropriate cable and connect nodes
                branch_distance = vertices_dict[branch_node_list[0]]
                cable, count = self.dbc.find_minimal_available_cable(
                    Imax, net, main_street_available_cables, branch_distance
                )

                if len(branch_node_list) >= 2:
                    local_length_dict = self.dbc.create_line_node_to_node(
                        self.plz, kcid, bcid, branch_node_list, branch_deviation,
                        vertices_dict, local_length_dict, cable, ont_vertice, count, net
                    )

                # Connect branch to transformer
                branch_start_node = branch_node_list[-1]
                if branch_start_node == ont_vertice:
                    self.dbc.create_line_ont_to_lv_bus(
                        self.plz, bcid, kcid, branch_start_node, branch_deviation, net, cable, count
                    )
                else:
                    length = self.dbc.create_line_start_to_lv_bus(
                        self.plz, bcid, kcid, branch_start_node, branch_deviation,
                        net, vertices_dict, cable, count, ont_vertice
                    )
                    local_length_dict[cable] += length

                # Update processed nodes and visualization
                for vertice in branch_node_list:
                    connection_node_list.remove(vertice)

                self.dbc.deviate_bus_geodata(branch_node_list, branch_deviation, net)
                branch_deviation += 1

            # Track and report progress
            ci_count += 1
            progress_increment = 10  # Report progress in 10% increments
            progress_threshold = max(1, len(cluster_list) / progress_increment)

            if ci_count >= progress_threshold:
                ci_process += progress_increment
                ci_count = 0
                self.logger.info(
                    f"Cable installation: {min(ci_process, 100)}% complete ({ci_process // progress_increment}/{progress_increment})")

            self.save_net(net, kcid, bcid)

    def analyse_results(self):
        try:
            self.logger.info("Start basic result analysis")
            self.dbc.analyse_basic_parameters(self.plz)
            self.logger.info("Start cable counting")
            self.dbc.analyse_cables(self.plz)
            self.logger.info("Start per trafo analysis")
            self.dbc.analyse_per_trafo_parameters(self.plz)
            self.logger.info("Result analysis finished")
            self.dbc.conn.commit()
        except Exception as e:
            self.logger.error(f"Error during analysis for PLZ {self.plz}: {e}")
            self.logger.info(f"Skipped PLZ {self.plz} due to analysis error.")
            self.dbc.delete_plz_from_sample_set_table(str(CLASSIFICATION_VERSION),self.plz)  # delete from sample set
            


    def save_net(self, net, kcid, bcid):
        """
        Save one grid to file and to database
        """
        if SAVE_GRID_FOLDER:
            savepath_folder = Path(RESULT_DIR, "grids", f"version_{VERSION_ID}", self.plz)
            savepath_folder.mkdir(parents=True, exist_ok=True)
            filename = f"kcid{kcid}bcid{bcid}.json"
            savepath_file = Path(savepath_folder, filename)
            pp.to_json(net, filename=savepath_file)

        json_string = pp.to_json(net, filename=None)

        self.dbc.save_pp_net_with_json(self.plz, kcid, bcid, json_string)

        self.logger.info(f"Grid with kcid:{kcid} bcid:{bcid} is stored. ")

    def generate_grid_for_multiple_plz(self, df_plz: pd.DataFrame, analyze_grids: bool = False) -> None:
        """generates grid for all plz contained in the column 'plz' of df_samples

        :param df_plz: table that contains PLZ for grid generation
        :type df_plz: pd.DataFrame
        :param analyze_grids: option to analyse the results after grid generation, defaults to False
        :type analyze_grids: bool
        """
        self.dbc.create_temp_tables() # create temp tables for the grid generation
        
        for index, row in df_plz.iterrows():
            self.plz = str(row['plz'])
            print('-------------------- start', self.plz, '---------------------------')
            try:
                self.generate_grid()
                self.dbc.save_tables(plz=self.plz) # Save data from temporary tables to result tables
                self.dbc.reset_tables() # Reset temporary tables
                if analyze_grids:
                    self.analyse_results()
            except ResultExistsError:
                print('Grids for this PLZ have already been generated.')
            except Exception as e:
                self.logger.error(f"Error during grid generation for PLZ {self.plz}: {e}")
                self.logger.info(f"Skipped PLZ {self.plz} due to generation error.")
                self.dbc.conn.rollback() # rollback the transaction
                self.dbc.delete_plz_from_sample_set_table(str(CLASSIFICATION_VERSION),self.plz)  # delete from sample set
                continue
            print('-------------------- end', self.plz, '-----------------------------')
        
        
        self.dbc.drop_temp_tables() # drop temp tables
        self.dbc.commit_changes() # commit the changes to the database
    
    def generate_grid_for_single_plz(self, plz: str, analyze_grids: bool = False) -> None:
        """
        Generates the grid for a single PLZ.

        :param plz: Postal code for which the grid should be generated.
        :type plz: str
        :param analyze_grids: Option to analyze the results after grid generation, defaults to False.
        :type analyze_grids: bool
        """
        self.plz = plz
        print('-------------------- start', self.plz, '---------------------------')
        
        self.dbc.create_temp_tables()  # create temp tables for the grid generation

        try:
            self.generate_grid()
            self.dbc.save_tables(plz=self.plz) # Save data from temporary tables to result tables
            if analyze_grids:
                self.analyse_results()
        except ResultExistsError:
            print('Grids for this PLZ have already been generated.')
        except Exception as e:
            self.logger.error(f"Error during grid generation for PLZ {self.plz}: {e}")
            self.logger.info(f"Skipped PLZ {self.plz} due to generation error.")
            self.dbc.conn.rollback()  # rollback the transaction
            self.dbc.delete_plz_from_sample_set_table(str(CLASSIFICATION_VERSION), self.plz)  # delete from sample set
            return

        self.dbc.drop_temp_tables()  # drop temp tables
        self.dbc.commit_changes()    # commit the changes to the database

        print('-------------------- end', self.plz, '-----------------------------')
