import math
import json
import statistics
from math import radians

import geopandas as gpd
import networkx as nx
import pandapower as pp
from sklearn.metrics.pairwise import haversine_distances
import pandapower.topology as top

import src.database.database_client as dbc
from src.config_loader import *
from src import utils


class ParameterCalculator:
    """Calculate and save grid parameters for a postal code area."""

    def __init__(self):
        self.dbc = dbc.DatabaseClient()
        self.version_id = VERSION_ID
        self.plz = None
        self.bcid = None
        self.kcid = None
        self.no_connection_buses = None
        self.no_branches = None
        self.no_house_connections = None
        self.no_house_connections_per_branch = None
        self.no_households = None
        self.no_household_equ = None
        self.no_households_per_branch = None
        self.max_no_of_households_of_a_branch = None
        self.house_distance_km = None
        self.transformer_mva = None
        self.osm_trafo = None
        self.max_trafo_dis = None
        self.avg_trafo_dis = None
        self.cable_length_km = None
        self.cable_len_per_house = None
        self.max_power_mw = None
        self.simultaneous_peak_load_mw = None
        self.resistance = None
        self.reactance = None
        self.ratio = None
        self.vsw_per_branch = None
        self.max_vsw_of_a_branch = None



    def calc_parameters_per_plz(self, plz):
        try:
            self.dbc.logger.info("Start basic result analysis")
            self.analyse_basic_parameters_per_plz(plz)
            self.dbc.logger.info("Start cable counting")
            self.analyse_cables_per_plz(plz)
            self.dbc.logger.info("Start per trafo analysis")
            self.analyse_trafo_parameters_per_plz(plz)
            self.dbc.logger.info("Result analysis finished")
            self.dbc.conn.commit()
        except Exception as e:
            self.dbc.logger.error(f"Error during analysis for PLZ {plz}: {e}")
            self.dbc.logger.info(f"Skipped PLZ {plz} due to analysis error.")
            self.dbc.delete_plz_from_sample_set_table(str(CLASSIFICATION_VERSION),plz)  # delete from sample set

    def calc_parameters_per_grid(self, plz: int) -> None:
        """Calculate parameters for all grids of a PLZ."""

        plz = str(plz)
        parameter_count = self.dbc.count_clustering_parameters(plz=plz)
        if parameter_count > 0:
            print(f"The parameters for the grids of postcode area {plz} and version {VERSION_ID} "
                  f"have already been calculated.")
            return

        cluster_list = self.dbc.get_list_from_plz(plz)
        for kcid, bcid in cluster_list:
            print(bcid, kcid)
            self.calc_grid_parameters(plz, bcid, kcid)

    def calc_grid_parameters(self, plz: int, bcid: int, kcid: int) -> None:
        """Calculate parameters for a single grid and save them."""

        self.plz = plz
        self.bcid = bcid
        self.kcid = kcid
        self.osm_trafo = self.has_osm_trafo()

        net = self.dbc.read_net(self.plz, self.kcid, self.bcid)
        self.compute_parameters(net)

        params = {"version_id": self.version_id, "plz": self.plz, "bcid": self.bcid, "kcid": self.kcid,
            "no_connection_buses": int(self.no_connection_buses), "no_branches": int(self.no_branches),
            "no_house_connections": int(self.no_house_connections),
            "no_house_connections_per_branch": float(self.no_house_connections_per_branch),
            "no_households": int(self.no_households), "no_household_equ": float(self.no_household_equ),
            "no_households_per_branch": float(self.no_households_per_branch),
            "max_no_of_households_of_a_branch": float(self.max_no_of_households_of_a_branch),
            "house_distance_km": float(self.house_distance_km), "transformer_mva": float(self.transformer_mva),
            "osm_trafo": bool(self.osm_trafo), "max_trafo_dis": float(self.max_trafo_dis),
            "avg_trafo_dis": float(self.avg_trafo_dis), "cable_length_km": float(self.cable_length_km),
            "cable_len_per_house": float(self.cable_len_per_house), "max_power_mw": float(self.max_power_mw),
            "simultaneous_peak_load_mw": float(self.simultaneous_peak_load_mw), "resistance": float(self.resistance),
            "reactance": float(self.reactance), "ratio": float(self.ratio),
            "vsw_per_branch": float(self.vsw_per_branch), "max_vsw_of_a_branch": float(self.max_vsw_of_a_branch), }

        self.dbc.insert_clustering_parameters(params)

    def compute_parameters(self, net: pp.pandapowerNet) -> None:
        """Compute all grid parameters."""

        self.no_house_connections = self.get_no_of_buses(net, "Consumer Nodebus")
        self.no_connection_buses = self.get_no_of_buses(net, "Connection Nodebus")
        self.no_households = self.get_no_households(net)
        self.max_power_mw = self.get_max_power(net)

        self.no_household_equ = self.max_power_mw * 1000.0 / PEAK_LOAD_HOUSEHOLD
        self.cable_length_km = self.get_cable_length(net)
        self.cable_len_per_house = self.cable_length_km / self.no_house_connections

        G = pp.topology.create_nxgraph(net)

        self.no_branches = self.get_no_branches(G, net)
        self.avg_trafo_dis, self.max_trafo_dis = self.get_distances_in_graph(net, G)
        self.no_house_connections_per_branch = self.no_house_connections / self.no_branches
        self.no_households_per_branch = self.max_power_mw * 1000.0 / (PEAK_LOAD_HOUSEHOLD * self.no_branches)

        self.transformer_mva = self.get_trafo_power(net)
        self.house_distance_km = self.calc_avg_house_distance(net)
        self.simultaneous_peak_load_mw = self.get_simultaneous_peak_load()
        (self.max_no_of_households_of_a_branch, self.resistance, self.reactance, self.ratio,
         self.max_vsw_of_a_branch,) = self.calc_resistance(net, G)

        self.vsw_per_branch = self.resistance / self.no_branches

    def get_parameters_as_dataframe(self) -> pd.DataFrame:
        params = [value for key, value in vars(self).items() if key != "dbc"]
        df_parameters = pd.DataFrame([params], columns=CLUSTERING_PARAMETERS)
        return df_parameters

    def get_simultaneous_peak_load(self) -> float:
        data_list, _, _ = self.dbc.read_per_trafo_dict(self.plz)
        transformer_type_str = str(int(self.transformer_mva * 1000))
        max_trafo_distance_list = data_list[3][transformer_type_str]
        if self.max_trafo_dis * 1000 in max_trafo_distance_list:
            sim_load_index = max_trafo_distance_list.index(self.max_trafo_dis * 1000)
            simultaneous_peak_load_mw = data_list[2][transformer_type_str][sim_load_index] / 1000
            return simultaneous_peak_load_mw
        return None

    def has_osm_trafo(self) -> bool:
        return self.bcid < 0

    def print_grid_parameters(self) -> None:
        params = [value for key, value in vars(self).items() if key != "dbc"]
        print(*params)


    def get_max_power(self, pandapower_net: pp.pandapowerNet) -> float:
        df_load = pandapower_net.load
        return df_load["max_p_mw"].sum()


    def get_no_households(self, pandapower_net: pp.pandapowerNet) -> int:
        df_load = pandapower_net.load
        return len(df_load["name"])


    def get_no_of_buses(self, pandapower_net: pp.pandapowerNet, bus_description: str) -> int:
        df_bus = pandapower_net.bus
        df_bus["type_bus"] = df_bus["name"].str.contains(bus_description)
        return df_bus["type_bus"].sum()


    def get_cable_length(self, pandapower_net: pp.pandapowerNet) -> float:
        df_line = pandapower_net.line
        return df_line["length_km"].sum()


    def calc_avg_house_distance(self, pandapower_net: pp.pandapowerNet) -> float:
        bus_geo = pandapower_net.bus_geodata
        bus_geo = gpd.GeoDataFrame(bus_geo, geometry=gpd.points_from_xy(bus_geo["x"], bus_geo["y"]))
        bus = pandapower_net.bus
        bus_geo = bus_geo.merge(bus, left_index=True, right_index=True)
        bus_geo["consumer_bus"] = bus_geo["name"].str.contains("Consumer Nodebus")
        bus_geo = bus_geo[bus_geo["consumer_bus"]]

        list_pt = []
        for pt in bus_geo["geometry"]:
            new_pt = [radians(pt.x), radians(pt.y)]
            list_pt.append(new_pt)

        dis_mat = haversine_distances(list_pt, list_pt)
        dis_mat = dis_mat * 6371000 / 1000
        df_distances = pd.DataFrame(dis_mat)
        list_avg_dis4pts = []

        for column in df_distances:
            smallest = df_distances[column].nsmallest(5)
            avg = smallest.sum() / 4
            list_avg_dis4pts.append(avg)

        median_dis = statistics.median(list_avg_dis4pts)
        return median_dis


    def get_root(self, pandapower_net: pp.pandapowerNet):
        root = pandapower_net.bus
        root["LV_bus"] = root["name"].str.contains("LVbus")
        root = root[root["LV_bus"]]
        root = list(root.index)[0]
        return root


    def get_no_branches(self, networkx_graph: nx.Graph, pandapower_net: pp.pandapowerNet) -> int:
        root = self.get_root(pandapower_net)
        return networkx_graph.degree(root) - 1


    def get_distances_in_graph(self, pandapower_net: pp.pandapowerNet, networkx_graph: nx.Graph) -> tuple[float, float]:
        root = self.get_root(pandapower_net)
        leaves = pandapower_net.bus
        leaves["consumer_bus"] = leaves["name"].str.contains("Consumer Nodebus")
        leaves = list(leaves[leaves["consumer_bus"]].index)
        no_leaves = len(leaves)

        path_length_to_leaves = []
        for leaf in leaves:
            weighted_length = nx.dijkstra_path_length(networkx_graph, root, leaf)
            path_length_to_leaves.append(weighted_length)

        max_path_length = max(path_length_to_leaves)
        avg_path_length = sum(path_length_to_leaves) / no_leaves

        return avg_path_length, max_path_length


    def get_trafo_power(self, pandapower_net: pp.pandapowerNet) -> float:
        df_trafo = pandapower_net.trafo.sn_mva
        return df_trafo.iloc[0]


    def calc_resistance(self, pandapower_net: pp.pandapowerNet, networkx_graph: nx.Graph) -> tuple[
        float, float, float, float, float]:
        df_load = pandapower_net.load
        df_vsw = df_load.groupby("bus")["max_p_mw"].sum() * 1000.0 / PEAK_LOAD_HOUSEHOLD
        df_vsw = df_vsw.to_frame().reset_index().rename(
            columns={"bus": "house_connection", "max_p_mw": "household_equivalents"})

        df_line = self.calculate_line_with_sim_factor(pandapower_net, networkx_graph)
        root = self.get_root(pandapower_net)

        df_vsw["path"] = ""
        for index, row in df_vsw.iterrows():
            df_vsw.at[index, "path"] = nx.shortest_path(networkx_graph, source=root,
                                                        target=df_vsw.at[index, "house_connection"])

        df_vsw["branch"] = ""
        for branch in networkx_graph.edges(root):
            for index, row in df_vsw.iterrows():
                if branch[1] in row["path"]:
                    df_vsw.at[index, "branch"] = branch

        max_no_of_households_of_a_branch = df_vsw.groupby("branch")["household_equivalents"].sum().max()

        df_vsw["resistance"] = ""
        df_vsw["resistance_sections"] = ""
        df_vsw["reactance"] = ""
        df_vsw["reactance_sections"] = ""
        for index, row in df_vsw.iterrows():
            path_list = df_vsw.at[index, "path"]
            length = len(path_list)
            no_load = df_vsw.at[index, "household_equivalents"]
            resistance_list = []
            reactance_list = []
            for i in range(length - 1):
                start_node = path_list[i]
                end_node = path_list[i + 1]
                line = df_line[df_line["from_bus"] == start_node]
                line = line[line["to_bus"] == end_node].head(1)
                length_km = line["length_km"]
                r_ohm_per_km = line["r_ohm_per_km"]
                x_ohm_per_km = line["x_ohm_per_km"]
                sim_factor = line["sim_factor_cumulated"]
                resistance_of_cable_section = no_load * length_km * r_ohm_per_km * sim_factor
                resistance_list.append(resistance_of_cable_section)
                reactance_of_cable_section = no_load * length_km * x_ohm_per_km
                reactance_list.append(reactance_of_cable_section)
            df_vsw.at[index, "resistance"] = math.fsum(resistance_list)
            df_vsw.at[index, "resistance_sections"] = resistance_list
            df_vsw.at[index, "reactance"] = math.fsum(reactance_list)
            df_vsw.at[index, "reactance_sections"] = reactance_list

        resistance = df_vsw["resistance"].sum()
        reactance = df_vsw["reactance"].sum()
        ratio = resistance / reactance
        max_vsw_of_a_branch = df_vsw.groupby("branch")["resistance"].sum().max()

        return max_no_of_households_of_a_branch, resistance, reactance, ratio, max_vsw_of_a_branch

    def calculate_line_with_sim_factor(self, pandapower_net, networkx_graph) -> pd.DataFrame:
        """calculate the sim factor for each line segment"""
        df_sim_factor_definitions = pd.DataFrame.from_dict(SIM_FACTOR, orient='index')
        df_sim_factor_definitions.reset_index(inplace=True)
        df_sim_factor_definitions.columns = ['description', 'sim_factor']

        # The idea is to add to each line a new attributes: these are needed
        # to calculate the simultaneity factor for each line (cable segment).
        # The simultaneity factor is needed to calculate the vsw

        net_line_with_sim_factor = pandapower_net.line
        net_line_with_sim_factor['sim_factor_cumulated'] = ''
        net_line_with_sim_factor['sim_load'] = ''
        net_line_with_sim_factor['no_commercial'] = ''
        net_line_with_sim_factor['load_commercial_mw'] = ''
        net_line_with_sim_factor['no_public'] = ''
        net_line_with_sim_factor['load_public_mw'] = ''
        net_line_with_sim_factor['no_residential'] = ''
        net_line_with_sim_factor['load_residential_mw'] = ''
        net_line_with_sim_factor.drop(['c_nf_per_km', 'g_us_per_km', 'max_i_ka', 'df', 'type', 'in_service'], axis=1,
                                      inplace=True)
        net_line_with_sim_factor = net_line_with_sim_factor.drop_duplicates()

        # First we calculate the sim factor for the consumers/ consumer buses

        level1 = pd.merge(left=pandapower_net.load, left_on='bus', right=pandapower_net.bus,
                          right_on=pandapower_net.bus.index)
        level1.replace(['MFH', 'SFH', 'AB', 'TH'], 'Residential', inplace=True)

        load_value = level1.groupby(['bus', 'zone'])['max_p_mw'].sum()
        load_value = pd.DataFrame(load_value)
        load_value = load_value.reset_index()

        load_count = level1.groupby(['bus', 'zone'])['name_x'].count()
        load_count = pd.DataFrame(load_count)
        load_count = load_count.reset_index()
        load_count = load_count.rename(columns={'name_x': 'count'})

        load_count = pd.merge(left=load_count, left_on='bus', right=load_value, right_on='bus')
        load_count.drop(['zone_y'], axis=1, inplace=True)

        load_count_cat = pd.merge(left=load_count, left_on='zone_x', right=df_sim_factor_definitions,
                                  right_on='description')

        load_count_cat = load_count_cat.assign(
            sim_factor_level1=lambda x: utils.oneSimultaneousLoad(installed_power=1, load_count=x['count'],
                                                            sim_factor=x['sim_factor']))

        load_count_cat = load_count_cat.assign(sim_load_level1=lambda x: x['max_p_mw'] * x['sim_factor_level1'])

        # we can now enter these values in our lines table

        for index, row in load_count_cat.iterrows():
            bus = row['bus']
            index_line = net_line_with_sim_factor.index[net_line_with_sim_factor['to_bus'] == bus].tolist()
            net_line_with_sim_factor.at[index_line[0], 'sim_factor_cumulated'] = row['sim_factor_level1']
            net_line_with_sim_factor.at[index_line[0], 'sim_load'] = row['sim_load_level1']
            if row['description'] == 'Commercial':
                net_line_with_sim_factor.at[index_line[0], 'no_commercial'] = row['count']
                net_line_with_sim_factor.at[index_line[0], 'load_commercial_mw'] = row['max_p_mw']
                net_line_with_sim_factor.at[index_line[0], 'no_public'] = 0
                net_line_with_sim_factor.at[index_line[0], 'load_public_mw'] = 0
                net_line_with_sim_factor.at[index_line[0], 'no_residential'] = 0
                net_line_with_sim_factor.at[index_line[0], 'load_residential_mw'] = 0
            elif row['description'] == 'Public':
                net_line_with_sim_factor.at[index_line[0], 'no_commercial'] = 0
                net_line_with_sim_factor.at[index_line[0], 'load_commercial_mw'] = 0
                net_line_with_sim_factor.at[index_line[0], 'no_public'] = row['count']
                net_line_with_sim_factor.at[index_line[0], 'load_public_mw'] = row['max_p_mw']
                net_line_with_sim_factor.at[index_line[0], 'no_residential'] = 0
                net_line_with_sim_factor.at[index_line[0], 'load_residential_mw'] = 0
            elif row['description'] == 'Residential':
                net_line_with_sim_factor.at[index_line[0], 'no_commercial'] = 0
                net_line_with_sim_factor.at[index_line[0], 'load_commercial_mw'] = 0
                net_line_with_sim_factor.at[index_line[0], 'no_public'] = 0
                net_line_with_sim_factor.at[index_line[0], 'load_public_mw'] = 0
                net_line_with_sim_factor.at[index_line[0], 'no_residential'] = row['count']
                net_line_with_sim_factor.at[index_line[0], 'load_residential_mw'] = row['max_p_mw']

        # lets work on the connection nodebuses and their sim factor

        connection_bus = pandapower_net.bus
        connection_bus['connection_bus'] = connection_bus['name'].str.contains("Connection Nodebus")
        connection_bus = connection_bus[connection_bus['connection_bus']]
        connection_bus = connection_bus.index
        connection_bus = list(connection_bus)

        # we sort them by their distance ( number of edges that need to be passed ) along the graph to the trafo.
        df_connection_bus = pd.DataFrame(connection_bus, columns=['bus'])
        df_connection_bus['source'] = 0

        len_path_list = []
        for index, row in df_connection_bus.iterrows():
            length = nx.shortest_path_length(networkx_graph, source=row['source'], target=row['bus'])
            len_path_list.append(length)
        df_connection_bus['len_to_trafo_in_graph'] = len_path_list
        # The connection nodebuses furthest away need to be adressed first.
        df_connection_bus = df_connection_bus.sort_values(by=['len_to_trafo_in_graph'], ascending=False)

        # turn it into a loop
        for index, row in df_connection_bus.iterrows():
            furthest_connection_bus = row['bus']
            connected_downstream = net_line_with_sim_factor[
                net_line_with_sim_factor['from_bus'] == furthest_connection_bus]
            # upstream: towards the trafo
            connected_upstream = net_line_with_sim_factor[net_line_with_sim_factor['to_bus'] == furthest_connection_bus]
            upstream_index = connected_upstream.index
            net_line_with_sim_factor.at[upstream_index[0], 'no_commercial'] = connected_downstream[
                'no_commercial'].sum()
            net_line_with_sim_factor.at[upstream_index[0], 'load_commercial_mw'] = connected_downstream[
                'load_commercial_mw'].sum()
            net_line_with_sim_factor.at[upstream_index[0], 'no_public'] = connected_downstream['no_public'].sum()
            net_line_with_sim_factor.at[upstream_index[0], 'load_public_mw'] = connected_downstream[
                'load_public_mw'].sum()
            net_line_with_sim_factor.at[upstream_index[0], 'no_residential'] = connected_downstream[
                'no_residential'].sum()
            net_line_with_sim_factor.at[upstream_index[0], 'load_residential_mw'] = connected_downstream[
                'load_residential_mw'].sum()

            load_commercial = utils.oneSimultaneousLoad(
                installed_power=net_line_with_sim_factor.at[upstream_index[0], 'load_commercial_mw'],
                load_count=net_line_with_sim_factor.at[upstream_index[0], 'no_commercial'],
                sim_factor=SIM_FACTOR['Commercial'])

            load_public = utils.oneSimultaneousLoad(
                installed_power=net_line_with_sim_factor.at[upstream_index[0], 'load_public_mw'],
                load_count=net_line_with_sim_factor.at[upstream_index[0], 'no_public'], sim_factor=SIM_FACTOR['Public'])

            load_residential = utils.oneSimultaneousLoad(
                installed_power=net_line_with_sim_factor.at[upstream_index[0], 'load_residential_mw'],
                load_count=net_line_with_sim_factor.at[upstream_index[0], 'no_residential'],
                sim_factor=SIM_FACTOR['Residential'])

            net_line_with_sim_factor.at[
                upstream_index[0], 'sim_load'] = load_commercial + load_public + load_residential

            peak_load_all_consumer_types = net_line_with_sim_factor.at[upstream_index[0], 'load_commercial_mw'] + \
                                           net_line_with_sim_factor.at[upstream_index[0], 'load_public_mw'] + \
                                           net_line_with_sim_factor.at[upstream_index[0], 'load_residential_mw']
            if peak_load_all_consumer_types == 0:
                net_line_with_sim_factor.at[
                    upstream_index[0], 'sim_factor_cumulated'] = 0  # print('Connection nodebus error')
            else:
                net_line_with_sim_factor.at[upstream_index[0], 'sim_factor_cumulated'] = (
                            net_line_with_sim_factor.at[upstream_index[0], 'sim_load'] / peak_load_all_consumer_types)

        return net_line_with_sim_factor

    def analyse_basic_parameters_per_plz(self, plz: int):
        cluster_list = self.dbc.get_list_from_plz(plz)
        count = len(cluster_list)
        time = 0
        percent = 0

        load_count_dict = {}
        bus_count_dict = {}
        cable_length_dict = {}
        trafo_dict = {}
        self.dbc.logger.debug("start basic parameter counting")
        for kcid, bcid in cluster_list:
            load_count = 0
            bus_list = []
            try:
                net = self.dbc.read_net(plz, kcid, bcid)
            except Exception as e:
                self.dbc.logger.warning(f" local network {kcid},{bcid} is problematic")
                raise e
            else:
                for row in net.load[["name", "bus"]].itertuples():
                    load_count += 1
                    bus_list.append(row.bus)
                bus_list = list(set(bus_list))
                bus_count = len(bus_list)
                cable_length = net.line["length_km"].sum()

                for row in net.trafo[["sn_mva", "lv_bus"]].itertuples():
                    capacity = round(row.sn_mva * 1e3)

                    if capacity in trafo_dict:
                        trafo_dict[capacity] += 1

                        load_count_dict[capacity].append(load_count)
                        bus_count_dict[capacity].append(bus_count)
                        cable_length_dict[capacity].append(cable_length)

                    else:
                        trafo_dict[capacity] = 1

                        load_count_dict[capacity] = [load_count]
                        bus_count_dict[capacity] = [bus_count]
                        cable_length_dict[capacity] = [cable_length]

            time += 1
            if time / count >= 0.1:
                percent += 10
                self.dbc.logger.info(f"{percent} percent finished")
                time = 0
        self.dbc.logger.info("analyse_basic_parameters finished.")
        trafo_string = json.dumps(trafo_dict)
        load_count_string = json.dumps(load_count_dict)
        bus_count_string = json.dumps(bus_count_dict)

        self.dbc.insert_plz_parameters(plz, trafo_string, load_count_string, bus_count_string)

    def analyse_cables_per_plz(self, plz: int):
        cluster_list = self.dbc.get_list_from_plz(plz)
        count = len(cluster_list)
        time = 0
        percent = 0

        # distributed according to cross_section
        cable_length_dict = {}
        for kcid, bcid in cluster_list:
            try:
                net = self.dbc.read_net(plz, kcid, bcid)
            except Exception as e:
                self.dbc.logger.debug(f" local network {kcid},{bcid} is problematic")
                raise e
            else:
                cable_df = net.line[net.line["in_service"] == True]

                cable_type = pd.unique(cable_df["std_type"]).tolist()
                for type in cable_type:

                    if type in cable_length_dict:
                        cable_length_dict[type] += (cable_df[cable_df["std_type"] == type]["parallel"] *
                                                    cable_df[cable_df["std_type"] == type]["length_km"]).sum()

                    else:
                        cable_length_dict[type] = (cable_df[cable_df["std_type"] == type]["parallel"] *
                                                   cable_df[cable_df["std_type"] == type]["length_km"]).sum()
            time += 1
            if time / count >= 0.1:
                percent += 10
                self.dbc.logger.info(f"{percent} % processed")
                time = 0
        self.dbc.logger.info("analyse_cables finished.")
        cable_length_string = json.dumps(cable_length_dict)
        self.dbc.insert_cable_length(plz, cable_length_string)

    def analyse_trafo_parameters_per_plz(self, plz: int):
        cluster_list = self.dbc.get_list_from_plz(plz)
        count = len(cluster_list)
        time = 0
        percent = 0

        trafo_load_dict = {}
        trafo_max_distance_dict = {}
        trafo_avg_distance_dict = {}

        for kcid, bcid in cluster_list:
            try:
                net = self.dbc.read_net(plz, kcid, bcid)
            except Exception as e:
                self.dbc.logger.warning(f" local network {kcid},{bcid} is problematic")
                raise e
            else:
                trafo_sizes = net.trafo["sn_mva"].tolist()[0]

                load_bus = pd.unique(net.load["bus"]).tolist()

                top.create_nxgraph(net, respect_switches=False)
                trafo_distance_to_buses = (
                    top.calc_distance_to_bus(net, net.trafo["lv_bus"].tolist()[0], weight="weight",
                                             respect_switches=False, ).loc[load_bus].tolist())

                # calculate total sim_peak_load
                residential_bus_index = net.bus[~net.bus["zone"].isin(["Commercial", "Public"])].index.tolist()
                commercial_bus_index = net.bus[net.bus["zone"] == "Commercial"].index.tolist()
                public_bus_index = net.bus[net.bus["zone"] == "Public"].index.tolist()

                residential_house_num = net.load[net.load["bus"].isin(residential_bus_index)].shape[0]
                public_house_num = net.load[net.load["bus"].isin(public_bus_index)].shape[0]
                commercial_house_num = net.load[net.load["bus"].isin(commercial_bus_index)].shape[0]

                residential_sum_load = (net.load[net.load["bus"].isin(residential_bus_index)]["max_p_mw"].sum() * 1e3)
                public_sum_load = (net.load[net.load["bus"].isin(public_bus_index)]["max_p_mw"].sum() * 1e3)
                commercial_sum_load = (net.load[net.load["bus"].isin(commercial_bus_index)]["max_p_mw"].sum() * 1e3)

                sim_peak_load = 0
                for building_type, sum_load, house_num in zip(["Residential", "Public", "Commercial"],
                                                              [residential_sum_load, public_sum_load,
                                                               commercial_sum_load],
                                                              [residential_house_num, public_house_num,
                                                               commercial_house_num], ):
                    if house_num:
                        sim_peak_load += utils.oneSimultaneousLoad(installed_power=sum_load, load_count=house_num,
                                                                   sim_factor=SIM_FACTOR[building_type], )

                avg_distance = (sum(trafo_distance_to_buses) / len(trafo_distance_to_buses)) * 1e3
                max_distance = max(trafo_distance_to_buses) * 1e3

                trafo_size = round(trafo_sizes * 1e3)

                if trafo_size in trafo_load_dict:
                    trafo_load_dict[trafo_size].append(sim_peak_load)

                    trafo_max_distance_dict[trafo_size].append(max_distance)

                    trafo_avg_distance_dict[trafo_size].append(avg_distance)

                else:
                    trafo_load_dict[trafo_size] = [sim_peak_load]
                    trafo_max_distance_dict[trafo_size] = [max_distance]
                    trafo_avg_distance_dict[trafo_size] = [avg_distance]

            time += 1
            if time / count >= 0.1:
                percent += 10
                self.dbc.logger.info(f"{percent} % processed")
                time = 0
        self.dbc.logger.info("analyse_per_trafo_parameters finished.")

        trafo_load_string = json.dumps(trafo_load_dict)
        trafo_max_distance_string = json.dumps(trafo_max_distance_dict)
        trafo_avg_distance_string = json.dumps(trafo_avg_distance_dict)
        self.dbc.insert_trafo_parameters(plz, trafo_load_string, trafo_max_distance_string, trafo_avg_distance_string)
