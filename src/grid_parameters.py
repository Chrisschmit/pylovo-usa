import math
import statistics
from math import radians
import pandas as pd
import geopandas as gpd
import networkx as nx
import pandapower as pp
from sklearn.metrics.pairwise import haversine_distances

from src.config_loader import SIM_FACTOR
from src.utils import oneSimultaneousLoad

from src.classification.parameter_calculation.sim_factor import calculate_line_with_sim_factor
from src.config_loader import *


class GridParameters:
    """A set of parameters and functions to calculate and save grid parameters
    for a plz that are generated with a certain VERSION_ID
    """

    def __init__(self, plz, bcid, kcid, databaseClient):
        self.dbc = databaseClient
        self.version_id = VERSION_ID
        self.plz = plz
        self.bcid = bcid
        self.kcid = kcid
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

    def calc_plz_parameters(self) -> None:
        """calculate parameters of each grid
        save results to table 'clustering_parameters' on database
        """
        self.osm_trafo = self.has_osm_trafo()

        # load network
        net = self.dbc.read_net(self.plz, self.kcid, self.bcid)

        # calculate parameters
        self.compute_parameters(net)

        # parameters to database
        self.write_parameters_to_db()

    def compute_parameters(self, net):
        """the individual parameters are computed in this function
        """

        # calculate parameters
        self.no_house_connections = get_no_of_buses(pandapower_net=net,
                                                    bus_description="Consumer Nodebus")  # todo: Benennung Knoten
        self.no_connection_buses = get_no_of_buses(pandapower_net=net,
                                                   bus_description="Connection Nodebus")  # todo: Benennung Knoten
        self.no_households = get_no_households(pandapower_net=net)
        self.max_power_mw = get_max_power(pandapower_net=net)  # todo: more general

        # no household expressed as a multiple of a household load (of 30 kW)
        self.no_household_equ = self.max_power_mw * 1000.0 / PEAK_LOAD_HOUSEHOLD  # todo: Abhängigkeit beachten
        self.cable_length_km = get_cable_length(pandapower_net=net)

        # cable length per house connection
        self.cable_len_per_house = self.cable_length_km / self.no_house_connections  # todo Abhängikeit beachten

        # convert pandapower net into graph
        G = pp.topology.create_nxgraph(net)

        self.no_branches = get_no_branches(pandapower_net=net, networkx_graph=G)  # todo: Benennung LV_Bus in get roots

        # distances in the graph:
        self.avg_trafo_dis, self.max_trafo_dis = get_distances_in_graph(pandapower_net=net,
                                                                        networkx_graph=G)  # todo: Benennung LV_Bus in get roots & Knoten

        # number of house connections per branch
        self.no_house_connections_per_branch = self.no_house_connections / self.no_branches

        # number of households (equivalents) per branch = total power divided by power of a single household
        self.no_households_per_branch = self.max_power_mw * 1000.0 / (PEAK_LOAD_HOUSEHOLD * self.no_branches)

        self.transformer_mva = get_trafo_power(pandapower_net=net)

        # average house distance
        self.house_distance_km = calc_avg_house_distance(pandapower_net=net)

        # simultaneous peak load
        self.simultaneous_peak_load_mw = self.get_simultaneous_peak_load()  # todo: geht wsl nicht wegen Kategorisierung MFH, SFH ...

        # electrical resistance (Verbrauchersummenwiderstand, vsw)
        (self.max_no_of_households_of_a_branch,
         self.resistance,
         self.reactance,
         self.ratio,
         self.max_vsw_of_a_branch) = calc_resistance(pandapower_net=net, networkx_graph=G)

        # vsw per branch (of capacitive resistance)
        self.vsw_per_branch = self.resistance / self.no_branches

    def get_parameters_as_dataframe(self) -> pd.DataFrame:
        """get grid parameters. Order of parameters is defined in CLUSTERING_PARAMETERS should equal database
        table clustering_parameters

        :return: table of grid parameters
        :rtype: pd.Dataframe
        """
        params = [value for key, value in vars(self).items() if key != 'dbc']
        df_parameters = pd.DataFrame([params], columns=CLUSTERING_PARAMETERS)
        return df_parameters

    def get_simultaneous_peak_load(self) -> float:
        # in data list at index 2 the simultaneous
        # peak load is saved grouped by transformer size
        # see also compare_grid_parameters_db.iynb for more details
        data_list, data_labels, trafo_dict = self.dbc.read_per_trafo_dict(self.plz)
        transformer_type_str = str(int(self.transformer_mva * 1000))
        max_trafo_distance_list = data_list[3][transformer_type_str]
        if self.max_trafo_dis * 1000 in max_trafo_distance_list:
            sim_load_index = max_trafo_distance_list.index(self.max_trafo_dis * 1000)
            simultaneous_peak_load_mw = data_list[2][transformer_type_str][sim_load_index] / 1000

            return simultaneous_peak_load_mw
        else:
            return None

    def has_osm_trafo(self) -> bool:
        # trafo from OSM: Boolean, OSM trafos have negative bcid
        return self.bcid < 0

    def print_grid_parameters_types(self) -> None:
        param_types = [type(value) for key, value in vars(self).items() if key != 'dbc']
        print(*param_types)

    def print_grid_parameters(self) -> None:
        params = [value for key, value in vars(self).items() if key != 'dbc']
        print(*params)

    def write_parameters_to_db(self):
        """writes parameters to database"""
        query = """INSERT INTO clustering_parameters (
                   grid_result_id,
                   no_connection_buses,
                   no_branches,
                   no_house_connections,
                   no_house_connections_per_branch,
                   no_households,
                   no_household_equ,
                   no_households_per_branch,
                   max_no_of_households_of_a_branch,
                   house_distance_km,
                   transformer_mva,
                   osm_trafo,
                   max_trafo_dis,
                   avg_trafo_dis,
                   cable_length_km,
                   cable_len_per_house,
                   max_power_mw,
                   simultaneous_peak_load_mw,
                   resistance,
                   reactance,
                   ratio,
                   vsw_per_branch,
                   max_vsw_of_a_branch
                  )
                  VALUES (
                  (SELECT grid_result_id FROM grid_result WHERE version_id = %(version_id)s AND plz = %(plz)s AND bcid = %(bcid)s AND kcid = %(kcid)s),
                  %(no_connection_buses)s,
                  %(no_branches)s,
                  %(no_house_connections)s,
                  %(no_house_connections_per_branch)s,
                  %(no_households)s,
                  %(no_household_equ)s,
                  %(no_households_per_branch)s,
                  %(max_no_of_households_of_a_branch)s,
                  %(house_distance_km)s,
                  %(transformer_mva)s,
                  %(osm_trafo)s,
                  %(max_trafo_dis)s,
                  %(avg_trafo_dis)s,
                  %(cable_length_km)s,
                  %(cable_len_per_house)s,
                  %(max_power_mw)s,
                  %(simultaneous_peak_load_mw)s,
                  %(resistance)s,
                  %(reactance)s,
                  %(ratio)s,
                  %(vsw_per_branch)s,
                  %(max_vsw_of_a_branch)s);"""

        self.dbc.cur.execute(query, {
            "version_id": self.version_id,
            "plz": self.plz,
            "bcid": self.bcid,
            "kcid": self.kcid,
            "no_connection_buses": int(self.no_connection_buses),
            "no_branches": int(self.no_branches),
            "no_house_connections": int(self.no_house_connections),
            "no_house_connections_per_branch": float(self.no_house_connections_per_branch),
            "no_households": int(self.no_households),
            "no_household_equ": float(self.no_household_equ),
            "no_households_per_branch": float(self.no_households_per_branch),
            "max_no_of_households_of_a_branch": float(self.max_no_of_households_of_a_branch),
            "house_distance_km": float(self.house_distance_km),
            "transformer_mva": float(self.transformer_mva),
            "osm_trafo": bool(self.osm_trafo),
            "max_trafo_dis": float(self.max_trafo_dis),
            "avg_trafo_dis": float(self.avg_trafo_dis),
            "cable_length_km": float(self.cable_length_km),
            "cable_len_per_house": float(self.cable_len_per_house),
            "max_power_mw": float(self.max_power_mw),
            "simultaneous_peak_load_mw": float(self.simultaneous_peak_load_mw),
            "resistance": float(self.resistance),
            "reactance": float(self.reactance),
            "ratio": float(self.ratio),
            "vsw_per_branch": float(self.vsw_per_branch),
            "max_vsw_of_a_branch": float(self.max_vsw_of_a_branch)
        })
        print(self.dbc.cur.statusmessage)
        self.dbc.conn.commit()


def get_max_power(pandapower_net) -> float:
    # cumulated maximum power
    df_load = pandapower_net.load
    max_power_mw = df_load['max_p_mw'].sum()
    return max_power_mw


def get_no_households(pandapower_net) -> int:
    # get number of households
    df_load = pandapower_net.load
    no_households = len(df_load['name'])
    return no_households


def get_no_of_buses(pandapower_net, bus_description: str) -> int:
    # gets number of buses of a type that contains string bus_description
    df_bus = pandapower_net.bus
    df_bus['type_bus'] = df_bus['name'].str.contains(bus_description)
    no_type_bus = df_bus['type_bus'].sum()
    return no_type_bus


def get_cable_length(pandapower_net) -> float:
    # cable length
    df_line = pandapower_net.line
    cable_length = df_line['length_km'].sum()
    return cable_length


def calc_avg_house_distance(pandapower_net) -> float:
    """
    calculates the median distance of four houses closest together for a pandapower net
    Remark: haversine_distances are computed between latitude and longitude coordinates to get distance in km
    According to Kerber 2010 p. 45 the average house distance is the average distance of a house to its four nearest
    neighbors
    """
    # bus information is extracted from network
    bus_geo = pandapower_net.bus_geodata
    bus_geo = gpd.GeoDataFrame(bus_geo, geometry=gpd.points_from_xy(bus_geo['x'], bus_geo['y']))
    bus = pandapower_net.bus
    bus_geo = bus_geo.merge(bus, left_index=True, right_index=True)
    bus_geo['consumer_bus'] = bus_geo['name'].str.contains("Consumer Nodebus")
    bus_geo = bus_geo[bus_geo['consumer_bus']]  # keep only consumer nodebuses

    # points are written into array
    list_pt = []
    for pt in bus_geo['geometry']:
        new_pt = [radians(pt.x), radians(pt.y)]
        list_pt.append(new_pt)

    # distance matrix is generated with haversine formula
    dis_mat = haversine_distances(list_pt, list_pt)
    dis_mat = dis_mat * 6371000 / 1000  # multiply by Earth radius to get kilometers
    df_distances = pd.DataFrame(dis_mat)
    list_avg_dis4pts = []

    # average distance of four nearest houses
    for column in df_distances:
        smallest = df_distances[column].nsmallest(5)
        avg = smallest.sum() / 4
        list_avg_dis4pts.append(avg)

    # median of average distance for network
    median_dis = statistics.median(list_avg_dis4pts)
    return median_dis


    def get_root(pandapower_net):
        # the trafo is the root of the graph
        root = pandapower_net.bus
        root['LV_bus'] = root['name'].str.contains("LVbus")
        root = root[root['LV_bus']]
        root = root.index
        root = list(root)
        root = root[0]
        return root


    def get_no_branches(networkx_graph, pandapower_net) -> int:
        # number of branches
        root = get_root(pandapower_net)
        # number of branches
        no_branches = networkx_graph.degree(root) - 1
        return no_branches


    def get_distances_in_graph(pandapower_net, networkx_graph) -> tuple[float, float]:
        """
        distances in the graph:
        - average trafo distance: distance from each house connection to the trafo
          divided by the number of house connections
        - max trafo distance: distance to the furthest house connection from the trafo
        """

        root = get_root(pandapower_net)
        # the consumers are leaves in the graph
        leaves = pandapower_net.bus
        leaves['consumer_bus'] = leaves['name'].str.contains("Consumer Nodebus")
        leaves = leaves[leaves['consumer_bus']]
        leaves = leaves.index
        leaves = list(leaves)
        no_leaves = len(leaves)

        # determine path lengths in graph:
        path_length_to_leaves = []
        for leaf in leaves:
            weighted_length = nx.dijkstra_path_length(networkx_graph, root,
                                                      leaf)  # calculates weigthed path from start to end note
            # in this case the weight of an edge is the actual length in km
            path_length_to_leaves.append(weighted_length)

        max_path_length = max(path_length_to_leaves)
        avg_path_length = sum(path_length_to_leaves) / no_leaves

        return avg_path_length, max_path_length


    def get_trafo_power(pandapower_net) -> float:
        # transformer
        df_trafo = pandapower_net.trafo.sn_mva
        return df_trafo.iloc[0]


    def calc_resistance(pandapower_net, networkx_graph) -> tuple[float, float, float, float, float]:
        """
        this function calculates the resistance and reactance in ohm of the network
        (Verbrauchersummenwiderstand accoding to Kerber 2010 p. 44)
        output:  resistance  and reactance and ratio of resistances
        """
        # get dataframe of load (consumer households) of the network
        df_load = pandapower_net.load

        # for the Verbrauchersummenwiderstand vsw the number of household equivalents (loads with 0.03 MW)
        # of each house connection are computed
        df_vsw = df_load.groupby('bus')['max_p_mw'].sum() * 1000.0 / PEAK_LOAD_HOUSEHOLD
        df_vsw = df_vsw.to_frame()
        df_vsw = df_vsw.reset_index()
        df_vsw = df_vsw.rename(columns={"bus": "house_connection"})
        df_vsw = df_vsw.rename(columns={"max_p_mw": "household_equivalents"})

        # get dataframe of lines (cables) with their length and resistances
        # get line with sim factor
        df_line = calculate_line_with_sim_factor(pandapower_net, networkx_graph)

        root = get_root(pandapower_net=pandapower_net)

        # compute path from root to each house connection
        df_vsw['path'] = ''
        for index, row in df_vsw.iterrows():
            df_vsw.at[index, 'path'] = nx.shortest_path(networkx_graph, source=root,
                                                        target=df_vsw.at[index, 'house_connection'])

        # assign each path to a branch
        df_vsw['branch'] = ''
        for branch in networkx_graph.edges(root):
            for index, row in df_vsw.iterrows():
                if branch[1] in row['path']:
                    df_vsw.at[index, 'branch'] = branch

        # maximum number of household (equivalents) of a single branch
        max_no_of_households_of_a_branch = df_vsw.groupby('branch')['household_equivalents'].sum().max()

        # compute the  part of the resistance and reactance for each edge of the graph (cable section)
        # the resistance depends on the cable type, length of the cable and the amount of households that are connected
        df_vsw['resistance'] = ''
        df_vsw['resistance_sections'] = ''
        df_vsw['reactance'] = ''
        df_vsw['reactance_sections'] = ''
        for index, row in df_vsw.iterrows():
            path_list = df_vsw.at[index, 'path']
            length = len(path_list)
            no_load = df_vsw.at[index, 'household_equivalents']
            resistance_list = []
            reactance_list = []
            for i in range(length - 1):
                start_node = path_list[i]
                end_node = path_list[i + 1]
                line = df_line[df_line['from_bus'] == start_node]  # here the correct cable is filtered that starts at start
                line = line[line['to_bus'] == end_node]  # node and ends at end node
                line = line.head(1)
                length_km = line['length_km']
                r_ohm_per_km = line['r_ohm_per_km']
                x_ohm_per_km = line['x_ohm_per_km']
                sim_factor = line['sim_factor_cumulated']
                resistance_of_cable_section = no_load * length_km * r_ohm_per_km * sim_factor
                resistance_of_cable_section = resistance_of_cable_section
                resistance_list.append(resistance_of_cable_section)
                reactance_of_cable_section = no_load * length_km * x_ohm_per_km
                reactance_of_cable_section = reactance_of_cable_section
                reactance_list.append(reactance_of_cable_section)
            df_vsw.at[index, 'resistance'] = math.fsum(resistance_list)
            df_vsw.at[index, 'resistance_sections'] = resistance_list
            df_vsw.at[index, 'reactance'] = math.fsum(reactance_list)
            df_vsw.at[index, 'reactance_sections'] = reactance_list

        # sum the capacitive and inductive resistance of each house_connection to obtain the total resistance of the network
        resistance = df_vsw['resistance'].sum()
        reactance = df_vsw['reactance'].sum()
        ratio = resistance / reactance

        # maximum vsw of a single branch in the network
        max_vsw_of_a_branch = df_vsw.groupby('branch')['resistance'].sum().max()

        return max_no_of_households_of_a_branch, resistance, reactance, ratio, max_vsw_of_a_branch

    def calculate_line_with_sim_factor(pandapower_net, networkx_graph) -> pd.DataFrame:
        """Calculate the simultaneity factor for each line segment."""

        # Prepare SIM_FACTOR definitions
        df_sim_def = pd.DataFrame.from_dict(SIM_FACTOR, orient='index', columns=['sim_factor']).reset_index()
        df_sim_def.rename(columns={'index': 'description'}, inplace=True)

        # Initialize new attributes in pandapower_net.line
        net_line = pandapower_net.line.drop(['c_nf_per_km', 'g_us_per_km', 'max_i_ka', 'df', 'type', 'in_service'],
            axis=1).drop_duplicates()
        new_columns = ['sim_factor_cumulated', 'sim_load', 'no_commercial', 'load_commercial_mw', 'no_public',
            'load_public_mw', 'no_residential', 'load_residential_mw']
        net_line[new_columns] = ''

        # Calculate simultaneity factor for consumer buses
        level1 = pd.merge(pandapower_net.load, pandapower_net.bus, left_on='bus', right_index=True).replace(
            ['MFH', 'SFH', 'AB', 'TH'], 'Residential')

        load_data = level1.groupby(['bus', 'zone'])['max_p_mw'].sum().reset_index()
        load_count = level1.groupby(['bus', 'zone'])['name_x'].count().reset_index().rename(columns={'name_x': 'count'})
        load_count = pd.merge(load_count, load_data, on='bus').merge(df_sim_def, left_on='zone_x', right_on='description')

        load_count['sim_factor_level1'] = load_count.apply(lambda x: oneSimultaneousLoad(1, x['count'], x['sim_factor']),
            axis=1)
        load_count['sim_load_level1'] = load_count['max_p_mw'] * load_count['sim_factor_level1']

        # Update net_line with calculated values
        for _, row in load_count.iterrows():
            bus, desc = row['bus'], row['description']
            idx = net_line.index[net_line['to_bus'] == bus].tolist()
            if not idx:
                continue
            idx = idx[0]

            net_line.at[idx, 'sim_factor_cumulated'] = row['sim_factor_level1']
            net_line.at[idx, 'sim_load'] = row['sim_load_level1']
            net_line.at[idx, f'no_{desc.lower()}'] = row['count']
            net_line.at[idx, f'load_{desc.lower()}_mw'] = row['max_p_mw']

        # Process connection node buses
        connection_buses = pandapower_net.bus[pandapower_net.bus['name'].str.contains("Connection Nodebus")].index.tolist()
        df_conn_bus = pd.DataFrame({'bus': connection_buses, 'source': 0})
        df_conn_bus['len_to_trafo'] = df_conn_bus['bus'].apply(
            lambda x: nx.shortest_path_length(networkx_graph, source=0, target=x))
        df_conn_bus.sort_values('len_to_trafo', ascending=False, inplace=True)

        # Aggregate data for upstream buses
        def aggregate_upstream(connected_downstream, upstream_idx):
            for col in new_columns[2:]:
                net_line.at[upstream_idx, col] = connected_downstream[col].sum()

            sim_load = sum([
                oneSimultaneousLoad(net_line.at[upstream_idx, f'load_{t}_mw'], net_line.at[upstream_idx, f'no_{t}'],
                                    SIM_FACTOR[t]) for t in ['commercial', 'public', 'residential']])
            peak_load = sum(net_line.at[upstream_idx, f'load_{t}_mw'] for t in ['commercial', 'public', 'residential'])

            net_line.at[upstream_idx, 'sim_load'] = sim_load
            net_line.at[upstream_idx, 'sim_factor_cumulated'] = sim_load / peak_load if peak_load else 0

        for _, row in df_conn_bus.iterrows():
            furthest_bus = row['bus']
            downstream = net_line[net_line['from_bus'] == furthest_bus]
            upstream_idx = net_line[net_line['to_bus'] == furthest_bus].index.tolist()
            if upstream_idx:
                aggregate_upstream(downstream, upstream_idx[0])

        return net_line
