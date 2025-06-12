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

from src import utils
from src.config_table_structure import *
from src.config_loader import *

import warnings
warnings.simplefilter(action='ignore', category=UserWarning)

class GridMixin:
        def get_transformer_data(self, settlement_type: int = None) -> tuple[np.array, dict]:
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
    
            query = """SELECT equipment_data.s_max_kva , cost_eur
                FROM equipment_data
                WHERE typ = 'Transformer' AND application_area IN %(tuple)s
                ORDER BY s_max_kva;"""
    
            self.cur.execute(query, {"tuple": application_area_tuple})
            data = self.cur.fetchall()
            capacities = [i[0] for i in data]
            transformer2cost = {i[0]: i[1] for i in data}
    
            self.logger.debug("Transformer data fetched.")
            return np.array(capacities), transformer2cost
        def create_cable_std_type(self, net: pp.pandapowerNet) -> None:
            """Create standard pandapower cable types from equipment_data table."""
            query = """
                SELECT name, r_mohm_per_km/1000.0 as r_ohm_per_km, x_mohm_per_km/1000.0 as x_ohm_per_km, 
                       max_i_a/1000.0 as max_i_ka
                FROM equipment_data
                WHERE typ = 'Cable'
            """
    
            # Execute query and fetch cable data
            self.cur.execute(query)
            cables = self.cur.fetchall()
    
            # Create standard type for each cable in the database
            for cable in cables:
                name, r_ohm_per_km, x_ohm_per_km, max_i_ka = cable
                pp_name = name.replace('_', ' ') # Extract name
                q_mm2 = int(name.split("_")[-1])  # Extract cross-section from name
    
                pp.create_std_type(
                    net,
                    {
                        "r_ohm_per_km": float(r_ohm_per_km),
                        "x_ohm_per_km": float(x_ohm_per_km),
                        "max_i_ka": float(max_i_ka),
                        "c_nf_per_km": float(0),  # Set to zero for our standard grids
                        "q_mm2": q_mm2
                    },
                    name=pp_name,
                    element="line",
                )
    
            self.logger.debug(f"Created {len(cables)} standard cable types from equipment_data table")
            return None
        def create_lvmv_bus(self, plz: int, kcid: int, bcid: int, net: pp.pandapowerNet) -> None:
            geodata = self.get_ont_geom_from_bcid(plz, kcid, bcid)
    
            pp.create_bus(
                net,
                name="LVbus 1",
                vn_kv=VN * 1e-3,
                geodata=geodata,
                max_vm_pu=V_BAND_HIGH,
                min_vm_pu=V_BAND_LOW,
                type="n",
            )
    
            # medium voltage external network and mvbus
            mv_data = (float(geodata[0]), float(geodata[1]) + 1.5 * 1e-4)
            mv_bus = pp.create_bus(
                net,
                name="MVbus 1",
                vn_kv=20,
                geodata=mv_data,
                max_vm_pu=V_BAND_HIGH,
                min_vm_pu=V_BAND_LOW,
                type="n",
            )
            pp.create_ext_grid(net, bus=mv_bus, vm_pu=1, name="External grid")
    
            return None
        def create_transformer(self, plz: int, kcid: int, bcid: int, net: pp.pandapowerNet) -> None:
            transformer_rated_power = self.get_transformer_rated_power_from_bcid(plz, kcid, bcid)
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
                trafo_std = f"{str(transformer_rated_power * 1e-3 * 0.5)} MVA 20/0.4 kV"
                parallel = 2
            else:
                trafo_name = "630 transformer"
                trafo_std = "0.63 MVA 20/0.4 kV"
                parallel = transformer_rated_power / 630
            trafo_index = pp.create_transformer(
                net,
                pp.get_element_index(net, "bus", "MVbus 1"),
                pp.get_element_index(net, "bus", "LVbus 1"),
                name=trafo_name,
                std_type=trafo_std,
                tap_pos=0,
                parallel=parallel,
            )
            net.trafo.at[trafo_index, "sn_mva"] = transformer_rated_power * 1e-3
            return None
        def create_connection_bus(self, connection_nodes: list, net: pp.pandapowerNet):
            for i in range(len(connection_nodes)):
                node_geodata = self.get_node_geom(connection_nodes[i])
                pp.create_bus(
                    net,
                    name=f"Connection Nodebus {connection_nodes[i]}",
                    vn_kv=VN * 1e-3,
                    geodata=node_geodata,
                    max_vm_pu=V_BAND_HIGH,
                    min_vm_pu=V_BAND_LOW,
                    type="n",
                )
        def create_consumer_bus_and_load(
                self, consumer_list: list, load_units: dict, net: pp.pandapowerNet, load_type: dict, building_df: pd.DataFrame
        ) -> None:
            for i in range(len(consumer_list)):
                node_geodata = self.get_node_geom(consumer_list[i])
    
                ltype = load_type[consumer_list[i]]
    
                if ltype in ["SFH", "MFH", "AB", "TH"]:
                    peak_load = CONSUMER_CATEGORIES.loc[
                        CONSUMER_CATEGORIES["definition"] == ltype, "peak_load"
                    ].values[0]
                else:
                    peak_load = building_df[building_df["vertice_id"] == consumer_list[i]][
                        "peak_load_in_kw"
                    ].tolist()[0]
    
                pp.create_bus(
                    net=net,
                    name=f"Consumer Nodebus {consumer_list[i]}",
                    vn_kv=VN * 1e-3,
                    geodata=node_geodata,
                    max_vm_pu=V_BAND_HIGH,
                    min_vm_pu=V_BAND_LOW,
                    type="n",
                    zone=ltype,
                )
                for j in range(1, load_units[consumer_list[i]] + 1):
                    pp.create_load(
                        net=net,
                        bus=pp.get_element_index(
                            net, "bus", f"Consumer Nodebus {consumer_list[i]}"
                        ),
                        p_mw=0,
                        name=f"Load {consumer_list[i]} household {j}",
                        max_p_mw=peak_load * 1e-3,
                    )
        def install_consumer_cables(
                self,
                plz: int,
                bcid: int,
                kcid: int,
                branch_deviation: float,
                branch_node_list: list,
                ont_vertice: int,
                vertices_dict: dict,
                Pd: dict,
                net: pp.pandapowerNet,
                connection_available_cables: list[str],
                local_length_dict: dict,
        ) -> dict:
            # lines
            # first draw house connections from consumer node to corresponding connection node
            consumer_list = self.get_vertices_from_connection_points(branch_node_list)
            branch_consumer_list = [n for n in consumer_list if n in vertices_dict.keys()]
            for vertice in branch_consumer_list: # TODO: looping for duplicate vertices
                path_list = self.get_path_to_bus(vertice, ont_vertice)
                start_vid = path_list[1]
                end_vid = path_list[0]
    
                geodata = self.get_node_geom(start_vid)
                start_node_geodata = (
                    float(geodata[0]) + 5 * 1e-6 * branch_deviation,
                    float(geodata[1]) + 5 * 1e-6 * branch_deviation,
                )
    
                end_node_geodata = self.get_node_geom(end_vid)
    
                line_geodata = [start_node_geodata, end_node_geodata]
    
                cost_km = (vertices_dict[end_vid] - vertices_dict[start_vid]) * 1e-3
    
                count = 1
                sim_load = Pd[end_vid]  # power in Watt
                Imax = sim_load * 1e-3 / (VN * V_BAND_LOW * np.sqrt(3))  # current in kA
                voltage_available_cables_df = None
                while True:
                    line_df = pd.DataFrame.from_dict(net.std_types["line"], orient="index")
                    current_available_cables_df = line_df[
                        (line_df["max_i_ka"] >= Imax / count)
                        & (line_df.index.isin(connection_available_cables))
                        ]
    
                    if len(current_available_cables_df) == 0:
                        count += 1
                        continue
    
                    current_available_cables_df["cable_impedence"] = np.sqrt(
                        current_available_cables_df["r_ohm_per_km"] ** 2
                        + current_available_cables_df["x_ohm_per_km"] ** 2
                    )  # impedence in ohm / km
                    if sim_load <= 100:
                        voltage_available_cables_df = current_available_cables_df[
                            current_available_cables_df["cable_impedence"]
                            <= 2 * 1e-3 / (Imax * cost_km / count)
                            ]
                    else:
                        voltage_available_cables_df = current_available_cables_df[
                            current_available_cables_df["cable_impedence"]
                            <= 4 * 1e-3 / (Imax * cost_km / count)
                            ]
    
                    if len(voltage_available_cables_df) == 0:
                        count += 1
                        continue
                    else:
                        break
    
                cable = voltage_available_cables_df.sort_values(
                    by=["q_mm2"]
                ).index.tolist()[0]
                local_length_dict[cable] += count * cost_km
    
                pp.create_line(
                    net,
                    from_bus=pp.get_element_index(net, "bus", f"Connection Nodebus {start_vid}"),
                    to_bus=pp.get_element_index(net, "bus", f"Consumer Nodebus {end_vid}"),
                    length_km=cost_km,
                    std_type=cable,
                    name=f"Line to {end_vid}",
                    geodata=line_geodata,
                    parallel=count,
                )
    
                self.insert_lines(geom=line_geodata,
                                  plz=plz,
                                  bcid=bcid,
                                  kcid=kcid,
                                  line_name=f"Line to {end_vid}",
                                  std_type=cable,
                                  from_bus=pp.get_element_index(net, "bus", f"Connection Nodebus {start_vid}"),
                                  to_bus=pp.get_element_index(net, "bus", f"Consumer Nodebus {end_vid}"),
                                  length_km=cost_km
                                  )
    
            return local_length_dict
        def find_minimal_available_cable(self, Imax: float, net: pp.pandapowerNet, cables_list: list, distance: int=0) -> tuple[str, int]:
            count = 1
            cable = None
            while 1:
                line_df = pd.DataFrame.from_dict(net.std_types["line"], orient="index")
                current_available_cables = line_df[
                    (line_df.index.isin(cables_list))
                    & (line_df["max_i_ka"] >= Imax / count)
                    ]
                if len(current_available_cables) == 0:
                    count += 1
                    continue
    
                if distance != 0:
                    current_available_cables["cable_impedence"] = np.sqrt(
                        current_available_cables["r_ohm_per_km"] ** 2
                        + current_available_cables["x_ohm_per_km"] ** 2
                    )  # impedence in ohm / km
                    voltage_available_cables = current_available_cables[
                        current_available_cables["cable_impedence"]
                        <= 400 * 0.045 / (Imax * distance / count)
                        ]
                    if len(voltage_available_cables) == 0:
                        count += 1
                        continue
                    else:
                        cable = voltage_available_cables.sort_values(
                            by=["q_mm2"]
                        ).index.tolist()[0]
                        break
                else:
                    cable = current_available_cables.sort_values(
                        by=["q_mm2"]
                    ).index.tolist()[0]
                    break
    
            return cable, count
        def create_line_node_to_node(
                self,
                plz: int,
                kcid: int,
                bcid: int,
                branch_node_list: list,
                branch_deviation: float,
                vertices_dict: dict,
                local_length_dict: dict,
                cable: str,
                ont_vertice: int,
                count: float,
                net: pp.pandapowerNet
        ) -> dict:
            """creates the lines / cables from one Connection Nodebus to the next. Adds them to the pandapower network
            and lines result table"""
            for i in range(len(branch_node_list) - 1):
                # to get the line geodata, we now need to consider all the nodes in database, not only connection points
                node_path_list = self.get_path_to_bus(branch_node_list[i], ont_vertice)  # gets the path along ways_result
                # end at next connection point
                if branch_node_list[i + 1] not in node_path_list:  # if next node of branch node list not in node path list
                    self.logger.debug(f"creating line to node i + 1: {i+1} node: {branch_node_list[i + 1]}")
                    node_path_list = self.get_path_to_bus(branch_node_list[i], branch_node_list[i + 1])
                    # node_path_list = [branch_node_list[i], branch_node_list[i + 1]]
                    # intermediate nodes up to next connection nodebus are neglected
                    # the cable will directly connect to next connection nodebus
    
                node_path_list = node_path_list[
                                 : node_path_list.index(branch_node_list[i + 1]) + 1
                                 ]  # the node path list goes up to the index (branch_node_list[i + 1]) +1
                node_path_list.reverse()  # to keep the correct direction
    
                start_vid = node_path_list[0]
                end_vid = node_path_list[-1]
    
                line_geodata = []
                for p in node_path_list:
                    node_geodata = self.get_node_geom(p)
                    node_geodata = (
                        float(node_geodata[0]) + 5 * 1e-6 * branch_deviation,
                        float(node_geodata[1]) + 5 * 1e-6 * branch_deviation,
                    )
                    line_geodata.append(node_geodata)
    
                cost_km = (vertices_dict[end_vid] - vertices_dict[start_vid]) * 1e-3
    
                local_length_dict[cable] += count * cost_km
                pp.create_line(
                    net,
                    from_bus=pp.get_element_index(
                        net, "bus", f"Connection Nodebus {start_vid}"
                    ),
                    to_bus=pp.get_element_index(
                        net, "bus", f"Connection Nodebus {end_vid}"
                    ),
                    length_km=cost_km,
                    std_type=cable,
                    name=f"Line to {end_vid}",
                    geodata=line_geodata,
                    parallel=count,
                )
    
                self.insert_lines(
                    geom=line_geodata,
                    plz=plz,
                    bcid=bcid,
                    kcid=kcid,
                    line_name=f"Line to {end_vid}",
                    std_type=cable,
                    from_bus=pp.get_element_index(net, "bus", f"Connection Nodebus {start_vid}"),
                    to_bus=pp.get_element_index(net, "bus", f"Connection Nodebus {end_vid}"),
                    length_km=cost_km
                )
            return local_length_dict
        def create_line_ont_to_lv_bus(
                self, plz:int, bcid:int, kcid:int, branch_start_node: int, branch_deviation:float, net:pp.pandapowerNet, cable: str, count: int
        ):
            end_vid = branch_start_node
            node_geodata = self.get_node_geom(end_vid)
            node_geodata = (
                float(node_geodata[0]) + 5 * 1e-6 * branch_deviation,
                float(node_geodata[1]) + 5 * 1e-6 * branch_deviation,
            )
            lvbus_geodata = (
                net.bus_geodata.loc[pp.get_element_index(net, "bus", "LVbus 1"), "x"]
                + 5 * 1e-6 * branch_deviation,
                net.bus_geodata.loc[pp.get_element_index(net, "bus", "LVbus 1"), "y"],
            )
            line_geodata = [lvbus_geodata, node_geodata]
    
            cost_km = 0
            pp.create_line(
                net,
                from_bus=pp.get_element_index(net, "bus", "LVbus 1"),
                to_bus=pp.get_element_index(net, "bus", f"Connection Nodebus {end_vid}"),
                length_km=cost_km,
                std_type=cable,
                name=f"Line to {end_vid}",
                geodata=line_geodata,
                parallel=count,
            )
    
            self.insert_lines(
                geom=line_geodata,
                plz=plz,
                bcid=bcid,
                kcid=kcid,
                line_name=f"Line to {end_vid}",
                std_type=cable,
                from_bus=pp.get_element_index(net, "bus", "LVbus 1"),
                to_bus=pp.get_element_index(net, "bus", f"Connection Nodebus {end_vid}"),
                length_km=cost_km
            )
        def create_line_start_to_lv_bus(
                self,
                plz: int,
                bcid: int,
                kcid: int,
                branch_start_node: int,
                branch_deviation: float,
                net: pp.pandapowerNet,
                vertices_dict: dict,
                cable: str,
                count: int,
                ont_vertice: int,
        ) -> int:
    
            node_path_list = self.get_path_to_bus(branch_start_node, ont_vertice)
    
            line_geodata = []
            for p in node_path_list:
                node_geodata = self.get_node_geom(p)
                node_geodata = (
                    float(node_geodata[0]) + 5 * 1e-6 * branch_deviation,
                    float(node_geodata[1]) + 5 * 1e-6 * branch_deviation,
                )
                line_geodata.append(node_geodata)
            lvbus_geodata = (
                net.bus_geodata.loc[pp.get_element_index(net, "bus", "LVbus 1"), "x"]
                + 5 * 1e-6 * branch_deviation,
                net.bus_geodata.loc[pp.get_element_index(net, "bus", "LVbus 1"), "y"],
            )
            line_geodata.append(lvbus_geodata)
            line_geodata.reverse()
    
            cost_km = vertices_dict[branch_start_node] * 1e-3
            length = count * cost_km  # distance in m
            pp.create_line(
                net,
                from_bus=pp.get_element_index(net, "bus", "LVbus 1"),
                to_bus=pp.get_element_index(
                    net, "bus", f"Connection Nodebus {branch_start_node}"
                ),
                length_km=cost_km,
                std_type=cable,
                name=f"Line to {branch_start_node}",
                geodata=line_geodata,
                parallel=count,
            )
    
            self.insert_lines(geom=line_geodata,
                              plz=plz,
                              bcid=bcid,
                              kcid=kcid,
                              line_name=f"Line to {branch_start_node}",
                              std_type=cable,
                              from_bus=pp.get_element_index(net, "bus", "LVbus 1"),
                              to_bus=pp.get_element_index(net, "bus", f"Connection Nodebus {branch_start_node}"),
                              length_km=cost_km
                              )
    
            return length
        def get_maximum_load_branch(
                self, furthest_node_path_list: list, buildings_df: pd.DataFrame, consumer_df: pd.DataFrame
        ) -> tuple[list, float]:
            # TOD O explanation?
            branch_node_list = []
            for node in furthest_node_path_list:
                branch_node_list.append(node)
                sim_load = utils.simultaneousPeakLoad(
                    buildings_df, consumer_df, branch_node_list
                )  # sim_peak load in kW
                Imax = sim_load / (VN * V_BAND_LOW * np.sqrt(3))  # current in kA
                if Imax >= 0.313 and len(
                        branch_node_list) > 1:  # 0.313 is the current limit of the largest allowed cable 4x185SE
                    branch_node_list.remove(node)
                    break
                elif Imax >= 0.313 and len(branch_node_list) == 1:
                    break
            sim_load = utils.simultaneousPeakLoad(
                buildings_df, consumer_df, branch_node_list
            )
            Imax = sim_load / (VN * V_BAND_LOW * np.sqrt(3))
    
            return branch_node_list, Imax
        def deviate_bus_geodata(self, branch_node_list: list, branch_deviation: float, net: pp.pandapowerNet):
            for node in branch_node_list:
                net.bus_geodata.at[
                    pp.get_element_index(net, "bus", f"Connection Nodebus {node}"), "x"
                ] += (5 * 1e-6 * branch_deviation)
                net.bus_geodata.at[
                    pp.get_element_index(net, "bus", f"Connection Nodebus {node}"), "y"
                ] += (5 * 1e-6 * branch_deviation)
        def get_vertices_from_connection_points(self, connection: list) -> list:
            query = """SELECT vertice_id FROM buildings_tem
                        WHERE connection_point IN %(c)s
                        AND type != 'Transformer';"""
            self.cur.execute(query, {"c": tuple(connection)})
            data = self.cur.fetchall()
            return [t[0] for t in data]
        def get_path_to_bus(self, vertice: int, ont: int) -> list:
            """routing problem: find the shortest path from vertice to the ont (ortsnetztrafo)"""
            query = """SELECT node FROM pgr_Dijkstra(
                        'SELECT way_id as id, source, target, cost, reverse_cost FROM ways_tem', %(v)s, %(o)s, false);"""
            """query = WITH
                        dijkstra AS(
                            SELECT * FROM pgr_Dijkstra(
                                            'SELECT way_id, source, target, cost, reverse_cost FROM ways_tem', %(v)s, %(o)s, false)
                        ),
                            get_geom AS(
                                SELECT dijkstra. *,
                                -- adjusting directionality
                                    CASE
                                        WHEN dijkstra.node = ways.source THEN geom
                                        ELSE ST_Reverse(geom)
                                    END AS route_geom
                                FROM dijkstra JOIN ways ON(edge=way_id)
                                ORDER BY seq)
                            SELECT seq, cost,
                            degrees(ST_azimuth(ST_StartPoint(route_geom), ST_EndPoint(route_geom))) AS azimuth,
                            ST_AsText(route_geom),
                            route_geom
                        FROM get_geom
                        ORDER BY seq;"""
            self.cur.execute(query, {"o": ont, "v": vertice})
            data = self.cur.fetchall()
            way_list = [t[0] for t in data]
    
            return way_list
        def get_transformer_rated_power_from_bcid(self, plz: int, kcid: int, bcid: int) -> int:
            query = """SELECT transformer_rated_power FROM grid_result
                        WHERE version_id = %(v)s 
                        AND plz = %(p)s 
                        AND kcid = %(k)s
                        AND bcid = %(b)s;"""
            self.cur.execute(query, {"v": VERSION_ID, "p": plz, "k": kcid, "b": bcid})
            transformer_rated_power = self.cur.fetchone()[0]
    
            return transformer_rated_power
        def get_building_connection_points_from_bc(self, kcid: int, bcid: int) -> list:
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
            except:
                cp = []
    
            return cp
        def get_single_connection_point_from_bc(self, kcid: int, bcid: int) -> int:
            count_query = """SELECT connection_point
                                    FROM buildings_tem AS b
                                    WHERE b.vertice_id IS NOT NULL
                                        AND b.bcid = %(b)s 
                                        AND b.kcid = %(k)s
                                        LIMIT 1;"""
            params = {"b": bcid, "k": kcid}
            self.cur.execute(count_query, params)
            conn_id = self.cur.fetchone()[0]
    
            return conn_id
        def upsert_transformer_selection(self, plz: int, kcid: int, bcid: int, connection_id: int):
            """Writes the vertice_id of chosen building as ONT location in the grid_result table"""
    
            query = """UPDATE grid_result SET ont_vertice_id = %(c)s
                       WHERE version_id = %(v)s AND plz = %(p)s AND kcid = %(k)s AND bcid = %(b)s;
                       
                       UPDATE grid_result SET model_status = 1 
                       WHERE version_id = %(v)s AND plz = %(p)s AND kcid = %(k)s AND bcid = %(b)s;
                       
                       INSERT INTO transformer_positions (grid_result_id, geom, comment)
                       VALUES (
                         (SELECT grid_result_id FROM grid_result WHERE version_id = %(v)s AND plz = %(p)s AND kcid = %(k)s AND bcid = %(b)s),
                         (SELECT the_geom FROM ways_tem_vertices_pgr WHERE id = %(c)s),
                         'on_way'
                       );"""
            params = {"v": VERSION_ID, "c": connection_id, "b": bcid, "k": kcid, "p": plz}
    
            self.cur.execute(query, params)
        def update_transformer_rated_power(self, plz: int, kcid: int, bcid: int, note: int):
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
                old_query = """SELECT transformer_rated_power FROM grid_result
                                WHERE  version_id = %(v)s
                                AND plz = %(p)s
                                AND kcid = %(k)s
                                AND bcid = %(b)s;"""
                self.cur.execute(
                    old_query, {"v": VERSION_ID, "p": plz, "k": kcid, "b": bcid}
                )
                transformer_rated_power = self.cur.fetchone()[0]
    
                new_transformer_rated_power = transformer_capacities[transformer_capacities > transformer_rated_power][0].item()
                update_query = """UPDATE grid_result SET transformer_rated_power = %(n)s
                                WHERE version_id = %(v)s 
                                AND plz = %(p)s
                                AND kcid = %(k)s
                                AND bcid = %(b)s;"""
                self.cur.execute(
                    update_query,
                    {"v": VERSION_ID, "p": plz, "k": kcid, "b": bcid, "n": new_transformer_rated_power},
                )
            else:
                double_trans = np.multiply(transformer_capacities[2:4], 2)
                combined = np.concatenate((transformer_capacities, double_trans), axis=None)
                np.sort(combined, axis=None)
                old_query = """SELECT transformer_rated_power FROM grid_result
                                            WHERE version_id = %(v)s 
                                            AND plz = %(p)s
                                            AND kcid = %(k)s
                                            AND bcid = %(b)s;"""
                self.cur.execute(
                    old_query, {"v": VERSION_ID, "p": plz, "k": kcid, "b": bcid}
                )
                transformer_rated_power = self.cur.fetchone()[0]
                if transformer_rated_power in combined.tolist():
                    return None
                new_transformer_rated_power = np.ceil(transformer_rated_power / 630) * 630
                update_query = """UPDATE grid_result SET transformer_rated_power = %(n)s
                                                WHERE version_id = %(v)s 
                                                AND plz = %(p)s 
                                                AND kcid = %(k)s 
                                                AND bcid = %(b)s;"""
                self.cur.execute(
                    update_query,
                    {"v": VERSION_ID, "p": plz, "k": kcid, "b": bcid, "n": new_transformer_rated_power},
                )
                self.logger.debug("double or multiple transformer group transformer_rated_power assigned")
        def get_cables(self,  anw: tuple) -> pd.DataFrame:
            query = """SELECT name, max_i_a, r_mohm_per_km, x_mohm_per_km, z_mohm_per_km, cost_eur FROM equipment_data
                        WHERE typ = 'Cable' AND application_area IN %(a)s ORDER BY max_i_a DESC; """
            cables_df = pd.read_sql_query(query, self.conn, params={"a": anw})
            self.logger.debug(f"{len(cables_df)} different cable types are imported...")
            return cables_df
        def get_included_transformers(self, kcid: int) -> list:
            """
            Reads the vertice ids of transformers from a given kcid
            :param kcid:
            :return: list
            """
            query = """SELECT vertice_id FROM buildings_tem WHERE kcid = %(k)s AND type = 'Transformer';"""
            self.cur.execute(query, {"k": kcid})
            transformers_list = (
                [t[0] for t in data] if (data := self.cur.fetchall()) else []
            )
            return transformers_list
        def get_consumer_to_transformer_df(self, kcid: int, transformer_list: list) -> pd.DataFrame:
            consumer_query = """SELECT DISTINCT connection_point FROM buildings_tem 
                                WHERE kcid = %(k)s AND type != 'Transformer';"""
            self.cur.execute(consumer_query, {"k": kcid})
            consumer_list = [t[0] for t in self.cur.fetchall()]
    
            cost_query = """SELECT * FROM pgr_dijkstraCost(
                    'SELECT way_id as id, source, target, cost, reverse_cost FROM ways_tem',
                    %(cl)s,%(tl)s,
                    false);"""
            cost_df = pd.read_sql_query(
                cost_query,
                con=self.conn,
                params={"cl": consumer_list, "tl": transformer_list},
                dtype={"start_vid": np.int16, "end_vid": np.int16, "agg_cost": np.int16},
            )
    
            return cost_df
        def position_brownfield_transformers(self, plz: int, kcid: int, transformer_list: list) -> None:
            """
            Assign buildings to the existing transformers and store them as bcid in buildings_tem.
            Args:
                plz: Postal code
                kcid: K-means cluster ID
                transformer_list: List of transformer IDs
            """
            self.logger.debug(f"{len(transformer_list)} transformers found")
    
            # Get cost dataframe between consumers and transformers
            cost_df = self.get_consumer_to_transformer_df(kcid, transformer_list)
    
            # Filter out connections with distance >= 300
            cost_df = cost_df[cost_df["agg_cost"] < 300].sort_values(by=["agg_cost"])
    
            # Initialize tracking variables
            pre_result_dict = {transformer_id: [] for transformer_id in transformer_list}
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
                sim_load = self.calculate_sim_load(pre_result_dict[end_transformer_id])
    
                # Check if transformer capacity exceeded
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
                    self.logger.debug(f"Transformer {transformer_id} has no assigned consumer, deleted")
                    self.delete_transformers_from_buildings_tem([transformer_id])
                    continue
    
                # Create building cluster with sequential negative ID
                building_cluster_count -= 1
    
                # Calculate the simulated load for all loads assigned to this transformer
                sim_load = self.calculate_sim_load(pre_result_dict[transformer_id])
    
                # Define the available standard transformer sizes in kVA
                possible_transformers = np.array([100, 160, 250, 400, 630]) #TODO: check with settlement_type approach
    
                # Select the smallest transformer that is larger than the simulated load
                transformer_rated_power = possible_transformers[possible_transformers > float(sim_load)][0].item()
    
                # Update database with new building cluster
                self.update_building_cluster(
                    transformer_id,
                    pre_result_dict[transformer_id],
                    building_cluster_count,
                    kcid,
                    plz,
                    transformer_rated_power
                )
    
            self.logger.debug("Brownfield clusters completed")
        def position_greenfield_transformers(self, plz, kcid, bcid):
            """
            Positions a transformer at the optimal location for a greenfield building cluster.
    
            The optimal location minimizes the sum of distance*load from each vertex to others.
    
            Args:
                plz: Postcode
                kcid: Kmeans cluster ID
                bcid: Building cluster ID
            """
            # Get all connection points in the building cluster
            connection_points = self.get_building_connection_points_from_bc(kcid, bcid)
    
            # If there's only one connection point, use it
            if len(connection_points) == 1:
                self.upsert_transformer_selection(plz, kcid, bcid, connection_points[0])
                return
    
            # Get distance matrix between all connection points
            localid2vid, dist_mat, _ = self.get_distance_matrix_from_building_cluster(kcid, bcid)
    
            # Get load vector for each connection point
            loads = self.generate_load_vector(kcid, bcid)
    
            # Calculate weighted distance (distance * load) for each potential location
            total_load_per_vertice = dist_mat.dot(loads)
    
            # Select the point with minimum weighted distance as transformer location
            min_localid = np.argmin(total_load_per_vertice)
            ont_connection_id = int(localid2vid[min_localid])
    
            # Update the database with the selected transformer position
            self.upsert_transformer_selection(plz, kcid, bcid, ont_connection_id)
        def draw_building_connection(self) -> None:
            """
            Updates ways_tem, creates pgr network topology in new tables:
            :return:
            """
            connection_query = """ SELECT draw_home_connections(); """
            self.cur.execute(connection_query)
    
            topology_query = """select pgr_createTopology('ways_tem', 0.01, id:='way_id', the_geom:='geom', clean:=true) """
            self.cur.execute(topology_query)
    
            # add_buildings_query = '''SELECT add_buildings();'''
            # self.cur.execute(add_buildings_query)
            # self.conn.commit()
    
            analyze_query = (
                """SELECT pgr_analyzeGraph('ways_tem',0.01, the_geom:='geom'); """
            )
            self.cur.execute(analyze_query)
        def insert_transformers(self, plz: int) -> None:
            """
            Add up the existing transformers from transformers table to the buildings_tem table
            :param plz:
            :return:
            """
            insert_query = """
                --UPDATE transformers SET geom = ST_Centroid(geom) WHERE ST_GeometryType(geom) =  'ST_Polygon';
                INSERT INTO buildings_tem (osm_id, geom)--(osm_id,center)
                    SELECT osm_id, geom 
                    --FROM transformers WHERE ST_Within(geom, (SELECT geom FROM postcode_result LIMIT 1)) IS FALSE;
                    FROM transformers as t
                        WHERE ST_Within(t.geom, (SELECT geom FROM postcode_result WHERE postcode_result_plz = %(p)s AND version_id = %(v)s)); --IS FALSE;
                UPDATE buildings_tem SET plz = %(p)s WHERE plz ISNULL;
                UPDATE buildings_tem SET center = ST_Centroid(geom) WHERE center ISNULL;
                UPDATE buildings_tem SET type = 'Transformer' WHERE type ISNULL;
                UPDATE buildings_tem SET peak_load_in_kw = -1 WHERE peak_load_in_kw ISNULL;"""
            self.cur.execute(insert_query, {"p": plz, "v": VERSION_ID})
        def delete_transformers(self) -> None:
            """all transformers are deleted from table transformers in database"""
            delete_query = """DELETE FROM transformers;"""
            self.cur.execute(delete_query)
            self.conn.commit()
            self.logger.info('Transformers deleted.')
        def count_indoor_transformers(self) -> None:
            """counts indoor transformers before deleting them"""
            query = """WITH union_table (ungeom) AS 
                    (SELECT ST_Union(geom) FROM buildings_tem WHERE peak_load_in_kw = 0)
                SELECT COUNT(*) 
                    FROM buildings_tem 
                    WHERE ST_Within(center, (SELECT ungeom FROM union_table))
                    AND type = 'Transformer';"""
            self.cur.execute(query)
            count = self.cur.fetchone()[0]
            self.logger.debug(f"{count} indoor transformers will be deleted")
        def drop_indoor_transformers(self) -> None:
            """
            Drop transformer if it is inside a building with zero load
            :return:
            """
            query = """WITH union_table (ungeom) AS 
                    (SELECT ST_Union(geom) FROM buildings_tem WHERE peak_load_in_kw = 0)
                DELETE FROM buildings_tem WHERE ST_Within(center, (SELECT ungeom FROM union_table))
                    AND type = 'Transformer';"""
            self.cur.execute(query)
        def delete_transformers_from_buildings_tem(self, vertices: list) -> None:
            """
            Deletes selected transformers from buildings_tem
            :param vertices:
            :return:
            """
            query = """
                    DELETE FROM buildings_tem WHERE vertice_id IN %(v)s;"""
            self.cur.execute(query, {"v": tuple(map(int, vertices))})
        def read_cable_dict(self, plz: int) -> dict:
            read_query = """SELECT cable_length FROM plz_parameters
            WHERE version_id = %(v)s AND plz = %(p)s;"""
            self.cur.execute(read_query, {"v": VERSION_ID, "p": plz})
            cable_length = self.cur.fetchall()[0][0]
    
            return cable_length
        def get_grid_versions_with_plz(self, plz: int) -> list[tuple]:
            query = (
                """SELECT DISTINCT version_id FROM grid_result WHERE plz = %(p)s"""
            )
            self.cur.execute(query, {"p": plz})
            result = self.cur.fetchall()
            return result
        def get_grids_of_version(self, plz: int, version_id: str) -> list[tuple]:
            query = (
                """SELECT kcid, bcid, grid
                    FROM grid_result 
                    WHERE plz = %(p)s AND version_id = %(v)s""")
            self.cur.execute(query, {"p": plz, "v": version_id})
            result = self.cur.fetchall()
            return result
        def get_grids_from_plz(self, plz: int) -> pd.DataFrame:
            grids_query = """SELECT version_id, plz, kcid, bcid, grid FROM grids_result
                             WHERE plz = %(p)s"""
            params = {"p": plz}
            grids_df = pd.read_sql_query(grids_query, con=self.conn, params=params)
            self.logger.debug(f"{len(grids_df)} grid data fetched.")
    
            return grids_df
        def insert_lines(
                self,
                geom: list,
                plz: int,
                bcid: int,
                kcid: int,
                line_name: str,
                std_type: str,
                from_bus: int,
                to_bus: int,
                length_km: float
        ) -> None:
            """writes lines / cables that belong to a network into the database"""
            line_insertion_query = """INSERT INTO lines_result (
                        grid_result_id, 
                        geom,
                        line_name,
                        std_type,
                        from_bus,
                        to_bus,
                        length_km
                    ) 
                    VALUES (
                        (
                            SELECT grid_result_id
                            FROM grid_result
                            WHERE version_id = %(v)s
                            AND plz = %(plz)s
                            AND kcid = %(kcid)s
                            AND bcid = %(bcid)s
                        ),
                        ST_SetSRID(%(geom)s::geometry,3035),
                        %(line_name)s,
                        %(std_type)s,
                        %(from_bus)s,
                        %(to_bus)s,
                        %(length_km)s
                    ); """
            self.cur.execute(line_insertion_query, {
                "v": VERSION_ID,
                "geom": LineString(geom).wkb_hex,
                "plz": int(plz),
                "bcid": int(bcid),
                "kcid": int(kcid),
                "line_name": line_name,
                "std_type": std_type,
                "from_bus": int(from_bus),
                "to_bus": int(to_bus),
                "length_km": length_km
            })
