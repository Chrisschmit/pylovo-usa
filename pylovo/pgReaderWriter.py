import json
import math
import time
from typing import *
from decimal import *

import geopandas as gpd
import pandapower as pp
import pandapower.topology as top
import psycopg2 as pg
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform
from shapely.geometry import LineString
from sqlalchemy import create_engine, text

from pylovo import utils
from pylovo.config_data import *
from pylovo.config_version import *

import warnings
warnings.simplefilter(action='ignore', category=UserWarning)

class PgReaderWriter:
    """
    This class is the interface with the database. Functions communicating with the database
    are listed under this class.
    Use this class to interact with the database and to utilize connection resources.
    """

    # Konstruktor
    def __init__(
            self, dbname=DBNAME, user=USER, pw=PASSWORD, host=HOST, port=PORT, **kwargs
        ):
        self.logger = utils.create_logger(
            "PgReaderWriter", log_file=kwargs.get("log_file", "log.txt"), log_level=LOG_LEVEL
        )
        try:
            self.conn = pg.connect(
                database=dbname, user=user, password=pw, host=host, port=port
            )
            self.cur = self.conn.cursor()
            self.db_path = f"postgresql+psycopg2://{user}:{pw}@{host}:{port}/{dbname}"
            self.sqla_engine = create_engine(self.db_path)
        except pg.OperationalError as err:
            self.logger.warning(
                f"Connecting to {dbname} was not successful. Make sure, that you have established the SSH "
                f"connection with correct port mapping."
            )
            raise err


        self.logger.debug(f"PgReaderWriter is constructed and connected to {self.db_path}.")

    # Dekonstruktor
    def __del__(self):
        self.cur.close()
        self.conn.close()

    def get_consumer_categories(self):
        """
        Returns: A dataframe with self-defined consumer categories and typical values
        """
        query = """SELECT * FROM public.consumer_categories"""
        cc_df = pd.read_sql_query(query, self.conn)
        cc_df.set_index("definition", drop=False, inplace=True)
        cc_df.sort_index(inplace=True)
        self.logger.debug("Consumer categories fetched.")
        return cc_df

    def get_settlement_type_from_plz(self, plz) -> int:
        """
        Args:
            plz:
        Returns: Settlement type: 1=City, 2=Village, 3=Rural
        """
        settlement_query = """SELECT settlement_type FROM public.postcode_result
            WHERE postcode_result_id = %(p)s 
            LIMIT 1; """
        self.cur.execute(settlement_query, {"p": plz})
        settlement_type = self.cur.fetchone()[0]

        return settlement_type

    def get_transformer_data(self, settlement_type: int = None) -> tuple[np.array, dict]:
        """
        Args:
            Settlement type: 1=City, 2=Village, 3=Rural
        Returns: Typical transformer capacities and costs depending on the settlement type
        """
        # if settlement_type == 1:
        #     application_area_tuple = (1, 2, 3)
        # elif settlement_type == 2:
        #     application_area_tuple = (2, 3, 4)
        # elif settlement_type == 3:
        #     application_area_tuple = (3, 4, 5)
        # else:
        #     print("Incorrect settlement type number specified.")
        #     return
        application_area_tuple = (1, 2, 3, 4, 5) # TODO:check selection

        query = """SELECT equipment_data.s_max_kva , cost_eur
            FROM public.equipment_data
            WHERE typ = 'Transformer' AND application_area IN %(tuple)s
            ORDER BY s_max_kva;"""

        self.cur.execute(query, {"tuple": application_area_tuple})
        data = self.cur.fetchall()
        capacities = [i[0] for i in data]
        transformer2cost = {i[0]: i[1] for i in data}

        self.logger.debug("Transformer data fetched.")
        return np.array(capacities), transformer2cost

    def get_buildings_from_kcid(
            self,
            kcid : int,
    ) -> pd.DataFrame:
        """
        Args:
            kcid: kmeans_cluster ID
        Returns: A dataframe with all building information
        """
        buildings_query = """SELECT * FROM public.buildings_tem 
                        WHERE connection_point IS NOT NULL
                        AND kcid = %(k)s
                        AND bcid ISNULL;"""
        params = {"k": kcid}

        buildings_df = pd.read_sql_query(buildings_query, con=self.conn, params=params)
        buildings_df.set_index("vertice_id", drop=False, inplace=True)
        buildings_df.sort_index(inplace=True)

        self.logger.debug(
            f"Building data fetched. {len(buildings_df)} buildings from kc={kcid} ..."
        )

        return buildings_df

    def get_buildings_from_bc(self, plz: int, kcid: int, bcid: int) -> pd.DataFrame:

        buildings_query = """SELECT * FROM buildings_tem
                        WHERE type != 'Transformer'
                        AND plz = %(p)s
                        AND bcid = %(b)s
                        AND kcid = %(k)s;"""
        params = {"p": plz, "b": bcid, "k": kcid}

        buildings_df = pd.read_sql_query(buildings_query, con=self.conn, params=params)
        buildings_df.set_index("vertice_id", drop=False, inplace=True)
        buildings_df.sort_index(inplace=True)
        # dropping duplicate indices
        # buildings_df = buildings_df[~buildings_df.index.duplicated(keep='first')]

        self.logger.debug(f"{len(buildings_df)} building data fetched.")

        return buildings_df

    def prepare_vertices_list(
            self, plz: int, kcid: int, bcid: int
        ) -> tuple[dict, int, list, pd.DataFrame, pd.DataFrame, list, list]:
        vertices_dict, ont_vertice = self.get_vertices_from_bcid(plz, kcid, bcid)
        vertices_list = list(vertices_dict.keys())

        buildings_df = self.get_buildings_from_bc(plz, kcid, bcid)
        consumer_df = self.get_consumer_categories()
        consumer_list = buildings_df.vertice_id.to_list()
        consumer_list = list(dict.fromkeys(consumer_list))  # removing duplicates

        connection_nodes = [i for i in vertices_list if i not in consumer_list]

        return (
            vertices_dict,
            ont_vertice,
            vertices_list,
            buildings_df,
            consumer_df,
            consumer_list,
            connection_nodes,
        )

    def get_consumer_simultaneous_load_dict(
            self, consumer_list: list, buildings_df: pd.DataFrame
        ) -> tuple[dict, dict, dict]:
        Pd = {
            consumer: 0 for consumer in consumer_list
        }  # dict of all vertices in bc, 0 as default
        load_units = {consumer: 0 for consumer in consumer_list}
        load_type = {consumer: "SFH" for consumer in consumer_list}

        for row in buildings_df.itertuples():
            load_units[row.vertice_id] = row.houses_per_building
            load_type[row.vertice_id] = row.type
            gzf = CONSUMER_CATEGORIES.loc[
                CONSUMER_CATEGORIES.definition == row.type, "sim_factor"
            ].item()

            Pd[row.vertice_id] = utils.oneSimultaneousLoad(
                row.peak_load_in_kw * 1e-3, row.houses_per_building, gzf
            )  # simultaneous load of each building in mW

        return Pd, load_units, load_type

    def create_cable_std_type(self, net: pp.pandapowerNet) -> None:
        pp.create_std_type(
            net,
            {
                "r_ohm_per_km": 1.15,
                "x_ohm_per_km": 0.09,
                "max_i_ka": 0.103,
                "c_nf_per_km": 0,
                "q_mm2": 16,
            },
            name="NYY 4x16 SE",
            element="line",
        )
        pp.create_std_type(
            net,
            {
                "r_ohm_per_km": 0.524,
                "x_ohm_per_km": 0.085,
                "max_i_ka": 0.159,
                "c_nf_per_km": 0,
                "q_mm2": 35,
            },
            name="NYY 4x35 SE",
            element="line",
        )
        pp.create_std_type(
            net,
            {
                "r_ohm_per_km": 0.164,
                "x_ohm_per_km": 0.08,
                "max_i_ka": 0.313,
                "c_nf_per_km": 0,
                "q_mm2": 185,
            },
            name="NAYY 4x185 SE",
            element="line",
        )
        pp.create_std_type(
            net,
            {
                "r_ohm_per_km": 0.32,
                "x_ohm_per_km": 0.082,
                "max_i_ka": 0.215,
                "c_nf_per_km": 0,
                "q_mm2": 95,
            },
            name="NAYY 4x95 SE",
            element="line",
        )
        pp.create_std_type(
            net,
            {
                "r_ohm_per_km": 0.268,
                "x_ohm_per_km": 0.082,
                "max_i_ka": 0.232,
                "c_nf_per_km": 0,
                "q_mm2": 70,
            },
            name="NYY 4x70 SE",
            element="line",
        )
        pp.create_std_type(
            net,
            {
                "r_ohm_per_km": 0.193,
                "x_ohm_per_km": 0.082,
                "max_i_ka": 0.280,
                "c_nf_per_km": 0,
                "q_mm2": 95,
            },
            name="NYY 4x95 SE",
            element="line",
        )
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

    def find_furthest_node_path_list(self, connection_node_list: list, vertices_dict: dict, ont_vertice: int) -> list:
        connection_node_dict = {n: vertices_dict[n] for n in connection_node_list}
        furthest_node = max(connection_node_dict, key=connection_node_dict.get)
        # all the connection nodes in the path from transformer to furthest node are considered as potential branch loads
        furthest_node_path_list = self.get_path_to_bus(furthest_node, ont_vertice)
        furthest_node_path = [
            p for p in furthest_node_path_list if p in connection_node_list
        ]

        return furthest_node_path

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

    def get_vertices_from_bcid(self, plz:int, kcid:int, bcid:int) -> tuple[dict, int]:
        ont = self.get_ont_info_from_bc(plz, kcid, bcid)["ont_vertice_id"]

        consumer_query = """SELECT vertice_id FROM buildings_tem
                    WHERE plz = %(p)s 
                    AND kcid = %(k)s
                    AND bcid = %(b)s;"""
        self.cur.execute(consumer_query, {"p": plz, "k": kcid, "b": bcid})
        consumer = [t[0] for t in self.cur.fetchall()]

        connection_query = """SELECT DISTINCT connection_point FROM buildings_tem
                    WHERE plz = %(p)s 
                    AND kcid = %(k)s
                    AND bcid = %(b)s;"""
        self.cur.execute(connection_query, {"p": plz, "k": kcid, "b": bcid})
        connection = [t[0] for t in self.cur.fetchall()]

        vertices_query = """ SELECT DISTINCT node, agg_cost FROM pgr_dijkstra(
                    'SELECT way_id as id, source, target, cost, reverse_cost FROM ways_tem'::text, %(o)s, %(c)s::integer[], false) ORDER BY agg_cost;"""
        self.cur.execute(vertices_query, {"o": ont, "c": consumer})
        data = self.cur.fetchall()
        vertice_cost_dict = {
            t[0]: t[1] for t in data if t[0] in consumer or t[0] in connection
        }

        return vertice_cost_dict, ont

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

    def get_ont_geom_from_bcid(self, plz: int, kcid: int, bcid: int):
        query = """SELECT ST_X(ST_Transform(geom,4326)), ST_Y(ST_Transform(geom,4326)) FROM transformer_positions
                    WHERE version_id = %(v)s 
                    AND plz = %(p)s 
                    AND kcid = %(k)s
                    AND bcid = %(b)s;"""
        self.cur.execute(query, {"v": VERSION_ID, "p": plz, "k": kcid, "b": bcid})
        geo = self.cur.fetchone()

        return geo

    def get_node_geom(self, vid: int):
        query = """SELECT ST_X(ST_Transform(the_geom,4326)), ST_Y(ST_Transform(the_geom,4326)) 
                    FROM ways_tem_vertices_pgr
                    WHERE id = %(id)s;"""
        self.cur.execute(query, {"id": vid})
        geo = self.cur.fetchone()

        return geo

    def get_transformer_rated_power_from_bcid(self, plz: int, kcid: int, bcid: int) -> int:
        query = """SELECT transformer_rated_power FROM building_clusters
                    WHERE version_id = %(v)s 
                    AND plz = %(p)s 
                    AND kcid = %(k)s
                    AND bcid = %(b)s;"""
        self.cur.execute(query, {"v": VERSION_ID, "p": plz, "k": kcid, "b": bcid})
        transformer_rated_power = self.cur.fetchone()[0]

        return transformer_rated_power

    def get_list_from_plz(self, plz: int) -> list:
        query = """SELECT DISTINCT kcid, bcid FROM building_clusters 
                    WHERE  version_id = %(v)s AND plz = %(p)s 
                    ORDER BY kcid, bcid;"""
        self.cur.execute(query, {"p": plz, "v": VERSION_ID})
        cluster_list = self.cur.fetchall()

        return cluster_list

    def get_building_connection_points_from_bc(self, kcid: int, bcid: int) -> list:
        """
        Args:
            kcid: kmeans_cluster ID
            bcid: building cluster ID
        Returns: A dataframe with all building information
        """
        count_query = """SELECT DISTINCT connection_point
                        FROM public.buildings_tem
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
                                FROM public.buildings_tem AS b
                                WHERE b.vertice_id IS NOT NULL
                                    AND b.bcid = %(b)s 
                                    AND b.kcid = %(k)s
                                    LIMIT 1;"""
        params = {"b": bcid, "k": kcid}
        self.cur.execute(count_query, params)
        conn_id = self.cur.fetchone()[0]

        return conn_id

    def get_distance_matrix_from_kcid(self, kcid: int) -> tuple[dict, np.ndarray, dict]:
        """
        Creates a distance matrix from the buildings in the kcid
        Args:
            kcid: k-means cluster id
        Returns: The distance matrix of the buildings as np.array and the mapping between vertice_id and local ID as dict
        """

        costmatrix_query = """SELECT * FROM pgr_dijkstraCostMatrix(
                            'SELECT way_id as id, source, target, cost, reverse_cost FROM public.ways_tem',
                            (SELECT array_agg(DISTINCT b.connection_point) FROM (SELECT * FROM buildings_tem 
                            WHERE kcid = %(k)s
                            AND bcid ISNULL
                            ORDER BY connection_point) AS b),
                            false);"""
        params = {"k": kcid}

        return self._calculate_cost_arr_dist_matrix(costmatrix_query, params)

    def get_distance_matrix_from_building_cluster(self, kcid: int, bcid: int) -> tuple[dict, np.ndarray, dict]:
        """
        Args:
            kcid: k mean cluster ID
            bcid: building cluster ID
        Returns: Die Distanzmatrix der Gebäuden als np.array und das Mapping zwischen vertice_id und lokale ID als dict
        """
        # Creates a distance matrix from the buildings in the postcode cluster or smaller in the building cluster

        costmatrix_query = """SELECT * FROM pgr_dijkstraCostMatrix(
                            'SELECT way_id as id, source, target, cost, reverse_cost FROM public.ways_tem',
                            (SELECT array_agg(DISTINCT b.connection_point) FROM (SELECT * FROM buildings_tem 
                                WHERE kcid = %(k)s
                                AND bcid = %(b)s
                                ORDER BY connection_point) AS b),
                            false);"""
        params = {"b": bcid, "k": kcid}

        return self._calculate_cost_arr_dist_matrix(costmatrix_query, params)

    def upsert_building_cluster(self, plz: int, kcid: int, bcid: int, vertices: list, transformer_rated_power: int):
        """
        Assign buildings in buildings_tem the bcid and stores the cluster in building_clusters
        Args:
            plz: postcode cluster ID - plz
            kcid: kmeans cluster ID
            bcid: building cluster ID
            vertices: List of vertice_id of selected buildings
            transformer_rated_power: Apparent power of the selected transformer
        """
        # Insert references to building elements in which cluster they are.
        building_query = """UPDATE public.buildings_tem 
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
        cluster_query = """INSERT INTO building_clusters (version_id, plz, kcid, bcid, transformer_rated_power) 
                VALUES(%(v)s, %(pc)s, %(kc)s, %(bc)s, %(s)s); """

        params = {"v": VERSION_ID, "pc": plz, "bc": bcid, "kc": kcid, "s": int(transformer_rated_power)}
        self.cur.execute(cluster_query, params)

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

    def clear_building_clusters_in_kmean_cluster(self, plz: int, kcid: int):
        # Remove old clustering at same postcode cluster
        clear_query = """DELETE FROM building_clusters
                WHERE  version_id = %(v)s 
                AND plz = %(pc)s
                AND kcid = %(kc)s
                AND bcid >= 0; """

        params = {"v": VERSION_ID, "pc": plz, "kc": kcid}
        self.cur.execute(clear_query, params)
        self.logger.debug(
            f"Building clusters with plz = {plz}, k_mean cluster = {kcid} area cleared."
        )

    def upsert_transformer_selection(self, plz: int, kcid: int, bcid: int, connection_id: int):
        """Writes the vertice_id of chosen building as ONT location in the building_clusters table"""

        query = """UPDATE building_clusters SET ont_vertice_id = %(c)s 
                    WHERE version_id = %(v)s AND plz = %(p)s AND kcid = %(k)s AND bcid = %(b)s; 
                    UPDATE building_clusters SET model_status = 1 
                    WHERE version_id = %(v)s AND plz = %(p)s AND kcid = %(k)s AND bcid = %(b)s;
                INSERT INTO transformer_positions (version_id, plz, kcid, bcid, geom, ogc_fid, comment)
                    VALUES (%(v)s, %(p)s, %(k)s, %(b)s, (SELECT the_geom FROM ways_tem_vertices_pgr WHERE id = %(c)s), %(c)s::varchar, 'on_way');"""
        params = {"v": VERSION_ID, "c": connection_id, "b": bcid, "k": kcid, "p": plz}

        self.cur.execute(query, params)

    def count_kmean_cluster_consumers(self, kcid: int) -> int:
        query = """SELECT COUNT(DISTINCT vertice_id) FROM buildings_tem WHERE kcid = %(k)s AND type != 'Transformer' AND bcid ISNULL;"""
        self.cur.execute(query, {"k": kcid})
        count = self.cur.fetchone()[0]

        return count

    def update_transformer_rated_power(self, plz: int, kcid: int, bcid: int, note: int):
        """
        Updates Smax in building_clusters
        :param plz:
        :param kcid:
        :param bcid:
        :param note:
        :return:
        """
        sdl = self.get_settlement_type_from_plz(plz)
        transformer_capacities, _ = self.get_transformer_data(sdl)

        if note == 0:
            old_query = """SELECT transformer_rated_power FROM building_clusters
                            WHERE  version_id = %(v)s
                            AND plz = %(p)s
                            AND kcid = %(k)s
                            AND bcid = %(b)s;"""
            self.cur.execute(
                old_query, {"v": VERSION_ID, "p": plz, "k": kcid, "b": bcid}
            )
            transformer_rated_power = self.cur.fetchone()[0]

            new_transformer_rated_power = transformer_capacities[transformer_capacities > transformer_rated_power][0].item()
            update_query = """UPDATE building_clusters SET transformer_rated_power = %(n)s
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
            old_query = """SELECT transformer_rated_power FROM building_clusters
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
            update_query = """UPDATE building_clusters SET transformer_rated_power = %(n)s
                                            WHERE version_id = %(v)s 
                                            AND plz = %(p)s 
                                            AND kcid = %(k)s 
                                            AND bcid = %(b)s;"""
            self.cur.execute(
                update_query,
                {"v": VERSION_ID, "p": plz, "k": kcid, "b": bcid, "n": new_transformer_rated_power},
            )
            self.logger.debug("double or multiple transformer group transformer_rated_power assigned")


    def get_greenfield_bcids(self, plz: int, kcid: int) -> list:
        """
        Args:
            plz: loadarea cluster ID
            kcid: kmeans cluster ID
        Returns: A list of greenfield building clusters for a given plz
        """
        query = """SELECT DISTINCT bcid FROM building_clusters
            WHERE version_id = %(v)s 
            AND kcid = %(kc)s
            AND plz = %(pc)s
            AND model_status ISNULL
            ORDER BY bcid; """
        params = {"v": VERSION_ID, "pc": plz, "kc": kcid}
        self.cur.execute(query, params)
        bcid_list = [t[0] for t in data] if (data := self.cur.fetchall()) else []
        return bcid_list


    def get_ont_info_from_bc(self, plz: int, kcid: int, bcid: int) -> dict | None:

        query = """SELECT ont_vertice_id, transformer_rated_power
                    FROM building_clusters
                    WHERE version_id = %(v)s 
                    AND kcid = %(k)s
                    AND bcid = %(b)s
                    AND plz = %(p)s; """
        params = {"v": VERSION_ID, "p": plz, "k": kcid, "b": bcid}
        self.cur.execute(query, params)
        info = self.cur.fetchall()
        if not info:
            self.logger.debug(f"found no ont information for kcid {kcid}, bcid {bcid}")
            return None

        return {"ont_vertice_id": info[0][0], "transformer_rated_power": info[0][1]}

    def get_cables(self,  anw: tuple) -> pd.DataFrame:
        query = """SELECT name, max_i_a, r_mohm_per_km, x_mohm_per_km, z_mohm_per_km, cost_eur FROM equipment_data
                    WHERE typ = 'Cable' AND application_area IN %(a)s ORDER BY max_i_a DESC; """
        cables_df = pd.read_sql_query(query, self.conn, params={"a": anw})
        self.logger.debug(f"{len(cables_df)} different cable types are imported...")
        return cables_df

    def get_next_unfinished_kcid(self, plz: int) -> int:
        """
        :return: one unmodeled k mean cluster ID - plz
        """
        query = """SELECT kcid FROM buildings_tem 
                    WHERE kcid NOT IN (
                        SELECT DISTINCT kcid FROM building_clusters
                        WHERE version_id = %(v)s AND  building_clusters.plz = %(plz)s) AND kcid IS NOT NULL
                    ORDER BY kcid
                    LIMIT 1;"""
        self.cur.execute(query, {"v": VERSION_ID, "plz": plz})
        kcid = self.cur.fetchone()[0]
        return kcid

    def count_no_kmean_buildings(self):
        """
        Counts relative buildings in buildings_tem, which could not be clustered via k-means
        :return: count
        """
        query = """SELECT COUNT(*) FROM buildings_tem WHERE peak_load_in_kw != 0 AND kcid ISNULL;"""
        self.cur.execute(query)
        count = self.cur.fetchone()[0]

        return count

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

    def get_kcid_length(self) -> int:
        query = """SELECT COUNT(DISTINCT kcid) FROM buildings_tem WHERE kcid IS NOT NULL; """
        self.cur.execute(query)
        kcid_length = self.cur.fetchone()[0]
        return kcid_length

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
                self.delete_transformers([transformer_id])
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
            pgr: PostgreSQL reader/writer instance
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
          - Insert a new record into 'building_clusters'.
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

            INSERT INTO building_clusters (version_id, plz, kcid, bcid, ont_vertice_id, transformer_rated_power)
            VALUES (%(v)s, %(pc)s, %(k)s, %(count)s, %(t)s, %(l)s);

            INSERT INTO transformer_positions (version_id, plz, kcid, bcid, geom, ogc_fid, comment)
            VALUES (
                %(v)s, %(pc)s, %(k)s, %(count)s,
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

    def connect_unconnected_ways(self) -> None:
        """
        Updates ways_tem
        :return:
        """
        query = """SELECT public.draw_way_connections();"""
        self.cur.execute(query)

    def calculate_sim_load(self, conn_list: Union[tuple, list]) -> Decimal:
        residential = """WITH residential AS 
        (SELECT b.peak_load_in_kw AS load, b.houses_per_building AS count, c.sim_factor
        FROM buildings_tem AS b
        LEFT JOIN consumer_categories AS c
        ON b.type = c.definition
        WHERE b.connection_point IN %(c)s AND b.type IN ('SFH','MFH','AB','TH')
        )
        SELECT SUM(load), SUM(count), sim_factor FROM residential GROUP BY sim_factor;
        """
        self.cur.execute(residential, {"c": tuple(conn_list)})

        data = self.cur.fetchone()
        if data:
            residential_load = Decimal(data[0])
            residential_count = Decimal(data[1])
            residential_factor = Decimal(data[2])
            residential_sim_load = residential_load * (
                    residential_factor
                    + (1 - residential_factor) * (residential_count ** Decimal(-3 / 4))
            )
        else:
            residential_sim_load = 0
        # TODO can the following 4 repetitions simplified with a general function?
        commercial = """WITH commercial AS 
                (SELECT b.peak_load_in_kw AS load, b.houses_per_building AS count, c.sim_factor
                FROM buildings_tem AS b
                LEFT JOIN consumer_categories AS c 
                ON c.definition = b.type
                WHERE b.connection_point IN %(c)s AND b.type = 'Commercial'
                )
                SELECT SUM(load), SUM(count), sim_factor FROM commercial GROUP BY sim_factor;
                """
        self.cur.execute(commercial, {"c": tuple(conn_list)})
        data = self.cur.fetchone()
        if data:
            commercial_load = Decimal(data[0])
            commercial_count = Decimal(data[1])
            commercial_factor = Decimal(data[2])
            commercial_sim_load = commercial_load * (
                    commercial_factor
                    + (1 - commercial_factor) * (commercial_count ** Decimal(-3 / 4))
            )
        else:
            commercial_sim_load = 0

        public = """WITH public AS 
                    (SELECT b.peak_load_in_kw AS load, b.houses_per_building AS count, c.sim_factor 
                    FROM buildings_tem AS b 
                    LEFT JOIN consumer_categories AS c 
                    ON c.definition = b.type
                    WHERE b.connection_point IN %(c)s AND b.type = 'Public')
                    SELECT SUM(load), SUM(count), sim_factor FROM public GROUP BY sim_factor;
                        """
        self.cur.execute(public, {"c": tuple(conn_list)})
        data = self.cur.fetchone()
        if data:
            public_load = Decimal(data[0])
            public_count = Decimal(data[1])
            public_factor = Decimal(data[2])
            public_sim_load = public_load * (
                    public_factor + (1 - public_factor) * (public_count ** Decimal(-3 / 4))
            )
        else:
            public_sim_load = 0

        industrial = """WITH industrial AS 
                    (SELECT b.peak_load_in_kw AS load, b.houses_per_building AS count, c.sim_factor FROM buildings_tem AS b
                     LEFT JOIN consumer_categories AS c 
                     ON c.definition = b.type
                     WHERE b.connection_point IN %(c)s AND b.type = 'Industrial')
                     SELECT SUM(load), SUM(count), sim_factor FROM industrial GROUP BY sim_factor;
                                """
        self.cur.execute(industrial, {"c": tuple(conn_list)})
        data = self.cur.fetchone()
        if data:
            industrial_load = Decimal(data[0])
            industrial_count = Decimal(data[1])
            industrial_factor = Decimal(data[2])
            industrial_sim_load = industrial_load * (
                    industrial_factor
                    + (1 - industrial_factor) * (industrial_count ** Decimal(-3 / 4))
            )
        else:
            industrial_sim_load = 0

        total_sim_load = (
                residential_sim_load
                + commercial_sim_load
                + industrial_sim_load
                + public_sim_load
        )

        return total_sim_load

    def copy_postcode_result_table(self, plz: int) -> None:
        """
        Copies the given plz entry from postcode to the postcode_result table
        :param plz:
        :return:
        """
        query = """INSERT INTO postcode_result (version_id, postcode_result_id, geom) 
                    SELECT %(v)s as version_id, plz, geom FROM postcode 
                    WHERE plz = %(p)s
                    LIMIT 1 
                    ON CONFLICT (version_id,postcode_result_id) DO NOTHING;"""

        self.cur.execute(query, {"v": VERSION_ID, "p": plz})

    def count_postcode_result(self, plz: int) -> int:
        """
        :param plz:
        :return:
        """
        query = """SELECT COUNT(*) FROM postcode_result
                    WHERE version_id = %(v)s
                    AND postcode_result_id::INT = %(p)s"""
        self.cur.execute(query, {"v": VERSION_ID, "p": plz})
        return int(self.cur.fetchone()[0])

    def count_clustering_parameters(self, plz: int) -> int:
        """
        :param plz:
        :return:
        """
        query = """SELECT COUNT(*) FROM clustering_parameters
                    WHERE version_id = %(v)s
                    AND plz = %(p)s"""
        self.cur.execute(query, {"v": VERSION_ID, "p": plz})
        return int(self.cur.fetchone()[0])

    def set_plz_settlement_type(self, plz: int) -> None:
        """
        Determine settlement_type in postcode_result table based on the house_distance metric for a given plz
        :param plz: Postleitzahl (postal code)
        :return: None
        """
        # Get average distance between buildings by sampling 50 random buildings
        # and finding their 4 nearest neighbors
        distance_query = """WITH some_buildings AS(
                            SELECT osm_id, center FROM buildings_tem
                            ORDER BY RANDOM()
                            LIMIT 50) 
                        SELECT b.osm_id, d.dist
                        FROM some_buildings AS b
                        LEFT JOIN LATERAL(
                        SELECT ST_Distance(b.center, b2.center) AS dist
                        FROM buildings_tem AS b2
                        WHERE b.osm_id <> b2.osm_id
                        ORDER BY b.center <-> b2.center
                        LIMIT 4) AS d
                        ON TRUE;"""
        self.cur.execute(distance_query)
        data = self.cur.fetchall()

        if not data:
            raise ValueError("There is no building in the buildings_tem table!")

        # Calculate average distance
        distance = [t[1] for t in data]
        avg_dis = int(sum(distance) / len(distance))

        # Update database with average distance and set settlement types based on threshold
        query = """
            UPDATE postcode_result
            SET house_distance = %(avg)s, 
                settlement_type = CASE
                    WHEN %(avg)s < 25 THEN 3
                    WHEN %(avg)s < 45 THEN 2
                    ELSE 1
                END
            WHERE version_id = %(v)s 
            AND postcode_result_id = %(p)s;"""

        self.cur.execute(query, {"v": VERSION_ID, "avg": avg_dis, "p": plz})

    def set_building_peak_load(self) -> int:
        """
        * Sets the area, type and peak_load in the buildings_tem table
        * Removes buildings with zero load from the buildings_tem table
        :return: Number of removed unloaded buildings from buildings_tem
        """
        query = """
            UPDATE buildings_tem SET area = ST_Area(geom);
            UPDATE buildings_tem SET houses_per_building = (CASE
            WHEN type IN ('TH','Commercial','Public','Industrial') THEN 1
            WHEN type = 'SFH' AND area < 160 THEN 1
            WHEN type = 'SFH' AND area >= 160 THEN 2
            WHEN type IN ('MFH','AB') THEN floor(area/50) * floors
            ELSE 0
            END);
            UPDATE buildings_tem b SET peak_load_in_kw = (CASE
            WHEN b.type IN ('SFH','TH','MFH','AB') THEN b.houses_per_building*(SELECT peak_load FROM consumer_categories WHERE definition = b.type)								  
            WHEN b.type IN ('Commercial','Public','Industrial') THEN b.area*(SELECT peak_load_per_m2 FROM consumer_categories WHERE definition = b.type)/1000 
            ELSE 0
            END);"""
        self.cur.execute(query)

        count_query = (
            """SELECT COUNT(*) FROM buildings_tem WHERE peak_load_in_kw = 0;"""
        )
        self.cur.execute(count_query)
        count = self.cur.fetchone()[0]

        delete_query = """DELETE FROM buildings_tem WHERE peak_load_in_kw = 0;"""
        self.cur.execute(delete_query)

        return count

    def set_residential_buildings_table(self, plz: int):
        """
        * Fills buildings_tem with residential buildings which are inside the plz area
        :param plz:
        :return:
        """

        # Fill table
        query = """INSERT INTO buildings_tem (osm_id, area, type, geom, center, floors)
                SELECT osm_id, area, building_t, geom, ST_Centroid(geom), floors::int FROM res
                WHERE ST_Contains((SELECT post.geom FROM postcode_result as post WHERE version_id = %(v)s
                AND postcode_result_id = %(plz)s LIMIT 1), ST_Centroid(res.geom));
                UPDATE buildings_tem SET plz = %(plz)s WHERE plz ISNULL;"""
        self.cur.execute(query, {"v": VERSION_ID, "plz": plz})

    def set_other_buildings_table(self, plz: int):
        """
        * Fills buildings_tem with other buildings which are inside the plz area
        * Sets all floors to 1
        :param plz:
        :return:
        """

        # Fill table
        query = """INSERT INTO buildings_tem(osm_id, area, type, geom, center)
                SELECT osm_id, area, use, geom, ST_Centroid(geom) FROM oth AS o 
                WHERE o.use in ('Commercial', 'Public')
                AND ST_Contains((SELECT post.geom FROM postcode_result as post WHERE version_id = %(v)s
                    AND  postcode_result_id = %(plz)s), ST_Centroid(o.geom));;
            UPDATE buildings_tem SET plz = %(plz)s WHERE plz ISNULL;
            UPDATE buildings_tem SET floors = 1 WHERE floors ISNULL;"""
        self.cur.execute(query, {"v": VERSION_ID, "plz": plz})


    def set_residential_buildings_table_from_osmid(self, plz: int, buildings: list) -> None:
        """
        * Fills buildings_tem with residential buildings which are inside the selected polygon
        * Sets the postcode cluster to first plz that intersects
        :param shape:
        :return:
        """

        # Fill table
        query = """INSERT INTO buildings_tem (osm_id, area, type, geom, center, floors)
                SELECT osm_id, area, building_t, geom, ST_Centroid(geom), floors::int FROM res
                WHERE res.osm_id = ANY(%(buildings)s);
                UPDATE buildings_tem SET plz = %(p)s WHERE plz ISNULL;"""
        self.cur.execute(query, {"v": VERSION_ID, "p": plz, "buildings": buildings})

    def set_other_buildings_table_from_osmid(self, plz: int, buildings: list) -> None:
        """
        * Fills buildings_tem with other buildings which are inside the selected polygon
        * Sets the postcode cluster to first plz that intersects shapefile
        * Sets all floors to 1
        :param shape:
        :return:
        """

        # Fill table
        query = """INSERT INTO buildings_tem(osm_id, area, type, geom, center)
                SELECT osm_id, area, use, geom, ST_Centroid(geom) FROM oth AS o 
                WHERE o.use in ('Commercial', 'Public')
                AND o.osm_id = ANY(%(buildings)s);
            UPDATE buildings_tem SET plz = %(p)s WHERE plz ISNULL;
            UPDATE buildings_tem SET floors = 1 WHERE floors ISNULL;"""
        self.cur.execute(query, {"v": VERSION_ID, "p": plz, "buildings": buildings})

    def get_connected_component(self) -> tuple[np.ndarray, np.ndarray]:
        """
        Reads from ways_tem
        :return:
        """
        component_query = """SELECT component,node FROM pgr_connectedComponents(
                'SELECT way_id as id, source, target, cost, reverse_cost FROM ways_tem');"""
        self.cur.execute(component_query)
        data = self.cur.fetchall()
        component = np.asarray([i[0] for i in data])
        node = np.asarray([i[1] for i in data])

        return component, node

    def count_buildings_kcid(self, kcid: int) -> int:
        query = """SELECT COUNT(*) FROM buildings_tem WHERE kcid = %(k)s;"""
        self.cur.execute(query, {"k": kcid})
        count = self.cur.fetchone()[0]

        return count

    def set_ways_tem_table(self, plz: int) -> int:
        """
        * Inserts ways inside the plz area to the ways_tem table
        :param plz:
        :return: number of ways in ways_tem
        """
        query = """INSERT INTO ways_tem
            SELECT * FROM ways AS w 
            WHERE ST_Intersects(w.geom,(SELECT geom FROM postcode_result WHERE version_id = %(v)s
                    AND  postcode_result_id = %(p)s));
            SELECT COUNT(*) FROM ways_tem;"""
        self.cur.execute(query, {"v": VERSION_ID, "p": plz})
        count = self.cur.fetchone()[0]

        if count == 0:
            raise ValueError(f"Ways table is empty for the given plz: {plz}")

        return count

    def set_ways_tem_table_from_shapefile(self, shape) -> int:
        """
        * Inserts ways inside the plz area to the ways_tem table
        :param shape:
        :return: number of ways in ways_tem
        """
        query = """INSERT INTO ways_tem
            SELECT * FROM ways AS w 
            WHERE ST_Intersects(w.geom, ST_Transform(ST_GeomFromGeoJSON(%(shape)s), 3035));
            SELECT COUNT(*) FROM ways_tem;"""
        self.cur.execute(query, {"v": VERSION_ID, "shape": shape})
        count = self.cur.fetchone()[0]

        if count == 0:
            raise ValueError(f"Ways table is empty for the given plz: {shape}")

        return count

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
                        valid_cluster_dict = dict(enumerate(valid_cluster_dict.values()))

                    # Process invalid clusters
                    if invalid_cluster_dict:
                        current_invalid_amount = len(invalid_trans_cluster_dict)
                        invalid_trans_cluster_dict.update(
                            {x + current_invalid_amount: y for x, y in invalid_cluster_dict.items()}
                        )
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
                self.clear_building_clusters_in_kmean_cluster(plz, kcid)
                for bcid, cluster_data in valid_cluster_dict.items():
                    self.upsert_building_cluster(
                        plz, kcid, bcid,
                        vertices=cluster_data[0],
                        transformer_rated_power=cluster_data[1]
                    )
                self.logger.debug(f"bcids for plz {plz} kcid {kcid} found...")

    def draw_building_connection(self) -> None:
        """
        Updates ways_tem, creates pgr network topology in new tables:
        :return:
        """
        connection_query = """ SELECT public.draw_home_connections(); """
        self.cur.execute(connection_query)

        topology_query = """select pgr_createTopology('ways_tem', 0.01, id:='way_id', the_geom:='geom', clean:=true) """
        self.cur.execute(topology_query)

        # add_buildings_query = '''SELECT public.add_buildings();'''
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
                    WHERE ST_Within(t.geom, (SELECT geom FROM postcode_result WHERE postcode_result_id = %(p)s AND version_id = %(v)s)); --IS FALSE;
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

    def set_vertice_id(self) -> int:
        """
        Updates buildings_tem with the vertice_id s from ways_tem_vertices_pgr
        :return:
        """
        query = """UPDATE public.buildings_tem b
                SET vertice_id = (SELECT id FROM ways_tem_vertices_pgr AS v 
                WHERE ST_Equals(v.the_geom,b.center));"""
        self.cur.execute(query)

        query2 = """UPDATE buildings_tem b
                SET connection_point = (SELECT target FROM ways_tem WHERE source = b.vertice_id LIMIT 1)
                WHERE vertice_id IS NOT NULL AND connection_point IS NULL;"""
        self.cur.execute(query2)

        count_query = """ SELECT COUNT(*) FROM buildings_tem
            WHERE connection_point IS NULL AND peak_load_in_kw != 0;"""
        self.cur.execute(count_query)
        count = self.cur.fetchone()[0]

        delete_query = """DELETE FROM buildings_tem WHERE connection_point IS NULL AND peak_load_in_kw != 0;"""
        self.cur.execute(delete_query)

        return count

    def generate_load_vector(self, kcid: int, bcid: int) -> np.ndarray:
        query = """SELECT SUM(peak_load_in_kw)::float FROM buildings_tem 
                WHERE kcid = %(k)s AND bcid = %(b)s 
                GROUP BY connection_point 
                ORDER BY connection_point;"""
        self.cur.execute(query, {"k": kcid, "b": bcid})
        load = np.asarray([i[0] for i in self.cur.fetchall()])

        return load

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

    def count_connected_buildings(self, vertices: Union[list, tuple]) -> int:
        """
        Get count from buildings_tem where type is not transformer
        :param vertices: np.array
        :return: count of buildings with given vertice_id s from buildings_tem
        """
        query = """SELECT COUNT(*) FROM buildings_tem WHERE vertice_id IN %(v)s AND type != 'Transformer';"""
        self.cur.execute(query, {"v": tuple(map(int, vertices))})
        count = self.cur.fetchone()[0]

        return count

    def update_ways_cost(self) -> None:
        """
        Calculates the length of each way and stores in ways_tem.cost as meter
        """
        query = """UPDATE ways_tem SET cost = ST_Length(geom); 
        UPDATE ways_tem SET reverse_cost = cost;"""
        self.cur.execute(query)

    def count_one_building_cluster(self) -> int:
        query = """SELECT COUNT(*) FROM building_clusters bc 
            WHERE (SELECT COUNT(*) FROM buildings_tem b WHERE b.kcid = bc.kcid AND b.bcid = bc.bcid) = 1;"""
        self.cur.execute(query)
        try:
            count = self.cur.fetchone()[0]
        except:
            count = 0

        return count

    def count_indoor_transformers(self) -> None:
        """counts indoor transformers before deleting them"""
        query = f"""WITH union_table (ungeom) AS 
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

    def delete_ways(self, vertices: list) -> None:
        """
        Deletes selected ways from ways_tem and ways_tem_vertices_pgr
        :param vertices:
        :return:
        """
        query = """DELETE FROM ways_tem WHERE target IN %(v)s;
                DELETE FROM ways_tem_vertices_pgr WHERE id IN %(v)s;"""
        self.cur.execute(query, {"v": tuple(map(int, vertices))})

    def delete_transformers(self, vertices: list) -> None:
        """
        Deletes selected transformers from buildings_tem
        :param vertices:
        :return:
        """
        query = """
                DELETE FROM buildings_tem WHERE vertice_id IN %(v)s;"""
        self.cur.execute(query, {"v": tuple(map(int, vertices))})

    def delete_isolated_building(self, plz: int, kcid):
        query = """DELETE FROM buildings_tem WHERE plz = %(p)s
                    AND kcid = %(k)s AND bcid ISNULL;"""
        self.cur.execute(query, {"p": plz, "k": kcid})

    def save_and_reset_tables(self, plz: int):

        # finding duplicates that violate the buildings_result_pkey constraint
        # the key of building result is (version_id, osm_id, plz)
        query = f"""
                DELETE FROM buildings_tem a USING(
                    SELECT MIN(ctid) as ctid, osm_id, plz
                    FROM buildings_tem
                    GROUP BY (osm_id, plz) HAVING COUNT(*) > 1
                    ) b
                WHERE a.osm_id = b.osm_id
                AND a.plz = b.plz
                AND a.ctid <> b.ctid;"""
        self.cur.execute(query)

        # Save building results
        query = f"""
                INSERT INTO buildings_result 
                    SELECT '{VERSION_ID}' as version_id, * FROM buildings_tem WHERE peak_load_in_kw != 0 AND peak_load_in_kw != -1;"""
        self.cur.execute(query)

        # Save ways results
        query = f"""INSERT INTO ways_result
                    SELECT '{VERSION_ID}' as version_id, clazz, source, target, cost, reverse_cost, geom, way_id,
                    %(p)s as plz FROM ways_tem;"""

        self.cur.execute(query, vars={"p": plz})

        # Clear temporary tables
        query = """DELETE FROM buildings_tem"""
        self.cur.execute(query)
        query = """DELETE FROM ways_tem"""
        self.cur.execute(query)
        query = """DELETE FROM ways_tem_vertices_pgr"""
        self.cur.execute(query)

        self.conn.commit()

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

    def insert_version_if_not_exists(self):
        count_query = f"""SELECT COUNT(*) 
        FROM version 
        WHERE "version_id" = '{VERSION_ID}'"""
        self.cur.execute(count_query)
        version_exists = self.cur.fetchone()[0]
        if not version_exists:
            # create new version
            consumer_categories_str = CONSUMER_CATEGORIES.to_json().replace("'", "''")
            cable_cost_dict_str = json.dumps(CABLE_COST_DICT).replace("'", "''")
            connection_available_cables_str = str(CONNECTION_AVAILABLE_CABLES).replace(
                "'", "''"
            )
            other_parameters_dict = {
                "LARGE_COMPONENT_LOWER_BOUND": LARGE_COMPONENT_LOWER_BOUND,
                "LARGE_COMPONENT_DIVIDER": LARGE_COMPONENT_DIVIDER,
                "VN": VN,
                "V_BAND_LOW": V_BAND_LOW,
                "V_BAND_HIGH": V_BAND_HIGH,
            }
            other_paramters_str = str(other_parameters_dict).replace("'", "''")

            insert_query = f"""INSERT INTO version (version_id, version_comment, consumer_categories, cable_cost_dict, connection_available_cables, other_parameters) VALUES
            ('{VERSION_ID}', '{VERSION_COMMENT}', '{consumer_categories_str}', '{cable_cost_dict_str}', '{connection_available_cables_str}', '{other_paramters_str}')"""
            self.cur.execute(insert_query)
            self.logger.info(f"Version: {VERSION_ID} (created for the first time)")

    def insert_parameter_tables(self, consumer_categories: pd.DataFrame):
        with self.sqla_engine.begin() as conn:
            conn.execute(text("DELETE FROM consumer_categories"))
            consumer_categories.to_sql(
                name="consumer_categories", con=conn, if_exists="append", index=False
            )

        self.logger.debug("Parameter tables are inserted")

    def analyse_basic_parameters(self, plz: int):
        cluster_list = self.get_list_from_plz(plz)
        count = len(cluster_list)
        time = 0
        percent = 0

        load_count_dict = {}
        bus_count_dict = {}
        cable_length_dict = {}
        trafo_dict = {}
        self.logger.debug("start basic parameter counting")
        for kcid, bcid in cluster_list:
            load_count = 0
            bus_list = []
            try:
                net = self.read_net(plz, kcid, bcid)
                # net = pp.from_json(os.path.join(RESULT_DIR, "grid_genereation_result", file))
            except Exception as e:
                self.logger.warning(f" local network {kcid},{bcid} is problematic")
                raise e
            else:
                for row in net.load[["name", "bus"]].itertuples():
                    load_count += 1
                    bus_list.append(row.bus)
                bus_list = list(set(bus_list))
                bus_count = len(bus_list)
                cable_length = net.line['length_km'].sum()

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
                self.logger.info(f"{percent} percent finished")
                time = 0
        self.logger.info("analyse_basic_parameters finished.")
        trafo_string = json.dumps(trafo_dict)
        load_count_string = json.dumps(load_count_dict)
        bus_count_string = json.dumps(bus_count_dict)

        self.insert_grid_parameters(plz, trafo_string, load_count_string, bus_count_string)


    def insert_grid_parameters(self, plz: int, trafo_string: str, load_count_string: str, bus_count_string: str):
        update_query = """INSERT INTO public.grid_parameters (version_id, plz, trafo_num, load_count_per_trafo, bus_count_per_trafo)
        VALUES(%s, %s, %s, %s, %s);"""  # TODO: check - should values be updated for same plz and version if analysis is started? And Add a column
        self.cur.execute(
            update_query,
            vars=(VERSION_ID, plz, trafo_string, load_count_string, bus_count_string),
        )

        self.logger.debug("basic parameter count finished")

    def analyse_cables(self, plz:int):
        cluster_list = self.get_list_from_plz(plz)
        count = len(cluster_list)
        time = 0
        percent = 0

        # distributed according to cross_section
        cable_length_dict = {}
        for kcid, bcid in cluster_list:
            try:
                net = self.read_net(plz, kcid, bcid)
            except Exception as e:
                self.logger.debug(f" local network {kcid},{bcid} is problematic")
                raise e
            else:
                cable_df = net.line[net.line["in_service"] == True]

                cable_type = pd.unique(cable_df["std_type"]).tolist()
                for type in cable_type:

                    if type in cable_length_dict:
                        cable_length_dict[type] += (
                                cable_df[cable_df["std_type"] == type]["parallel"]
                                * cable_df[cable_df["std_type"] == type]["length_km"]
                        ).sum()

                    else:
                        cable_length_dict[type] = (
                                cable_df[cable_df["std_type"] == type]["parallel"]
                                * cable_df[cable_df["std_type"] == type]["length_km"]
                        ).sum()
            time += 1
            if time / count >= 0.1:
                percent += 10
                self.logger.info(f"{percent} % processed")
                time = 0
        self.logger.info("analyse_cables finished.")
        cable_length_string = json.dumps(cable_length_dict)

        update_query = """UPDATE public.grid_parameters
        SET cable_length = %(c)s 
        WHERE version_id = %(v)s AND plz = %(p)s;"""
        self.cur.execute(
            update_query, {"v": VERSION_ID, "c": cable_length_string, "p": plz}
        )  # TODO: change to cable_length_per_type, add cable_length_per_trafo

        self.logger.debug("cable count finished")

    def analyse_per_trafo_parameters(self, plz: int):
        cluster_list = self.get_list_from_plz(plz)
        count = len(cluster_list)
        time = 0
        percent = 0

        trafo_load_dict = {}
        trafo_max_distance_dict = {}
        trafo_avg_distance_dict = {}

        for kcid, bcid in cluster_list:
            try:
                net = self.read_net(plz, kcid, bcid)
            except Exception as e:
                self.logger.warning(f" local network {kcid},{bcid} is problematic")
                raise e
            else:
                trafo_sizes = net.trafo["sn_mva"].tolist()[0]

                load_bus = pd.unique(net.load["bus"]).tolist()

                top.create_nxgraph(net, respect_switches=False)
                trafo_distance_to_buses = (
                    top.calc_distance_to_bus(
                        net,
                        net.trafo["lv_bus"].tolist()[0],
                        weight="weight",
                        respect_switches=False,
                    )
                    .loc[load_bus]
                    .tolist()
                )

                # calculate total sim_peak_load
                residential_bus_index = net.bus[
                    ~net.bus["zone"].isin(["Commercial", "Public"])
                ].index.tolist()
                commercial_bus_index = net.bus[
                    net.bus["zone"] == "Commercial"
                    ].index.tolist()
                public_bus_index = net.bus[net.bus["zone"] == "Public"].index.tolist()

                residential_house_num = net.load[
                    net.load["bus"].isin(residential_bus_index)
                ].shape[0]
                public_house_num = net.load[
                    net.load["bus"].isin(public_bus_index)
                ].shape[0]
                commercial_house_num = net.load[
                    net.load["bus"].isin(commercial_bus_index)
                ].shape[0]

                residential_sum_load = (
                        net.load[net.load["bus"].isin(residential_bus_index)][
                            "max_p_mw"
                        ].sum()
                        * 1e3
                )
                public_sum_load = (
                        net.load[net.load["bus"].isin(public_bus_index)]["max_p_mw"].sum()
                        * 1e3
                )
                commercial_sum_load = (
                        net.load[net.load["bus"].isin(commercial_bus_index)][
                            "max_p_mw"
                        ].sum()
                        * 1e3
                )

                sim_peak_load = 0
                for building_type, sum_load, house_num in zip(
                        ["Residential", "Public", "Commercial"],
                        [residential_sum_load, public_sum_load, commercial_sum_load],
                        [residential_house_num, public_house_num, commercial_house_num],
                ):
                    if house_num:
                        sim_peak_load += utils.oneSimultaneousLoad(
                            installed_power=sum_load,
                            load_count=house_num,
                            sim_factor=SIM_FACTOR[building_type],
                        )

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
                self.logger.info(f"{percent} % processed")
                time = 0
        self.logger.info("analyse_per_trafo_parameters finished.")
        trafo_load_string = json.dumps(trafo_load_dict)
        trafo_max_distance_string = json.dumps(trafo_max_distance_dict)
        trafo_avg_distance_string = json.dumps(trafo_avg_distance_dict)

        update_query = """UPDATE public.grid_parameters
        SET sim_peak_load_per_trafo = %(l)s, max_distance_per_trafo = %(m)s, avg_distance_per_trafo = %(a)s
        WHERE version_id = %(v)s AND plz = %(p)s;
        """
        self.cur.execute(
            update_query,
            {
                "v": VERSION_ID,
                "p": plz,
                "l": trafo_load_string,
                "m": trafo_max_distance_string,
                "a": trafo_avg_distance_string,
            },
        )

        self.logger.debug("per trafo analysis finished")

    def read_trafo_dict(self, plz: int) -> dict:
        read_query = """SELECT trafo_num FROM public.grid_parameters 
        WHERE version_id = %(v)s AND plz = %(p)s;"""
        self.cur.execute(read_query, {"v": VERSION_ID, "p": plz})
        trafo_num_dict = self.cur.fetchall()[0][0]

        return trafo_num_dict

    def read_per_trafo_dict(self, plz: int) -> tuple[list[dict], list[str], dict]:
        read_query = """SELECT load_count_per_trafo, bus_count_per_trafo, sim_peak_load_per_trafo,
        max_distance_per_trafo, avg_distance_per_trafo FROM public.grid_parameters 
        WHERE version_id = %(v)s AND plz = %(p)s;"""
        self.cur.execute(read_query, {"v": VERSION_ID, "p": plz})
        result = self.cur.fetchall()

        # Sort all parameters according to transformer size
        load_dict = dict(sorted(result[0][0].items(), key=lambda x: int(x[0])))
        bus_dict = dict(sorted(result[0][1].items(), key=lambda x: int(x[0])))
        peak_dict = dict(sorted(result[0][2].items(), key=lambda x: int(x[0])))
        max_dict = dict(sorted(result[0][3].items(), key=lambda x: int(x[0])))
        avg_dict = dict(sorted(result[0][4].items(), key=lambda x: int(x[0])))

        trafo_dict = dict(sorted(self.read_trafo_dict(plz).items(), key=lambda x: int(x[0]), reverse=True))
        # Create list with all parameter dicts
        data_list = [load_dict, bus_dict, peak_dict, max_dict, avg_dict]
        data_labels = ['Load Number [-]', 'Bus Number [-]', 'Simultaneous peak load [kW]',
                       'Max. Trafo-Distance [m]', 'Avg. Trafo-Distance [m]']

        return data_list, data_labels, trafo_dict

    def read_cable_dict(self, plz: int) -> dict:
        read_query = """SELECT cable_length FROM public.grid_parameters
        WHERE version_id = %(v)s AND plz = %(p)s;"""
        self.cur.execute(read_query, {"v": VERSION_ID, "p": plz})
        cable_length = self.cur.fetchall()[0][0]

        return cable_length

    def save_net(self, plz:int, kcid:int, bcid:int, json_string:str) -> None:
        insert_query = "INSERT INTO grids VALUES (%s, %s, %s, %s, %s)"
        self.cur.execute(insert_query, vars=(VERSION_ID, plz, kcid, bcid, json_string))

    def read_net(self, plz: int, kcid: int, bcid: int) -> pp.pandapowerNet:
        """
        Reads a pandapower network from the database for the specified grid.

        Args:
            plz: Postal code ID
            kcid: Kmeans cluster ID
            bcid: Building cluster ID

        Returns:
            A pandapower network object

        Raises:
            ValueError: If the requested grid does not exist in the database
        """
        read_query = "SELECT grid FROM grids WHERE version_id = %s AND plz = %s AND kcid = %s AND bcid = %s LIMIT 1"
        self.cur.execute(read_query, vars=(VERSION_ID, plz, kcid, bcid))

        result = self.cur.fetchall()
        if not result:
            self.logger.error(f"Grid not found for plz={plz}, kcid={kcid}, bcid={bcid}, version_id={VERSION_ID}")
            raise ValueError(f"Grid not found for plz={plz}, kcid={kcid}, bcid={bcid}")

        grid_tuple = result[0]
        grid_dict = grid_tuple[0]
        grid_json_string = json.dumps(grid_dict)
        net = pp.from_json_string(grid_json_string)

        return net

    # Getter functions with Geopandas

    def get_geo_df(
            self,
            table: str,
            **kwargs,
    ) -> gpd.GeoDataFrame:
        """
        Args:
            **kwargs: equality filters matching with the table column names
        Returns: A geodataframe with all building information
        :param table: table name
        """
        if kwargs:
            filters = " AND " + " AND ".join(
                [f"{key} = {value}" for key, value in kwargs.items() if key != 'version_id']
            )
        else:
            filters = ""
        query = (
                f"""SELECT * FROM public.{table} 
                    WHERE version_id = %(v)s """
                + filters
        )
        version = VERSION_ID
        if 'version_id' in kwargs:
            version = kwargs.get('version_id')

        params = {"v": version}
        with self.sqla_engine.begin() as connection:
            gdf = gpd.read_postgis(query, con=connection, params=params)

        return gdf

    def get_grid_versions_with_plz(self, plz: int) -> list[tuple]:
        query = (
            f"""SELECT DISTINCT version_id FROM grids WHERE plz = %(p)s"""
        )
        self.cur.execute(query, {"p": plz})
        result = self.cur.fetchall()
        return result

    def get_grids_of_version(self, plz: int, version_id: str) -> list[tuple]:
        query = (
            f"""SELECT kcid, bcid, grid
                FROM grids 
                WHERE plz = %(p)s AND version_id = %(v)s""")
        self.cur.execute(query, {"p": plz, "v": version_id})
        result = self.cur.fetchall()
        return result

    def get_grids_from_plz(self, plz: int) -> pd.DataFrame:
        grids_query = """SELECT * FROM grids
                        WHERE plz = %(p)s"""
        params = {"p": plz}
        grids_df = pd.read_sql_query(grids_query, con=self.conn, params=params)
        self.logger.debug(f"{len(grids_df)} grid data fetched.")

        return grids_df

    def copy_postcode_result_table_with_new_shape(self, plz: int, shape) -> None:
        """
        Copies the given plz entry from postcode to the postcode_result table
        :param plz:
        :return:
        """
        query = """INSERT INTO postcode_result (version_id, postcode_result_id, settlement_type, geom, house_distance) 
                    VALUES(%(v)s, %(p)s::INT, null, ST_Multi(ST_Transform(ST_GeomFromGeoJSON(%(shape)s), 3035)), null)
                    ON CONFLICT (version_id,postcode_result_id) DO NOTHING;"""

        self.cur.execute(query, {"v": VERSION_ID, "p": plz, "shape": shape})

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
                    version_id, 
                    geom,
                    plz,
                    bcid,
                    kcid,
                    line_name,
                    std_type,
                    from_bus,
                    to_bus,
                    length_km
                    ) 
                VALUES(
                %(v)s, 
                ST_SetSRID(%(geom)s::geometry,3035),
                %(plz)s, 
                %(bcid)s, 
                %(kcid)s, 
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

    def delete_plz_from_all_tables(self, plz: int, version_id: str) -> None:
        """
        Deletes all entries of corresponding networks in all tables for the given Version ID and plz.
        :param plz: Postal code
        :param version_id: Version ID
        """
        tables = [
            "building_clusters", "buildings_result", "grid_parameters", "grids",
            "lines_result", "transformer_positions", "ways_result"
        ]

        for table in tables:
            query = f"DELETE FROM {table} WHERE version_id = %(v)s AND plz = %(p)s;"
            self.cur.execute(query, {"v": version_id, "p": plz})
            self.conn.commit()

        query = """DELETE FROM postcode_result
        WHERE version_id = %(v)s AND postcode_result_id = %(p)s;"""
        self.cur.execute(query, {"v": version_id, "p": int(plz)})
        self.conn.commit()
        self.logger.info(f"All data for PLZ {plz} and version {version_id} deleted")

    def delete_version_from_all_tables(self, version_id: str) -> None:
        """Delete all entries of the given version ID from all tables."""
        tables = [
            "building_clusters", "buildings_result", "grid_parameters", "grids",
            "lines_result", "postcode_result", "transformer_positions", "ways_result", "version"
        ]
        for table in tables:
            query = f"DELETE FROM {table} WHERE version_id = %(v)s;"
            self.cur.execute(query, {"v": version_id})
            self.conn.commit()
        self.logger.info(f"Version {version_id} deleted from all tables")

    def get_clustering_parameters_for_plz_list(self, plz_tuple: tuple) -> pd.DataFrame:
        """get clustering parameter for multiple plz"""
        query = """
                WITH plz_table(plz) AS (
                    VALUES (%(p)s)
                    ),
                clustering AS(
                    SELECT * 
                    FROM public.clustering_parameters 
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

    def get_municipal_register_for_plz(self, plz: str) -> pd.DataFrame:
        """get entry of table municipal register for given PLZ"""
        query = """SELECT * 
        FROM public.municipal_register
        WHERE plz = %(p)s;"""
        self.cur.execute(query, {"p": plz})
        register = self.cur.fetchall()
        df_register = pd.DataFrame(register, columns=MUNICIPAL_REGISTER)
        return df_register

    def get_municipal_register(self) -> pd.DataFrame:
        """get municipal register """
        query = """SELECT * 
        FROM public.municipal_register;"""
        self.cur.execute(query)
        register = self.cur.fetchall()
        df_register = pd.DataFrame(register, columns=MUNICIPAL_REGISTER)
        return df_register

    def get_ags_log(self) -> pd.DataFrame:
        """get ags log: the amtliche gemeindeschluessel of the municipalities of which the buildings
        have already been imported to the database
        :return: table with column of ags
        :rtype: DataFrame
         """
        query = """SELECT * 
        FROM public.ags_log;"""
        df_query = pd.read_sql_query(query, con=self.conn, )
        return df_query

    def write_ags_log(self, ags: int) -> None:
        """write ags log to database: the amtliche gemeindeschluessel of the municipalities of which the buildings
        have already been imported to the database
        :param ags:  ags to be added
        :rtype ags: numpy integer 64
         """
        query = """INSERT INTO ags_log (ags) 
                VALUES(%(a)s); """
        self.cur.execute(query, {"a": int(ags), })
        self.conn.commit()

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