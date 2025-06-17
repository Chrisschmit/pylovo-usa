import json
import warnings

import geopandas as gpd
import pandapower.topology as top

from src import utils
from src.config_loader import *

warnings.simplefilter(action="ignore", category=UserWarning)


class AnalysisMixin:

    def insert_plz_parameters(self, plz: int, trafo_string: str, load_count_string: str, bus_count_string: str):
        update_query = """INSERT INTO plz_parameters (version_id, plz, trafo_num, load_count_per_trafo, bus_count_per_trafo)
                          VALUES (%s, %s, %s, %s,
                                  %s);"""  # TODO: check - should values be updated for same plz and version if analysis is started? And Add a column
        self.cur.execute(update_query, vars=(VERSION_ID, plz, trafo_string, load_count_string, bus_count_string), )

        self.logger.debug("basic parameter count finished")

    def save_pp_net_with_json(self, plz: int, kcid: int, bcid: int, json_string: str) -> None:
        insert_query = ("""UPDATE grid_result
                           SET grid = %s
                           WHERE version_id = %s
                             AND plz = %s
                             AND kcid = %s
                             AND bcid = %s;""")
        self.cur.execute(insert_query, vars=(json_string, VERSION_ID, plz, kcid, bcid))

    def analyse_cables(self, plz: int):
        cluster_list = self.dbc.get_list_from_plz(plz)
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
                        cable_length_dict[type] += (cable_df[cable_df["std_type"] == type]["parallel"] *
                                                    cable_df[cable_df["std_type"] == type]["length_km"]).sum()

                    else:
                        cable_length_dict[type] = (cable_df[cable_df["std_type"] == type]["parallel"] *
                                                   cable_df[cable_df["std_type"] == type]["length_km"]).sum()
            time += 1
            if time / count >= 0.1:
                percent += 10
                self.logger.info(f"{percent} % processed")
                time = 0
        self.logger.info("analyse_cables finished.")
        cable_length_string = json.dumps(cable_length_dict)

        update_query = """UPDATE plz_parameters
            SET cable_length = %(c)s 
            WHERE version_id = %(v)s AND plz = %(p)s;"""
        self.cur.execute(update_query, {"v": VERSION_ID, "c": cable_length_string,
                                        "p": plz})  # TODO: change to cable_length_per_type, add cable_length_per_trafo

        self.logger.debug("cable count finished")

    def analyse_per_trafo_parameters(self, plz: int):
        cluster_list = self.dbc.get_list_from_plz(plz)
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
                self.logger.info(f"{percent} % processed")
                time = 0
        self.logger.info("analyse_per_trafo_parameters finished.")
        trafo_load_string = json.dumps(trafo_load_dict)
        trafo_max_distance_string = json.dumps(trafo_max_distance_dict)
        trafo_avg_distance_string = json.dumps(trafo_avg_distance_dict)

        update_query = """UPDATE plz_parameters
            SET sim_peak_load_per_trafo = %(l)s, max_distance_per_trafo = %(m)s, avg_distance_per_trafo = %(a)s
            WHERE version_id = %(v)s AND plz = %(p)s;
            """
        self.cur.execute(update_query,
                         {"v": VERSION_ID, "p": plz, "l": trafo_load_string, "m": trafo_max_distance_string,
                          "a": trafo_avg_distance_string, }, )

        self.logger.debug("per trafo analysis finished")

    def count_clustering_parameters(self, plz: int) -> int:
        """
        :param plz:
        :return:
        """
        query = """SELECT COUNT(cp.grid_result_id)
                   FROM clustering_parameters cp
                            JOIN grid_result gr ON gr.grid_result_id = cp.grid_result_id
                   WHERE version_id = %(v)s
                     AND plz = %(p)s"""
        self.cur.execute(query, {"v": VERSION_ID, "p": plz})
        return int(self.cur.fetchone()[0])

    def read_per_trafo_dict(self, plz: int) -> tuple[list[dict], list[str], dict]:
        read_query = """SELECT load_count_per_trafo,
                               bus_count_per_trafo,
                               sim_peak_load_per_trafo,
                               max_distance_per_trafo,
                               avg_distance_per_trafo
                        FROM plz_parameters
                        WHERE version_id = %(v)s
                          AND plz = %(p)s;"""
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
        data_labels = ['Load Number [-]', 'Bus Number [-]', 'Simultaneous peak load [kW]', 'Max. Trafo-Distance [m]',
                       'Avg. Trafo-Distance [m]']

        return data_list, data_labels, trafo_dict

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
        read_query = "SELECT grid FROM grid_result WHERE version_id = %s AND plz = %s AND kcid = %s AND bcid = %s LIMIT 1"
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

    def get_geo_df(self, table: str, **kwargs, ) -> gpd.GeoDataFrame:
        """
        Args:
            **kwargs: equality filters matching with the table column names
        Returns: A geodataframe with all building information
        :param table: table name
        """
        if kwargs:
            filters = " AND " + " AND ".join(
                [f"{key} = {value}" for key, value in kwargs.items() if key != 'version_id'])
        else:
            filters = ""
        query = (f"""SELECT * FROM {table}
                        WHERE version_id = %(v)s """ + filters)
        version = VERSION_ID
        if 'version_id' in kwargs:
            version = kwargs.get('version_id')

        params = {"v": version}
        with self.sqla_engine.begin() as connection:
            gdf = gpd.read_postgis(query, con=connection, params=params)

        return gdf

    def get_geo_df_join(self, select: list[str], from_table: str, join_table: str, on: tuple[str, str],
            **kwargs, ) -> gpd.GeoDataFrame:
        """
        Args:
            **kwargs: equality filters matching with the table column names
        Returns: A geodataframe with all building information
        :param select: list of column names
        :param from_table: table name
        :param join_table: table name
        :param on: join on on[0] = on[1]
        """
        if kwargs:
            filters = " AND " + " AND ".join(
                [f"{key} = {value}" for key, value in kwargs.items() if key != 'version_id'])
        else:
            filters = ""

        column_names = ", ".join(select)

        jt_prefix = join_table
        parts = join_table.split(" ")
        if len(parts) == 2:
            jt_prefix = parts[1]

        query = (f"""SELECT {column_names}
                        FROM {from_table}
                        JOIN {join_table}
                          ON {on[0]} = {on[1]}
                        WHERE {jt_prefix}.version_id = %(v)s """ + filters)
        version = VERSION_ID
        if 'version_id' in kwargs:
            version = kwargs.get('version_id')

        params = {"v": version}
        with self.sqla_engine.begin() as connection:
            gdf = gpd.read_postgis(query, con=connection, params=params)

        return gdf

    def get_municipal_register_for_plz(self, plz: str) -> pd.DataFrame:
        """get entry of table municipal register for given PLZ"""
        query = """SELECT *
                   FROM municipal_register
                   WHERE plz = %(p)s;"""
        self.cur.execute(query, {"p": plz})
        register = self.cur.fetchall()
        df_register = pd.DataFrame(register, columns=MUNICIPAL_REGISTER)
        return df_register

    def get_municipal_register(self) -> pd.DataFrame:
        """get municipal register """
        query = """SELECT *
                   FROM municipal_register;"""
        self.cur.execute(query)
        register = self.cur.fetchall()
        df_register = pd.DataFrame(register, columns=MUNICIPAL_REGISTER)
        return df_register

    def read_trafo_dict(self, plz: int) -> dict:
        read_query = """SELECT trafo_num
                        FROM plz_parameters
                        WHERE version_id = %(v)s
                          AND plz = %(p)s;"""
        self.cur.execute(read_query, {"v": VERSION_ID, "p": plz})
        trafo_num_dict = self.cur.fetchall()[0][0]

        return trafo_num_dict

    def read_cable_dict(self, plz: int) -> dict:
        read_query = """SELECT cable_length
                        FROM plz_parameters
                        WHERE version_id = %(v)s
                          AND plz = %(p)s;"""
        self.cur.execute(read_query, {"v": VERSION_ID, "p": plz})
        cable_length = self.cur.fetchall()[0][0]

        return cable_length
