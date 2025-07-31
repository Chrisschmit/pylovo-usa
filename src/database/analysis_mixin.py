import json
import warnings
from abc import ABC

import geopandas as gpd
import pandapower as pp

from src.config_loader import *
from src.database.base_mixin import BaseMixin

warnings.simplefilter(action="ignore", category=UserWarning)


class AnalysisMixin(BaseMixin, ABC):
    def __init__(self):
        super().__init__()

    def insert_regional_identifier_parameters(
            self, regional_identifier: int, trafo_string: str, load_count_string: str, bus_count_string: str):
        update_query = """INSERT INTO regional_identifier_parameters (version_id, regional_identifier, trafo_num, load_count_per_trafo, bus_count_per_trafo)
                          VALUES (%s, %s, %s, %s,
                                  %s);"""  # TODO: check - should values be updated for same regional_identifier and version if analysis is started? And Add a column
        self.cur.execute(
            update_query,
            vars=(
                VERSION_ID,
                regional_identifier,
                trafo_string,
                load_count_string,
                bus_count_string),
        )
        self.logger.debug("basic parameter count finished")

    def insert_cable_length(self, regional_identifier: int,
                            cable_length_string: str):
        update_query = """UPDATE regional_identifier_parameters
                          SET cable_length = %(c)s
                          WHERE version_id = %(v)s
                            AND regional_identifier = %(p)s;"""
        self.cur.execute(update_query, {"v": VERSION_ID, "c": cable_length_string,
                                        # TODO: change to
                                        # cable_length_per_type, add
                                        # cable_length_per_trafo
                                        "p": regional_identifier})
        self.logger.debug("cable count finished")

    def insert_trafo_parameters(self, regional_identifier: int, trafo_load_string: str, trafo_max_distance_string: str,
                                trafo_avg_distance_string: str):
        update_query = """UPDATE regional_identifier_parameters
                          SET sim_peak_load_per_trafo = %(l)s,
                              max_distance_per_trafo  = %(m)s,
                              avg_distance_per_trafo  = %(a)s
                          WHERE version_id = %(v)s
                            AND regional_identifier = %(p)s; \
                       """
        self.cur.execute(update_query,
                         {"v": VERSION_ID, "p": regional_identifier, "l": trafo_load_string, "m": trafo_max_distance_string,
                          "a": trafo_avg_distance_string, }, )
        self.logger.debug("per trafo analysis finished")

    def save_pp_net_with_json(
            self, regional_identifier: int, kcid: int, bcid: int, json_string: str) -> None:
        insert_query = ("""UPDATE grid_result
                           SET grid = %s
                           WHERE version_id = %s
                             AND regional_identifier = %s
                             AND kcid = %s
                             AND bcid = %s;""")
        self.cur.execute(
            insert_query,
            vars=(
                json_string,
                VERSION_ID,
                regional_identifier,
                kcid,
                bcid))

    def count_clustering_parameters(self, regional_identifier: int) -> int:
        """
        :param regional_identifier:
        :return:
        """
        query = """SELECT COUNT(cp.grid_result_id)
                   FROM clustering_parameters cp
                            JOIN grid_result gr ON gr.grid_result_id = cp.grid_result_id
                   WHERE version_id = %(v)s
                     AND regional_identifier = %(p)s"""
        self.cur.execute(query, {"v": VERSION_ID, "p": regional_identifier})
        return int(self.cur.fetchone()[0])

    def read_per_trafo_dict(
            self, regional_identifier: int) -> tuple[list[dict], list[str], dict]:
        read_query = """SELECT load_count_per_trafo,
                               bus_count_per_trafo,
                               sim_peak_load_per_trafo,
                               max_distance_per_trafo,
                               avg_distance_per_trafo
                        FROM regional_identifier_parameters
                        WHERE version_id = %(v)s
                          AND regional_identifier = %(p)s;"""
        self.cur.execute(
            read_query, {
                "v": VERSION_ID, "p": regional_identifier})
        result = self.cur.fetchall()

        # Sort all parameters according to transformer size
        load_dict = dict(sorted(result[0][0].items(), key=lambda x: int(x[0])))
        bus_dict = dict(sorted(result[0][1].items(), key=lambda x: int(x[0])))
        peak_dict = dict(sorted(result[0][2].items(), key=lambda x: int(x[0])))
        max_dict = dict(sorted(result[0][3].items(), key=lambda x: int(x[0])))
        avg_dict = dict(sorted(result[0][4].items(), key=lambda x: int(x[0])))

        trafo_dict = dict(
            sorted(
                self.read_trafo_dict(regional_identifier).items(),
                key=lambda x: int(
                    x[0]),
                reverse=True))
        # Create list with all parameter dicts
        data_list = [load_dict, bus_dict, peak_dict, max_dict, avg_dict]
        data_labels = ['Load Number [-]', 'Bus Number [-]', 'Simultaneous peak load [kW]', 'Max. Trafo-Distance [m]',
                       'Avg. Trafo-Distance [m]']

        return data_list, data_labels, trafo_dict

    def read_net(self, regional_identifier: int, kcid: int,
                 bcid: int) -> pp.pandapowerNet:
        """
        Reads a pandapower network from the database for the specified grid.

        Args:
            regional_identifier: Postal code ID
            kcid: Kmeans cluster ID
            bcid: Building cluster ID

        Returns:
            A pandapower network object

        Raises:
            ValueError: If the requested grid does not exist in the database
        """
        read_query = "SELECT grid FROM grid_result WHERE version_id = %s AND regional_identifier = %s AND kcid = %s AND bcid = %s LIMIT 1"
        self.cur.execute(
            read_query,
            vars=(
                VERSION_ID,
                regional_identifier,
                kcid,
                bcid))

        result = self.cur.fetchall()
        if not result:
            self.logger.error(
                f"Grid not found for regional_identifier={regional_identifier}, kcid={kcid}, bcid={bcid}, version_id={VERSION_ID}")
            raise ValueError(
                f"Grid not found for regional_identifier={regional_identifier}, kcid={kcid}, bcid={bcid}")

        grid_tuple = result[0]
        grid_dict = grid_tuple[0]
        grid_json_string = json.dumps(grid_dict)
        net = pp.from_json_string(grid_json_string)

        return net

    def insert_clustering_parameters(self, params: dict) -> None:
        """Insert calculated grid parameters into clustering_parameters table."""

        insert_query = """INSERT INTO clustering_parameters (
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
                  (SELECT grid_result_id FROM grid_result WHERE version_id = %(version_id)s AND regional_identifier = %(regional_identifier)s AND bcid = %(bcid)s AND kcid = %(kcid)s),
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

        self.cur.execute(insert_query, params)
        self.conn.commit()

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

    def get_municipal_register_for_regional_identifier(
            self, regional_identifier: str) -> pd.DataFrame:
        """get entry of table municipal register for given regional_identifier"""
        query = """SELECT *
                   FROM municipal_register
                   WHERE regional_identifier = %(p)s;"""
        self.cur.execute(query, {"p": regional_identifier})
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

    def read_trafo_dict(self, regional_identifier: int) -> dict:
        read_query = """SELECT trafo_num
                        FROM regional_identifier_parameters
                        WHERE version_id = %(v)s
                          AND regional_identifier = %(p)s;"""
        self.cur.execute(
            read_query, {
                "v": VERSION_ID, "p": regional_identifier})
        trafo_num_dict = self.cur.fetchall()[0][0]

        return trafo_num_dict

    def read_cable_dict(self, regional_identifier: int) -> dict:
        read_query = """SELECT cable_length
                        FROM regional_identifier_parameters
                        WHERE version_id = %(v)s
                          AND regional_identifier = %(p)s;"""
        self.cur.execute(
            read_query, {
                "v": VERSION_ID, "p": regional_identifier})
        cable_length = self.cur.fetchall()[0][0]

        return cable_length

    def is_grid_analyzed(self, regional_identifier: int):
        """
        Check if grid has been analyzed.

        Args:
            regional_identifier: Postal code to be checked

        Returns:
            bool: True if record exists, False otherwise
        """
        query = f"""
            SELECT 1
            FROM regional_identifier_parameters
            WHERE version_id = %(version_id)s AND regional_identifier = %(regional_identifier)s
            LIMIT 1;
        """

        self.cur.execute(
            query, {
                "version_id": VERSION_ID, "regional_identifier": regional_identifier})
        result = self.cur.fetchone()
        return result is not None
