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

class ReadMixin:
        def get_consumer_categories(self):
            """
            Returns: A dataframe with self-defined consumer categories and typical values
            """
            query = """SELECT * FROM consumer_categories"""
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
            settlement_query = """SELECT settlement_type FROM postcode_result
                WHERE postcode_result_plz = %(p)s 
                LIMIT 1; """
            self.cur.execute(settlement_query, {"p": plz})
            settlement_type = self.cur.fetchone()[0]
    
            return settlement_type
        def get_buildings_from_kcid(
                self,
                kcid : int,
        ) -> pd.DataFrame:
            """
            Args:
                kcid: kmeans_cluster ID
            Returns: A dataframe with all building information
            """
            buildings_query = """SELECT * FROM buildings_tem 
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
        def get_consumer_simultaneous_load_dict(self, consumer_list: list, buildings_df: pd.DataFrame) -> tuple[dict, dict, dict]:
            Pd = {consumer: 0 for consumer in consumer_list}  # dict of all vertices in bc, 0 as default
            load_units = {consumer: 0 for consumer in consumer_list}
            load_type = {consumer: "SFH" for consumer in consumer_list}
    
            for row in buildings_df.itertuples():
                load_units[row.vertice_id] = row.houses_per_building
                load_type[row.vertice_id] = row.type
                gzf = CONSUMER_CATEGORIES.loc[CONSUMER_CATEGORIES.definition == row.type, "sim_factor"].item()
    
                # Determine simultaneous load of each building in MW
                Pd[row.vertice_id] = utils.oneSimultaneousLoad(row.peak_load_in_kw * 1e-3, row.houses_per_building, gzf)
    
            return Pd, load_units, load_type
        def find_furthest_node_path_list(self, connection_node_list: list, vertices_dict: dict, ont_vertice: int) -> list:
            connection_node_dict = {n: vertices_dict[n] for n in connection_node_list}
            furthest_node = max(connection_node_dict, key=connection_node_dict.get)
            # all the connection nodes in the path from transformer to furthest node are considered as potential branch loads
            furthest_node_path_list = self.get_path_to_bus(furthest_node, ont_vertice)
            furthest_node_path = [
                p for p in furthest_node_path_list if p in connection_node_list
            ]
    
            return furthest_node_path
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
        def get_ont_geom_from_bcid(self, plz: int, kcid: int, bcid: int):
            query = """SELECT ST_X(ST_Transform(geom,4326)), ST_Y(ST_Transform(geom,4326))
                       FROM transformer_positions tp
                       JOIN grid_result gr
                       ON tp.grid_result_id = gr.grid_result_id
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
        def get_list_from_plz(self, plz: int) -> list:
            query = """SELECT DISTINCT kcid, bcid FROM grid_result 
                        WHERE  version_id = %(v)s AND plz = %(p)s 
                        ORDER BY kcid, bcid;"""
            self.cur.execute(query, {"p": plz, "v": VERSION_ID})
            cluster_list = self.cur.fetchall()
    
            return cluster_list
        def get_distance_matrix_from_kcid(self, kcid: int) -> tuple[dict, np.ndarray, dict]:
            """
            Creates a distance matrix from the buildings in the kcid
            Args:
                kcid: k-means cluster id
            Returns: The distance matrix of the buildings as np.array and the mapping between vertice_id and local ID as dict
            """
    
            costmatrix_query = """SELECT * FROM pgr_dijkstraCostMatrix(
                                'SELECT way_id as id, source, target, cost, reverse_cost FROM ways_tem',
                                (SELECT array_agg(DISTINCT b.connection_point) FROM (SELECT * FROM buildings_tem 
                                WHERE kcid = %(k)s
                                AND bcid ISNULL
                                ORDER BY connection_point) AS b),
                                false);"""
            params = {"k": kcid}
    
            return self._calculate_cost_arr_dist_matrix(costmatrix_query, params)
        def get_greenfield_bcids(self, plz: int, kcid: int) -> list:
            """
            Args:
                plz: loadarea cluster ID
                kcid: kmeans cluster ID
            Returns: A list of greenfield building clusters for a given plz
            """
            query = """SELECT DISTINCT bcid FROM grid_result
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
                        FROM grid_result
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
        def get_next_unfinished_kcid(self, plz: int) -> int:
            """
            :return: one unmodeled k mean cluster ID - plz
            """
            query = """SELECT kcid FROM buildings_tem 
                        WHERE kcid NOT IN (
                            SELECT DISTINCT kcid FROM grid_result
                            WHERE version_id = %(v)s AND grid_result.plz = %(plz)s) AND kcid IS NOT NULL
                        ORDER BY kcid
                        LIMIT 1;"""
            self.cur.execute(query, {"v": VERSION_ID, "plz": plz})
            kcid = self.cur.fetchone()[0]
            return kcid
        def get_kcid_length(self) -> int:
            query = """SELECT COUNT(DISTINCT kcid) FROM buildings_tem WHERE kcid IS NOT NULL; """
            self.cur.execute(query)
            kcid_length = self.cur.fetchone()[0]
            return kcid_length
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
        def count_postcode_result(self, plz: int) -> int:
            """
            :param plz:
            :return:
            """
            query = """SELECT COUNT(*) FROM postcode_result
                        WHERE version_id = %(v)s
                        AND postcode_result_plz::INT = %(p)s"""
            self.cur.execute(query, {"v": VERSION_ID, "p": plz})
            return int(self.cur.fetchone()[0])
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
        def generate_load_vector(self, kcid: int, bcid: int) -> np.ndarray:
            query = """SELECT SUM(peak_load_in_kw)::float FROM buildings_tem 
                    WHERE kcid = %(k)s AND bcid = %(b)s 
                    GROUP BY connection_point 
                    ORDER BY connection_point;"""
            self.cur.execute(query, {"k": kcid, "b": bcid})
            load = np.asarray([i[0] for i in self.cur.fetchall()])
    
            return load
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
        def read_trafo_dict(self, plz: int) -> dict:
            read_query = """SELECT trafo_num FROM plz_parameters 
            WHERE version_id = %(v)s AND plz = %(p)s;"""
            self.cur.execute(read_query, {"v": VERSION_ID, "p": plz})
            trafo_num_dict = self.cur.fetchall()[0][0]
    
            return trafo_num_dict
        def read_per_trafo_dict(self, plz: int) -> tuple[list[dict], list[str], dict]:
            read_query = """SELECT load_count_per_trafo, bus_count_per_trafo, sim_peak_load_per_trafo,
            max_distance_per_trafo, avg_distance_per_trafo FROM plz_parameters 
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
                    f"""SELECT * FROM {table}
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
        def get_geo_df_join(
                self,
                select: list[str],
                from_table: str,
                join_table: str,
                on: tuple[str, str],
                **kwargs,
        ) -> gpd.GeoDataFrame:
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
                    [f"{key} = {value}" for key, value in kwargs.items() if key != 'version_id']
                )
            else:
                filters = ""
    
            column_names = ", ".join(select)
    
            jt_prefix = join_table
            parts = join_table.split(" ")
            if len(parts) == 2:
                jt_prefix = parts[1]
    
            query = (
                    f"""SELECT {column_names}
                        FROM {from_table}
                        JOIN {join_table}
                          ON {on[0]} = {on[1]}
                        WHERE {jt_prefix}.version_id = %(v)s """
                    + filters
            )
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
        def get_ags_log(self) -> pd.DataFrame:
            """get ags log: the amtliche gemeindeschluessel of the municipalities of which the buildings
            have already been imported to the database
            :return: table with column of ags
            :rtype: DataFrame
             """
            query = """SELECT * 
            FROM ags_log;"""
            df_query = pd.read_sql_query(query, con=self.conn, )
            return df_query
