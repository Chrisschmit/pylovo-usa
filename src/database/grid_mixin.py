import warnings
from abc import ABC

import pandapower as pp
from shapely.geometry import LineString

from src.config_loader import *
from src.database.base_mixin import BaseMixin

warnings.simplefilter(action='ignore', category=UserWarning)


class GridMixin(BaseMixin, ABC):
    def __init__(self):
        super().__init__()

    def create_cable_std_type(self, net: pp.pandapowerNet) -> None:
        """Create standard pandapower cable types from equipment_data table."""
        query = """
                SELECT name,
                       r_mohm_per_km / 1000.0 as r_ohm_per_km,
                       x_mohm_per_km / 1000.0 as x_ohm_per_km,
                       max_i_a / 1000.0       as max_i_ka
                FROM equipment_data
                WHERE typ = 'Cable' \
                """

        # Execute query and fetch cable data
        self.cur.execute(query)
        cables = self.cur.fetchall()

        # Create standard type for each cable in the database
        for cable in cables:
            name, r_ohm_per_km, x_ohm_per_km, max_i_ka = cable
            pp_name = name.replace('_', ' ')  # Extract name
            q_mm2 = int(name.split("_")[-1])  # Extract cross-section from name

            pp.create_std_type(net,
                               {"r_ohm_per_km": float(r_ohm_per_km), "x_ohm_per_km": float(x_ohm_per_km), "max_i_ka": float(max_i_ka),
                                # Set to zero for our standard grids
                                "c_nf_per_km": float(0),
                                "q_mm2": q_mm2}, name=pp_name, element="line", )

        self.logger.debug(
            f"Created {len(cables)} standard cable types from equipment_data table")
        return None

    def get_vertices_from_bcid(
            self, plz: int, kcid: int, bcid: int) -> tuple[dict, int]:
        ont = self.get_ont_info_from_bc(plz, kcid, bcid)["ont_vertice_id"]

        consumer_query = """SELECT vertice_id
                            FROM buildings_tem
                            WHERE plz = %(p)s
                              AND kcid = %(k)s
                              AND bcid = %(b)s;"""
        self.cur.execute(consumer_query, {"p": plz, "k": kcid, "b": bcid})
        consumer = [t[0] for t in self.cur.fetchall()]

        connection_query = """SELECT DISTINCT connection_point
                              FROM buildings_tem
                              WHERE plz = %(p)s
                                AND kcid = %(k)s
                                AND bcid = %(b)s;"""
        self.cur.execute(connection_query, {"p": plz, "k": kcid, "b": bcid})
        connection = [t[0] for t in self.cur.fetchall()]

        vertices_query = """ SELECT DISTINCT node, agg_cost
                             FROM pgr_dijkstra(
                                     'SELECT way_id as id, source, target, cost, reverse_cost FROM ways_tem'::text,
                                     %(o)s, %(c)s::integer[], false)
                             ORDER BY agg_cost;"""
        self.cur.execute(vertices_query, {"o": ont, "c": consumer})
        data = self.cur.fetchall()
        vertice_cost_dict = {t[0]: t[1]
                             for t in data if t[0] in consumer or t[0] in connection}

        return vertice_cost_dict, ont

    def get_ont_info_from_bc(self, plz: int, kcid: int,
                             bcid: int) -> dict | None:

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
            self.logger.debug(
                f"found no ont information for kcid {kcid}, bcid {bcid}")
            return None

        return {"ont_vertice_id": info[0][0],
                "transformer_rated_power": info[0][1]}

    def get_ont_geom_from_bcid(self, plz: int, kcid: int, bcid: int):
        query = """SELECT ST_X(ST_Transform(geom, 4326)), ST_Y(ST_Transform(geom, 4326))
                   FROM transformer_positions tp
                            JOIN grid_result gr
                                 ON tp.grid_result_id = gr.grid_result_id
                   WHERE gr.version_id = %(v)s
                     AND plz = %(p)s
                     AND kcid = %(k)s
                     AND bcid = %(b)s;"""
        self.cur.execute(
            query, {
                "v": VERSION_ID, "p": plz, "k": kcid, "b": bcid})
        geo = self.cur.fetchone()

        return geo

    def get_transformer_rated_power_from_bcid(
            self, plz: int, kcid: int, bcid: int) -> int:
        query = """SELECT transformer_rated_power
                   FROM grid_result
                   WHERE version_id = %(v)s
                     AND plz = %(p)s
                     AND kcid = %(k)s
                     AND bcid = %(b)s;"""
        self.cur.execute(
            query, {
                "v": VERSION_ID, "p": plz, "k": kcid, "b": bcid})
        transformer_rated_power = self.cur.fetchone()[0]

        return transformer_rated_power

    def get_node_geom(self, vid: int):
        query = """SELECT ST_X(ST_Transform(the_geom, 4326)), ST_Y(ST_Transform(the_geom, 4326))
                   FROM ways_tem_vertices_pgr
                   WHERE id = %(id)s;"""
        self.cur.execute(query, {"id": vid})
        geo = self.cur.fetchone()

        return geo

    def get_vertices_from_connection_points(self, connection: list) -> list:
        query = """SELECT vertice_id
                   FROM buildings_tem
                   WHERE connection_point IN %(c)s
                     AND type != 'Transformer';"""
        self.cur.execute(query, {"c": tuple(connection)})
        data = self.cur.fetchall()
        return [t[0] for t in data]

    def get_path_to_bus(self, vertice: int, ont: int) -> list:
        """routing problem: find the shortest path from vertice to the ont (ortsnetztrafo)"""
        query = """SELECT node
                   FROM pgr_Dijkstra(
                           'SELECT way_id as id, source, target, cost, reverse_cost FROM ways_tem', %(v)s, %(o)s,
                           false);"""
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

    def insert_lines(self, geom: list, plz: int, bcid: int, kcid: int, line_name: str, std_type: str, from_bus: int,
                     to_bus: int, length_km: float) -> None:
        """writes lines / cables that belong to a network into the database"""
        line_insertion_query = """INSERT INTO lines_result (grid_result_id,
                                                            geom,
                                                            line_name,
                                                            std_type,
                                                            from_bus,
                                                            to_bus,
                                                            length_km)
                                  VALUES ((SELECT grid_result_id
                                           FROM grid_result
                                           WHERE version_id = %(v)s
                                             AND plz = %(plz)s
                                             AND kcid = %(kcid)s
                                             AND bcid = %(bcid)s),
                                          ST_SetSRID(%(geom)s::geometry, %(epsg)s),
                                          %(line_name)s,
                                          %(std_type)s,
                                          %(from_bus)s,
                                          %(to_bus)s,
                                          %(length_km)s); """
        self.cur.execute(line_insertion_query,
                         {"v": VERSION_ID, "geom": LineString(geom).wkb_hex, "plz": int(plz), "bcid": int(bcid),
                             "kcid": int(kcid), "line_name": line_name, "std_type": std_type, "from_bus": int(from_bus),
                             "to_bus": int(to_bus), "length_km": length_km, "epsg": EPSG})

    def is_grid_generated(self, plz: int):
        """
        Check if grid exists.

        Args:
            plz: Postal code to be checked

        Returns:
            bool: True if record exists, False otherwise
        """
        query = f"""
            SELECT 1
            FROM postcode_result
            WHERE version_id = %(version_id)s AND postcode_result_plz = %(plz)s
            LIMIT 1;
        """

        self.cur.execute(query, {"version_id": VERSION_ID, "plz": plz})
        result = self.cur.fetchone()
        return result is not None
