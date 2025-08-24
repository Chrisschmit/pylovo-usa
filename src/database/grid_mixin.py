import warnings
from abc import ABC
from typing import Any, Dict, List, Optional, Union

import pandapower as pp
from shapely.geometry import LineString

from src.config_loader import *
from src.database.base_mixin import BaseMixin
from src.equipment_data_schema import (CableEquipment, TransformerEquipment,
                                       create_equipment_from_database_row)

warnings.simplefilter(action='ignore', category=UserWarning)


class GridMixin(BaseMixin, ABC):
    def __init__(self):
        super().__init__()

    def create_cable_std_type(self, net: pp.pandapowerNet) -> None:
        # TODO Refactor
        """Create standard pandapower cable types from equipment_data table."""
        query = """
                SELECT name,
                       r_ohm_per_km,
                       x_ohm_per_km,
                       max_i_a / 1000.0       as max_i_ka
                FROM equipment_data
                WHERE type = 'Cable' \
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

    # TODO: Refactor not compatible with new grid structure
    def get_vertices_from_bcid(
            self, regional_identifier: int, kcid: int, bcid: int, scid: int = None) -> tuple[dict, int]:
        # get Transformer_vertice_ids from lv_grid_result table (for LV transformers)
        # If scid is provided, query the hierarchical table
        if scid is not None:
            transformer_query = """SELECT dist_transformer_vertice_id
                                 FROM lv_grid_result
                                 WHERE version_id = %s
                                   AND regional_identifier = %s
                                   AND kcid = %s
                                   AND scid = %s
                                   AND bcid = %s"""
            self.cur.execute(
                transformer_query,
                (VERSION_ID,
                 regional_identifier,
                 kcid,
                 scid,
                 bcid))
            result = self.cur.fetchone()
            transformer = result[0] if result else None
        else:
            # Fallback to old method for backward compatibility
            transformer = self.get_transformer_info_from_bc(
                regional_identifier, kcid, bcid)["transformer_vertice_id"]

        # Build queries with optional scid filter
        scid_filter = " AND scid = %(s)s" if scid is not None else ""

        consumer_query = f"""SELECT vertice_id
                            FROM buildings_tem
                            WHERE regional_identifier = %(p)s
                              AND kcid = %(k)s
                              AND bcid = %(b)s{scid_filter};"""

        params = {"p": regional_identifier, "k": kcid, "b": bcid}
        if scid is not None:
            params["s"] = scid

        self.cur.execute(consumer_query, params)
        consumer = [t[0] for t in self.cur.fetchall()]

        connection_query = f"""SELECT DISTINCT connection_point
                              FROM buildings_tem
                              WHERE regional_identifier = %(p)s
                                AND kcid = %(k)s
                                AND bcid = %(b)s{scid_filter};"""
        self.cur.execute(connection_query, params)
        connection = [t[0] for t in self.cur.fetchall()]

        vertices_query = """ SELECT DISTINCT node, agg_cost
                             FROM pgr_dijkstra(
                                     'SELECT way_id as id, source, target, cost, reverse_cost FROM ways_tem'::text,
                                     %(o)s, %(c)s::integer[], false)
                             ORDER BY agg_cost;"""

        self.cur.execute(vertices_query, {"o": transformer, "c": consumer})
        data = self.cur.fetchall()
        # data contains tuples of (vertex_id, routing_cost) from the Dijkstra query
        # t[0] = vertex ID, t[1] = routing cost
        vertice_cost_dict = {t[0]: t[1]
                             for t in data if t[0] in consumer or t[0] in connection}

        return vertice_cost_dict, transformer

    def get_transformer_info_from_bc(self, regional_identifier: int, kcid: int,
                                     bcid: int) -> dict | None:
        """
        get transformer information from grid_result table
        """

        query = """SELECT transformer_vertice_id, transformer_rated_power
                   FROM grid_result
                   WHERE version_id = %(v)s
                     AND kcid = %(k)s
                     AND bcid = %(b)s
                     AND regional_identifier = %(p)s; """
        params = {
            "v": VERSION_ID,
            "p": regional_identifier,
            "k": kcid,
            "b": bcid}
        self.cur.execute(query, params)
        info = self.cur.fetchall()
        if not info:
            self.logger.debug(
                f"found no transformer information for kcid {kcid}, bcid {bcid}")
            return None

        return {"transformer_vertice_id": info[0][0],
                "transformer_rated_power": info[0][1]}

    def get_transformer_geom_from_bcid(
            self, regional_identifier: int, kcid: int, bcid: int):
        query = """SELECT ST_X(ST_Transform(geom, 4326)), ST_Y(ST_Transform(geom, 4326))
                   FROM transformer_positions tp
                            JOIN grid_result gr
                                 ON tp.grid_result_id = gr.grid_result_id
                   WHERE gr.version_id = %(v)s
                     AND regional_identifier = %(p)s
                     AND kcid = %(k)s
                     AND bcid = %(b)s;"""
        self.cur.execute(
            query, {
                "v": VERSION_ID, "p": regional_identifier, "k": kcid, "b": bcid})
        geo = self.cur.fetchone()

        return geo

    def get_transformer_rated_power_from_bcid(
            self, regional_identifier: int, kcid: int, bcid: int) -> int:
        query = """SELECT transformer_rated_power
                   FROM grid_result
                   WHERE version_id = %(v)s
                     AND regional_identifier = %(p)s
                     AND kcid = %(k)s
                     AND bcid = %(b)s;"""
        self.cur.execute(
            query, {
                "v": VERSION_ID, "p": regional_identifier, "k": kcid, "b": bcid})
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

    def get_path_to_bus(self, vertice: int, transformer: int) -> list:
        """routing problem: find the shortest path from vertice to the transformer of the cluster"""
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
        self.cur.execute(query, {"o": transformer, "v": vertice})
        data = self.cur.fetchall()
        way_list = [t[0] for t in data]

        return way_list

    def insert_mv_line(self, geom: list, kcid: int, scid: int, line_name: str,
                       equipment_id: str, from_bus: int, to_bus: int, length_km: float) -> None:
        """Insert MV line (20kV) into the database."""
        query = """
        INSERT INTO lines_result (
            grid_result_id, lv_grid_result_id, grid_level,
            line_name, std_type, equipment_id,
            from_bus, to_bus, length_km, geom
        )
        VALUES (
            (SELECT grid_result_id FROM grid_result
             WHERE version_id = %(v)s AND kcid = %(kcid)s AND scid = %(scid)s),
            NULL,
            'MV',
            %(line_name)s,
            %(equipment_id)s,
            %(equipment_id)s,
            %(from_bus)s,
            %(to_bus)s,
            %(length_km)s,
            ST_SetSRID(%(geom)s::geometry, %(epsg)s)
        )
        """
        self.cur.execute(query, {
            "v": VERSION_ID,
            "kcid": int(kcid),
            "scid": int(scid),
            "line_name": line_name,
            "equipment_id": equipment_id,
            "from_bus": int(from_bus),
            "to_bus": int(to_bus),
            "length_km": float(length_km),
            "geom": LineString(geom).wkb_hex,
            "epsg": EPSG
        })

    def insert_lv_line(self, geom: list, kcid: int, scid: int, bcid: int,
                       line_name: str, equipment_id: str,
                       from_bus: int, to_bus: int, length_km: float) -> None:
        """Insert LV line (400V) into the database."""
        query = """
        INSERT INTO lines_result (
            grid_result_id, lv_grid_result_id, grid_level,
            line_name, std_type, equipment_id,
            from_bus, to_bus, length_km, geom
        )
        VALUES (
            (SELECT grid_result_id FROM grid_result
             WHERE version_id = %(v)s AND kcid = %(kcid)s AND scid = %(scid)s),
            (SELECT lv_grid_result_id FROM lv_grid_result
             WHERE version_id = %(v)s AND kcid = %(kcid)s
               AND scid = %(scid)s AND bcid = %(bcid)s),
            'LV',
            %(line_name)s,
            %(equipment_id)s,
            %(equipment_id)s,
            %(from_bus)s,
            %(to_bus)s,
            %(length_km)s,
            ST_SetSRID(%(geom)s::geometry, %(epsg)s)
        )
        """
        self.cur.execute(query, {
            "v": VERSION_ID,
            "kcid": int(kcid),
            "scid": int(scid),
            "bcid": int(bcid),
            "line_name": line_name,
            "equipment_id": equipment_id,
            "from_bus": int(from_bus),
            "to_bus": int(to_bus),
            "length_km": float(length_km),
            "geom": LineString(geom).wkb_hex,
            "epsg": EPSG
        })

    # Legacy method - kept for backward compatibility but marked deprecated
    def insert_lines(self, geom: list, regional_identifier: int, bcid: int, kcid: int,
                     line_name: str, std_type: str, from_bus: int, to_bus: int,
                     length_km: float) -> None:
        """DEPRECATED: Use insert_lv_line or insert_mv_line instead."""
        self.logger.warning(
            "insert_lines is deprecated. Use insert_lv_line() for LV networks.")
        # Convert to new format - assume LV line if bcid is provided
        scid = 0  # Default scid for legacy compatibility
        self.insert_lv_line(
            geom=geom, kcid=kcid, scid=scid, bcid=bcid,
            line_name=line_name, equipment_id=std_type,
            from_bus=from_bus, to_bus=to_bus, length_km=length_km
        )

    def is_grid_generated(self, regional_identifier: int):
        """
        Check if grid exists.

        Args:
            regional_identifier: Postal code to be checked

        Returns:
            bool: True if record exists, False otherwise
        """
        query = f"""
            SELECT 1
            FROM postcode_result
            WHERE version_id = %(version_id)s AND postcode_result_regional_identifier = %(regional_identifier)s
            LIMIT 1;
        """

        self.cur.execute(
            query, {
                "version_id": VERSION_ID, "regional_identifier": regional_identifier})
        result = self.cur.fetchone()
        return result is not None

    # ==== EQUIPMENT RETRIEVAL METHODS ====

    def get_equipment_by_id(
            self, equipment_id: str) -> Union[TransformerEquipment, CableEquipment]:
        """
        Retrieve equipment specifications from equipment_data table by ID.
        Used during grid construction to get pre-selected equipment.

        Args:
            equipment_id: Primary key (name) from equipment_data table

        Returns:
            TransformerEquipment or CableEquipment object with full specifications
        """
        query = """
        SELECT * FROM equipment_data
        WHERE name = %s
        """
        self.cur.execute(query, (equipment_id,))
        row = self.cur.fetchone()

        if not row:
            raise ValueError(f"Equipment not found: {equipment_id}")

        # Convert to dict and create appropriate equipment object
        columns = [desc[0] for desc in self.cur.description]
        equipment_dict = dict(zip(columns, row))

        return create_equipment_from_database_row(equipment_dict)

    def get_substation_for_scid(self, kcid: int, scid: int) -> dict:
        """
        Get substation data including equipment_id for MV network construction.

        Returns:
            Dictionary with substation_vertice_id, substation_rated_power, and equipment_id
        """
        query = """
        SELECT substation_vertice_id, substation_rated_power, equipment_id
        FROM grid_result
        WHERE version_id = %s
          AND kcid = %s
          AND scid = %s
        """
        self.cur.execute(query, (VERSION_ID, kcid, scid))
        result = self.cur.fetchone()

        if result:
            return {
                'substation_vertice_id': result[0],
                'substation_rated_power': result[1],
                'equipment_id': result[2]  # Foreign key to equipment_data.name
            }
        return None

    def get_lv_transformers_for_scid(self, kcid: int, scid: int) -> List[dict]:
        """
        Get all LV transformers with their equipment_ids for a given scid.

        Returns:
            List of dicts with bcid, transformer vertex, power rating, and equipment_id
        """
        query = """
        SELECT bcid, dist_transformer_vertice_id,
               dist_transformer_rated_power, equipment_id
        FROM lv_grid_result
        WHERE version_id = %s
          AND kcid = %s
          AND scid = %s
        """
        self.cur.execute(query, (VERSION_ID, kcid, scid))
        results = self.cur.fetchall()

        return [{
            'bcid': row[0],
            'transformer_vertice_id': row[1],
            'transformer_rated_power': row[2],
            'equipment_id': row[3]  # Foreign key to equipment_data.name
        } for row in results]

    def get_mv_buildings_for_scid(self, kcid: int, scid: int) -> List[dict]:
        """
        Get MV buildings (high load buildings that connect directly to MV) for a given scid.

        Returns:
            List of dicts with building information for direct MV connections
        """
        query = """
        SELECT osm_id, connection_point, peak_load_in_kw, type, vertice_id
        FROM buildings_tem
        WHERE kcid = %s
          AND scid = %s
          AND grid_level_connection = 'MV'
        ORDER BY peak_load_in_kw DESC
        """
        self.cur.execute(query, (kcid, scid))
        results = self.cur.fetchall()

        return [{
            'osm_id': row[0],
            'connection_point': row[1],
            'peak_load_kw': row[2],
            'building_type': row[3],
            'vertice_id': row[4]  # Building centroid vertex ID
        } for row in results]

    def get_bcids_for_scid(self, kcid: int, scid: int) -> List[int]:
        """
        Get all bcid clusters under a given substation cluster (scid).

        Returns:
            List of bcid integers
        """
        query = """
        SELECT DISTINCT bcid
        FROM lv_grid_result
        WHERE version_id = %s
          AND kcid = %s
          AND scid = %s
        ORDER BY bcid
        """
        self.cur.execute(query, (VERSION_ID, kcid, scid))
        results = self.cur.fetchall()

        return [row[0] for row in results]

    def get_lv_transformer_for_bcid(
            self, kcid: int, scid: int, bcid: int) -> dict:
        """
        Get LV transformer data for a specific bcid cluster.

        Returns:
            Dictionary with transformer vertex, power rating, and equipment_id
        """
        query = """
        SELECT dist_transformer_vertice_id, dist_transformer_rated_power, equipment_id
        FROM lv_grid_result
        WHERE version_id = %s
          AND kcid = %s
          AND scid = %s
          AND bcid = %s
        """
        self.cur.execute(query, (VERSION_ID, kcid, scid, bcid))
        result = self.cur.fetchone()

        if result:
            return {
                'transformer_vertice_id': result[0],
                'transformer_rated_power': result[1],
                'equipment_id': result[2]  # Foreign key to equipment_data.name
            }
        return None

    # ==== LINE RETRIEVAL METHODS FOR VISUALIZATION ====

    def get_lines_by_voltage_level(self, voltage_level: str, kcid: Optional[int] = None,
                                   scid: Optional[int] = None, bcid: Optional[int] = None) -> List[dict]:
        """
        Get lines filtered by voltage level for visualization.

        Args:
            voltage_level: 'MV' or 'LV'
            kcid: Optional K-means cluster filter
            scid: Optional substation cluster filter
            bcid: Optional building cluster filter (LV only)

        Returns:
            List of line dictionaries with geometry and metadata
        """
        query = """
        SELECT
            lr.lines_result_id,
            lr.line_name,
            lr.equipment_id,
            lr.length_km,
            lr.grid_level,
            lr.network_identifier,
            ST_AsGeoJSON(lr.geom) as geometry,
            lr.kcid,
            lr.scid,
            lr.bcid
        FROM lines_result_with_grid lr
        WHERE lr.grid_level = %s
        """
        params = [voltage_level]

        if kcid is not None:
            query += " AND lr.kcid = %s"
            params.append(kcid)

        if scid is not None:
            query += " AND lr.scid = %s"
            params.append(scid)

        if bcid is not None and voltage_level == 'LV':
            query += " AND lr.bcid = %s"
            params.append(bcid)

        query += " ORDER BY lr.kcid, lr.scid, lr.bcid, lr.line_name"

        self.cur.execute(query, params)
        results = self.cur.fetchall()

        columns = [desc[0] for desc in self.cur.description]
        return [dict(zip(columns, row)) for row in results]

    def get_mv_lines_for_visualization(
            self, kcid: Optional[int] = None, scid: Optional[int] = None) -> List[dict]:
        """Get MV lines (20kV) for visualization."""
        return self.get_lines_by_voltage_level('MV', kcid=kcid, scid=scid)

    def get_lv_lines_for_visualization(self, kcid: Optional[int] = None,
                                       scid: Optional[int] = None, bcid: Optional[int] = None) -> List[dict]:
        """Get LV lines (400V) for visualization."""
        return self.get_lines_by_voltage_level(
            'LV', kcid=kcid, scid=scid, bcid=bcid)

    def get_line_statistics_by_voltage_level(self) -> dict:
        """
        Get line statistics grouped by voltage level.

        Returns:
            Dictionary with statistics for MV and LV networks
        """
        query = """
        SELECT
            grid_level,
            COUNT(*) as line_count,
            COUNT(DISTINCT equipment_id) as unique_cables,
            SUM(length_km) as total_length_km,
            AVG(length_km) as avg_length_km
        FROM lines_result
        WHERE grid_level IN ('MV', 'LV')
        GROUP BY grid_level
        ORDER BY grid_level
        """

        self.cur.execute(query)
        results = self.cur.fetchall()

        stats = {}
        for row in results:
            grid_level, line_count, unique_cables, total_length, avg_length = row
            stats[grid_level] = {
                'line_count': line_count,
                'unique_cable_types': unique_cables,
                'total_length_km': float(total_length) if total_length else 0.0,
                'average_length_km': float(avg_length) if avg_length else 0.0
            }

        return stats

    def save_grid_cluster(self, regional_identifier: int,
                          kcid: int, scid: int, grid_data: Dict[str, Any]) -> None:
        """
        Save grid construction results for a single cluster to database.

        Args:
            kcid: K-means cluster ID
            scid: Substation cluster ID
            grid_data: Complete grid data from backend export
        """
        import json

        # Convert grid_data to JSON string if needed
        grid_json = json.dumps(grid_data) if not isinstance(
            grid_data, str) else grid_data

        query = """
        UPDATE grid_result
        SET grid = %s
        WHERE version_id = %s
          AND regional_identifier = %s
          AND kcid = %s
          AND scid = %s
        """
        self.cur.execute(
            query,
            (grid_json,
             str(VERSION_ID),
                int(regional_identifier),
                int(kcid),
                int(scid)))
