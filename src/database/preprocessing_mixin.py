import json
import warnings
from abc import ABC

from src.config_loader import *
from src.database.base_mixin import BaseMixin

warnings.simplefilter(action='ignore', category=UserWarning)


class PreprocessingMixin(BaseMixin, ABC):
    def __init__(self):
        super().__init__()

    def insert_parameter_tables(self, consumer_categories: pd.DataFrame):
        self.cur.execute("SELECT count(*) FROM consumer_categories")
        categories_exist = self.cur.fetchone()[0]
        with self.sqla_engine.begin() as conn:
            if not categories_exist:
                consumer_categories.to_sql(
                    name="consumer_categories",
                    con=conn,
                    if_exists="append",
                    index=False)
                self.logger.debug("Parameter tables are inserted")

    def insert_version_if_not_exists(self):
        count_query = f"""SELECT COUNT(*)
            FROM version
            WHERE "version_id" = '{VERSION_ID}'"""
        self.cur.execute(count_query)
        version_exists = self.cur.fetchone()[0]
        if not version_exists:
            # create new version
            consumer_categories_str = CONSUMER_CATEGORIES.to_json().replace("'", "''")
            cable_cost_dict_str = json.dumps(
                CABLE_COST_DICT).replace("'", "''")
            connection_available_cables_str = str(
                CONNECTION_AVAILABLE_CABLES).replace("'", "''")
            other_parameters_dict = {"LARGE_COMPONENT_LOWER_BOUND": LARGE_COMPONENT_LOWER_BOUND,
                                     "LARGE_COMPONENT_DIVIDER": LARGE_COMPONENT_DIVIDER, "VN": VN,
                                     "V_BAND_LOW": V_BAND_LOW, "V_BAND_HIGH": V_BAND_HIGH, }
            other_paramters_str = str(other_parameters_dict).replace("'", "''")

            insert_query = f"""INSERT INTO version (version_id, version_comment, consumer_categories, cable_cost_dict, connection_available_cables, other_parameters) VALUES
                ('{VERSION_ID}', '{VERSION_COMMENT}', '{consumer_categories_str}', '{cable_cost_dict_str}', '{connection_available_cables_str}', '{other_paramters_str}')"""
            self.cur.execute(insert_query)
            self.logger.info(
                f"Version: {VERSION_ID} (created for the first time)")

    def get_postcode_table_for_regional_identifier(
            self, regional_identifier: int) -> pd.DataFrame:
        """get postcode table for given regional_identifier"""
        query = """SELECT *
                   FROM postcode
                   WHERE regional_identifier = %(p)s;"""
        df_postcode = pd.read_sql_query(
            query, con=self.conn, params={
                "p": regional_identifier})
        if len(df_postcode) == 0:
            raise ValueError(
                f"No entry in postcode table found for regional_identifier: {regional_identifier}")
        return df_postcode

    def copy_postcode_result_table(self, regional_identifier: int) -> None:
        """
        Copies the given regional_identifier entry from postcode to the postcode_result table
        :param regional_identifier:
        :return:
        """
        query = """INSERT INTO postcode_result (version_id, postcode_result_regional_identifier, geom)
                   SELECT %(v)s as version_id, regional_identifier, geom
                   FROM postcode
                   WHERE regional_identifier = %(p)s
                   LIMIT 1
                   ON CONFLICT (version_id,postcode_result_regional_identifier) DO NOTHING;"""

        self.cur.execute(query, {"v": VERSION_ID, "p": regional_identifier})

    def set_residential_buildings_table(self, regional_identifier: int):
        """
        * Fills buildings_tem with residential buildings which are inside the regional_identifier area
        :param regional_identifier:
        :return:
        """

        # Fill table
        query = """INSERT INTO buildings_tem (osm_id, area, type, geom, center, floors)
                   SELECT osm_id, area, building_t, geom, ST_Centroid(geom), floors::int
                   FROM res
                   WHERE ST_Contains((SELECT post.geom
                                      FROM postcode_result as post
                                      WHERE version_id = %(v)s
                                        AND postcode_result_regional_identifier = %(regional_identifier)s
                                      LIMIT 1), ST_Centroid(res.geom));
        UPDATE buildings_tem
        SET regional_identifier = %(regional_identifier)s
        WHERE regional_identifier ISNULL;"""
        self.cur.execute(
            query, {
                "v": VERSION_ID, "regional_identifier": regional_identifier})

    def set_other_buildings_table(self, regional_identifier: int):
        """
        * Fills buildings_tem with other buildings which are inside the regional_identifier area
        * Sets all floors to 1
        :param regional_identifier:
        :return:
        """

        # Fill table
        query = """INSERT INTO buildings_tem(osm_id, area, type, geom, center)
                   SELECT osm_id, area, use, geom, ST_Centroid(geom)
                   FROM oth AS o
                   WHERE o.use in ('Commercial', 'Public')
                     AND ST_Contains((SELECT post.geom
                                      FROM postcode_result as post
                                      WHERE version_id = %(v)s
                                        AND postcode_result_regional_identifier = %(regional_identifier)s), ST_Centroid(o.geom));;
        UPDATE buildings_tem
        SET regional_identifier = %(regional_identifier)s
        WHERE regional_identifier ISNULL;
        UPDATE buildings_tem
        SET floors = 1
        WHERE floors ISNULL;"""
        self.cur.execute(
            query, {
                "v": VERSION_ID, "regional_identifier": regional_identifier})

    def remove_duplicate_buildings(self):
        """
        * Remove buildings without geometry or osm_id
        * Remove buildings which are duplicates of other buildings and have a copied id
        :return:
        """
        remove_query = """DELETE
                          FROM buildings_tem
                          WHERE geom ISNULL;"""
        self.cur.execute(remove_query)

        remove_noid_building = """DELETE
                                  FROM buildings_tem
                                  WHERE osm_id ISNULL;"""
        self.cur.execute(remove_noid_building)

        query = """DELETE
                   FROM buildings_tem
                   WHERE geom IN
                         (SELECT geom FROM buildings_tem GROUP BY geom HAVING count(*) > 1)
                     AND osm_id LIKE '%copy%';"""
        self.cur.execute(query)

    def set_regional_identifier_settlement_type(
            self, regional_identifier: int) -> None:
        """
        Determine settlement_type in postcode_result table based on the house_distance metric for a given regional_identifier
        :param regional_identifier: Postleitzahl (postal code)
        :return: None
        """
        # Get average distance between buildings by sampling 50 random buildings
        # and finding their 4 nearest neighbors
        distance_query = """WITH some_buildings AS (SELECT osm_id, center
                                                    FROM buildings_tem
                                                    ORDER BY RANDOM()
                                                    LIMIT 50)
                            SELECT b.osm_id, d.dist
                            FROM some_buildings AS b
                                     LEFT JOIN LATERAL (
                                SELECT ST_Distance(b.center, b2.center) AS dist
                                FROM buildings_tem AS b2
                                WHERE b.osm_id <> b2.osm_id
                                ORDER BY b.center <-> b2.center
                                LIMIT 4) AS d
                                               ON TRUE;"""
        self.cur.execute(distance_query)
        data = self.cur.fetchall()

        if not data:
            raise ValueError(
                "There is no building in the buildings_tem table!")

        # Calculate average distance
        distance = [t[1] for t in data]
        avg_dis = int(sum(distance) / len(distance))

        # Update database with average distance and set settlement types based
        # on threshold
        query = """
                UPDATE postcode_result
                SET house_distance  = %(avg)s,
                    settlement_type = CASE
                                          WHEN %(avg)s < 25 THEN 3
                                          WHEN %(avg)s < 45 THEN 2
                                          ELSE 1
                        END
                WHERE version_id = %(v)s
                  AND postcode_result_regional_identifier = %(p)s;"""

        self.cur.execute(
            query, {
                "v": VERSION_ID, "avg": avg_dis, "p": regional_identifier})

    def set_building_peak_load(self) -> int:
        """
        * Sets the area, type and peak_load in the buildings_tem table
        * Removes buildings with zero load from the buildings_tem table
        :return: Number of removed unloaded buildings from buildings_tem
        """
        query = """
                UPDATE buildings_tem
                SET area = ST_Area(geom);
                UPDATE buildings_tem
                SET houses_per_building = (CASE
                                               WHEN type IN ('TH', 'Commercial', 'Public', 'Industrial') THEN 1
                                               WHEN type = 'SFH' AND area < 160 THEN 1
                                               WHEN type = 'SFH' AND area >= 160 THEN 2
                                               WHEN type IN ('MFH', 'AB') THEN floor(area / 50) * floors
                                               ELSE 0
                    END);
                UPDATE buildings_tem b
                SET peak_load_in_kw = (CASE
                                           WHEN b.type IN ('SFH', 'TH', 'MFH', 'AB') THEN b.houses_per_building *
                                                                                          (SELECT peak_load FROM consumer_categories WHERE definition = b.type)
                                           WHEN b.type IN ('Commercial', 'Public', 'Industrial') THEN b.area *
                                                                                                      (SELECT peak_load_per_m2
                                                                                                       FROM consumer_categories
                                                                                                       WHERE definition = b.type) /
                                                                                                      1000
                                           ELSE 0
                    END);"""
        self.cur.execute(query)

        count_query = ("""SELECT COUNT(*)
                          FROM buildings_tem
                          WHERE peak_load_in_kw = 0;""")
        self.cur.execute(count_query)
        count = self.cur.fetchone()[0]

        delete_query = """DELETE
                          FROM buildings_tem
                          WHERE peak_load_in_kw = 0;"""
        self.cur.execute(delete_query)

        return count

    def update_too_large_consumers_to_zero(self) -> int:
        """
        Sets the load to zero if the peak load is too large (> 100)
        :return: number of the large customers
        """
        query = """
                UPDATE buildings_tem
                SET peak_load_in_kw = 0
                WHERE peak_load_in_kw > 100
                  AND type IN ('Commercial', 'Public');
                SELECT COUNT(*)
                FROM buildings_tem
                WHERE peak_load_in_kw = 0;"""
        self.cur.execute(query)
        too_large = self.cur.fetchone()[0]

        return too_large

    def assign_close_buildings(self) -> None:
        """
        * Set peak load to zero, if a building is too near or touching to a too large customer?
        :return:
        """
        while True:
            remove_query = """WITH close (un) AS (SELECT ST_Union(geom)
                                                  FROM buildings_tem
                                                  WHERE peak_load_in_kw = 0)
                              UPDATE buildings_tem b
                              SET peak_load_in_kw = 0
                              FROM close AS c
                              WHERE ST_Touches(b.geom, c.un)
                                AND b.type IN ('Commercial', 'Public', 'Industrial')
                                AND b.peak_load_in_kw != 0;"""
            self.cur.execute(remove_query)

            count_query = """WITH close (un) AS (SELECT ST_Union(geom)
                                                 FROM buildings_tem
                                                 WHERE peak_load_in_kw = 0)
                             SELECT COUNT(*)
                             FROM buildings_tem AS b,
                                  close AS c
                             WHERE ST_Touches(b.geom, c.un)
                               AND b.type IN ('Commercial', 'Public', 'Industrial')
                               AND b.peak_load_in_kw != 0;"""
            self.cur.execute(count_query)
            count = self.cur.fetchone()[0]
            if count == 0 or count is None:
                break

        return None

    def insert_transformers(self, regional_identifier: int) -> None:
        """
        Add up the existing transformers from transformers table to the buildings_tem table
        :param regional_identifier:
        :return:
        """
        insert_query = """
                       --UPDATE transformers SET geom = ST_Centroid(geom) WHERE ST_GeometryType(geom) =  'ST_Polygon';
                       INSERT INTO buildings_tem (osm_id, geom)--(osm_id,center)
                       SELECT osm_id, geom
                       --FROM transformers WHERE ST_Within(geom, (SELECT geom FROM postcode_result LIMIT 1)) IS FALSE;
                       FROM transformers as t
                       WHERE ST_Within(t.geom, (SELECT geom
                                                FROM postcode_result
                                                WHERE postcode_result_regional_identifier = %(p)s
                                                  AND version_id = %(v)s)); --IS FALSE;
                       UPDATE buildings_tem
                       SET regional_identifier = %(p)s
                       WHERE regional_identifier ISNULL;
                       UPDATE buildings_tem
                       SET center = ST_Centroid(geom)
                       WHERE center ISNULL;
                       UPDATE buildings_tem
                       SET type = 'Transformer'
                       WHERE type ISNULL;
                       UPDATE buildings_tem
                       SET peak_load_in_kw = -1
                       WHERE peak_load_in_kw ISNULL;"""
        self.cur.execute(
            insert_query, {
                "p": regional_identifier, "v": VERSION_ID})

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
                   DELETE
                   FROM buildings_tem
                   WHERE ST_Within(center, (SELECT ungeom FROM union_table))
                     AND type = 'Transformer';"""
        self.cur.execute(query)

    def set_ways_tem_table(self, regional_identifier: int) -> int:
        """
        * Inserts ways inside the regional_identifier area (with buffer) to the ways_tem table
        :param regional_identifier:
        :return: number of ways in ways_tem
        """
        query = """INSERT INTO ways_tem
                   SELECT *
                   FROM ways AS w
                   WHERE ST_Intersects(w.geom, (SELECT ST_Buffer(geom, 50)
                                                FROM postcode_result
                                                WHERE version_id = %(v)s
                                                  AND postcode_result_regional_identifier = %(p)s));
        SELECT COUNT(*)
        FROM ways_tem;"""
        self.cur.execute(query, {"v": VERSION_ID, "p": regional_identifier})
        count = self.cur.fetchone()[0]

        if count == 0:
            raise ValueError(
                f"Ways table is empty for the given regional_identifier: {regional_identifier}")

        return count

    def connect_unconnected_ways(self) -> None:
        """
        Updates ways_tem
        :return:
        """
        query = """SELECT draw_way_connections();"""
        self.cur.execute(query)

    def draw_building_connection(self) -> None:
        """
        Updates ways_tem, creates pgr network topology in new tables and performs component analysis:
        :return:
        """
        # setup_query = """SELECT setup_temp_tables();"""
        # self.cur.execute(setup_query)

        # # Deduplicate ways_tem before running home connections
        # self._deduplicate_ways_table()

        connection_query = """ SELECT draw_home_connections(); """
        # connection_query = """ SELECT draw_home_connections_set_based(); """
        self.cur.execute(connection_query)

        topology_query = """select pgr_createTopology('ways_tem', 0.01, id:='way_id', the_geom:='geom', clean:=true) """
        self.cur.execute(topology_query)

        analyze_query = """SELECT pgr_analyzeGraph('ways_tem', 0.01, the_geom:='geom', id:='way_id');"""

        self.cur.execute(analyze_query)

    def _deduplicate_ways_table(self) -> None:
        """
        Remove duplicate geometries from ways_tem table before running home connections.
        Reports the number of duplicates found and removed.
        """
        # Count duplicates before removal
        count_query = """
        WITH duplicate_groups AS (
            SELECT geom, COUNT(*) as count
            FROM ways_tem
            GROUP BY geom
            HAVING COUNT(*) > 1
        )
        SELECT
            COUNT(*) as duplicate_groups,
            SUM(count) as total_duplicate_segments,
            SUM(count - 1) as segments_to_remove
        FROM duplicate_groups
        """

        self.cur.execute(count_query)
        result = self.cur.fetchone()

        if result and result[0] > 0:
            groups, total_segments, to_remove = result
            self.logger.info(
                f"Found {groups} duplicate geometry groups with {total_segments} total segments")
            self.logger.info(
                f"Removing {to_remove} duplicate segments (keeping 1 per group)")

            # Remove duplicates - keep the one with the smallest way_id
            remove_query = """
            DELETE FROM ways_tem w1
            USING ways_tem w2
            WHERE w1.way_id > w2.way_id
              AND ST_Equals(w1.geom, w2.geom)
            """

            self.cur.execute(remove_query)

            # Get number of affected rows from cursor
            removed_count = self.cur.rowcount

            self.logger.info(
                f"Successfully removed {removed_count} duplicate segments")

            # Final count
            self.cur.execute("SELECT COUNT(*) FROM ways_tem")
            final_count = self.cur.fetchone()[0]
            self.logger.info(
                f"ways_tem now contains {final_count} unique segments")

        else:
            self.logger.info("No duplicate geometries found in ways_tem table")

    def update_ways_cost(self) -> None:
        """
        Calculates the length of each way and stores in ways_tem.cost as meter
        """
        query = """UPDATE ways_tem
                   SET cost = ST_Length(geom);
        UPDATE ways_tem
        SET reverse_cost = cost;"""
        self.cur.execute(query)

    def set_vertice_id(self) -> int:
        """
        Updates buildings_tem with the vertice_id s from ways_tem_vertices_pgr
        :return:
        """
        query = """UPDATE buildings_tem b
                   SET vertice_id = (SELECT id
                                     FROM ways_tem_vertices_pgr AS v
                                     WHERE ST_Equals(v.the_geom, b.center));"""
        self.cur.execute(query)

        query2 = """UPDATE buildings_tem b
                    SET connection_point = (SELECT target FROM ways_tem WHERE source = b.vertice_id LIMIT 1)
                    WHERE vertice_id IS NOT NULL
                      AND connection_point IS NULL;"""
        self.cur.execute(query2)

        count_query = """ SELECT COUNT(*)
                          FROM buildings_tem
                          WHERE connection_point IS NULL
                            AND peak_load_in_kw != 0;"""
        self.cur.execute(count_query)
        count = self.cur.fetchone()[0]

        delete_query = """DELETE
                          FROM buildings_tem
                          WHERE connection_point IS NULL
                            AND peak_load_in_kw != 0;"""
        self.cur.execute(delete_query)

        return count

    def get_fips_log(self) -> pd.DataFrame:
        """get fips log: the fips code of the region of interest of which the buildings
        have already been imported to the database
        :return: table with column of fips_code
        :rtype: DataFrame
         """
        query = """SELECT *
                   FROM fips_log;"""
        df_query = pd.read_sql_query(query, con=self.conn, )
        return df_query
