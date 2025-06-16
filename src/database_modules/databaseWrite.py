import json
import warnings

from src.config_loader import *

warnings.simplefilter(action='ignore', category=UserWarning)


class WriteMixin:
    def copy_postcode_result_table(self, plz: int) -> None:
        """
        Copies the given plz entry from postcode to the postcode_result table
        :param plz:
        :return:
        """
        query = """INSERT INTO postcode_result (version_id, postcode_result_plz, geom)
                   SELECT %(v)s as version_id, plz, geom
                   FROM postcode
                   WHERE plz = %(p)s
                   LIMIT 1
                   ON CONFLICT (version_id,postcode_result_plz) DO NOTHING;"""

        self.cur.execute(query, {"v": VERSION_ID, "p": plz})

    def set_plz_settlement_type(self, plz: int) -> None:
        """
        Determine settlement_type in postcode_result table based on the house_distance metric for a given plz
        :param plz: Postleitzahl (postal code)
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
            raise ValueError("There is no building in the buildings_tem table!")

        # Calculate average distance
        distance = [t[1] for t in data]
        avg_dis = int(sum(distance) / len(distance))

        # Update database with average distance and set settlement types based on threshold
        query = """
                UPDATE postcode_result
                SET house_distance  = %(avg)s,
                    settlement_type = CASE
                                          WHEN %(avg)s < 25 THEN 3
                                          WHEN %(avg)s < 45 THEN 2
                                          ELSE 1
                        END
                WHERE version_id = %(v)s
                  AND postcode_result_plz = %(p)s;"""

        self.cur.execute(query, {"v": VERSION_ID, "avg": avg_dis, "p": plz})

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

    def set_residential_buildings_table(self, plz: int):
        """
        * Fills buildings_tem with residential buildings which are inside the plz area
        :param plz:
        :return:
        """

        # Fill table
        query = """INSERT INTO buildings_tem (osm_id, area, type, geom, center, floors)
                   SELECT osm_id, area, building_t, geom, ST_Centroid(geom), floors::int
                   FROM res
                   WHERE ST_Contains((SELECT post.geom
                                      FROM postcode_result as post
                                      WHERE version_id = %(v)s
                                        AND postcode_result_plz = %(plz)s
                                      LIMIT 1), ST_Centroid(res.geom));
        UPDATE buildings_tem
        SET plz = %(plz)s
        WHERE plz ISNULL;"""
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
                   SELECT osm_id, area, use, geom, ST_Centroid(geom)
                   FROM oth AS o
                   WHERE o.use in ('Commercial', 'Public')
                     AND ST_Contains((SELECT post.geom
                                      FROM postcode_result as post
                                      WHERE version_id = %(v)s
                                        AND postcode_result_plz = %(plz)s), ST_Centroid(o.geom));;
        UPDATE buildings_tem
        SET plz = %(plz)s
        WHERE plz ISNULL;
        UPDATE buildings_tem
        SET floors = 1
        WHERE floors ISNULL;"""
        self.cur.execute(query, {"v": VERSION_ID, "plz": plz})

    def set_ways_tem_table(self, plz: int) -> int:
        """
        * Inserts ways inside the plz area to the ways_tem table
        :param plz:
        :return: number of ways in ways_tem
        """
        query = """INSERT INTO ways_tem
                   SELECT *
                   FROM ways AS w
                   WHERE ST_Intersects(w.geom, (SELECT geom
                                                FROM postcode_result
                                                WHERE version_id = %(v)s
                                                  AND postcode_result_plz = %(p)s));
        SELECT COUNT(*)
        FROM ways_tem;"""
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
                   SELECT *
                   FROM ways AS w
                   WHERE ST_Intersects(w.geom, ST_Transform(ST_GeomFromGeoJSON(%(shape)s), 3035));
        SELECT COUNT(*)
        FROM ways_tem;"""
        self.cur.execute(query, {"v": VERSION_ID, "shape": shape})
        count = self.cur.fetchone()[0]

        if count == 0:
            raise ValueError(f"Ways table is empty for the given plz: {shape}")

        return count

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

    def update_ways_cost(self) -> None:
        """
        Calculates the length of each way and stores in ways_tem.cost as meter
        """
        query = """UPDATE ways_tem
                   SET cost = ST_Length(geom);
        UPDATE ways_tem
        SET reverse_cost = cost;"""
        self.cur.execute(query)

    def delete_ways(self, vertices: list) -> None:
        """
        Deletes selected ways from ways_tem and ways_tem_vertices_pgr
        :param vertices:
        :return:
        """
        query = """DELETE
                   FROM ways_tem
                   WHERE target IN %(v)s;
        DELETE
        FROM ways_tem_vertices_pgr
        WHERE id IN %(v)s;"""
        self.cur.execute(query, {"v": tuple(map(int, vertices))})

    def delete_isolated_building(self, plz: int, kcid):
        query = """DELETE
                   FROM buildings_tem
                   WHERE plz = %(p)s
                     AND kcid = %(k)s
                     AND bcid ISNULL;"""
        self.cur.execute(query, {"p": plz, "k": kcid})

    def save_tables(self, plz: int):

        # finding duplicates that violate the buildings_result_pkey constraint
        # the key of building result is (version_id, osm_id, plz)
        query = """
                DELETE
                FROM buildings_tem a USING (SELECT MIN(ctid) as ctid, osm_id, plz
                                            FROM buildings_tem
                                            GROUP BY (osm_id, plz)
                                            HAVING COUNT(*) > 1) b
                WHERE a.osm_id = b.osm_id
                  AND a.plz = b.plz
                  AND a.ctid <> b.ctid;"""
        self.cur.execute(query)

        # Save building results
        query = f"""
                    INSERT INTO buildings_result
                    (version_id, osm_id, grid_result_id, area, type, geom, houses_per_building, center,
                    peak_load_in_kw, vertice_id, floors, connection_point)
                    SELECT '{VERSION_ID}' as version_id, osm_id, gr.grid_result_id, area, type, geom, houses_per_building,
                    center, peak_load_in_kw, vertice_id, floors, bt.connection_point
                    FROM buildings_tem bt
                    JOIN grid_result gr
                    ON bt.plz = gr.plz AND bt.kcid = gr.kcid AND bt.bcid = gr.bcid and gr.version_id = '{VERSION_ID}'
                    WHERE peak_load_in_kw != 0 AND peak_load_in_kw != -1;"""
        self.cur.execute(query)

        # Save ways results
        query = f"""INSERT INTO ways_result
                        SELECT '{VERSION_ID}' as version_id, clazz, source, target, cost, reverse_cost, geom, way_id,
                        %(p)s as plz FROM ways_tem;"""

        self.cur.execute(query, vars={"p": plz})

    def reset_tables(self):
        """
        Clears the temporary tables.
        """
        self.cur.execute("TRUNCATE TABLE buildings_tem")
        self.cur.execute("TRUNCATE TABLE ways_tem")
        self.cur.execute("TRUNCATE TABLE ways_tem_vertices_pgr")
        self.conn.commit()

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
            connection_available_cables_str = str(CONNECTION_AVAILABLE_CABLES).replace("'", "''")
            other_parameters_dict = {"LARGE_COMPONENT_LOWER_BOUND": LARGE_COMPONENT_LOWER_BOUND,
                                     "LARGE_COMPONENT_DIVIDER": LARGE_COMPONENT_DIVIDER, "VN": VN,
                                     "V_BAND_LOW": V_BAND_LOW, "V_BAND_HIGH": V_BAND_HIGH, }
            other_paramters_str = str(other_parameters_dict).replace("'", "''")

            insert_query = f"""INSERT INTO version (version_id, version_comment, consumer_categories, cable_cost_dict, connection_available_cables, other_parameters) VALUES
                ('{VERSION_ID}', '{VERSION_COMMENT}', '{consumer_categories_str}', '{cable_cost_dict_str}', '{connection_available_cables_str}', '{other_paramters_str}')"""
            self.cur.execute(insert_query)
            self.logger.info(f"Version: {VERSION_ID} (created for the first time)")

    def insert_parameter_tables(self, consumer_categories: pd.DataFrame):
        self.cur.execute("SELECT count(*) FROM consumer_categories")
        categories_exist = self.cur.fetchone()[0]
        with self.sqla_engine.begin() as conn:
            if not categories_exist:
                consumer_categories.to_sql(name="consumer_categories", con=conn, if_exists="append", index=False)
                self.logger.debug("Parameter tables are inserted")

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

    def copy_postcode_result_table_with_new_shape(self, plz: int, shape) -> None:
        """
        Copies the given plz entry from postcode to the postcode_result table
        :param plz:
        :return:
        """
        query = """INSERT INTO postcode_result (version_id, postcode_result_plz, settlement_type, geom, house_distance)
                   VALUES (%(v)s, %(p)s::INT, null, ST_Multi(ST_Transform(ST_GeomFromGeoJSON(%(shape)s), 3035)), null)
                   ON CONFLICT (version_id,postcode_result_plz) DO NOTHING;"""

        self.cur.execute(query, {"v": VERSION_ID, "p": plz, "shape": shape})

    def delete_plz_from_all_tables(self, plz: int, version_id: str) -> None:
        """
        Deletes all entries of corresponding networks in all tables for the given Version ID and plz.
        :param plz: Postal code
        :param version_id: Version ID
        """
        query = """DELETE
                   FROM postcode_result
                   WHERE version_id = %(v)s
                     AND postcode_result_plz = %(p)s;"""
        self.cur.execute(query, {"v": version_id, "p": int(plz)})
        self.conn.commit()
        self.logger.info(f"All data for PLZ {plz} and version {version_id} deleted")

    def delete_version_from_all_tables(self, version_id: str) -> None:
        """Delete all entries of the given version ID from all tables."""
        query = "DELETE FROM version WHERE version_id = %(v)s;"
        self.cur.execute(query, {"v": version_id})
        self.conn.commit()
        self.logger.info(f"Version {version_id} deleted from all tables")

    def delete_classification_version_from_related_tables(self, classification_id: str) -> None:
        """
        Deletes all rows with the given classification_id from related tables:
        transformer_classified, sample_set, and classification_version.

        :param classification_id: ID of the classification version to delete
        """
        query = "DELETE FROM classification_version WHERE classification_id = %(cid)s;"
        self.cur.execute(query, {"cid": classification_id})
        self.conn.commit()

        self.logger.info(f"Deleted classification ID {classification_id}.")

    def delete_plz_from_sample_set_table(self, classification_id: str, plz: int) -> None:
        """
        Deletes the row corresponding to the given classification ID and PLZ from the sample_set table.

        :param classification_id: ID of the classification version
        :param plz: Postal code to be removed
        """
        query = """
                DELETE
                FROM sample_set
                WHERE classification_id = %(cid)s
                  AND plz = %(p)s; \
                """
        self.cur.execute(query, {"cid": classification_id, "p": plz})
        self.conn.commit()
        self.logger.info(f"Deleted PLZ {plz} for classification ID {classification_id} from sample_set table.")

    def write_ags_log(self, ags: int) -> None:
        """write ags log to database: the amtliche gemeindeschluessel of the municipalities of which the buildings
        have already been imported to the database
        :param ags:  ags to be added
        :rtype ags: numpy integer 64
         """
        query = """INSERT INTO ags_log (ags)
                   VALUES (%(a)s); """
        self.cur.execute(query, {"a": int(ags), })
        self.conn.commit()

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