import warnings
from typing import override

import psycopg2 as psy
from sqlalchemy import create_engine

from src import utils
from src.config_loader import *
from src.database.analysis_mixin import AnalysisMixin
from src.database.clustering_mixin import ClusteringMixin
from src.database.grid_mixin import GridMixin
from src.database.preprocessing_mixin import PreprocessingMixin
from src.database.utils_mixin import UtilsMixin

warnings.simplefilter(action="ignore", category=UserWarning)


class DatabaseClient(
    PreprocessingMixin, ClusteringMixin, GridMixin, AnalysisMixin, UtilsMixin
):
    """Main database client handling connections."""

    def __init__(
        self, dbname=DBNAME, user=USER, pw=PASSWORD, host=HOST, port=PORT, **kwargs
    ):
        self.logger = utils.create_logger(
            name="DatabaseClient",
            log_level=LOG_LEVEL,
            log_file=LOG_FILE,
        )
        try:
            self.conn = psy.connect(
                database=dbname,
                user=user,
                password=pw,
                host=host,
                port=port,
                options=f"-c search_path={TARGET_SCHEMA},public",
            )
            self.cur = self.conn.cursor()
            self.conn.autocommit = True
            self.db_path = f"postgresql+psycopg2://{user}:{pw}@{host}:{port}/{dbname}"
            self.sqla_engine = create_engine(
                self.db_path,
                connect_args={
                    "options": f"-c search_path={TARGET_SCHEMA},public"},
            )
        except psy.OperationalError as err:
            self.logger.warning(
                f"Connecting to {dbname} was not successful. Make sure, that you have established the SSH connection with correct port mapping."
            )
            raise err

        # init supers after everything is set up
        super().__init__()

        self.logger.debug(
            f"DatabaseClient is constructed and connected to {self.db_path}."
        )

    def __del__(self):
        self.cur.close()
        self.conn.close()

    @override
    def get_connection(self):
        return self.conn

    @override
    def get_logger(self):
        return self.logger

    @override
    def get_sqla_engine(self):
        return self.sqla_engine

    def save_tables(self, regional_identifier: int):

        # finding duplicates that violate the buildings_result_pkey constraint
        # the key of building result is (version_id, osm_id,
        # regional_identifier)
        query = """
                DELETE
                FROM buildings_tem a USING (SELECT MIN(ctid) as ctid, osm_id, regional_identifier
                                            FROM buildings_tem
                                            GROUP BY (osm_id, regional_identifier)
                                            HAVING COUNT(*) > 1) b
                WHERE a.osm_id = b.osm_id
                  AND a.regional_identifier = b.regional_identifier
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
                    ON bt.regional_identifier = gr.regional_identifier AND bt.kcid = gr.kcid AND bt.bcid = gr.bcid and gr.version_id = '{VERSION_ID}'
                    WHERE peak_load_in_kw != 0 AND peak_load_in_kw != -1;"""
        self.cur.execute(query)

        # Save ways results
        query = f"""INSERT INTO ways_result (version_id, way_id, regional_identifier, clazz, source, target, cost, reverse_cost, geom)
                        SELECT '{VERSION_ID}' as version_id, way_id, %(p)s as regional_identifier, clazz, source, target, cost, reverse_cost, geom
                        FROM ways_tem;"""

        self.cur.execute(query, vars={"p": regional_identifier})

    def reset_tables(self):
        """
        Clears the temporary tables.
        """
        self.cur.execute("TRUNCATE TABLE buildings_tem")
        self.cur.execute("TRUNCATE TABLE ways_tem")
        self.cur.execute("TRUNCATE TABLE ways_tem_vertices_pgr")
        self.conn.commit()

    def delete_regional_identifier_from_all_tables(
            self, regional_identifier: int, version_id: str) -> None:
        """
        Deletes all entries of corresponding networks in all tables for the given Version ID and regional_identifier.
        :param regional_identifier: Postal code
        :param version_id: Version ID
        """
        query = """DELETE
                   FROM postcode_result
                   WHERE version_id = %(v)s
                     AND postcode_result_regional_identifier = %(p)s;"""
        self.cur.execute(
            query, {
                "v": version_id, "p": int(regional_identifier)})
        self.conn.commit()
        self.logger.info(
            f"All data for regional_identifier {regional_identifier} and version {version_id} deleted")

    def delete_version_from_all_tables(self, version_id: str) -> None:
        """Delete all entries of the given version ID from all tables."""
        query = "DELETE FROM version WHERE version_id = %(v)s;"
        self.cur.execute(query, {"v": version_id})
        self.conn.commit()
        self.logger.info(f"Version {version_id} deleted from all tables")

    def delete_classification_version_from_related_tables(
        self, classification_id: str
    ) -> None:
        """
        Deletes all rows with the given classification_id from related tables:
        transformer_classified, sample_set, and classification_version.

        :param classification_id: ID of the classification version to delete
        """
        query = "DELETE FROM classification_version WHERE classification_id = %(cid)s;"
        self.cur.execute(query, {"cid": classification_id})
        self.conn.commit()

        self.logger.info(f"Deleted classification ID {classification_id}.")

    def delete_regional_identifier_from_sample_set_table(
        self, classification_id: str, regional_identifier: int
    ) -> None:
        """
        Deletes the row corresponding to the given classification ID and regional_identifier from the sample_set table.

        :param classification_id: ID of the classification version
        :param regional_identifier: Postal code to be removed
        """
        query = """
                DELETE
                FROM sample_set
                WHERE classification_id = %(cid)s
                  AND regional_identifier = %(p)s; \
                """
        self.cur.execute(
            query, {
                "cid": classification_id, "p": regional_identifier})
        self.conn.commit()
        self.logger.info(
            f"Deleted regional_identifier {regional_identifier} for classification ID {classification_id} from sample_set table."
        )

    def delete_transformers(self) -> None:
        """all transformers are deleted from table transformers in database"""
        delete_query = "TRUNCATE TABLE transformers;"
        self.cur.execute(delete_query)
        self.conn.commit()
        self.logger.info("Transformers deleted.")

    def write_fips_log(self, fips_code: int) -> None:
        """write ags log to database: the amtliche gemeindeschluessel of the municipalities of which the buildings
        have already been imported to the database
        :param fips_code:  fips_code to be added
        :rtype ags: numpy integer 64
        """
        query = """INSERT INTO fips_log (fips_code)
                   VALUES (%(f)s); """
        self.cur.execute(
            query,
            {
                "f": int(fips_code),
            },
        )
        self.conn.commit()
