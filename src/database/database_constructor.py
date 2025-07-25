import os
import subprocess
import time
import warnings
from pathlib import Path

import pandas as pd
import psycopg2 as psy
import sqlparse

import src.database.database_client as dbc
from config.config_table_structure import *
from src.config_loader import *
from src.load_data.load_transformers import (
    EPSG, RELATION_ID, fetch_trafos, get_trafos_processed_3035_geojson_path,
    get_trafos_processed_geojson_path, process_trafos)

# uncomment for automated building import of buildings in regiostar_samples
# from raw_data.import_building_data import OGR_FILE_LIST


class DatabaseConstructor:
    """
    Constructs a ready to use src database. Be careful about overwriting the tables.
    It uses databaseClient to connect to the database and create tables and import data.
    """

    def __init__(self, dbc_obj=None):
        self.extensions_added = False

        if dbc_obj:
            self.dbc = dbc_obj
        else:
            self.dbc = dbc.DatabaseClient()

    def get_table_name_list(self):
        with self.dbc.conn.cursor() as cur:
            cur.execute(
                """SELECT table_name FROM information_schema.tables
                   WHERE table_schema = %s""", (TARGET_SCHEMA,)
            )
            table_name_list = [tup[0] for tup in cur.fetchall()]

        return table_name_list

    def table_exists(self, table_name):
        if table_name in self.get_table_name_list():
            warnings.warn(f"{table_name} table is overwritten!")
            return True
        else:
            return False

    def create_table(self, table_name):
        # create extension if not exists for recognition of geom datatypes
        if not self.extensions_added:
            with self.dbc.conn.cursor() as cur:
                # create schema if not exists
                cur.execute(f"CREATE SCHEMA IF NOT EXISTS {TARGET_SCHEMA};")
                print(f"CREATE SCHEMA {TARGET_SCHEMA}")
                self.dbc.conn.commit()
                # create extension if not exists for recognition of geom
                # datatypes
                cur.execute("CREATE EXTENSION IF NOT EXISTS postgis;")
                print("CREATE EXTENSION postgis")
                cur.execute("CREATE EXTENSION IF NOT EXISTS pgRouting;")
                print("CREATE EXTENSION pgRouting")
                self.dbc.conn.commit()
                self.extensions_added = True

        if table_name == "all":
            try:
                with self.dbc.conn.cursor() as cur:
                    for table_name, query in CREATE_QUERIES.items():
                        cur.execute(query, {"epsg": EPSG})
                        print(f"CREATE TABLE {table_name}")
                self.dbc.conn.commit()
            except (Exception, psy.DatabaseError) as error:
                raise error
        elif table_name in CREATE_QUERIES:
            try:
                with self.dbc.conn.cursor() as cur:
                    cur.execute(CREATE_QUERIES[table_name], {"epsg": EPSG})
                    print(f"CREATE TABLE {table_name}")
                self.dbc.conn.commit()
            except (Exception, psy.DatabaseError) as error:
                raise error
        else:
            raise ValueError(
                f"Table name {table_name} is not a valid parameter value for the function create_table. See config.py"
            )

    def ogr_to_db(self, ogr_file_list, skip_failures: bool = False):
        """
            OGR/GDAL is a translator library for raster and vector geospatial data formats
            inserts building data specified into database
        """

        for file_dict in ogr_file_list:
            st = time.time()
            file_path = Path(file_dict["path"])
            assert file_path.exists(), file_path
            file_name = file_path.stem
            table_name = file_dict.get("table_name", file_name)

            table_exists = self.table_exists(table_name=table_name)
            print("ogr working for table", table_name)
            command = [
                "ogr2ogr",
                "-append" if table_exists else "-overwrite",
                "-progress",
                "-f",
                "PostgreSQL",
                f"PG:dbname={DBNAME} user={USER} password={PASSWORD} host={HOST} port={PORT}",
                file_path,
                "-nln",
                # explicitly tells ogr2ogr where to append (for the case of
                # table already existing)
                f"{TARGET_SCHEMA}.{table_name}",
                "-nlt",
                # "MULTIPOLYGON",
                "PROMOTE_TO_MULTI",
                "-t_srs",
                f"EPSG:{EPSG}",
                "-lco",
                "geometry_name=geom",
                # ensures creation happens in correct schema
                "-lco", f"SCHEMA={TARGET_SCHEMA}",
            ]
            if skip_failures:
                command.append("-skipfailures")

            result = subprocess.run(
                command,
                check=True,
                shell=False,
                stderr=subprocess.PIPE if skip_failures else None)
            if skip_failures:
                error_list = result.stderr.decode().replace("\r", "").split("\n")
                error_list = [e[e.find("ERROR: "):e.find("DETAIL: ")]
                              for e in error_list]
                error_list = [e.strip("\n")
                              for e in error_list if "ERROR: " in e]
                error_set = set(error_list)

                print(
                    f"Warning: Error(s) occurred while processing {file_name}:")
                for error in error_set:
                    print("\t" + error)
                    if "duplicate key value violates unique constraint" in error:
                        print(
                            "\tThis is likely due to importing already existing data.")

            et = time.time()
            print(f"{file_name} is successfully imported to db in {int(et - st)} s")

    def transformers_to_db(self):
        """Call the overpass api for transformer data and populate the transformers table.
        Delete raw_data/transformer_data/processed_trafos/*_trafos_processed.geojson to
        fetch fresh data from OSM.

        """
        trafos_processed_geojson_path = get_trafos_processed_geojson_path(
            RELATION_ID)
        trafos_processed_3035_geojson_path = get_trafos_processed_3035_geojson_path(
            RELATION_ID)

        update_trafos = not os.path.isfile(trafos_processed_geojson_path)

        if update_trafos:
            print(
                f"{trafos_processed_geojson_path} does not exist -> fetch transformer data from API and process it")
            fetch_trafos(RELATION_ID)
            process_trafos(RELATION_ID)

        in_file = trafos_processed_geojson_path
        out_file = trafos_processed_3035_geojson_path

        if update_trafos or not os.path.isfile(out_file):
            # Convert the GeoJSON file to EPSG and write to a new file
            subprocess.run(
                [
                    "ogr2ogr",
                    "-f", "GeoJSON",
                    "-s_srs", f"EPSG:{str(EPSG)}",
                    "-t_srs", f"EPSG:{str(EPSG)}",
                    out_file,  # output
                    in_file  # input
                ],
                shell=False
            )

        trafo_dict = [
            {
                "path": out_file,
                "table_name": "transformers"
            }
        ]
        self.ogr_to_db(trafo_dict)

    def csv_to_db(self, file_dict, overwrite=False):
        """
        Import a single CSV file to database.

        Args:
            file_dict: Dict with 'path' and optional 'table_name'
            overwrite: If True, delete existing data first. If False, just append.
        """
        st = time.time()
        file_path = Path(file_dict["path"])
        assert file_path.exists(), file_path
        file_name = file_path.stem
        table_name = file_dict.get("table_name", file_name)

        # Read CSV data
        df = pd.read_csv(file_path, index_col=False)
        df = df.rename(
            columns={
                "subdivision_name": "subdivision_name",
                "fipscode": "plz"})

        if overwrite and self.table_exists(table_name=table_name):
            # Delete all existing data if overwrite is True
            with self.dbc.conn.cursor() as cur:
                cur.execute(f"DELETE FROM {table_name}")
                self.dbc.conn.commit()
            print(f"Cleared existing data from {table_name}")

        df.to_sql(
            name=table_name,
            con=self.dbc.sqla_engine,
            if_exists="append",
            index=False,
        )

        et = time.time()
        print(f"{file_name}: {len(df)} records processed in {int(et - st)}s")

    def create_public_2po_table(self):
        """
        Reads the large SQL file in 10% chunks, executes complete statements on-the-fly,
        and defers incomplete statements until the next chunk.
        """
        cur = self.dbc.conn.cursor()

        # Path to your SQL file, which includes creation of the table
        sc_path = os.path.join(
            os.getcwd(),
            "raw_data",
            "imports",
            REGION['STATE'].replace(' ', '_'),
            REGION['COUNTY'].replace(' ', '_'),
            REGION['COUNTY_SUBDIVISION'].replace(' ', '_'),
            "STREET_NETWORK",
            "ways_public_2po_4pgr.sql")

        file_size = os.path.getsize(sc_path)

        # We read 10% at a time.  (Or pick a chunk size in bytes that works for
        # your environment.)
        chunk_size = max(1, file_size // 100)
        chars_read = 0

        leftover = ""  # Holds any partial statement that didn't end with a semicolon

        print("\nStart inserting ways into public_2po_4pgr table.")
        with open(sc_path, 'r', encoding='utf-8') as sc_file:
            while True:
                # Read next chunk
                data = sc_file.read(chunk_size)
                if not data:
                    # No more data to read
                    break

                chars_read += len(data)
                progress = round(chars_read * 100 / file_size)
                print(f"\rProgress: {progress}%", end="", flush=True)

                # Combine leftover from previous read with current chunk
                combined = leftover + data

                # Use sqlparse to split out complete statements
                statements = sqlparse.split(combined)

                # If sqlparse.split() returns multiple statements, the last one
                # might be incomplete. We’ll keep it as leftover if needed.
                if len(statements) > 1:
                    # Execute all statements except possibly the last
                    for stmt in statements[:-1]:
                        stmt = stmt.strip()
                        if stmt and not stmt.startswith('--'):
                            cur.execute(stmt)
                            self.dbc.conn.commit()

                    # Check if the last statement ends with a semicolon or not
                    last_stmt = statements[-1].strip()
                    if last_stmt.endswith(
                            ';') and not last_stmt.startswith('--'):
                        # It's a complete statement
                        cur.execute(last_stmt)
                        self.dbc.conn.commit()
                        leftover = ""
                    else:
                        leftover = last_stmt
                else:
                    # 0 or 1 statements from sqlparse
                    if len(statements) == 1:
                        # Could be complete or incomplete
                        stmt = statements[0].strip()
                        if stmt.endswith(';') and not stmt.startswith('--'):
                            # It's complete, execute it
                            cur.execute(stmt)
                            self.dbc.conn.commit()
                            leftover = ""
                        else:
                            # It's incomplete, keep it
                            leftover = stmt
                    else:
                        # No statements found. This can happen if combined was empty or whitespace.
                        # Just continue reading next chunk
                        pass
        print("\nInserted all ways into public_2po_4pgr table.")

    def ways_to_db(self):
        """This function transform the output of osm2po to the ways table, refer to the issue
        https://github.com/TongYe1997/Connector-syn-grid/issues/19"""

        st = time.time()

        cur = self.dbc.conn.cursor()

        # Transform to ways table
        query = """INSERT INTO ways
            SELECT  clazz,
                    source,
                    target,
                    cost,
                    reverse_cost,
                    ST_Transform(geom_way, %(epsg)s) as geom,
                    id AS way_id
            FROM public_2po_4pgr"""
        cur.execute(query, {"epsg": EPSG})

        # Drop public_2po_4pgr table, as it is not needed anymore
        query = "DROP TABLE public_2po_4pgr"
        cur.execute(query)

        self.dbc.conn.commit()

        et = time.time()
        print(f"Ways are successfully imported to db in {int(et - st)} s")

    def dump_functions(self):
        """
        Creates the SQL functions that are needed for the app to operate
        """
        cur = self.dbc.conn.cursor()
        sc_path = os.path.join(
            os.getcwd(),
            "src",
            "postgres_dump_functions.sql")
        with open(sc_path, 'r') as sc_file:
            print(
                f"Executing postgres_dump_functions.sql script with schema '{TARGET_SCHEMA}'.")
            sql = sc_file.read()
            cur.execute(sql)
            self.dbc.conn.commit()

    def drop_all_tables(self):
        """
        Drops all tables in the database
        """
        cur = self.dbc.conn.cursor()

        cur.execute("DROP EXTENSION IF EXISTS pgRouting CASCADE;")
        print("Dropped pgRouting extension.")
        cur.execute("DROP EXTENSION IF EXISTS postgis CASCADE;")
        print("Dropped postgis extension.")

        cur.execute(f"DROP SCHEMA {TARGET_SCHEMA} CASCADE")
        self.dbc.conn.commit()
