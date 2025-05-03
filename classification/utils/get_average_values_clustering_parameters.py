import warnings
import os
import sys
import pandas as pd
import psycopg2 as pg

# Determine the project's root directory and add to Python's module search path
PROJECT_ROOT = os.path.abspath(os.path.join(os.getcwd(), "../../.."))
sys.path.append(PROJECT_ROOT)

from classification.config_loader import *
from pylovo.config_data import *


def get_clustering_parameters_for_kmeans_cluster_0() -> pd.DataFrame:
    """
    Get clustering parameters for rows where kmeans_clusters = 0 in transformer_classified.

    :return: A DataFrame with matched clustering parameters.
    """
    # Connect to the database
    conn = pg.connect(
        database=DBNAME, user=USER, password=PASSWORD, host=HOST, port=PORT
    )

    try:
        # Run the query
        query = """
            SELECT cp.*
            FROM public.clustering_parameters cp
            JOIN (
                SELECT version_id, plz, kcid, bcid
                FROM public.transformer_classified
                WHERE kmeans_clusters = 0
                GROUP BY version_id, plz, kcid, bcid
            ) tc
            ON cp.version_id = tc.version_id
            AND cp.plz = tc.plz
            AND cp.kcid = tc.kcid
            AND cp.bcid = tc.bcid;
        """

        df = pd.read_sql_query(query, con=conn)

    finally:
        # Close the connection
        conn.close()

    return df

def calculate_average_clustering_parameters(df: pd.DataFrame, parameters: list) -> dict:
    """
    Calculate the average values for the given clustering parameters.

    :param df: DataFrame with clustering parameters.
    :param parameters: List of parameter names to calculate averages for.
    :return: Dictionary with average values.
    """
    avg_values = {}

    for field in parameters:
        avg = df[field].mean()
        avg_values[field] = round(avg, 3)  # Rounded to 3 decimals
    return avg_values


def main():
    # Get all clustering parameters
    df_clustering_parameters = get_clustering_parameters_for_kmeans_cluster_0()

    # Calculate average values using LIST_OF_CLUSTERING_PARAMETERS
    averages = calculate_average_clustering_parameters(df_clustering_parameters, LIST_OF_CLUSTERING_PARAMETERS)

    # Print the average values
    print("Average Clustering Parameter Values:")
    for param, avg in averages.items():
        print(f"{param}: {avg}")


if __name__ == "__main__":
    main()
