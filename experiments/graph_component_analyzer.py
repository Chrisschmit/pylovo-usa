"""
Graph Component Analysis and Visualization

This module provides functionality to analyze and visualize connected components
of street networks using PostGIS routing functions and matplotlib.
"""

import io
import logging
import sys
from contextlib import redirect_stderr
from typing import Dict, List, Tuple

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import psycopg2
from matplotlib.colors import ListedColormap
from sqlalchemy import create_engine

from src.config_loader import *


class GraphComponentAnalyzer:
    """Analyzes and visualizes connected components of street networks."""
    
    def __init__(self, dbname=DBNAME, user=USER, pw=PASSWORD, host=HOST, port=PORT):
        """Initialize database connection."""
        self.conn = psycopg2.connect(
            database=dbname,
            user=user,
            password=pw,
            host=host,
            port=port,
            options=f"-c search_path={TARGET_SCHEMA},public",
        )
        self.cur = self.conn.cursor()
        self.conn.autocommit = True
        
        # SQLAlchemy engine for geopandas
        self.db_path = f"postgresql+psycopg2://{user}:{pw}@{host}:{port}/{dbname}"
        self.engine = create_engine(
            self.db_path,
            connect_args={"options": f"-c search_path={TARGET_SCHEMA},public"},
        )
    
    def __del__(self):
        """Clean up database connections."""
        if hasattr(self, 'cur'):
            self.cur.close()
        if hasattr(self, 'conn'):
            self.conn.close()
    
    def capture_pgr_analyze_notices(self) -> List[str]:
        """
        Capture notices from pgr_analyzeGraph execution.
        
        Returns:
            List of notice messages from the pgr_analyzeGraph function
        """
        notices = []
        
        # Clear any existing notices
        self.conn.notices[:] = []
        
        # Execute the analyze graph query
        analyze_query = "SELECT pgr_analyzeGraph('ways_tem', 0.1, the_geom:='geom', id:='way_id');"
        self.cur.execute(analyze_query)
        
        # Capture notices from the connection
        for notice in self.conn.notices:
            notices.append(notice.strip())
        
        return notices
    
    def get_component_statistics(self) -> pd.DataFrame:
        """
        Get statistics for each connected component.
        
        Returns:
            DataFrame with component statistics including vertex and edge counts
        """
        stats_query = """
        WITH comps AS (
            SELECT component, node
            FROM pgr_connectedComponents(
                'SELECT way_id AS id,
                        source,
                        target,
                        COALESCE(cost, 1.0) AS cost,
                        COALESCE(reverse_cost, 1.0) AS reverse_cost
                 FROM ways_tem'
            )
        ),
        component_vertices AS (
            SELECT component,
                   COUNT(DISTINCT node) AS vertex_count
            FROM comps
            GROUP BY component
        ),
        component_edges AS (
            SELECT wt.way_id,
                   c1.component
            FROM ways_tem wt
            JOIN comps c1 ON wt.source = c1.node
            JOIN comps c2 ON wt.target = c2.node AND c1.component = c2.component
        ),
        edge_counts AS (
            SELECT component,
                   COUNT(*) AS edge_count
            FROM component_edges
            GROUP BY component
        )
        SELECT cv.component,
               COALESCE(ec.edge_count, 0) AS edge_count,
               cv.vertex_count
        FROM component_vertices cv
        LEFT JOIN edge_counts ec ON cv.component = ec.component
        ORDER BY COALESCE(ec.edge_count, 0) DESC;
        """
        
        return pd.read_sql_query(stats_query, self.engine)
    
    def get_component_geometries(self) -> gpd.GeoDataFrame:
        """
        Get geometries for each connected component.
        
        Returns:
            GeoDataFrame with component geometries
        """
        geom_query = """
        WITH comps AS (
            SELECT component, node
            FROM pgr_connectedComponents(
                'SELECT way_id AS id,
                        source,
                        target,
                        COALESCE(cost, 1.0) AS cost,
                        COALESCE(reverse_cost, 1.0) AS reverse_cost
                 FROM ways_tem'
            )
        ),
        way_components AS (
            SELECT wt.way_id,
                   wt.geom,
                   c1.component
            FROM ways_tem wt
            JOIN comps c1 ON wt.source = c1.node
            JOIN comps c2 ON wt.target = c2.node AND c1.component = c2.component
        )
        SELECT way_id, geom, component
        FROM way_components
        ORDER BY component;
        """
        
        return gpd.read_postgis(geom_query, self.engine, geom_col='geom')
    
    def visualize_components(self, save_path: str = None, figsize: Tuple[int, int] = (15, 10)):
        """
        Create a visualization of connected components in different colors.
        
        Args:
            save_path: Optional path to save the plot
            figsize: Figure size as (width, height)
        """
        # Get component data
        gdf = self.get_component_geometries()
        stats_df = self.get_component_statistics()
        
        if gdf.empty:
            print("No component data found. Make sure ways_tem table exists and has data.")
            return
        
        # Create figure and axis
        fig, ax = plt.subplots(1, 1, figsize=figsize)
        
        # Get unique components and create color map
        unique_components = sorted(gdf['component'].unique())
        n_components = len(unique_components)
        
        # Use a colormap that provides good contrast
        if n_components <= 10:
            colors = plt.cm.tab10(np.linspace(0, 1, n_components))
        else:
            colors = plt.cm.hsv(np.linspace(0, 1, n_components))
        
        # Plot each component in a different color
        for i, component in enumerate(unique_components):
            component_data = gdf[gdf['component'] == component]
            component_data.plot(
                ax=ax, 
                color=colors[i], 
                linewidth=2,
                label=f'Component {component}'
            )
        
        # Customize the plot
        ax.set_title(f'Street Network Connected Components\n{n_components} Components Found', 
                    fontsize=16, fontweight='bold')
        ax.set_xlabel('Longitude', fontsize=12)
        ax.set_ylabel('Latitude', fontsize=12)
        
        # Add legend only if components are manageable in number
        if n_components <= 20:
            ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
        
        # Add grid
        ax.grid(True, alpha=0.3)
        
        # Adjust layout
        plt.tight_layout()
        
        # Add statistics text box
        stats_text = f"Total Components: {n_components}\n"
        stats_text += f"Total Edges: {stats_df['edge_count'].sum()}\n"
        stats_text += f"Total Vertices: {stats_df['vertex_count'].sum()}"
        
        # Add text box with statistics
        props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
        ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, fontsize=10,
                verticalalignment='top', bbox=props)
        
        # Save or show plot
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Plot saved to: {save_path}")
        else:
            plt.show()
        
        return fig, ax
    
    def print_component_summary(self):
        """Print a summary of component analysis."""
        # Get statistics
        stats_df = self.get_component_statistics()
        
        if stats_df.empty:
            print("No component data found.")
            return
        
        print("=" * 60)
        print("CONNECTED COMPONENTS ANALYSIS SUMMARY")
        print("=" * 60)
        
        total_components = len(stats_df)
        total_edges = stats_df['edge_count'].sum()
        total_vertices = stats_df['vertex_count'].sum()
        
        print(f"Total number of connected components: {total_components}")
        print(f"Total edges: {total_edges}")
        print(f"Total vertices: {total_vertices}")
        print(f"Average edges per component: {total_edges / total_components:.2f}")
        print(f"Average vertices per component: {total_vertices / total_components:.2f}")
        
        print("\nLargest components (by edge count):")
        print("-" * 40)
        top_10 = stats_df.head(10)
        for _, row in top_10.iterrows():
            print(f"Component {row['component']:3d}: {row['edge_count']:4d} edges, "
                  f"{row['vertex_count']:4d} vertices")
        
        print("\nSmallest components (by edge count):")
        print("-" * 40)
        bottom_10 = stats_df.tail(10)
        for _, row in bottom_10.iterrows():
            print(f"Component {row['component']:3d}: {row['edge_count']:4d} edges, "
                  f"{row['vertex_count']:4d} vertices")
        
        print("=" * 60)
    
    def run_full_analysis(self, output_dir: str = "experiments/"):
        """
        Run complete component analysis including topology creation, analysis, and visualization.
        
        Args:
            output_dir: Directory to save output files
        """
        print("Starting graph component analysis...")
        
        # Step 1: Create topology
        print("Creating topology...")
        topology_query = "SELECT pgr_createTopology('ways_tem', 0.1, id:='way_id', the_geom:='geom', clean:=true)"
        self.cur.execute(topology_query)
        
        # Step 2: Analyze graph and capture notices
        print("Analyzing graph and capturing notices...")
        notices = self.capture_pgr_analyze_notices()
        
        # Print captured notices
        if notices:
            print("\nPgRouting Analysis Notices:")
            print("-" * 40)
            for notice in notices:
                print(f"  {notice}")
        else:
            print("No notices captured from pgr_analyzeGraph")
        
        # Step 3: Get and print component statistics
        print("\nGenerating component statistics...")
        self.print_component_summary()
        
        # Step 4: Create visualization
        print("\nCreating visualization...")
        output_path = f"{output_dir}/component_visualization.png"
        self.visualize_components(save_path=output_path)
        
        # Step 5: Save detailed statistics to CSV
        stats_df = self.get_component_statistics()
        stats_path = f"{output_dir}/component_statistics.csv"
        stats_df.to_csv(stats_path, index=False)
        print(f"Detailed statistics saved to: {stats_path}")
        
        print("\nAnalysis complete!")


def main():
    """Main function to run the analysis."""
    analyzer = GraphComponentAnalyzer()
    analyzer.run_full_analysis()


if __name__ == "__main__":
    main()