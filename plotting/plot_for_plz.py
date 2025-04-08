from plotting.config_plots import ACCESS_TOKEN_PLOTLY
from pylovo.GridGenerator import GridGenerator
from pylovo.config_data import *
from pylovo.config_version import *

import os
import sys
import pandas as pd
import numpy as np
import plotly.express as px
from matplotlib import pyplot as plt
import pandapower as pp
import plotly
from pandapower.plotting.plotly import vlevel_plotly
from pandapower.plotting.plotly.mapbox_plot import set_mapbox_token
from pathlib import Path

sys.path.append(os.path.abspath('..'))
px.set_mapbox_access_token(ACCESS_TOKEN_PLOTLY)


def plot_pie_of_trafo_cables(plz):
    """
    Plots a pie chart of the trafos and cable types for a plz
    """
    gg = GridGenerator(plz=plz)
    pg = gg.pgr
    data_list, data_labels, trafo_dict = pg.read_per_trafo_dict(plz=plz)
    fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(16, 4))
    # Plot Transformer size distribution
    axs[0].pie(trafo_dict.values(), labels=trafo_dict.keys(), autopct='%1.1f%%',
               pctdistance=1.15, labeldistance=.6)
    axs[0].set_title('Transformer Size Distribution', fontsize=14)
    # Plot cable length distribution
    cable_dict = pg.read_cable_dict(plz)
    axs[1].pie(cable_dict.values(), labels=cable_dict.keys(), autopct="%.1f%%")
    axs[1].set_title("Installed Cable Length", fontsize=14)
    plt.show()


def plot_hist_trafos(plz):
    """
    plots histogram of trafo sizes in plz
    """
    gg = GridGenerator(plz=plz)
    pg = gg.pgr
    data_list, data_labels, trafo_dict = pg.read_per_trafo_dict(plz=plz)
    plt.bar(trafo_dict.keys(), height=trafo_dict.values(), width=0.3)
    plt.title('Transformer Size Distribution', fontsize=14)
    plt.xlabel("Trafo size")
    plt.ylabel("Count")
    plt.show()


def plot_boxplot_plz(plz):
    """
    Boxplot of load number, bus number, simultaneaous load peak, max trafo distance, avg trafo distance
    """
    gg = GridGenerator(plz=plz)
    pg = gg.pgr
    data_list, data_labels, trafo_dict = pg.read_per_trafo_dict(plz=plz)
    trafo_sizes = list(data_list[0].keys())
    values = [list(d.values()) for d in data_list]

    # Create the figure and axes objects
    fig, axs = plt.subplots(nrows=1, ncols=len(data_list), figsize=(16, 4), sharey=True)
    for i, data_label in enumerate(data_labels):
        axs[i].boxplot(values[i], labels=trafo_sizes, vert=False, showfliers=False, patch_artist=True, notch=False)
        axs[i].set_title(data_label, fontsize=12)
    fig.supxlabel('Values', fontsize=12)
    fig.supylabel('Transformer Size (kVA)', fontsize=12)

    # Adjust the layout and save the plot
    plt.tight_layout()
    plt.show()


def plot_cable_length_of_types(plz):
    """
    Plots distribution of cable length by length
    """
    gg = GridGenerator(plz=plz)
    pg = gg.pgr
    # distributed according to cross_section
    cluster_list = pg.get_list_from_plz(plz)
    cable_length_dict = {}
    for kcid, bcid in cluster_list:
        try:
            net = pg.read_net(plz, kcid, bcid)
        except Exception as e:
            print(f" local network {kcid},{bcid} is problematic")
            raise e
        else:
            cable_df = net.line[net.line["in_service"] == True]

            cable_type = pd.unique(cable_df["std_type"]).tolist()
            for type in cable_type:

                if type in cable_length_dict:
                    cable_length_dict[type] += (
                            cable_df[cable_df["std_type"] == type]["parallel"]
                            * cable_df[cable_df["std_type"] == type]["length_km"]
                    ).sum()

                else:
                    cable_length_dict[type] = (
                            cable_df[cable_df["std_type"] == type]["parallel"]
                            * cable_df[cable_df["std_type"] == type]["length_km"]
                    ).sum()
    plt.bar(cable_length_dict.keys(), height=cable_length_dict.values(), width=0.3)
    plt.title('Cable Type Distribution', fontsize=14)
    plt.xlabel("Cable type")
    plt.ylabel("Length in m")
    plt.show()


def get_trafo_dicts(plz):
    """
    Retrieve load count, bus count and cable lenth per type for a plz
    """
    gg = GridGenerator(plz=plz)
    pg = gg.pgr
    cluster_list = pg.get_list_from_plz(plz)
    load_count_dict = {}
    bus_count_dict = {}
    cable_length_dict = {}
    trafo_dict = {}
    print("start basic parameter counting")
    for kcid, bcid in cluster_list:
        load_count = 0
        bus_list = []
        net = pg.read_net(plz, kcid, bcid)
        for row in net.load[["name", "bus"]].itertuples():
            load_count += 1
            bus_list.append(row.bus)
        bus_list = list(set(bus_list))
        bus_count = len(bus_list)
        cable_length = net.line['length_km'].sum()

        for row in net.trafo[["sn_mva", "lv_bus"]].itertuples():
            capacity = round(row.sn_mva * 1e3)

            if capacity in trafo_dict:
                trafo_dict[capacity] += 1

                load_count_dict[capacity].append(load_count)
                bus_count_dict[capacity].append(bus_count)
                cable_length_dict[capacity].append(cable_length)

            else:
                trafo_dict[capacity] = 1

                load_count_dict[capacity] = [load_count]
                bus_count_dict[capacity] = [bus_count]
                cable_length_dict[capacity] = [cable_length]
    return load_count_dict, bus_count_dict, cable_length_dict

def plot_trafo_on_map(plz, save_plots: bool = False) -> None:
    """
    Transformer types are plotted by their capacity on a plotly basemap
    """

    net_plot = pp.create_empty_network()
    gg = GridGenerator(plz=plz)
    pg = gg.pgr
    cluster_list = pg.get_list_from_plz(plz)
    grid_index = 1
    set_mapbox_token("pk.eyJ1IjoiYmVuZWhhcm8iLCJhIjoiY205OGdwejJ1MDJsbzJsczl1ajdyYmlzaSJ9.HWA8ZLQm1Sp0Whs5PADxrw") # public token
    for kcid, bcid in cluster_list:
        net = pg.read_net(plz, kcid, bcid)
        for row in net.trafo[["sn_mva", "lv_bus"]].itertuples():
            trafo_size = round(row.sn_mva * 1e3)
            trafo_geom = np.array(net.bus_geodata.loc[row.lv_bus, ["x", "y"]])
            pp.create_bus(
                net_plot,
                name="Distribution_grid_"
                     + str(grid_index)
                     + "<br>"
                     + "transformer: "
                     + str(trafo_size)
                     + "_kVA",
                vn_kv=trafo_size,
                geodata=trafo_geom,
                type="b",
            )
            grid_index += 1

    figure = vlevel_plotly(
        net_plot, on_map=True, colors_dict=PLOT_COLOR_DICT, projection="epsg:4326"
    )

    if save_plots:
        savepath_folder = Path(
            RESULT_DIR, "figures", f"version_{VERSION_ID}", plz
        )
        savepath_folder.mkdir(parents=True, exist_ok=True)
        savepath_file = Path(savepath_folder, "trafo_on_map.html")
        plotly.offline.plot(
            figure,
            filename=savepath_file,
        )