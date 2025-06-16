import json
import warnings

import pandapower.topology as top

from src import utils
from src.config_loader import *

warnings.simplefilter(action="ignore", category=UserWarning)


class AnalysisMixin:
    def analyse_basic_parameters(self, plz: int):
        cluster_list = self.get_list_from_plz(plz)
        count = len(cluster_list)
        time = 0
        percent = 0

        load_count_dict = {}
        bus_count_dict = {}
        cable_length_dict = {}
        trafo_dict = {}
        self.logger.debug("start basic parameter counting")
        for kcid, bcid in cluster_list:
            load_count = 0
            bus_list = []
            try:
                net = self.read_net(plz, kcid, bcid)
            except Exception as e:
                self.logger.warning(f" local network {kcid},{bcid} is problematic")
                raise e
            else:
                for row in net.load[["name", "bus"]].itertuples():
                    load_count += 1
                    bus_list.append(row.bus)
                bus_list = list(set(bus_list))
                bus_count = len(bus_list)
                cable_length = net.line["length_km"].sum()

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

            time += 1
            if time / count >= 0.1:
                percent += 10
                self.logger.info(f"{percent} percent finished")
                time = 0
        self.logger.info("analyse_basic_parameters finished.")
        trafo_string = json.dumps(trafo_dict)
        load_count_string = json.dumps(load_count_dict)
        bus_count_string = json.dumps(bus_count_dict)

        self.insert_plz_parameters(plz, trafo_string, load_count_string, bus_count_string)

    def analyse_cables(self, plz: int):
        cluster_list = self.get_list_from_plz(plz)
        count = len(cluster_list)
        time = 0
        percent = 0

        # distributed according to cross_section
        cable_length_dict = {}
        for kcid, bcid in cluster_list:
            try:
                net = self.read_net(plz, kcid, bcid)
            except Exception as e:
                self.logger.debug(f" local network {kcid},{bcid} is problematic")
                raise e
            else:
                cable_df = net.line[net.line["in_service"] == True]

                cable_type = pd.unique(cable_df["std_type"]).tolist()
                for type in cable_type:

                    if type in cable_length_dict:
                        cable_length_dict[type] += (cable_df[cable_df["std_type"] == type]["parallel"] *
                                                    cable_df[cable_df["std_type"] == type]["length_km"]).sum()

                    else:
                        cable_length_dict[type] = (cable_df[cable_df["std_type"] == type]["parallel"] *
                                                   cable_df[cable_df["std_type"] == type]["length_km"]).sum()
            time += 1
            if time / count >= 0.1:
                percent += 10
                self.logger.info(f"{percent} % processed")
                time = 0
        self.logger.info("analyse_cables finished.")
        cable_length_string = json.dumps(cable_length_dict)

        update_query = """UPDATE plz_parameters
            SET cable_length = %(c)s 
            WHERE version_id = %(v)s AND plz = %(p)s;"""
        self.cur.execute(update_query, {"v": VERSION_ID, "c": cable_length_string,
                                        "p": plz})  # TODO: change to cable_length_per_type, add cable_length_per_trafo

        self.logger.debug("cable count finished")

    def analyse_per_trafo_parameters(self, plz: int):
        cluster_list = self.get_list_from_plz(plz)
        count = len(cluster_list)
        time = 0
        percent = 0

        trafo_load_dict = {}
        trafo_max_distance_dict = {}
        trafo_avg_distance_dict = {}

        for kcid, bcid in cluster_list:
            try:
                net = self.read_net(plz, kcid, bcid)
            except Exception as e:
                self.logger.warning(f" local network {kcid},{bcid} is problematic")
                raise e
            else:
                trafo_sizes = net.trafo["sn_mva"].tolist()[0]

                load_bus = pd.unique(net.load["bus"]).tolist()

                top.create_nxgraph(net, respect_switches=False)
                trafo_distance_to_buses = (
                    top.calc_distance_to_bus(net, net.trafo["lv_bus"].tolist()[0], weight="weight",
                                             respect_switches=False, ).loc[load_bus].tolist())

                # calculate total sim_peak_load
                residential_bus_index = net.bus[~net.bus["zone"].isin(["Commercial", "Public"])].index.tolist()
                commercial_bus_index = net.bus[net.bus["zone"] == "Commercial"].index.tolist()
                public_bus_index = net.bus[net.bus["zone"] == "Public"].index.tolist()

                residential_house_num = net.load[net.load["bus"].isin(residential_bus_index)].shape[0]
                public_house_num = net.load[net.load["bus"].isin(public_bus_index)].shape[0]
                commercial_house_num = net.load[net.load["bus"].isin(commercial_bus_index)].shape[0]

                residential_sum_load = (net.load[net.load["bus"].isin(residential_bus_index)]["max_p_mw"].sum() * 1e3)
                public_sum_load = (net.load[net.load["bus"].isin(public_bus_index)]["max_p_mw"].sum() * 1e3)
                commercial_sum_load = (net.load[net.load["bus"].isin(commercial_bus_index)]["max_p_mw"].sum() * 1e3)

                sim_peak_load = 0
                for building_type, sum_load, house_num in zip(["Residential", "Public", "Commercial"],
                                                              [residential_sum_load, public_sum_load,
                                                               commercial_sum_load],
                                                              [residential_house_num, public_house_num,
                                                               commercial_house_num], ):
                    if house_num:
                        sim_peak_load += utils.oneSimultaneousLoad(installed_power=sum_load, load_count=house_num,
                                                                   sim_factor=SIM_FACTOR[building_type], )

                avg_distance = (sum(trafo_distance_to_buses) / len(trafo_distance_to_buses)) * 1e3
                max_distance = max(trafo_distance_to_buses) * 1e3

                trafo_size = round(trafo_sizes * 1e3)

                if trafo_size in trafo_load_dict:
                    trafo_load_dict[trafo_size].append(sim_peak_load)

                    trafo_max_distance_dict[trafo_size].append(max_distance)

                    trafo_avg_distance_dict[trafo_size].append(avg_distance)

                else:
                    trafo_load_dict[trafo_size] = [sim_peak_load]
                    trafo_max_distance_dict[trafo_size] = [max_distance]
                    trafo_avg_distance_dict[trafo_size] = [avg_distance]

            time += 1
            if time / count >= 0.1:
                percent += 10
                self.logger.info(f"{percent} % processed")
                time = 0
        self.logger.info("analyse_per_trafo_parameters finished.")
        trafo_load_string = json.dumps(trafo_load_dict)
        trafo_max_distance_string = json.dumps(trafo_max_distance_dict)
        trafo_avg_distance_string = json.dumps(trafo_avg_distance_dict)

        update_query = """UPDATE plz_parameters
            SET sim_peak_load_per_trafo = %(l)s, max_distance_per_trafo = %(m)s, avg_distance_per_trafo = %(a)s
            WHERE version_id = %(v)s AND plz = %(p)s;
            """
        self.cur.execute(update_query,
                         {"v": VERSION_ID, "p": plz, "l": trafo_load_string, "m": trafo_max_distance_string,
                          "a": trafo_avg_distance_string, }, )

        self.logger.debug("per trafo analysis finished")
