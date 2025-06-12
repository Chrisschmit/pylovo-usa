from src.classification.database_communication.DatabaseCommunication import DatabaseCommunication


def apply_filter_to_grids(additional_filtering: bool = False) -> None:
    """apply thresholds set in config_clustering to clustering parameters.
    according to those thresholds the value in column 'filtered' of 'clustering_parameters'
    is set true or false
    """
    dc = DatabaseCommunication()
    dc.apply_max_trafo_dis_threshold()
    dc.apply_households_per_building_threshold()
    if additional_filtering:
        dc.apply_list_of_clustering_parameters_thresholds()
    dc.set_remaining_filter_values_false()
