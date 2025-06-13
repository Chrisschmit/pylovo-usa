from src.classification.parameter_calculation.grid_parameters import GridParameters
import src.database_client as dbc
from src.config_loader import VERSION_ID


class ParameterCalculator:
    def __init__(self, plz):
        self.plz = plz
        # connect to database
        self.dbc = dbc.DatabaseClient()
        

    def calc_parameters_for_grids(self) -> None:
        """
        this function extracts and calculates parameters for each net in a plz
        """
        # check if for this version ID and PLZ the parameters have already been calculated
        parameter_count = self.dbc.count_clustering_parameters(plz=self.plz)
        if parameter_count > 0:
            print(f"The parameters for the grids of postcode area {self.plz} and version {VERSION_ID} "
                  f"have already been calculated.")
            return
        # all nets within plz
        cluster_list = self.dbc.get_list_from_plz(self.plz)
        # loop over all networks
        for kcid, bcid in cluster_list:
            gp = GridParameters(self.plz, bcid, kcid, self.dbc)
            print(bcid, kcid)
            gp.calc_plz_parameters()
