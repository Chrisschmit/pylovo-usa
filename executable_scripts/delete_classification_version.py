from pylovo.GridGenerator import GridGenerator

# select plz and version you want to delete the networks for
classification_version = "9"

# delete networks
gg = GridGenerator(plz="91301") # just a dummy plz for the initialization of the class
gg.pgr.delete_classification_version_from_related_tables(classification_version)
