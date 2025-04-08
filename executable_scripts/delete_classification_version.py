from pylovo.GridGenerator import GridGenerator

# select plz and version you want to delete the networks for
classification_version = "1"

# delete networks
gg = GridGenerator() # initialization of the class
gg.pgr.delete_classification_version_from_related_tables(classification_version)
