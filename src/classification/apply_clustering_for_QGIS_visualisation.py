# 7. cluster and write to transformer_classified
# --> clustered grids and representative grids
from src.classification.database_communication.database_communication import DatabaseCommunication


def apply_clustering_for_visualisation() -> None:
    dc = DatabaseCommunication()
    df_parameters_of_grids = dc.municipal_register_with_clustering_parameters_for_classification_version()

    dc.save_transformers_with_classification_info()


def main():
    apply_clustering_for_visualisation()


if __name__ == "__main__":
    main()
