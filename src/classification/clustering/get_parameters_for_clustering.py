import warnings

from src.classification.clustering.cluster_settings import print_parameters_for_clustering_for_classification_version

warnings.filterwarnings('ignore')

def get_parameters_for_clustering() -> list[str]:
    return print_parameters_for_clustering_for_classification_version()


def main():
    get_parameters_for_clustering()


if __name__ == '__main__':
    main()