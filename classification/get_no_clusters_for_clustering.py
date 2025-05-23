import pandas as pd
import warnings

from classification.clustering.cluster_settings import get_best_no_of_clusters_ch_index_for_classification_version

warnings.filterwarnings('ignore')


def get_no_clusters_for_clustering() -> pd.DataFrame:
    return get_best_no_of_clusters_ch_index_for_classification_version()


def main() -> None:
    df_no = get_no_clusters_for_clustering()
    print(df_no)


if __name__ == "__main__":
    main()