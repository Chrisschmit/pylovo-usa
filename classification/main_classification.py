import subprocess
import yaml
import os
import re
import sys

# Define paths to YAML config files
CONFIG_CLASSIFICATION_PATH = os.path.join(os.path.dirname(__file__), "config_classification.yaml")
CONFIG_CLUSTERING_PATH = os.path.join(os.path.dirname(__file__), "clustering/config_clustering.yaml")


def load_yaml(filepath):
    """Load a YAML file."""
    with open(filepath, "r", encoding="utf-8") as file:
        return yaml.safe_load(file)


def save_yaml(filepath, data):
    """Save a YAML file."""
    with open(filepath, "w", encoding="utf-8") as file:
        yaml.dump(data, file, default_flow_style=False, sort_keys=False)


def run_script(script_name):
    """Runs a Python script inside the classification module, streams output"""
    process = subprocess.Popen(
        ["python", "-m", f"classification.{script_name}"],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True
    )

    output_lines = []

    for line in process.stdout:
        print(line, end="")  # Always print logs live

        # Only store output if it's not prepare_data_for_clustering
        if script_name != "prepare_data_for_clustering":
            output_lines.append(line)

    process.wait()

    if process.returncode != 0:
        print(f"\nError while running {script_name}. Exit code: {process.returncode}")
        sys.exit(1)

    return "".join(output_lines) if script_name != "prepare_data_for_clustering" else None


def get_user_confirmation():
    """Asks the user if they want to manually assign clustering parameters."""
    while True:
        user_input = input("\nDo you prefer to assign clustering parameters and number of clusters? "
                           "Otherwise, optimal values will be calculated and assigned accordingly. (yes/no): ").strip().lower()
        if user_input in ["yes", "no"]:
            return user_input == "yes"
        print("Invalid input. Please enter 'yes' or 'no'.")


def get_custom_clustering_parameters():
    """Gets custom clustering parameters from user input."""
    while True:
        user_input = input("\nPlease enter desired clustering parameters (comma-separated). "
                           "For example: no_branches,max_no_of_households_of_a_branch,avg_trafo_dis\n> ").strip()
        params = [param.strip() for param in user_input.split(",") if param.strip()]
        if params:
            return params
        print("Invalid input. Please enter at least one parameter.")


def get_custom_cluster_numbers():
    """Gets custom cluster numbers for KMedoids, KMeans, and GMM from user input while ensuring valid ranges."""
    
    # Load allowed cluster values from config_classification.yaml
    config_classification = load_yaml(CONFIG_CLASSIFICATION_PATH)
    allowed_values = config_classification.get("NO_OF_CLUSTERS_ALLOWED", [])

    if not allowed_values:
        print("Warning: NO_OF_CLUSTERS_ALLOWED is missing or empty in config_classification.yaml. Defaulting to [3, 4, 5, 6, 7].")
        allowed_values = [3, 4, 5, 6, 7]  # Fallback in case the value is missing

    # Convert to a set for quick lookup
    allowed_values_set = set(allowed_values)
    min_val, max_val = min(allowed_values), max(allowed_values)

    while True:
        user_input = input(f"\nPlease enter desired number of clusters for KMedoid, KMeans, and GMM "
                           f"(comma-separated, allowed: {allowed_values}). "
                           f"For example: 4,5,4\n> ").strip()
        try:
            values = [int(num.strip()) for num in user_input.split(",") if num.strip()]
            
            # Check if exactly 3 values are provided
            if len(values) != 3:
                print(f"Invalid input. Please enter exactly 3 integer values separated by commas.")
                continue

            # Check if all values are within the allowed range
            if all(val in allowed_values_set for val in values):
                return values

            print(f"Invalid input. All values must be within the allowed range {min_val}-{max_val}. Try again.")

        except ValueError:
            print("Invalid input. Please enter numeric values only.")




def update_list_of_clustering_parameters():
    """Runs get_parameters_for_clustering and updates LIST_OF_CLUSTERING_PARAMETERS in config_clustering.yaml."""
    output = run_script("get_parameters_for_clustering")
    
    # Extract clustering parameters from the output 
    params = [line.strip() for line in output.split("\n") if line.strip() and "Database connection" not in line]
    
    # Update YAML file
    config = load_yaml(CONFIG_CLUSTERING_PATH)
    config["LIST_OF_CLUSTERING_PARAMETERS"] = params
    save_yaml(CONFIG_CLUSTERING_PATH, config)
    print("LIST_OF_CLUSTERING_PARAMETERS updated in config_clustering.yaml")


def update_number_of_clusters():
    """Runs get_no_clusters_for_clustering and updates cluster numbers in config_clustering.yaml."""
    output = run_script("get_no_clusters_for_clustering")

    # Define the direct mapping from algorithm names to YAML keys
    cluster_mappings = {
        "kmeans": "N_CLUSTERS_KMEANS",
        "KMedoids": "N_CLUSTERS_KMEDOID",
        "GMM tied": "N_CLUSTERS_GMM"
    }

    cluster_counts = {}

    for line in output.split("\n"):
        # Match table lines with the format: algorithm, no_clusters, ch_index
        match = re.match(r"^\s*\d+\s+([A-Za-z\s]+)\s+(\d+)\s+[\d.]+", line)
        if match:
            algorithm, num_clusters = match.groups()
            algorithm = algorithm.strip()  

            # Only update parameters that are in our mapping
            if algorithm in cluster_mappings:
                yaml_key = cluster_mappings[algorithm]
                cluster_counts[yaml_key] = int(num_clusters)


    # Load existing YAML configuration
    config = load_yaml(CONFIG_CLUSTERING_PATH)

    # Update YAML configuration with extracted values
    config.update(cluster_counts)

    # Save updated YAML file
    save_yaml(CONFIG_CLUSTERING_PATH, config)
    print("Number of clusters updated in config_clustering.yaml")


def main():
    """Main function to execute the classification pipeline."""
    print("Running classification pipeline...")

    # Step 1: Ensure user has configured `config_classification.yaml`
    config_classification = load_yaml(CONFIG_CLASSIFICATION_PATH)
    print(f"Using classification version: {config_classification['CLASSIFICATION_VERSION']}")
    
    # Step 2: Run prepare_data_for_clustering.py
    print("\nRunning prepare_data_for_clustering.py...")
    run_script("prepare_data_for_clustering")

    # Step 3: Ask user for manual input or automatic assignment
    if get_user_confirmation():
        # User wants to enter clustering parameters manually
        clustering_parameters = get_custom_clustering_parameters()
        cluster_numbers = get_custom_cluster_numbers()

        # Update YAML file manually
        config = load_yaml(CONFIG_CLUSTERING_PATH)
        config["LIST_OF_CLUSTERING_PARAMETERS"] = clustering_parameters
        config["N_CLUSTERS_KMEDOID"], config["N_CLUSTERS_KMEANS"], config["N_CLUSTERS_GMM"] = cluster_numbers
        save_yaml(CONFIG_CLUSTERING_PATH, config)

        print("\nManually assigned clustering parameters and number of clusters updated in config_clustering.yaml")
    else:
        # Step 4: Automatically update clustering parameters and cluster numbers
        print("\nGetting parameters for clustering...")
        update_list_of_clustering_parameters()

        print("\nGetting number of clusters for clustering...")
        update_number_of_clusters()

    # Step 5: Run apply_clustering_for_QGIS_visualisation.py
    print("\nRunning apply_clustering_for_QGIS_visualisation.py...")
    run_script("apply_clustering_for_QGIS_visualisation")
    

    print("\nClassification process completed successfully!")


if __name__ == "__main__":
    main()
