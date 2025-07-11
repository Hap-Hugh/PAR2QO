import os
import json
import statistics
import time
import csv
import psycopg2
import matplotlib.pyplot as plt
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed

# PostgreSQL configuration
db_config = {
    'dbname': 'imdbloadbase',
    'user': 'kepler',
    'password': 'kepler',
    'host': 'localhost',
    'port': '5431'
}

# Set the base path and format for query files
# query valid testing
base_path = 'imdb_7-0_sample'
methods = ['cardinality', 'csv', 'kepler']
query_ids = ['7-0']
robustness_choices = ["sliding", "random", "category"]
ranging = [0, 1, 2, 3, 4, 5, 6, 7, 8]
# ranging = [2]


##############
# execute query helper methods
def execute_query(query):
    """
    Connect to PostgreSQL using psycopg2 and execute the given query.
    """
    # print(query)
    try:
        # connect to db
        conn = psycopg2.connect(
            dbname=db_config['dbname'],
            user=db_config['user'],
            password=db_config['password'],
            host=db_config['host'],
            port=db_config['port']
        )
        cursor = conn.cursor()
        
        # Record the start time of query execution
        start_time = time.time()
        cursor.execute(query)
        conn.commit()
        end_time = time.time()

        # Calculate query execution time
        execution_time = end_time - start_time
        
        # Check the number of affected rows or returned rows
        if cursor.rowcount == 0:
            raise ValueError("The query did not affect or return any rows.")
        
        cursor.close()
        conn.close()

        return execution_time
    except Exception as e:
        print(f"Error executing query: {e}")
        return None


def execute_query_median_latency(query, num=5):
    """
    Execute a query 5 times and return the median latency.
    """
    latencies = []
    
    for _ in range(num):
        latency = execute_query(query)
        if latency is not None:
            latencies.append(latency)

    if latencies:
        median_latency = statistics.median(latencies)
        return median_latency
    else:
        return None


def save_to_csv(results, method, run_dir):
    """
    Save query names and latencies to a CSV file for each method.
    """
    # Generate filename that includes a timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = os.path.join(run_dir, f"{method}_queries_results_{timestamp}.csv")

    with open(filename, 'w', newline='') as csvfile:
        fieldnames = ['query_name', 'latency']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        # Write the header
        writer.writeheader()

        # Write each query name and latency to the file
        for query_name, latency in results:
            writer.writerow({'query_name': query_name, 'latency': latency})

    print(f"Results saved to {filename}")
    return filename


def plot_results(csv_file, method, run_dir):
    """
    Plot latency vs. query index from the generated CSV file for each method.
    """
    query_names = []
    latencies = []

    # Read data from CSV file
    with open(csv_file, 'r') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            query_names.append(row['query_name'])
            latencies.append(float(row['latency']))

    # Generate the plot
    plt.figure(figsize=(10, 6))
    plt.scatter(range(len(latencies)), latencies, color='blue', alpha=0.7)
    plt.title(f'Query Latencies for {method} ({csv_file})')
    plt.xlabel('Query Index')
    plt.ylabel('Latency (seconds)')
    plt.grid(True)
    
    # Save plot as image
    plot_filename = os.path.join(run_dir, f"{method}_latencies_plot.png")
    plt.savefig(plot_filename)
    print(f"Plot saved as {plot_filename}")
    plt.close()


def save_errors_to_json(errors, method, run_dir):
    """
    Save the list of errors to a JSON file.
    """
    # Define the JSON file path
    json_filename = os.path.join(run_dir, f"{method}_errors.json")

    try:
        # Write the errors list to a JSON file
        with open(json_filename, 'w') as f:
            json.dump(errors, f, indent=2)
        print(f"Errors saved to {json_filename}")
    except Exception as e:
        print(f"Failed to save errors to JSON: {e}")
        
        
def run_queries_for_method(method, queries, run_dir, save=True):
    """
    Run all queries for a given method, save results in CSV and generate a plot.
    """
    print(f"Running queries for method: {method} {'testing' if save else 'training'}")
    results = []
    errors = []

    # Execute all queries for the method
    for query_name, query_text in queries.items():
        print(f"Executing query {query_name} for method {method}...")
        latency = execute_query_median_latency(query_text, num=1) # TODO: change num for faster execution
        if latency is not None:
            results.append((query_name, latency))  # Save query name and latency
        else:
            errors.append(query_name)

    if save:
        # Save results to CSV file
        csv_filename = save_to_csv(results, method, run_dir)

        # Plot the results from the CSV file
        plot_results(csv_filename, method, run_dir)
        
        # Save errors to JSON file if there are any
        if errors:
            save_errors_to_json(errors, method, run_dir)
        
        


##############
# test query valid
# TODO: training set to 2000
def load_queries_input(robustness, method, query_id, instance=1):
    """
    Load the content of the JSON files from the specified path.
    If testing exists but neither training file is found, return None.
    """
    base_query_path = f"{base_path}/{robustness}/db_instance_{instance}/{method}/inputs/PQO/query"
    testing_file = f"{base_query_path}/{query_id}_testing.json"
    training_file_2000 = f"{base_query_path}/{query_id}_training_2000.json"
    training_file_400 = f"{base_query_path}/{query_id}_training_400.json"
    
    # Check if the testing file exists
    if not os.path.exists(testing_file):
        print(f"Query file {testing_file} does not exist.")
        return None

    # Check if at least one of the training files exists
    if os.path.exists(training_file_2000):
        training_file = training_file_2000
    elif os.path.exists(training_file_400):
        training_file = training_file_400
    else:
        print(f"Neither {training_file_2000} nor {training_file_400} exists.")
        return None

    # Load the contents of both files
    try:
        with open(testing_file, 'r') as f:
            testing_content = json.load(f)
        
        with open(training_file, 'r') as f:
            training_content = json.load(f)
    except json.JSONDecodeError as e:
        print(f"Error reading JSON file: {e}")
        return None

    # Return the contents of both files
    return testing_content, training_content


def input_testing(query_ids):
    """
    Schedule all methods (cardinality, csv, kepler) in parallel.
    """
    
    with ThreadPoolExecutor() as executor:
        print(f"Starting run...")

        futures = []
        # Submit each method to run its queries in parallel
        for method in methods:
            for query_id in query_ids:
                for robustness in robustness_choices:
                    for i in ranging:
                        if robustness == "category" and i in [0, 5, 8]:
                            continue
                        # Create a directory for this run
                        run_dir = os.path.join(base_path, robustness, f"db_instance_{i}", f"testing_time_check")
                        os.makedirs(run_dir, exist_ok=True)
                
                        testing, training = load_queries_input(robustness, method, query_id, i)
                        
                        # Submit the execution of testing queries if available
                        if testing is not None:
                            futures.append(
                                executor.submit(run_queries_for_method, method, testing, run_dir)
                            )

                        # Submit the execution of training queries if available
                        if training is not None:
                            futures.append(
                                executor.submit(run_queries_for_method, method, training, run_dir, save=False)
                            )
                            
                        # Wait for all futures to complete
                        for future in futures:
                            try:
                                future.result()  # Ensure all tasks complete without error
                            except Exception as e:
                                print(f"An error occurred during query execution: {e}")




def main():
    input_testing(query_ids)


if __name__ == "__main__":
    main()
