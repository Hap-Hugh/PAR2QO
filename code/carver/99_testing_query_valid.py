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
base_path = '.'
# methods = ['cardinality', 'csv', 'kepler']
methods = ['cardinality_full']
query_ids = ['29-0']

# # average time testing
# base_path = '/home/lsh/test_kepler/kepler/imdb_sampling_test'
# query_ids = ['7-1']


##############
# execute query helper methods
def execute_query(query_id, method, train, query, param = ""):
    """
    Connect to PostgreSQL using psycopg2 and execute the given query.
    Returns tuple of (execution_time, row_count)
    """
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

        def modify_query(query) -> str:
            """
            Modify the query by splitting on FROM and replacing the SELECT clause with SELECT *
            """
            if 'FROM' not in query:
                return query
                
            parts = query.split('FROM', 1)
            modified_query = 'SELECT * FROM' + parts[1]
            # modified_query = modified_query.rstrip(';') + ' LIMIT 5;' # TODO - just for '15-0' for oom error
            return modified_query

        query = modify_query(query)

        if not query.startswith('SELECT * FROM'):
            raise ValueError("The query not contains wildcard")

        # Record the start time of query execution
        start_time = time.time()
        cursor.execute(query)
        conn.commit()
        end_time = time.time()

        # Get the actual number of rows
        rows = cursor.fetchall()
        row_count = len(rows)

        # Calculate query execution time
        execution_time = end_time - start_time

        # Log to CSV
        log_query_results(param, query, row_count, execution_time, query_id, method, train)

        cursor.close()
        conn.close()

        return execution_time, row_count

    except Exception as e:
        print(f"Error executing query: {e}")
        # Log failed queries too
        log_query_results(param, query, 0, None, str(e), query_id, method, train)
        return None, 0

def log_query_results(param, query, row_count, execution_time, query_id, method, train="train", error = ""):
    """
    Log query results to a CSV file
    """
    # Create directory path
    dir_path = f"0_testing_time_check/{query_id}"
    file_path = f"{dir_path}/{method}_{train}_result.csv"
    
    # Create directories if they don't exist
    os.makedirs(dir_path, exist_ok=True)
    
    # Check if file exists
    file_exists = os.path.exists(file_path)
    
    with open(file_path, 'a', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        if not file_exists:
            # Write header if file is new
            writer.writerow(['Row Count', 'Execution Time (s)', 'Error', 'Query'])
        
        # Write the results
        writer.writerow([
            row_count,
            f"{execution_time * 1000:.4f}" if execution_time is not None else "",
            error,
            query.replace('\n', ' '),  # Remove newlines for better CSV formatting
        ])

def execute_query_median_latency(query, num=5, query_id='1-0', method='cardinality', train="train"):
    """
    Execute a query 5 times and return the median latency.
    """
    latencies = []
    
    for _ in range(num):
        latency = execute_query(query_id, method, train, query)
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

    with open(filename, 'w', newline='', encoding='utf-8') as csvfile:
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
    with open(csv_file, 'r', encoding='utf-8') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            query_names.append(row['query_name'])
            latencies.append(row['latency'])

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
        with open(json_filename, 'w', encoding='utf-8') as f:
            json.dump(errors, f, indent=2)
        print(f"Errors saved to {json_filename}")
    except Exception as e:
        print(f"Failed to save errors to JSON: {e}")
        
        
def run_queries_for_method(query_id, method, train, queries, run_dir, save=True):
    """
    Run all queries for a given method, save results in CSV and generate a plot.
    """
    print(f"Running queries for method: {method} {'testing' if save else 'training'}")
    results = []
    errors = []

    # Execute all queries for the method
    for query_name, query_text in queries.items():
        print(f"Executing query {query_name} for method {method}...")
        latency = execute_query_median_latency(query_text, num=1, query_id=query_id, method=method, train=train) # TODO: change num for faster execution
        if latency is not None:
            results.append((query_name, latency))  # Save query name and latency
        else:
            errors.append(query_name)

    if save:
        # Save results to CSV file
        csv_filename = save_to_csv(results, method, run_dir)

        # Plot the results from the CSV file
        # plot_results(csv_filename, method, run_dir)
        
        # Save errors to JSON file if there are any
        if errors:
            save_errors_to_json(errors, method, run_dir)
        
        


##############
# test query valid
# TODO: training set to 2000
def load_queries_input(method, query_id):
    """
    Load the content of the JSON files from the specified path.
    If testing exists but neither training file is found, return None.
    """
    base_query_path = f"{base_path}/imdb_{query_id}_original/{method}/inputs/PQO/query"
    testing_file = f"{base_query_path}/{query_id}_testing.json"
    training_file_2000 = f"{base_query_path}/{query_id}_training_2000.json"
    training_file_400 = f"{base_query_path}/{query_id}_training_400.json"
    
    # Check if the testing file exists
    if not os.path.exists(testing_file):
        print(f"Query file {testing_file} does not exist.")
        return None

    # Check if at least one of the training files exists
    # if os.path.exists(training_file_2000):
    #     training_file = training_file_2000
    if os.path.exists(training_file_400):
        training_file = training_file_400
    else:
        print(f"Neither {training_file_2000} nor {training_file_400} exists.")
        return None

    # Load the contents of both files
    try:
        with open(testing_file, 'r', encoding='utf-8') as f:
            testing_content = json.load(f)
        
        with open(training_file, 'r', encoding='utf-8') as f:
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
        for query_id in query_ids:
            for method in methods:
                # Create a directory for this run
                run_dir = os.path.join(base_path, f"0_testing_time_check", query_id)
                os.makedirs(run_dir, exist_ok=True)
        
                testing, training = load_queries_input(method, query_id)
                
                # Submit the execution of testing queries if available
                if testing is not None:
                    futures.append(
                        executor.submit(run_queries_for_method, query_id, method, "test", testing, run_dir)
                    )

                # Submit the execution of training queries if available
                if training is not None:
                    futures.append(
                        executor.submit(run_queries_for_method, query_id, method, "train", training, run_dir, save=False)
                    )
                    
                # Wait for all futures to complete
                for future in futures:
                    try:
                        future.result()  # Ensure all tasks complete without error
                    except Exception as e:
                        print(f"An error occurred during query execution: {e}")


##############
# test time intervals
def load_queries(method, query_id):
    """
    Load the content of the JSON file from the specified path.
    """
    query_file = f"{base_path}/{method}/inputs/PQO/query/{query_id}_testing.json"
    
    if not os.path.exists(query_file):
        print(f"Query file {query_file} does not exist.")
        return None
    
    with open(query_file, 'r', encoding='utf-8') as f:
        return json.load(f)


# def schedule_queries_parallel(query_ids, max_runtime_hours=24):
#     """
#     Schedule all methods (cardinality, csv, kepler) in parallel.
#     Every 30 minutes, a new batch of queries is started, regardless of whether the previous batch is complete.
#     """
#     total_runs = int(max_runtime_hours * 2)  # 48 runs (24 hours, 30 minutes per run)
    
#     with ThreadPoolExecutor() as executor:
#         for run_num in range(total_runs):
#             print(f"Starting run {run_num + 1}/{total_runs}...")
            
#             # Create a directory for this run
#             run_dir = os.path.join(base_path, f"testing_time_check/run_{run_num + 1}")
#             os.makedirs(run_dir, exist_ok=True)

#             futures = []
#             # Submit each method to run its queries in parallel
#             for method in methods:
#                 for query_id in query_ids:
#                     queries = load_queries(method, query_id)
#                     if queries is None:
#                         continue

#                     # Submit the execution of queries to the thread pool
#                     futures.append(executor.submit(run_queries_for_method, method, queries, run_dir))

#             # We don't wait for futures to complete, we immediately schedule the next run after 30 minutes
#             print(f"Run {run_num + 1} scheduled. ")
#             time.sleep(1800)  # Wait for 30 minutes before scheduling the next batch

#             # Note: The previous batch will continue running in parallel if not finished


# def schedule_queries_single(query_ids, max_runtime_hours=24):
#     """
#     Schedule all methods (cardinality, csv, kepler) sequentially.
#     A batch of queries is started every 30 minutes, but the next batch will not start until the current one finishes.
#     """
#     total_runs = int(max_runtime_hours)
    
#     with ThreadPoolExecutor() as executor:
#         for run_num in range(total_runs):
#             print(f"Starting run {run_num + 1}/{total_runs}...")

#             # Create a directory for this run
#             run_dir = os.path.join(base_path, f"testing_time_check/run_{run_num + 1}")
#             os.makedirs(run_dir, exist_ok=True)

#             futures = []
#             # Submit each method to run its queries one by one
#             query_file_path = f"imdb_sampling_test/7-1_testing_base_card_between_kepler.json"
#             if not os.path.exists(query_file_path):
#                 print(f"Query file {query_file_path} does not exist.")
#                 return None
#             with open(query_file_path, 'r', encoding='utf-8') as f:
#                 queries =  json.load(f)
#             if queries is None:
#                 continue
            
#             method = "default"

#             # Submit the execution of queries to the thread pool
#             futures.append(executor.submit(run_queries_for_method, method, queries, run_dir))

#             # Wait for all futures to complete before scheduling the next run
#             for future in futures:
#                 future.result()  # This will block until the query is complete

#             print(f"Run {run_num + 1} completed.")


# def schedule_queries_parallel_test(query_ids, max_runtime_hours=24):
#     """
#     Schedule all methods (cardinality, csv, kepler) in parallel.
#     Every 30 minutes, a new batch of queries is started, regardless of whether the previous batch is complete.
#     """
#     total_runs = int(max_runtime_hours * 1)  # 48 runs (24 hours, 30 minutes per run)
    
#     with ThreadPoolExecutor() as executor:
#         for run_num in range(total_runs):
#             print(f"Starting run {run_num + 1}/{total_runs}...")
            
#             # Create a directory for this run
#             run_dir = os.path.join(base_path, f"testing_time_check/run_{run_num + 1}")
#             os.makedirs(run_dir, exist_ok=True)

#             futures = []
#             # Submit each method to run its queries in parallel
#             for method in methods:
#                 for query_id in query_ids:
#                     query_file = f"imdb_sampling_test/{query_id}_{method}_testing.json"
#                     if not os.path.exists(query_file):
#                         print(f"Query file {query_file} does not exist.")
#                         return None
                    
#                     with open(query_file, 'r', encoding='utf-8') as f:
#                         queries = json.load(f)

#                     if queries is None:
#                         continue

#                     # Submit the execution of queries to the thread pool
#                     futures.append(executor.submit(run_queries_for_method, method, queries, run_dir))

#             # We don't wait for futures to complete, we immediately schedule the next run after 30 minutes
#             print(f"Run {run_num + 1} scheduled. ")
#             time.sleep(300)  # Wait for 30 minutes before scheduling the next batch

#             # Note: The previous batch will continue running in parallel if not finished



def main():
    # Run queries for all methods in parallel for 24 hours (48 runs in total)
    # schedule_queries_parallel(query_ids, max_runtime_hours=1)
    # schedule_queries_single(query_ids, max_runtime_hours=1)
    # schedule_queries_parallel_test(query_ids, max_runtime_hours=1)
    input_testing(query_ids)


if __name__ == "__main__":
    main()
