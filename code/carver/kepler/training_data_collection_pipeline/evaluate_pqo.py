import pandas as pd
import psycopg2
import time
import csv
import os
import numpy as np
from absl import app
from absl import flags
import re
import json

_OUTPUT_DIR = flags.DEFINE_string(
    "output_dir", None,
    "Directory to store parameter values per query.")
flags.mark_flag_as_required("output_dir")

_PQO_RESULT_FILE = flags.DEFINE_string(
    "pqo_result_file", None,
    "CSV file regarding PQO result.")
flags.mark_flag_as_required("pqo_result_file")

_QUERY_ID = flags.DEFINE_string("query_id", None, "Name of query to train on")
flags.mark_flag_as_required("query_id")

_TRAINING_SIZE = flags.DEFINE_string("training_size", None, "Training set length")
flags.mark_flag_as_required("training_size")

_DATABASE = flags.DEFINE_string("database", "imdbloadbase", "Name of the database")


# def execute_query(cursor, query):
#     start_time = time.time()
#     cursor.execute(query)
#     return time.time() - start_time

def execute_query(cursor, query):
    """
    Execute the given query and return the execution time, result row count, and cost.
    """
    start_time = time.time()
    cursor.execute(f"EXPLAIN (ANALYZE, BUFFERS, FORMAT JSON) {query}")
    execution_time = time.time() - start_time

    # Parse EXPLAIN JSON to extract row count and cost
    explain_result = cursor.fetchone()[0]
    
    # Ensure explain_result is a JSON string
    if isinstance(explain_result, list):
        # If it's a list, convert it back to a JSON string
        explain_result = json.dumps(explain_result)
        
    explain_data = json.loads(explain_result)

    # Extract row count and cost
    total_cost = explain_data[0]['Plan']['Total Cost']
    row_count = explain_data[0]['Plan']['Plan Rows']

    return execution_time, row_count, total_cost


def test_query_latency(query_id, pqo_data, db_config):
    # connect to db
    conn = psycopg2.connect(
        dbname=db_config['dbname'],
        user=db_config['user'],
        password=db_config['password'],
        host=db_config['host'],
        port=db_config['port']
    )

    cursor = conn.cursor()
    results = []
    counter = 0
    
    # Load hint plan extension
    cursor.execute("LOAD 'pg_hint_plan';")
    cursor.execute('COMMIT')
    
    # Iterate through each row in pqo_data (DataFrame)
    for _, row in pqo_data.iterrows():
        query = row['query']
        plan_content = row['plan']
        frequency = row['freq']
        
        # Initialize latencies storage
        hinted_latencies = []
        default_latencies = []
        hinted_cardinality, hinted_cost, default_cardinality, default_cost = 0, 0, 0, 0
    
        # Run each query five times
        for range_num in range(5):
            # TODO: Apply plan hint if available, otherwise use the default query
            hinted_query = f"{plan_content} {query}" if plan_content != "N/A" else query
            
            hinted_latency, h_cardinality, h_cost = execute_query(cursor, hinted_query)
            cursor.execute('COMMIT')  # commit after each execution to clear any locks
            hinted_latencies.append(hinted_latency)

            default_latency, d_cardinality, d_cost = execute_query(cursor, query)
            cursor.execute('COMMIT')  # commit after each execution to clear any locks
            default_latencies.append(default_latency)
            
            if range_num == 0:
                hinted_cardinality = h_cardinality
                hinted_cost = h_cost
                default_cardinality = d_cardinality
                default_cost = d_cost

        # Calculate median of latencies
        median_hinted_latency = np.median(hinted_latencies)
        median_default_latency = np.median(default_latencies)

        results.append({
            'query': query,
            'frequency': frequency,
            'hinted_latency': median_hinted_latency * 1000,
            'default_latency': median_default_latency * 1000,
            'hinted_row_count': hinted_cardinality,
            'default_row_count': default_cardinality,
            'hinted_cost': hinted_cost,
            'default_cost': default_cost,
            'plan_content': plan_content
        })

        print(f"{query_id} - {counter} - Query: {query}")
        print(f"Hinted Latency (median): {median_hinted_latency * 1000} ms")
        print(f"Default Latency (median): {median_default_latency * 1000} ms")
        print(f"Hinted row count: {hinted_cardinality}")
        print(f"Default row count: {default_cardinality}")
        print(f"Hinted cost: {hinted_cost}")
        print(f"Default cost: {default_cost}")
        print("="*50)
        
        counter += 1

    # close db
    cursor.close()
    conn.close()

    return results


def save_results_to_csv(results, output_file):
    output_dir = os.path.dirname(output_file)
    os.makedirs(output_dir, exist_ok=True)
    
    with open(output_file, mode='w', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=['query', 'frequency', 'hinted_latency', 'default_latency', 'hinted_row_count', 'default_row_count', 'hinted_cost', 'default_cost', 'plan_content'])
        writer.writeheader()
        for result in results:
            writer.writerow(result)
    
    print(f"Results saved to {output_file}")


def calculate_hint_efficiency(results):
    total = sum(result['frequency'] for result in results)
    improved = sum(result['frequency'] for result in results if result['hinted_latency'] < result['default_latency'])
    
    # Total latency calculation
    total_hinted_latency = sum(result['hinted_latency'] * result['frequency'] for result in results)
    total_default_latency = sum(result['default_latency'] * result['frequency'] for result in results)
    avg_hinted_latency = total_hinted_latency / total if total > 0 else 0
    avg_default_latency = total_default_latency / total if total > 0 else 0
    
    # Total cost calculation
    total_hinted_cost = sum(result['hinted_cost'] * result['frequency'] for result in results)
    total_default_cost = sum(result['default_cost'] * result['frequency'] for result in results)
    avg_hinted_cost = total_hinted_cost / total if total > 0 else 0
    avg_default_cost = total_default_cost / total if total > 0 else 0
    
    # Improved latency calculation
    improved_hinted_latency = sum(result['hinted_latency'] * result['frequency'] for result in results if result['hinted_latency'] < result['default_latency'])
    improved_default_latency = sum(result['default_latency'] * result['frequency'] for result in results if result['hinted_latency'] < result['default_latency'])
    avg_improved_hinted_latency = improved_hinted_latency / improved if improved > 0 else 0
    avg_improved_default_latency = improved_default_latency / improved if improved > 0 else 0


    return {
        'improved_count': improved,
        'total': total,
        'avg_improved_hinted_latency': avg_improved_hinted_latency,
        'avg_improved_default_latency': avg_improved_default_latency,
        'avg_hinted_latency': avg_hinted_latency,
        'avg_default_latency': avg_default_latency,
        'avg_hinted_cost': avg_hinted_cost,
        'avg_default_cost': avg_default_cost
    }


def main(unused_arg):
    pqo_result_file = _PQO_RESULT_FILE.value
    output_dir = _OUTPUT_DIR.value
    db = _DATABASE.value    
    query_id = _QUERY_ID.value
    training_size = _TRAINING_SIZE.value
    
    summary_results = []
    
    output_file = os.path.join(output_dir, f'{query_id}_training_{training_size}_latency_comparison.csv')

    # db_config
    db_config = {
        'dbname': db,
        'user': 'kepler',
        'password': 'kepler',
        'host': 'localhost',
        'port': '5431'
    }

    # load data
    pqo_data = pd.read_csv(pqo_result_file)

    # test latency
    results = test_query_latency(query_id, pqo_data, db_config)

    # save result
    save_results_to_csv(results, output_file)

    # calculate efficiency and add to summary
    efficiency = calculate_hint_efficiency(results)
    summary_results.append({
        'query_id': query_id,
        'improved_count': efficiency['improved_count'],
        'total': efficiency['total'],
        'percentage_improved': (efficiency['improved_count'] / efficiency['total']) * 100 if efficiency['total'] > 0 else 0,  # Convert to percentage
        'avg_improved_hinted_latency': efficiency['avg_improved_hinted_latency'],
        'avg_improved_default_latency': efficiency['avg_improved_default_latency'],
        'avg_hinted_latency': efficiency['avg_hinted_latency'],
        'avg_default_latency': efficiency['avg_default_latency'],
        'avg_hinted_cost': efficiency['avg_hinted_cost'],
        'avg_default_cost': efficiency['avg_default_cost']
    })
    
    # Save summary results to CSV
    summary_file = os.path.join(output_dir, f'{query_id}_training_{training_size}_efficiency_summary.csv')

    # Check if the file already exists and has content
    file_exists = os.path.isfile(summary_file) and os.path.getsize(summary_file) > 0
    
    summary_dir = os.path.dirname(summary_file)
    os.makedirs(summary_dir, exist_ok=True)

    with open(summary_file, mode='a', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=[
            'query_id', 'improved_count', 'total', 'percentage_improved', 
            'avg_improved_hinted_latency', 'avg_improved_default_latency',
            'avg_hinted_latency', 'avg_default_latency',
            'avg_hinted_cost', 'avg_default_cost'
        ])

        # Only write the header if the file is new or empty
        if not file_exists:
            writer.writeheader()

        for result in summary_results:
            writer.writerow(result)

    print(f"Summary results saved to {summary_file}")


if __name__ == "__main__":
    app.run(main)
