import psycopg2
import time
import json
import csv
from collections import defaultdict
import os
import numpy as np
from absl import app
from absl import flags
import statistics
import re

_OUTPUT_DIR = flags.DEFINE_string(
    "output_dir", None,
    "Directory to store parameter values per query.")
flags.mark_flag_as_required("output_dir")

_MODEL_RESULT_CSV_FILE = flags.DEFINE_string(
    "model_result_csv_file", None,
    "CSV file regarding kepler model result.")
flags.mark_flag_as_required("model_result_csv_file")

_TESTING_PARAMETER_VALUES = flags.DEFINE_string(
    "testing_parameter_values",
    None,
    "File containing parameter values for testing set."
)
flags.mark_flag_as_required("testing_parameter_values")
_QUERY_ID = flags.DEFINE_string("query_id", None, "Name of query to train on")
flags.mark_flag_as_required("query_id")
_DATABASE = flags.DEFINE_string("database", "imdbloadbase", "Name of the database")


def load_data(json_file_path, csv_file_path, iteration):
    # load testing data
    with open(json_file_path, 'r') as json_file:
        query_data = json.load(json_file)

    # Use defaultdict to store param_plan_map with frequencies
    param_plan_map = defaultdict(lambda: {'confidence': -1, 'plan_id': -1, 'plan_content': '', 'frequency': 0})

    with open(csv_file_path, mode='r') as csv_file:
        reader = csv.DictReader(csv_file)
        for row in reader:
            # print(row['iteration'])
            if int(row['iteration']) == iteration:
                params = row['params']
                confidence = float(row['confidence'])

                # If this param already exists and the new one has a higher confidence
                if confidence > param_plan_map[params]['confidence']:
                    param_plan_map[params] = {
                        'confidence': confidence,
                        'plan_id': row['plan_id'],
                        'plan_content': row['plan_content'],
                        'frequency': param_plan_map[params]['frequency'] + 1,
                    }
                else:
                    # Just increment the frequency if not updating the plan
                    param_plan_map[params]['frequency'] += 1

    # Convert the defaultdict to a list of dicts
    param_plan_map = [{'params': params, **details} for params, details in param_plan_map.items()]
    
    # Extract all confidence values from param_plan_map
    confidence_values = [entry['confidence'] for entry in param_plan_map]
    
    # Calculate average and standard deviation of confidence values
    avg_confidence = statistics.mean(confidence_values) if confidence_values else 0
    std_confidence = statistics.stdev(confidence_values) if len(confidence_values) > 1 else 0
    
    return query_data, param_plan_map, avg_confidence, std_confidence

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


def test_query_latency(query_id, query_data, param_plan_map, db_config):
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
    
    # TODO: Only test first 100 entries with the highest confidence
    dbname = db_config['dbname']
    if dbname == "stack":
        filtered_param_plan_map = sorted(param_plan_map, key=lambda x: x['confidence'], reverse=True)[:100]
    else:
        filtered_param_plan_map = param_plan_map
    
    counter = 0

    # test hinted query
    for entry in filtered_param_plan_map:
        params = entry['params']
        plan_id = int(entry['plan_id'])
        plan_content = entry['plan_content']
        frequency = entry['frequency']
        confidence = entry['confidence']
        
        # Convert params from string representation of a list to an actual list
        params_list = eval(params)
        
        # get full query instance
        query = query_data[query_id]['query']
        for i, param in enumerate(params_list):
            param = str(param).strip()
            pattern = re.compile(rf"@param{i}\b")
            query = pattern.sub(param, query)
            # query_str = query_str.replace(f"@param{i}", str(param))
        # print(query)

        # add hint
        if plan_id != -1: # CHANGE: default plan regression
            hinted_query = f"{plan_content} {query}"
        else:
            hinted_query = query

        hinted_latencies = []
        default_latencies = []
        hinted_cardinality, hinted_cost, default_cardinality, default_cost = 0, 0, 0, 0

        # Run each query 5 times
        cursor.execute("LOAD 'pg_hint_plan';")
        cursor.execute('COMMIT')
        
        for range_num in range(5):
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
            'params': params,
            'frequency': frequency,
            'run': 'median',
            'hinted_latency': median_hinted_latency * 1000,
            'default_latency': median_default_latency * 1000,
            'hinted_row_count': hinted_cardinality,
            'default_row_count': default_cardinality,
            'hinted_cost': hinted_cost,
            'default_cost': default_cost,
            'confidence': confidence,
            'plan_id': plan_id,
            'plan_content': plan_content
        })

        print(f"{query_id} - {counter} - Params: {params}")
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
    with open(output_file, mode='w', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=['params', 'frequency', 'run', 'hinted_latency', 'default_latency', 'hinted_row_count', 'default_row_count', 'hinted_cost', 'default_cost', 'confidence', 'plan_id', 'plan_content'])
        writer.writeheader()
        for result in results:
            writer.writerow(result)
    
    print(f"Results saved to {output_file}")

def calculate_hint_efficiency(results, avg_confidence, std_confidence):
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
    
    # Improved confidence info
    improved_confidences = [result['confidence'] for result in results if result['hinted_latency'] < result['default_latency']]
    avg_improved_confidence = statistics.mean(improved_confidences) if improved_confidences else 0
    std_improved_confidence = statistics.stdev(improved_confidences) if len(improved_confidences) > 1 else 0


    return {
        'improved_count': improved,
        'total': total,
        'avg_improved_hinted_latency': avg_improved_hinted_latency,
        'avg_improved_default_latency': avg_improved_default_latency,
        'avg_improved_confidence': avg_improved_confidence,
        'std_improved_confidence': std_improved_confidence,
        'avg_hinted_latency': avg_hinted_latency,
        'avg_default_latency': avg_default_latency,
        'avg_hinted_cost': avg_hinted_cost,
        'avg_default_cost': avg_default_cost,
        'avg_confidence': avg_confidence,
        'std_confidence': std_confidence
    }

def main(unused_arg):
    query_id = _QUERY_ID.value
    model_result_csv = _MODEL_RESULT_CSV_FILE.value
    testing_json = _TESTING_PARAMETER_VALUES.value
    output_dir = _OUTPUT_DIR.value
    db = _DATABASE.value
    
    summary_results = []

    # Determine iteration based on file name
    if model_result_csv.endswith('_batch_0.csv'):
        iteration = 0
    else:
        iteration = 10
    
    output_file = os.path.join(output_dir, f'{query_id}_latency_comparison.csv')

    # db_config
    db_config = {
        'dbname': db,
        'user': 'kepler',
        'password': 'kepler',
        'host': 'localhost',
        'port': '5431'
    }

    # load data
    query_data, param_plan_map, avg_confidence, std_confidence = load_data(testing_json, model_result_csv, iteration)

    # test latency
    results = test_query_latency(query_id, query_data, param_plan_map, db_config)

    # save result
    save_results_to_csv(results, output_file)

    # calculate efficiency and add to summary
    efficiency = calculate_hint_efficiency(results, avg_confidence, std_confidence)
    summary_results.append({
        'query_id': query_id,
        'improved_count': efficiency['improved_count'],
        'total': efficiency['total'],
        'percentage_improved': (efficiency['improved_count'] / efficiency['total']) * 100,  # convert to percentage
        'avg_improved_hinted_latency': efficiency['avg_improved_hinted_latency'],
        'avg_improved_default_latency': efficiency['avg_improved_default_latency'],
        'avg_improved_confidence': efficiency['avg_improved_confidence'],
        'std_improved_confidence': efficiency['std_improved_confidence'],
        'avg_hinted_latency': efficiency['avg_hinted_latency'],
        'avg_default_latency': efficiency['avg_default_latency'],
        'avg_hinted_cost': efficiency['avg_hinted_cost'],
        'avg_default_cost': efficiency['avg_default_cost'],
        'avg_confidence': efficiency['avg_confidence'],
        'std_confidence': efficiency['std_confidence']
    })
    
    # Save summary results to CSV
    summary_file = os.path.join(output_dir, 'all_hint_efficiency_summary.csv')

    # Check if the file already exists and has content
    file_exists = os.path.isfile(summary_file) and os.path.getsize(summary_file) > 0

    with open(summary_file, mode='a', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=['query_id', 'improved_count', 
                                                  'total', 'percentage_improved', 
                                                  'avg_improved_hinted_latency', 'avg_improved_default_latency', 
                                                  'avg_improved_confidence', 'std_improved_confidence', 
                                                  'avg_hinted_latency', 'avg_default_latency', 
                                                  'avg_hinted_cost', 'avg_default_cost',
                                                  'avg_confidence', 'std_confidence'])

        # Only write the header if the file is new or empty
        if not file_exists:
            writer.writeheader()

        for result in summary_results:
            writer.writerow(result)

    print(f"Summary results saved to {summary_file}")


if __name__ == "__main__":
    app.run(main)
