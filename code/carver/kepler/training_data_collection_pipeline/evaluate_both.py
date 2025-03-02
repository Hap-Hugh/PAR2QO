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
import pandas as pd
import re
import json

_KEPLER_OUTPUT_DIR = flags.DEFINE_string(
    "kepler_output_dir", None,
    "Directory to store parameter values per query.")
flags.mark_flag_as_required("kepler_output_dir")

_PQO_OUTPUT_DIR = flags.DEFINE_string(
    "pqo_output_dir", None,
    "Directory to store parameter values per query.")
flags.mark_flag_as_required("pqo_output_dir")


_KEPLER_MODEL_RESULT_CSV_FILE = flags.DEFINE_string(
    "kepler_model_result_csv_file", None,
    "CSV file regarding kepler model result.")
flags.mark_flag_as_required("kepler_model_result_csv_file")

_PQO_RESULT_FILE = flags.DEFINE_string(
    "pqo_result_file", None,
    "CSV file regarding PQO result.")
flags.mark_flag_as_required("pqo_result_file")


_KEPLER_TESTING_PARAMETER_VALUES = flags.DEFINE_string(
    "kepler_testing_parameter_values",
    None,
    "File containing parameter values for testing set."
)
flags.mark_flag_as_required("kepler_testing_parameter_values")

_PQO_TRAINING_SIZE = flags.DEFINE_string("pqo_training_size", None, "Training set length")
flags.mark_flag_as_required("pqo_training_size")

_QUERY_ID = flags.DEFINE_string("query_id", None, "Name of query to train on")
flags.mark_flag_as_required("query_id")
_DATABASE = flags.DEFINE_string("database", "imdbload", "Name of the database")

# kepler related
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


# Helper methods
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


def test_query_latency(query_id, query_data, param_plan_map, pqo_data, db_config):
    # connect to db
    conn = psycopg2.connect(
        dbname=db_config['dbname'],
        user=db_config['user'], 
        password=db_config['password'],
        host=db_config['host'],
        port=db_config['port']
    )
    
    cursor = conn.cursor()
    kepler_results = []
    pqo_results = []
    
    # Filter kepler plans if needed
    dbname = db_config['dbname']
    if dbname == "stack":
        filtered_param_plan_map = sorted(param_plan_map, key=lambda x: x['confidence'], reverse=True)[:100]
    else:
        filtered_param_plan_map = param_plan_map
        
    counter = 0
    checking_counter = 0
    
    # Load hint plan extension
    cursor.execute("LOAD 'pg_hint_plan';")
    cursor.execute('COMMIT')
    
    # Get base query
    base_query = query_data[query_id]['query']
    
    def normalize_query(query):
        # 1. remove extra space
        query = ' '.join(query.split())
        
        # 2. lower
        query = query.lower()
        
        # 3. remove ;
        query = query.rstrip(';')
        
        return query

    def find_matching_pqo_plan(query, pqo_data):
        normalized_query = normalize_query(query)
        
        for _, row in pqo_data.iterrows():
            if normalize_query(row['query']) == normalized_query:
                return row['plan']
                
        return None
    
    # Process each parameter combination
    for entry in filtered_param_plan_map:
        params = entry['params']
        kepler_plan_id = int(entry['plan_id'])
        kepler_plan = entry['plan_content']
        frequency = entry['frequency']
        confidence = entry['confidence']
        
        # Convert params and build full query
        params_list = eval(params)
        query = base_query
        
        # get full query instance
        for i, param in enumerate(params_list):
            param = str(param).strip()
            pattern = re.compile(rf"@param{i}\b")
            query = pattern.sub(param, query)
            # query_str = query_str.replace(f"@param{i}", str(param))
                
        # Find matching PQO plan from pqo_data
        pqo_plan = find_matching_pqo_plan(query, pqo_data)
                
        # Skip if no matching PQO plan found
        if pqo_plan is None:
            print(f"################ {counter} no pqo_plan found - CHECK")
            checking_counter += 1
            pqo_plan = "N/A"
            
        # Initialize latency storage
        kepler_latencies = []
        pqo_latencies = []
        default_latencies = []
        kepler_cardinality = kepler_cost = 0
        pqo_cardinality = pqo_cost = 0
        default_cardinality = default_cost = 0
        
        # Prepare queries with hints
        kepler_query = f"{kepler_plan} {query}" if kepler_plan_id != -1 else query
        pqo_query = f"{pqo_plan} {query}" if pqo_plan != "N/A" else query
        
        # Run each version 5 times
        for range_num in range(5):
            # Test Kepler plan
            kepler_latency, k_cardinality, k_cost = execute_query(cursor, kepler_query)
            cursor.execute('COMMIT')
            kepler_latencies.append(kepler_latency)
            
            # Test PQO plan
            pqo_latency, p_cardinality, p_cost = execute_query(cursor, pqo_query)
            cursor.execute('COMMIT')
            pqo_latencies.append(pqo_latency)
            
            # Test default
            default_latency, d_cardinality, d_cost = execute_query(cursor, query)
            cursor.execute('COMMIT')
            default_latencies.append(default_latency)
            
            # save cardinality and cost (only once per query)
            if range_num == 0:
                # Store cardinality and cost values
                kepler_cardinality = k_cardinality
                kepler_cost = k_cost
                pqo_cardinality = p_cardinality
                pqo_cost = p_cost
                default_cardinality = d_cardinality
                default_cost = d_cost
        
        # Calculate medians
        median_kepler_latency = np.median(kepler_latencies)
        median_pqo_latency = np.median(pqo_latencies)
        median_default_latency = np.median(default_latencies)
        
        # Store Kepler results
        kepler_results.append({
            'params': params,
            'frequency': frequency,
            'run': 'median',
            'hinted_latency': median_kepler_latency * 1000,
            'default_latency': median_default_latency * 1000,
            'hinted_row_count': kepler_cardinality,
            'default_row_count': default_cardinality,
            'hinted_cost': kepler_cost,
            'default_cost': default_cost,
            'confidence': confidence,
            'plan_id': kepler_plan_id,
            'plan_content': kepler_plan
        })
        
        # Store PQO results
        pqo_results.append({
            'query': query,
            'frequency': frequency,
            'hinted_latency': median_pqo_latency * 1000,
            'default_latency': median_default_latency * 1000,
            'hinted_row_count': pqo_cardinality,
            'default_row_count': default_cardinality,
            'hinted_cost': pqo_cost,
            'default_cost': default_cost,
            'plan_content': pqo_plan
        })
        
        # Print progress
        print(f"{query_id} - {counter} - Query: {query}")
        print(f"Kepler Latency (median): {median_kepler_latency * 1000} ms")
        print(f"PQO Latency (median): {median_pqo_latency * 1000} ms") 
        print(f"Default Latency (median): {median_default_latency * 1000} ms")
        print("="*50)
        
        counter += 1
    
    # close db
    cursor.close()
    conn.close()
    
    print(f"################ checking num - {checking_counter} ################")
    
    return kepler_results, pqo_results


def save_results_to_csv(kepler_results, pqo_results, kepler_output_file, pqo_output_file):
    # save kepler result
    with open(kepler_output_file, mode='w', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=['params', 'frequency', 'run', 'hinted_latency', 'default_latency', 'hinted_row_count', 'default_row_count', 'hinted_cost', 'default_cost', 'confidence', 'plan_id', 'plan_content'])
        writer.writeheader()
        for result in kepler_results:
            writer.writerow(result)
    print(f"Results saved to {kepler_output_file}")
    
    # save pqo result
    output_dir = os.path.dirname(pqo_output_file)
    os.makedirs(output_dir, exist_ok=True)
    with open(pqo_output_file, mode='w', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=['query', 'frequency', 'hinted_latency', 'default_latency', 'hinted_row_count', 'default_row_count', 'hinted_cost', 'default_cost', 'plan_content'])
        writer.writeheader()
        for result in pqo_results:
            writer.writerow(result)
    print(f"Results saved to {pqo_output_file}")


def calculate_hint_efficiency(results, avg_confidence=None, std_confidence=None):
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

    # Initialize return dictionary with basic metrics
    efficiency_dict = {
        'improved_count': improved,
        'total': total,
        'avg_improved_hinted_latency': avg_improved_hinted_latency,
        'avg_improved_default_latency': avg_improved_default_latency,
        'avg_hinted_latency': avg_hinted_latency,
        'avg_default_latency': avg_default_latency,
        'avg_hinted_cost': avg_hinted_cost,
        'avg_default_cost': avg_default_cost
    }
    
    # Add confidence metrics if provided
    if avg_confidence is not None:
        # Improved confidence info
        improved_confidences = [result['confidence'] for result in results if result['hinted_latency'] < result['default_latency']]
        avg_improved_confidence = statistics.mean(improved_confidences) if improved_confidences else 0
        std_improved_confidence = statistics.stdev(improved_confidences) if len(improved_confidences) > 1 else 0
        
        # Add confidence metrics to the dictionary
        efficiency_dict.update({
            'avg_improved_confidence': avg_improved_confidence,
            'std_improved_confidence': std_improved_confidence,
            'avg_confidence': avg_confidence,
            'std_confidence': std_confidence
        })
    
    return efficiency_dict


def main(unused_arg):
    # initialize
    kepler_output_dir = _KEPLER_OUTPUT_DIR.value
    pqo_output_dir = _PQO_OUTPUT_DIR.value
    
    kepler_model_result_csv = _KEPLER_MODEL_RESULT_CSV_FILE.value
    pqo_result_file = _PQO_RESULT_FILE.value
    
    kepler_testing_json = _KEPLER_TESTING_PARAMETER_VALUES.value
    pqo_training_size = _PQO_TRAINING_SIZE.value
    
    query_id = _QUERY_ID.value
    db = _DATABASE.value
    
    kepler_summary_results = []
    pqo_summary_results = []
    
    # db_config
    db_config = {
        'dbname': db,
        'user': 'kepler',
        'password': 'kepler',
        'host': 'localhost',
        'port': '5432'
    }

    ## Kepler
    # Determine iteration based on file name
    if kepler_model_result_csv.endswith('_batch_0.csv'):
        iteration = 0
    else:
        iteration = 10
    
    kepler_output_file = os.path.join(kepler_output_dir, f'{query_id}_latency_comparison.csv')

    # load data
    query_data, param_plan_map, avg_confidence, std_confidence = load_data(kepler_testing_json, kepler_model_result_csv, iteration)

    ## PQO
    # load data
    pqo_data = pd.read_csv(pqo_result_file)
    pqo_output_file = os.path.join(pqo_output_dir, f'{query_id}_training_{pqo_training_size}_latency_comparison.csv')


    ## Both
    # test latency
    kepler_results, pqo_results = test_query_latency(query_id, query_data, param_plan_map, pqo_data, db_config)

    # save result
    save_results_to_csv(kepler_results, pqo_results, kepler_output_file, pqo_output_file)

    # calculate efficiency and add to summary
    kepler_efficiency = calculate_hint_efficiency(kepler_results, avg_confidence, std_confidence)
    pqo_efficiency = calculate_hint_efficiency(pqo_results)
    
    kepler_summary_results.append({
        'query_id': query_id,
        'improved_count': kepler_efficiency['improved_count'],
        'total': kepler_efficiency['total'],
        'percentage_improved': (kepler_efficiency['improved_count'] / kepler_efficiency['total']) * 100,  # convert to percentage
        'avg_improved_hinted_latency': kepler_efficiency['avg_improved_hinted_latency'],
        'avg_improved_default_latency': kepler_efficiency['avg_improved_default_latency'],
        'avg_improved_confidence': kepler_efficiency['avg_improved_confidence'],
        'std_improved_confidence': kepler_efficiency['std_improved_confidence'],
        'avg_hinted_latency': kepler_efficiency['avg_hinted_latency'],
        'avg_default_latency': kepler_efficiency['avg_default_latency'],
        'avg_hinted_cost': kepler_efficiency['avg_hinted_cost'],
        'avg_default_cost': kepler_efficiency['avg_default_cost'],
        'avg_confidence': kepler_efficiency['avg_confidence'],
        'std_confidence': kepler_efficiency['std_confidence']
    })
    pqo_summary_results.append({
        'query_id': query_id,
        'improved_count': pqo_efficiency['improved_count'],
        'total': pqo_efficiency['total'],
        'percentage_improved': (pqo_efficiency['improved_count'] / pqo_efficiency['total']) * 100 if pqo_efficiency['total'] > 0 else 0,  # Convert to percentage
        'avg_improved_hinted_latency': pqo_efficiency['avg_improved_hinted_latency'],
        'avg_improved_default_latency': pqo_efficiency['avg_improved_default_latency'],
        'avg_hinted_latency': pqo_efficiency['avg_hinted_latency'],
        'avg_default_latency': pqo_efficiency['avg_default_latency'],
        'avg_hinted_cost': pqo_efficiency['avg_hinted_cost'],
        'avg_default_cost': pqo_efficiency['avg_default_cost']
    })
    
    # Save summary results to CSV
    kepler_summary_file = os.path.join(kepler_output_dir, 'all_hint_efficiency_summary.csv')
    pqo_summary_file = os.path.join(pqo_output_dir, f'{query_id}_training_{pqo_training_size}_efficiency_summary.csv')

    # Check if the file already exists and has content
    kepler_file_exists = os.path.isfile(kepler_summary_file) and os.path.getsize(kepler_summary_file) > 0
    pqo_file_exists = os.path.isfile(pqo_summary_file) and os.path.getsize(pqo_summary_file) > 0

    with open(kepler_summary_file, mode='a', newline='') as file:
        kepler_writer = csv.DictWriter(file, fieldnames=['query_id', 'improved_count', 
                                                  'total', 'percentage_improved', 
                                                  'avg_improved_hinted_latency', 'avg_improved_default_latency', 
                                                  'avg_improved_confidence', 'std_improved_confidence', 
                                                  'avg_hinted_latency', 'avg_default_latency', 
                                                  'avg_hinted_cost', 'avg_default_cost',
                                                  'avg_confidence', 'std_confidence'])

        # Only write the header if the file is new or empty
        if not kepler_file_exists:
            kepler_writer.writeheader()

        for result in kepler_summary_results:
            kepler_writer.writerow(result)
    print(f"Summary results saved to {kepler_summary_file}")
    
    with open(pqo_summary_file, mode='a', newline='') as file:
        pqo_writer = csv.DictWriter(file, fieldnames=[
            'query_id', 'improved_count', 'total', 'percentage_improved', 
            'avg_improved_hinted_latency', 'avg_improved_default_latency',
            'avg_hinted_latency', 'avg_default_latency',
            'avg_hinted_cost', 'avg_default_cost'
        ])

        # Only write the header if the file is new or empty
        if not pqo_file_exists:
            pqo_writer.writeheader()

        for result in pqo_summary_results:
            pqo_writer.writerow(result)
    print(f"Summary results saved to {pqo_summary_file}")


if __name__ == "__main__":
    app.run(main)
