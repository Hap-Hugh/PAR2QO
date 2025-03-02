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


def execute_query(cursor, query):
    """
    Execute the given query and return the execution time, result row count, and cost.
    """
    start_time = time.time()
    cursor.execute(f"EXPLAIN (FORMAT JSON) {query}")
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

    return total_cost


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

        hinted_cost, default_cost = 0, 0

        # Run each query 5 times
        cursor.execute("LOAD 'pg_hint_plan';")
        cursor.execute('COMMIT')
        
        for range_num in range(1):
            h_cost = execute_query(cursor, hinted_query)
            cursor.execute('COMMIT')  # commit after each execution to clear any locks

            d_cost = execute_query(cursor, query)
            cursor.execute('COMMIT')  # commit after each execution to clear any locks
            
            if range_num == 0:
                hinted_cost = h_cost
                default_cost = d_cost

        results.append({
            'params': params,
            'frequency': frequency,
            'run': 'median',
            'hinted_cost': hinted_cost,
            'default_cost': default_cost,
            'confidence': confidence,
            'plan_id': plan_id,
            'plan_content': plan_content
        })

        print(f"{query_id} - {counter} - Params: {params}")
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
        writer = csv.DictWriter(file, fieldnames=['params', 'frequency', 'run', 'hinted_cost', 'default_cost', 'confidence', 'plan_id', 'plan_content'])
        writer.writeheader()
        for result in results:
            writer.writerow(result)
    
    print(f"Results saved to {output_file}")


def main(unused_arg):
    query_id = _QUERY_ID.value
    model_result_csv = _MODEL_RESULT_CSV_FILE.value
    testing_json = _TESTING_PARAMETER_VALUES.value
    output_dir = _OUTPUT_DIR.value
    db = _DATABASE.value

    # Determine iteration based on file name
    if model_result_csv.endswith('_batch_0.csv'):
        iteration = 0
    else:
        iteration = 10
    
    output_file = os.path.join(output_dir, f'{query_id}_cost_comparison.csv')

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


if __name__ == "__main__":
    app.run(main)
