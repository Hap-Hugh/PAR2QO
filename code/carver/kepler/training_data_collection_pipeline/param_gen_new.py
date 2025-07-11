import bisect
import collections
import dataclasses
import datetime
import logging
import random
from typing import Any, List, Dict, Tuple
import urllib.parse
import collections
from concurrent import futures
import functools
import json
import logging
import os
import time
import math
from itertools import product
import numpy as np

from absl import app
from absl import flags

from kepler.training_data_collection_pipeline import parameter_generator
from kepler.training_data_collection_pipeline import query_utils

from kepler.training_data_collection_pipeline import query_utils

# Define flags
_RANDOM_SEED = flags.DEFINE_integer(
    "random_seed", 2024,
    "random seed setting")
_TEMPLATE_FILE = flags.DEFINE_string("template_file", None, "Parameterized query template file.")
flags.mark_flag_as_required("template_file")
_OUTPUT_DIR = flags.DEFINE_string(
    "output_dir", None,
    "Directory to store parameter values per query.")
flags.mark_flag_as_required("output_dir")


def get_param_and_cardinality(select_column, table, not_null_column, join_condition, left_or_right):
    
    db_config = query_utils.DatabaseConfiguration(
      dbname="imdbloadbase", user="kepler", password="kepler", host="localhost"
    )
    query_manager = query_utils.QueryManager(db_config)

    query = f"""SELECT {select_column}, COUNT(*) as count_all FROM {table}
            WHERE {not_null_column}
            {join_condition}
            GROUP BY {select_column} ORDER BY COUNT(*) DESC"""

    return [(value[0], value[1]) for value in query_manager.execute(query)]


def create_buckets(data, num_buckets):
    # Extract the second values from the pairs
    second_values = [value for _, value in data]
    
    # Find the minimum and maximum of the second values
    min_value = min(second_values)
    max_value = max(second_values)
    
    # Calculate the interval width
    interval_width = (max_value - min_value) / num_buckets
    
    # Initialize the buckets
    buckets = [[] for _ in range(num_buckets)]
    
    # Assign each item to the appropriate bucket
    for item in data:
        _, value = item
        bucket_index = min(int((value - min_value) / interval_width), num_buckets - 1)
        buckets[bucket_index].append(item)
    
    return buckets


def sample_from_buckets(buckets, total_samples):
    num_buckets = len(buckets)
    samples_per_bucket = total_samples // num_buckets
    remainder_samples = total_samples % num_buckets
    
    sampled_data = []
    
    # Sample from each bucket
    for i, bucket in enumerate(buckets):
        num_samples = samples_per_bucket + (1 if i < remainder_samples else 0)
        if len(bucket) > num_samples:
            sampled_data.extend(random.sample(bucket, num_samples))
        else:
            sampled_data.extend(bucket)  # If fewer than num_samples, take all
    
    # Check if we have enough samples; if not, sample randomly from all buckets
    if len(sampled_data) < total_samples:
        remaining_samples = total_samples - len(sampled_data)
        all_items = [item for bucket in buckets for item in bucket]
        additional_samples = random.sample(all_items, remaining_samples)
        sampled_data.extend(additional_samples)
    
    # Ensure the result is exactly total_samples in case of rounding issues
    return sampled_data[:total_samples]


def gen_param_by_cardinality(template_file, N = 10, K = 50, debug=False):
    """
    Input:
    N: number of buckets to split the cardinality range, default is 10
    K: number of total samples
    
    Output:
    A list of param list: [[@param1_1, @param1_2, ..., @param1_N], [@param2_1, ..., @param2_N]]
    """
    with open(template_file) as f:
        all_param_lists = []
        templates = json.load(f)
        for query_id, template in templates.items():
            for predicate in template['predicates']:
                left_or_right = predicate['left_or_right'][0]
                column_pair = [predicate['column'], predicate['join_tables_column'][0]]
                join_condition = predicate['join_conditions'][0]
                from_table = predicate['table'] + " AS " + predicate['alias'] + ", " + predicate['join_tables'][0] + " AS " + predicate['join_tables_alias'][0]
                
                if left_or_right == 'both':
                    select_column = column_pair[0] + ',' + column_pair[1]
                    not_null_column = column_pair[0] + ' IS NOT NULL AND ' + column_pair[1] + ' IS NOT NULL AND '
                elif left_or_right == 'r':
                    select_column = column_pair[1]
                    not_null_column = column_pair[1] + ' IS NOT NULL AND '
                elif left_or_right == 'l':
                    select_column = column_pair[0]
                    not_null_column = column_pair[0] + ' IS NOT NULL AND '
                
                ranking_result = get_param_and_cardinality(select_column, from_table, not_null_column, join_condition, left_or_right)
                if debug: print(ranking_result)
                buckets = create_buckets(ranking_result, N)
                sampled_data = sample_from_buckets(buckets, K)
                for i, bucket in enumerate(buckets):
                    if debug: print(f"Bucket {i + 1}: {bucket}")

                if debug: print("\nSampled data:")
                if debug: print(sampled_data)
                if debug: input()

                temp_params_list = [i[0] for i in sampled_data]
                random.shuffle(temp_params_list)
                all_param_lists.append(temp_params_list)
        return all_param_lists
 

# Function to save PQO query files   
def save_pqo_files(query_id, data, output_dir, mode):
    os.makedirs(output_dir, exist_ok=True)
    output_json = {}

    new_sql_template = data[query_id]["query"]
    literals = data[query_id]["params"]

    for index, combination in enumerate(literals):
        query_str = new_sql_template
        for i, param in enumerate(combination):
            query_str = query_str.replace(f"@param{i}", str(param))
        
        key = f"{query_id}_{mode}_{index}"
        output_json[key] = query_str

    output_file = os.path.join(output_dir, f'{query_id}_{mode}.json')
    with open(output_file, 'w') as json_file:
        json.dump(output_json, json_file, indent=4)

    print(f"Finished writing {output_file}")


# Function to save PQO predicate files
def save_pqo_predicates(query_id, training_data, testing_data, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    def save_combinations(data, file_name):
        output_file = os.path.join(output_dir, file_name)
        with open(output_file, 'w') as file:
            for i, combination in enumerate(data["params"]):
                formatted_items = []
                for j, item in enumerate(combination):
                    predicate = data["predicates"][j]
                    table = predicate["alias"]
                    column = predicate["column"]
                    operator = predicate["operator"]
                    data_type = predicate["data_type"]
                    
                    # Format the param based on its data type
                    if data_type == "text":
                        formatted_param = f"'{item}'"
                    else:
                        formatted_param = item
                    
                    # Format the param if the operator is "in"
                    if operator.lower() == "in":
                        formatted_param = f"({formatted_param})"
                    
                    # Create the formatted string as table.column operator param
                    formatted_items.append(f"{table}.{column} {operator} {formatted_param}")
                
                # Create the combination string
                combination_str = '["' + '", "'.join(formatted_items) + '"]'
                
                # Write to the file
                if i < len(data["params"]) - 1:
                    file.write(combination_str + ',\n')
                else:
                    file.write(combination_str + '\n')

        print(f"Finished writing data to {output_file}")

    # Save training combinations
    save_combinations(training_data[query_id], f'{query_id}_training.txt')

    # Save testing combinations
    save_combinations(testing_data[query_id], f'{query_id}_testing.txt')
    

def escape_single_quotes(param):
    # If param is a string and contains single quotes, replace them with two single quotes
    if isinstance(param, str):
        return param.replace("'", "''")
    return param

# Function to generate both kepler & PQO input files
def gen_full_template_file(output_dir, template_file, param_choice_list, split_ratio, cross_product=True):
    # preparation
    os.makedirs(output_dir, exist_ok=True)
    training_dir = os.path.join(output_dir, 'training')
    original_dir = os.path.join(output_dir, 'original')
    testing_dir = os.path.join(output_dir, 'testing')
    os.makedirs(training_dir, exist_ok=True)
    os.makedirs(original_dir, exist_ok=True)
    os.makedirs(testing_dir, exist_ok=True)

    with open(template_file, 'r') as file:
        templates = json.load(file)

    # for now ONLY one query_id per template file
    query_id, template = next(iter(templates.items()))
    query = template['query']
    predicates = template['predicates']
    
    # TODO: check
    print([len(params) for params in param_choice_list])
    print(param_choice_list)
    param_choice_list = [list(set(params)) for params in param_choice_list]
    print([len(params) for params in param_choice_list])
    print(param_choice_list)
    
    # CROSS PRODUCT or 1-1 to get params
    if cross_product:
        param_list = list(product(*param_choice_list))
    else:
        # 1-1
        param_list = list(zip(*param_choice_list))
    
    # TODO: Escape single quotes
    param_list = [[escape_single_quotes(param) for param in params] for params in param_list]
    param_list = np.array(param_list)
    np.random.shuffle(param_list)
    param_list = param_list.tolist()

    # Calculate the split index
    split_index = math.ceil(len(param_list) * split_ratio)  
    
    # Split the params into training and testing based on the ratio
    training_params = param_list[:split_index]  
    testing_params = param_list[split_index:]   

    # full_template format (contains all params)
    full_template = {
        query_id: {
            "query": query,
            "predicates": predicates,
            "params": param_list
        }
    }

    # training format (contains only training params)
    training_template = {
        query_id: {
            "query": query,
            "predicates": predicates,
            "params": training_params
        }
    }

    # testing format (contains only testing params)
    testing_template = {
        query_id: {
            "query": query,
            "predicates": predicates,
            "params": testing_params
        }
    }

    # Save full template
    full_output_file = os.path.join(original_dir, f"{query_id}_original.json")
    with open(full_output_file, 'w') as out_file:
        json.dump(full_template, out_file, indent=4)
    print(f"Generated {full_output_file} successfully.")

    # Save training template
    training_output_file = os.path.join(training_dir, f"{query_id}_training.json")
    with open(training_output_file, 'w') as out_file:
        json.dump(training_template, out_file, indent=4)
    print(f"Generated {training_output_file} successfully.")

    # Save testing template
    testing_output_file = os.path.join(testing_dir, f"{query_id}_testing.json")
    with open(testing_output_file, 'w') as out_file:
        json.dump(testing_template, out_file, indent=4)
    print(f"Generated {testing_output_file} successfully.")
    
    # PQO directories for predicates and queries
    pqo_query_dir = os.path.join(output_dir, 'PQO', 'query')
    pqo_predicates_dir = os.path.join(output_dir, 'PQO', 'predicates')
    os.makedirs(pqo_query_dir, exist_ok=True)
    os.makedirs(pqo_predicates_dir, exist_ok=True)
    
    # Save PQO files
    save_pqo_files(query_id, training_template, pqo_query_dir, 'training')
    save_pqo_files(query_id, testing_template, pqo_query_dir, 'testing')
    save_pqo_predicates(query_id, training_template, testing_template, pqo_predicates_dir)  
    
def main(unused_argv):
    # template_file = "/home/lsh/PARQO_backend/query/join-order-benchmark/templates/2-0.json"
    # gen_param_by_cardinality(template_file, N = 10, K = 50)
    
    seed_value = _RANDOM_SEED.value
    template_file = _TEMPLATE_FILE.value
    output_dir = _OUTPUT_DIR.value
    
    random.seed(seed_value)
    param_choice_list = gen_param_by_cardinality(template_file, N = 10, K = 50)
    
    # save input files
    base_dir = "cardinality/inputs"
    for cross_product in [True, False]:
        sub_dir = "cross_product" if cross_product else "one_on_one"
        full_output_dir = os.path.join(output_dir, base_dir, sub_dir)
        gen_full_template_file(full_output_dir, template_file, param_choice_list, split_ratio=0.8, cross_product=cross_product)
    


if __name__ == "__main__":
    app.run(main)