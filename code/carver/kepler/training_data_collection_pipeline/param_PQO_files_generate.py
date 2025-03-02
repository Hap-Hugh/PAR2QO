import collections
import json
import os

from absl import app
from absl import flags

import re


_OUTPUT_DIR = flags.DEFINE_string(
    "output_dir", None,
    "Directory to store parameter values per query.")
flags.mark_flag_as_required("output_dir")

_INPUT_DIR = flags.DEFINE_string(
    "input_dir", None,
    "Directory of the input data")
flags.mark_flag_as_required("input_dir")

_TRAIN_SIZE = flags.DEFINE_string("train_size", None, "Training size value")
flags.mark_flag_as_required("train_size")

_QUERY_ID = flags.DEFINE_string("query_id", None, "Name of query to train on")
flags.mark_flag_as_required("query_id")


###################
# PQO input related
def save_pqo_files(query_id, data, output_dir, description): # description = {train_size}_training / testing
    os.makedirs(output_dir, exist_ok=True)
    output_json = {}

    new_sql_template = data[query_id]["query"]
    literals = data[query_id]["params"]

    for index, combination in enumerate(literals):
        query_str = new_sql_template
        for i, param in enumerate(combination):
            param = str(param).strip()
            pattern = re.compile(rf"@param{i}\b")
            query_str = pattern.sub(param, query_str)
        
        key = f"{query_id}_{description}_{index}"
        output_json[key] = query_str

    # PQO query file naming
    file_name = f'{query_id}_{description}.json'
    output_file = os.path.join(output_dir, file_name)
    if os.path.exists(output_file):
        print(f"{file_name} already exists, skipping save.")
        return
    
    with open(output_file, 'w', encoding='utf-8') as json_file:
        json.dump(output_json, json_file, indent=4)


def save_pqo_predicates(query_id, training_data, testing_data, output_dir, train_size):
    os.makedirs(output_dir, exist_ok=True)

    def save_combinations(data, file_name):
        output_file = os.path.join(output_dir, file_name)
        if os.path.exists(output_file):
            print(f"{file_name} already exists, skipping save.")
            return
        
        with open(output_file, 'w', encoding='utf-8') as file:
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


    # Save training combinations
    save_combinations(training_data[query_id], f'{query_id}_{train_size}_training.txt')

    # Save testing combinations
    save_combinations(testing_data[query_id], f'{query_id}_testing.txt')


def generate_pqo_from_existing_templates(training_file, testing_file, query_id, dirs, train_size):
    """
    Generate PQO files and predicates using existing training and testing JSON templates.

    Args:
    training_file (str): Path to the training template file.
    testing_file (str): Path to the testing template file.
    query_id (str): Query ID.
    dirs (dict): A dictionary containing paths for saving PQO files.
    train_size (int): Training size.
    """
    # Load existing templates
    with open(training_file, 'r', encoding='utf-8') as train_file:
        training_template = json.load(train_file)
    
    with open(testing_file, 'r', encoding='utf-8') as test_file:
        testing_template = json.load(test_file)

    # PQO directories
    pqo_query_dir = os.path.join(dirs['PQO_query'])
    pqo_predicates_dir = os.path.join(dirs['PQO_predicates'])

    # Ensure directories exist
    os.makedirs(pqo_query_dir, exist_ok=True)
    os.makedirs(pqo_predicates_dir, exist_ok=True)

    # Save PQO files
    save_pqo_files(query_id, training_template, pqo_query_dir, f'training_{train_size}')
    save_pqo_files(query_id, testing_template, pqo_query_dir, 'testing')

    # Save predicates
    save_pqo_predicates(query_id, training_template, testing_template, pqo_predicates_dir, train_size)

    print(f"PQO files and predicates have been saved successfully for {query_id}.")


def main(unused_argv):
    # Input from flags
    query_id = _QUERY_ID.value
    train_size = _TRAIN_SIZE.value
    output_dir = _OUTPUT_DIR.value
    input_dir = _INPUT_DIR.value

    # User-provided file paths
    training_file = os.path.join(input_dir, "training", f"{query_id}_training_original_{train_size}.json")
    testing_file = os.path.join(input_dir, "testing", f"{query_id}_testing_original.json")

    # PQO directories
    dirs = {
        "PQO_query": os.path.join(output_dir, "PQO_query"),
        "PQO_predicates": os.path.join(output_dir, "PQO_predicates"),
    }

    # Validate file existence
    if not os.path.exists(training_file):
        raise FileNotFoundError(f"Training file {training_file} does not exist.")
    if not os.path.exists(testing_file):
        raise FileNotFoundError(f"Testing file {testing_file} does not exist.")

    # Generate PQO files and predicates
    generate_pqo_from_existing_templates(training_file, testing_file, query_id, dirs, train_size)


if __name__ == "__main__":
    app.run(main)

