import os
import json
import shutil
import re
from collections import defaultdict

def load_json_data(file_path):
   """
   Load data from a JSON file
   Args:
       file_path: path to the JSON file
   Returns:
       dictionary containing the JSON data
   """
   try:
       with open(file_path, 'r', encoding='utf-8') as f:
           return json.load(f)
   except Exception as e:
       print(f"Error loading JSON file {file_path}: {str(e)}")
       raise


def save_pqo_predicates(query_id, training_data, testing_data, output_dir, train_size):
    """
    Save PQO predicates to output files, grouping predicates by alias
    Args:
        query_id: ID of the query
        training_data: dictionary containing training data
        testing_data: dictionary containing testing data
        output_dir: directory to save output files
        train_size: size of training data
    """
    os.makedirs(output_dir, exist_ok=True)
    
    def save_combinations(data, file_name):
        """
        Save combinations to a file, grouping predicates by alias and handling OR groups
        Args:
            data: dictionary containing predicates, parameters, and query
            file_name: name of the output file
        """
        output_file = os.path.join(output_dir, file_name)
        if os.path.exists(output_file):
            return

        # First, analyze the query to find OR groups
        query = data.get("query", "")
        or_groups = []
        
        # Find all patterns matching "(* OR *)" or similar patterns at start/end of string
        or_patterns = re.finditer(r'\(([^()]+?\bOR\b[^()]+?)\)', query, re.IGNORECASE)
        
        # Store information about OR groups and their tables
        for match in or_patterns:
            or_expr = match.group(1)
            if ' OR ' in or_expr.upper():
                # Split the OR expression and extract table names
                predicates = [p.strip() for p in or_expr.split(' OR ')]
                
                # Extract table name from first part of predicate (before the dot)
                tables = [p.split('.')[0].strip() for p in predicates]
                
                # get @param indices
                param_indices = []
                for p in predicates:
                    param_match = re.search(r'@param(\d+)', p)
                    if param_match:
                        param_indices.append(int(param_match.group(1)))
                    
                # Only store if all predicates are from the same table
                if len(set(tables)) == 1:
                    or_groups.append({
                        'table': tables[0],
                        'predicates': predicates,
                        'param_indices': param_indices
                    })
                    
            # print(or_groups)

        with open(output_file, 'w', encoding='utf-8') as file:
            original_alias_used = False

            for i, combination in enumerate(data["params"]):
                # Create a map of table -> predicates for current combination
                table_predicates = defaultdict(list)
                table_param_indices = defaultdict(list)
                
                # Format predicates and group by table
                for j, item in enumerate(combination):
                    predicate = data["predicates"][j]
                    # Get the grouping key (original_alias if exists, else alias)
                    group_key = predicate.get("original_alias", predicate["alias"])
                    table = predicate["alias"]
                    column = predicate["column"]
                    operator = predicate["operator"]
                    data_type = predicate["data_type"]

                    # Format parameter based on data type
                    if data_type == "text":
                        formatted_param = f"'{item}'"
                    else:
                        formatted_param = item

                    # Handle IN operator
                    if operator.lower() == "in":
                        formatted_param = f"({formatted_param})"

                    # Create predicate string
                    predicate_str = f"{table}.{column} {operator} {formatted_param}"
                    table_predicates[group_key].append(predicate_str)
                    table_param_indices[table].append(j)

                    if "original_alias" in predicate:
                        original_alias_used = True

                # Process predicates table by table
                formatted_groups = []
                processed_tables = set()

                # First, handle OR groups
                for group in or_groups:
                    table = group['table']
                    # print("table predicates", table_predicates)
                    # print("table param indices", table_param_indices)
                    if table in table_predicates and table not in processed_tables:
                        predicates = []
                        or_predicates = []
                        
                        # get all predicates
                        for pred, idx in zip(table_predicates[table], table_param_indices[table]):
                            if idx in group['param_indices']:
                                or_predicates.append(pred)
                            else:
                                predicates.append(pred)
                        # print("OR predicates", or_predicates)
                        # print("and predicates", predicates)
                        
                        # form or_predicates
                        if or_predicates:
                            or_str = f"({' OR '.join(or_predicates)})"
                            if predicates:
                                # add the other predicates
                                group_str = f"{or_str} AND {' AND '.join(predicates)}"
                            else:
                                group_str = or_str
                            formatted_groups.append(group_str)
                            processed_tables.add(table)

                # Handle remaining predicates
                for table, predicates in table_predicates.items():
                    if table not in processed_tables:
                        if len(predicates) == 1:
                            formatted_groups.append(predicates[0])
                        else:
                            group_str = ' AND '.join(predicates)
                            formatted_groups.append(group_str)

                # Create final combination string
                combination_str = '["' + '", "'.join(formatted_groups) + '"]'

                # Write to file with appropriate line ending
                if i < len(data["params"]) - 1:
                    file.write(combination_str + ',\n')
                else:
                    file.write(combination_str + '\n')

            if original_alias_used:
                print(f"Used original_alias in file: {file_name}")
            
    # Save training combinations
    save_combinations(training_data[query_id], f'{query_id}_{train_size}_training.txt')
    
    # Save testing combinations
    save_combinations(testing_data[query_id], f'{query_id}_testing.txt')

    
def clear_join_predicates(query_ids, methods, robustness_methods, db_instances):
   """
   Clear all join_predicates directories before processing
   Args:
       query_ids: list of query IDs
       methods: list of methods
   """
   for robustness in robustness_methods:
       for i in db_instances:
            for method in methods:
                for query_id in query_ids:
                    base_dir = f"0_sample_repo/imdb_{query_id}_sample"
                    join_predicates_dir = os.path.join(
                        base_dir,
                        robustness,
                        f"db_instance_{i}",
                        method, 
                        "inputs",
                        "PQO",
                        "join_predicates"
                    )
                    
                    if os.path.exists(join_predicates_dir):
                        shutil.rmtree(join_predicates_dir)
                        print(f"Removed existing directory: {join_predicates_dir}")


def process_query(query_id, train_size, method, robustness, i):
   """
   Process a single query with specific method
   Args:
       query_id: ID of the query
       train_size: size of training data
       method: processing method (cardinality/csv/kepler)
   """
   try:
       # Define base directory
       base_dir = f"0_sample_repo/imdb_{query_id}_sample"
       
       # Define join_predicates directory
       join_predicates_dir = os.path.join(
           base_dir,
           robustness,
           f"db_instance_{i}",
           method, 
           "inputs",
           "PQO",
           "join_predicates"
       )
       
       # Define file paths
       training_file = os.path.join(
           base_dir,
           robustness,
           f"db_instance_{i}",
           method,
           "inputs",
           "training",
           f"{query_id}_training_original_{train_size}.json"
       )
       
       testing_file = os.path.join(
           base_dir,
           robustness,
           f"db_instance_{i}",
           method,
           "inputs",
           "testing",
           f"{query_id}_testing_original.json"
       )
       
       
       # Load data
       training_data = load_json_data(training_file)
       testing_data = load_json_data(testing_file)
       
       # Process and save predicates
       save_pqo_predicates(query_id, training_data, testing_data, join_predicates_dir, train_size)
       
       print(f"Successfully processed query {query_id} with method {method}")
       
   except Exception as e:
       print(f"Error processing query {query_id} with method {method}: {str(e)}")
       raise


def main():
   """
   Main function to process all queries with all methods and training sizes
   """
   # Configuration
   methods = ['cardinality', 'csv', 'kepler']
   robustness_methods = ["category", "random", "sliding"]
   db_instances = [1, 4]
#    query_ids = ['1-0', '7-0', '9-0', '11-0', 
#                 '19-0', '20-0', '21-0', '23-0', 
#                 '24-0','26-0', '27-0']
   query_ids = ['1-0']
   training_sizes = [50]
   
   # First clear all join_predicates directories
   print("Clearing existing join_predicates directories...")
   clear_join_predicates(query_ids, methods, robustness_methods, db_instances)
   print("Finished clearing directories\n")
   
   # Then process all combinations
   print("Starting to process queries...")
   for robustness in robustness_methods:
       for i in db_instances:
            for method in methods:
                for query_id in query_ids:
                    for train_size in training_sizes:
                        try:
                            process_query(query_id, train_size, method, robustness, i)
                        except Exception as e:
                            print(f"Failed to process query {query_id} with train size {train_size} and method {method}: {str(e)}")
                            continue

if __name__ == "__main__":
   main()