import os
import json
import shutil
from collections import defaultdict
import re

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
   

# PQO input related
def save_pqo_files(query_id, data, output_dir, description): # description = testing
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
        

def save_pqo_predicates(query_id, testing_data, output_dir):
    """
    Save PQO predicates to output files, grouping predicates by alias
    Args:
        query_id: ID of the query
        testing_data: dictionary containing testing data
        output_dir: directory to save output files
    """
    os.makedirs(output_dir, exist_ok=True)
    
    def save_combinations(data, file_name):
        """
        Save combinations to a file, grouping predicates by alias
        Args:
            data: dictionary containing predicates and parameters
            file_name: name of the output file
        """
        output_file = os.path.join(output_dir, file_name)
        if os.path.exists(output_file):
            return
        
        with open(output_file, 'w', encoding='utf-8') as file:
            original_alias_used = False
            
            for i, combination in enumerate(data["params"]):
                # Group predicates by original_alias
                alias_groups = defaultdict(list)
                
                # Format each predicate and group by original_alias  
                for j, item in enumerate(combination):
                    predicate = data["predicates"][j]
                    # Get the grouping key (original_alias if exists, else alias)
                    group_key = predicate.get("original_alias", predicate["alias"])
                    # Always use alias for the predicate string
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
                    
                    # Add to corresponding group using original_alias as key
                    alias_groups[group_key].append(predicate_str)
                    
                    # Track if original_alias is used
                    if "original_alias" in predicate:
                        original_alias_used = True
                
                # Process each alias group
                formatted_groups = []
                for group_key, predicates in alias_groups.items():
                    if len(predicates) == 1:
                        formatted_groups.append(predicates[0])
                    else:
                        # Join multiple predicates with AND
                        group_str = ' AND '.join(predicates)
                        formatted_groups.append(group_str)
                
                # Create final combination string
                combination_str = '["' + '", "'.join(formatted_groups) + '"]'
                
                # Write to file with appropriate line ending
                if i < len(data["params"]) - 1:
                    file.write(combination_str + ',\n')
                else:
                    file.write(combination_str + '\n')

            # Print message if original_alias was used
            if original_alias_used:
                print(f"Used original_alias in file: {file_name}")
    
    # Save testing combinations
    save_combinations(testing_data[query_id], f'{query_id}_mixture_test.txt')
    

def process_query(query_id):
    """
    Process a single query with specific method for testing data only
    Args:
        query_id: ID of the query
    """
    try:
        # Define base directory
        base_dir = f"0_mixture_test/{query_id}"
        
        # Define output directories
        pqo_dir = os.path.join(
            base_dir, 
            "PQO"
        )
        
        # Define testing file path
        testing_file = os.path.join(
            base_dir,
            f"{query_id}_mixture_test.json"
        )
        
        # Load testing data
        testing_data = load_json_data(testing_file)
        
        # Process and save PQO files
        save_pqo_files(query_id, testing_data, pqo_dir, "mixture_test")
        
        # Process and save predicates
        save_pqo_predicates(query_id, testing_data, pqo_dir)
        
        print(f"Successfully processed testing data for query {query_id}")
        
    except Exception as e:
        print(f"Error processing query {query_id}: {str(e)}")
        raise


def clear_directories(query_ids):
    """
    Clear all PQO directories before processing
    Args:
        query_ids: list of query IDs
    """
    for query_id in query_ids:
        base_dir = f"0_mixture_test/{query_id}/PQO"
        
        if os.path.exists(base_dir):
            shutil.rmtree(base_dir)
            print(f"Removed existing directory: {base_dir}")


def main():
    """
    Main function to process testing data for all queries with all methods
    """
    # Configuration
    query_ids = ['33-0']
    
    # First clear all PQO directories
    print("Clearing existing PQO directories...")
    clear_directories(query_ids)
    print("Finished clearing directories\n")
    
    # Process all combinations for testing data
    print("Starting to process queries...")
    for query_id in query_ids:
        try:
            process_query(query_id)
        except Exception as e:
            print(f"Failed to process query {query_id}: {str(e)}")
            continue

if __name__ == "__main__":
    main()