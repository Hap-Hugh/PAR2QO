import os
import json
import shutil
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

    # Save training combinations
    save_combinations(training_data[query_id], f'{query_id}_{train_size}_training.txt')
    
    # Save testing combinations
    save_combinations(testing_data[query_id], f'{query_id}_testing.txt')
    
def clear_join_predicates(query_ids, methods):
   """
   Clear all join_predicates directories before processing
   Args:
       query_ids: list of query IDs
       methods: list of methods
   """
   for method in methods:
       for query_id in query_ids:
           base_dir = f"imdb_{query_id}_original"
           join_predicates_dir = os.path.join(
               base_dir,
               method, 
               "inputs",
               "PQO",
               "join_predicates"
           )
           
           if os.path.exists(join_predicates_dir):
               shutil.rmtree(join_predicates_dir)
               print(f"Removed existing directory: {join_predicates_dir}")

def process_query(query_id, train_size, method):
   """
   Process a single query with specific method
   Args:
       query_id: ID of the query
       train_size: size of training data
       method: processing method (cardinality/csv/kepler)
   """
   try:
       # Define base directory
       base_dir = f"imdb_{query_id}_original"
       
       # Define join_predicates directory
       join_predicates_dir = os.path.join(
           base_dir,
           method, 
           "inputs",
           "PQO",
           "join_predicates"
       )
       
       # Define file paths
       training_file = os.path.join(
           base_dir,
           method,
           "inputs",
           "training",
           f"{query_id}_training_original_{train_size}.json"
       )
       
       testing_file = os.path.join(
           base_dir,
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
#    methods = ['cardinality', 'csv', 'kepler']
   methods = ['kepler']
#    query_ids = ['1-0', '2-0', '3-0', '4-0', '5-0', '6-0', '7-0', '8-0', '9-0', 
#                 '10-0', '11-0', '12-0', '13-0', '14-0', '15-0', '16-0', '17-0', 
#                 '18-0', '19-0', '20-0', '21-0', '22-0', '23-0', '25-0', '26-0', 
#                 '27-0', '28-0', '30-0', '31-0', '32-0', '33-0']
   query_ids = ['33-0']
   training_sizes = [50]
   
   # First clear all join_predicates directories
   print("Clearing existing join_predicates directories...")
   clear_join_predicates(query_ids, methods)
   print("Finished clearing directories\n")
   
   # Then process all combinations
   print("Starting to process queries...")
   for method in methods:
       for query_id in query_ids:
           for train_size in training_sizes:
               try:
                   process_query(query_id, train_size, method)
               except Exception as e:
                   print(f"Failed to process query {query_id} with train size {train_size} and method {method}: {str(e)}")
                   continue

if __name__ == "__main__":
   main()