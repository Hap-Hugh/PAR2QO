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
    
    # Save testing combinations
    save_combinations(testing_data[query_id], f'{query_id}_mixture_test.txt')

              
def clear_files(query_ids):
    """
    Clear all PQO result files before processing
    
    Args:
        query_ids: list of query IDs
    """
    for query_id in query_ids:
        file_path = f"0_mixture_test/{query_id}/PQO/{query_id}_mixture_test.txt"
        
        if os.path.exists(file_path):
            os.remove(file_path)
            print(f"Removed existing file: {file_path}")


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
        
        # Process and save predicates
        save_pqo_predicates(query_id, testing_data, pqo_dir)
        
        print(f"Successfully processed testing data for query {query_id}")
        
    except Exception as e:
        print(f"Error processing query {query_id}: {str(e)}")
        raise


def main():
    """
    Main function to process all queries with all methods and training sizes
    """
    # Configuration
    #    query_ids = ['1-0', '7-0', '9-0', '11-0', 
    #                 '19-0', '20-0', '21-0', '23-0', 
    #                 '24-0','26-0', '27-0']
    query_ids = ['1-0', '9-0', '11-0', 
                '19-0', '20-0', '21-0', '23-0', 
                '24-0','26-0', '27-0']
   
    # First clear all PQO predicate file
    print("Clearing existing PQO predicate file...")
    clear_files(query_ids)
    print("Finished clearing predicate file\n")

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