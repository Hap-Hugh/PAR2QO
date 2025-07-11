# import os

# def check_candidate_metadata(base_dir, query_id, method, training_size):
#    # Build the file path
#    path = os.path.join(
#        f"imdb_{query_id}_original",
#        method,
#        "outputs",
#        "hints",
#        str(query_id),
#        f"training_{training_size}",
#        "candidate_metadata.json"
#    )
   
#    # Check if file exists
#    exists = os.path.exists(path)
   
#    # Return path if file does not exist
#    if not exists:
#        return path
#    return None

# # Define parameter lists
# methods = ['cardinality', 'csv', 'kepler']
# query_ids = ['1-0', '2-0', '3-0', '4-0', '5-0', '6-0', '7-0', '8-0', '9-0', '10-0', 
#             '11-0', '12-0', '13-0', '14-0', '15-0', '16-0', '17-0', '18-0', '19-0', 
#             '20-0', '21-0', '22-0', '23-0', '25-0', '26-0', '27-0', '28-0', '30-0', 
#             '31-0', '32-0', '33-0']
# training_sizes = [50, 400]

# # Store paths where files were not found
# not_found_paths = []

# # Iterate through all combinations
# for method in methods:
#    for query_id in query_ids:
#        for training_size in training_sizes:
#            result = check_candidate_metadata(os.getcwd(), query_id, method, training_size)
#            if result:
#                not_found_paths.append(result)

# # Print paths where files were not found
# print("\nPaths where candidate_metadata.json was not found:")
# for path in not_found_paths:
#    print(path)
# print(f"\nTotal number of missing files: {len(not_found_paths)}")


import json
import csv
import os
from typing import Dict, List, Tuple
import pandas as pd
from collections import defaultdict

def load_query_data(base_path: str, query_id: str, method: str) -> str:
    """Load query data from JSON file."""
    json_path = os.path.join(
        base_path, 
        f"imdb_{query_id}_original",
        method,
        "inputs",
        "testing",
        f"{query_id}_testing_distinct.json"
    )
    
    try:
        with open(json_path, 'r') as f:
            data = json.load(f)
            return data[query_id]['query']
    except Exception as e:
        print(f"Error loading query data for {method}: {e}")
        return None

def verify_queries(base_path: str, query_id: str) -> bool:
    """Verify if queries are identical across methods."""
    methods = ['cardinality', 'csv', 'kepler']
    queries = {}
    
    # Load queries for each method
    for method in methods:
        query = load_query_data(base_path, query_id, method)
        if query is None:
            return False
        queries[method] = query
    
    # Compare queries
    reference_query = queries[methods[0]]
    for method in methods[1:]:
        if queries[method] != reference_query:
            print(f"Query mismatch found for {method}")
            print(f"Reference query ({methods[0]}): {reference_query}")
            print(f"{method} query: {queries[method]}")
            return False
    
    print(f"All queries for {query_id} are identical!")
    return True

def collect_metadata(base_path: str, query_id: str) -> List[Tuple[str, str, str, str]]:
    """Collect metadata from CSV files."""
    methods = ['cardinality', 'csv', 'kepler']
    target_keys = [
        'distinct_50_training_params',
        'distinct_400_training_params',
        'distinct_2000_training_params'
    ]
    results = []
    
    for method in methods:
        csv_path = os.path.join(
            base_path,
            f"imdb_{query_id}_original",
            method,
            "inputs",
            "metadata",
            f"{query_id}.csv"
        )
        
        try:
            df = pd.read_csv(csv_path)
            # Assuming the CSV has 'Key' and 'Value' columns
            for _, row in df.iterrows():
                if row['Key'] in target_keys:
                    results.append((query_id, method, row['Key'], row['Value']))
        except Exception as e:
            print(f"Error processing metadata for {method}: {e}")
    
    return results

def main():
    # Configure these parameters
    base_path = "."  # Update this
    query_ids = ['1-0', '2-0', '3-0', '4-0', '5-0', '6-0', '7-0', '8-0', '9-0', '10-0', '11-0', '12-0', '13-0', '14-0', '15-0', '16-0', '17-0', '18-0', '19-0', '20-0', '21-0', '22-0', '23-0', '25-0', '26-0', '27-0', '28-0', '30-0', '31-0', '32-0', '33-0']
    
    # Output file for metadata
    output_file = "0_query_metadata_analysis.csv"
    
    # Verify queries and collect metadata
    all_metadata = []
    for query_id in query_ids:
        print(f"\nProcessing query {query_id}...")
        
        # Verify queries
        is_identical = verify_queries(base_path, query_id)
        if not is_identical:
            print(f"Warning: Queries for {query_id} are not identical across methods!")
        
        # Collect metadata
        metadata = collect_metadata(base_path, query_id)
        all_metadata.extend(metadata)
    
    # Write metadata to CSV
    with open(output_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['query_id', 'method', 'Key', 'Value'])
        writer.writerows(all_metadata)
    
    print(f"\nMetadata analysis has been written to {output_file}")

if __name__ == "__main__":
    main()