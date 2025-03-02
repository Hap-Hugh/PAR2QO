import json
import os
import pandas as pd

def read_json_file(file_path):
    """Read and return JSON file content"""
    with open(file_path, 'r') as f:
        return json.load(f)

def get_hints_by_indices(plans_data, indices):
    """Get hints from plans data based on provided indices"""
    return [plans_data[int(index)]['hints'] for index in indices]

def process_method_data(base_path, method):
    """Process data for a specific method
    
    Args:
        base_path: Base directory path
        method: Method name (cardinality/csv/kepler)
    
    Returns:
        dict: Processed data including method, plan indices and hints
    """
    # Construct file paths
    metadata_path = os.path.join(base_path, method, 'outputs/results/7-0/training_50/execution_output/imdbloadbase_7-0_metadata.json')
    plans_path = os.path.join(base_path, 'cardinality/outputs/hints/7-0/training_50/imdbloadbase/imdb_plans.json')
    
    # Read files
    try:
        metadata = read_json_file(metadata_path)
        plans_data = read_json_file(plans_path)
        
        # Extract plan cover indices from metadata
        plan_cover = metadata.get('7-0', {}).get('plan_cover', [])
        
        # Get corresponding hints for each plan index
        hints = get_hints_by_indices(plans_data['7-0'], plan_cover)
        
        return {
            'method': method,
            'plan_indices': plan_cover,
            'hints': hints
        }
    except FileNotFoundError as e:
        print(f"Warning: Could not process {method}: {e}")
        return None

def main():
    # Set base directory path
    base_path = '/home/lsh/test_kepler/kepler/imdb_repo_history/imdb_7-0_original'
    
    # Methods to process
    methods = ['cardinality', 'csv', 'kepler']
    
    # Collect data from all methods
    data = []
    for method in methods:
        result = process_method_data(base_path, method)
        if result:
            # Create a row for each hint
            for idx, (plan_idx, hint) in enumerate(zip(result['plan_indices'], result['hints'])):
                data.append({
                    'method': method,
                    'plan_index': plan_idx,
                    'hint': hint
                })
    
    # Create DataFrame and save to CSV
    df = pd.DataFrame(data)
    output_path = 'imdb_hints_comparison.csv'
    df.to_csv(output_path, index=False)
    print(f"CSV file has been created at: {output_path}")
    
    # Print summary statistics
    print("\nSummary:")
    for method in methods:
        method_data = df[df['method'] == method]
        print(f"\n{method.upper()}:")
        print(f"Number of plans: {len(method_data)}")

if __name__ == "__main__":
    main()