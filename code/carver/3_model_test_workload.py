import os
import json

# Original db
methods = ['cardinality', 'kepler']
query_ids = ['13-0', '17-0', '19-0', '23-0', '33-0']
training_sizes = [50]
confidence_thresholds = [0]

# command template
template_command = (
    "python3 -m kepler.examples.active_learning_for_training_data_collection_main "
    "--query_metadata imdb_{query_id}_original/kepler/inputs/testing/{query_id}_testing_original.json " # must use kepler's due to min/max modification
    "--vocab_data_dir imdb_input/training_metadata "
    "--execution_metadata imdb_{query_id}_original/{train_method}/outputs/results/{query_id}/training_{training_size}/execution_output/imdbloadbase_{query_id}_metadata.json "
    "--execution_data imdb_{query_id}_original/{train_method}/outputs/results/{query_id}/training_{training_size}/execution_output/imdbloadbase_{query_id}.json "
    "--query_id {query_id} "
    "--testing_parameter_values imdb_{query_id}_original/{test_method}/inputs/testing/{query_id}_testing_original.json "
    "--plan_hints_file imdb_{query_id}_original/{train_method}/outputs/hints/{query_id}/training_{training_size}/imdbloadbase/imdb_plans.json "
    "--num_initial_query_instances {num_initial} "
    "--num_next_query_instances_per_iteration {num_per_iteration} "
    "--output_dir 0_workload_testing/{query_id}/{test_method}_test_{train_method}_train/training_{training_size}/confidence_{confidence_threshold} "
    "--frequency_file imdb_{query_id}_original/{train_method}/inputs/frequency/{query_id}_train_{training_size}_freq.json "
    "--confidence_threshold {confidence_threshold}"
)

# Cross-method testing
for train_method in methods:
    for test_method in methods:
        if train_method == test_method:
            continue  # Skip when train and test methods are the same
        
        for query_id in query_ids:
            for training_size in training_sizes:
                for confidence_threshold in confidence_thresholds:
                    training_param_file = f"imdb_{query_id}_original/{train_method}/inputs/training/{query_id}_training_distinct_{training_size}.json"
                    
                    with open(training_param_file, 'r') as file:
                        data = json.load(file)
                    
                    param_length = len(data[query_id]["params"]) - 1
                    num_initial = param_length
                    num_per_iteration = 0

                    # Format the command with the calculated sizes
                    command = template_command.format(
                        query_id=query_id,
                        train_method=train_method,
                        test_method=test_method,
                        training_size=training_size,
                        confidence_threshold=confidence_threshold,
                        num_initial=num_initial,
                        num_per_iteration=num_per_iteration
                    )
                    
                    print(command)
                    os.system(command)
                    

import json
import shutil
import os
import pandas as pd

def process_outputs(query_ids, test_method, train_method, training_sizes, confidence_thresholds):
    for query_id in query_ids:
        for training_size in training_sizes:
            # Source directory (confidence_0)
            # 0_workload_testing/{query_id}/{test_method}_test_{train_method}_train/training_{training_size}/confidence_{confidence_threshold} 
            source_base = f'0_workload_testing/{query_id}/{test_method}_test_{train_method}_train/training_{training_size}/confidence_0'
            
            # Copy metadata.json
            source_metadata = os.path.join(source_base, 'metadata.json')
            
            # Process predictions
            source_predictions = os.path.join(source_base, 'predictions')
            
            for confidence_threshold in confidence_thresholds:
                if confidence_threshold == 0:
                    continue  # Skip processing for confidence_0 as it's our source
                    
                # Target directory
                target_base = f'0_workload_testing/{query_id}/{test_method}_test_{train_method}_train/training_{training_size}/confidence_{confidence_threshold}'
                os.makedirs(target_base, exist_ok=True)
                
                # Copy metadata.json
                target_metadata = os.path.join(target_base, 'metadata.json')
                shutil.copy2(source_metadata, target_metadata)
                
                # Create predictions directory
                target_predictions = os.path.join(target_base, 'predictions')
                os.makedirs(target_predictions, exist_ok=True)
                
                # Process each CSV file in the predictions directory
                for filename in os.listdir(source_predictions):
                    if filename.endswith('.csv'):
                        source_csv = os.path.join(source_predictions, filename)
                        target_csv = os.path.join(target_predictions, filename)
                        
                        # Read and process CSV
                        df = pd.read_csv(source_csv)
                        
                        # Convert confidence values to float and apply threshold
                        df['confidence'] = pd.to_numeric(df['confidence'], errors='coerce')
                        mask = df['confidence'] < confidence_threshold
                        df.loc[mask, 'plan_id'] = -1
                        df.loc[mask, 'plan_content'] = 'No plan selected'
                        
                        # Save modified CSV
                        df.to_csv(target_csv, index=False)


# Configuration matching your original script
# Cross-method testing
for train_method in methods:
    for test_method in methods:
        if train_method == test_method:
            continue
        new_confidence_thresholds = [0.2, 0.4, 0.6, 0.8]  # Add your desired confidence thresholds here
        process_outputs(query_ids, test_method, train_method, training_sizes, new_confidence_thresholds)
