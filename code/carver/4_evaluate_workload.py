import os
import json         

# original db
methods = ['cardinality', 'kepler']
query_ids = ['7-0']
training_sizes = [50, 400]
confidence_thresholds = [0, 0.2, 0.4, 0.6, 0.8]

# command template
template_command = (
    "python -m kepler.training_data_collection_pipeline.evaluate "
    "--output_dir imdb_{query_id}_original/{test_method}_test_{train_method}_train/training_{training_size}/confidence_{confidence_threshold}/predictions "
    "--model_result_csv_file imdb_{query_id}_original/{test_method}_test_{train_method}_train/training_{training_size}/confidence_{confidence_threshold}/predictions/{prediction_file} "
    "--testing_parameter_values imdb_{query_id}_original/{test_method}/inputs/testing/{query_id}_testing_original.json "
    "--query_id {query_id}"
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
                    prediction_file = f'{query_id}_init_{param_length}_batch_0.csv'

                    # Format the command with the calculated sizes
                    command = template_command.format(
                        query_id=query_id, 
                        train_method=train_method, 
                        test_method=test_method,
                        training_size=training_size, 
                        confidence_threshold=confidence_threshold, 
                        prediction_file=prediction_file
                    )
                    
                    print(command)
                    os.system(command)
