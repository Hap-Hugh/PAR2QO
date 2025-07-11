import os
import json         

# original db
methods = ['cardinality', 'kepler']
query_ids = ['5-0']
training_sizes = [50, 400]
confidence_thresholds = [0, 0.2, 0.4, 0.6, 0.8]


# command template
template_command = (
    "python -m kepler.training_data_collection_pipeline.end_visualize "
    "--output_dir imdb_{query_id}_original/{test_method}_test_{train_method}_train/training_{training_size}/confidence_{confidence_threshold} "
    "--model_result_csv_file imdb_{query_id}_original/{test_method}_test_{train_method}_train/training_{training_size}/confidence_{confidence_threshold}/predictions/{query_id}_latency_comparison.csv "
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
                    # Format the command with the calculated sizes
                    command = template_command.format(
                        query_id=query_id, 
                        train_method=train_method, 
                        test_method=test_method,
                        training_size=training_size, 
                        confidence_threshold=confidence_threshold
                    )
                    
                    print(command)
                    os.system(command)
