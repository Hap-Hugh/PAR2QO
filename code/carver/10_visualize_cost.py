import os
import json

# original db
methods = ['cardinality', 'kepler', 'csv']
query_ids = ['7-0', '1-0', '5-0']
training_sizes = [50, 400]
confidence_thresholds = [0, 0.2, 0.4, 0.6, 0.8]


# command template
template_command = (
    "python -m kepler.training_data_collection_pipeline.end_visualize_cost "
    "--output_dir imdb_{query_id}_original/{method}/outputs/visualization/{query_id}/training_{training_size}/confidence_{confidence_threshold}/cost "
    "--model_result_csv_file imdb_{query_id}_original/{method}/outputs/evaluation/{query_id}/training_{training_size}/confidence_{confidence_threshold}/predictions/{query_id}_latency_comparison.csv "
    "--cost_file imdb_{query_id}_original/{method}/outputs/evaluation/{query_id}/training_{training_size}/confidence_{confidence_threshold}/predictions/{query_id}_cost_comparison.csv "
    "--query_id {query_id}"
)

for method in methods:
    for query_id in query_ids:
        for training_size in training_sizes:
            for confidence_threshold in confidence_thresholds:
                # Format the command with the calculated sizes
                command = template_command.format(query_id=query_id, method=method, training_size=training_size, confidence_threshold=confidence_threshold)
                
                print(command)
                os.system(command)
