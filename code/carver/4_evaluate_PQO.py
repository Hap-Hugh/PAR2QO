import os
import json

methods = ['kepler']
query_ids = ['q7-t0']
training_sizes = [50, 400]

# command template
template_command = (
    "python -m kepler.training_data_collection_pipeline.evaluate_pqo "
    "--output_dir imdb_{kepler_query_id}_original_PQO/{method}/evaluation "
    "--pqo_result_file imdb_{kepler_query_id}_original_PQO/{method}/pqo-{query_id}-{training_size}-b0.5_freq.csv "
    "--query_id {query_id} "
    "--training_size {training_size}"
)

for method in methods:
    for query_id in query_ids:
        for training_size in training_sizes:
            # Format the command with the calculated sizes
            kepler_query_id = f"{query_id.split('-')[0][1:]}-{query_id.split('-')[1][1:]}"
            command = template_command.format(kepler_query_id=kepler_query_id, query_id=query_id, method=method, training_size=training_size)
            
            print(command)
            os.system(command)
