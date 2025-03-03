import os
import json

# # robustness db
# methods = ['cardinality', 'kepler', 'csv']
# query_ids = ['2-0']
# training_sizes = [50, 400, 2000]
# confidence_thresholds = [0, 0.2, 0.4, 0.6, 0.8]
# robustness_choices = ["category", "random", "sliding"]
# ranging = [1, 4]

# # command template
# template_command = (
#     "python -m kepler.training_data_collection_pipeline.evaluate "
#     "--output_dir imdb_{query_id}_sample/{robustness}/db_instance_{i}/{method}/outputs/evaluation/{query_id}/training_{training_size}/confidence_{confidence_threshold}/predictions "
#     "--model_result_csv_file imdb_{query_id}_sample/{robustness}/db_instance_{i}/{method}/outputs/evaluation/{query_id}/training_{training_size}/confidence_{confidence_threshold}/predictions/{prediction_file} "
#     "--testing_parameter_values imdb_{query_id}_sample/{robustness}/db_instance_{i}/{method}/inputs/testing/{query_id}_testing_original.json "
#     "--query_id {query_id}"
# )

# for query_id in query_ids:
#     for robustness in robustness_choices:
#         for i in ranging:
#             if robustness == "category" and i in [0, 5, 8]:
#                 continue
#             for method in methods:
#                 for training_size in training_sizes:
#                     for confidence_threshold in confidence_thresholds:
#                         training_param_file = f"imdb_{query_id}_sample/{robustness}/db_instance_{i}/{method}/inputs/training/{query_id}_training_distinct_{training_size}.json"
#                         print(f"####### Training param file {training_param_file}")
                        
#                         with open(training_param_file, 'r') as file:
#                             data = json.load(file)
                            
#                         param_length = len(data[query_id]["params"]) - 1
#                         prediction_file = f'{query_id}_init_{param_length}_batch_0.csv'

#                         # Format the command with the calculated sizes
#                         command = template_command.format(query_id=query_id, method=method, training_size=training_size, confidence_threshold=confidence_threshold, prediction_file=prediction_file, robustness=robustness, i=i)
                        
#                         print(command)
#                         os.system(command)
                

# original db
methods = ['cardinality', 'kepler', 'csv']
query_ids = ['16-0', '18-0']
training_sizes = [50]
confidence_thresholds = [0, 0.2, 0.4, 0.6, 0.8]

# command template
template_command = (
    "python -m kepler.training_data_collection_pipeline.evaluate_both "
    "--kepler_output_dir imdb_{query_id}_original/{method}/outputs/evaluation/{query_id}/training_{training_size}/confidence_{confidence_threshold}/predictions "
    "--kepler_model_result_csv_file imdb_{query_id}_original/{method}/outputs/evaluation/{query_id}/training_{training_size}/confidence_{confidence_threshold}/predictions/{prediction_file} "
    "--kepler_testing_parameter_values imdb_{query_id}_original/{method}/inputs/testing/{query_id}_testing_original.json "
    "--query_id {query_id} "
    "--pqo_output_dir imdb_{query_id}_original_PQO/{method}/kepler_confidence_{confidence_threshold} "
    "--pqo_result_file imdb_{query_id}_original_PQO/{method}/pqo-{pqo_query_id}-{training_size}-b0.5_freq.csv "
    "--pqo_training_size {training_size}"
)

for query_id in query_ids:
    pqo_query_id = f"q{query_id.split('-')[0]}-t{query_id.split('-')[1]}"
    
    for method in methods:
        for training_size in training_sizes:
            for confidence_threshold in confidence_thresholds:                
                training_param_file = f"imdb_{query_id}_original/{method}/inputs/training/{query_id}_training_distinct_{training_size}.json"
                
                with open(training_param_file, 'r') as file:
                    data = json.load(file)
                    
                param_length = len(data[query_id]["params"]) - 1
                prediction_file = f'{query_id}_init_{param_length}_batch_0.csv'

                # Format the command with the calculated sizes
                command = template_command.format(query_id=query_id, method=method, training_size=training_size, confidence_threshold=confidence_threshold, prediction_file=prediction_file, pqo_query_id=pqo_query_id)
                
                print(command)
                os.system(command)
