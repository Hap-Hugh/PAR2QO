import os
import json

# # robustness db
# query_ids = ['7-0']
# training_sizes = [50, 400]
# methods = ['cardinality', 'csv', 'kepler']
# robustness_choices = ["category", "random", "sliding"]
# ranging = [0, 1, 2, 3]
# confidence_thresholds = [0, 0.2, 0.4, 0.6, 0.8]


# # command template
# template_command = (
#     "python -m kepler.examples.active_learning_for_training_data_collection_main "
#     "--query_metadata imdb_{query_id}_sample/{robustness}/db_instance_{i}/{method}/inputs/testing/{query_id}_testing_original.json "
#     "--vocab_data_dir imdb_input/training_metadata "
#     "--execution_metadata imdb_{query_id}_sample/{robustness}/db_instance_{i}/{method}/outputs/results/{query_id}/training_{training_size}/execution_output/imdbloadbase_{query_id}_metadata.json "
#     "--execution_data imdb_{query_id}_sample/{robustness}/db_instance_{i}/{method}/outputs/results/{query_id}/training_{training_size}/execution_output/imdbloadbase_{query_id}.json "
#     "--query_id {query_id} "
#     "--testing_parameter_values imdb_{query_id}_sample/{robustness}/db_instance_{i}/{method}/inputs/testing/{query_id}_testing_original.json "
#     "--plan_hints_file imdb_{query_id}_sample/{robustness}/db_instance_{i}/{method}/outputs/hints/{query_id}/training_{training_size}/imdbloadbase/imdb_plans.json "
#     "--num_initial_query_instances {num_initial} "
#     "--num_next_query_instances_per_iteration {num_per_iteration} "
#     "--output_dir imdb_{query_id}_sample/{robustness}/db_instance_{i}/{method}/outputs/evaluation/{query_id}/training_{training_size}/confidence_{confidence_threshold} "
#     "--frequency_file imdb_{query_id}_sample/{robustness}/db_instance_{i}/{method}/inputs/frequency/{query_id}_train_{training_size}_freq.json "
#     "--confidence_threshold {confidence_threshold}"
# )

# for query_id in query_ids:
#     for training_size in training_sizes:
#         for method in methods:
#             for confidence_threshold in confidence_thresholds:
#                 for robustness in robustness_choices:
#                     for i in ranging:
#                         if robustness == "category" and i in [0, 5, 8]:
#                             continue
                        
#                         training_param_file = f"imdb_{query_id}_sample/{robustness}/db_instance_{i}/{method}/inputs/training/{query_id}_training_distinct_{training_size}.json"
#                         print(training_param_file)
                        
#                         with open(training_param_file, 'r') as file:
#                             data = json.load(file)
                            
#                         param_length = len(data[query_id]["params"]) - 1
#                         num_initial = param_length
#                         num_per_iteration = 0

#                         # Format the command with the calculated sizes
#                         command = template_command.format(query_id=query_id, method=method, training_size=training_size, confidence_threshold=confidence_threshold, num_initial=num_initial, num_per_iteration=num_per_iteration, robustness=robustness, i=i)
                        
#                         print(command)
#                         os.system(command)


# Original db
# # command template
# template_command = (
#     "python3 -m kepler.examples.active_learning_for_training_data_collection_main "
#     "--query_metadata dsb_cardrange_workload/original_template_mixture/{query_id}.json "
#     "--vocab_data_dir dsb_training_metadata "
#     "--execution_metadata dsb_cardrange_workload/dsb_{query_id}_original/{method}/outputs/results/{query_id}/training_{training_size}/execution_output/dsb_{query_id}_metadata.json "
#     "--execution_data dsb_cardrange_workload/dsb_{query_id}_original/{method}/outputs/results/{query_id}/training_{training_size}/execution_output/dsb_{query_id}.json "
#     "--query_id {query_id} "
#     "--testing_parameter_values dsb_cardrange_workload/0_mixture_test/{query_id}/{query_id}_mixture_test.json "
#     "--plan_hints_file dsb_cardrange_workload/dsb_{query_id}_original/{method}/outputs/hints/{query_id}/training_{training_size}/dsb/dsb_plans.json "
#     "--num_initial_query_instances {num_initial} "
#     "--num_next_query_instances_per_iteration {num_per_iteration} "
#     "--output_dir dsb_cardrange_workload/0_mixture_test/{query_id}/{method}/training_{training_size}/confidence_{confidence_threshold} "
#     "--confidence_threshold {confidence_threshold}"
# )

# python 3_model_test.py 2>imdb_input/{query_id}/model_test.log
methods = ['kepler']
query_ids = ['4-0']
training_sizes = [400]

# ONLY need one, modify based on the given threshold
confidence_thresholds = [0]

# command template
template_command = (
    "python3 -m kepler.examples.active_learning_for_training_data_collection_main "
    "--query_metadata 0_finished_repo/imdb_{query_id}_original/kepler/inputs/testing/{query_id}_testing_original.json " # must use kepler's due to min/max modification
    "--vocab_data_dir imdb_input/training_metadata "
    "--execution_metadata 0_finished_repo/imdb_{query_id}_original/{method}/outputs/results/{query_id}/training_{training_size}/execution_output/imdbloadbase_{query_id}_metadata.json "
    "--execution_data 0_finished_repo/imdb_{query_id}_original/{method}/outputs/results/{query_id}/training_{training_size}/execution_output/imdbloadbase_{query_id}.json "
    "--query_id {query_id} "
    "--testing_parameter_values 0_mixture_test/{query_id}/{query_id}_mixture_test.json "
    "--plan_hints_file 0_finished_repo/imdb_{query_id}_original/{method}/outputs/hints/{query_id}/training_{training_size}/imdbloadbase/imdb_plans.json "
    "--num_initial_query_instances {num_initial} "
    "--num_next_query_instances_per_iteration {num_per_iteration} "
    "--output_dir 0_mixture_test/{query_id}/{method}/training_{training_size}/confidence_{confidence_threshold} "
    "--confidence_threshold {confidence_threshold}"
)

for query_id in query_ids:
    for method in methods:
        for training_size in training_sizes:
            for confidence_threshold in confidence_thresholds:
                training_param_file = f"0_finished_repo/imdb_{query_id}_original/{method}/inputs/training/{query_id}_training_distinct_{training_size}.json"
                
                with open(training_param_file, 'r') as file:
                    data = json.load(file)
                    
                param_length = len(data[query_id]["params"]) - 1

                num_initial = param_length
                num_per_iteration = 0

                # Format the command with the calculated sizes
                command = template_command.format(query_id=query_id, method=method, training_size=training_size, confidence_threshold=confidence_threshold, num_initial=num_initial, num_per_iteration=num_per_iteration)
                
                print(command)
                os.system(command)


# import json
# import shutil
# import os
# import pandas as pd

# def process_outputs(query_ids, methods, training_sizes, confidence_thresholds):
#     for query_id in query_ids:
#         for method in methods:
#             for training_size in training_sizes:
#                 # Source directory (confidence_0)
#                 source_base = f'0_mixture_test/{query_id}/{method}/training_{training_size}/confidence_0'
                
#                 # Copy metadata.json
#                 source_metadata = os.path.join(source_base, 'metadata.json')
                
#                 # Process predictions
#                 source_predictions = os.path.join(source_base, 'predictions')
                
#                 for confidence_threshold in confidence_thresholds:
#                     if confidence_threshold == 0:
#                         continue  # Skip processing for confidence_0 as it's our source
                        
#                     # Target directory
#                     target_base = f'0_mixture_test/{query_id}/{method}/training_{training_size}/confidence_{confidence_threshold}'
#                     os.makedirs(target_base, exist_ok=True)
                    
#                     # Copy metadata.json
#                     target_metadata = os.path.join(target_base, 'metadata.json')
#                     shutil.copy2(source_metadata, target_metadata)
                    
#                     # Create predictions directory
#                     target_predictions = os.path.join(target_base, 'predictions')
#                     os.makedirs(target_predictions, exist_ok=True)
                    
#                     # Process each CSV file in the predictions directory
#                     for filename in os.listdir(source_predictions):
#                         if filename.endswith('.csv'):
#                             source_csv = os.path.join(source_predictions, filename)
#                             target_csv = os.path.join(target_predictions, filename)
                            
#                             # Read and process CSV
#                             df = pd.read_csv(source_csv)
                            
#                             # Convert confidence values to float and apply threshold
#                             df['confidence'] = pd.to_numeric(df['confidence'], errors='coerce')
#                             mask = df['confidence'] < confidence_threshold
#                             df.loc[mask, 'plan_id'] = -1
#                             df.loc[mask, 'plan_content'] = 'No plan selected'
                            
#                             # Save modified CSV
#                             df.to_csv(target_csv, index=False)


# # Configuration matching your original script
# new_confidence_thresholds = [0.2, 0.4, 0.6, 0.8]  # Add your desired confidence thresholds here
# process_outputs(query_ids, methods, training_sizes, new_confidence_thresholds)