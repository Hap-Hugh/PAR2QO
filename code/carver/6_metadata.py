import os
import json
import pandas as pd

# # robustness db
# methods = ['cardinality', 'kepler', 'csv']
# query_ids = ['7-0']
# training_sizes = [50, 400]
# confidence_thresholds = [0, 0.2, 0.4, 0.6, 0.8]
# robustness_choices = ["category", "random", "sliding"]
# # ranging = [1, 4]
# ranging = [0, 1, 2, 3]

# final_result = {}

# for query_id in query_ids:
#     for method in methods:
#         for training_size in training_sizes:
#             for confidence_threshold in confidence_thresholds:
#                 for robustness in robustness_choices:
#                     for i in ranging:
#                         if robustness == "category" and i in [0, 5, 8]:
#                             continue
#                         base_path = f"imdb_{query_id}_sample/{robustness}/db_instance_{i}/{method}/outputs"
                        
#                         # candidate_plan_generation_time_seconds
#                         candidate_metadata_path = f"{base_path}/hints/{query_id}/training_{training_size}/candidate_metadata.json"
#                         with open(candidate_metadata_path, 'r') as f:
#                             candidate_metadata = json.load(f)
#                             candidate_plan_time = candidate_metadata.get('candidate_plan_generation_time_seconds', None)

#                         # plan_cover_execution_time & training_data_collection_time
#                         result_metadata_path = f"{base_path}/results/{query_id}/training_{training_size}/execution_output/metadata.json"
#                         with open(result_metadata_path, 'r') as f:
#                             result_metadata = json.load(f)
#                             plan_cover_time = result_metadata.get('plan_cover_execution_time', None)
#                             training_data_time = result_metadata.get('training_data_collection_time', None)
                        
#                         # model_prediction_time
#                         evaluation_metadata_path = f"{base_path}/evaluation/{query_id}/training_{training_size}/confidence_{confidence_threshold}/metadata.json"
#                         with open(evaluation_metadata_path, 'r') as f:
#                             evaluation_metadata = json.load(f)
#                             model_prediction_time = evaluation_metadata.get('model_prediction_time', None)

#                         # # of robust plans
#                         plan_cover_path = f"{base_path}/results/{query_id}/training_{training_size}/execution_output/imdbloadbase_{query_id}_metadata.json"
#                         with open(plan_cover_path, 'r') as f:
#                             plan_cover_metadata = json.load(f)
#                             robust_plans_count = len(plan_cover_metadata.get(query_id, {}).get('plan_cover', []))

#                         # all_hint_efficiency_summary.csv
#                         hint_efficiency_csv_path = f"{base_path}/evaluation/{query_id}/training_{training_size}/confidence_{confidence_threshold}/predictions/all_hint_efficiency_summary.csv"
#                         df = pd.read_csv(hint_efficiency_csv_path)
#                         row = df[df['query_id'] == query_id]

#                         improved_count = row['improved_count'].values[0]
#                         total = row['total'].values[0]
#                         avg_hinted_latency = row['avg_hinted_latency'].values[0]
#                         avg_default_latency = row['avg_default_latency'].values[0]

#                         # Ratio
#                         if avg_default_latency > avg_hinted_latency:
#                             ratio = avg_default_latency / avg_hinted_latency
#                         else:
#                             ratio = -(avg_hinted_latency / avg_default_latency)
                            
#                         # Cost related
#                         avg_hinted_cost = row['avg_hinted_cost'].values[0]
#                         avg_default_cost = row['avg_default_cost'].values[0]
                        
#                         # Ratio
#                         if avg_default_cost > avg_hinted_cost:
#                             ratio_cost = avg_default_cost / avg_hinted_cost
#                         else:
#                             ratio_cost = -(avg_hinted_cost / avg_default_cost)

#                         # Robust is faster
#                         robust_is_faster = f'{improved_count}/{total}'

#                         # distinct/original
#                         metadata_csv_path = f"imdb_{query_id}_sample/{robustness}/db_instance_{i}/{method}/inputs/metadata/{query_id}.csv"
#                         metadata_df = pd.read_csv(metadata_csv_path)
                        
#                         distinct_value = metadata_df[metadata_df['Key'] == f'distinct_{training_size}_training_params']['Value'].values[0]
#                         original_value = metadata_df[metadata_df['Key'] == f'original_{training_size}_training_params']['Value'].values[0]

#                         distinct_original_ratio = f'{distinct_value}/{original_value}'

#                         # save the result
#                         final_result[f'{method}_training_{training_size}_confidence_{confidence_threshold}_robustness_{robustness}_db_{i}'] = {
#                             'candidate_plan_generation_time_seconds': candidate_plan_time,
#                             'plan_cover_execution_time': plan_cover_time,
#                             'training_data_collection_time': training_data_time,
#                             'model_prediction_time': model_prediction_time,
#                             '#_of_robust_plans': robust_plans_count,
#                             'Avg Robust (ms)': avg_hinted_latency,
#                             'Avg Postgres (ms)': avg_default_latency,
#                             'Ratio': ratio,
#                             'Avg Robust Cost': avg_hinted_cost,
#                             'Avg Postgres Cost': avg_default_cost,
#                             'Cost_ratio': ratio_cost,
#                             'Robust_is_faster': robust_is_faster,
#                             f'#_of_distinct_in_{training_size}': distinct_original_ratio
#                         }

#     output_file = f"imdb_{query_id}_sample/performance/{query_id}_final_output.json"
#     output_dir = os.path.dirname(output_file)

#     # Ensure the directory exists, create it if it doesn't
#     os.makedirs(output_dir, exist_ok=True)

#     with open(output_file, 'w') as f:
#         json.dump(final_result, f, indent=4)

#     print(f"Output saved to {output_file}")


# original db
methods = ['cardinality', 'csv', 'kepler']
query_ids = ['16-0']
training_sizes = [50]
confidence_thresholds = [0, 0.2, 0.4, 0.6, 0.8]

final_result = {}

for query_id in query_ids:
    for method in methods:
        for training_size in training_sizes:
            for confidence_threshold in confidence_thresholds:
                base_path = f"imdb_{query_id}_original/{method}/outputs"
                
                # candidate_plan_generation_time_seconds
                candidate_metadata_path = f"{base_path}/hints/{query_id}/training_{training_size}/candidate_metadata.json"
                with open(candidate_metadata_path, 'r') as f:
                    candidate_metadata = json.load(f)
                    candidate_plan_time = candidate_metadata.get('candidate_plan_generation_time_seconds', None)

                # plan_cover_execution_time & training_data_collection_time
                result_metadata_path = f"{base_path}/results/{query_id}/training_{training_size}/execution_output/metadata.json"
                with open(result_metadata_path, 'r') as f:
                    result_metadata = json.load(f)
                    plan_cover_time = result_metadata.get('plan_cover_execution_time', None)
                    training_data_time = result_metadata.get('training_data_collection_time', None)
                
                # model_prediction_time
                evaluation_metadata_path = f"{base_path}/evaluation/{query_id}/training_{training_size}/confidence_{confidence_threshold}/metadata.json"
                with open(evaluation_metadata_path, 'r') as f:
                    evaluation_metadata = json.load(f)
                    model_prediction_time = evaluation_metadata.get('model_prediction_time', None)

                # # of robust plans
                plan_cover_path = f"{base_path}/results/{query_id}/training_{training_size}/execution_output/imdbloadbase_{query_id}_metadata.json"
                with open(plan_cover_path, 'r') as f:
                    plan_cover_metadata = json.load(f)
                    robust_plans_count = len(plan_cover_metadata.get(query_id, {}).get('plan_cover', []))

                # all_hint_efficiency_summary.csv
                hint_efficiency_csv_path = f"{base_path}/evaluation/{query_id}/training_{training_size}/confidence_{confidence_threshold}/predictions/all_hint_efficiency_summary.csv"
                df = pd.read_csv(hint_efficiency_csv_path)
                row = df[df['query_id'] == query_id]

                improved_count = row['improved_count'].values[0]
                total = row['total'].values[0]
                avg_hinted_latency = row['avg_hinted_latency'].values[0]
                avg_default_latency = row['avg_default_latency'].values[0]

                # Ratio
                if avg_default_latency > avg_hinted_latency:
                    ratio = avg_default_latency / avg_hinted_latency
                else:
                    ratio = -(avg_hinted_latency / avg_default_latency)
                
                # Cost related
                avg_hinted_cost = row['avg_hinted_cost'].values[0]
                avg_default_cost = row['avg_default_cost'].values[0]
                
                # Ratio
                if avg_default_cost > avg_hinted_cost:
                    ratio_cost = avg_default_cost / avg_hinted_cost
                else:
                    ratio_cost = -(avg_hinted_cost / avg_default_cost)

                # Robust is faster
                robust_is_faster = f'{improved_count}/{total}'

                # distinct/original
                metadata_csv_path = f"imdb_{query_id}_original/{method}/inputs/metadata/{query_id}.csv"
                metadata_df = pd.read_csv(metadata_csv_path)
                
                distinct_value = metadata_df[metadata_df['Key'] == f'distinct_{training_size}_training_params']['Value'].values[0]
                original_value = metadata_df[metadata_df['Key'] == f'original_{training_size}_training_params']['Value'].values[0]

                distinct_original_ratio = f'{distinct_value}/{original_value}'

                # save the result
                final_result[f'{method}_training_{training_size}_confidence_{confidence_threshold}'] = {
                    'candidate_plan_generation_time_seconds': candidate_plan_time,
                    'plan_cover_execution_time': plan_cover_time,
                    'training_data_collection_time': training_data_time,
                    'model_prediction_time': model_prediction_time,
                    '#_of_robust_plans': robust_plans_count,
                    'Avg Robust (ms)': avg_hinted_latency,
                    'Avg Postgres (ms)': avg_default_latency,
                    'Ratio': ratio,
                    'Avg Robust Cost': avg_hinted_cost,
                    'Avg Postgres Cost': avg_default_cost,
                    'Cost_ratio': ratio_cost,
                    'Robust_is_faster': robust_is_faster,
                    f'#_of_distinct_in_{training_size}': distinct_original_ratio
                }

    output_file = f"imdb_{query_id}_original/performance/{query_id}_final_output.json"
    output_dir = os.path.dirname(output_file)

    # Ensure the directory exists, create it if it doesn't
    os.makedirs(output_dir, exist_ok=True)

    with open(output_file, 'w') as f:
        json.dump(final_result, f, indent=4)

    print(f"Output saved to {output_file}")
