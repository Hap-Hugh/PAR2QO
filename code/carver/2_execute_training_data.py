import os
import json

# # robustness db
# query_ids = ['19-0']
# training_sizes = [50]
# methods = ['cardinality', 'csv', 'kepler']
# robustness_choices = ["sliding"] # "category", "random"
# ranging = [4]

# # command template
# template_command = (
#     "python3 -m kepler.training_data_collection_pipeline.pg_execute_training_data_queries_main "
#     "--database imdbloadbase "
#     "--user kepler "
#     "--password kepler "
#     "--query_templates_file 0_sample_repo/imdb_{query_id}_sample/{robustness}/db_instance_{i}/{method}/inputs/training/{query_id}_training_distinct_{training_size}.json "
#     "--parameter_values_file 0_sample_repo/imdb_{query_id}_sample/{robustness}/db_instance_{i}/{method}/outputs/hints/{query_id}/training_{training_size}/imdbloadbase/imdb_plans_plan_index.json "
#     "--plan_hints_file 0_sample_repo/imdb_{query_id}_sample/{robustness}/db_instance_{i}/{method}/outputs/hints/{query_id}/training_{training_size}/imdbloadbase/imdb_plans.json "
#     "--output_dir 0_sample_repo/imdb_{query_id}_sample/{robustness}/db_instance_{i}/{method}/outputs/results/{query_id}/training_{training_size} "
#     "--plan_cover_num_params {plan_cover_num_params} "
#     "--near_optimal_threshold 1.2"
# )

# for query_id in query_ids:
#     for robustness in robustness_choices:
#         for i in ranging:
#             if robustness == "category" and i in [0, 5, 8]:
#                 continue
#             for method in methods:
#                 for training_size in training_sizes:
#                     training_param_file = f"0_sample_repo/imdb_{query_id}_sample/{robustness}/db_instance_{i}/{method}/inputs/training/{query_id}_training_distinct_{training_size}.json"
                    
#                     with open(training_param_file, 'r') as file:
#                         data = json.load(file)
                        
#                     param_length = len(data[query_id]["params"]) - 1
                    
#                     if param_length >= 500:
#                         param_length = 500
                    
#                     # command format
#                     command = template_command.format(query_id=query_id, method=method, training_size=training_size, plan_cover_num_params=param_length, robustness=robustness, i=i)
                    
#                     print(command)
#                     os.system(command)

# original db
methods = ['cardinality']
query_ids = ['10-0']
training_sizes = [400]

# command template
template_command = (
    "python3 -m kepler.training_data_collection_pipeline.pg_execute_training_data_queries_main "
    "--database imdbloadbase "
    "--user kepler "
    "--password kepler "
    "--query_templates_file 0_finished_repo/imdb_{query_id}_original/{method}/inputs/training/{query_id}_training_distinct_{training_size}.json "
    "--parameter_values_file 0_finished_repo/imdb_{query_id}_original/{method}/outputs/hints/{query_id}/training_{training_size}/imdbloadbase/imdb_plans_plan_index.json "
    "--plan_hints_file 0_finished_repo/imdb_{query_id}_original/{method}/outputs/hints/{query_id}/training_{training_size}/imdbloadbase/imdb_plans.json "
    "--output_dir 0_finished_repo/imdb_{query_id}_original/{method}/outputs/results/{query_id}/training_{training_size} "
    "--plan_cover_num_params {plan_cover_num_params} "
    "--near_optimal_threshold 1.2"
)

for training_size in training_sizes:
    for query_id in query_ids:
        for method in methods:
            training_param_file = f"0_finished_repo/imdb_{query_id}_original/{method}/inputs/training/{query_id}_training_distinct_{training_size}.json"
            
            with open(training_param_file, 'r') as file:
                data = json.load(file)
                
            param_length = len(data[query_id]["params"]) - 1
            
            if param_length >= 500:
                param_length = 500
            
            # command format
            command = template_command.format(query_id=query_id, method=method, training_size=training_size, plan_cover_num_params=param_length)
            
            print(command)
            os.system(command)
