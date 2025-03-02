import os
import json

# # robustness db - imdb_{query_id}_sample/{robustness}/db_instance_{i}/{method}/inputs/PQO
# methods = ['cardinality', 'kepler', 'csv']
# query_ids = ['7-0']
# training_sizes = [50, 400]
# robustness_choices = ["category", "random", "sliding"]
# ranging = [1, 2, 3]

# # command template
# template_command = (
#     "python -m kepler.training_data_collection_pipeline.param_PQO_files_generate "
#     "--output_dir imdb_{query_id}_sample/{robustness}/db_instance_{i}/{method}/inputs/PQO "
#     "--input_dir imdb_{query_id}_sample/{robustness}/db_instance_{i}/{method}/inputs "
#     "--query_id {query_id} "
#     "--train_size {training_size}"
# )

# # Cross-method testing
# for query_id in query_ids:
#     for method in methods:
#         for training_size in training_sizes:
#             for robustness in robustness_choices:
#                 for i in ranging:
#                         if robustness == "category" and i in [0, 5, 8]:
#                             continue
                        
#                         command = template_command.format(
#                             query_id=query_id,
#                             method=method,
#                             training_size=training_size,
#                             robustness=robustness,
#                             i=i
#                         )
                        
#                         print(command)
#                         os.system(command)


# original db - imdb_{query_id}_original/{method}/inputs/PQO
methods = ['cardinality', 'csv', 'kepler']
query_ids = ['14-0', '15-0', '17-0', '19-0', '20-0', '21-0', '22-0', '23-0', '25-0', '26-0', '27-0', '28-0', '30-0', '31-0', '32-0', '33-0']
training_sizes = [50, 400, 2000]

# command template
template_command = (
    "python -m kepler.training_data_collection_pipeline.param_PQO_files_generate "
    "--output_dir imdb_{query_id}_original/{method}/inputs/PQO "
    "--input_dir imdb_{query_id}_original/{method}/inputs "
    "--query_id {query_id} "
    "--train_size {training_size}"
)

# Cross-method testing
for query_id in query_ids:
    for method in methods:
        for training_size in training_sizes:
            command = template_command.format(
                query_id=query_id,
                method=method,
                training_size=training_size
            )
            
            print(command)
            os.system(command)
