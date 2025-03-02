import os
import json

# # robustness db
# methods = ['cardinality', 'kepler', 'csv']
# query_ids = ['7-0']
# training_sizes = [50, 400]
# confidence_thresholds = [0, 0.2, 0.4, 0.6, 0.8]
# robustness_choices = ["category", "random", "sliding"]
# # ranging = [1, 4]
# ranging = [0, 1, 2, 3]


# # command template
# template_command = (
#     "python -m kepler.training_data_collection_pipeline.end_visualize "
#     "--output_dir imdb_{query_id}_sample/{robustness}/db_instance_{i}/{method}/outputs/visualization/{query_id}/training_{training_size}/confidence_{confidence_threshold} "
#     "--model_result_csv_file imdb_{query_id}_sample/{robustness}/db_instance_{i}/{method}/outputs/evaluation/{query_id}/training_{training_size}/confidence_{confidence_threshold}/predictions/{query_id}_latency_comparison.csv "
#     "--query_id {query_id}"
# )

# for query_id in query_ids:
#     for training_size in training_sizes:
#         for method in methods:
#             for confidence_threshold in confidence_thresholds:
#                 for robustness in robustness_choices:
#                     for i in ranging:
#                         if robustness == "category" and i in [0, 5, 8]:
#                             continue
#                         # Format the command with the calculated sizes
#                         command = template_command.format(query_id=query_id, method=method, training_size=training_size, confidence_threshold=confidence_threshold, robustness=robustness, i=i)
                        
#                         print(command)
#                         os.system(command)
                

# original db
methods = ['cardinality', 'csv', 'kepler']
query_ids = ['16-0']
training_sizes = [50]
confidence_thresholds = [0, 0.2, 0.4, 0.6, 0.8]


# command template
template_command = (
    "python -m kepler.training_data_collection_pipeline.end_visualize "
    "--output_dir imdb_{query_id}_original/{method}/outputs/visualization/{query_id}/training_{training_size}/confidence_{confidence_threshold} "
    "--model_result_csv_file imdb_{query_id}_original/{method}/outputs/evaluation/{query_id}/training_{training_size}/confidence_{confidence_threshold}/predictions/{query_id}_latency_comparison.csv "
    "--query_id {query_id}"
)

for query_id in query_ids:
    for method in methods:
        for training_size in training_sizes:
            for confidence_threshold in confidence_thresholds:
                # Format the command with the calculated sizes
                command = template_command.format(query_id=query_id, method=method, training_size=training_size, confidence_threshold=confidence_threshold)
                
                print(command)
                os.system(command)
