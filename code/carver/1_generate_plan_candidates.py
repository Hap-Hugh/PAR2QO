import os
# TODO: need manual check, since sometimes it stucks

# # robustness db
# methods = ['cardinality', 'csv', 'kepler']
# query_ids = []
# training_sizes = [50]
# robustness_choices = ["category", "random", "sliding"]
# ranging = [1, 4]

# # command template
# template_command = (
#     "python3 -m kepler.training_data_collection_pipeline.pg_generate_plan_candidates_main "
#     "--database imdbloadbase "
#     "--user kepler "
#     "--password kepler "
#     "--host localhost "
#     "--query_params_file 0_sample_repo/imdb_{query_id}_sample/{robustness}/db_instance_{i}/{method}/inputs/training/{query_id}_training_distinct_{training_size}.json "
#     "--output_dir 0_sample_repo/imdb_{query_id}_sample/{robustness}/db_instance_{i}/{method}/outputs/hints/{query_id}/training_{training_size} "
#     "--plans_output_file imdb_plans.json "
#     "--generation_function row_num_evolution "
#     "--soft_total_plans_limit 30"
# )

# for query_id in query_ids:
#     for robustness in robustness_choices:
#         for i in ranging:
#                 if robustness == "category" and i in [0, 5, 8]:
#                     continue
#                 for method in methods:
#                     for training_size in training_sizes:
#                         command = template_command.format(method=method, query_id=query_id, training_size=training_size, robustness=robustness, i=i)
#                         print(command)
#                         os.system(command)


# original database
# command template
# template_command = (
#     "python3 -m kepler.training_data_collection_pipeline.pg_generate_plan_candidates_main "
#     "--database dsb "
#     "--user kepler "
#     "--password kepler "
#     "--host localhost "
#     "--query_params_file dsb_cardrange_workload/dsb_{query_id}_original/{method}/inputs/training/{query_id}_training_distinct_{training_size}.json "
#     "--output_dir dsb_cardrange_workload/dsb_{query_id}_original/{method}/outputs/hints/{query_id}/training_{training_size} "
#     "--plans_output_file dsb_plans.json "
#     "--generation_function row_num_evolution "
#     "--soft_total_plans_limit 30"
# )

methods = ['cardinality_full']
query_ids = ['29-0']
training_sizes = [50]

template_command = (
    "python3 -m kepler.training_data_collection_pipeline.pg_generate_plan_candidates_main "
    "--database imdbloadbase "
    "--user kepler "
    "--password kepler "
    "--host localhost "
    "--query_params_file imdb_{query_id}_original/{method}/inputs/training/{query_id}_training_distinct_{training_size}.json "
    "--output_dir imdb_{query_id}_original/{method}/outputs/hints/{query_id}/training_{training_size} "
    "--plans_output_file imdb_plans.json "
    "--generation_function row_num_evolution "
    "--soft_total_plans_limit 30"
)


for query_id in query_ids:
    for method in methods:
        for training_size in training_sizes:
            command = template_command.format(method=method, query_id=query_id, training_size=training_size)
            print(command)
            os.system(command)
