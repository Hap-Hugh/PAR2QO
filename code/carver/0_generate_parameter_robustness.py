import os
import json

# methods = ['kepler', 'csv', 'cardinality']
methods = ['cardinality']
# methods = ['cardinality_full'] # query_ids = ['16-0 - category'] # no result
# query_ids = [f"{i}-0" for i in range(23, 34) if i != 29]
query_ids = ['3-0']
counts = ['300000']
robustness_choices = ["category", "random", "sliding"]
# TODO: only for db_instance_1 and db_instance_4 -> change code in param_gen_test_output_robustness

# command template
template_command = (
    "python3 -m kepler.training_data_collection_pipeline.param_gen_test_output_robustness "
    "--template_file imdb_input/sample_template/{query_id}.json "
    "--output_dir 0_sample_repo/imdb_{query_id}_sample "
    "--robustness {robustness} "
    "--metadata_file imdb_input/sample_template/metadata/query_{query_num}a_json_output.json "
    "--query_id {query_id} "
    "--selection {method} "
    "--count {count}"
)

for query_id in query_ids:
    for method in methods:
        for count in counts:
            for robustness in robustness_choices:
                count = 1000000 if method != "kepler" else count
                query_num = query_id.split('-')[0]
                
                # command format
                command = template_command.format(query_id=query_id, method=method, count=count, query_num=query_num, robustness=robustness)
                
                print(command)
                os.system(command)
