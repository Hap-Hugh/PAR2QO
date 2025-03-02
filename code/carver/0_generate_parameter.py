import os
import json

# methods = ['cardinality', 'kepler', 'csv']
# query_ids = ['1-1', '1-2', '1-3'] # cannot do csv since they have range operator, not exist in sensitive_dict
# query_ids = ['19-0', '24-0] # cardinality killed
# query_ids = ['14-0', '19-0', '22-0', '24-0', '25-0', '28-0', '29-0', '30-0', '31-0] # kepler psycopg2.errors.DiskFull No space left on device

# command template
# template_command = (
#     "python3 -m kepler.training_data_collection_pipeline.param_gen_test_output "
#     "--template_file dsb_cardrange_workload/original_template/{query_id}.json "
#     "--output_dir dsb_cardrange_workload/dsb_{query_id}_original "
#     "--query_id {query_id} "
#     "--selection {method} "
#     "--count {count} "
#     "--db_name dsb" # imdbloadbase or dsb
# )

methods = ['cardinality_full']
query_ids = ['29-0']
counts = ['10000']

template_command = (
    "python3 -m kepler.training_data_collection_pipeline.param_gen_test_output "
    "--template_file imdb_input/original_template/{query_id}.json "
    "--output_dir imdb_{query_id}_original "
    "--metadata_file imdb_input/original_template/metadata/query_{query_num}a_json_output.json "
    "--query_id {query_id} "
    "--selection {method} "
    "--count {count} "
)

for query_id in query_ids:
    for method in methods:
        for count in counts:
            count = 500000 if method != "kepler" else count
            query_num = query_id.split('-')[0]
            
            # command format
            command = template_command.format(query_id=query_id, method=method, count=count, query_num=query_num)
            
            print(command)
            os.system(command)
