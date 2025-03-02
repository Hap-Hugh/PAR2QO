import os
import json
import pandas as pd

methods = ['cardinality', 'kepler', 'csv']
query_ids = ['q7-t0']
training_sizes = [50, 400]

final_result = {}

for query_id in query_ids:
    kepler_query_id = f"{query_id.split('-')[0][1:]}-{query_id.split('-')[1][1:]}"
    
    for method in methods:
        for training_size in training_sizes:
            base_path = f"imdb_{kepler_query_id}_original_PQO/{method}"

            # all_hint_efficiency_summary.csv
            hint_efficiency_csv_path = f"{base_path}/evaluation/{query_id}_training_{training_size}_efficiency_summary.csv"
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

            # save the result
            final_result[f'{method}_training_{training_size}'] = {
                'Avg Robust (ms)': avg_hinted_latency,
                'Avg Postgres (ms)': avg_default_latency,
                'Ratio': ratio,
                'Avg Robust Cost': avg_hinted_cost,
                'Avg Postgres Cost': avg_default_cost,
                'Cost_ratio': ratio_cost,
                'Robust_is_faster': robust_is_faster
            }

    output_file = f"imdb_{kepler_query_id}_original_PQO/performance/{kepler_query_id}_final_output.json"
    output_dir = os.path.dirname(output_file)
    os.makedirs(output_dir, exist_ok=True)
    
    with open(output_file, 'w') as f:
        json.dump(final_result, f, indent=4)

    print(f"Output saved to {output_file}")
