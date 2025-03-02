import os
import json
import pandas as pd

# original db
methods = ['cardinality', 'kepler']
query_ids = ['1-0', '7-0']
training_sizes = [50, 400]
confidence_thresholds = [0, 0.2, 0.4, 0.6, 0.8]

for query_id in query_ids:
    final_result = {}
    
    # Cross-method testing
    for train_method in methods:
        for test_method in methods:
            if train_method == test_method:
                continue  # Skip when train and test methods are the same
        
            for training_size in training_sizes:
                for confidence_threshold in confidence_thresholds:
                    base_path = f"imdb_{query_id}_original/{test_method}_test_{train_method}_train"

                    # all_hint_efficiency_summary.csv
                    hint_efficiency_csv_path = f"{base_path}/training_{training_size}/confidence_{confidence_threshold}/predictions/all_hint_efficiency_summary.csv"
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
                    final_result[f'{test_method}_test_{train_method}_train_size_{training_size}_confidence_{confidence_threshold}'] = {
                        'Avg Robust (ms)': avg_hinted_latency,
                        'Avg Postgres (ms)': avg_default_latency,
                        'Ratio': ratio,
                        'Avg Robust Cost': avg_hinted_cost,
                        'Avg Postgres Cost': avg_default_cost,
                        'Cost_ratio': ratio_cost,
                        'Robust_is_faster': robust_is_faster
                    }

    output_file = f"imdb_{query_id}_original/workload_performance/{query_id}_final_output.json"
    output_dir = os.path.dirname(output_file)

    # Ensure the directory exists, create it if it doesn't
    os.makedirs(output_dir, exist_ok=True)

    with open(output_file, 'w') as f:
        json.dump(final_result, f, indent=4)

    print(f"Output saved to {output_file}")
