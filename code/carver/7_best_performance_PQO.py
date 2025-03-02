import json
import pandas as pd

for query_id in ['q7-t0']:
    kepler_query_id = f"{query_id.split('-')[0][1:]}-{query_id.split('-')[1][1:]}"
    
    input_file = f"imdb_{kepler_query_id}_original_PQO/performance/{kepler_query_id}_final_output.json"
    with open(input_file, 'r') as f:
        data = json.load(f)

    rows = []

    for key, values in data.items():
        method, training_size = key.split('_training_')[0], key.split('_training_')[1]

        rows.append({
            'method': method,
            'training_size': int(training_size),
            'Avg_Robust_ms': values['Avg Robust (ms)'],
            'Avg_Postgres_ms': values['Avg Postgres (ms)'],
            'Ratio': values['Ratio'],
            'Avg_Robust_Cost': values['Avg Robust Cost'],
            'Avg_Postgres_Cost': values['Avg Postgres Cost'],
            'Cost_ratio': values['Cost_ratio'],
            'Robust_is_faster': values['Robust_is_faster']
        })

    df = pd.DataFrame(rows)

    full_performance_file = f"imdb_{kepler_query_id}_original_PQO/performance/{kepler_query_id}_full_performance.csv"
    df.to_csv(full_performance_file, index=False)
    print(f"Full performance CSV saved to {full_performance_file}")
