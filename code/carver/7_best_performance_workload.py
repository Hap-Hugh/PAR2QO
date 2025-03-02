import json
import pandas as pd

# original db
for query_id in ['1-0', '7-0']:
    input_file = f"imdb_{query_id}_original/workload_performance/{query_id}_final_output.json"
    with open(input_file, 'r') as f:
        data = json.load(f)

    rows = []

    for key, values in data.items():
        train_method, test_method, training_size, confidence_threshold = key.split('_test_')[0], key.split('_test_')[1].split('_train_size_')[0], key.split('_train_size_')[1].split('_confidence_')[0], key.split('_confidence_')[1]
        rows.append({
            'training_method': train_method,
            'testing_method': test_method,
            'training_size': int(training_size),
            'confidence_threshold': float(confidence_threshold),
            'Avg_Robust_ms': values['Avg Robust (ms)'],
            'Avg_Postgres_ms': values['Avg Postgres (ms)'],
            'Ratio': values['Ratio'],
            'Avg_Robust_Cost': values['Avg Robust Cost'],
            'Avg_Postgres_Cost': values['Avg Postgres Cost'],
            'Cost_ratio': values['Cost_ratio'],
            'Robust_is_faster': values['Robust_is_faster']
        })

    df = pd.DataFrame(rows)

    full_performance_file = f"imdb_{query_id}_original/workload_performance/{query_id}_full_performance.csv"
    df.to_csv(full_performance_file, index=False)
    print(f"Full performance CSV saved to {full_performance_file}")

    best_performance = df.loc[df.groupby(['training_method', 'testing_method', 'training_size'])['Ratio'].idxmax()]
    output_file = f"imdb_{query_id}_original/workload_performance/{query_id}_best_performance.csv"
    best_performance.to_csv(output_file, index=False)
    print(f"Best performance CSV saved to {output_file}")
