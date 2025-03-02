import json
import pandas as pd

# # robustness db
# for query_id in ['7-0']:
#     input_file = f"imdb_{query_id}_sample/performance/{query_id}_final_output.json"
#     with open(input_file, 'r') as f:
#         data = json.load(f)

#     rows = []

#     for key, values in data.items():
#         method, training_size, confidence_threshold, robustness, db_instance = key.split('_training_')[0], key.split('_training_')[1].split('_confidence_')[0], key.split('_confidence_')[1].split('_robustness_')[0], key.split('_robustness_')[1].split('_db_')[0], key.split('_db_')[1]
#         # E2E Training Time
#         # e2e_training_time = (values['candidate_plan_generation_time_seconds'] + 
#         #                      values['plan_cover_execution_time'] + 
#         #                      values['training_data_collection_time'] + 
#         #                      values['model_prediction_time'])
#         e2e_training_time = f"= {values['candidate_plan_generation_time_seconds']:.2f} + {values['plan_cover_execution_time']:.2f} + {values['training_data_collection_time']:.2f} + {values['model_prediction_time']:.2f}"
        
#         rows.append({
#             'robustness': robustness,
#             'db_instance': int(db_instance),
#             'method': method,
#             'training_size': int(training_size),
#             'confidence_threshold': float(confidence_threshold),
#             'E2E_Training_Time': e2e_training_time,
#             'candidate_plan_generation_time_seconds': values['candidate_plan_generation_time_seconds'],
#             'plan_cover_execution_time': values['plan_cover_execution_time'],
#             'training_data_collection_time': values['training_data_collection_time'],
#             'model_prediction_time': values['model_prediction_time'],
#             '#_of_robust_plans': values['#_of_robust_plans'],
#             'Avg_Robust_ms': values['Avg Robust (ms)'],
#             'Avg_Postgres_ms': values['Avg Postgres (ms)'],
#             'Ratio': values['Ratio'],
#             'Avg_Robust_Cost': values['Avg Robust Cost'],
#             'Avg_Postgres_Cost': values['Avg Postgres Cost'],
#             'Cost_ratio': values['Cost_ratio'],
#             'Robust_is_faster': values['Robust_is_faster']
#         })

#     df = pd.DataFrame(rows)

#     full_performance_file = f"imdb_{query_id}_sample/performance/{query_id}_full_performance.csv"
#     df.to_csv(full_performance_file, index=False)
#     print(f"Full performance CSV saved to {full_performance_file}")

#     best_performance = df.loc[df.groupby(['robustness', 'db_instance', 'method', 'training_size'])['Ratio'].idxmax()]
#     output_file = f"imdb_{query_id}_sample/performance/{query_id}_best_performance.csv"
#     best_performance.to_csv(output_file, index=False)
#     print(f"Best performance CSV saved to {output_file}")


# original db
for query_id in ['16-0']:
    input_file = f"imdb_{query_id}_original/performance/{query_id}_final_output.json"
    with open(input_file, 'r') as f:
        data = json.load(f)

    rows = []

    for key, values in data.items():
        method, training_size, confidence_threshold = key.split('_training_')[0], key.split('_training_')[1].split('_confidence_')[0], key.split('_confidence_')[1]
        # E2E Training Time
        # e2e_training_time = (values['candidate_plan_generation_time_seconds'] + 
        #                      values['plan_cover_execution_time'] + 
        #                      values['training_data_collection_time'] + 
        #                      values['model_prediction_time'])
        e2e_training_time = f"= {values['candidate_plan_generation_time_seconds']:.2f} + {values['plan_cover_execution_time']:.2f} + {values['training_data_collection_time']:.2f} + {values['model_prediction_time']:.2f}"
        
        rows.append({
            'method': method,
            'training_size': int(training_size),
            'confidence_threshold': float(confidence_threshold),
            'E2E_Training_Time': e2e_training_time,
            'candidate_plan_generation_time_seconds': values['candidate_plan_generation_time_seconds'],
            'plan_cover_execution_time': values['plan_cover_execution_time'],
            'training_data_collection_time': values['training_data_collection_time'],
            'model_prediction_time': values['model_prediction_time'],
            '#_of_robust_plans': values['#_of_robust_plans'],
            'Avg_Robust_ms': values['Avg Robust (ms)'],
            'Avg_Postgres_ms': values['Avg Postgres (ms)'],
            'Ratio': values['Ratio'],
            'Avg_Robust_Cost': values['Avg Robust Cost'],
            'Avg_Postgres_Cost': values['Avg Postgres Cost'],
            'Cost_ratio': values['Cost_ratio'],
            'Robust_is_faster': values['Robust_is_faster']
        })

    df = pd.DataFrame(rows)

    full_performance_file = f"imdb_{query_id}_original/performance/{query_id}_full_performance.csv"
    df.to_csv(full_performance_file, index=False)
    print(f"Full performance CSV saved to {full_performance_file}")

    best_performance = df.loc[df.groupby(['method', 'training_size'])['Ratio'].idxmax()]
    output_file = f"imdb_{query_id}_original/performance/{query_id}_best_performance.csv"
    best_performance.to_csv(output_file, index=False)
    print(f"Best performance CSV saved to {output_file}")
