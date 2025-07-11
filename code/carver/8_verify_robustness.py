import os
import json
import pandas as pd

# robustness db
robustness = "sliding"
query_ids = ['7-0']
ranging = [1, 2, 3]

# command template
template_command = (
    "python -m kepler.training_data_collection_pipeline.verify_robustness "
    "--output_dir imdb_{query_id}_sample/{robustness}/db_instance_{i}/{method}/outputs/robustness/{query_id}/training_{training_size}/confidence_{confidence_threshold}/predictions "
    "--model_result_csv_file imdb_{query_id}_sample/{robustness}/db_instance_{i}/{method}/outputs/evaluation/{query_id}/training_{training_size}/confidence_{confidence_threshold}/predictions/{prediction_file} "
    "--testing_parameter_values imdb_{query_id}_sample/{robustness}/db_instance_{i}/{method}/inputs/testing/{query_id}_testing_original.json "
    "--template imdb_input/sample_template/{query_id}.json "
    "--query_id {query_id} "
    "--i {i}"
)

for query_id in query_ids:
    performance_file = f"imdb_{query_id}_sample/performance/{query_id}_best_performance.csv"
    df = pd.read_csv(performance_file)
    
    db_instance_combinations = {}
    filtered_df = df[(df['robustness'] == robustness) & (df['db_instance'].isin(ranging))]
    
    for db_instance in ranging:
        # get corresponding db_instance
        instance_df = filtered_df[filtered_df['db_instance'] == db_instance]
        
        # get (method, training_size, confidence_threshold)
        db_instance_combinations[db_instance] = [
            (row['method'], row['training_size'], row['confidence_threshold']) 
            for _, row in instance_df.iterrows()
        ]
    
    print(db_instance_combinations)
    
    for i in ranging:
        if robustness == "category" and i in [0, 5, 8]:
            continue
        
        # get the best performance param combination
        if i in db_instance_combinations:
            for method, training_size, confidence_threshold in db_instance_combinations[i]:
                if confidence_threshold == 0.0:
                    confidence_threshold = int(confidence_threshold)
                    
                training_param_file = f"imdb_{query_id}_sample/{robustness}/db_instance_{i}/{method}/inputs/training/{query_id}_training_distinct_{training_size}.json"
                
                with open(training_param_file, 'r') as file:
                    data = json.load(file)
                    
                param_length = len(data[query_id]["params"]) - 1
                prediction_file = f'{query_id}_init_{param_length}_batch_0.csv'

                # Format the command with the calculated sizes
                command = template_command.format(query_id=query_id, method=method, training_size=training_size, confidence_threshold=confidence_threshold, prediction_file=prediction_file, robustness=robustness, i=i)
                
                print(command)
                os.system(command)
