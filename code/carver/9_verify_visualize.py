import os
import json
import pandas as pd

query_ids = ['7-0']
ranging = [1, 2, 3]
robustness_choices = ["random"]

# sliding, random, category - 1, 4

# command template
template_command = (
    "python -m kepler.training_data_collection_pipeline.verify_visualize "
    "--output_dir imdb_{query_id}_sample/{robustness}/db_instance_{i}/{method}/outputs/{robustness_method}/{query_id}/training_{training_size}/confidence_{confidence_threshold} "
    "--model_result_csv_file imdb_{query_id}_sample/{robustness}/db_instance_{i}/{method}/outputs/{robustness_method}/{query_id}/training_{training_size}/confidence_{confidence_threshold}/predictions/verify_{i}/{query_id}_latency_comparison "
    "--query_id {query_id} "
    "--robustness {robustness}"
)


for query_id in query_ids:
    for robustness in robustness_choices:
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
                        
                    robustness_method = "robustness" if robustness == "sliding" else f"robustness_{robustness}"

                    # Format the command with the calculated sizes
                    command = template_command.format(query_id=query_id, method=method, training_size=training_size, confidence_threshold=confidence_threshold, robustness_method=robustness_method, robustness=robustness, i=i)
                    
                    print(command)
                    os.system(command)

