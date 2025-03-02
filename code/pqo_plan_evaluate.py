import csv
import pandas as pd
from postgres import *



def default_latency(pqo_data, db_name = 'imdbloadbase'):
    results = []
    for id, row in pqo_data.iterrows():
        if 'plan_content' in pqo_data.columns:
            pqo_data.rename(columns={'plan_content': 'plan'}, inplace=True)
        query = row['query']
        pg_plan_latency = round(get_real_latency(db_name, query, hint=None, times=5, limit_time=200000), 5) 
        results.append({
            'query': query,
            'default_latency': pg_plan_latency,
        })
    

def pqo_latency(pqo_data, template_id, workload_name, threshold=0, db_name = 'imdbloadbase'):
    results = []
    latencies = []
    pg_latencies = []
    if 'plan_content' in pqo_data.columns:
        pqo_data.rename(columns={'plan_content': 'plan'}, inplace=True)
    for id, row in pqo_data.iterrows():
        query = row['query']
        plan_content = row['plan']
        if plan_content == "No plan selected":
            pqo_plan_latency = round(get_real_latency(db_name, query, hint=None, times=5, limit_time=200000), 5)

        else:
            pqo_plan_latency = round(get_real_latency(db_name, query, hint=plan_content, times=5, limit_time=200000), 5)
            # pqo_plan_latency = 1
        
        if False:
            pg_plan_latency = round(get_real_latency(db_name, query, hint=None, times=5, limit_time=200000), 5) 
            
        else:
            pg_plan_latency = 1
        
        results.append({
            'query': query,
            'pqo_latency': pqo_plan_latency,
            'pg_plan_latency': pg_plan_latency,
            'plan_content': plan_content
        })

        latencies.append(pqo_plan_latency)
        pg_latencies.append(pg_plan_latency)
        print(f"{template_id} - {id} - {workload_name} - {threshold} - Query: \n {query}")
        print(f"PQO Latency: {pqo_plan_latency} ms, Default latency:", pg_plan_latency, "ms")
        print(f"Current avg Latency: {round(sum(latencies) / len(latencies), 2)} ms / {round(sum(pg_latencies) / len(pg_latencies), 2)} ms")
        print("="*50)
    return results

def save_results_to_csv(results, output_file):
    output_dir = os.path.dirname(output_file)
    os.makedirs(output_dir, exist_ok=True)
    
    with open(output_file, mode='w', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=['query', 'pqo_latency', 'plan_content', 'pg_plan_latency'])
        writer.writeheader()
        for result in results:
            writer.writerow(result)
    
    print(f"Results saved to {output_file}")



def kepler_robust_latency(pqo_data, template_id, workload_name, split, rob_verify=4, log_file=None, db_name = 'imdbloadbase'):
    result_verify_pqo = []
    log_output = []
    if 'plan_content' in pqo_data.columns:
        pqo_data.rename(columns={'plan_content': 'plan'}, inplace=True)
    for id, row in pqo_data.iterrows():
        para_sql = row['query']
        plan_content = row['plan']
        

        pqo_verify_one_query = []
        if split == 'random' or split == 'sliding':
            ins_list = [0, 1, 2, 3, 4, 5, 6, 7, 8]
        else:
            ins_list = [1, 2, 3, 4, 6, 7] # category

        for new_ins_id in ins_list:
            para_sql_per_ins = para_sql.replace(f"_{rob_verify} AS", f"_{new_ins_id} AS")
            robust_plan_latency = round(get_real_latency(db_name, para_sql_per_ins, hint=plan_content, times=3, limit_time=200000), 3)
            pqo_verify_one_query.append(robust_plan_latency)

        result_verify_pqo.append(pqo_verify_one_query)

        pqo_avg = [sum(column) / len(column) for column in list(zip(*result_verify_pqo))]

        output_string = f"Q{template_id}-{workload_name}-{id}:\n {[round(i, 3) for i in pqo_avg]}"
        
        print(output_string)
        log_output.append({
            'query': para_sql,
            'pqo_latency': pqo_verify_one_query,
            'pg_plan_latency': 0,
            'plan_content': plan_content
        })
    save_results_to_csv(log_output, log_file)
    return [round(i, 3) for i in pqo_avg]