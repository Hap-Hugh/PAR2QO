from diagram import *
from diagram_nearest import *
from diagram_best_cost import *
from diagram_querylog import *
from diagram_querylog_w_sample import * 
from pqo_plan_evaluate import *

# kepler
if False:
    for q in [4]:
        for threshold in ["0.0"]:
            q = str(q)
            template_id = q + "-0"
            train_size = 50
            for w in ["cardinality", "kepler", "csv", ]:
                for mixture in [False]:
                    if not mixture:
                        pqo_result_file = f"./reuse/imdbloadbase/kepler/{threshold}/{template_id}_workload_{w}_training_size_{train_size}.csv"
                        output_file = f"./pqo_result/imdbloadbase/kepler/{threshold}/{template_id}_workload_{w}_kepler_training_size_{train_size}_result.csv"
                        summary_file = f"./pqo_result/imdbloadbase/kepler/{threshold}/{template_id}_workload_{w}_kepler_training_size_{train_size}_summary.csv"
                    else:
                        pqo_result_file = f"./reuse/imdbloadbase/kepler/mixture_test/{template_id}_workload_{w}_training_size_{train_size}.csv"
                        output_file = f"./pqo_result/imdbloadbase/kepler/mixture_test/{template_id}_workload_{w}_kepler_training_size_{train_size}_mixture_result.csv"
                        summary_file = f"./pqo_result/imdbloadbase/kepler/mixture_test/{template_id}_workload_{w}_kepler_training_size_{train_size}_mixture_summary.csv"
                    # pqo_data = pd.read_csv(pqo_result_file)
                    # results = pqo_latency(pqo_data, template_id, w, threshold)
                    # save_results_to_csv(results, output_file)
                    # average_latency = round(sum([r['pqo_latency'] for r in results]) / len(results))
                    # print("*"*50)
                    # print(average_latency)
                    df = pd.read_csv(output_file)
                    print(round(df['pqo_latency'].mean(), 0))
                    print(round(df['pg_plan_latency'].mean(), 0))

# kepler robust:
if False:
    for q in [19]:
        q = str(q)
        template_id = q + "-0"
        for w in ["cardinality", "kepler", "csv", ]:
            for split in ['random']:
                for db_ins in [1]:
                    pqo_result_file = f"./reuse/imdbloadbase/kepler/robust/{template_id}_{split}_db{db_ins}_{w}_training_size_50.csv"
                    summary_file = f"./pqo_result/imdbloadbase/kepler/robust/{split}/{template_id}_db{db_ins}_{w}_kepler_training_size_50_mixture_result.txt"
                    log_file = f"./pqo_result/imdbloadbase/kepler/robust/{split}/{template_id}_db{db_ins}_{w}_kepler_training_size_50_mixture_result.csv"
                    
                    pqo_data = pd.read_csv(pqo_result_file)
                    results = kepler_robust_latency(pqo_data, template_id, w, split=split, rob_verify=db_ins, log_file=log_file)
                    with open(summary_file, 'w') as file:
                        for value in results:
                            file.write(f"{value}\n")
                        file.write(f"\n")


# on-demand pqo
if False:
    for q in [25]:
        q = str(q)
        template_id = q + "-0"
        for w in ["cardinality", "kepler", "csv", ]:
            pqo_result_file= f"reuse/imdbloadbase/on-demand/{w}_workload/pqo-q{q}-t0-50-b0.5.csv"
            output_file = f"./pqo_result/imdbloadbase/ondemand/{template_id}_workload_{w}_ondemand_training_size_50.csv"
            summary_file = f"./pqo_result/imdbloadbase/ondemand/{template_id}_workload_{w}_ondemand_training_size_50_summary.csv"
            # pqo_data = pd.read_csv(pqo_result_file, encoding='ISO-8859-1')
            # results = pqo_latency(pqo_data, template_id, w)
            # save_results_to_csv(results, output_file)
            # average_latency = round(sum([r['pqo_latency'] for r in results]) / len(results))
            # print("*"*50)
            # print(average_latency)
            df = pd.read_csv(output_file)
            print(round(df['pqo_latency'].mean(), 3))
            # print(round(df['pg_plan_latency'].mean(), 0))

# exit()

if __name__ == "__main__":
    # for q_id in [3, 4, 14, 17]:
    # this is robust experiment
    for q_id in []:
        q_id = str(q_id)
        for split in ['category', 'sliding', 'random']:
            if split == 'sliding': instance_id = 4
            else: instance_id = 1
            for t_id in ["0"]:
                for N in [50]:
                    for b in [0.5]:
                        for wn in ["cardinality", "kepler", "csv", ]:
                            test = Diagram(db_name="imdbloadbase", 
                                        workload_name=wn, 
                                        query_id=q_id, 
                                        template_id=t_id, 
                                        n_in_name=50, 
                                        b=b, 
                                        debug=False, 
                                        mixture_test=False, 
                                        rob_verify=split, 
                                        ins_id=str(instance_id))
                            try:
                                test.pqoByFeatureCollection(N=N)
                                test.evaluate(exe=True, rob_verify=instance_id, split=split, R=0)
                            except Exception as e:
                                print("emmm", e)
    # exit()
    for q_id in []:
        print("run", q_id)
        q_id = str(q_id)
        t_id = "0"
        n = 50
        b = 0.5
        for N in [50]:
        
            for remained_plan_size in [0]:
                for wn in ["cardinality", "kepler", "csv", ]:
                    for mixture_test in [False, True]:
                        test = Diagram(db_name="imdbloadbase", 
                                    workload_name=wn, 
                                    query_id=q_id, 
                                    template_id=t_id, 
                                    n_in_name=n, 
                                    b=b, 
                                    debug=False, 
                                    mixture_test=mixture_test, 
                                    rob_verify=None, 
                                    ins_id=None)
                        test.pqoByFeatureCollection(N=N)
                        test.evaluate(exe=True, rob_verify=False, R=remained_plan_size, prune='sim')
                    print("*"*50 + f" {q_id} - {wn} - {N} finished" + "*"*50)
        print("="*50)
        print()
