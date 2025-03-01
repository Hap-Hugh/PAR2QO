import json
import re
import pandas as pd
import numpy as np
import logging
import matplotlib.pyplot as plt
from matplotlib import cycler
import matplotlib.cm as cm
import psycopg2
import seaborn as sns
from postgres import get_plan_cost
# from prep_error_list import err_files_dict_job

from utility import gen_center_from_err_dist, find_bin_id_from_err_hist_list
from prep_cardinality import get_maps, ori_cardest, get_raw_table_size, write_to_file, write_pointers_to_file
from prep_error_list import prepare_error_data, cal_pdf
from prep_selectivity import prep_sel

with open("cached_info/sensitive_dict.json", 'r') as f:
    data = json.load(f)
    # if db_name == 'imdbloadbase':
sen_dict_name = "sen_dict_sobol"
    # if db_name == 'dsb':
    #     sen_dict_name =  "dsb_sen_dict_sobol"
    # if db_name == 'stats':
    #     sen_dict_name =  "stats_sen_dict_sobol"
sen_dict = data[sen_dict_name]

explain = "EXPLAIN (SUMMARY, COSTS, FORMAT JSON)"
file_of_base_sel = './cardinality/new_single.txt'  # file to be sent to pg folder, contains cardinality for base_rel
file_of_join_sel = './cardinality/join.txt'  # file to be sent to pg folder, contains cardinality for join_rel

def check_common(workload,n,query,template,sql=None):
    def filter_join(hint):
        join_method_pattern = re.compile(r'(NestLoop|HashJoin|MergeJoin)\s*\(.*?\)')
        leading_pattern = re.compile(r'Leading\s*\(.*?\)')

        # Extract join methods
        join_methods = join_method_pattern.findall(hint)

        # Extract the Leading clause (join order)
        leading_clause = leading_pattern.findall(hint)

        # Combine the filtered join methods and Leading clause
        filtered_hint = ''.join(join_methods + leading_clause)

        # Use a better match extraction to ensure we're capturing the content inside parentheses
        join_methods_full = re.findall(r'(NestLoop|HashJoin|MergeJoin)\s*\(([^)]+)\)', hint)
        leading_clause_full = re.findall(r'Leading\s*\(([^)]+)\)', hint)

        # Prepare the output strings
        filtered_plan_hint = ''.join([f"{method}({tables})" for method, tables in join_methods_full])
        filtered_plan_hint += 'Leading(' + leading_clause_full[0] + ')' if leading_clause_full else ''

        return filtered_plan_hint

    logging.basicConfig(filename='./log/test.log', level=logging.INFO)

    logging.info('---------------------------------------------')
    ########### Kepler
    if 1:
        query_template = f"{query}-{template}"
        kepler_plans_folder = f"/home/lsh/test_kepler/kepler/imdb_{query_template}/plan/"
        kepler_plan_path = f"{kepler_plans_folder}{workload}/training_{n}/imdb_plans.json"
        kepler_plan_cover_path = f"{kepler_plans_folder}{workload}/training_{n}/imdbloadbase_{query_template}_metadata.json"

        with open(kepler_plan_path, 'r') as file:
            kepler_plan = json.load(file)

        with open(kepler_plan_cover_path, 'r') as file:
            kepler_plan_cover = json.load(file)

        kepler_robust_plans = []
        for plan_id in kepler_plan_cover[query_template]["plan_cover"]:
            # print(plan_id)
            plan = kepler_plan[query_template][plan_id]["hints"]
            # plan["hints"] = plan["hints"].replace("BitmapScan", "SeqScan")
            # print(re.sub(r'\((\w+)\s+\w+_\w+\)', r'(\1)', plan["hints"]).replace(" ",""))

            # conn = psycopg2.connect(host="/tmp", dbname="imdbloadbase", user="lsh")
            # conn.set_session(autocommit=True)
            # cursor = conn.cursor()
            # _, ori_join_order, ori_scan_mtd = get_plan_cost(cursor, sql, plan["hints"], explain=explain, debug=True)
            # plan = gen_final_hint(scan_mtd=ori_scan_mtd, str=ori_join_order)
            # # print("".join(plan.split()))

            # plan = re.sub(r'\((\w+)\s+\w+_\w+\)', r'(\1)', plan).replace(" ","")
            # plan = filter_join(plan)
            # kepler_robust_plans.append(plan)


            kepler_robust_plans.append("".join(plan.split()))

    ########### PQO
    if 1:
        meta_file = f"/home/lsh/PARQO_backend/reuse/on-demand/anchors/{workload}_workload/q{query}-t{template}-{n}-optimized-queries-b0.5.json"

        with open(meta_file, 'r') as file:
            pqo_num_query = len(json.load(file))

        pqo_robust_plans = []
        for i in range(pqo_num_query):
            log_file_path = f"/home/lsh/PARQO_backend/log/on-base/imdbloadbase/on-demand/anchors/{workload}_workload/imdbloadbase_q{query}-t{template}-{n}-{i}-b0.5_0.2.log"

            with open(log_file_path, 'r') as file:
                log_data = file.read()

            pattern = r'INFO:root:### Best plan by exp penalty: \[([\d,\s]+)\]'
            matches = re.findall(pattern, log_data)

            if matches:
                for match in matches:
                    best_plan = list(map(int, match.split(',')))
                    # print(i, best_plan)

            pqo_plans_path = f"/home/lsh/PARQO_backend/plan/on-base/imdbloadbase/pqo-entire-space/on-demand/{workload}_workload/on-basetmp_plan_dict_imdbloadbase_q{query}-t{template}-{n}-{i}-b0.5.txt"
            with open(pqo_plans_path, 'r') as file:
                pqo_plans = json.load(file)["0"]
            
            # for plan_id in [best_plan[0]]:
            for plan_id in best_plan:
                plan = ' '.join(pqo_plans[plan_id].split())
                # plan = filter_join(plan)
                pqo_robust_plans.append((plan))
            
            
            # # all candidate plans
            # for plan_id in range(len(pqo_plans)):
            #     pqo_robust_plans.append((' '.join(pqo_plans[plan_id].split())
            #                             ,i,plan_id))

    ########### Comparison
    if 1:
        kepler_common = []
        pqo_common = []
        pqo_robust_plans = set(pqo_robust_plans)
        for i, kepler_hint in enumerate(kepler_robust_plans):
            for pqo_hint in pqo_robust_plans:
                print(kepler_hint)
                print(pqo_hint)
                print()
                if kepler_hint == pqo_hint:
                    # print(f"PQO anchor{j}-plan{plan_id} is in Kepler's plan cover.")
                    pqo_common.append((pqo_hint))
                    print(kepler_hint)
                    print(pqo_hint)
                    print()
                    kepler_common.append(i)

        logging.info(f"{workload}-{n} workload:")
        logging.info(f"{len(set(kepler_common))}/{len(kepler_robust_plans)} of Kepler's plan_cover is in PQO's all candidate plans")
        logging.info(f"There are {len([1 for h in set(kepler_robust_plans) if 'BitmapScan' in h])} plans in Kepler with Bitmap Heap Scan")
        logging.info(f"{len(set(pqo_common))}/{len(set(pqo_robust_plans))} of PQO's all candidate plans are in Kepler's plan_cover")

    ########### Used ratio
    if 1:
        test_result = pd.read_csv(f"/home/lsh/PARQO_backend/reuse/imdbloadbase/on-demand/{workload}_workload/pqo-q{query}-t{template}-{n}-b0.5_freq.csv")
        used = len([1 for hint in test_result["plan"] if hint.replace(" ","") in pqo_common])
        logging.info(f"The used ratio is: {used}/{test_result.freq.sum()}")
    logging.info('---------------------------------------------')

def get_latency_detail(workload,n,query,template):
    print(f"for {workload}-{query}-{template}-{n}")
    pqo_latency_results_path = f"/home/lsh/test_kepler/kepler/imdb_repo/imdb_{query}-{template}_original_PQO/{workload}/evaluation/q{query}-t{template}_training_{n}_latency_comparison.csv"
    meta_info_path = f"/home/lsh/PARQO_backend/reuse/imdbloadbase/on-demand/{workload}_workload/pqo-q{query}-t{template}-{n}-b0.5.json"
    kepler_best_performance_path = f"/home/lsh/test_kepler/kepler/imdb_repo_history/imdb_{query}-{template}_original/performance/{query}-{template}_best_performance.csv"
    df = pd.read_csv(kepler_best_performance_path)
    confidence_threshold = df[(df["method"]==workload) & (df["training_size"]==n)]["confidence_threshold"].values[0]
    print(confidence_threshold)
    if confidence_threshold == 0:
        confidence_threshold = int(confidence_threshold)
    kepler_latency_results_path = f"/home/lsh/test_kepler/kepler/imdb_repo_history/imdb_{query}-{template}_original/{workload}/outputs/evaluation/{query}-{template}/training_{n}/confidence_{confidence_threshold}/predictions/{query}-{template}_latency_comparison.csv"
    pqo_df = pd.read_csv(pqo_latency_results_path, encoding='latin1')
    kepler_df = pd.read_csv(kepler_latency_results_path, encoding='latin1')
    pqo_df['kepler_latency'] = kepler_df['hinted_latency']


    with open(meta_info_path, 'r') as json_file:
        json_data = json.load(json_file)

    def normalize_string(text):
        return "".join(text.split())

    for _, row in pqo_df.iterrows():
        # Normalize and combine the CSV query and plan_content to create the CSV identifier
        csv_query = normalize_string(row['query'])
        csv_plan_content = normalize_string(row['plan_content'])
        csv_combined = csv_query + csv_plan_content

        # Now loop through the JSON entries to find a matching query and hint[robust_plan]
        for json_entry in json_data:
            # Normalize and combine the JSON query and hint[robust_plan]
            json_query = normalize_string(json_entry['sql'])
            json_hint = normalize_string(json_entry['hint'][json_entry['robust_plan']])
            json_combined = json_query + json_hint

            # If they match, add latency data from CSV to JSON
            if csv_combined == json_combined:
                json_entry['pqo_latency'] = row['hinted_latency']
                json_entry['default_latency'] = row['default_latency']
                json_entry['kepler_latency'] = row['kepler_latency']
                break  # Move to the next CSV row after finding the match

    updated_json_path = f"/home/lsh/PARQO_backend/reuse/imdbloadbase/on-demand/{workload}_workload/hint_dist/latency-pqo-q{query}-t{template}-{n}-b0.5.json"
    with open(updated_json_path, 'w') as outfile:
        json.dump(json_data, outfile, indent=4)

    print(f"Updated JSON saved to {updated_json_path}")
    pass

def plot_latency_hist(workload,n,query,template):
    latency_detail_path = f"/home/lsh/PARQO_backend/reuse/imdbloadbase/on-demand/{workload}_workload/hint_dist/latency-pqo-q{query}-t{template}-{n}-b0.5.json"
    with open(latency_detail_path, 'r') as json_file:
        json_data = json.load(json_file)

    df = pd.DataFrame(json_data)
    df['pqo_over_kepler_ratio'] = df['pqo_latency'] / df['kepler_latency']

    # Define bins for the latency (x-axis)
    # bins = [0, 50, 100, 200, 500, 1000, 1500]
    bins = [0, 1, 2, 3, 5, 10]

    # # Create a new column representing the binned latencies for both hinted and default
    # df['hinted_latency_bin'] = pd.cut(df['hinted_latency'], bins=bins)
    # df['default_latency_bin'] = pd.cut(df['default_latency'], bins=bins)

    # # Combine hinted and default latencies into one dataset for plotting
    # # df_combined = pd.concat([df[['nearest_query', 'hinted_latency_bin']].rename(columns={'hinted_latency_bin': 'latency_bin'}),
    # #                         df[['nearest_query', 'default_latency_bin']].rename(columns={'default_latency_bin': 'latency_bin'})])
    # df_combined = df[['nearest_query', 'hinted_latency_bin']].rename(columns={'hinted_latency_bin': 'latency_bin'})

    # # Count the frequency of each nearest_query in each latency bin
    # latency_bin_counts = df_combined.groupby(['latency_bin', 'nearest_query']).size().unstack(fill_value=0)
    
    # Create a new column representing the binned ratios
    df['ratio_bin'] = pd.cut(df['pqo_over_kepler_ratio'], bins=bins)

    # Count the frequency of each nearest_query in each ratio bin
    ratio_bin_counts = df.groupby(['ratio_bin', 'nearest_query']).size().unstack(fill_value=0)

    # Plot settings
    bar_width = 0.05
    positions = np.arange(len(bins) - 1)  # Positions for the bins
    opacity = 0.7

    # Initialize the plot
    fig, ax = plt.subplots(figsize=(10, 6))
    colormaps = ['Set1', 'Set2', 'Set3', 'tab20']
    colors = []

    # Combine the colors from different colormaps to get more than 20 high-contrast colors
    for cmap_name in colormaps:
        cmap = cm.get_cmap(cmap_name)
        colors.extend([cmap(i) for i in range(cmap.N)])  # Add all colors from each colormap

    # Plot the bars for each nearest query ID
    for i, q in enumerate(ratio_bin_counts.columns):
        ax.bar(positions + i * bar_width, ratio_bin_counts[q], bar_width, alpha=opacity, color=colors[i], label=f'Nearest Query {q}')

    # Customize the plot
    ax.set_xlabel('(pqo_latency/kepler_latency) Ratio Bins')
    ax.set_ylabel('Frequency')
    ax.set_title('Histogram of Latency Ratio Grouped by Nearest Query')
    ax.set_xticks(positions + bar_width)
    ax.set_xticklabels([f'{bins[i]}-{bins[i+1]}' for i in range(len(bins)-1)])
    ax.legend()

    # Show the plot
    plt.tight_layout()
    plt.savefig(f"/home/lsh/PARQO_backend/reuse/imdbloadbase/on-demand/{workload}_workload/hint_dist/latency-pqo-q{query}-t{template}-{n}-b0.5.png")

def get_hint_dist(hint,n,query,template,workload,db_name='imdbloadbase',anchor=0):
    query_path = f"/home/lsh/PARQO_backend/query/join-order-benchmark/on-demand/{workload}_workload/q{query}-t{template}-{n}-{anchor}-b0.5.sql"
    with open(query_path) as p:
        sql = p.read()
    
    conn = psycopg2.connect(host="/tmp", dbname=db_name, user="lsh")
    conn.set_session(autocommit=True)
    cursor = conn.cursor()

    ### original Postgres's est_card cardinality
    table_name_id_dict, join_maps, join_info, pair_rel_info = get_maps(db_name, sql, debug=False)
    est_base_card, est_join_card_info = ori_cardest(db_name, sql)
    est_join_card = list(est_join_card_info[:, 2])
    est_card = est_base_card + est_join_card

    ### number of rows of base_rel
    raw_base_card = get_raw_table_size(sql, -2, db_name)


    ### raw_join_card: number of rows of left_table * number of rows of right_table
    raw_join_card = [i[2] for i in join_info]
    raw_card = raw_base_card + raw_join_card

    num_of_base_rel = len(raw_base_card)
    num_of_pair_rel = len(pair_rel_info)
    num_of_join_rel = len(raw_join_card)
    all_basic_rels = list(range(num_of_base_rel + num_of_pair_rel)) # basic includes single and pair
    all_rels = list(range(num_of_base_rel + num_of_join_rel)) # all include all

    assert len(est_base_card) == len(raw_base_card)
    assert len(est_join_card) == len(raw_join_card)

    ### selectivity = est_card / raw_card
    est_base_sel = [est_base_card[i]/raw_base_card[i] for i in range(num_of_base_rel)]
    est_join_sel = [est_join_card[i]/raw_join_card[i] for i in range(num_of_join_rel)]

    err_info_dict = {}
    for i in range(num_of_base_rel + num_of_pair_rel):

        cur_err_list, cur_err_hist = prepare_error_data(db_name, f"{query}a", sensi_dim=i, max_sel=1.0, 
                                                        rel_error=True, div=2, debug=False,
                                                        pqo=True, template_id=template, num=n, workload=workload)
        if cur_err_list == [] and cur_err_hist == []: # Don't need to build err profile for this dimension
            err_info_dict[i] = []
            continue
        cur_kde_list = cal_pdf(cur_err_hist, rel_error=True, bandwidth=0.5, naive=False)
        err_info_dict[i] = [cur_err_list, cur_err_hist, cur_kde_list]

    center_err = gen_center_from_err_dist(est_card, raw_card, all_basic_rels, err_info_dict, num_of_samples=1000, debug=False, naive=False)


    def gen_samples_from_joint_err_dist(N, relations, random_seeds=True, naive=False):
        if random_seeds:
            np.random.seed(2023)
        joint_error_samples = []
        for table_id in relations:
            r = find_bin_id_from_err_hist_list(est_card, raw_card, cur_dim=table_id, err_info_dict=err_info_dict)
            pdf_of_err = err_info_dict[table_id][2][r]
            err_sample = pdf_of_err.sample(N)
            joint_error_samples.append(err_sample)

        joint_error_samples = np.array(joint_error_samples).T.tolist()[0]

        return joint_error_samples

    def cal_penalty_at_sample(error, hint, cur_dim, 
                            est_base_sel=est_base_sel, est_join_sel=est_join_sel, 
                            return_penalty_val=True, recentered_error=center_err):
        new_base_sel, new_join_sel = prep_sel(table_name_id_dict, join_maps, join_info, 
                                              est_base_sel, file_of_base_sel, 
                                              est_join_sel, file_of_join_sel, 
                                              error=error, recentered_error=recentered_error,
                                              relation_list=cur_dim, rela_error=True)
        cost_value_with_hint, join_order_with_hint, scan_mtd_with_hint = get_plan_cost(cursor, sql=sql, hint=hint, explain=explain, debug=True)
        cost_value_opt, join_order_opt, scan_mtd_opt = get_plan_cost(cursor, sql=sql, explain=explain, debug=True)
        if return_penalty_val:
            return max(cost_value_with_hint - cost_value_opt, 0)
        else:
            return cost_value_with_hint, cost_value_opt, new_base_sel, new_join_sel

    sensitive_rels = sen_dict[workload+"-od-"+f"q{query}-t{template}-{n}-{anchor}-b0.5"] # read from file
    # sensitive_rels = sorted(sen_dict[workload+"-od-"+f"q{query}-t{template}-{n}-{anchor}-b0.5"]) # read from file

    write_to_file(est_base_sel, file_of_base_sel)
    write_to_file(est_join_sel, file_of_join_sel)
    write_pointers_to_file(list(range(num_of_base_rel + num_of_join_rel)))

    cost_dist = []
    cost_ori_dist = []
    # print("sen d: ", sensitive_rels)
    joint_err_samples = gen_samples_from_joint_err_dist(50, relations=sensitive_rels, random_seeds=True, naive=False)
    # print(joint_err_samples[:10])
    # input()

    # exp_penalty_by_samples(cur_plan_list, sensitive_rels, joint_err_samples, tolerance=tolerance, save_samples=True)
    
    # print(f"Got {len(joint_err_samples)} samples")
    for error in joint_err_samples:
        cost_value_with_hint, cost_value_opt, new_sel_base, new_sel_join = cal_penalty_at_sample(error=error, hint=hint,
                                                                                                cur_dim=sensitive_rels, return_penalty_val=False,       
                                                                                                est_base_sel=est_base_sel, est_join_sel=est_join_sel)
        cost_dist.append(cost_value_with_hint)
        cost_ori_dist.append(cost_value_opt)
    return cost_dist, cost_ori_dist, joint_err_samples

def get_all_hint_dist(workload,n,query,template,robust_only=True):
    hint_dist = {}
    # Load Kepler plan cover
    if 1:
        query_template = f"{query}-{template}"
        kepler_plans_folder = f"/home/lsh/test_kepler/kepler/imdb_repo_history/imdb_{query_template}_original/"
        kepler_plan_path = f"{kepler_plans_folder}{workload}/outputs/hints/{query_template}/training_{n}/imdbloadbase/imdb_plans.json"
        kepler_plan_cover_path = f"{kepler_plans_folder}{workload}/outputs/results/{query_template}/training_{n}/execution_output/imdbloadbase_{query_template}_metadata.json"
        
        # Filter kepler plans that are actually used
        all_dataframes = []
        for confidence in ["0", "0.2", "0.4", "0.6", "0.8"]:
            kepler_plan_used_path = f"{kepler_plans_folder}{workload}/outputs/evaluation/{query_template}/training_{n}/confidence_{confidence}/predictions/{query_template}_latency_comparison.csv"
            df = pd.read_csv(kepler_plan_used_path)
            all_dataframes.append(df)
        concatenated_df = pd.concat(all_dataframes, ignore_index=True)
        kepler_plan_used = concatenated_df['plan_id'].unique()
        # print(kepler_plan_used)

        with open(kepler_plan_path, 'r') as file:
            kepler_plan = json.load(file)

        with open(kepler_plan_cover_path, 'r') as file:
            kepler_plan_cover = json.load(file)
    
    # Load PQO plans per anchor and test kepler's plan with the samples + sen_dim fo this anchor
    if 1:
        meta_file = f"/home/lsh/PARQO_backend/reuse/on-demand/anchors/{workload}_workload/q{query}-t{template}-{n}-optimized-queries-b0.5.json"

        with open(meta_file, 'r') as file:
            pqo_num_query = len(json.load(file))

        pqo_hint_dist = {}
        pqo_hint_ori_dist = {}
        kepler_hint_dist = {}
        kepler_hint_ori_dist = {}
        for i in range(pqo_num_query):
            kepler_per_anchor = {}
            for plan_id in kepler_plan_cover[query_template]["plan_cover"]:
                if plan_id in kepler_plan_used:
                    plan = kepler_plan[query_template][plan_id]["hints"]
                    dist, ori_dist, joint_err_samples = get_hint_dist(plan,n,query,template,workload, anchor=i)
                    kepler_per_anchor[plan_id]=(dist, joint_err_samples)
            kepler_hint_ori_dist[i] = ori_dist
            kepler_hint_dist[i] = kepler_per_anchor
            
            # Load pqo plans
            log_file_path = f"/home/lsh/PARQO_backend/log/on-base/imdbloadbase/on-demand/anchors/{workload}_workload/imdbloadbase_q{query}-t{template}-{n}-{i}-b0.5_0.2.log"

            with open(log_file_path, 'r') as file:
                log_data = file.read()

            pattern = r'INFO:root:### Best plan by exp penalty: \[([\d,\s]+)\]'
            matches = re.findall(pattern, log_data)

            if matches:
                for match in matches:
                    best_plan = list(map(int, match.split(',')))
                    # print(i, best_plan)

            pqo_plans_path = f"/home/lsh/PARQO_backend/plan/on-base/imdbloadbase/pqo-entire-space/on-demand/{workload}_workload/on-basetmp_plan_dict_imdbloadbase_q{query}-t{template}-{n}-{i}-b0.5.txt"
            with open(pqo_plans_path, 'r') as file:
                pqo_plans = json.load(file)["0"]
            pqo_plans = list(sorted(set(pqo_plans)))

            if robust_only:
                plans = best_plan
            else:
                plans = range(len(pqo_plans))
            hint_per_anchor = []
            for plan_id in plans:
                plan = pqo_plans[plan_id]
                dist, ori_dist, joint_err_samples = get_hint_dist(plan,n,query,template,workload,anchor=i)
                hint_per_anchor.append((dist, plan_id, plan_id in best_plan, joint_err_samples))

            pqo_hint_dist[i] = hint_per_anchor
            pqo_hint_ori_dist[i] = ori_dist
            print(f"pqo {i+1}/{pqo_num_query} done")
        hint_dist["pqo"] = pqo_hint_dist
        hint_dist["kepler"] = kepler_hint_dist
        hint_dist["kepler_ori"] = kepler_hint_ori_dist
        hint_dist["pqo_ori"] = pqo_hint_ori_dist
        print("all done")
    
    if robust_only:
        hint_dist_path = f"/home/lsh/PARQO_backend/reuse/imdbloadbase/on-demand/{workload}_workload/hint_dist/hint_dist-pqo-q{query}-t{template}-{n}-b0.5_robust.json"
    else:
        hint_dist_path = f"/home/lsh/PARQO_backend/reuse/imdbloadbase/on-demand/{workload}_workload/hint_dist/hint_dist-pqo-q{query}-t{template}-{n}-b0.5_all.json"
    with open(hint_dist_path, 'w') as outfile:
        json.dump(hint_dist, outfile, indent=4)
    return hint_dist_path

def evaluate_hint_dist(path,workload,n,query,template,robust_only=True):
    color_cycle = plt.cm.get_cmap('tab20', 20)
    plt.gca().set_prop_cycle(cycler('color', [color_cycle(i) for i in range(20)]))  # Cycle through 20 colors

    with open(path, 'r') as file:
        data = json.load(file)
    
    with open("cached_info/error_profile_dict.json", "r") as f:
        err_files_dict = json.load(f)
    err_files_dict[f"{query}-{template}"] = {int(k):v for k, v in err_files_dict[f"{query}-{template}"].items()}

    sensitive_rels_name = {k:v.split(".txt")[0] for k,v in err_files_dict[f"{query}-{template}"].items()}
    
    for anchor, dists in data["pqo"].items():
        fig, ax1 = plt.subplots(figsize=(10, 6))
        mean_dist_kepler = 0
        # sensitive_rels = sorted(sen_dict[workload+"-od-"+f"q{query}-t{template}-{n}-{anchor}-b0.5"]) # read from file
        sensitive_rels = sen_dict[workload+"-od-"+f"q{query}-t{template}-{n}-{anchor}-b0.5"] # read from file
        sen_dim_name = sensitive_rels_name[sensitive_rels[0]]
        for plan_id, (dist, joint_err_samples) in data["kepler"][anchor].items():
            # if plan_id == "54":
            if 1:
                mean_dist_kepler = np.mean(dist)
                # sort the costs by most sensitive dim error
                top_sen_dim_err = [e[0] for e in joint_err_samples]
                dist_sorted = sorted(zip(top_sen_dim_err, dist))
                s, c = zip(*dist_sorted)

                ax2 = ax1.twinx()
                sns.kdeplot(x=s, ax=ax2, fill=True, color="lightblue", alpha=0.1, label="Density of Errors")
                ax2.set_yticks([]) 

                ax1.plot(s,c,label=f'kepler-{plan_id}',marker='o',linestyle='--',linewidth=2)
                # plt.plot(dist,label=f'kepler-{plan_id}',marker='o',linestyle='--',linewidth=2)
        min_exp = np.min([np.sum(d[0]) for d in dists])
        for dist, plan_id, isrobust, joint_err_samples in dists:
            # if np.mean(dist) < 5 *mean_dist_kepler:
            if True:
                # print(isrobust)
                top_sen_dim_err = [e[0] for e in joint_err_samples]
                dist_sorted = sorted(zip(top_sen_dim_err, dist))
                s, c = zip(*dist_sorted)
                if isrobust:
                    label = f"pqo-{anchor}-{plan_id}*"
                    if not robust_only:
                        if np.sum(dist) == min_exp:
                            label += "+"
                    ax1.plot(s,c,label=label,linestyle='-',marker='o',linewidth=2)
                    # plt.plot(dist,label=label,marker='o',linestyle='--',linewidth=2)
                else:
                    label = f"pqo-{anchor}-{plan_id}"
                    if not robust_only:
                        if np.sum(dist) == min_exp:
                            label += "+"
                    ax1.plot(s,c,label=label,linestyle='-',marker='o',linewidth=1)
                    # plt.plot(dist,label=label,marker='o',linestyle='--',linewidth=2)
        # plt.plot(data["kepler_ori"][anchor],label="kepler_ori",linestyle='dotted',linewidth=5)
        # plt.plot(data["pqo_ori"][anchor],label="pqo_ori",linestyle='dotted',linewidth=3)
        plt.xlim(min(s)*1.1, max(s)*1.1)
        ax1.set_ylabel("Cost")
        ax2.set_ylabel("")
        ax1.set_xlabel(f"Log-relative Selectivity Error\n Dimension: {sen_dim_name}")
        plt.title(f"Plan Cost Distribution on PQO Anchor {query}-{template}-{workload}-{n}-{anchor}")
        ax1.legend(loc="upper left")
        ax2.legend(loc="upper right")
        # plt.legend()
        if robust_only:
            plt.savefig(path.replace("_robust.json", f"-{anchor}_robust.png"))
        else:
            plt.savefig(path.replace("_all.json", f"-{anchor}_all.png"))

if __name__ == "__main__":
    # workload = 'csv'
    # n = 50
    query = 7
    template = 0

    for workload in ['csv', 'kepler', 'cardinality']:
        for n in [50, 400]:
            robust_only=False
            path = get_all_hint_dist(workload,n,query,template,robust_only=robust_only)
            # path = "/home/lsh/PARQO_backend/reuse/imdbloadbase/on-demand/kepler_workload/hint_dist/hint_dist-pqo-q7-t1-400-b0.5.json"
            evaluate_hint_dist(path,workload,n,query,template,robust_only=robust_only)
    #         pass
    
    # for workload in ['csv', 'kepler', 'cardinality']:
    #     for n in [50, 400]:
    #         get_latency_detail(workload,n,query,template)
    #         plot_latency_hist(workload,n,query,template)
    
    # check_common(workload,n,query,template)