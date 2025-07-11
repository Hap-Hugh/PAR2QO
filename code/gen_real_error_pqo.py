
from postgres import *
from psql_explain_decoder import *
from prep_error_list import plot_error, cal_rel_error
from querylets import *

from sklearn.neighbors import KernelDensity
import random
import pandas as pd
import copy
import logging
import itertools
import argparse
import re

# file_name_to_save_real_error = 'mc_ct_both'
kk = '1=1'
cache_right = {}
db = 'imdb'
# global num
# query_id = 7
# t_id = 1



# Enumerate all possible local selection conditions (lsc) in the template
# to generate abs error list in pqo
def gen_real_error(db: str, # name of databse
                   query_id: int, 
                   t_id: int, # id of template for given query_id
                   num: int, # how many samples used 
                   left: str, # the name of left table to load predicates in csv, e.g: mc, mi_idx
                   left_qlet_name: str, # template of left: e.g. mc (mc should have lsc) or mc_full (when mc w/o lsc), see querylet.py 
                   right: str, # the name of right table to load predicates in csv
                   right_qlet_name: str, # template of right: similar as left_qlet_name
                   querylet_name: str, # querylet name, can be found in querylet.py 
                   SINGLE_TABLE_QUERYLET: bool, # true: 1-table querylet; false: 2-table querylet
                   workload: str, # the workload name (to identify the data and save_to): 'csv', 'kepler', 'cardinality'
                   base_path=None,
                   split=None,
                   instance=None
                   ):
    """
    For example: to generate mc_ct_both-q1-t1-20.txt
    q1: query_id; t1: t_id; 20: num
    querylet_name: mc_ct_both; 
    left: mc; left_qlet_name: mc;
    right: cn; right_qlet_name: cn;
    SINGLE_TABLE_QUERYLET = False

    For example to generate ct-q1-t1-10.txt
    q1: query_id; t1: t_id; 10: num
    left = 'x' --> since we only have 1 table in this template
    left_qlet_name = ''
    right = 'ct'
    right_qlet_name = ''
    querylet_name = 'template_ct'
    SINGLE_TABLE_QUERYLET = True

    
    ---Other examples to set querylet---
    #### Example 1: mc_ct_both in 1a 
    left = 'mc' 
    left_qlet_name = 'mc' # left template name
    right = 'ct'
    right_qlet_name = 'ct' # right template name
    querylet_name = f'template_mc_ct_both'
    SINGLE_TABLE_QUERYLET = False

    #### Example 2: mi_idx_it_r in 1a !! Note, mi_idx is a table's name
    left = 'x' #  --> since we only have 1 table in this template
    left_qlet_name = 'mi_idx_full' 
    right = 'it_miidx' !! Note we use it_miidx to make sure the predicate on it can join miidx with results
    right_qlet_name = 'it' 
    querylet_name = f'template_mi_idx_it_r' 
    SINGLE_TABLE_QUERYLET = False


    #### Example 3: mi_idx_mc_r in 1a 
    left = 'x'  --> since we only have 1 table in this template
    left_qlet_name = 'mi_idx_full' 
    right = "mc" 
    right_qlet_name = 'mc' 
    querylet_name = f'template_mi_idx_mc_r' 
    SINGLE_TABLE_QUERYLET = False


    #### Example 4: t_mi_idx__it in 1a 
    left = 'x' 
    left_qlet_name = 't_full' 
    right = 'it_miidx' !! Note we use it_miidx to make sure the predicate on it can join miidx with results
    right_qlet_name = 'mi_idx_it_r' # template name of right that joining it with mi_idx
    querylet_name = f'template_t_mi_idx__it' 
    SINGLE_TABLE_QUERYLET = False

    #### Example 5: n_ci_l 
    left = 'x'
    left_qlet_name = 'ci_full'
    right = 'n'
    right_qlet_name = 'n'
    querylet_name = 'n_ci_l'
    SINGLE_TABLE_QUERYLET = False

    #### Example 5: n_ci_pure Note: after generate .txt, change 1 row to 2 rows
    left = 'x'
    left_qlet_name = 'ci_full'
    right = 'x'
    right_qlet_name = 'n_full'
    querylet_name = 'n_ci_pure'
    SINGLE_TABLE_QUERYLET = False
    """
    global file_name_to_save_real_error
    if split is not None:
        split_name_dict = {"category":"cat", "random": "random", "sliding": "sampled"}
        instance_name_dict = {"db_instance_1": "1", "db_instance_4": "4"}
    if db == 'imdb':
        if base_path is None:
            base_path = "./data/imdb-new/"
        path = f'{base_path}{query_id}-{t_id}_{workload}/sample-{num}.csv'
        print(path)
        local_selections = pd.read_csv(path, encoding='ISO-8859-1')
        # ['Table', 'Condition', 'Frequency']

        tables = local_selections['Table'].dropna().unique()
        condition_dict = {}
        for i in tables:
            condition_dict[i] = []
        # condition_dict = {'k': [], 't': [], 'cn': [], 'n': [], 'mc': [], 'mi': [], 'it_pi': [], 'it_mi': [], 'it_miidx': [], 'an': [], 'lt': [], 'pi': [], 'ci':[], 'mi_idx':[], 'kt':[], 'ct':[], 'rt':[], 'cct':[], 'chn':[]}
    
    if db == 'dsb':
        local_selections = pd.read_csv('lsc/dsb/DSB-072.csv')
        table_names_from_csv = local_selections['Table'].unique()

        condition_dict = {key: [] for key in table_names_from_csv}
        # condition_dict = {
        #     "call_center": [], "catalog_returns": [],
        #     "catalog_sales": [], "customer": [], 
        #     "customer_address": [], "customer_demographics": [],
        #     "date_dim": [],
        #     "household_demographics": [],
        #     "income_band": [], "item": [],
        #     "ship_mode": [], "store": [], "store_sales": [],
        #     "warehouse": [], "web_sales": [], 
        #     }
        
    if db == 'stats': 
        local_selections = pd.read_csv('lsc/stats/LSC-Stats.csv')
        condition_dict = {'b': [], 'c': [], 'u': [], 'ph': [], 'p': [], 'pl': [], 'v': []}

    local_selections_grouped = local_selections.groupby('Table')
    frequency_dict = copy.deepcopy(condition_dict)
    frequency_dict['x'] = [1]
    

    for table in condition_dict.keys():
        for _, row in local_selections_grouped.get_group(table).iterrows():
            if row['Condition'] == '1=1':
                continue
            condition_dict[table].append(row['Condition'])
            frequency_dict[table].append(int(row['Frequency']))
    condition_dict['x'] = ['1=1']
    

    if db == 'imdb': 
        frequency_dict['mk'] =[1]
        condition_dict['mk'] = ['1=1']
        frequency_dict['akat'] =[1]
        condition_dict['akat'] = ['1=1']
        frequency_dict['cc'] =[1]
        condition_dict['cc'] = ['1=1']

        if query_id in [11, 21, 27]:
            frequency_dict['mc'] =[1]
            condition_dict['mc'] = ['mc.note IS NULL']

        data_list = []
        print(condition_dict)
        # input()
        while True:
            if len(condition_dict[left]) > 50:
                left_conditions = random.sample(condition_dict[left], 50)
            else:
                left_conditions = condition_dict[left]
            if len(condition_dict[right]) > 50:
                right_conditions = random.sample(condition_dict[right], 50)
            else:
                right_conditions = condition_dict[right]

            combinations = list(itertools.product(enumerate(left_conditions), enumerate(right_conditions)))[:100]
            random.shuffle(combinations)

            for (id_1, left_condition), (id_2, right_condition) in combinations:
            # for id_1, left_condition in enumerate(left_conditions):
            #     for id_2, right_condition in enumerate(right_conditions):
                # right_condition = "k.keyword ='character-name-in-title'"
                template = querylet(db, left_condition, right_condition, querylet_name)
                print(left_condition, right_condition, querylet_name)
                # print(template)

                if not SINGLE_TABLE_QUERYLET:
                    left_template = querylet(db, right_condition, left_condition, 'template_'+left_qlet_name)
                    right_template = querylet(db, left_condition, right_condition, 'template_'+right_qlet_name)
                    if split is not None:
                        left_template = querylet(db, right_condition, left_condition, 'template_'+left_qlet_name, split=split_name_dict[split], instance=instance_name_dict[instance])
                        right_template = querylet(db, left_condition, right_condition, 'template_'+right_qlet_name, split=split_name_dict[split], instance=instance_name_dict[instance])
                    print(left_template, right_template)
                    print(template)
                    data = cal_join_selectivity(template, left_template, right_template, id_2)

                if SINGLE_TABLE_QUERYLET:
                    template_full = querylet(db, left_condition, right_condition, querylet_name.replace("1","").replace("2","") + '_full')
                    if split is not None:
                        template_full = querylet(db, left_condition, right_condition, querylet_name.replace("1","").replace("2","") + '_full', split=split_name_dict[split], instance=instance_name_dict[instance])
                    print(querylet_name)
                    print(template_full)
                    print(template)
                    data = cal_local_selectivity(template, template_full)

                file_name_to_save_real_error = querylet_name.split('template_')[1]+'-q'+str(query_id)+'-t'+str(t_id)

                # print(template, template_full, file_name_to_save_real_error)

                # print(len(data), len(data_list))
                if data:
                    print(data, cal_rel_error(data[0], data[1]), math.log(data[1] / data[0]))
                    data_list.extend([data]*frequency_dict[right][id_2]*frequency_dict[left][id_1])
                    print("debug: ", len(data), len(data_list))

            if len(data_list) > 0:
                break
        output = [str(data[0]) +" "+ str(data[1]) for data in data_list]
        print(output)
        # input()
        save_to = f'{base_path}{query_id}-{t_id}_{workload}/error_profile/{file_name_to_save_real_error}-{num}.txt'
        with open(save_to, 'w') as fp:
            fp.write('\n'.join(output))
        plot_pdf(query_id, save_to)
        return

        for right in condition_dict.keys():
            for left in condition_dict.keys():
                for l_r_b in ['l', 'r', 'both']:
                    data_list = []
                    right = right
                    querylet_name = f'{left}_{right}_{l_r_b}'
                    
                    # fix 'it'
                    # if left.split('_')[0] == 'it' and left.split('_')[1] == right:
                    #     querylet_name = f'{left}_{l_r_b}'
                    # elif right.split('_')[0] == 'it' and right.split('_')[1] == left:
                    #     querylet_name = f'{right.split('_')[1]}_{right.split('_')[0]}_{l_r_b}'
                    # else:
                    #     continue

                    # if querylet_name.split('_')[0] == 'it' or querylet_name.split('_')[1] == 'it':
                    #     print(querylet_name)
                    #     input()
                    # else:
                    #     continue

                    # don't care those redundent template; only care simplest ones now 
                    if not check_tempalte(db, 'template_'+querylet_name):
                        continue
                    if os.path.exists(f'./data/abs-error-imdb/{querylet_name}.txt'):
                        print("yes")
                        continue
                    else:
                        print("No", querylet_name)
                        input()
                        # continue
                    for id_1, left_condition in enumerate(condition_dict[left]):
                        for id_2, right_condition in enumerate(condition_dict[right]):

                            if l_r_b == 'l':
                                template = querylet(db, right_condition, left_condition, 'template_'+querylet_name)
                            else:
                                template = querylet(db, left_condition, right_condition, 'template_'+querylet_name)
                            
                            if l_r_b == 'l':
                                right_qlet_name = right+'_full'
                                freq_right = 'x'
                            else:
                                right_qlet_name = right
                                freq_right = right
                            if l_r_b == 'r':
                                left_qlet_name = left + '_full'
                                freq_left = 'x'
                            else:
                                left_qlet_name = left
                                freq_left = left
                            left_template = querylet(db, right_condition, left_condition, 'template_'+left_qlet_name)
                            right_template = querylet(db, left_condition, right_condition, 'template_'+right_qlet_name)
                            # template_full = querylet(db, left_condition, right_condition, 'template_'+querylet_name + '_full')
                            print(template)
                            # input()
                            print(left_template)
                            print(right_template)
                            
                            
                            file_name_to_save_real_error = querylet_name
                            # data = cal_local_selectivity(template, template_full)
                            # input() # WARNING
                            data = cal_join_selectivity(template, left_template, right_template, id_2)
                            
                            if data:
                                print(data, cal_rel_error(data[0], data[1]), math.log(data[1] / data[0]))
                                data_list.extend([data]*frequency_dict[freq_right][id_2]*frequency_dict[freq_left][id_1])
                            if l_r_b == 'l':
                                break # since we don't need to go through right conditions
                        if l_r_b == 'r':
                            break # since we don't need to go through left conditions
                    output = [str(data[0]) +" "+ str(data[1]) for data in data_list]
                    # print(output)
                    input()
                    with open('./data/abs-error-'+db+'/' + file_name_to_save_real_error + '.txt', 'w') as fp:
                        fp.write('\n'.join(output))
                    plot_pdf()

    if db == 'stats':

        data_list = []
        left = 'x'
        left_qlet_name = 'b_full'
        right = 'c'
        right_qlet_name = 'c_ph_l'
        querylet_name = f'template_ph_b__c'

        

        for id_1, left_condition in enumerate(condition_dict[left]):
            for id_2, right_condition in enumerate(condition_dict[right]):
                template = stats_complex_querylet(cc=right_condition, template=querylet_name)
                left_template = stats_single_querylet(left_condition, left_qlet_name)
                right_template = stats_join_querylet(left_alias='c', right_alias='ph', l_r_b='l', 
                                                     cc=right_condition, kk=left_condition)

                print(template)
                file_name_to_save_real_error = 'ph_b__c'

                # data = cal_local_selectivity(template, template_full)
                # input() # WARNING
                data = cal_join_selectivity(template, left_template, right_template, id_2)
                
                if data:
                    print(data, cal_rel_error(data[0], data[1]), math.log(data[1] / data[0]))
                    data_list.extend([data]*frequency_dict[right][id_2]*frequency_dict[left][id_1])
        output = [str(data[0]) +" "+ str(data[1]) for data in data_list]
        # print(output)
        input()
        with open('./data/abs-error-'+db+'/' + file_name_to_save_real_error + '.txt', 'w') as fp:
            fp.write('\n'.join(output))
        plot_pdf()
        exit()
    
        # for l_id, left in enumerate(['p', 'c', 'ph', 'pl', 'v']):

        #     for r_id, right in enumerate(['p', 'c', 'ph', 'pl', 'v']):
        # for l_id, left in enumerate(['u', 'c', 'b', 'v']):

        #     for r_id, right in enumerate(['u', 'c', 'b', 'v']):
        for l_id, left in enumerate(['ph']):

            for r_id, right in enumerate(['b']):
                # if r_id <= l_id: continue
            
                for l_r_b in ['both']:

                    data_list = []

                    # right = 'income_band'
                    querylet_name = f'{left}_{right}_{l_r_b}'
                    template_id = '_2'

                    for id_1, left_condition in enumerate(random.sample(condition_dict[left], 10)):
                        for id_2, right_condition in enumerate(random.sample(condition_dict[right], 10)):

                            file_name_to_save_real_error = querylet_name + template_id

                            
                            template = stats_join_querylet(left, right, l_r_b, left_condition, right_condition)
                            if l_r_b == 'l':
                                right_qlet_name = right+'_full'
                                freq_right = 'x'
                            else:
                                right_qlet_name = right
                                freq_right = right
                            if l_r_b == 'r':
                                left_qlet_name = left + '_full'
                                freq_left = 'x'
                            else:
                                left_qlet_name = left
                                freq_left = left
                            left_template = stats_single_querylet(left_condition, left_qlet_name)
                            right_template = stats_single_querylet(right_condition, right_qlet_name)
                            print(template)
                            print(left_template)
                            print(right_template)
                            
                            # data = cal_local_selectivity(template, template_full)
                            # input() # WARNING
                            data = cal_join_selectivity(template, left_template, right_template, id_2)
                            
                            if data:
                                print(data, cal_rel_error(data[0], data[1]), math.log(data[1] / data[0]))
                                data_list.extend([data]*frequency_dict[freq_right][id_2]*frequency_dict[freq_left][id_1])
                                                        
                            # input()
                            if l_r_b == 'l':
                                break # since we don't need to go through right conditions
                        if l_r_b == 'r':
                            break # since we don't need to go through left conditions

                    output = [str(data[0]) +" "+ str(data[1]) for data in data_list]
                    # print(output)
                    # input()
                    with open('./data/abs-error-'+db+'/' + file_name_to_save_real_error + '.txt', 'w') as fp:
                        fp.write('\n'.join(output))
                    plot_pdf()
        
    if db == 'dsb':
        data_list = []
        
        left = 'x'
        left_qlet_name = 'warehouse_full'
        
        right = 'catalog_sales'
        right_qlet_name = 'inventory_catalog_sales_r'
        
        querylet_name = f'inventory_warehouse__catalog_sales'
        query_let_type_is_join = True
        file_name_to_save_real_error = 'inventory_warehouse__catalog_sales_072'
        for id_1, left_condition in enumerate(condition_dict[left]):
            for id_2, right_condition in enumerate(condition_dict[right]):
                
                # left_condition = left_condition.replace("s2.", "")
                right_condition = right_condition.replace("d1.", "")

                if query_let_type_is_join:
                    template = querylet(db, left_condition, right_condition, 'template_'+ querylet_name)
                    left_template = querylet(db, right_condition, left_condition.replace('d1.', ''), 'template_'+left_qlet_name)
                    right_template = querylet(db, left_condition, right_condition.replace('d2.', ''), 'template_'+right_qlet_name)
                
                    print(template)
                    
                    data = cal_join_selectivity(template, left_template, right_template, id_2)
                else:
                    template = querylet(db, left_condition, right_condition, 'template_'+ querylet_name)
                    template_full = querylet(db, left_condition, right_condition, 'template_'+ querylet_name + '_full')
                    print(template_full)
                    print(template)
                    
                    data = cal_local_selectivity(template, template_full)
                
                if data:
                    print(data, cal_rel_error(data[0], data[1]), math.log(data[1] / data[0]))
                    data_list.extend([data]*frequency_dict[right][id_2]*frequency_dict[left][id_1])
        output = [str(data[0]) +" "+ str(data[1]) for data in data_list]
        # print(output)
        input()
        with open('./data/abs-error-'+db+'/' + file_name_to_save_real_error + '.txt', 'w') as fp:
            fp.write('\n'.join(output))
        plot_pdf()
        exit()



def cal_join_selectivity(join_template, left_template, right_template, id):
    global cache_right
    est_join_count, act_join_count = get_est_act_count(join_template)
    est_left_count, act_left_count = get_est_act_count(left_template)
    if id not in cache_right.keys():
        est_right_count, act_right_count = get_est_act_count(right_template)
        cache_right[id] = [est_right_count, act_right_count]
    else:
        est_right_count, act_right_count = cache_right[id]
        print("=== Use cached")
    print(f"join rows est: {est_join_count}, act: {act_join_count}")
    print(f"left rows est: {est_left_count},  act {act_left_count}")
    print(f"right rows est: {est_right_count}, act: {act_right_count}")
    if est_left_count == 0 or act_left_count == 0 or est_right_count == 0 or act_right_count == 0 or act_join_count == 0:
        return False
    est_sel_join = max(1, est_join_count) / (est_left_count * est_right_count)
    act_sel_join = max(1, act_join_count) / (act_left_count * act_right_count)
    return [act_sel_join, est_sel_join]


def cal_local_selectivity(local_template, full_table_template):
    est_count, act_count = get_est_act_count(local_template)
    est_count_full, act_count_full = get_est_act_count(full_table_template)
    if act_count_full == 0 or est_count_full == 0:
        return False
    else:
        return [max(1, act_count)/act_count_full, max(1, est_count)/act_count_full]


def get_est_act_count(template):
    if db=='imdb':
        join_plans = get_real_latency('imdbloadbase', template, times=1, return_json=True, limit_time=False, limit_worker=True, drop_buffer=False)
    else:
        join_plans = get_real_latency(db, template, times=1, return_json=True, limit_time=False, limit_worker=True, drop_buffer=False)

    join_plans = join_plans[0][0][0]['Plan']
    node_type = join_plans['Node Type']
    while True:
        if node_type in ['Aggregate', 'Gather', 'Sort', 'Materialize', 'Sort', 'Hash', 'Gather Merge']:
            join_plans = join_plans["Plans"][0]
            node_type = join_plans['Node Type']
        else:
            break
    print(join_plans)
    est_join_count = join_plans['Plan Rows']
    act_join_count = join_plans['Actual Rows']
    return est_join_count, act_join_count




def plot_pdf(query_id=1, txt_file=None):
    if txt_file:
        with open(txt_file, 'r') as fp:
            lines = fp.readlines()
    else:
        with open('./data/abs-error-'+db+'/' + file_name_to_save_real_error + '.txt', 'r') as fp:
            lines = fp.readlines()
    data = [x.strip().split() for x in lines]
    abs_error_list = []
    cleaned_data = []
    for x in data:
        # if float(x[0]) > 0.001:
        #     continue
        # if float(x[0]) > 1 or float(x[1]) > 1 or float(x[0]) < 0 or float(x[1]) < 0:
        #     continue
        # else:
        cleaned_data.append([float(x[0])/134170, float(x[1])/134170])
    # Err = true - est
    abs_err = [float(x[0]) - float(x[1]) for x in cleaned_data]
    abs_err = sorted(abs_err)
    abs_err = np.array(abs_err).reshape(-1, 1)
    print(max(abs_err), "max abs (true-est) error")
    print(min(abs_err), "min abs (true-est) error")
    count_1 = 0
    count_2 = 0
    for i in abs_err:
        if i > 0:
            count_1 += 1
        else:
            count_2 += 1
    print(count_1/(count_1 + count_2), ": true>est ", count_2/(count_1 + count_2), ": true<est")
    kde = KernelDensity(kernel="gaussian", bandwidth=0.3).fit(abs_err)
    # plot_error(abs_err, kde, name="data/abs-error-dsb/cn-mc_abs")
    
    relative_error_list = []
    for x in data:
        if float(x[0]) == 0 or 0 == float(x[1]):
            continue
        relative_error_list.append(-cal_rel_error(float(x[0]), float(x[1])))
    # print(relative_error_list)
    relative_error_list = np.array(relative_error_list).reshape(-1, 1)
    print(max(relative_error_list), "max rel error")
    print(min(relative_error_list), "min rel error")
    kde = KernelDensity(kernel="gaussian", bandwidth=1).fit(relative_error_list)
    plot_error(relative_error_list, kde, rel_error=True, name=txt_file[:-4])
    

def check_tempalte(db, querylet_name):
    if not querylet(db, '', '', querylet_name):
        return False
    else:
        return True

def modify_table_name(inner_table, q=0, outer_table=None):
    '''Given the exact table name in the query template, provides the table information
        for the querylet.
    '''
    if outer_table: # join
        # inner_table, outer_table: table name in querylet
        # inner_table_name, outer_table_name: table name in sample.csv
        if inner_table == "it" or inner_table == "it1" or inner_table == "it2":
            if outer_table == "pi":
                inner_table_name = "it_pi"
            if outer_table == "mi":
                inner_table_name = "it_mi"
            if outer_table in ["mi_idx", "miidx"]:
                inner_table_name = "it_miidx"
            if outer_table in ["mi_idx1", "mi_idx2"]:   # Q33
                inner_table_name = inner_table
            inner_table = "it"
        else:
            inner_table_name = inner_table
            if inner_table[-1] in ["1", "2"]:
                inner_table = inner_table[:-1]
            if inner_table=="miidx":
                inner_table_name = "mi_idx"
                inner_table = "mi_idx"
        if outer_table == "it" or outer_table == "it1" or outer_table == "it2":
            if inner_table == "pi":
                outer_table_name = "it_pi"
            if inner_table == "mi":
                outer_table_name = "it_mi"
            if inner_table in ["mi_idx", "miidx"]:
                outer_table_name = "it_miidx"
            if inner_table in ["mi_idx1", "mi_idx2"]:   # Q33
                outer_table_name = inner_table
            outer_table = "it"
        else:
            outer_table_name = outer_table
            if outer_table[-1] in ["1", "2"]:
                outer_table = outer_table[:-1]
            if outer_table=="miidx":
                outer_table_name = "mi_idx"
                outer_table = "mi_idx"
        return inner_table, inner_table_name, outer_table, outer_table_name
    else:   # single table
        # table: table name in querylet
        # table_name: table name in sample.csv
        table_name = inner_table
        if inner_table == "it" or inner_table == "it1" or inner_table == "it2":
            if q==7:
                table_name = "it_pi"
            if q==13:
                if inner_table == "it":
                    table_name = "it_miidx"
                if inner_table == "it2":
                    table_name = "it_mi"
            if q in [18, 12, 14, 22, 25, 28, 30, 31]:
                if inner_table == "it1":
                    table_name = "it_mi"
                if inner_table == "it2":
                    table_name = "it_miidx"
            if q in [1, 4, 26]:
                table_name = "it_miidx"
            if q in [19, 15, 23, 24, 29]:
                table_name = "it_mi"
            table = "it"
        else:
            table = inner_table
            if table[-1] in ["1", "2"]:
                table = table[:-1]
            if inner_table=="miidx":
                table = "mi_idx"
        return table, table_name

def parse_query(sql,q,t,split=None,instance=None):
    if split is None:
        basic=True
    else:
        basic=False
    conn = psycopg2.connect(host="/tmp", dbname="imdbloadbase", user="lsh")
    conn.set_session(autocommit=True)
    cursor = conn.cursor()
    explain = "EXPLAIN (SUMMARY, COSTS, FORMAT JSON)"
    _ = get_plan_cost(cursor=cursor, sql=sql, explain=explain)
    result_dict = {}
    param_dict = {}

    with open("/nfs/lshh/imdb/single_tbl_est_record.txt", "r") as file:
        log_data = file.read()
    single_table_regex = re.compile(
        r"query: (\d+)\nRELOPTINFO \((\w+)\): rows=\d+ width=\d+\n(?:\s+baserestrictinfo: (.+?)\n)?",re.DOTALL)
    for match in single_table_regex.finditer(log_data):
        dim = int(match.group(1))
        table = match.group(2)
        table, table_name = modify_table_name(table, q=q)

        baserestrictinfo = match.group(3)  # May be None if baserestrictinfo is missing

        # Only add entries with baserestrictinfo
        if baserestrictinfo:
            params = ["x", "", table_name, "", "template_"+table, True]
            result_dict[int(dim)] = table+".txt"
            param_dict[int(dim)] = params

    with open("/nfs/lshh/imdb/join_est_record_job.txt", "r") as file:
        log_data = file.read()
    query_regex = re.compile(r"query: (\d+)\n(.+?)(?=query: |\Z)", re.DOTALL)
    inner_rel_regex = re.compile(r"==================inner_rel======(\d+)============: \nRELOPTINFO \((\w+)\):.+?\n.+?(baserestrictinfo: .+?\n|$)?")
    outer_rel_regex = re.compile(r"==================outer_rel======(\d+)============: \nRELOPTINFO \((\w+)\):.+?\n.+?(baserestrictinfo: .+?\n|$)?")
    for match in query_regex.finditer(log_data):
        dim = match.group(1)
        query_text = match.group(2)
        
        inner_match = inner_rel_regex.search(query_text)
        outer_match = outer_rel_regex.search(query_text)
   
        if inner_match and outer_match:
            inner_rel, inner_table, inner_predicate = inner_match.groups()
            outer_rel, outer_table, outer_predicate = outer_match.groups()
            if int(inner_rel) * int(outer_rel) == 0: break

            inner_has_predicate = inner_predicate is not None
            outer_has_predicate = outer_predicate is not None

            inner_table_name = inner_table
            outer_table_name = outer_table

            inner_table, inner_table_name, outer_table, outer_table_name = modify_table_name(inner_table,outer_table=outer_table)
                
            params = []
            if inner_has_predicate and outer_has_predicate:
                join_type = "both"
                params += [inner_table_name, inner_table, outer_table_name, outer_table]
            elif inner_has_predicate:
                join_type = "l"
                params += [inner_table_name, inner_table, "x", outer_table+"_full"]
            elif outer_has_predicate:
                join_type = "r"
                params += ["x", inner_table+"_full", outer_table_name, outer_table]
            else:
                continue
            
            join_key = f"{inner_table}_{outer_table}_{join_type}"
            if join_key in ["n_an_both", "pi_n_both"]: continue
            params.append("template_"+join_key)
            params.append(False)
            result_dict[int(dim)] = join_key+".txt"
            param_dict[int(dim)] = params
    if basic:
        err_file = "cached_info/error_profile_dict.json"
        param_file = "cached_info/gen_real_error_params.json"
        key = f"{q}-{t}"
    else:
        err_file = "cached_info/error_profile_dict_rob.json"
        param_file = "cached_info/gen_real_error_params_rob.json"
        key = f"{q}-{t}-{split}-{instance}"
    with open(err_file, "r") as f:
        data = json.load(f)
    data[key] = result_dict
    with open(err_file, "w") as f:
        json.dump(data, f, indent=4)
    
    with open(param_file, "r") as f:
        data = json.load(f)
    data[key] = param_dict
    with open(param_file, "w") as f:
        json.dump(data, f, indent=4)
    return data

params = {}
params['7a'] = {
    'an': ['x', '', 'an', '', 'template_an', True],
    'n': ['x', '', 'n', '', 'template_n', True],
    'pi': ['x', '', 'pi', '', 'template_pi', True],
    't': ['x', '', 't', '', 'template_t', True],
    'ci_an_r': ['x', 'ci_full', 'an', 'an', 'template_ci_an_r', False],
    'n_an_both': ['n', 'n', 'an', 'an', 'template_n_an_both', False],
    'pi_an_both': ['pi', 'pi', 'an', 'an', 'template_pi_an_both', False],
    'ml_ci_pure': ['x', 'ml_full', 'x', 'ci_full', 'template_ml_ci_pure', False],
    'n_ci_l': ['x', 'ci_full', 'n', 'n', 'template_n_ci_l', False],
    'pi_ci_l': ['x', 'ci_full', 'pi', 'pi', 'template_pi_ci_l', False],
    't_ci_l': ['x', 'ci_full', 't', 't', 'template_t_ci_l', False],
    'pi_it_both': ['pi', 'pi', 'it_pi', 'it', 'template_pi_it_both', False],
    'ml_lt_r': ['x', 'ml_full', 'lt', 'lt', 'template_ml_lt_r', False],
    't_ml_l_2': ['x', 'ml_full', 't', 't', 'template_t_ml_l_2', False],
    'pi_n_both': ['pi', 'pi', 'n', 'n', 'template_pi_n_both', False],
}

params['2a'] = {
    'cn': ['x', '', 'cn', '', 'template_cn', True],
    'k': ['x', '', 'k', '', 'template_k', True],
    'mc_cn_r': ['x', 'mc_full', 'cn', 'cn', 'template_mc_cn_r', False],
    'mk_k_r': ['x', 'mk_full', 'k', 'k', 'template_mk_k_r', False],
    'mk_mc__cn': ['x', 'mk_full', 'cn', 'mc_cn_r', 'template_mk_mc__cn', False],
    'mk_mc__k': ['x', 'mc_full', 'k', 'mk_k_r', 'template_mk_mc__k', False],
    't_mc__cn': ['x', 't_full', 'cn', 'mc_cn_r', 'template_t_mc__cn', False],
    't_mk__k': ['x', 't_full', 'k', 'mk_k_r', 'template_t_mk__k', False]
}

if __name__ == "__main__":
    # Set up argument parsing
    parser = argparse.ArgumentParser()
    parser.add_argument('--q', type=int, help='query')
    parser.add_argument('--t', type=int, help='template')
    parser.add_argument('--n', type=int, help='Number of samples')
    parser.add_argument('--workload', type=str, help='workload')
    parser.add_argument('--gen_err_profile', action='store_true', help='generate meta info first')
    parser.add_argument('--gen_meta_info', action='store_true', help='generate error profile')
    parser.add_argument('--manual', action='store_true', help='manually generate error profile')
    parser.add_argument('--base_path', default=None, type=str, help='base path to raw data folder, inside should be in the form of q-t_workload')

    args = parser.parse_args()
    if args.q is None:
        t = 1
        q = 7
        num = 50
        workload = 'csv'
    else:
        q = args.q
        t = args.t
        num = args.n
        workload = args.workload
        gen_err_profile = args.gen_err_profile
        gen_meta_info = args.gen_meta_info
        manual = args.manual
        base_path = args.base_path
        basic, split, instance = True, None, None
        if base_path is None:
            base_path = "/home/lsh/PARQO_backend/data/imdb-new/"
        else:
            split = base_path.split("/")[-3]
            instance = base_path.split("/")[-2]
            basic = False
    #/home/lsh/PARQO_backend/data/imdb-robustness/category/db_instance_1/3-0_csv/error_profile/k-q3-t0-50.txt
    log_fname = f"{base_path}{q}-{t}_{workload}/log/error-profile-{q}-{t}.log"
    log_dir = os.path.dirname(log_fname)
    os.makedirs(log_dir, exist_ok=True)
    logging.basicConfig(filename=log_fname, level=logging.INFO)
    if gen_meta_info:
        start = time.time()
        path = f"{base_path}{q}-{t}_{workload}/raw_data/{q}-{t}_training_{num}.json"
        with open(path, "r") as f:
            data = json.load(f)
        sql = list(data.values())[0]
        results = parse_query(sql,q,t,split,instance)
        logging.info(f"Generating error profile meta info: {time.time() - start}")
    if gen_err_profile:
        start = time.time()
        path = f'{base_path}{q}-{t}_{workload}/error_profile/'
        if split is None:
            fn = "cached_info/gen_real_error_params.json"
            key = f'{q}-{t}'
        else:
            fn = "cached_info/gen_real_error_params_rob.json"
            key = f'{q}-{t}-{split}-{instance}'
        with open(fn, "r") as f:
            results = json.load(f)
        for qlet, error_info in results[key].items():
            cache_right = {}
            param_data = error_info
            print(f'Generating error profile for {qlet}')
            print(f'Parameters are {param_data}')
            # Unpack the parameter data into variables
            left, left_qlet_name, right, right_qlet_name, querylet_name, single_table_querylet = param_data
            
            # Call the 'gen_real_error' function with unpacked parameters
            gen_real_error(
                db='imdb',  # Database name
                query_id=q,  # Query ID
                t_id=t,  # Template ID
                num=num,  # Number parameter
                left=left,  # Left parameter from 'params'
                left_qlet_name=left_qlet_name,  # Left Qlet name from 'params'
                right=right,  # Right parameter from 'params'
                right_qlet_name=right_qlet_name,  # Right Qlet name from 'params'
                querylet_name=querylet_name,  # Querylet name from 'params'
                SINGLE_TABLE_QUERYLET=single_table_querylet,  # Boolean value from 'params'
                workload=workload,
                base_path=base_path,
                split=split,
                instance=instance
            )
        logging.info(f"Generating error profile at {path} for sample-{num}.csv: {time.time() - start}")
    if manual:
        with open("cached_info/gen_real_error_params_manual.json", "r") as f:
            results = json.load(f)
        for qlet, error_info in results[f'{q}-{t}'].items():
            cache_right = {}
            param_data = error_info
            print(f'Generating error profile for {qlet}')
            print(f'Parameters are {param_data}')
            # Unpack the parameter data into variables
            left, left_qlet_name, right, right_qlet_name, querylet_name, single_table_querylet = param_data
            
            # Call the 'gen_real_error' function with unpacked parameters
            gen_real_error(
                db='imdb',  # Database name
                query_id=q,  # Query ID
                t_id=t,  # Template ID
                num=num,  # Number parameter
                left=left,  # Left parameter from 'params'
                left_qlet_name=left_qlet_name,  # Left Qlet name from 'params'
                right=right,  # Right parameter from 'params'
                right_qlet_name=right_qlet_name,  # Right Qlet name from 'params'
                querylet_name=querylet_name,  # Querylet name from 'params'
                SINGLE_TABLE_QUERYLET=single_table_querylet,  # Boolean value from 'params'
                workload=workload
            )