import json
import csv
from prep_cardinality import *
from prep_error_list import *
from prep_selectivity import *
import pandas as pd

explain = "EXPLAIN (SUMMARY, COSTS, FORMAT JSON)"
file_of_base_sel = './cardinality/new_single.txt'  # file to be sent to pg folder, contains cardinality for base_rel
file_of_join_sel = './cardinality/join.txt'  # file to be sent to pg folder, contains cardinality for join_rel


class PQOMethod():

    def __init__(self, db_name, workload_name, query_id, template_id, n_in_name, debug, tolerance=0.2, b=0.5, mixture_test=False, rob_verify=None, ins_id=None):
        self.debug = debug
        
        self.workload_name, self.db_name = workload_name, db_name
        self.tolerance, self.b, self.n = tolerance, b, n_in_name
        self.query_id, self.template_id  = query_id, template_id
        
        self.cursor = None
        self.join_map, self.join_info = None, None
        
        self.queries_train, self.queries_test = [], [] # training/testing parametric queries
        
        self.dimension_space = [] # all querylets, e.g. 7-0: [0, 2, 3, 5, 6, 8], see error_profile_dict
        self.all_dim = [] # all dimensions, e.g. 7-0: [0, 1, ..., 90]
        
        self.raw_base_card = [] # size of tables in current query template (should only be initialized once)
        self.err_info_dict = {} # contain error profile for each dimension in dimension_space
        self.output_result = [] # [sql_id, para_sql, cur_plan_list[rob_plan_id]] for performance test

        self.initParameterizedQueries(mixture_test=mixture_test, rob_verify=rob_verify, ins_id=ins_id)
        self.initRawBaseCard()
        self.initDimensions()
        self.initErrInfoDict(rob_verify, ins_id)
        self.connectToPG()

    '''
    Load training and testing query workload
    '''
    def initParameterizedQueries(self, mixture_test=False, rob_verify=None, ins_id=None):
        if self.db_name == 'imdbloadbase':
            if not rob_verify:
                train_file = f"./data/imdb-new/{self.query_id}-{self.template_id}_{self.workload_name}/raw_data/{self.query_id}-{self.template_id}_training_{self.n}.json"
                if not mixture_test:
                    test_file = f"./data/imdb-new/{self.query_id}-{self.template_id}_{self.workload_name}/raw_data/{self.query_id}-{self.template_id}_testing.json"
                else:
                    print("== Loading mixture testing queries...")
                    test_file = f"./reuse/imdbloadbase/kepler/mixture_test/{self.query_id}-{self.template_id}_workload_{self.workload_name}_training_size_{self.n}.csv"
                    pqo_data = pd.read_csv(test_file)
                    self.queries_test = pqo_data['query'].tolist()
            else:
                assert ins_id != None, "Instance id should not be none"
                train_file = f"./data/imdb-robustness/{rob_verify}/db_instance_{ins_id}/{self.query_id}-{self.template_id}_{self.workload_name}/raw_data/{self.query_id}-{self.template_id}_training_{self.n}.json"
                test_file = f"./data/imdb-robustness/{rob_verify}/db_instance_{ins_id}/{self.query_id}-{self.template_id}_{self.workload_name}/raw_data/{self.query_id}-{self.template_id}_testing.json"
        else:
            train_file = f"./data/dsb-new/{self.query_id}_{self.workload_name}/raw_data/{self.query_id}_training_{self.n}.json"
            test_file = f"./data/dsb-new/{self.query_id}_{self.workload_name}/raw_data/{self.query_id}_testing_200.json"

            if not mixture_test:
                if self.workload_name == 'gaussian':
                    test_file = f"./data/dsb-new/{self.query_id}_{self.workload_name}/raw_data/{self.query_id}_testing_200.json"
                else:
                    test_file = f"./data/dsb-new/{self.query_id}_{self.workload_name}/raw_data/{self.query_id}_testing.json"
            else:
                print("== Loading mixture testing queries...")
                test_file = f"./reuse/dsb/kepler/mixture_test/{self.query_id}_workload_{self.workload_name}_training_size_{self.n}.csv"
                pqo_data = pd.read_csv(test_file)
                self.queries_test = pqo_data['query'].tolist()
                    
        with open(train_file, 'r') as file:
            train_template = json.load(file)
            self.queries_train = list(train_template.values()) # current workload (list of parameterized sqls)

        if not mixture_test:
            with open(test_file, 'r') as file:
                test_template = json.load(file)
                self.queries_test = list(test_template.values()) # current workload (list of parameterized sqls)
            
        
    '''
    Init size of base tables (raw cardinality -- only call this once, since table sizes won't change)
    '''
    def initRawBaseCard(self):
        sql = self.queries_train[0]
        _, self.join_map, self.join_info, self.pair_rel_info = get_maps(self.db_name, sql, debug=self.debug)
        _, _ = ori_cardest(self.db_name, sql)
        raw_join_card = [i[2] for i in self.join_info]
        self.raw_base_card = get_raw_table_size(sql, -2, self.db_name) # raw cardinality of all base tables
        self.all_dim = list(range(len(self.raw_base_card) + len(raw_join_card)))

    '''
    Dimensions (sub-queries/querylets) we are considering. In this pqo method, we consider all querylets upto 2-d.
    '''
    def initDimensions(self):
        
        if self.db_name == "imdbloadbase":
            with open("cached_info/error_profile_dict.json", "r") as f:
                err_files_dict = json.load(f)
            self.dimension_space = list(err_files_dict[f"{self.query_id}-{self.template_id}"].keys())
        elif self.db_name == "dsb":
            with open("cached_info/error_profile_dict_dsb.json", "r") as f:
                err_files_dict = json.load(f)
            self.dimension_space = list(err_files_dict[f"{self.query_id}"].keys())
            

    def connectToPG(self):
        conn = psycopg2.connect(host="/tmp", dbname=self.db_name, user="hx68")
        conn.set_session(autocommit=True)
        self.cursor = conn.cursor()

    '''
    Init err_info_dict, which stores the error distribution for each dimension; Only call this once
    '''
    def initErrInfoDict(self, rob_verify, ins_id):
        err_info_dict = {}
        for i in range(len(self.raw_base_card) + len(self.pair_rel_info)):

            cur_err_list, cur_err_hist = prepare_error_data(self.db_name, self.query_id, sensi_dim=i, max_sel=1.0, 
                                                            div=2, debug=self.debug, rel_error=True, 
                                                            pqo=True, template_id=self.template_id, num=self.n, workload=self.workload_name, 
                                                            rob_verify=rob_verify, ins_id=ins_id)

            if cur_err_list == [] and cur_err_hist == []: # Don't need to build err profile for this dimension
                err_info_dict[i] = []
                continue
            cur_kde_list = cal_pdf(cur_err_hist, bandwidth=self.b)
            err_info_dict[i] = [cur_err_list, cur_err_hist, cur_kde_list]
        self.err_info_dict = err_info_dict

    '''
    Preprocess a new parameterized query:
    -- Get estimated cardinality, selectivity and raw cardinality
    '''
    def preProcessQuery(self, para_sql):
        para_table_name_id_dict, para_join_maps, para_join_info, para_pair_rel_info = get_maps(self.db_name, para_sql, debug=self.debug)
        para_est_base_card, para_est_join_card_info = ori_cardest(self.db_name, para_sql)
        para_est_join_card = list(para_est_join_card_info[:, 2])
        para_est_card = para_est_base_card + para_est_join_card


        ### raw_join_card: number of rows of left_table * number of rows of right_table
        para_raw_join_card = [i[2] for i in para_join_info]
        para_raw_card = self.raw_base_card + para_raw_join_card
        para_est_sel_base = [para_est_base_card[i]/self.raw_base_card[i] for i in range(len(para_est_base_card))]
        para_est_sel_join = [para_est_join_card[i]/para_raw_join_card[i] for i in range(len(para_est_join_card))]
        para_est_sel = [para_est_card[i]/para_raw_card[i] for i in range(len(para_est_card))]

        return para_est_card, para_est_sel, para_raw_card, para_join_maps, para_join_info, para_est_sel_base, para_est_sel_join

    def writePlanToFile(self):
        with open(f'reuse/{self.db_name}/on-demand/{self.workload_name}_workload/new-pqo-q{self.query_id}-t{self.template_id}-{self.n}-b{self.b}.csv', mode='w', newline='') as file:
            writer = csv.writer(file)
            
            # Optionally, write the header row
            writer.writerow(['id', 'query', 'plan'])
            
            # Write each inner list to the file as a row
            self.output_result = [(item[0], item[1].replace('\n', ' '), item[2].replace('\n', ' ')) for item in self.output_result]
            writer.writerows(self.output_result)

def sendAllSelToPG(base_output_sel, f_base_sel, join_output_sel, f_join_sel, changed_relation_list):
    write_to_file(base_output_sel, f_base_sel)
    write_pointers_to_file(changed_relation_list)
    write_to_file(join_output_sel, f_join_sel)


def convertErrToSel(error_samples, dimension_space, est_sel, num_of_base_rel):
    base_sel_samples = []
    join_sel_samples = []
    for error in error_samples:
        base_sel = []
        join_sel = []
        for i in range(len(error)):
            table_id = dimension_space[i]
            table_id = int(table_id)
            cur_e = error[i]
            new_sel = cal_new_sel_by_err(cur_e, est_sel[table_id])
            # print(table_id, cur_e, new_sel, est_sel[table_id])
            # print(f"table_id {table_id}, num_of_base {num_of_base_rel}")
            if table_id < num_of_base_rel:
                base_sel.append(new_sel)
            else:
                join_sel.append(new_sel)  
        # input() 
        base_sel_samples.append(base_sel)
        join_sel_samples.append(join_sel)

    return base_sel_samples, join_sel_samples


def gen_samples_from_joint_err_dist(N, relations, est_card, raw_card, err_info_dict, random_seeds=True, naive=False):
    if random_seeds:
        np.random.seed(2023)
    joint_error_samples = []
    for table_id in relations:
        table_id = int(table_id)
        r = find_bin_id_from_err_hist_list(est_card, raw_card, cur_dim=table_id, err_info_dict=err_info_dict)
        pdf_of_err = err_info_dict[table_id][2][r]
        err_sample = pdf_of_err.sample(N)
        joint_error_samples.append(err_sample)
    joint_error_samples = np.array(joint_error_samples).T.tolist()[0]
    return joint_error_samples