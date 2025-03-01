import logging
from prep_selectivity import *
from kl import *
from parse import *
from pqo_method import *
from plan_reduction_by_similarity import *
from plan_reduction_by_opt_range import * 
from tqdm import tqdm
import json
import os
from diagram import *
import random


'''
New PQO method, see pqoByFeatureCollection()
''' 
class DiagramQueryLogWithSample(Diagram):

    def __init__(self, db_name, workload_name, query_id, template_id, n_in_name, debug, mixture_test, rob_verify=None, ins_id=None, tolerance=0.2, b=0.5):
        
        super().__init__(db_name, workload_name, query_id, template_id, n_in_name, debug, mixture_test, rob_verify=rob_verify, ins_id=ins_id, tolerance=tolerance, b=b)


    def initLogFile(self):
        filename = './log/on-base/' + self.db_name + '/diagram/qlog-sample-' + str(self.query_id) + '-' + str(self.template_id) + '_' + self.workload_name + '_workload_b' + str(self.b) +'_N'+str(self.N)+ '.log'
        print("####### Log saved at ", filename)
        logging.basicConfig(filename=filename, level=logging.INFO)


    def pqoByFeatureCollection(self, N):
        """
        PQO approach:

        1) If a cache exists, simply load it and skip re-collection.
        2) Otherwise:
        - Collect features (selectivity samples) from each representative query.
        - Collect candidate plans at each sample.
        - Compute cost and penalty for each plan at each sample.
        - Reduce plan space by removing similar plans.
        - Compute re-weighted probabilities.
        - Save the final model to the cache.

        Parameters:
            N (int): Number of samples drawn from each query (selectivity distribution).
            R (int): Number of cached plans to retain.
        """
        self.N=N
        self.initLogFile()
        # Construct a filename (same logic used in saveModeltoCache)
        filename = f'./reuse/{self.db_name}/diagram/qlog/qlog-sample-{self.query_id}-{self.template_id}_{self.workload_name}_b{self.b}_N{self.N}.json'

        # 1. If a cache file already exists, load it and skip the steps:
        if os.path.exists(filename):
            print(f"####### Found existing cache file at: {filename}")
            try:
                with open(filename, "r") as f:
                    model_data = json.load(f)
                self.all_base_features = model_data["all_base_features"]
                self.all_join_features = model_data["all_join_features"]
                self.joint_probabilities = model_data["joint_probabilities"]
                self.plan_list = model_data["plan_list"]
                self.penaltyCollection = model_data["penaltyCollection"]
                self.costCollection = model_data["costCollection"]
                print("####### Model successfully loaded from cache. Skipping feature and plan collection.")
            except Exception as e:
                print(f"Error loading cache file: {e}")
            return

        # 2. Otherwise, perform the steps to build the PQO model from scratch:
        # 0. Collect all samples
        self.collectFeatures()
        # 1. Collect plan at each sample
        self.collectPlans()
        # 2. Get "cost" & "penalty" for each plan at each sample
        self.collectPlanCost()
        # self.collectOptCostAndPenalty()
        # 3. Reduce plan space by removing similar plans
        # self.planSpaceReductionKL(R=R)
        # Alternatively: self.planSpaceReductionJS(R=R)

        # 4. Compute the re-weighted probabilities
        # self.calReweightProbability()

        # 5. Finally, save everything to the cache
        self.saveModeltoCache()



    def collectFeatures(self):

        # collect features from basic training query
        for sql_id, para_sql in enumerate(self.queries_train):
            para_est_card, est_sel, para_raw_card, para_join_maps, para_join_info, para_est_sel_base, para_est_sel_join = super().preProcessQuery(para_sql)
            self.all_base_features.append(para_est_sel_base)
            self.all_join_features.append(para_est_sel_join)

        
        min_vals_base, max_vals_base = [], []
        for i in range(len(self.all_base_features[0])):
            column_values = [row[i] for row in self.all_base_features]
            min_vals_base.append(min(column_values))
            max_vals_base.append(max(column_values))

        min_vals_join, max_vals_join = [], []
        for i in range(len(self.all_join_features[0])):
            column_values = [row[i] for row in self.all_join_features]
            min_vals_join.append(min(column_values))
            max_vals_join.append(max(column_values))

        # collect additional random features from selectivity space constrained by workload
        for i in range(300):
            new_list_base = [
                random.uniform(min_vals_base[i], max_vals_base[i])  # random float in [min_i, max_i]
                for i in range(len(self.all_base_features[0]))
            ]
            self.all_base_features.append(new_list_base)                
            new_list_join = [
                random.uniform(min_vals_join[i], max_vals_join[i])  # random float in [min_i, max_i]
                for i in range(len(self.all_join_features[0]))
            ]
            self.all_join_features.append(new_list_join)


    def planSpaceReductionOptRange(self, R):
        centers, sel_to_plan_dict = reduce_by_opt_range(self.costCollection, R)
        self.costCollection = [self.costCollection[i] for i in centers]
        # self.penaltyCollection = [self.penaltyCollection[i] for i in centers]
        self.plan_list = [self.plan_list[i] for i in centers]
        return sel_to_plan_dict


    def evaluate(self, R, exe=True, rob_verify=False, split=None, prune='', bound=False):

        plan_candidate_size = len(self.plan_list)
        if plan_candidate_size == 1:
            return
        # self.planSpaceReductionKL(R=R)
        # Alternatively: 
        if R == 0:
            R = max(10, int(plan_candidate_size/5))
        else:
            R = min(plan_candidate_size, R)
        if prune == 'sim':
            self.planSpaceReductionJS(R=R)
        elif prune == 'rob-range':
            sel_to_plan_dict = self.planSpaceReductionOptRange(R=R)            

        # find the robust plan for each query
        result_pqo, result_pg, result_verify_pqo, result_verify_refresh_pg, result_verify_base_pg = [], [], [], [], []
        total_pg, total_pqo, robust_is_better  = 0, 0, 0
        avg_costing_time, avg_planning_time = 0, 0
        out_bound_count = 0

        if bound:
            for sql_id, para_sql in enumerate(self.queries_test[:100]):
                min_recost = float('inf')
                robust_plan_id = -1
                for id, i in enumerate(self.plan_list):
                    recost_of_i = get_plan_cost_simple(self.cursor, sql=para_sql, hint=i, explain=explain, debug=False)
                    if recost_of_i < min_recost:
                        min_recost = recost_of_i
                        robust_plan_id = id
                
                robust_plan = self.plan_list[robust_plan_id]
                # add the sub-optimality boundary
                if bound:
                    cur_opt_cost, cur_join_order_opt, cur_scan_mtd_opt = get_plan_cost_simple(self.cursor, sql=para_sql, explain=explain, debug=True)
                    if min_recost > 1.2 * cur_opt_cost:
                        out_bound_count += 1
                        robust_plan = gen_final_hint(cur_join_order_opt, cur_scan_mtd_opt)
            print(f"Found {out_bound_count} queries fall back to pg.")
            if out_bound_count == 0 or out_bound_count == 100:
                return

        
        for sql_id, para_sql in enumerate(tqdm(self.queries_test[:100], desc=f"{self.query_id}-{self.workload_name}: Testing queries: template {self.query_id}")):
            logging.info(f"Query {sql_id}")
            logging.info(f"{para_sql}")
            # Reweight
            start_1 = time.time()
            para_est_card, est_sel, para_raw_card, _, _, _, _ = super().preProcessQuery(para_sql)
            end_1 = time.time()
            avg_costing_time += (end_1 - start_1)*1000

            cached_sel = [i+j for i, j in zip(self.all_base_features, self.all_join_features)]
            # nearest_sel = find_nearest_sample(cached_sel, est_sel)


            # robust_plan_id = sel_to_plan_dict[nearest_sel]
            
            # find the plan with the best recost(new query)
            min_recost = float('inf')
            robust_plan_id = -1
            for id, i in enumerate(self.plan_list):
                recost_of_i = get_plan_cost_simple(self.cursor, sql=para_sql, hint=i, explain=explain, debug=False)
                if recost_of_i < min_recost:
                    min_recost = recost_of_i
                    robust_plan_id = id
            
            robust_plan = self.plan_list[robust_plan_id]
            # add the sub-optimality boundary
            if bound:
                cur_opt_cost, cur_join_order_opt, cur_scan_mtd_opt = get_plan_cost_simple(self.cursor, sql=para_sql, explain=explain, debug=True)
                if min_recost > 1.2 * cur_opt_cost:
                    out_bound_count += 1
                    robust_plan = gen_final_hint(cur_join_order_opt, cur_scan_mtd_opt)


            # print(f"{nearest_sel} is the nearest sel sample, and we select plan {robust_plan_id}")
            self.output_result.append([sql_id, para_sql, self.plan_list[robust_plan_id]])
            
            if exe:
            
                if self.debug:
                    print(f"Robust plan is {robust_plan_id}")
                    for i in range(len(self.plan_list)):
                        latency = round(get_real_latency(self.db_name, para_sql, hint=self.plan_list[i], times=5, limit_time=200000), 5)
                        print(f"Cur plan is {i}: {latency} {self.plan_list[i]}")
                        if i == robust_plan_id: robust_plan_latency = latency
                else:
                    robust_plan_latency = round(get_real_latency(self.db_name, para_sql, hint=robust_plan, times=3, limit_time=200000), 3)
                    
                # if self.workload_name == "cardinality":
                if False:
                    pg_plan_latency = round(get_real_latency(self.db_name, para_sql, hint=None, times=3, limit_time=200000), 3)        
                else:
                    pg_plan_latency = 1
                
                result_pqo.append(robust_plan_latency)
                result_pg.append(pg_plan_latency)
                total_pqo += robust_plan_latency
                total_pg += pg_plan_latency
                if (robust_plan_latency < pg_plan_latency): robust_is_better+=1
                
                output_string = f"Q{self.query_id}-{self.workload_name}-{self.b}-{len(self.cluster_weights)}-{len(self.plan_list)}:" \
                    f"Robust plan is {robust_plan_id}: {robust_plan_latency}, Postgres plan: {pg_plan_latency}, {robust_is_better} / {sql_id+1} speedup," \
                    f"avg: pg {round(total_pg/len(result_pg), 3)} / pqo {round(total_pqo/len(result_pqo), 3)} = {round(total_pg / total_pqo, 3)}"
                # print(output_string)
                logging.info(output_string)
        if exe:
            if not rob_verify:
                output_string = f"PG avg: {round(sum(result_pg)/len(result_pg), 3)} ms, PQO avg: {round(sum(result_pqo)/len(result_pqo), 3)} ms, Ratio is {round(sum(result_pg) / sum(result_pqo), 3)}, {robust_is_better} / {len(self.queries_test)} speedup! "
                print(output_string)
                logging.info(output_string)

        logging.info(f"# of samples per cluster: {self.N}, # of clusters {len(self.costCollection[0])/self.N} # of cached plans {len(self.plan_list)}/{plan_candidate_size}")
        # self.writePlanToFile()
        for handler in logging.root.handlers[:]:
            logging.root.removeHandler(handler)
        return
    

    def saveModeltoCache(self):
        """
        Save the relevant learned/cached information to a JSON file for future reuse.
        The dictionary includes:
          - self.all_base_features
          - self.all_join_features
          - self.joint_probabilities
          - self.plan_list
          - self.penaltyCollection
        """
        # Construct a filename similar to how we do with the log files
        filename = f'./reuse/{self.db_name}/diagram/qlog/qlog-sample-{self.query_id}-{self.template_id}_{self.workload_name}_b{self.b}_N{self.N}.json'


        # Ensure directory exists
        os.makedirs(os.path.dirname(filename), exist_ok=True)

        # Create the data dictionary
        model_data = {
            "all_base_features": self.all_base_features,
            "all_join_features": self.all_join_features,
            "joint_probabilities": self.joint_probabilities,
            "plan_list": self.plan_list,
            "costCollection": self.costCollection,
            "penaltyCollection": self.penaltyCollection
        }

        # Dump to JSON file
        with open(filename, "w") as f:
            json.dump(model_data, f)

        logging.info(f"Model cache saved to {filename}")
        print(f"####### Model cache saved to {filename}")
