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


'''
PQO method, using same initial plan set as diagram, reduce by JS/robustness-cover
at runtime select the plan from the nearest (L2) selectivity sample
''' 
class Diagram_Nearest(Diagram):

    def __init__(self, db_name, workload_name, query_id, template_id, n_in_name, debug, mixture_test, rob_verify=None, ins_id=None, tolerance=0.2, b=0.5):        
        super().__init__(db_name, workload_name, query_id, template_id, n_in_name, debug, mixture_test, rob_verify, ins_id, tolerance=tolerance, b=b)

    def initLogFile(self):
        filename = './log/on-base/' + self.db_name + '/diagram/naive_' + str(self.query_id) + '-' + str(self.template_id) + '_' + self.workload_name + '_workload_b' + str(self.b) +'_N'+str(self.N)+ '.log'
        print("####### Log saved at ", filename)
        logging.basicConfig(filename=filename, level=logging.INFO)


    def evaluate(self, R, exe=True, rob_verify=False, split=None):

        plan_candidate_size = len(self.plan_list)
        # self.planSpaceReductionKL(R=R)
        # Alternatively: 
        if R == 0:
            R = max(10, int(plan_candidate_size/5))
        else:
            R = min(plan_candidate_size, R)
        # self.planSpaceReductionJS(R=R)
        sel_to_plan_dict = self.planSpaceReductionOptRange(R=R)


        # find the robust plan for each query
        result_pqo, result_pg, result_verify_pqo, result_verify_refresh_pg, result_verify_base_pg = [], [], [], [], []
        total_pg, total_pqo, robust_is_better  = 0, 0, 0
        avg_costing_time, avg_planning_time = 0, 0

        for sql_id, para_sql in enumerate(tqdm(self.queries_test[:100], desc=f"{self.query_id}-{self.workload_name}: Testing queries: template {self.query_id}")):
            logging.info(f"Query {sql_id}")
            logging.info(f"{para_sql}")
            # Reweight
            start_1 = time.time()
            para_est_card, est_sel, para_raw_card, _, _, _, _ = super().preProcessQuery(para_sql)
            end_1 = time.time()
            avg_costing_time += (end_1 - start_1)*1000

            cached_sel = [i+j for i, j in zip(self.all_base_features, self.all_join_features)]
            nearest_sel = find_nearest_sample(cached_sel, est_sel)


            robust_plan_id = sel_to_plan_dict[nearest_sel]
            # print(f"{nearest_sel} is the nearest sel sample, and we select plan {robust_plan_id}")
            self.output_result.append([sql_id, para_sql, self.plan_list[robust_plan_id]])
            

            robust_plan_latency = round(get_real_latency(self.db_name, para_sql, hint=self.plan_list[robust_plan_id], times=3, limit_time=200000), 3)

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
    

    