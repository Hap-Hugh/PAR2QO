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


'''
New PQO method, see pqoByFeatureCollection()
''' 
class Diagram(PQOMethod):

    def __init__(self, db_name, workload_name, query_id, template_id, n_in_name, debug, mixture_test, rob_verify=None, ins_id=None, tolerance=0.2, b=0.5):
        
        super().__init__(db_name, workload_name, query_id, template_id, n_in_name, debug, tolerance, b, mixture_test, rob_verify, ins_id)

        # only for This PQO approach
        self.all_base_features, self.all_join_features = [], []
        self.joint_probabilities = [] # probability of each sel sample being sampled from the overall distribution
        self.probability_of_sampled = [] # probability of each sel sample being sampled from each cluster: (N * cluster) * cluster
        self.clusters = [] # est cardinality and base cardinality of anchors
        self.cluster_weights = [] # weight of each anchor
        self.plan_list = set()
        self.costCollection, self.penaltyCollection = [], []
        self.optCostCollection = []
        self.rob_verify, self.mixture_test, self.ins_id = rob_verify, mixture_test, ins_id


    def initLogFile(self):
        if not self.rob_verify:
            if not self.mixture_test:
                filename = './log/on-base/' + self.db_name + '/diagram/' + str(self.query_id) + '-' + str(self.template_id) + '_' + self.workload_name + '_workload_b' + str(self.b) +'_N'+str(self.N)+ '.log'
            else:
                filename = './log/on-base/' + self.db_name + '/diagram/mixture_' + str(self.query_id) + '-' + str(self.template_id) + '_' + self.workload_name + '_workload_b' + str(self.b) +'_N'+str(self.N)+ '.log'
        else:
            filename = f'./log/{self.rob_verify}/db_instance_{self.ins_id}' + '/diagram/' + str(self.query_id) + '-' + str(self.template_id) + '_' + self.workload_name + '_workload_b' + str(self.b) +'_N'+str(self.N)+ '.log'
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
        # self.initLogFile()
        # Construct a filename (same logic used in saveModeltoCache)
        if not self.rob_verify:
            filename = f'./reuse/{self.db_name}/diagram/{self.query_id}-{self.template_id}_{self.workload_name}_b{self.b}_N{self.N}.json'
        else:
            # For robustness verification runs
            filename = f'./reuse/{self.db_name}/{self.rob_verify}/db_instance_{self.ins_id}/{self.query_id}-{self.template_id}_{self.workload_name}_b{self.b}_N{self.N}.json'

        # 1. If a cache file already exists, load it and skip the steps:
        if os.path.exists(filename):
            try:
                with open(filename, "r") as f:
                    model_data = json.load(f)
                self.all_base_features = model_data["all_base_features"]
                self.all_join_features = model_data["all_join_features"]
                self.joint_probabilities = model_data["joint_probabilities"]
                self.plan_list = model_data["plan_list"]
                self.penaltyCollection = model_data["penaltyCollection"]
                self.costCollection = model_data["costCollection"]
                print(f"####### Model successfully loaded from {filename}. Skipping samples and plan collection.")
            except Exception as e:
                print(f"Error loading cache file: {e}")
            return

        # 2. Otherwise, perform the steps to build the PQO model from scratch:
        
        # 0) Collect all samples
        self.collectFeatures()
        
        # 1) Collect plan at each sample
        self.collectPlans()
        
        # 2) Get "cost" & "penalty" for each plan at each sample
        self.collectPlanCost()
        self.collectOptCostAndPenalty()

        # 3) Compute the re-weighted probabilities
        self.calReweightProbability()

        # 4) Finally, save everything to the cache
        self.saveModeltoCache()


    '''
    Collect features (samples):
    From training query, if this query is not close to any historical query, we sample
    Basically, each sample is a selectivity vector with d=self.dimension_space
    We measure the distance between queries by KL divergence (each query is a conditional selectivity distribution).
    '''
    def collectFeatures(self):
        for sql_id, para_sql in enumerate(self.queries_train):
            para_est_card, est_sel, para_raw_card, para_join_maps, para_join_info, para_est_sel_base, para_est_sel_join = super().preProcessQuery(para_sql)

            nearest_id = -1 # Find the nearest historical sql by calculating KL
            smallest_kl = float('inf')

            for history_id in range(len(self.clusters)):
                # Calculate the KL of sel distribution on querylets' dimensions (not all dimensions of the query)
                tmp_kl = cal_kl(self.err_info_dict, self.clusters[history_id][0], self.clusters[history_id][1],
                        self.err_info_dict, para_est_card, para_raw_card,
                        dim=self.dimension_space, debug=False)

                if (tmp_kl < smallest_kl): 
                    smallest_kl = tmp_kl
                    nearest_id = history_id

            if math.exp(smallest_kl) < 200: # the first query won't go into this either
                self.cluster_weights[nearest_id] += 1
                continue
            else:
                # print(f"Training query {sql_id}: need to sample, KL is {round(smallest_kl, 3)}")
                self.cluster_weights.append(1)
                self.collectFeatureFromOneQuery(self.N, para_est_card, para_raw_card, est_sel, para_join_maps, para_join_info, para_est_sel_base, para_est_sel_join)

            self.clusters.append([para_est_card, para_raw_card])

            # print("current sample size is: ", len(self.all_base_features))
            assert len(self.all_base_features) == len(self.all_join_features)


    '''
    Collect features (samples): each sample is a selectivity vector (NOT error vector!!)
    '''
    def collectFeatureFromOneQuery(self, N, para_est_card, para_raw_card, est_sel, para_join_maps, para_join_info, para_est_sel_base, para_est_sel_join):
        error_samples = gen_samples_from_joint_err_dist(N, self.dimension_space, para_est_card, para_raw_card, self.err_info_dict)
        # base_sel_samples, join_sel_samples = convertErrToSel(error_samples, self.dimension_space, est_sel, len(self.raw_base_card))
        for err in error_samples:
            base_sel_samples, join_sel_samples = prep_sel(None, para_join_maps, para_join_info, 
                                                        para_est_sel_base, file_of_base_sel, 
                                                        para_est_sel_join, file_of_join_sel, 
                                                        error=err, recentered_error=None, 
                                                        relation_list=[int(i) for i in self.dimension_space])
            self.all_base_features.append(base_sel_samples)
            self.all_join_features.append(join_sel_samples)


    # Probability considers weighted contribution, weight is the "reuse frequency"
    def calReweightProbability(self):
        all_samples = [i+j for i, j in zip(self.all_base_features, self.all_join_features)]
        all_samples = np.array(all_samples)

        self.probability_of_sampled = [[] for _ in range(len(all_samples))]
        # for each sel sample, calculate the weighted probability of it being sampled from the cluster 
        for cluster_id, cluster in enumerate(tqdm(self.clusters, desc="calculate probability of each sample")):
            para_est_card, para_raw_card = cluster[0], cluster[1]
            probability = cal_prob_of_sample(samples=all_samples, 
                                            sensitive_rels=[int(i) for i in self.dimension_space], 
                                            est_card=para_est_card, 
                                            raw_card=para_raw_card, 
                                            err_info_dict=self.err_info_dict, 
                                            is_error_sample=False)
            # self.probability_of_sampled = [i.append(j) for i, j in zip(self.probability_of_sampled, probability)]
            
            if len(self.joint_probabilities) == 0:
                self.joint_probabilities = [j * self.cluster_weights[cluster_id] for j in probability]
            else:
                self.joint_probabilities = [i + j * self.cluster_weights[cluster_id] for i, j in zip(self.joint_probabilities, probability)]
        
        self.joint_probabilities = [i for i in self.joint_probabilities]
    

    # Not a good function: build a new kde over all samples first
    def calProbabilityOverall(self):
        # calculate the overall distribution first
        all_samples = [i+j for i, j in zip(self.all_base_features, self.all_join_features)]
        all_samples = np.array(all_samples)
        kdes = [KernelDensity(kernel="gaussian", 
                              bandwidth=0.5).fit(all_samples[:, int(i)].reshape(-1, 1)) 
                              for i in self.dimension_space]

        # calculate the probability of being sampled
        probabilities_per_dim = np.array([
            np.exp(kde.score_samples(all_samples[:, int(i)].reshape(-1, 1)))
            for i, kde in zip(self.dimension_space, kdes)
        ]).T
        print(probabilities_per_dim)
        input()
        self.joint_probabilities = np.prod(probabilities_per_dim, axis=1)


    def collectPlans(self):
        # add the optimal plan at each sample
        for base_sel, join_sel in zip(tqdm(self.all_base_features, desc=f"{self.query_id}-{self.workload_name}: collect plans"), self.all_join_features):
            sendAllSelToPG(base_output_sel=base_sel, f_base_sel=file_of_base_sel,
                           join_output_sel=join_sel, f_join_sel=file_of_join_sel, 
                            changed_relation_list=self.all_dim)
            cost_value_opt, join_order_opt, scan_mtd_opt = get_plan_cost(self.cursor, sql=self.queries_train[0], explain=explain, debug=True)
            opt_plan_at_sel_sample = gen_final_hint(join_order_opt, scan_mtd_opt)
            # if opt_plan_at_sel_sample not in self.plan_list:
            #     latency = round(get_real_latency(self.db_name, self.queries_train[0], hint=opt_plan_at_sel_sample, times=5, limit_time=200000), 5)
            #     print(latency)
            #     print(base_sel, join_sel)
            #     print(opt_plan_at_sel_sample)
            self.plan_list.add(opt_plan_at_sel_sample)
        self.plan_list = sorted(list(self.plan_list))
        print(f"{self.query_id}-{self.workload_name}: collected {len(self.plan_list)} candidate plans.")

    '''
    For each plan, collect the cost of this plan at each selectivity sample self.all_base_sample / self.all_join_features
    '''
    def collectPlanCost(self):
        result = []
        for base_sel, join_sel in zip(tqdm(self.all_base_features, desc=f"{self.query_id}-{self.workload_name}: costing each plan"), self.all_join_features):
            sendAllSelToPG(base_output_sel=base_sel, f_base_sel=file_of_base_sel,
                        join_output_sel=join_sel, f_join_sel=file_of_join_sel, 
                        changed_relation_list=self.all_dim)
            tmp = []
            for plan in self.plan_list:
                cost = get_plan_cost(self.cursor, sql=self.queries_train[0], hint=plan, explain=explain)
                tmp.append(cost)
            result.append(tmp)
        self.costCollection = np.array(result).T.tolist()
        # print(len(self.costCollection), len(self.costCollection[0]))


    '''
    For each sample, collect the cost of optimal plan (from PG); record the penalties
    Calculate the penalty based on costCollection and optCostCollection
    '''
    def collectOptCostAndPenalty(self):
        for base_sel, join_sel in zip(self.all_base_features, self.all_join_features):
            sendAllSelToPG(base_output_sel=base_sel, f_base_sel=file_of_base_sel,
                        join_output_sel=join_sel, f_join_sel=file_of_join_sel, 
                        changed_relation_list=self.all_dim)
            cost = get_plan_cost(self.cursor, sql=self.queries_train[0], hint=None, explain=explain)
            self.optCostCollection.append(cost)

        for plan_id in range(len(self.plan_list)):
            cur_penalties = []
            for i in range(len(self.optCostCollection)):
                if self.costCollection[plan_id][i] / self.optCostCollection[i] > 1 + self.tolerance:
                    penalty = self.costCollection[plan_id][i] - self.optCostCollection[i]
                else:
                    penalty = 0
                cur_penalties.append(penalty)
            self.penaltyCollection.append(cur_penalties)

    '''
    Again, using KL to measure plan similarity
    '''
    def planSpaceReductionKL(self, R):
        # plot_all_cost_distribution(self.costCollection, 
        #                            file_name=f"working/{self.query_id}_{self.template_id}_{self.workload_name}_plan_cost.pdf")
        # plot_all_cost_distribution(self.costCollection, sort=True,
        #                            file_name=f"working/{self.query_id}_{self.template_id}_{self.workload_name}_plan_cost_sorted.pdf")
        
        start_1 = time.time()
        all_kl_list = []
        for i in self.costCollection:
            kl_list = []
            for j in self.costCollection:
                kl = JS_distance(i, j)
                kl_list.append(kl)
            all_kl_list.append(kl_list)
        end_1 = time.time()

        # plot_2d_matrix(all_kl_list, 
        #             filename=f'working/{self.query_id}_{self.template_id}_{self.workload_name}_plan_cost_KL_matrix_all_plan.pdf')
        start_2 = time.time()
        remained_kl_list, resulting_plan_ids = reduce_matrix(all_kl_list, target_rows=R)
        end_2 = time.time()
        print(len(all_kl_list), len(remained_kl_list), resulting_plan_ids)
        

        #### plot cost distribution of similar plan groupsdsf
        anchor_to_plan_dict = {}
        for plan_id, kl_list in enumerate(all_kl_list):
            smallest_kl = 1e10
            nearest_anchor = -1
            for target_id in resulting_plan_ids:
                if kl_list[target_id] < smallest_kl:
                    smallest_kl = kl_list[target_id]
                    nearest_anchor = target_id
            if nearest_anchor in anchor_to_plan_dict:
                anchor_to_plan_dict[nearest_anchor].append(plan_id)
            else:
                anchor_to_plan_dict[nearest_anchor] = [plan_id]
        # print(anchor_to_plan_dict)
        # {1: [0, 1, 15, 17], 2: [2, 3, 5, 6, 18], 16: [4, 9, 10, 11, 12, 14, 16, 20, 21, 22], 7: [7], 8: [8, 13, 19]}

        # for anchor in anchor_to_plan_dict.keys():
        #     plot_all_cost_distribution([self.costCollection[i] for i in anchor_to_plan_dict[anchor]], labels=anchor_to_plan_dict[anchor], anchor=anchor,
        #                            file_name=f"working/17/20/{self.query_id}_{self.template_id}_{self.workload_name}_plan_cost_anchor_{anchor}.pdf")



        # input()
        print(f"== Plan reduction time {round((end_1-start_1+end_2-start_2)*1000, 1)}(ms)")
        # plot_2d_matrix(remained_kl_list, resulting_plan_ids, 
        #             filename=f'working/{self.query_id}_plan_cost_KL_matrix_after_remove.pdf')
        
        self.costCollection = [self.costCollection[i] for i in resulting_plan_ids]
        self.penaltyCollection = [self.penaltyCollection[i] for i in resulting_plan_ids]
        self.plan_list = [self.plan_list[i] for i in resulting_plan_ids]
        # plot_all_cost_distribution(self.costCollection, labels=resulting_plan_ids, 
        #                            file_name=f"working/{self.query_id}_{self.template_id}_{self.workload_name}_plan_cost_after_remove.pdf")
        # plot_all_cost_distribution(self.costCollection, labels=resulting_plan_ids, sort=True,
        #                            file_name=f"working/{self.query_id}_{self.template_id}_{self.workload_name}_plan_cost_after_remove_sorted.pdf")
        
        print(f"After reducing, we have {len(self.plan_list)} plans")


    '''
    Again, using JS to measure plan similarity and reduce plan space
    '''
    def planSpaceReductionJS(self, R):
        
        # test: find the plan with largest robust coverage first
        # first_plan, _ = reduce_by_opt_range(self.costCollection, 1)
        first_plan = [None]


        start_1 = time.time()
        centers, assignments = k_center_greedy(self.costCollection, R, JS_distance, first_plan=first_plan[0])
        end_1 = time.time()
        output_string = f"== Plan reduction time {round((end_1-start_1)*1000, 1)}(ms)\n== Reduce from {len(self.costCollection)} to {len(centers)}"

        print(output_string)
        logging.info(output_string)
        # print(assignments)
        # # input()
        # for anchor in assignments.keys():
        #     plot_all_cost_distribution([self.costCollection[i] for i in assignments[anchor]], 
        #                                labels=assignments[anchor], 
        #                                anchor=anchor,
        #                                file_name=f"working/17/JS-{self.N}/{self.query_id}_{self.template_id}_{self.workload_name}_plan_cost_anchor_{anchor}.pdf")
        # exit()
        
        self.costCollection = [self.costCollection[i] for i in centers]
        self.penaltyCollection = [self.penaltyCollection[i] for i in centers]
        self.plan_list = [self.plan_list[i] for i in centers]


    def planSpaceReductionOptRange(self, R):
        centers, sel_to_plan_dict = reduce_by_opt_range(self.costCollection, R)
        self.costCollection = [self.costCollection[i] for i in centers]
        self.penaltyCollection = [self.penaltyCollection[i] for i in centers]
        self.plan_list = [self.plan_list[i] for i in centers]

        # for id, anchor in enumerate(self.costCollection):
        #     plot_all_cost_distribution([self.costCollection[id]], 
        #                                labels=None, 
        #                                anchor=None,
        #                                file_name=f"working/17/RobustRange-{self.N}/{self.query_id}_{self.template_id}_{self.workload_name}_plan_cost_{id}.pdf")
        # exit()


        return sel_to_plan_dict

    def evaluate(self, R, exe=True, rob_verify=False, split=None, prune='rob-range'):
        plan_candidate_size = len(self.plan_list)
        # self.planSpaceReductionKL(R=R)
        # Alternatively: 
        start_1 = time.time()
        if R == 0:
            R = max(10, int(plan_candidate_size/5))
        else:
            R = min(plan_candidate_size, R)
        if prune == 'sim':
            # list1 = self.costCollection
            self.planSpaceReductionJS(R=R)
            # list2 = self.costCollection
            # reduce_by_opt_range_evaluate(plan_cost_list=list1, sampled_plan_cost_list=list2, K=R)      
        elif prune == 'rob-range':
            sel_to_plan_dict = self.planSpaceReductionOptRange(R=R)  
        elif prune == 'no':
            print("Don't prune any plan...")
        start_2 = time.time()


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


            start_2 = time.time()
            probability = cal_prob_of_sample(samples=cached_sel, 
                                            sensitive_rels=[int(i) for i in self.dimension_space], 
                                            est_card=para_est_card, raw_card=para_raw_card, 
                                            err_info_dict=self.err_info_dict)

            # calculate expected penalty for this query
            exp_penalty_list = []
            for plan_id, plan in enumerate(self.plan_list):

                cur_penalty_list = [cached_penalty 
                                    * new_p 
                                    / old_p
                                   for cached_penalty, new_p, old_p
                                   in zip(self.penaltyCollection[plan_id], probability, self.joint_probabilities)]
                exp_penalty_list.append(sum(cur_penalty_list))
            # for i in range(len(self.joint_probabilities)):
            #     print(i, "prob new: ", probability[i], "prob old: ", self.joint_probabilities[i],probability[i]/self.joint_probabilities[i])
            #     print(np.array(self.penaltyCollection)[:, i])
                # input()
            robust_plan_id = [i for i, x in enumerate(exp_penalty_list) if x == min(exp_penalty_list)][0]
            self.output_result.append([sql_id, para_sql, self.plan_list[robust_plan_id]])
            end_2 = time.time()
            avg_planning_time += (end_2 - start_2)*1000
            logging.info(f"== Inference time {round((end_1-start_1 + end_2-start_2)*1000, 2)}(ms)")
            
            if exe:
                if rob_verify and split != None:
                    # first, get the original pg plan at the base instance 
                    cost_value_opt, join_order_opt, scan_mtd_opt = get_plan_cost_simple(self.cursor, sql=para_sql, explain=explain, debug=True)
                    base_pg_plan = gen_final_hint(join_order_opt, scan_mtd_opt)

                    pqo_verify_one_query = []
                    refreshed_pg_verify_one_query = []
                    base_pg_verify_one_query = []

                    if split == 'random' or split == 'sliding':
                        ins_list = [0, 1, 2, 3, 4, 5, 6, 7, 8]
                    else:
                        ins_list = [1, 2, 3, 4, 6, 7] # category

                    for new_ins_id in ins_list:
                        para_sql_per_ins = para_sql.replace(f"_{rob_verify} AS", f"_{new_ins_id} AS")
                        
                        if self.query_id == '30' or  self.query_id == '19' :
                        
                            base_pg_plan_latecy = round(get_real_latency(self.db_name, para_sql_per_ins, hint=base_pg_plan, times=3, limit_time=200000), 3)        
                            
                            cost_value_opt, join_order_opt, scan_mtd_opt = get_plan_cost_simple(self.cursor, sql=para_sql_per_ins, explain=explain, debug=True)
                            new_pg_plan = gen_final_hint(join_order_opt, scan_mtd_opt)

                            refreshed_pg_plan_latency = round(get_real_latency(self.db_name, para_sql_per_ins, hint=new_pg_plan, times=3, limit_time=200000), 3)        
                        else:
                            base_pg_plan_latecy, refreshed_pg_plan_latency = 1, 1
                        robust_plan_latency = round(get_real_latency(self.db_name, para_sql_per_ins, hint=self.plan_list[robust_plan_id], times=3, limit_time=200000), 3)
                        pqo_verify_one_query.append(robust_plan_latency)
                        refreshed_pg_verify_one_query.append(refreshed_pg_plan_latency)
                        base_pg_verify_one_query.append(base_pg_plan_latecy)

                    result_verify_pqo.append(pqo_verify_one_query)
                    result_verify_refresh_pg.append(refreshed_pg_verify_one_query)
                    result_verify_base_pg.append(base_pg_verify_one_query)

                    pqo_avg = [sum(column) / len(column) for column in list(zip(*result_verify_pqo))]
                    pg_refreshed_avg = [sum(column) / len(column) for column in list(zip(*result_verify_refresh_pg))]
                    pg_base_avg = [sum(column) / len(column) for column in list(zip(*result_verify_base_pg))]

                    output_string = f"Q{self.query_id}-{self.workload_name}-{sql_id}-{self.b}-{len(self.cluster_weights)}-{len(self.plan_list)}:\n" \
                    f"Robust plan is {robust_plan_id}: {pqo_verify_one_query},\nPostgres refresh plan: {refreshed_pg_verify_one_query} \n" \
                    f"Base pg plan: {base_pg_verify_one_query}\n" \
                    f"avg: pg refresh {[round(i, 3) for i in pg_refreshed_avg]} / pqo {[round(i, 3) for i in pqo_avg]} = {[round(i / j, 3) for i, j in zip(pg_refreshed_avg, pqo_avg)]}\n" \
                    f"avg: pg refresh {[round(i, 3) for i in pg_refreshed_avg]} / base pg {[round(i, 3) for i in pg_base_avg]} = {[round(i / j, 3) for i, j in zip(pg_refreshed_avg, pg_base_avg)]} "
                    
                    print(output_string)
                    logging.info(output_string)

                else:
                    if self.debug:
                        print("exp penalty: ", exp_penalty_list)
                        print(f"Robust plan is {robust_plan_id}")
                        for i in range(len(self.plan_list)):
                            latency = round(get_real_latency(self.db_name, para_sql, hint=self.plan_list[i], times=5, limit_time=200000), 5)
                            print(f"Cur plan is {i}: {latency} {self.plan_list[i]}")
                            if i == robust_plan_id: robust_plan_latency = latency
                    else:
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
        if not self.rob_verify:
            filename = f'./reuse/{self.db_name}/diagram/{self.query_id}-{self.template_id}_{self.workload_name}_b{self.b}_N{self.N}.json'
        else:
            # For robustness verification runs
            filename = f'./reuse/{self.db_name}/{self.rob_verify}/db_instance_{self.ins_id}/{self.query_id}-{self.template_id}_{self.workload_name}_b{self.b}_N{self.N}.json'


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
