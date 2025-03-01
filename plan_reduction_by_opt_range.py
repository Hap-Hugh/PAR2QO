import numpy as np


def reduce_by_opt_range(plan_cost_list, K):
    """
    Greedy pick K plans with largest coverages
    """
    sample_size = len(plan_cost_list[0])
    plan_size = len(plan_cost_list)
    # Minimum cost for each sample index across all plans
    opt_costs = [
        min(plan_cost_list[plan_id][s] for plan_id in range(plan_size))
        for s in range(sample_size)
    ]
    
    uncovered_sels = set(list(range(sample_size))) # covered sel samples sets
    available_plan_sets = set(list(range(len(plan_cost_list)))) # available plans
    sample_to_plan_dict = {}
    saved_plan = []
    coverage_percentage = []
    
    while len(uncovered_sels) != 0 and len(saved_plan) < K and len(available_plan_sets) > 0:
        # find the largest opt range plan in the plan sets
        best_coverage = set()
        best_plan_id = -1
        for plan_id in available_plan_sets:
            
            cur_covered_sel = cal_opt_range(plan_cost_list[plan_id], opt_costs, uncovered_sels)
            if len(cur_covered_sel) > len(best_coverage):
                best_coverage = cur_covered_sel
                best_plan_id = plan_id
        if best_plan_id == -1:
            break
        saved_plan.append(best_plan_id)
        coverage_percentage.append(round(len(best_coverage)/sample_size, 5))
        available_plan_sets.remove(best_plan_id)
        uncovered_sels -= best_coverage
        for sel_id in best_coverage:
            sample_to_plan_dict[sel_id] = len(saved_plan)-1 # new plan id
    
    print("Robustness range: ", coverage_percentage, sum(coverage_percentage), "original plan size: ", plan_size, f"reduce to {len(saved_plan)} {saved_plan}")
    return saved_plan, sample_to_plan_dict


def reduce_by_opt_range_evaluate(plan_cost_list, sampled_plan_cost_list, K):
    """
    Greedy pick K plans with largest coverages
    """
    sample_size = len(plan_cost_list[0])
    plan_size = len(plan_cost_list)
    # Minimum cost for each sample index across all plans
    opt_costs = [
        min(plan_cost_list[plan_id][s] for plan_id in range(plan_size))
        for s in range(sample_size)
    ]
    
    uncovered_sels = set(list(range(sample_size))) # covered sel samples sets
    available_plan_sets = set(list(range(len(sampled_plan_cost_list)))) # available plans
    sample_to_plan_dict = {}
    saved_plan = []
    coverage_percentage = []
    
    while len(uncovered_sels) != 0 and len(saved_plan) < K and len(available_plan_sets) > 0:
        # find the largest opt range plan in the plan sets
        best_coverage = set()
        best_plan_id = -1
        for plan_id in available_plan_sets:
            
            cur_covered_sel = cal_opt_range(sampled_plan_cost_list[plan_id], opt_costs, uncovered_sels)
            if len(cur_covered_sel) > len(best_coverage):
                best_coverage = cur_covered_sel
                best_plan_id = plan_id
        if best_plan_id == -1:
            print("-- Can not find more robust plans")
            coverage_percentage.append(0)
            available_plan_sets.pop()
            continue
        saved_plan.append(best_plan_id)
        coverage_percentage.append(round(len(best_coverage)/sample_size, 5))
        available_plan_sets.remove(best_plan_id)
        uncovered_sels -= best_coverage
        for sel_id in best_coverage:
            sample_to_plan_dict[sel_id] = len(saved_plan)-1 # new plan id
    
    print("-- uncovered: ", round(len(uncovered_sels)/sample_size, 3) )
    print("-- Robustness range: ", coverage_percentage, sum(coverage_percentage), "original plan size: ", plan_size, f"reduce to {len(saved_plan)} {saved_plan}")
    return saved_plan, sample_to_plan_dict






def cal_opt_range(plan_cost, opt_cost, uncovered_sels):
    covered_sels = set()
    for i in uncovered_sels:
        if plan_cost[i] < opt_cost[i] * 1.2:
            covered_sels.add(i)
    return covered_sels


def find_nearest_sample(cached_sels, est_sel):
    nearest_sel_id = -1
    min_dist = float('inf')
    for id, sel in enumerate(cached_sels):
        dist = sum((x - y) ** 2 for x, y in zip(est_sel, sel)) ** 0.5
        if dist < min_dist:
            nearest_sel_id = id
            min_dist = dist
    return nearest_sel_id