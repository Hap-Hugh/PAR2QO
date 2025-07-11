import json

from cached_robust_plan_dict import *

data = {}

data["cached_rob_plan_dict_on_demand"] = cached_rob_plan_dict_on_demand

with open("cached_info/robust_plan_dict.json", 'w') as f:
    json.dump(data, f, indent=4)

with open("cached_info/robust_plan_dict.json", 'r') as f:
    data = json.load(f)
