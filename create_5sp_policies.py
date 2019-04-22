import json 
import os

def LoadPolicy(policy_path):
    policy_json = None

    with open(policy_path, "r") as f:
        policy_json =  json.loads(f.read())

    return policy_json


def SplitPolicy(policy_json,new_policy_size = 5):

    new_policies = []

    full_policy = policy_json["policy"]["policy"]
    new_policy_count = 0
    for new_policy_start_i in range(0,len(full_policy),new_policy_size):
        new_policy_count += 1
        new_policy = full_policy[new_policy_start_i:new_policy_start_i+new_policy_size]

        new_policy_json = {"policy":{}}

        new_policy_json["policy"]["id"] = policy_json["policy"]["id"] + "_short_policy_"+str(new_policy_count)

        new_policy_json["policy"]["policy"] = new_policy

        new_policies.append(new_policy_json)

    return new_policies


policy_path = os.path.join("policies","autoaugment_paper_cifar10.json")

policy_json = LoadPolicy(policy_path)

new_small_policies = SplitPolicy(policy_json)

for small_policy in new_small_policies:

    new_path = os.path.join("policies_of_interest",small_policy["policy"]["id"]+".json")

    with open(new_path,"w") as f:
        f.write(json.dumps(small_policy,indent=4))