import json 
import os

def LoadPolicy(policy_path):
    policy_json = None

    with open(policy_path, "r") as f:
        policy_json =  json.loads(f.read())

    return policy_json


def ThresholdPolicy(policy_json, probability_threshold = 0.5, magnitude_threshold = 0):

    new_policy = []

    for sub_policy in policy_json["policy"]["policy"]:
        new_sub = []
        for technique in sub_policy:
            if(technique[1] >= probability_threshold and technique[2] >= magnitude_threshold):
                new_sub.append(technique)
        
        if(len(new_sub) >0):
            new_policy.append(new_sub)
    
    policy_json["policy"]["id"] += "_thresholded_"+str(probability_threshold)+"_"+str(magnitude_threshold)

    policy_json["policy"]["policy"] = new_policy

    return policy_json


policy_path = os.path.join("policies","autoaugment_paper_cifar10.json")

policy_json = LoadPolicy(policy_path)

thresholded_policy = ThresholdPolicy(policy_json,probability_threshold = 0.3, magnitude_threshold = 0)

new_path = os.path.join("policies_of_interest",thresholded_policy["policy"]["id"]+".json")

with open(new_path,"w") as f:
    f.write(json.dumps(thresholded_policy,indent=4))