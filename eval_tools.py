import subprocess
import json
import os
import sys

def EvaluatePolicy(policy_id, num_epochs):
    validation_result = subprocess.call(['./evaluator/evaluate.sh',str(policy_id),str(num_epochs)])

    return CleanValidationResult(validation_result)

def CleanValidationResult(validation_result):
    return validation_result


def LoadPolicy(policy_id):
    policy_dir = "policies"

    policy_path = os.path.join(policy_dir,policy_id+".json")
    policy_json = None
    with open(policy_path) as f:
        policy_json = json.load(f)
    
    return PolicyFromJson(policy_json)

def PolicyFromJson(policy_json):
    return  [[(str(a[0]),a[1],a[2]) for a in sp] for sp in policy_json["policy"]["policy"]]


if __name__ == "__main__":
    print(LoadPolicy("000001"))