from ARCCAPythonTool import ArccaTool

import os
import time

from simple_toolbar import ProgressBar

class RemoteGATool(object):
    def __init__(self,local_ga_dir,remote_ga_dir,host="hawklogin.cf.ac.uk"):
        self.local_ga_directory = local_ga_dir
        self.remote_ga_directory = remote_ga_dir
        
        self.local_policies_directory = os.path.join(self.local_ga_directory,"policies")
        self.remote_policies_directory = os.path.join(self.remote_ga_directory,"policies")

        self.host_key = b'SHA256:P8MxFCLE7+ROcYqIdFRZSZ1WI7CKGIWsJ96o5vjZluo' #ECDSA may not be supported by paramiko,tool uses system approved host keys
        self.host = host

        self.arcca_tool = ArccaTool(host)
        self.arcca_tool.DangerousAutoAddHost()

        self.ACCOUNT = "scw1427"
        self.RUN_FROM_PATH = "/home/c.c0919382/fyp_scw1427/genetic_augment"
        self.SCRIPT_NAME = "arcca_evaluate_child_model.sh"
        
        self.DATA_PATH="/home/c.c0919382/datasets/cifar-10-batches-py"

        self.training_tracker = {}
        self.job_map = {}
        self.current_generation = []
        self.running_jobs_of_generation = []

    def SendPolicyFile(self,policy_id):
        local_policy_path = os.path.join(self.local_policies_directory, policy_id+".json")
        remote_policy_path = os.path.join(self.remote_policies_directory, policy_id+".json")
        self.arcca_tool.SendFileToServer(local_policy_path,remote_policy_path)
    
    
    def StartRemoteChromosomeTrain(self, policy_id, num_epochs, data_path, dataset="cifar10", model_name="wrn", use_cpu=0):
        _, _, _, job_id, was_error = self.arcca_tool.StartBatchJob(self.ACCOUNT,self.RUN_FROM_PATH,self.SCRIPT_NAME,'"'+str(policy_id)+'" "'+str(num_epochs)+'"')
        return job_id, was_error
        
    def HandleJobError(self):
        #TODO: handle job errors 
        print("job error handling not implemented")
    

    def StartGenerationTraining(self,policy_ids, num_epochs):
        for policy_id in policy_ids:
            job_id, was_error = self.StartRemoteChromosomeTrain(policy_id, num_epochs,self.DATA_PATH)
            if(not was_error):
                print(str(policy_id) + " posted as job: "+str(job_id))
                self.training_tracker[policy_id] = {"job_id":job_id,"last_known_status":"submitted"}
                self.job_map[job_id] = policy_id
                self.running_jobs_of_generation.append(policy_id)
            else:
                self.HandleJobError()
        self.current_generation = policy_ids
    

    def JobListToPolicyList(self,job_list):
        return [self.job_map[j_id]for j_id in job_list]
    
    def PolicyListToJobList(self,policy_list):
        return [self.training_tracker[p_id]["job_id"] for p_id in policy_list]
    

    def UpdateCurrentGenerationJobs(self):

        # "job_id":result[0]
        #         ,"partition":result[1]
        #         ,"name":result[2]
        #         ,"user":result[3]
        #         ,"st":result[4]
        #         ,"time":result[5]
        #         ,"nodes":result[6]
        #         ,"nodelist":result[7]
        #         }
        job_queue = self.arcca_tool.CheckJobs(job_ids=self.PolicyListToJobList(self.current_generation))

        jobs = []
        for job_line in job_queue[1:]:
            jobs.append(self.arcca_tool.ProcessJobLine(job_line))
        
        jobs_in_queue = []
        for job in jobs:
            jobs_in_queue.append(job["job_id"])
            policy_id = self.job_map[job["job_id"]]
            self.training_tracker[policy_id]["last_known_status"] = self.arcca_tool.JOB_STATUS_CODES[job["st"]]["name"]

        self.running_jobs_of_generation = jobs_in_queue


    def WaitForGenerationComplete(self):
        num_jobs = len(self.current_generation)

        progress = ProgressBar(num_jobs, width=20, fmt=ProgressBar.FULL)

        while len(self.running_jobs_of_generation) > 0:
            self.UpdateCurrentGenerationJobs()
            jobs_strings=""
            for job_id in self.running_jobs_of_generation:
                policy_id = self.job_map[job_id]
                jobs_strings+= str(job_id)+","
            jobs_strings = jobs_strings[:-1] 
            progress.current = num_jobs - len(self.running_jobs_of_generation)
            progress(jobs_strings)
            time.sleep(5)
        progress.done()
        

    def ReadResultsFile(self,local_file_path):
        results_headings = ["policy_id","num_epochs","model_name","dataset","use_cpu","time_taken"]
        
        # for results_heading in results_headings:
        #     results_string += str(configuration_dict[results_heading]) +","
        
        # results_string += str(test_accuracy)
        results_string = ""
        with open(local_file_path,"r") as f:
            results_string = f.read()
        
        result_split = results_string.split(",")
        
        result = {}

        for results_heading_i in range(len(results_headings)):
            result[results_headings[results_heading_i]] = result_split[results_heading_i]
        
        result["test_accuracy"] = float(result_split[-1])
        
        return result

    def GetGenerationResults(self):
        local_results_dir = os.path.join(self.local_ga_directory,"results")
        remote_results_dir = os.path.join(self.remote_ga_directory,"results")
        
        results = []
        for policy_id in self.current_generation:
            local_result_path = os.path.join(local_results_dir,policy_id+".csv")
            remote_result_path = os.path.join(remote_results_dir,policy_id+".csv")

            try:
                self.arcca_tool.FetchFileFromServer(remote_result_path,local_result_path)

                results.append(self.ReadResultsFile(local_result_path))
            except:
                print("Policy Failed: "+ policy_id)
                print("")
        return results
    

    def CleanDirectoriesAndStoreCurrentGen(self,policy_ids):
        previous_generation_path = os.path.join(self.remote_ga_directory,"previous_generation")

        #clean previous_generation folders
        self.CleanPreviousGeneration(previous_generation_path)
        
        #copy current generations to previous_generation folders
        self.CopyCurrentGenerationToPreviousGenerationFolder(policy_ids,previous_generation_path)
        
        #clean main directories
        self.CleanCurrentGeneration(policy_ids)


    def CleanPreviousGeneration(self,previous_generation_path):
        pass


    def CopyCurrentGenerationToPreviousGenerationFolder(self,policy_ids,previous_generation_path):
        pass


    def CleanCurrentGeneration(self,policy_ids):
        pass



if __name__ == "__main__":
    test_after_posting_jobs = False
    submitted_policies = [("000001","2833647"),("000002","2833648") ] #for testing after jobs are posted
        
    local_ga_directory = "/media/harborned/ShutUpN/repos/final_year_project/genetic_augment"
    remote_ga_directory = "/home/c.c0919382/fyp_scw1427/genetic_augment"

    remote_tool = RemoteGATool(local_ga_directory,remote_ga_directory)

    #TODO: uncomment section after testing
    test_policy_ids = ["000001","000002"]

    for policy_id in test_policy_ids:
        remote_tool.SendPolicyFile(policy_id)

    
    if(test_after_posting_jobs):
        #TODO: remove code after testing:
        for submitted_policy in submitted_policies:
            remote_tool.training_tracker[submitted_policy[0]] = {"job_id":submitted_policy[1],"last_known_status":"submitted"}
            remote_tool.job_map[submitted_policy[1]] = submitted_policy[0]
            remote_tool.running_jobs_of_generation.append(submitted_policy[0])
        remote_tool.current_generation = [p[0] for p in submitted_policies]
    else:
        remote_tool.StartGenerationTraining(test_policy_ids,5)

        # remote_tool.arcca_tool.PollJobs()


        # remote_tool.arcca_tool.CheckJobs(job_ids=remote_tool.current_generation)

    remote_tool.WaitForGenerationComplete()

    time.sleep(2)
    results = remote_tool.GetGenerationResults()

    for r in results:
        print(r)