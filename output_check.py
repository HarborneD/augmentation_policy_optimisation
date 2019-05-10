import os
import csv

def AccumulateOutputs():
	output_dir = "outputs"

	output_files = os.listdir(output_dir)

	accuracies = []
	print(len(output_files))
	for output_file in output_files[:]:
		file_path = os.path.join(output_dir,output_file)

		final_accuracy = ""
		with open(file_path,"r") as f:
			try:
				file_text = f.read()
			except:
				print(file_path, " could not be read")
				continue
			text_lines = file_text.split("\n")
			if(len(text_lines) < 2):
				continue
			final_accuracy = text_lines[-2].split(" ")[-1]

		accuracies.append( (output_file, final_accuracy, text_lines) )

	highest_accuracy = 0
	highest_output = ""
	for accuracy in accuracies:
		try:
			accuracy_val = float(accuracy[1])
		except:
			print("failed to find accuracy:")
			# print(accuracy)
		if(accuracy_val > float(highest_accuracy)):
			highest_accuracy = accuracy[1]
			highest_output = accuracy[0]

	accuracies = sorted(accuracies,key=lambda x: x[1],reverse=True)

	print(highest_output)
	print(highest_accuracy)

	output_lines = [ ",".join(["output_file", "policy_id","accuracy" ]) ]

	for accuracy in accuracies:
		output_lines.append( ",".join([ accuracy[0],str(accuracy[2][1]), str(accuracy[1]) ]) )


	with open("outputs_accumulated.csv" , "w") as f:
		f.write( "\n".join(output_lines) )


def LoadOutputsToDict():
	outputs_dict = {}
	policy_dict = {}
	sorted_outputs_array = []

	with open("outputs_accumulated.csv" , "r") as f:
		output_lines = f.read().split("\n")
	
	for line in output_lines:
		values = line.split(",")
	#output-4539521.o,policy_id: tabu_fTransf_AutoAugmentBasedPopulation_exp_0001_60e_100ls_25-2_00001_00174,busy
		try:
			accuracy = float(values[2])
		except:
			continue

		job_id = values[0].replace("output-","").replace(".o","")
		policy_id = values[1].replace("policy_id: ","")

		outputs_dict[job_id] = len(sorted_outputs_array)

		policy_dict[policy_id] = len(sorted_outputs_array)

		sorted_outputs_array.append({
			"job_id":job_id
			,"policy_id":policy_id
			,"accuracy":accuracy
		})

	
	return outputs_dict, policy_dict, sorted_outputs_array


def FetchEntryByPolicyId(policy_id,policy_dict,sorted_outputs_array):
	if(policy_id in policy_dict):
		return sorted_outputs_array[ policy_dict[policy_id] ]
	else:
		return None

def FetchEntryByJobId(job_id,output_dict,sorted_outputs_array):
	if(job_id in output_dict):
		return sorted_outputs_array[ output_dict[job_id] ]
	else:
		return None


def PopulateResultsFile(path_to_results_file, outputs_dict, policy_dict, sorted_outputs_array, output_file_path="", skip_missing_accuracies=False):
	if(output_file_path == ""):
		output_file_path = path_to_results_file.replace(".csv","_updated.csv")
	
	entries = []
	with open(path_to_results_file, "r", encoding='utf-8-sig') as f:
		results_reader = csv.DictReader(f)

		for row in results_reader:
			policy_id = row["policy_id"]
			job_id = row["job_id"] 
			accuracy = row["accuracy"]

			if(row["accuracy"] == ""):
				if(job_id != ""):
					result = FetchEntryByJobId(job_id,outputs_dict, sorted_outputs_array)
					if result is None:
						print("could not find job: "+str(job_id))
						if(skip_missing_accuracies):
							continue
					else:
						row["policy_id"] = result["policy_id"]
						row["accuracy"] = str(result["accuracy"])
				# elif(policy_id != ""):
				# 	result = FetchEntryByPolicyId(policy_id,policy_dict, sorted_outputs_array)
				# 	if result is None:
				# 		continue
				# 	row["job_id"] = result["job_id"]
				else:
					print(policy_id,"|", row["trial_num"],"|",row["description"],"|", "job id empty")
					if(skip_missing_accuracies):
						continue
				
				
			
			entries.append(row)

	fieldnames = ["policy_id","algorithm","model","description","child_epochs","trial_num","job_id","accuracy","confirmed_not_old_checkpoints"]
	
	with open(output_file_path, "w") as csvfile:
		csvwriter = csv.DictWriter(csvfile,  delimiter=',',  fieldnames= fieldnames)
		
		for entry in entries:
			csvwriter.writerow(entry)
		
if __name__ == "__main__":
	AccumulateOutputs()

	outputs_dict, policy_dict,sorted_outputs_array = LoadOutputsToDict()

	path_to_results_file = "/media/harborned/ShutUpN/google_drive/From Dropbox/COMSC/Year 4/Final Year Project/results.csv"
	PopulateResultsFile(path_to_results_file, outputs_dict, policy_dict, sorted_outputs_array, output_file_path="")
	print("")