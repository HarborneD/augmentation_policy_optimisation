import os
import sys
import getopt

import tensorflow as tf

import evaluator.evaluate_policies_without_flags

import time

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 


tf.flags.DEFINE_string('model_name', "wrn",
                    'wrn, shake_shake_32, shake_shake_96, shake_shake_112, '
                    'pyramid_net')
tf.flags.DEFINE_string('checkpoint_dir', "default_checkpoints", 'Training Directory.')
tf.flags.DEFINE_string('data_path', "/data/cifar10",
                    'Directory where dataset is located.')
tf.flags.DEFINE_string('dataset', "cifar10",
                    'Dataset to train with. Either cifar10 or cifar100')
tf.flags.DEFINE_integer('use_cpu', 0, '1 if use CPU, else GPU.')
tf.flags.DEFINE_string('policy_id', "default_policy_0001", 'id of policy to be evaluated')
tf.flags.DEFINE_integer('num_epochs', 2, 'Number of epochs to train model before evaluating')
tf.flags.DEFINE_integer('num_training_images', 4000, 'Number of training images to train on.')

FLAGS = tf.flags.FLAGS

print("created flags")


def TrainWithPolicy(policy_id, num_epochs, data_path, dataset="cifar10", model_name="wrn", use_cpu=0, num_training_images=4000):
    print("Training with policy:"+str(policy_id))
    
    checkpoints_dir = os.path.join(os.getcwd(),"checkpoints")
    
    if(not os.path.exists(checkpoints_dir)):
        os.mkdir(checkpoints_dir)

    checkpoint_dir = os.path.join(checkpoints_dir,"checkpoints_"+policy_id)

    FLAGS.model_name = model_name
    FLAGS.checkpoint_dir = checkpoint_dir 
    FLAGS.data_path = data_path 
    FLAGS.dataset = dataset 
    FLAGS.use_cpu = use_cpu 
    FLAGS.policy_id = policy_id 
    FLAGS.num_epochs = num_epochs 
    FLAGS.num_training_images = num_training_images 

    valid_accuracy, test_accuracy = evaluator.evaluate_policies_without_flags.TrainModelWithPolicies(FLAGS)
    return valid_accuracy, test_accuracy


def StoreResults(configuration_dict, test_accuracy):
    results_dir = os.path.join(os.getcwd(),"results")
    result_path = os.path.join(results_dir,configuration_dict["policy_id"]+".csv")
    results_string = ""
    results_headings = ["policy_id","num_epochs","model_name","dataset","use_cpu","time_taken"]
    
    for results_heading in results_headings:
        results_string += str(configuration_dict[results_heading]) +","
    
    results_string += str(test_accuracy)

    with open(result_path,"w") as f:
        f.write(results_string)
    

if __name__ == "__main__":
    try:
      opts, args = getopt.getopt(sys.argv[1:],"",["data_path=","policy_id=","num_epochs=","model_name","dataset","use_cpu","num_training_images"])
    
    except getopt.GetoptError:
      print('evaluate_child_model.py --data_path <data_path> --policy_id <policy_id> --num_epochs <num_epochs> --model_name <model_name> --dataset <dataset> --use_cpu <use_cpu> --num_training_images <num_training_images>')
      sys.exit(2)
    
    #set_defaults
    model_name = "wrn"
    dataset = "cifar10"
    use_cpu = 0 
    num_training_images=50000

    for opt, arg in opts:
        if opt in ("--data_path"):
            data_path = arg
        elif opt in ("--policy_id"):
            policy_id = arg
        elif opt in ("--num_epochs"):
            num_epochs = arg
        elif opt in ("--model_name"):
            model_name = arg
        elif opt in ("--dataset"):
            dataset = arg
        elif opt in ("--use_cpu"):
            use_cpu = arg
        elif opt in ("--num_training_images"):
            print("num training images arg found")
            num_training_images = arg
    print("Traing Model: "+str(model_name))
    print("Training Child Model Using Policy: "+str(policy_id))
    print("Number of Training Images: "+str(num_training_images))
    start_time = time.time()
    valid_accuracy, test_accuracy = TrainWithPolicy(policy_id, num_epochs, data_path, dataset=dataset, model_name=model_name, use_cpu=use_cpu, num_training_images=num_training_images)
    time_taken = time.time() - start_time

    print("valid_accuracy", "test_accuracy")
    print(valid_accuracy, test_accuracy)

    configuration_dict = {
        "policy_id":policy_id
        ,"num_epochs":num_epochs
        ,"model_name":model_name
        ,"dataset":dataset
        ,"use_cpu":use_cpu
        ,"time_taken":time_taken
        ,"num_training_images":num_training_images
    }

    StoreResults(configuration_dict, test_accuracy)