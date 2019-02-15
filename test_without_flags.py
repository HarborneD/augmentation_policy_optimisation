print("start of file:test_without_flags.py")
import tensorflow as tf
print("imported tensorflow")
import evaluator.evaluate_policies_without_flags
print("imported evaluate_policies_without_flags")
import os
import sys

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

FLAGS = tf.flags.FLAGS

print("created flags")


def TrainWithPolicy(policy_id, num_epochs, data_path, dataset="cifar10", model_name="wrn", use_cpu=0):
    print(FLAGS)
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

    valid_accuracy, test_accuracy = evaluator.evaluate_policies_without_flags.TrainModelWithPolicies(FLAGS)
    return valid_accuracy, test_accuracy


if __name__ == "__main__":
    print("This is the start of __name__==__main__ block test without flags")
    data_path = "/media/harborned/ShutUpN/datasets/cifar/cifar-10-batches-py"
    if(len(sys.argv) > 1):
        data_path = sys.argv[1]
    print("Using data path:"+data_path)
    policy_id = "000001"
    num_epochs = 5
    model_name = "wrn"
    data_path = data_path
    dataset = "cifar10"
    use_cpu = 0 

    print("Training Child Model Using Policy: "+str(policy_id))
    valid_accuracy, test_accuracy = TrainWithPolicy(policy_id, num_epochs, data_path, dataset="cifar10", model_name="wrn", use_cpu=0)

    print("valid_accuracy", "test_accuracy")
    print(valid_accuracy, test_accuracy)
