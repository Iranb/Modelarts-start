import os
import pprint
from pathlib import Path
from modelarts.session import Session
from modelarts.estimator import Estimator

JOB_STATUS = {
    0:"JOBSTAT_UNKNOWN，作业状态未知",
    1:"JOBSTAT_INIT，作业初始化状态。",
    2:"JOBSTAT_IMAGE_CREATING，作业镜像正在创建。",
    3:"JOBSTAT_IMAGE_FAILED，作业镜像创建失败。",
    4:"JOBSTAT_SUBMIT_TRYING，作业正在提交。",
    5:"JOBSTAT_SUBMIT_FAILED，作业提交失败。",
    6:"JOBSTAT_DELETE_FAILED，作业删除失败。",
    7:"JOBSTAT_WAITING，作业正在排队中。",
    8:"JOBSTAT_RUNNING，作业正在运行中。",
    9:"JOBSTAT_KILLING，作业正在删除。",
    10:"JOBSTAT_COMPLETED，作业已经完成。",
    11:"JOBSTAT_FAILED，作业运行失败。",
    12:"JOBSTAT_KILLED，作业取消成功。",
    13:"JOBSTAT_CANCELED，作业取消。",
    14:"JOBSTAT_LOST，作业丢失。",
    15:"JOBSTAT_SCALING，作业正在扩容"
}

def free_gpu_useble(sess):
    """
    检查免费 gpu 是否可用
    :param sess: modelarts session
    :return: if useable return True， eles return False
    """
    algo_info = Estimator.get_train_instance_types(modelarts_session=sess)  # 获取资源规格列表
    if 'modelarts.vm.gpu.free' in algo_info:
        return True
    return False

def upload2obs(sess, local_file, obs_location):
    path = Path(local_file)
    if path.is_file() and path.exists():
        sess.obs.upload_file(src_local_file=local_file, dst_obs_dir=obs_location)
    elif path.is_dir() and path.exists():
        sess.obs.upload_dir(src_local_dir=local_file, dst_obs_dir=obs_location)
    else:
        raise FileNotFoundError("File {} is not exist".format(local_file))

def framework_list(sess):
    """
    return :
        [{'framework_type': 'Ascend-Powered-Engine', 'framework_version': 'Mindspore-1.1.1-python3.7-aarch64'},
         {'framework_type': 'Ascend-Powered-Engine', 'framework_version': 'TF-1.15-python3.7-aarch64'},
         {'framework_type': 'Caffe', 'framework_version': 'Caffe-1.0.0-python2.7'},
         {'framework_type': 'MXNet', 'framework_version': 'MXNet-1.2.1-python2.7'},
         {'framework_type': 'MXNet', 'framework_version': 'MXNet-1.2.1-python3.6'},
         {'framework_type': 'MindSpore-GPU', 'framework_version': 'MindSpore-1.1.0-python3.7'},
         {'framework_type': 'PyTorch', 'framework_version': 'PyTorch-1.0.0-python2.7'},
         {'framework_type': 'PyTorch', 'framework_version': 'PyTorch-1.0.0-python3.6'},
         {'framework_type': 'PyTorch', 'framework_version': 'PyTorch-1.3.0-python2.7'},
         {'framework_type': 'PyTorch', 'framework_version': 'PyTorch-1.3.0-python3.6'},
         {'framework_type': 'PyTorch', 'framework_version': 'PyTorch-1.4.0-python3.6'},
         {'framework_type': 'Ray', 'framework_version': 'RAY-0.7.4-python3.6'},
         {'framework_type': 'Spark_MLlib', 'framework_version': 'Spark-2.3.2-python2.7'},
         {'framework_type': 'Spark_MLlib', 'framework_version': 'Spark-2.3.2-python3.6'},
         {'framework_type': 'TensorFlow', 'framework_version': 'TF-1.13.1-python2.7'},
         {'framework_type': 'TensorFlow', 'framework_version': 'TF-1.13.1-python3.6'},
         {'framework_type': 'TensorFlow', 'framework_version': 'TF-1.8.0-python2.7'},
         {'framework_type': 'TensorFlow', 'framework_version': 'TF-1.8.0-python3.6'},
         {'framework_type': 'TensorFlow', 'framework_version': 'TF-2.1.0-python3.6'},
         {'framework_type': 'XGBoost-Sklearn', 'framework_version': 'XGBoost-0.80-Sklearn-0.18.1-python2.7'},
         {'framework_type': 'XGBoost-Sklearn', 'framework_version': 'XGBoost-0.80-Sklearn-0.18.1-python3.6'}]"
    """
    avaiable_framework_list = Estimator.get_framework_list(modelarts_session=sess)
    print("avaiable framework list")
    pprint.pprint(avaiable_framework_list)
    return avaiable_framework_list
