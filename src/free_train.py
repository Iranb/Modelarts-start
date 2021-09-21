import uuid
import yaml
import pprint
from config import config
from utils import free_gpu_useble, upload2obs
from modelarts.session import Session
from modelarts.estimator import Estimator

if __name__ == "__main__":
    # 1. first create session and load config
    print(10*"=", "init train session and load config", 10*"=")
    pprint.pprint(config)
    session = Session(access_key=config.AK,secret_key=config.SK, project_id=config.PROJECT_ID, region_name=config.REGION_NAME)

    # 2. upload your code to obs file
    upload2obs(session, "code/", config.CODE_UPLOAD_URL)

    # 3. create a train job and update it's version
    if not free_gpu_useble:
        raise Exception("free gpu is not useable")

    # 3.1 create train mission
    estimator = Estimator(
                          modelarts_session=session,
                          framework_type=config.TRAIN.FRAMEWORK_TYPE,                   # AI引擎名称
                          framework_version=config.TRAIN.FRAME_WORK_TYPE,               # AI引擎版本
                          code_dir=config.TRAIN.CODE_OBS_DIR,                           # 训练脚本目录
                          boot_file=config.TRAIN.BOOT_FILE,                             # 训练启动脚本目录
                          log_url=config.TRAIN.LOG_OBS_PATH,                            # 训练日志目录
                          hyperparameters=config.TRAIN.HYPERPARAMETERS,                 # 训练超参数
                          output_path=config.TRAIN.OUTPUT_OBS_PATH,                     # 训练输出目录
                          train_instance_type='modelarts.vm.gpu.free',                  # nvidia v100 with 32GB gpu ram
                          train_instance_count=1
    )
    print(10*"=","first test train pipeline", 10*"=")
    job_instance = estimator.fit(inputs=config.TRAIN.DATA_OBS_DIR, wait=True, job_name=config.TRAIN.JOB_NAME + "_0_____" + str(uuid.uuid1())) # wait to train end
    print(10*"=","train pipeline OK", 10*"=")

    log_file_list = job_instance.get_job_log_file_list()
    print("================= log =================")
    pprint.pprint(job_instance.get_job_log(log_file = log_file_list["log_file_list"][-1]).get("content"))
    print("================= end of log =================")

    job_info = job_instance.get_job_info()
    print("current job id:")
    job_id = job_instance.job_id
    print("version id: {}".format(job_info["version_id"]))
    version_id = job_info["version_id"]


    for i in range(1, config.TRAIN.CONTINUE_ITERS):
        print("current run version of {}".format(i), "with version_id {}".format(version_id))
        job_version_instance = estimator.create_job_version(job_id=job_id, pre_version_id=version_id,
                                                            inputs=config.TRAIN.DATA_OBS_DIR, wait=True,
                                                            job_desc='train of iter {}'.format(i))
        job_version_info = estimator.get_job_version_info()
        # print("show current job info")
        # pprint.pprint(job_version_info)
        job_id = job_version_instance.job_id
        version_id = job_version_instance.get_job_version_info()
        # log = job_version_instance.get_job_log()
        log_file_list = job_version_instance.get_job_log_file_list()
        print("================= log ========================")
        pprint.pprint(job_version_instance.get_job_log(log_file = log_file_list["log_file_list"][-1]).get("content"))
        print("================= end of log =================")
        print(20 * "=", "train of {} end ".format(i), 20 * "=")
    print(20 * "=", "ALL TRAIN JOB FINISHED SUCCESS", 20 * "=")



