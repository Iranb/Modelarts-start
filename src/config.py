from easydict import EasyDict

config = EasyDict()
config.OBS_NAME = "ma-iranb"
config.CODE_UPLOAD_URL = "obs://ma-iranb/free_train/"
config.RUN_SCRIPT = "code/test.py"
config.AK = "LHXULDXMBQ7KZF2Z5AGG"
config.SK = "mYdC9OnGioLhzZs0lmiVQfi3PTzVOkeCZVojCrxL"
config.PROJECT_ID = "0a9bd4035700103b2f7ec000f6853aed"
config.REGION_NAME = 'cn-north-4' # do not change this line

config.TRAIN = EasyDict()
config.TRAIN.JOB_NAME = "test_code"
config.TRAIN.HYPERPARAMETERS = []  # json list [ {para_name1: value1}, {para_name2: value2}]
config.TRAIN.OUTPUT_OBS_PATH = "/{}/free_train/out/".format(config.OBS_NAME)   # must end with "/"
config.TRAIN.LOG_OBS_PATH = "/{}/free_train/log/".format(config.OBS_NAME)   # must end with "/"
config.TRAIN.BOOT_FILE = "/{}/free_train/{}".format(config.OBS_NAME, config.RUN_SCRIPT)
config.TRAIN.CODE_OBS_DIR = "/{}/free_train/code/".format(config.OBS_NAME)   # must end with "/"
config.TRAIN.FRAMEWORK_TYPE = "PyTorch" # see framework_list in in utils.__init__.py
config.TRAIN.FRAME_WORK_TYPE = 'PyTorch-1.4.0-python3.6'   # see utils.framework_list in utils.__init__.py
config.TRAIN.DATA_OBS_DIR = "/{}/free_train/data/".format(config.OBS_NAME)   # see utils.framework_list in utils.__init__.py

config.TRAIN.CONTINUE_ITERS = -1




