import pprint
from config import config
from utils import upload2obs
from modelarts.session import Session

if __name__ == "__main__":
    # 1. first create session and load config
    print(10*"=", "init train session and load config", 10*"=")
    pprint.pprint(config)
    session = Session(access_key=config.AK,secret_key=config.SK, project_id=config.PROJECT_ID, region_name=config.REGION_NAME)
    upload2obs(session, "your data location", config.TRAIN.DATA_OBS_DIR)
    # 上传你的数据文件到 config.TRAIN.DATA_OBS_DIR 这个obs 目录当中