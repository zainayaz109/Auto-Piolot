import tensorflow
print(tensorflow.__version__)
from trainner import tusimple_lanenet_single_gpu_trainner as single_gpu_trainner
from local_utils.log_util import init_logger
from local_utils.config_utils import parse_config_utils

LOG = init_logger.get_logger(log_file_name_prefix='lanenet_train')
CFG = parse_config_utils.lanenet_cfg

LOG.info('Using multi gpu trainner ...')
worker = single_gpu_trainner.LaneNetTusimpleTrainer(cfg=CFG)

worker.train()