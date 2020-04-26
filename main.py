import os
import math
import json
import logging
from shutil import copy

import tensorflow as tf

from train import train
from config import conf
from utils import get_environment_variable, create_directory

if __name__ == '__main__':
    # device placement
    tf.debugging.set_log_device_placement(conf.log.device_placement)

    # set distribution strategy
    if conf.general.strategy == 'default':
        conf.general.strategy = tf.distribute.get_strategy()
        conf.general.is_chief = True
        conf.general.is_cluster = False
        conf.general.nnodes = 1
    elif conf.general.strategy == 'mirrored':
        conf.general.strategy = tf.distribute.MirroredStrategy()
        conf.general.is_chief = True
        conf.general.is_cluster = False
        conf.general.nnodes = 1
    elif conf.general.strategy == 'multimirrored':
        # parse info from $TF_CONFIG, index 0 is chief by definition, chief is also a worker (handling logging etc.)
        conf.general.strategy = tf.distribute.experimental.MultiWorkerMirroredStrategy()
        tf_config = get_environment_variable('TF_CONFIG')
        tf_config = json.loads(tf_config)
        conf.general.nnodes = len(tf_config['cluster']['worker'])
        conf.general.is_chief = tf_config['task']['index'] == 0
        conf.general.is_cluster = True

    # store certain attributes in configs
    conf.general.summary = tf.summary.create_file_writer(conf.general.log_dir)
    conf.model.final_stage = int(math.log2(conf.model.resolution))
    conf.model.alpha = conf.train.alpha_init
    conf.log.log_file_path = os.path.join(conf.general.log_dir, conf.log.filename)
    conf.data.data_dir = conf.general.data_dir

    # resolve caching file (incorrect configuration might lead to OOM - log and print warning)
    if conf.data.caching:
        if conf.data.cache_file:
            if os.path.exists(conf.data.cache_file):
                raise FileExistsError(f"cache_file='{conf.data.cache_file}' already exists")
            logging.info(f"using cache_file='{conf.data.cache_file}' for dataset caching")
        else:
            msg = f"dataset caching is activated and cache_file was specified as \"\". TensorFlow will attempt to " \
                  f"load the entire dataset into system memory. In case of OOM specify a temporary cache_file."
            print(msg)
            logging.warning(msg)

    # chief creates directories as well as logfile
    if conf.general.is_chief and (conf.general.save or conf.general.evaluate):
        create_directory(conf.general.out_dir)
    if conf.general.is_chief and conf.general.logging:
        create_directory(conf.general.log_dir)
        logging.basicConfig(filename=conf.log.log_file_path, format=conf.log.format, level=conf.log.level, datefmt=conf.log.datefmt)
        if conf.log.adapt_tf_logger:
            tf_log = tf.get_logger()
            file_hdlr = logging.FileHandler(filename=conf.log.log_file_path, mode='w')
            file_hdlr.setFormatter(logging.Formatter(fmt=conf.log.format, datefmt=conf.log.datefmt))
            tf_log.addHandler(hdlr=file_hdlr)
            tf_log.setLevel(conf.log.tf_level)
        logging.info("Successfully set up logging")
        logging.info(f"XLA Compiler is {'disabled' if not conf.general.XLA else 'enabled'}")
        logging.info(f"Number of nodes in sync: {conf.general.nnodes}")
        logging.info(f"Number of replicas in sync: {conf.general.strategy.num_replicas_in_sync}")
        config_backup_file = os.path.join(conf.general.out_dir, os.path.basename(conf.general.config_file))
        copy(src=conf.general.config_file, dst=config_backup_file)
        logging.info(f"Backed up {conf.general.config_file} under {config_backup_file}")
        logging.info(f"Started {__name__}")

    # start job
    with conf.general.summary.as_default():
        train()

    # job done
    logging.info(f"{__name__} successfully terminated - job done!")
