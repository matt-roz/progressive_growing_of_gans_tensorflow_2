import os
import math
import json
import logging
from shutil import copy

import tensorflow as tf

from experimental_train_v2 import train
from config import conf
from utils import get_environment_variable, create_directory

if __name__ == '__main__':
    # device placement, global seed
    tf.debugging.set_log_device_placement(conf.log.device_placement)
    tf.random.set_seed(conf.general.global_seed)

    # set distribution strategy
    if conf.general.strategy == 'mirrored':
        # del os.environ['TF_CONFIG']
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
    elif isinstance(conf.general.strategy, list):
        # del os.environ['TF_CONFIG']
        conf.general.strategy = tf.distribute.MirroredStrategy(devices=conf.general.strategy)
        conf.general.is_chief = True
        conf.general.is_cluster = False
        conf.general.nnodes = 1
    else:
        msg = "strategy must be either a device list or one of ['mirrored', 'multimirrored'] but found "
        msg += f"{conf.general.strategy} instead"
        raise RuntimeError(msg)

    # store certain attributes in configs that are only determined at runtime
    conf.model.final_stage = int(math.log2(conf.model.resolution))
    conf.model.alpha = conf.train.alpha_init
    conf.model.alpha_step = (1.0 - conf.train.alpha_init) / (conf.train.epochs_per_stage * conf.data.num_examples / 2)
    conf.data.data_dir = conf.general.data_dir
    conf.log.log_file_path = os.path.join(conf.general.log_dir, conf.log.filename)

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
    if conf.general.is_chief and conf.general.logging:
        create_directory(conf.general.log_dir)
        logging.basicConfig(filename=conf.log.log_file_path, format=conf.log.format, level=conf.log.level, datefmt=conf.log.datefmt)
        logging.info("Successfully set up logging")
        logging.info(f"XLA Compiler is {'disabled' if not conf.general.XLA else 'enabled'}")
        logging.info(f"Number of nodes in sync: {conf.general.nnodes}")
        logging.info(f"Number of replicas in sync: {conf.general.strategy.num_replicas_in_sync}")
        conf.general.summary = tf.summary.create_file_writer(conf.general.log_dir)
    if conf.general.is_chief and (conf.general.save or conf.general.evaluate):
        create_directory(conf.general.out_dir)
        config_backup_file = os.path.join(conf.general.out_dir, os.path.basename(conf.general.config_file))
        copy(src=conf.general.config_file, dst=config_backup_file)
        logging.info(f"Backed up {conf.general.config_file} under {config_backup_file}")
    if conf.log.adapt_tf_logger:
        tf_log = tf.get_logger()
        tf_log.setLevel(conf.log.tf_level)
        if conf.general.is_chief and conf.general.logging:
            file_hdlr = logging.FileHandler(filename=conf.log.log_file_path, mode='w')
            file_hdlr.setFormatter(logging.Formatter(fmt=conf.log.format, datefmt=conf.log.datefmt))
            tf_log.addHandler(hdlr=file_hdlr)

    # start job
    logging.info(f"Started {__name__} with config: {conf}")
    train()

    # job done
    logging.info(f"{__name__} successfully terminated - job done!")
