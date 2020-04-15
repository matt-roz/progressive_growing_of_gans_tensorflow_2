import os
import sys
import math
import json
import logging
import platform
import argparse
from datetime import datetime

import tensorflow as tf

from utils import get_environment_variable, create_directory
from train import train

if __name__ == '__main__':
    # directories and paths
    source_dir = os.path.abspath(os.path.realpath(os.path.expanduser(sys.path[0])))
    working_dir = os.path.abspath(os.path.realpath(os.path.expanduser(os.getcwd())))
    data_dir = os.path.abspath(os.path.realpath(os.path.expanduser('~/tensorflow_datasets')))
    output_dir = os.path.join(source_dir, "outs")

    # log and temporary files
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    host = platform.node()
    log_file = f"{timestamp}-{host}-logfile.log"
    cache_file_path = os.path.join("/tmp", f"{timestamp}-tf-dataset.cache")

    # argparsing and parsing choices
    parser = argparse.ArgumentParser()
    tf_strategy_choices = ['default', 'mirrored', 'multimirrored']
    log_level_choices = ['INFO', 'CRITICAL', 'ERROR', 'WARNING', 'DEBUG', 'NOTSET']
    data_split_choices = ['train', 'test', 'validation']
    resolution_choices = [2**i for i in range(2, 11)]

    # input directories
    dir_parser = parser.add_argument_group(title="directory arguments")
    dir_parser.add_argument('--out-dir', dest="out_dir", type=str, default=output_dir,
                            help="root directory for outputs (default: '%(default)s')")
    dir_parser.add_argument('--data-dir', dest="data_dir", type=str, default=data_dir,
                            help="root directory for data input (default: '%(default)s')")

    # output, saving
    store_parser = parser.add_argument_group(title="output arguments")
    store_parser.add_argument('--no-save', dest="save", default=True, action="store_false", help="deactivates saving")
    store_parser.add_argument('--no-eval', dest="evaluate", default=True, action="store_false",
                              help="deactivates evaluation")

    # model configuration
    model_parser = parser.add_argument_group(title="model arguments")
    model_parser.add_argument('--epochs', dest='epochs', type=int, default=700,
                              help="depicts default number of epochs to train for (default: '%(default)s')")
    model_parser.add_argument('--epochs-per-stage', dest='epochs_per_stage', type=int, default=54,
                              help="epochs until stage is increased (default: '%(default)s')")
    model_parser.add_argument('--leaky-alpha', dest='leaky_alpha', type=float, default=0.2,
                              help="alpha for LeakyReLU activations (default: '%(default)s')")
    model_parser.add_argument('--resolution', dest='resolution', type=int, choices=resolution_choices,
                              default=resolution_choices[-1], help="stopping resolution for progressive GAN "
                              "(default: '%(default)s')")
    model_parser.add_argument('--noise-dim', dest="noise_dim", type=int, default=512,
                              help="noise dim for generator to create images from (default: '%(default)s')")
    model_parser.add_argument('--wgan-lambda', dest='wgan_lambda', type=float, default=10.0,
                              help="lambda for discriminator gradient penalty loss (default: '%(default)s')")
    model_parser.add_argument('--wgan-target', dest='wgan_target', type=float, default=1.0,
                              help="target for discriminator gradient penalty loss loss (default: '%(default)s')")
    model_parser.add_argument('--wgan-epsilon', dest='wgan_epsilon', type=float, default=0.001,
                              help="epsilon for discriminator epsilon drift loss loss (default: '%(default)s')")
    model_parser.add_argument('--no-bias', dest='use_bias', action='store_false', default=True,
                              help="deactivates use of bias in all layers")
    model_parser.add_argument('--no-weight-scaling', dest='use_weight_scaling', action='store_false', default=True,
                              help="deactivates use of weight scaling")
    model_parser.add_argument('--no-alpha-smoothing', dest='use_alpha_smoothing', action='store_false', default=True,
                              help="deactivates last block alpha smoothing")
    model_parser.add_argument('--no-stages', dest='use_stages', action='store_false', default=True,
                              help="deactivates progressive training (only trains final stage)")
    model_parser.add_argument('--no-epsilon-drift', dest='use_epsilon_drift', action='store_false', default=True,
                              help="deactivates epsilon drift for discriminator loss (will be set to 0.0)")
    model_parser.add_argument('--no-gradient-penalty', dest='use_gradient_penalty', action='store_false', default=True,
                              help="deactivates gradient penalty for discriminator loss (will be set to 0.0)")
    # batch_size input
    batch_size_group = model_parser.add_mutually_exclusive_group(required=True)
    batch_size_group.add_argument('--batch-size', dest='global_batch_size', type=int, default=0,
                                  help="depicts default global_batch_size to train with")
    batch_size_group.add_argument('--replica-batch-size', dest='replica_batch_size', type=int, default=0,
                                  help="depicts default per replica_batch_size to train with")
    batch_size_group.add_argument('--node-batch-size', dest='node_batch_size', type=int, default=0,
                                  help="depicts default per node_batch_size to train with")

    # tensorflow optimizer hyperparams
    optim_parser = parser.add_argument_group(title="optimizer arguments")
    optim_parser.add_argument('--learning-rate', dest='learning_rate', type=float, default=0.001,
                              help="learningrate to train both AdamOptimizers with (default: '%(default)s')")
    optim_parser.add_argument('--beta1', dest="beta1", type=float, default=0.0,
                              help="beta1 momentum to train both AdamOptimizers with (default: '%(default)s')")
    optim_parser.add_argument('--beta2', dest="beta2", type=float, default=0.99,
                              help="beta2 momentum to train both AdamOptimizers with (default: '%(default)s')")
    optim_parser.add_argument('--adam-epsilon', dest="adam_epsilon", type=float, default=1e-8,
                              help="epsilon to train both AdamOptimizers with (default: '%(default)s')")
    optim_parser.add_argument('--disc-repeats', dest="disc_repeats", type=float, default=1.0,
                              help="learning_rate advantage of discriminator over generator (default: '%(default)s')")

    # tensorflow execution mode
    exec_parser = parser.add_argument_group(title="execution mode arguments")
    exec_parser.add_argument('--no-eager', dest="neager", default=False, action="store_true",
                             help="deactivates eager execution (activates TF2.0+ graph mode) (default: '%(default)s')")
    exec_parser.add_argument('--XLA', dest='XLA', default=False, action="store_true",
                             help="activates XLA JIT compilation (default: '%(default)s')")
    exec_parser.add_argument('--strategy', dest='strategy', type=str, choices=tf_strategy_choices,
                             default=tf_strategy_choices[0], help="depicts distribution strategy "
                             "(default: '%(default)s')")

    # tensorflow data input pipeline
    data_parser = parser.add_argument_group(title="data pipeline arguments")
    data_parser.add_argument('--split', dest="split", choices=data_split_choices, default=data_split_choices[0],
                             help="depicts the data split for tensorflow_datasets to load (default: '%(default)s')")
    data_parser.add_argument('--buffer-size', dest="buffer_size", type=int, default=5000,
                             help="buffersize for TensorFlow Dataset shuffle (default: '%(default)s')")
    data_parser.add_argument('--map-parallel-calls', dest="map_calls", type=int,
                             default=tf.data.experimental.AUTOTUNE, help="num threads for mapping functions "
                             "over dataset (default: 'tf.data.experimental.AUTOTUNE')")
    data_parser.add_argument('--prefetch-parallel-calls', dest="prefetch_calls", type=int,
                             default=tf.data.experimental.AUTOTUNE, help="num threads for prefetching "
                             "dataset to accelerators (default: 'tf.data.experimental.AUTOTUNE')")
    data_parser.add_argument('--interleave-parallel-calls', dest="interleave_calls", type=int,
                             default=tf.data.experimental.AUTOTUNE, help="num threads for interleaving "
                             "dataset (default: 'tf.data.experimental.AUTOTUNE')")
    data_parser.add_argument('--cache', dest="caching", default=False, action="store_true",
                             help="activates TensorFlow's dataset caching (default: '%(default)s')")
    data_parser.add_argument('--cache-file', dest="cache_file", type=str, default=cache_file_path,
                             help="path to temporary cachefile or set to empty string \"\" to cache entire dataset in "
                             "system memory (default: '%(default)s')")

    # TensorBoard callbacks
    tboard_parser = parser.add_argument_group(title="tensorboard callback arguments")
    # tboard_parser.add_argument('--early-stopping', dest='early_stopping', type=int, default=0,
    #                           help="stops train when val_loss stopped improving for number of epochs. 0 = disabled "
    #                           "(default: '%(default)s')")
    tboard_parser.add_argument('--checkpoint-freq', dest='checkpoint_freq', type=int, default=27,
                               help="defines ModelCheckpoint save frequency. 0 = disabled (default: '%(default)s')")
    tboard_parser.add_argument('--eval-freq', dest='eval_freq', type=int, default=1,
                               help="defines generator image evaluation frequency. 0 = disabled "
                               "(default: '%(default)s')")
    tboard_parser.add_argument('--log-freq', dest='log_freq', type=str, choices=['epoch', 'batch', ''],
                               default='epoch', help="defines TensorBoard logging frequency. \"\" = disabled "
                               "(default: '%(default)s')")

    # logging configuration
    log_parser = parser.add_argument_group(title="logging/debug arguments")
    log_parser.add_argument('--device-placement', dest='device_placement', default=False, action="store_true",
                            help="activates TensorFlow device placement")
    log_parser.add_argument('--no-log', dest="logging", default=True, action="store_false", help="deactivates logging")
    log_parser.add_argument('--log-level', dest='log_level', type=str, default=log_level_choices[0],
                            choices=log_level_choices, help="depicts logging level (default: '%(default)s')")
    log_parser.add_argument('--log-dir', dest="log_dir", type=str, default=output_dir,
                            help="depicts root directory for logging (default: '%(default)s')")
    log_parser.add_argument('--log-filename', dest='log_file', type=str, default=log_file,
                            help="name of log file (default: '%(default)s')")
    log_parser.add_argument('--log-format', dest='log_format', type=str,
                            default='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                            help="defines logger message format (default: '%(default)s')")
    log_parser.add_argument('--log-date-format', dest='log_date_format', type=str, default='%m/%d/%Y %I:%M:%S %p',
                            help="defines logger timestamp format (default: '%(default)s')")

    # parse
    args = parser.parse_args()

    # tensorflow execution mode
    if args.neager:
        tf.compat.v1.disable_eager_execution()

    # device placement
    tf.debugging.set_log_device_placement(args.device_placement)

    # set distribution strategy
    if args.strategy == 'default':
        args.strategy = tf.distribute.get_strategy()
        args.is_chief = True
        args.is_cluster = False
        args.nnodes = 1
    elif args.strategy == 'mirrored':
        args.strategy = tf.distribute.MirroredStrategy()
        args.is_chief = True
        args.is_cluster = False
        args.nnodes = 1
    elif args.strategy == 'multimirrored':
        # parse info from $TF_CONFIG, index 0 is chief by definition, chief is also a worker (handling logging etc.)
        args.strategy = tf.distribute.experimental.MultiWorkerMirroredStrategy()
        tf_config = get_environment_variable('TF_CONFIG')
        tf_config = json.loads(tf_config)
        args.nnodes = len(tf_config['cluster']['worker'])
        args.is_chief = tf_config['task']['index'] == 0
        args.is_cluster = True

    # get local gpu devices
    gpu_physical_devices = tf.config.experimental.list_physical_devices('GPU')
    gpu_logical_devices = tf.config.experimental.list_logical_devices('GPU')
    args.ngpus = len(gpu_physical_devices)

    # GPU-cluster config assertions (further batch_size logic does not support CPU-cluster computations)
    if args.is_cluster:
        assert args.ngpus > 0, f"{host}: could not resolve local GPU devices for cluster computation"

    # set batch_size based on user input (if no gpu devices are found, each batch is computed on one local CPU device)
    if args.global_batch_size:
        args.node_batch_size = args.global_batch_size // args.nnodes
        args.replica_batch_size = args.node_batch_size // (args.ngpus if args.ngpus else 1)
    elif args.node_batch_size:
        args.replica_batch_size = args.node_batch_size // (args.ngpus if args.ngpus else 1)
        args.global_batch_size = args.node_batch_size * args.nnodes
    elif args.replica_batch_size:
        args.node_batch_size = args.replica_batch_size * (args.ngpus if args.ngpus else 1)
        args.global_batch_size = args.node_batch_size * args.nnodes

    # parse path inputs
    if args.is_cluster:
        slurm_job_id = get_environment_variable('SLURM_JOBID')
        dir_suffix = f"{slurm_job_id}"
    else:
        dir_suffix = f"{timestamp}-{host}"
    args.log_dir = os.path.join(os.path.abspath(os.path.realpath(os.path.expanduser(args.log_dir))), dir_suffix)
    args.out_dir = os.path.join(os.path.abspath(os.path.realpath(os.path.expanduser(args.out_dir))), dir_suffix)
    args.data_dir = os.path.abspath(os.path.realpath(os.path.expanduser(args.data_dir)))
    # args.cache_file might be "" if user wants to cache entire dataset in memory
    if args.cache_file:
        args.cache_file = os.path.abspath(os.path.realpath(os.path.expanduser(args.cache_file)))
    log_file_path = os.path.join(args.log_dir, args.log_file)

    # store certain attributes in args
    args.summary = tf.summary.create_file_writer(args.log_dir)
    args.stop_stage = int(math.log2(args.resolution))
    args.alpha = 0.0

    # chief creates directories as well as logfile
    if args.is_chief and args.logging:
        create_directory(args.log_dir)
        logging.basicConfig(filename=log_file_path, format=args.log_format, level=args.log_level, datefmt=args.log_date_format)
        # tf.get_logger().setLevel('ERROR')
        logging.info(f"{host}: successfully set up logging")
        logging.info(f"{host}: TensorFlow Eager Execution is {'disabled' if args.neager else 'enabled'}.")
        logging.info(f"{host}: XLA Compiler is {'disabled' if not args.XLA else 'enabled'}.")
        if args.is_cluster:
            logging.info(f"{host}: {len(gpu_physical_devices)}-GPU devices: physical={gpu_physical_devices}")
            logging.info(f"{host}: number of nodes in sync: {args.nnodes}")
            logging.info(f"{host}: number of replicas in sync: {args.strategy.num_replicas_in_sync}")
            logging.info(f"{host}: global_batch_size={args.global_batch_size}, node_batch_size={args.node_batch_size}, "
                         f"replica_batch_size={args.replica_batch_size}")
        logging.debug(f"{host}: started {__name__} with args={vars(args)}")
    if args.is_chief and (args.save or args.evaluate):
        create_directory(args.out_dir)

    # resolve caching file, log configuration for user (incorrect configuration might lead to OOM)
    if args.caching:
        if args.cache_file:
            if os.path.exists(args.cache_file):
                raise FileExistsError(f"--cache-file {args.cache_file} already exists")
            logging.info(f"using dataset_cache_file={args.cache_file} for dataset caching")
        else:
            msg = f"dataset caching is activated with --cache and --cache-file was specified as \"\". TensorFlow will "\
                  f"attempt to load the entire dataset into memory. In case of OOM specify a temporary cachefile!"
            logging.warning(msg)

    # start job
    with args.summary.as_default():
        train(args)

    # job done
    print(f"{host}: {__name__} terminated")
    logging.info(f"{host}: {__name__} terminated")
