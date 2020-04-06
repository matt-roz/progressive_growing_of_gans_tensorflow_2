import os
import sys
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
    resolution_choices = [2**i for i in range(11)]

    # input directories
    dir_parser = parser.add_argument_group(title="directory arguments")
    dir_parser.add_argument('--out-dir', dest="outdir", type=str, default=output_dir,
                            help="root directory for outputs (default: '%(default)s')")
    dir_parser.add_argument('--data-dir', dest="datadir", type=str, default=data_dir,
                            help="root directory for data input (default: '%(default)s')")

    # output, saving
    store_parser = parser.add_argument_group(title="output arguments")
    store_parser.add_argument('--no-save', dest="saving", default=True, action="store_false", help="deactivates saving")
    store_parser.add_argument('--no-eval', dest="evaluate", default=True, action="store_false",
                              help="deactivates evaluation")

    # model configuration
    model_parser = parser.add_argument_group(title="model arguments")
    model_parser.add_argument('--epochs', dest='epochs', type=int, default=25,
                              help="depicts default number of epochs to train for (default: '%(default)s')")
    model_parser.add_argument('--min-resolution', dest='minresolution', type=int, choices=resolution_choices,
                              default=resolution_choices[2], help="starting resolution for progressive GAN "
                              "(default: '%(default)s')")
    model_parser.add_argument('--max-resolution', dest='maxresolution', type=int, choices=resolution_choices,
                              default=resolution_choices[-1], help="stopping resolution for progressive GAN "
                              "(default: '%(default)s')")
    model_parser.add_argument('--noise-dim', dest="noisedim", type=int, default=512,
                              help="noise dim for generator to create images from (default: '%(default)s')")
    model_parser.add_argument('--alpha-step', dest='alphastep', type=float, default=0.001,
                              help="alpha step for soothing in new layers (default: '%(default)s')")

    # batch_size input
    batch_size_group = model_parser.add_mutually_exclusive_group(required=True)
    batch_size_group.add_argument('--batch-size', dest='globalbatchsize', type=int, default=0,
                                  help="depicts default global_batch_size to train with")
    batch_size_group.add_argument('--replica-batch-size', dest='replicabatchsize', type=int, default=0,
                                  help="depicts default per replica_batch_size to train with")
    batch_size_group.add_argument('--node-batch-size', dest='nodebatchsize', type=int, default=0,
                                  help="depicts default per node_batch_size to train with")

    # tensorflow optimizer hyperparams
    optim_parser = parser.add_argument_group(title="optimizer arguments")
    optim_parser.add_argument('--learning-rate', dest='learningrate', type=float, default=0.001,
                              help="learningrate to train both AdamOptimizers with (default: '%(default)s')")
    optim_parser.add_argument('--beta1', dest="beta1", type=float, default=0.0,
                              help="beta1 momentum to train both AdamOptimizers with (default: '%(default)s')")
    optim_parser.add_argument('--beta2', dest="beta2", type=float, default=0.99,
                              help="beta2 momentum to train both AdamOptimizers with (default: '%(default)s')")
    optim_parser.add_argument('--epsilon', dest="epsilon", type=float, default=1e-8,
                              help="epsilon to train both AdamOptimizers with (default: '%(default)s')")
    optim_parser.add_argument('--disc-repeats', dest="discrepeats", type=float, default=1.0,
                              help="learningrate advantage of discriminator over generator (default: '%(default)s')")

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
    data_parser.add_argument('--buffer-size', dest="buffersize", type=int, default=5000,
                             help="buffersize for TensorFlow Dataset shuffle (default: '%(default)s')")
    data_parser.add_argument('--map-parallel-calls', dest="mapcalls", type=int,
                             default=tf.data.experimental.AUTOTUNE, help="num threads for mapping functions "
                             "over dataset (default: 'tf.data.experimental.AUTOTUNE')")
    data_parser.add_argument('--prefetch-parallel-calls', dest="prefetchcalls", type=int,
                             default=tf.data.experimental.AUTOTUNE, help="num threads for prefetching "
                             "dataset to accelerators (default: 'tf.data.experimental.AUTOTUNE')")
    data_parser.add_argument('--interleave-parallel-calls', dest="interleavecalls", type=int,
                             default=tf.data.experimental.AUTOTUNE, help="num threads for interleaving "
                             "dataset (default: 'tf.data.experimental.AUTOTUNE')")
    data_parser.add_argument('--cache', dest="caching", default=False, action="store_true",
                             help="activates TensorFlow's dataset caching (default: '%(default)s')")
    data_parser.add_argument('--cache-file', dest="cachefile", type=str, default=cache_file_path,
                             help="path to temporary cachefile or set to empty string \"\" to cache entire dataset in "
                             "system memory (default: '%(default)s')")

    # TensorBoard callbacks
    tboard_parser = parser.add_argument_group(title="tensorboard callback arguments")
    tboard_parser.add_argument('--early-stopping', dest='earlystopping', type=int, default=0,
                               help="stops train when val_loss stopped improving for number of epochs. 0 = disabled "
                               "(default: '%(default)s')")
    tboard_parser.add_argument('--checkpoint-frequency', dest='checkpointfrequency', type=int, default=1,
                               help="defines ModelCheckpoint save frequency. 0 = disabled (default: '%(default)s')")
    tboard_parser.add_argument('--eval-frequency', dest='evalfrequency', type=int, default=1,
                               help="defines generator image evaluation frequency. 0 = disabled "
                               "(default: '%(default)s')")
    tboard_parser.add_argument('--log-frequency', dest='logfrequency', type=str, choices=['epoch', 'batch', ''],
                               default='epoch', help="defines TensorBoard logging frequency. \"\" = disabled "
                               "(default: '%(default)s')")

    # logging configuration
    log_parser = parser.add_argument_group(title="logging/debug arguments")
    log_parser.add_argument('--device-placement', dest='deviceplacement', default=False, action="store_true",
                            help="activates TensorFlow device placement")
    log_parser.add_argument('--no-log', dest="logging", default=True, action="store_false", help="deactivates logging")
    log_parser.add_argument('--log-level', dest='loglevel', type=str, default=log_level_choices[0],
                            choices=log_level_choices, help="depicts logging level (default: '%(default)s')")
    log_parser.add_argument('--log-dir', dest="logdir", type=str, default=output_dir,
                            help="depicts root directory for logging (default: '%(default)s')")
    log_parser.add_argument('--log-filename', dest='logfile', type=str, default=log_file,
                            help="name of log file (default: '%(default)s')")
    log_parser.add_argument('--log-format', dest='logformat', type=str,
                            default='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                            help="defines logger message format (default: '%(default)s')")
    log_parser.add_argument('--log-date-format', dest='logdateformat', type=str, default='%m/%d/%Y %I:%M:%S %p',
                            help="defines logger timestamp format (default: '%(default)s')")

    # parse
    args = parser.parse_args()

    # tensorflow execution mode
    if args.neager:
        tf.compat.v1.disable_eager_execution()

    # device placement
    tf.debugging.set_log_device_placement(args.deviceplacement)

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
    if args.globalbatchsize:
        args.nodebatchsize = args.globalbatchsize // args.nnodes
        args.replicabatchsize = args.nodebatchsize // (args.ngpus if args.ngpus else 1)
    elif args.nodebatchsize:
        args.replicabatchsize = args.nodebatchsize // (args.ngpus if args.ngpus else 1)
        args.globalbatchsize = args.nodebatchsize * args.nnodes
    elif args.replicabatchsize:
        args.nodebatchsize = args.replicabatchsize * (args.ngpus if args.ngpus else 1)
        args.globalbatchsize = args.nodebatchsize * args.nnodes

    # parse path inputs
    if args.is_cluster:
        slurm_job_id = get_environment_variable('SLURM_JOBID')
        dir_suffix = f"{slurm_job_id}"
    else:
        dir_suffix = f"{timestamp}-{host}"
    args.logdir = os.path.join(os.path.abspath(os.path.realpath(os.path.expanduser(args.logdir))), dir_suffix)
    args.outdir = os.path.join(os.path.abspath(os.path.realpath(os.path.expanduser(args.outdir))), dir_suffix)
    args.datadir = os.path.abspath(os.path.realpath(os.path.expanduser(args.datadir)))
    # args.cachefile might be "" if user wants to cache entire dataset in memory
    if args.cachefile:
        args.cachefile = os.path.abspath(os.path.realpath(os.path.expanduser(args.cachefile)))
    log_file_path = os.path.join(args.logdir, args.logfile)

    # store certain attributes in args
    args.sourcedir = source_dir
    args.workingdir = working_dir
    args.timestamp = timestamp
    args.host = host
    args.summary = tf.summary.create_file_writer(args.logdir)

    # chief creates directories as well as logfile
    if args.is_chief and args.logging:
        create_directory(args.logdir)
        logging.basicConfig(filename=log_file_path, format=args.logformat, level=args.loglevel, datefmt=args.logdateformat)
        logging.info(f"{host}: successfully set up logging")
        logging.info(f"{host}: TensorFlow Eager Execution is {'disabled' if args.neager else 'enabled'}.")
        logging.info(f"{host}: XLA Compiler is {'disabled' if not args.XLA else 'enabled'}.")
        if args.is_cluster:
            logging.info(f"{host}: {len(gpu_physical_devices)}-GPU devices: physical={gpu_physical_devices}")
            logging.info(f"{host}: number of nodes in sync: {args.nnodes}")
            logging.info(f"{host}: number of replicas in sync: {args.strategy.num_replicas_in_sync}")
            logging.info(f"{host}: global_batch_size={args.globalbatchsize}, node_batch_size={args.nodebatchsize}, "
                         f"replica_batch_size={args.replicabatchsize}")
        logging.debug(f"{host}: started {__name__} with args={vars(args)}")
    if args.is_chief and (args.saving or args.evaluate):
        create_directory(args.outdir)

    # assert certain invalid inputs
    assert args.minresolution <= args.maxresolution, f"--min-resolution must not exceed --max-resolution"

    # resolve caching file, log configuration for user (incorrect configuration might lead to OOM)
    if args.caching:
        if args.cachefile:
            if os.path.exists(args.cachefile):
                raise FileExistsError(f"--cache-file {args.cachefile} already exists")
            logging.info(f"using dataset_cache_file={args.cachefile} for dataset caching")
        else:
            msg = f"dataset caching is activated with --cache and --cache-file was specified as \"\". TensorFlow will "\
                  f"attempt to load the entire dataset into memory. In case of OOM specify a temporary cachefile!"
            logging.warning(msg)

    # start job
    train(args)

    # job done
    print(f"{host}: {__name__} terminated")
    logging.info(f"{host}: {__name__} terminated")
