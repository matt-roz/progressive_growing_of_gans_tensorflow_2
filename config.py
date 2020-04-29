import os
import sys
import platform
from datetime import datetime

import tensorflow as tf
from data import celeb_a_hq_process_func

# ----------------------------------------------------------------------------------------------------------------------
# Convenience class that behaves exactly like dict(), but allows accessing
# the keys and values using the attribute syntax, i.e., "mydict.key = value".
# taken from: https://github.com/tkarras/progressive_growing_of_gans/blob/master/config.py#L8
# ----------------------------------------------------------------------------------------------------------------------


class EasyDict(dict):
    def __init__(self, *args, **kwargs): super().__init__(*args, **kwargs)
    def __getattr__(self, name): return self[name]
    def __setattr__(self, name, value): self[name] = value
    def __delattr__(self, name): del self[name]


# ----------------------------------------------------------------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------------------------------------------------------------

timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
host = platform.node()

general_config = EasyDict()             # generic configurations/options for output, checkpoints, etc.
general_config.save                     = True      # bool: de-/activates model saving and checkpointing
general_config.evaluate                 = True      # bool: de-/activates model evaluation
general_config.logging                  = True      # bool: de-/activates file logging (incl. TensorBoard)
general_config.out_dir                  = os.path.join('/media', 'storage', 'outs', f'{timestamp}-{host}')
general_config.log_dir                  = os.path.join('/media', 'storage', 'outs', f'{timestamp}-{host}')
general_config.data_dir                 = os.path.join('/media', 'storage', 'tensorflow_datasets')
general_config.XLA                      = False     # bool: de-/activates XLA JIT compilation
general_config.strategy                 = 'default' # str: distribution strategy; options are ['default', 'mirrored', 'multimirrored']
general_config.checkpoint_freq          = 27        # uint: epoch frequency to checkpoint models with (0 = disabled)
general_config.eval_freq                = 1         # uint: epoch frequency to evaluate models with (0 = disabled)
general_config.log_freq                 = 1         # uint: epoch frequency to log with (0 = disabled)

model_config = EasyDict()               # configuration of model building
model_config.leaky_alpha                = 0.3       # float: leakiness of LeakyReLU activations
model_config.generator_ema              = 0.999     # float: exponential moving average of final_generator
model_config.resolution                 = 256       # uint: final resolution in [4, 8, 16, 32, 64, 128, 256, 512, 1024]
model_config.noise_dim                  = 512       # uint: noise_dim generator projects from
model_config.epsilon                    = 1e-8      # float: small constant for numerical stability in PixelNormalization as well as StandardDeviation Layer
model_config.use_bias                   = True      # bool: de-/activates usage of biases in all trainable layers
model_config.use_stages                 = True      # bool: de-/activates progressive training of model in stages (if deactivated only last stage for final resolution is trained)
model_config.use_fused_scaling          = True      # bool: de-/activates up- and downsampling of images via strides=(2, 2) in Conv2D and Conv2DTranspose (else UpSampling2D and AveragePooling2D is used)
model_config.use_weight_scaling         = True      # bool: de-/activates the weight scaling trick on Conv and Dense layers as proposed in https://arxiv.org/abs/1710.10196
model_config.use_alpha_smoothing        = True      # bool: de-/activates smoothing in an image from a previous block after increasing the model to a new stage as proposed in https://arxiv.org/abs/1710.10196
model_config.use_noise_normalization    = True      # bool: de-/activates pixel_normalization on noise input at generator start

train_config = EasyDict()               # configuration of train parameters
train_config.epochs                     = 432       # uint: number of epochs to train for
train_config.epochs_per_stage           = 54        # uint: number of epochs per stage; alpha is increased linearly from alpha_init to 1.0 in halfway through
train_config.alpha_init                 = 0.0       # float: initial alpha value to smooth in images from previous block after stage in model has been increased
train_config.use_epsilon_penalty        = True      # bool: de-/activates epsilon_drift_penalty applied to discriminator loss as described in https://arxiv.org/abs/1710.10196
train_config.drift_epsilon              = 0.001     # float: epsilon scalar for epsilon_drift_penalty as described in https://arxiv.org/abs/1710.10196
train_config.use_gradient_penalty       = True      # bool: de-/activates gradient_penalty applied to discriminator loss as described in https://arxiv.org/abs/1704.00028
train_config.wgan_lambda                = 10.0      # float: lambda scalar for gradient_penalty as described in https://arxiv.org/abs/1704.00028
train_config.wgan_target                = 1.0       # float: target scalar for gradient_penalty as described in https://arxiv.org/abs/1704.00028

data_config = EasyDict()                # configuration of data set pipeline
data_config.registered_name             = 'celeb_a_hq'                   # str: name argument for tensorflow_datasets.load
data_config.split                       = 'train'                        # str: split argument for tensorflow_datasets.load
data_config.num_examples                = 30000                          # uint: number of examples train dataset will contain according to loaded split
data_config.caching                     = False                          # bool: de-/activates dataset caching to file or system memory (see cache_file)
data_config.cache_file                  = os.path.join('/tmp', f'{timestamp}-tf-dataset.cache')  # str: ignored if caching is false, else location of temporary cache_file ("" = load entire dataset into system memory)
data_config.process_func                = celeb_a_hq_process_func        # callable: function to process each dataset entry with
data_config.map_parallel_calls          = tf.data.experimental.AUTOTUNE  # int: number of parallel entries to apply 'process_functions' asynchronously
data_config.prefetch_parallel_calls     = tf.data.experimental.AUTOTUNE  # int: number of parallel threads to prefetch entries with concurrently
data_config.interleave_parallel_calls   = tf.data.experimental.AUTOTUNE  # int: number of parallel threads to access dataset shards/files
data_config.replica_batch_sizes         = {2: 512, 3: 378, 4: 256, 5: 128, 6: 64, 7: 32, 8: 14, 9: 6, 10: 3}             # dict: batch_size at stage
data_config.buffer_sizes                = {2: 5000, 3: 5000, 4: 2500, 5: 1250, 6: 500, 7: 400, 8: 300, 9: 250, 10: 200}  # dict: buffer_size at stage

optimizer_config = EasyDict()           # configuration of adam optimizers for both generator and discriminator
optimizer_config.learning_rates         = {2: 1e-3, 3: 1e-3, 4: 1e-3, 5: 1e-3, 6: 1e-3, 7: 1e-3, 8: 1e-3, 9: 1e-3, 10: 1e-3}  # dict: learning_rate at stage
optimizer_config.beta1                  = 0.0   # float: exponential decay rate for the 1st moment estimates
optimizer_config.beta2                  = 0.99  # float: exponential decay rate for the 2nd moment estimates
optimizer_config.epsilon                = 1e-8  # float: small constant for numerical stability

log_config = EasyDict()                 # configuration of general purpose logging
log_config.device_placement             = False   # bool: de-/activates TensorFlow device placement
log_config.level                        = 'INFO'  # str: log level of project logger; options are ['INFO', 'CRITICAL', 'ERROR', 'WARNING', 'DEBUG', 'NOTSET']
log_config.filename                     = f'{timestamp}-{host}-logfile.log'
log_config.format                       = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
log_config.datefmt                      = '%m/%d/%Y %I:%M:%S %p'
log_config.adapt_tf_logger              = True    # bool: de-/activates overriding of tf_logger configuration
log_config.tf_level                     = 'ERROR' # str: log level of TensorFlow logging logger; options are ['INFO', 'CRITICAL', 'ERROR', 'WARNING', 'DEBUG', 'NOTSET']

conf = EasyDict()                       # configuration summary
conf.general                            = general_config
conf.model                              = model_config
conf.train                              = train_config
conf.data                               = data_config
conf.optimizer                          = optimizer_config
conf.log                                = log_config

# uncomment next line to apply full training according to the original contribution: https://arxiv.org/abs/1710.10196
# data_config.replica_batch_sizes = {2: 16, 3: 16, 4: 16, 5: 16, 6: 16, 7: 16, 8: 14, 9: 6, 10: 3}; model_config.resolution = 1024; train_config.epochs = 540

# ----------------------------------------------------------------------------------------------------------------------
# Placeholders (these configurations are automatically set at runtime)
# ----------------------------------------------------------------------------------------------------------------------

general_config.source_dir               = os.path.abspath(os.path.realpath(os.path.expanduser(sys.path[0])))
general_config.working_dir              = os.path.abspath(os.path.realpath(os.path.expanduser(os.getcwd())))
general_config.config_file              = os.path.abspath(os.path.realpath(os.path.expanduser(__file__)))
general_config.is_chief                 = None  # bool: whether or not local machine is chief worker
general_config.is_cluster               = None  # bool: whether or not project is run in cluster
general_config.nnodes                   = None  # uint: number of nodes training
general_config.summary                  = None  # tf.FileWriter: FileWriter instance for TensorBoard logging (only for chief)
data_config.data_dir                    = None  # str: path to data_dir (same as general_config.data_dir)
model_config.final_stage                = None  # uint: final model stage
model_config.alpha                      = None  # uint: current alpha for smoothing
model_config.alpha_step                 = None  # float: delta-alpha increase per image during training

