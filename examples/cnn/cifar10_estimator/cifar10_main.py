# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""ResNet model for classifying images from CIFAR-10 dataset.

Support single-host training with one or multiple devices.

ResNet as proposed in:
Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
Deep Residual Learning for Image Recognition. arXiv:1512.03385

CIFAR-10 as in:
http://www.cs.toronto.edu/~kriz/cifar.html


"""
from __future__ import division
from __future__ import print_function

import argparse
import datetime
import functools
import itertools
import os

# Silence tf for prettier logging of Bayesian Optimization
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

from bayes_opt import BayesianOptimization
from bayes_opt import UtilityFunction
import cifar10
import cifar10_model
import cifar10_utils
import numpy as np
import six
from six.moves import xrange  # pylint: disable=redefined-builtin

import tensorflow as tf
# import tensorflow_addons as tfa

# Setting verbosity to INFO will log training and evaluation details.
tf.logging.set_verbosity(tf.logging.ERROR)

import ray
from ray.tune import run, Trainable
from ray.tune.schedulers import AsyncHyperBandScheduler
from ray.tune.suggest.bayesopt import BayesOptSearch

import logging

logging.getLogger("tensorflow").setLevel(logging.ERROR)

parser = argparse.ArgumentParser()
parser.add_argument(
    "--data-dir",
    type=str,
    required=True,
    help="The directory where the CIFAR-10 input data is stored.",
)
parser.add_argument(
    "--job-dir",
    type=str,
    required=True,
    help="The directory where the model will be stored.",
)
parser.add_argument(
    "--variable-strategy",
    choices=["CPU", "GPU"],
    type=str,
    default="CPU",
    help="Where to locate variable operations",
)
parser.add_argument(
    "--num-gpus",
    type=int,
    default=1,
    help="The number of gpus used. Uses only CPU if set to 0.",
)
parser.add_argument(
    "--num-layers",
    type=int,
    default=20,
    help="The number of layers of the model.",
)
parser.add_argument(
    "--train-steps",
    type=int,
    default=80000,
    help="The number of steps to use for training.",
)
# parser.add_argument(
#     "--train-batch-size",
#     type=int,
#     default=128,
#     help="Batch size for training.",
# )
parser.add_argument(
    "--eval-batch-size",
    type=int,
    default=500,
    help="Batch size for validation.",
)
parser.add_argument(
    "--num-batches-for-eval",
    type=int,
    default=10,
    help="Number of batches for validation.",
)
# parser.add_argument(
#     "--momentum",
#     type=float,
#     default=0.9,
#     help="Momentum for MomentumOptimizer.",
# )
# parser.add_argument(
#     "--weight-decay",
#     type=float,
#     default=2e-4,
#     help="Weight decay for convolutions.",
# )
# parser.add_argument(
#     "--learning-rate",
#     type=float,
#     default=0.1,
#     help="""\
#     This is the inital learning rate value. The learning rate will decrease
#     during training. For more details check the model_fn implementation in
#     this file.\
#     """,
# )
parser.add_argument(
    "--use-distortion-for-training",
    type=bool,
    default=True,
    help="If doing image distortion for training.",
)
parser.add_argument(
    "--sync",
    action="store_true",
    default=False,
    help="""\
    If present when running in a distributed environment will run on sync mode.\
    """,
)
parser.add_argument(
    "--num-intra-threads",
    type=int,
    default=0,
    help="""\
    Number of threads to use for intra-op parallelism. When training on CPU
    set to 0 to have the system pick the appropriate number or alternatively
    set it to the number of physical CPU cores.\
    """,
)
parser.add_argument(
    "--num-inter-threads",
    type=int,
    default=0,
    help="""\
    Number of threads to use for inter-op parallelism. If set to 0, the
    system will pick an appropriate number.\
    """,
)
parser.add_argument(
    "--data-format",
    type=str,
    default=None,
    help="""\
    If not set, the data format best for the training device is used.
    Allowed values: channels_first (NCHW) channels_last (NHWC).\
    """,
)
parser.add_argument(
    "--log-device-placement",
    action="store_true",
    default=False,
    help="Whether to log device placement.",
)
# parser.add_argument(
#     "--batch-norm-decay",
#     type=float,
#     default=0.997,
#     help="Decay for batch norm.",
# )
# parser.add_argument(
#     "--batch-norm-epsilon",
#     type=float,
#     default=1e-5,
#     help="Epsilon for batch norm.",
# )

# Add arguments related to BayesOpt

parser.add_argument(
    "--smoke-test",
    action="store_true",
    default=False,
    help="Finish quickly for testing",
)
# parser.add_argument(
#     "--verbose", type=bool, default=False, help="Verbose output of training."
# )
parser.add_argument(
    "--strategy",
    type=str,
    default="proposed",
    help="Strategy for discretizing. Possible options are: basic, proposed.",
)
parser.add_argument(
    "--metric",
    type=str,
    default="accuracy",
    help="""\
    Whether to use accuracy or loss for Bayesian optimization.\
    """,
)
# TODO: better name?
parser.add_argument(
    "--precision",
    type=int,
    default=1000,
    help="""\
    Size of grid\
    """,
)
parser.add_argument(
    "--log-path",
    type=str,
    default=os.getcwd() + "/train.log",
    help="""
    """,
)
parser.add_argument(
    "--ray-address",
    type=str,
    default="",
    help="""
    """,
)
args = parser.parse_args()

# Filling in shared values here
hparams = {}
hparams["num_layers"] = args.num_layers
hparams["eval_batch_size"] = args.eval_batch_size
hparams["sync"] = args.sync
hparams["num_inter_threads"] = args.num_inter_threads
hparams["data_format"] = args.data_format


def get_model_fn(num_gpus, variable_strategy, num_workers):
    """Returns a function that will build the resnet model."""

    def _resnet_model_fn(features, labels, mode, params):
        """Resnet model body.

        Support single host, one or more GPU training. Parameter distribution can
        be either one of the following scheme.
        1. CPU is the parameter server and manages gradient updates.
        2. Parameters are distributed evenly across all GPUs, and the first GPU
        manages gradient updates.

        Args:
        features: a list of tensors, one for each tower
        labels: a list of tensors, one for each tower
        mode: ModeKeys.TRAIN or EVAL
        params: Hyperparameters suitable for tuning
        Returns:
        A EstimatorSpec object.
        """
        is_training = mode == tf.estimator.ModeKeys.TRAIN
        weight_decay = params.weight_decay
        momentum = params.momentum

        tower_features = features
        tower_labels = labels
        tower_losses = []
        tower_gradvars = []
        tower_preds = []

        # channels first (NCHW) is normally optimal on GPU and channels last (NHWC)
        # on CPU. The exception is Intel MKL on CPU which is optimal with
        # channels_last.
        data_format = params.data_format
        if not data_format:
            if num_gpus == 0:
                data_format = "channels_last"
            else:
                data_format = "channels_first"

        if num_gpus == 0:
            num_devices = 1
            device_type = "cpu"
        else:
            num_devices = num_gpus
            device_type = "gpu"

        for i in range(num_devices):
            worker_device = "/{}:{}".format(device_type, i)
            if variable_strategy == "CPU":
                device_setter = cifar10_utils.local_device_setter(
                    worker_device=worker_device
                )
            elif variable_strategy == "GPU":
                device_setter = cifar10_utils.local_device_setter(
                    ps_device_type="gpu",
                    worker_device=worker_device,
                    ps_strategy=tf.contrib.training.GreedyLoadBalancingStrategy(
                        num_gpus, tf.contrib.training.byte_size_load_fn
                    ),
                )
            with tf.variable_scope("resnet", reuse=bool(i != 0)):
                with tf.name_scope("tower_%d" % i) as name_scope:
                    with tf.device(device_setter):
                        loss, gradvars, preds = _tower_fn(
                            is_training,
                            weight_decay,
                            tower_features[i],
                            tower_labels[i],
                            data_format,
                            params.num_layers,
                            params.batch_norm_decay,
                            params.batch_norm_epsilon,
                        )
                        tower_losses.append(loss)
                        tower_gradvars.append(gradvars)
                        tower_preds.append(preds)
                        if i == 0:
                            # Only trigger batch_norm moving mean and variance update from
                            # the 1st tower. Ideally, we should grab the updates from all
                            # towers but these stats accumulate extremely fast so we can
                            # ignore the other stats from the other towers without
                            # significant detriment.
                            update_ops = tf.get_collection(
                                tf.GraphKeys.UPDATE_OPS, name_scope
                            )

        # Now compute global loss and gradients.
        gradvars = []
        with tf.name_scope("gradient_averaging"):
            all_grads = {}
            for grad, var in itertools.chain(*tower_gradvars):
                if grad is not None:
                    all_grads.setdefault(var, []).append(grad)
            for var, grads in six.iteritems(all_grads):
                # Average gradients on the same device as the variables
                # to which they apply.
                with tf.device(var.device):
                    if len(grads) == 1:
                        avg_grad = grads[0]
                    else:
                        avg_grad = tf.multiply(
                            tf.add_n(grads), 1.0 / len(grads)
                        )
                gradvars.append((avg_grad, var))

        # Device that runs the ops to apply global gradient updates.
        consolidation_device = (
            "/gpu:0" if variable_strategy == "GPU" else "/cpu:0"
        )
        with tf.device(consolidation_device):
            # Suggested learning rate scheduling from
            # https://github.com/ppwwyyxx/tensorpack/blob/master/examples/ResNet/cifar10-resnet.py#L155
            num_batches_per_epoch = cifar10.Cifar10DataSet.num_examples_per_epoch(
                "train"
            ) // (
                params.train_batch_size * num_workers
            )
            boundaries = [
                num_batches_per_epoch * x
                for x in np.array([80, 120, 160], dtype=np.int64)
            ]
            staged_lr = [
                params.learning_rate * x for x in [1, 0.1, 0.01, 0.001]
            ]

            learning_rate = tf.train.piecewise_constant(
                tf.train.get_global_step(), boundaries, staged_lr
            )

            loss = tf.reduce_mean(tower_losses, name="loss")

            # examples_sec_hook = cifar10_utils.ExamplesPerSecondHook(
            #     params.train_batch_size, every_n_steps=10
            # )

            # tensors_to_log = {"learning_rate": learning_rate, "loss": loss}

            # logging_hook = tf.train.LoggingTensorHook(
            #     tensors=tensors_to_log, every_n_iter=100
            # )

            # train_hooks = [logging_hook, examples_sec_hook]
            train_hooks = []

            # Hyper-parameter "momentum" is only used for the Momentum Optimizer
            # Other optimizers use their default parameters.
            if params.optimizer == "momentum":
                optimizer = tf.train.MomentumOptimizer(
                    learning_rate=learning_rate, momentum=momentum
                )
            elif params.optimizer == "adam":
                optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
            elif params.optimizer == "adagrad":
                optimizer = tf.train.AdagradOptimizer(
                    learning_rate=learning_rate
                )
            elif params.optimizer == "adadelta":
                optimizer = tf.train.AdadeltaOptimizer(
                    learning_rate=learning_rate
                )
            elif params.optimizer == "sgd":
                optimizer = tf.train.GradientDescentOptimizer(
                    learning_rate=learning_rate
                )
            elif params.optimizer == "rmsprop":
                optimizer = tf.train.RMSPropOptimizer(
                    learning_rate=learning_rate
                )
            else:
                raise ValueError("unrecognized optimizer name")
            # TODO: RAdam is implemented in tensorflow-addons v0.6, which requires tf 2.0
            #       Upgrade code by removing tf.contrib modules.
            # optimizer = tfa.optimizers.RectifiedAdam(lr=learning_rate)

            if params.sync:
                optimizer = tf.train.SyncReplicasOptimizer(
                    optimizer, replicas_to_aggregate=num_workers
                )
                sync_replicas_hook = optimizer.make_session_run_hook(
                    params.is_chief
                )
                train_hooks.append(sync_replicas_hook)

            # Create single grouped train op
            train_op = [
                optimizer.apply_gradients(
                    gradvars, global_step=tf.train.get_global_step()
                )
            ]
            train_op.extend(update_ops)
            train_op = tf.group(*train_op)

            predictions = {
                "classes": tf.concat(
                    [p["classes"] for p in tower_preds], axis=0
                ),
                "probabilities": tf.concat(
                    [p["probabilities"] for p in tower_preds], axis=0
                ),
            }
            stacked_labels = tf.concat(labels, axis=0)
            metrics = {
                "accuracy": tf.metrics.accuracy(
                    stacked_labels, predictions["classes"]
                )
            }

        return tf.estimator.EstimatorSpec(
            mode=mode,
            predictions=predictions,
            loss=loss,
            train_op=train_op,
            training_hooks=train_hooks,
            eval_metric_ops=metrics,
        )

    return _resnet_model_fn


def _tower_fn(
    is_training,
    weight_decay,
    feature,
    label,
    data_format,
    num_layers,
    batch_norm_decay,
    batch_norm_epsilon,
):
    """Build computation tower (Resnet).

    Args:
        is_training: true if is training graph.
        weight_decay: weight regularization strength, a float.
        feature: a Tensor.
        label: a Tensor.
        data_format: channels_last (NHWC) or channels_first (NCHW).
        num_layers: number of layers, an int.
        batch_norm_decay: decay for batch normalization, a float.
        batch_norm_epsilon: epsilon for batch normalization, a float.

    Returns:
        A tuple with the loss for the tower, the gradients and parameters, and
        predictions.

    """
    model = cifar10_model.ResNetCifar10(
        num_layers,
        batch_norm_decay=batch_norm_decay,
        batch_norm_epsilon=batch_norm_epsilon,
        is_training=is_training,
        data_format=data_format,
    )
    logits = model.forward_pass(feature, input_data_format="channels_last")
    tower_pred = {
        "classes": tf.argmax(input=logits, axis=1),
        "probabilities": tf.nn.softmax(logits),
    }

    tower_loss = tf.losses.sparse_softmax_cross_entropy(
        logits=logits, labels=label
    )
    tower_loss = tf.reduce_mean(tower_loss)

    model_params = tf.trainable_variables()
    tower_loss += weight_decay * tf.add_n(
        [tf.nn.l2_loss(v) for v in model_params]
    )

    tower_grad = tf.gradients(tower_loss, model_params)

    return tower_loss, zip(tower_grad, model_params), tower_pred


def input_fn(
    data_dir, subset, num_shards, batch_size, use_distortion_for_training=True
):
    """Create input graph for model.

    Args:
        data_dir: Directory where TFRecords representing the dataset are located.
        subset: one of 'train', 'validation' and 'eval'.
        num_shards: num of towers participating in data-parallel training.
        batch_size: total batch size for training to be divided by the number of
        shards.
        use_distortion_for_training: True to use distortions.
    Returns:
        two lists of tensors for features and labels, each of num_shards length.
    """
    with tf.device("/cpu:0"):
        use_distortion = subset == "train" and use_distortion_for_training
        dataset = cifar10.Cifar10DataSet(data_dir, subset, use_distortion)
        image_batch, label_batch = dataset.make_batch(batch_size)
        if num_shards <= 1:
            # No GPU available or only 1 GPU.
            return [image_batch], [label_batch]

        # Note that passing num=batch_size is safe here, even though
        # dataset.batch(batch_size) can, in some cases, return fewer than batch_size
        # examples. This is because it does so only when repeating for a limited
        # number of epochs, but our dataset repeats forever.
        image_batch = tf.unstack(image_batch, num=batch_size, axis=0)
        label_batch = tf.unstack(label_batch, num=batch_size, axis=0)
        feature_shards = [[] for i in range(num_shards)]
        label_shards = [[] for i in range(num_shards)]
        for i in xrange(batch_size):
            idx = i % num_shards
            feature_shards[idx].append(image_batch[i])
            label_shards[idx].append(label_batch[i])
        feature_shards = [tf.parallel_stack(x) for x in feature_shards]
        label_shards = [tf.parallel_stack(x) for x in label_shards]
        return feature_shards, label_shards


def build_estimator(
    data_dir,
    num_gpus,
    variable_strategy,
    run_config,
    hparams,
    use_distortion_for_training=True,
    ws=None,
):
    """Returns an Experiment function.

    Experiments perform training on several workers in parallel,
    in other words experiments know how to invoke train and eval in a sensible
    fashion for distributed training. Arguments passed directly to this
    function are not tunable, all other arguments should be passed within
    tf.HParams, passed to the enclosed function.

    Args:
        data_dir: str. Location of the data for input_fns.
        num_gpus: int. Number of GPUs on each worker.
        variable_strategy: String. CPU to use CPU as the parameter server
        and GPU to use the GPUs as the parameter server.
        use_distortion_for_training: bool. See cifar10.Cifar10DataSet.
    Returns:
        A function (tf.estimator.RunConfig, tf.contrib.training.HParams) ->
        tf.contrib.learn.Experiment.

        Suitable for use by tf.contrib.learn.learn_runner, which will run various
        methods on Experiment (train, evaluate) based on information
        about the current runner in `run_config`.
    """

    # Create estimator.
    train_input_fn = functools.partial(
        input_fn,
        data_dir,
        subset="train",
        num_shards=num_gpus,
        batch_size=hparams.train_batch_size,
        use_distortion_for_training=use_distortion_for_training,
    )

    eval_input_fn = functools.partial(
        input_fn,
        data_dir,
        subset="validation",
        batch_size=hparams.eval_batch_size,
        num_shards=num_gpus,
    )

    # validation: 5000, eval:10000
    num_eval_examples = cifar10.Cifar10DataSet.num_examples_per_epoch(
        "validation"
    )

    if num_eval_examples % hparams.eval_batch_size != 0:
        raise ValueError(
            "validation set size must be multiple of eval_batch_size"
        )

    classifier = tf.estimator.Estimator(
        model_fn=get_model_fn(
            num_gpus, variable_strategy, run_config.num_worker_replicas or 1
        ),
        config=run_config,
        params=hparams,
        warm_start_from=ws,
    )

    return train_input_fn, eval_input_fn, classifier


def get_idx(pbounds, names):
    param_names = list(pbounds.keys())
    param_names.sort()

    param_list = [0] * len(param_names)
    for i in range(len(param_names)):
        if param_names[i] in names:
            param_list[i] = 1

    return param_list


class MyTrainableEstimator(Trainable):
    def _setup(self, config):

        # The env variable is on deprecation path, default is set to off.
        os.environ["TF_SYNC_ON_FINISH"] = "0"
        os.environ["TF_ENABLE_WINOGRAD_NONFUSED"] = "1"

        # Session configuration.
        sess_config = tf.ConfigProto(
            allow_soft_placement=True,
            log_device_placement=args.log_device_placement,
            intra_op_parallelism_threads=args.num_intra_threads,
            gpu_options=tf.GPUOptions(
                force_gpu_compatible=True, allow_growth=True
            ),
        )

        # Convert to actual hyperparameter values here using the grid (discrete) input
        hparams["train_batch_size"] = 2 ** (int(config["batch_size"]) + 5)
        hparams["momentum"] = 0.4 + (
            0.55 * int(config["momentum"]) / args.precision
        )
        hparams["weight_decay"] = 1e-4 + (
            1e-4 * int(config["weight_decay"]) / args.precision
        )
        hparams["batch_norm_decay"] = 0.8 + (
            0.199 * int(config["batch_norm_decay"]) / args.precision
        )
        hparams["batch_norm_epsilon"] = 1e-5 + (
            0.00099 * int(config["batch_norm_epsilon"]) / args.precision
        )
        hparams["learning_rate"] = 0.01 + (
            0.1 * int(config["learning_rate"]) / args.precision
        )
        opt = int(config["optimizer"])
        if opt == 0:
            hparams["optimizer"] = "momentum"
        elif opt == 1:
            hparams["optimizer"] = "adam"
        elif opt == 2:
            hparams["optimizer"] = "adagrad"
        elif opt == 3:
            hparams["optimizer"] = "adadelta"
        elif opt == 4:
            hparams["optimizer"] = "sgd"
        else:
            hparams["optimizer"] = "rmsprop"

        # Calculate number of steps per one epoch
        self.train_steps = cifar10.Cifar10DataSet.num_examples_per_epoch(
            "train"
        ) // (hparams["train_batch_size"])

        # TODO: Fix checkpoint dir
        run_config = cifar10_utils.RunConfig(
            session_config=sess_config,
            model_dir=None,
            save_checkpoints_secs=None,
            save_checkpoints_steps=self.train_steps,
            keep_checkpoint_max=None,
            keep_checkpoint_every_n_hours=None,
        )
        self.run_config = run_config

        self.train_input_fn, self.eval_input_fn, self.estimator = build_estimator(
            data_dir=args.data_dir,
            num_gpus=args.num_gpus,
            variable_strategy=args.variable_strategy,
            use_distortion_for_training=args.use_distortion_for_training,
            run_config=run_config,
            hparams=tf.contrib.training.HParams(
                is_chief=run_config.is_chief, **hparams
            ),
        )

        self.logger = logging.getLogger("metrics")
        self.logger.setLevel(logging.INFO)
        file_handler = logging.FileHandler(args.log_path)
        self.logger.addHandler(file_handler)

        self.logger.info(f"[CONFIG] ID={self._experiment_id} config={hparams}")
        # self.steps = self.train_steps

    def _train(self):
        self.estimator.train(
            input_fn=self.train_input_fn, steps=self.train_steps
        )
        metrics = self.estimator.evaluate(
            input_fn=self.eval_input_fn,
            steps=args.eval_batch_size * args.num_batches_for_eval,
        )
        # self.steps = self.steps + self.train_steps
        self.logger.info(
            f"[RESULT] ID={self._experiment_id} iter={self._iteration} result={metrics}"
        )
        return metrics

    def _stop(self):
        self.estimator = None

    def _save(self, checkpoint_dir):
        lastest_checkpoint = self.estimator.latest_checkpoint()
        tf.logging.info(
            "Saving checkpoint {} for tune".format(lastest_checkpoint)
        )
        f = open(checkpoint_dir + "/path.txt", "w")
        f.write(lastest_checkpoint)
        f.flush()
        f.close()
        return checkpoint_dir + "/path.txt"

    def _restore(self, checkpoint_path):
        f = open(checkpoint_path, "r")
        path = f.readline().strip()
        tf.logging.info("Opening checkpoint {} for tune".format(path))
        f.flush()
        f.close()

        ws = tf.estimator.WarmStartSettings(ckpt_to_initialize_from=path)
        self.train_input_fn, self.eval_input_fn, self.estimator = build_estimator(
            data_dir=args.data_dir,
            num_gpus=args.num_gpus,
            variable_strategy=args.variable_strategy,
            use_distortion_for_training=args.use_distortion_for_training,
            run_config=self.run_config,
            hparams=tf.contrib.training.HParams(
                is_chief=self.run_config.is_chief, **hparams
            ),
            warm_start_from=ws,
        )


def main():
    # print(args)

    # Minor hack of generating a grid of 100 values each.
    # By setting all parameters to be discrete values over range (0,100),
    # we can map each integer value to corresponding hyperparameter value in training code.
    pbounds = {
        "batch_size": (0, 6),
        "momentum": (0, args.precision),
        "weight_decay": (0, args.precision),
        "batch_norm_decay": (0, args.precision),
        "batch_norm_epsilon": (0, args.precision),
        "learning_rate": (0, args.precision),
        "optimizer": (0, 6),
    }

    discrete = [
        "batch_size",
        "momentum",
        "weight_decay",
        "batch_norm_decay",
        "batch_norm_epsilon",
        "learning_rate",
        "optimizer",
    ]

    categorical = []

    discrete_indices = get_idx(pbounds, discrete)
    categorical_indices = get_idx(pbounds, categorical)

    train_spec = {
        "resources_per_trial": {"cpu": 12, "gpu": 1},
        "stop": {
            "accuracy": 93,
            "training_iteration": 2 if args.smoke_test else 99999,
        },
        "config": {
            "exp": "ckpt",  # the name of directory where training results are saved
            "log_level": "ERROR",
        },
        "num_samples": 100000,
        "local_dir": "/home/ddoyoon/BayesianOptimization/examples/cnn/cifar10_estimator/ckpt",
        "checkpoint_at_end": True,
    }

    algo = BayesOptSearch(
        args.strategy,
        pbounds,
        discrete=discrete_indices,
        categorical=categorical_indices,
        max_concurrent=12,
        metric="accuracy",
        mode="max",
        utility_kwargs={"kind": "ucb", "kappa": 2.5, "xi": 0.0},
    )

    # TODO: Initial values will not be discretized as of now.
    #       Manually probing with discrete values instead.
    # algo.optimizer.probe(
    #     params={
    #         "batch_size": 0,
    #         "momentum": 0,
    #         "weight_decay": 0,
    #         "batch_norm_decay": 0,
    #         "batch_norm_epsilon": 0,
    #         "learning_rate": 0,
    #     },
    #     lazy=True,
    # )

    scheduler = AsyncHyperBandScheduler(
        metric="accuracy",
        mode="max",
        max_t=200,
        grace_period=20,
        reduction_factor=2,
    )

    experiment_start = datetime.datetime.utcnow()
    logger = logging.getLogger("metrics")
    logger.setLevel(logging.INFO)
    file_handler = logging.FileHandler(args.log_path)
    logger.addHandler(file_handler)
    logger.info(f"[ TIME ] start={experiment_start}")

    run(
        MyTrainableEstimator,
        name="bo_resnet_cifar10",
        search_alg=algo,
        scheduler=scheduler,
        **train_spec,
    )

    experiment_end = datetime.datetime.utcnow()
    experiment_duration = experiment_end - experiment_start
    logger.info(f"[ TIME ] end={experiment_end}")
    logger.info(
        f"[ TIME ] end-to-end (min)={experiment_duration.total_seconds() / 60}"
    )


if __name__ == "__main__":

    if args.ray_address != "":
        ray.init(redis_address=args.ray_address, logging_level=logging.ERROR)
    else:
        ray.init()

    if args.num_gpus > 0:
        assert tf.test.is_gpu_available(), "Requested GPUs but none found."
    if args.num_gpus < 0:
        raise ValueError(
            'Invalid GPU count: "--num-gpus" must be 0 or a positive integer.'
        )
    if args.num_gpus == 0 and args.variable_strategy == "GPU":
        raise ValueError(
            "num-gpus=0, CPU must be used as parameter server. Set"
            "--variable-strategy=CPU."
        )
    if (args.num_layers - 2) % 6 != 0:
        raise ValueError("Invalid --num-layers parameter.")

    main()

