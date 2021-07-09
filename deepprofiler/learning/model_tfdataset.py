import gc
import os
import random
import abc

import comet_ml
import numpy as np
import tensorflow as tf

import deepprofiler.dataset.utils
import deepprofiler.imaging.cropping
import deepprofiler.learning.validation
from deepprofiler.dataset.image_dataset import make_dataset, make_cropped_dataset

tf.config.run_functions_eagerly(True)

#

##################################################
# This class should be used as an abstract base
# class for plugin models.
##################################################


class DeepProfilerModel(abc.ABC):

    def __init__(self, config, dset, is_training):
        self.feature_model = None
        self.loss = None
        self.optimizer = None
        self.config = config
        self.dset = dset
        self.random_seed = None
        self.is_training = is_training

    def seed(self, seed):
        self.random_seed = seed
        random.seed(seed)
        np.random.seed(seed)
        tf.compat.v1.set_random_seed(seed)

    def train(self, epoch=1, metrics=["accuracy"], verbose=1):
        # Raise ValueError if feature model isn't properly defined
        check_feature_model(self)

        # Print model summary
        self.feature_model.summary()

        # Compile model
        self.feature_model.compile(self.optimizer, self.loss, metrics, run_eagerly = True)

        # Create comet ml experiment
        experiment = setup_comet_ml(self)


        path = self.config["paths"]["single_cell_sample"]
        dataset, self.steps_per_epoch, num_classes = make_dataset(path, self.config["train"]["model"]["params"]["batch_size"])

        # Get training parameters
        epochs, steps, schedule_epochs, schedule_lr, freq = setup_params(self, experiment)

        # Load weights
        self.load_weights(epoch)

        # Create callbacks
        callbacks = setup_callbacks(self, schedule_epochs, schedule_lr, self.dset, experiment)



        validation_dataset = make_cropped_dataset(self.config["train"]["model"]["params"]["batch_size"],
                                                  'val', self.dset.meta, self.config)  # batch_size, scope, meta, config

        self.feature_model.fit(
            dataset,
            epochs=epochs,
            callbacks=callbacks,
            verbose=verbose,
            initial_epoch=epoch - 1,
            #validation_data=validation_dataset,
            #validation_freq=freq,
            steps_per_epoch=steps
        )
        # Return the feature model and validation data
        return self.feature_model

    def copy_pretrained_weights(self):
        # Override this method if the model can load pretrained weights
        print("This model does not support ImageNet pretrained weights initialization")
        return

    def load_weights(self, epoch):
        output_file = self.config["paths"]["checkpoints"] + "/checkpoint_{epoch:04d}.hdf5"
        previous_model = output_file.format(epoch=epoch - 1)
        # Initialize all tf variables
        if epoch >= 1 and os.path.isfile(previous_model):
            self.feature_model.load_weights(previous_model)
            print("Weights from previous model loaded:", previous_model)
            return True
        else:
            #if self.config["train"]["model"]["initialization"] == "ImageNet":
                #self.copy_pretrained_weights()
            return False


def check_feature_model(dpmodel):
    if "feature_model" not in vars(dpmodel):  # or not isinstance(dpmodel.feature_model, keras.Model):
        raise ValueError("Feature model is not properly defined.")


def setup_comet_ml(dpmodel):
    if 'comet_ml' in dpmodel.config["train"].keys():
        experiment = comet_ml.Experiment(
            api_key=dpmodel.config["train"]["comet_ml"]["api_key"],
            project_name=dpmodel.config["train"]["comet_ml"]["project_name"]
        )
        if dpmodel.config["experiment_name"] != "results":
            experiment.set_name(dpmodel.config["experiment_name"])
        experiment.log_others(dpmodel.config)
    else:
        experiment = None
    return experiment


def setup_callbacks(dpmodel, lr_schedule_epochs, lr_schedule_lr, dset, experiment):
    # Checkpoints
    output_file = dpmodel.config["paths"]["checkpoints"] + "/checkpoint_{epoch:04d}.hdf5"
    period = 1
    save_best = False
    if "checkpoint_policy" in dpmodel.config["train"]["model"] and isinstance(
            dpmodel.config["train"]["model"]["checkpoint_policy"], int):
        period = int(dpmodel.config["train"]["model"]["checkpoint_policy"])
    elif "checkpoint_policy" in dpmodel.config["train"]["model"] and dpmodel.config["train"]["model"][
        "checkpoint_policy"] == 'best':
        save_best = True

    callback_model_checkpoint = tf.keras.callbacks.ModelCheckpoint(
        filepath=output_file,
        save_weights_only=True,
        save_best_only=save_best,
        period=period
    )

    # CSV Log
    csv_output = dpmodel.config["paths"]["logs"] + "/log.csv"
    callback_csv = tf.keras.callbacks.CSVLogger(filename=csv_output)

    # Queue stats
    qstats = tf.keras.callbacks.LambdaCallback(
        #on_train_begin=lambda logs: dset.show_setup(),
        #on_epoch_end=lambda epoch, logs: experiment.log_metrics(dset.show_stats()) if experiment  # else dset.show_stats()
    )

    # Learning rate schedule
    def lr_schedule(epoch, lr):
        if epoch in lr_schedule_epochs:
            return lr_schedule_lr[lr_schedule_epochs.index(epoch)]
        else:
            return lr

    # Collect all callbacks
    if lr_schedule_epochs:
        callback_lr_schedule = tf.keras.callbacks.LearningRateScheduler(lr_schedule, verbose=1)
        callbacks = [callback_model_checkpoint, callback_csv, callback_lr_schedule, qstats]
    else:
        callbacks = [callback_model_checkpoint, callback_csv, qstats]
    return callbacks


def setup_params(dpmodel, experiment):
    epochs = dpmodel.config["train"]["model"]["epochs"]
    steps = dpmodel.steps_per_epoch
    lr_schedule_epochs = []
    lr_schedule_lr = []
    if 'comet_ml' in dpmodel.config["train"].keys():
        params = dpmodel.config["train"]["model"]["params"]
        experiment.log_others(params)
    if "lr_schedule" in dpmodel.config["train"]["model"]:
        if dpmodel.config["train"]["model"]["lr_schedule"] == "cosine":
            lr_schedule_epochs = [x for x in range(epochs)]
            init_lr = dpmodel.config["train"]["model"]["params"]["learning_rate"]
            # Linear warm up
            lr_schedule_lr = [init_lr / (5 - t) for t in range(5)]
            # Cosine decay
            lr_schedule_lr += [0.5 * (1 + np.cos((np.pi * t) / epochs)) * init_lr for t in range(5, epochs)]
        else:
            assert len(dpmodel.config["train"]["model"]["lr_schedule"]["epoch"]) == \
                   len(dpmodel.config["train"]["model"]["lr_schedule"]["lr"]), "Make sure that the length of " \
                                                                               "lr_schedule->epoch equals the length of " \
                                                                               "lr_schedule->lr in the config file."

            lr_schedule_epochs = dpmodel.config["train"]["model"]["lr_schedule"]["epoch"]
            lr_schedule_lr = dpmodel.config["train"]["model"]["lr_schedule"]["lr"]

    # Validation frequency
    if "frequency" in dpmodel.config["train"]["validation"].keys():
        freq = dpmodel.config["train"]["validation"]["frequency"]
    else:
        freq = 1

    return epochs, steps, lr_schedule_epochs, lr_schedule_lr, freq

