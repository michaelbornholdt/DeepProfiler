import numpy as np
import tensorflow as tf
from deepprofiler.learning.model_tfdataset import DeepProfilerModel

##################################################
# ResNet architecture as defined in "Identity Mappings
# in Deep Residual Networks" by Kaiming He,
# Xiangyu Zhang, Shaoqing Ren, Jian Sun
# https://arxiv.org/abs/1603.05027
##################################################


class ModelClass_TFRecord(DeepProfilerModel):
    def __init__(self, config, dset, is_training):
        super(ModelClass_TFRecord, self).__init__(config, dset, is_training)
        self.feature_model, self.optimizer, self.loss = self.define_model(config, dset)

    # Define supported models
    def get_supported_models(self):
        return {
            50: tf.keras.applications.resnet_v2.ResNet50V2,
            101: tf.keras.applications.resnet_v2.ResNet101V2,
            152: tf.keras.applications.resnet_v2.ResNet152V2,
        }

    ## Load a supported model
    def get_model(self, config, input_image=None, weights=None, pooling=None, include_top=False):
        supported_models = self.get_supported_models()
        SM = "ResNet supported models: " + ",".join([str(x) for x in supported_models.keys()])
        num_layers = config["train"]["model"]["params"]["conv_blocks"]
        error_msg = str(num_layers) + " conv_blocks not in " + SM
        assert num_layers in supported_models.keys(), error_msg
        if pooling is not None:
            model = supported_models[num_layers](input_tensor=input_image, pooling=pooling, include_top=include_top,
                                                 weights=weights)
        else:
            model = supported_models[num_layers](input_tensor=input_image, include_top=include_top, weights=weights)
        return model

    ## Model definition
    def define_model(self, config, dset):
        # 1. Create ResNet architecture to extract features
        loss_func = tf.keras.losses.CategoricalCrossentropy(from_logits=True)
        optimizer = tf.keras.optimizers.SGD(learning_rate=config["train"]["model"]["params"]["learning_rate"],
                                                      momentum=0.9, nesterov=True)
        if "use_pretrained_input_size" in config["profile"].keys() and self.is_training is False:
            input_tensor = tf.keras.layers.Input(
                (config["profile"]["use_pretrained_input_size"], config["profile"]["use_pretrained_input_size"], 3),
                name="input")
            model = self.get_model(
                config,
                input_image=input_tensor,
                weights='imagenet',
                pooling="avg",
                include_top=True
            )
            model.summary()

        elif self.is_training is True or "use_pretrained_input_size" not in config["profile"].keys():
            input_shape = (
                config["dataset"]["locations"]["box_size"],  # height
                config["dataset"]["locations"]["box_size"],  # width
                len(config["dataset"]["images"][
                        "channels"])  # channels
            )
            input_image = tf.compat.v1.keras.layers.Input(input_shape)
            model = self.get_model(config, input_image=input_image)
            features = tf.keras.layers.GlobalAveragePooling2D(name="pool5")(model.output)
            features = tf.keras.layers.BatchNormalization()(features)
            y = tf.keras.layers.Dense(len(set(dset.targets)))(features)

            # 4. Create and compile model
            model = tf.compat.v1.keras.models.Model(inputs=input_image, outputs=y)

            # Added weight decay following tricks reported in:
            # https://github.com/keras-team/keras/issues/2717
            regularizer = tf.compat.v1.keras.regularizers.l2(0.00001)
            for layer in model.layers:
                if hasattr(layer, "kernel_regularizer"):
                    setattr(layer, "kernel_regularizer", regularizer)

            model = tf.compat.v1.keras.models.model_from_json(
                model.to_json()
            )

        return model, optimizer, loss_func

    # Support for ImageNet initialization
    def copy_pretrained_weights(self):
        base_model = self.get_model(self.config, weights="imagenet")

        # => Transfer all weights except conv1.1
        total_layers = len(base_model.layers)
        for i in range(3, total_layers):
            if len(base_model.layers[i].weights) > 0:
                print("Setting pre-trained weights: {:.2f}%".format((i/total_layers)*100), end="\r")
                self.feature_model.layers[i].set_weights(base_model.layers[i].get_weights())
        
        # => Replicate filters of first layer as needed
        weights = base_model.layers[2].get_weights()
        available_channels = weights[0].shape[2]
        target_shape = self.feature_model.layers[2].weights[0].shape
        new_weights = np.zeros(target_shape)

        for i in range(new_weights.shape[2]):
            j = i % available_channels
            new_weights[:, :, i, :] = weights[0][:, :, j, :]

        weights_array = [new_weights]
        if len(weights) > 1: 
            weights_array += weights[1:]

        self.feature_model.layers[2].set_weights(weights_array)
        print("Network initialized with pretrained ImageNet weights")


