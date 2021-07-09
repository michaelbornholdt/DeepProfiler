import os
import numpy as np
import pandas as pd
import tensorflow as tf
tf.config.run_functions_eagerly(True)
from glob import glob
import random

import deepprofiler.dataset.pixels
import deepprofiler.dataset.utils
import deepprofiler.dataset.metadata
import deepprofiler.dataset.target
import deepprofiler.imaging.boxes


class ImageLocations(object):

    def __init__(self, metadata_training, getImagePaths, targets):
        self.keys = []
        self.images = []
        self.targets = []
        self.outlines = []
        for i, r in metadata_training.iterrows():
            key, image, outl = getImagePaths(r)
            self.keys.append(key)
            self.images.append(image)
            self.targets.append([t.get_values(r) for t in targets])
            self.outlines.append(outl)
        print("Reading single-cell locations")


    def load_loc(self, params):
        # Load cell locations for one image
        i, config = params
        loc = deepprofiler.imaging.boxes.get_locations(self.keys[i], config)
        loc["ID"] = loc.index
        loc["ImageKey"] = self.keys[i]
        loc["ImagePaths"] = "#".join(self.images[i])
        loc["Target"] = self.targets[i][0]
        loc["Outlines"] = self.outlines[i]
        print("Image", i, ":", len(loc), "cells", end="\r")
        return loc


    def load_locations(self, config):
        # Use parallel tools to read all cells as quickly as possible
        process = deepprofiler.dataset.utils.Parallel(config, numProcs=config["train"]["sampling"]["workers"])
        data = process.compute(self.load_loc, [x for x in range(len(self.keys))])
        process.close()
        return data


class ImageDataset():

    def __init__(self, metadata, sampling_field, channels, dataRoot, keyGen, config):
        self.meta = metadata      # Metadata object with a valid dataframe
        self.channels = channels  # List of column names corresponding to each channel file
        self.root = dataRoot      # Path to the directory of images
        self.keyGen = keyGen      # Function that returns the image key given its record in the metadata
        self.sampling_field = sampling_field  # Field in the metadata used to sample images evenly
        self.sampling_values = metadata.data[sampling_field].unique()
        self.targets = []         # Array of tasks in a multi-task setting (only one task supported)
        self.outlines = None      # Use of outlines if available
        self.config = config      # The configuration file


    def get_image_paths(self, r):
        key = self.keyGen(r)
        image = [self.root + "/" + r[ch] for ch in self.channels]
        outlines = self.outlines
        if outlines is not None:
            outlines = self.outlines + r["Outlines"]
        return (key, image, outlines)

    def prepare_training_locations(self):
        # Load single cell locations in one data frame
        image_loc = ImageLocations(self.meta.train, self.get_image_paths, self.targets)
        locations = image_loc.load_locations(self.config)
        locations = pd.concat(locations)

        # Group by image and count the number of single cells per image in the column ID
        self.training_images = locations.groupby(["ImageKey", "Target"])["ID"].count().reset_index()

        workers = self.config["train"]["sampling"]["workers"]
        batch_size = self.config["train"]["model"]["params"]["batch_size"]
        cache_size = self.config["train"]["sampling"]["cache_size"]
        self.sampling_factor = self.config["train"]["sampling"]["factor"]

        # Count the total number of single cells
        self.total_single_cells = len(locations)
        # Median number of images per class
        self.sample_images = int(np.median(self.training_images.groupby("Target").count()["ID"]))
        # Number of classes
        targets = len(self.training_images["Target"].unique())
        # Median number of single cells per image (column ID has counts as a result of groupby above)
        self.sample_locations = int(np.median(self.training_images["ID"]))
        # Set the target of single cells per epoch asuming a balanced set
        self.cells_per_epoch = int(targets * self.sample_images * self.sample_locations * self.sampling_factor)
        # Number of images that each worker should load at a time
        self.images_per_worker = int(batch_size / workers)
        # Percent of all cells that will be loaded in memory at a given moment in the queue
        self.cache_coverage = 100*(cache_size / self.cells_per_epoch)
        # Number of gradient updates required to approximately use all cells in an epoch
        self.steps_per_epoch = int(self.cells_per_epoch / batch_size)

        self.data_rotation = 0
        self.cache_records = 0
        self.shuffle_training_images()


    def show_setup(self):
        print(" || => Total single cells:", self.total_single_cells)
        print(" || => Median # of images per class:", self.sample_images)
        print(" || => Number of classes:", len(self.training_images["Target"].unique()))
        print(" || => Median # of cells per image:", self.sample_locations)
        print(" || => Approx. cells per epoch (with balanced sampling):", self.cells_per_epoch)
        print(" || => Images sampled per worker:", self.images_per_worker)
        print(" || => Cache data coverage: {}%".format(int(self.cache_coverage)))
        print(" || => Steps per epoch:", self.steps_per_epoch)
 

    def show_stats(self): ## Deprecated?
        # Proportion of images loaded by workers from all images that they should load in one epoch (recall)
        worker_efficiency = int(100 * (self.data_rotation / self.training_sample.shape[0]))
        # Proportion of single cells placed in the cache from all those that should be used in one epoch
        cache_usage = int(100 * self.cache_records / self.cells_per_epoch)
        self.data_rotation = 0
        self.cache_records = 0
        return {'worker_efficiency': worker_efficiency, 'cache_usage': cache_usage}

    def shuffle_training_images(self):                                                                                                                                                                                                                                  
        # Images in the original metadata file are resampled at each epoch
        sample = []                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                         
        for c in self.meta.train[self.sampling_field].unique():                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                   
            # Sample the same number of images per class. Oversample if the class has less images than needed
            mask = self.meta.train[self.sampling_field] == c
            available = self.meta.train[mask].shape[0]                                                                                                                                                                                                                                                                                                                      
            rec = self.meta.train[mask].sample(n=self.sample_images, replace=available < self.sample_images)                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                        
            sample.append(rec)

        # Shuffle and restart pointers. Note that training sample has images instead of single cells.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                               
        self.training_sample = pd.concat(sample)
        self.training_sample = self.training_sample.sample(frac=1.0).reset_index(drop=True)
        self.batch_pointer = 0

    def get_train_batch(self, lock):
        # Select the next group of available images for cropping
        lock.acquire()
        df = self.training_sample[self.batch_pointer:self.batch_pointer + self.images_per_worker].copy()
        self.batch_pointer += self.images_per_worker
        self.data_rotation += self.images_per_worker
        if self.batch_pointer > self.training_sample.shape[0]:
            self.shuffle_training_images()
        lock.release()

        # Prepare the batch and cropping information for these images
        batch = {"keys": [], "images": [], "targets": [], "locations": []}
        sample = max(1, int(self.sample_locations * self.sampling_factor))
        for k, r in df.iterrows():
            key, image, outl = self.get_image_paths(r)
            batch["keys"].append(key)
            batch["targets"].append([t.get_values(r) for t in self.targets])
            batch["images"].append(deepprofiler.dataset.pixels.openImage(image, outl))
            batch["locations"].append(deepprofiler.imaging.boxes.get_locations(key, self.config, random_sample=sample))

        return batch

    def scan(self, f, frame="train", check=lambda k: True):
        if frame == "all":
            frame = self.meta.data.iterrows()
        elif frame == "val":
            frame = self.meta.val.iterrows()
        else:
            frame = self.meta.train.iterrows()

        images = [(i, self.get_image_paths(r), r) for i, r in frame]
        for img in images:
            # img => [0] index key, [1] => [0:key, 1:paths, 2:outlines], [2] => metadata
            index = img[0]
            meta = img[2]
            if check(meta):
                #image = deepprofiler.dataset.pixels.openImage(img[1][1], img[1][2])
                #f(index, image, meta)
                f(index, self.config)
        return

    def number_of_records(self, dataset):
        if dataset == "all":
            return len(self.meta.data)
        elif dataset == "val":
            return len(self.meta.val)
        elif dataset == "train":
            return len(self.meta.train)
        else:
            return 0

    def add_target(self, new_target):
        self.targets.append(new_target)

def read_dataset(config, mode = 'train'):
    # Read metadata and split dataset in training and validation
    metadata = deepprofiler.dataset.metadata.Metadata(config["paths"]["index"], dtype=None)
    if config["prepare"]["compression"]["implement"]:
        metadata.data.replace({'.tiff': '.png', '.tif': '.png'}, inplace=True, regex=True)

    # Add outlines if specified
    outlines = None
    if "outlines" in config["prepare"].keys() and config["prepare"]["outlines"] != "":
        df = pd.read_csv(config["paths"]["metadata"] + "/outlines.csv")
        metadata.mergeOutlines(df)
        outlines = config["paths"]["root"] + "inputs/outlines/"

    print(metadata.data.info())

    # Split training data
    if mode == 'train':
        split_field = config["train"]["partition"]["split_field"]
        trainingFilter = lambda df: df[split_field].isin(config["train"]["partition"]["training_values"])
        validationFilter = lambda df: df[split_field].isin(config["train"]["partition"]["validation_values"])
        metadata.splitMetadata(trainingFilter, validationFilter)


    # Create a dataset
    keyGen = lambda r: "{}/{}-{}".format(r["Metadata_Plate"], r["Metadata_Well"], r["Metadata_Site"])
    dset = ImageDataset(
        metadata,
        config["dataset"]["metadata"]["label_field"],
        config["dataset"]["images"]["channels"],
        config["paths"]["images"],
        keyGen,
        config
    )

    # Add training targets
    if mode == 'train':
        for t in config["train"]["partition"]["targets"]:
            new_target = deepprofiler.dataset.target.MetadataColumnTarget(t, metadata.data[t].unique())
            dset.add_target(new_target)

    # Activate outlines for masking if needed
    if config["dataset"]["locations"]["mask_objects"]:
        dset.outlines = outlines

    #dset.prepare_training_locations()

    return dset

# This function is supposed to load a full image, get crops, then make a tf dataset of crops and labels
def make_cropped_dataset(batch_size, scope, meta, config):
    # We probably want the code from pixels.py as we discussed on Friday
    @tf.function
    def fold_channels(image):
        assert tf.executing_eagerly()
        # Expected input image shape: (h, w * c), with h = w
        # Output image shape: (h, w, c), with h = w

        image = image.numpy()
        print(image.shape)
        output = np.reshape(image, (image.shape[0], image.shape[1], -1), order="F").astype(np.float)
        for i in range(output.shape[-1]):
            mean = np.mean(output[:, :, i])
            std = np.std(output[:, :, i])
            output[:, :, i] = (output[:, :, i] - mean) / std
        return tf.convert_to_tensor(output, dtype=tf.float32)

    # Almost the same as get_single_cell_locations in the boxes.py
    @tf.function
    def get_boxes(metadata_row, locations_path):
        metadata_row = metadata_row.numpy()
        locations_path = locations_path.numpy()
        print(metadata_row)
        # indicies for plate\map\site are hardcoded to fit example data
        locations_path = os.path.join(str(locations_path), '{}/{}-{}'.format(metadata_row[3],
                                                        metadata_row[4], metadata_row[5]))
        if os.path.exists(locations_path):
            locations = pd.read_csv(locations_path)
            locations.values.astype(float).tolist()
            # sizes are hardcoded for example data
            for i in range(len(locations)):
                locations[i] = [locations[i][0] - 48, locations[i][1] - 48, locations[i][0] + 48, locations[i][1] + 48]

            box_indicies = [i for i in range(len(locations))]
            return tf.convert_to_tensor(locations, dtype=tf.float32), tf.convert_to_tensor(box_indicies, dtype=tf.int32)
        else:
            return tf.convert_to_tensor(np.array([]), dtype=tf.float32), tf.convert_to_tensor(np.array([]), dtype=tf.int32)

    # Read the image (tf), do crops
    def parse_image(dataframe_row):
        locations, box_indices = tf.py_function(func=get_boxes,
                                   inp=[dataframe_row, config["paths"]["locations"]],
                                   Tout=[tf.float32, tf.int32])

        def read_image(oi):
            image = tf.io.read_file(config["paths"]["images"] + dataframe_row[oi])
            image = tf.image.decode_png(image, channels=1)
            return image

        # the indices are cells with image paths in the row
        for oi in range(6, 9):
            if oi == 6:
                image = read_image(oi)
                outputs = image
            else:
                outputs = tf.concat([outputs, read_image(oi)], axis=1)

        image = tf.py_function(func=fold_channels, inp=[outputs], Tout=tf.float32)

        print('shape locations', tf.shape(locations))
        crops = tf.image.crop_and_resize(tf.expand_dims(image, 0), locations, box_indices=box_indices, crop_size=[96, 96])
        return crops

    def configure_for_performance(ds):
        ds = ds.shuffle(buffer_size=1000)
        ds = ds.batch(batch_size)
        #ds = ds.repeat()
        ds = ds.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
        return ds

    def scan(meta, scope="train"):
        if scope == "all":
            frame = meta.data
        elif scope == "val":
            frame = meta.val
        else:
            frame = meta.train
        return frame

    # Load metadata scope, merge with locations, make it tf data, img load-crop func, map images and labels

    df = scan(meta, scope)
    df.to_csv('scan.csv')

    ds_meta = []
    for index, row in df.iterrows():
        ds_meta.append(row.values.astype(str).tolist())

    # TODO targets are probably messed
    df['Compound_Concentration'] = pd.Categorical(df['Compound_Concentration'])
    labels = df['Compound_Concentration'].cat.codes.astype(int)
    print(labels)
    #df = df.drop(columns=['Compound_Concentration'])

    #df_ts = tf.data.Dataset.from_tensor_slices(df.values.astype(str))
    df_ts = tf.data.Dataset.from_tensor_slices(ds_meta)

    images_ds = df_ts.map(parse_image, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    labels_ds = tf.data.Dataset.from_tensor_slices(labels)

    num_classes = len(set(labels))

    ds = tf.data.Dataset.zip((images_ds, labels_ds))
    ds = configure_for_performance(ds)
    if scope == 'val':
        ds.take(len(ds_meta)).cache()

    print("classes", num_classes)
    return ds

# Dataset for the cropped single cells (after sample-sc)
def make_dataset(path, batch_size):

    @tf.function
    def fold_channels(crop):
        assert tf.executing_eagerly()
        # Expected input image shape: (h, w * c), with h = w
        # Output image shape: (h, w, c), with h = w

        crop = crop.numpy()
        output = np.reshape(crop, (crop.shape[0], crop.shape[0], -1), order="F").astype(np.float)
        for i in range(output.shape[-1]):
            mean = np.mean(output[:, :, i])
            std = np.std(output[:, :, i])
            output[:, :, i] = (output[:, :, i] - mean) / std
        return tf.convert_to_tensor(output, dtype=tf.float32)

        # TODO: move from numpy to tensors, example below, though tensors are immutable so the code below does not work
        #h = tf.shape(crop)[0]
        #output = tf.transpose(tf.reshape(crop, shape=(96, 96, -1)))
        #output = tf.cast(output, dtype=tf.float32)
        #for i in tf.range(tf.shape(output)[-1]):
        #    mean = tf.math.reduce_mean(output[:, :, i])
        #    std = tf.math.reduce_std(output[:, :, i])
        #    output[:, :, i] = (output[:, :, i] - mean) / std
        #return tf.convert_to_tensor(output)

    def parse_image(filename):
        image = tf.io.read_file(filename)
        image = tf.image.decode_png(image, channels=1)
        tf.print('image shape after loading', tf.shape(image))
        image = tf.py_function(func=fold_channels, inp=[image], Tout=tf.float32)
        tf.print('image shape after folding', tf.shape(image))
        return image

    def configure_for_performance(ds):
        ds = ds.shuffle(buffer_size=1000)
        ds = ds.batch(batch_size)
        ds = ds.repeat()
        ds = ds.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
        return ds

    filenames = [i for i in glob(path + '/*') if '.csv' not in i]
    steps = np.math.ceil(len(filenames) / batch_size)
    random.shuffle(filenames)
    labels = [int(name.split('@')[0].split('+')[1]) for name in filenames]
    num_classes = len(set(labels))
    print(labels)
    filenames_ds = tf.data.Dataset.from_tensor_slices(filenames)
    images_ds = filenames_ds.map(parse_image, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    labels_ds = tf.data.Dataset.from_tensor_slices(labels)
    ds = tf.data.Dataset.zip((images_ds, labels_ds))
    ds = configure_for_performance(ds)
    print("classes", num_classes)
    return ds, steps, num_classes

