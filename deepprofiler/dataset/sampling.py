import pandas as pd
import numpy as np
import skimage.io
import threading
import tqdm
import os
from io import BytesIO

import tensorflow as tf

import deepprofiler.imaging.boxes
import deepprofiler.imaging.cropping
import deepprofiler.dataset.utils

class SingleCellSampler(deepprofiler.imaging.cropping.CropGenerator):

    def start(self, session):
        self.session = session
        # Define input data batches
        with tf.compat.v1.variable_scope("train_inputs"):
            self.config["train"]["model"]["params"]["batch_size"] = self.config["train"]["validation"]["batch_size"]
            self.build_input_graph()

    def process_batch(self, batch):
        for i in range(len(batch["keys"])):
            batch["locations"][i]["Key"] = batch["keys"][i]
            batch["locations"][i]["Target"] = batch["targets"][i][0]
            batch["locations"][i]["Class_Name"] = self.dset.targets[0].values[batch["targets"][i][0]]
        metadata = pd.concat(batch["locations"])
        cols = ["Key", "Target", "Nuclei_Location_Center_X", "Nuclei_Location_Center_Y"]
        seps = ["+", "@", "x", ".png"]
        metadata["Image_Name"] = ""
        for c in range(len(cols)):
            metadata["Image_Name"] += metadata[cols[c]].astype(str).str.replace("/", "-") + seps[c]
        
        boxes, box_ind, targets, masks = deepprofiler.imaging.boxes.prepare_boxes(batch, self.config)

        feed_dict = {
            self.input_variables["image_ph"]:batch["images"],
            self.input_variables["boxes_ph"]:boxes,
            self.input_variables["box_ind_ph"]:box_ind,
            self.input_variables["mask_ind_ph"]:masks
        }
        for i in range(len(targets)):
            tname = "target_" + str(i)
            feed_dict[self.input_variables["targets_phs"][tname]] = targets[i]

        output = self.session.run(self.input_variables["labeled_crops"], feed_dict)
        return output[0], metadata.reset_index(drop=True)


def start_session():
    configuration = tf.compat.v1.ConfigProto()
    configuration.gpu_options.allow_growth = True
    main_session = tf.compat.v1.Session(config=configuration)
    tf.compat.v1.keras.backend.set_session(main_session)
    return main_session


def is_directory_empty(outdir):
    # Verify that the output directory is empty
    os.makedirs(outdir, exist_ok=True)
    files = os.listdir(outdir)
    if len(files) > 0:
        erase = ""
        while erase != "y" and erase != "n":
            erase = input("Delete " + str(len(files)) + " existing files in " + outdir + "? (y/n) ")
            print(erase)
        if erase == "n":
            print("Terminating sampling.")
            return False
        elif erase == "y":
            print("Removing previous sampled files")
            for f in tqdm.tqdm(files):
                os.remove(os.path.join(outdir, f))
    return True


def sample_dataset(config, dset):

    def make_tfrecord(locX, locY, key, target, class_name, image_array, box_size):
        tf_record = tf.train.Example(features=tf.train.Features(feature={
            'cell/nuclei_Location_Center_X': deepprofiler.dataset.utils.int64_feature(locX),
            'cell/nuclei_Location_Center_Y': deepprofiler.dataset.utils.int64_feature(locY),
            'cell/image_array': deepprofiler.dataset.utils.bytes_feature(image_array),
            'cell/key': deepprofiler.dataset.utils.bytes_feature(u'{}'.format(key).encode('utf-8')),
            'cell/target': deepprofiler.dataset.utils.int64_feature(target),
            'cell/class_name': deepprofiler.dataset.utils.bytes_feature(u'{}'.format(class_name).encode('utf-8')),
            'cell/box': deepprofiler.dataset.utils.int64_feature(box_size)
        }))
        return tf_record

    outdir = config["paths"]["single_cell_sample"]
    if not is_directory_empty(outdir):
        return

    # Start GPU session
    session = start_session()
    dset.show_setup()
    lock = threading.Lock()
    cropper = SingleCellSampler(config, dset)
    cropper.start(session)

    # Loop through a random sample of single cells
    pointer = dset.batch_pointer
    total_single_cells = 0
    total_images = 0
    all_metadata = []
    tfrecord_filename = os.path.join(outdir, 'sampled_single_cells.tfrecord')
    writer = tf.compat.v1.python_io.TFRecordWriter(tfrecord_filename)

    while dset.batch_pointer >= pointer:
        pointer = dset.batch_pointer
        batch = dset.get_train_batch(lock)

        # Store each single cell in a separate unfolded image
        if len(batch["keys"]) > 0:
            crops, metadata = cropper.process_batch(batch)
            for j in range(crops.shape[0]):
                #image = deepprofiler.imaging.cropping.unfold_channels(crops[j,:,:,:])
                #skimage.io.imsave(os.path.join(outdir, metadata.loc[j, "Image_Name"]), image)
                np_bytes = BytesIO()
                np.save(np_bytes, crops[j, :, :, :], allow_pickle=True)
                record = make_tfrecord(locX=metadata["Nuclei_Location_Center_X"][j], locY=metadata["Nuclei_Location_Center_Y"][j],
                                           key=metadata["Key"][j], target=metadata["Target"][j], class_name=metadata["Class_Name"][j],
                                           #image_array=crops[j, :, :, :].tobytes(),
                                           image_array=np_bytes.getvalue(),
                                           box_size=config['dataset']['locations']['box_size'])
                writer.write(record.SerializeToString())

            all_metadata.append(metadata)

            total_single_cells += len(metadata)
            total_images += len(batch["keys"])
            print(total_single_cells, "single cells sampled from", total_images, "images", end="\r")
    print()
    writer.close()
    # Save metadata
    all_metadata = pd.concat(all_metadata).reset_index(drop=True)
    all_metadata.to_csv(os.path.join(outdir, "sc-metadata.csv"), index=False)

