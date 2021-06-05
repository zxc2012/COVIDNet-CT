import os
import numpy as np
import tensorflow as tf
from tqdm import tqdm
import augmentations
import h5py
from math import ceil

class COVIDxCTDataset:
    """COVIDx-CT dataset class, which handles construction of train/validation datasets"""
    def __init__(self, data_dir, image_height=512, image_width=512, max_bbox_jitter=0.025, max_rotation=10,
                 max_shear=0.15, max_pixel_shift=10, max_pixel_scale_change=0.2, shuffle_buffer=1000):
        # General parameters
        self.data_dir = data_dir
        self.image_height = image_height
        self.image_width = image_width
        self.shuffle_buffer = shuffle_buffer

        # Augmentation parameters
        self.max_bbox_jitter = max_bbox_jitter
        self.max_rotation = max_rotation
        self.max_shear = max_shear
        self.max_pixel_shift = max_pixel_shift
        self.max_pixel_scale_change = max_pixel_scale_change

    def train_dataset(self, train_split_file='train.txt', batch_size=1):
        """Returns training dataset"""
        return self._make_dataset(train_split_file, batch_size, True)

    def validation_dataset(self, val_split_file='val.txt', batch_size=1):
        """Returns validation dataset (also used for testing)"""
        return self._make_dataset(val_split_file, batch_size, False)

    def _make_dataset(self, split_file, batch_size, is_training, balanced=True):
        """Creates COVIDX-CT dataset for train or val split"""
        files, classes, bboxes = self._get_files(split_file)
        count = len(files)
        # Create balanced dataset if required
        if is_training and balanced:
            files = np.asarray(files)
            classes = np.asarray(classes, dtype=np.int32)
            bboxes = np.asarray(bboxes, dtype=np.int32)
            class_nums = np.unique(classes)
            class_wise_datasets = []
            for cls in class_nums:
                indices = np.where(classes == cls)[0]
                class_wise_datasets.append(
                    tf.data.Dataset.from_tensor_slices((files[indices], classes[indices], bboxes[indices])))
            class_weights = [1.0 / len(class_nums) for _ in class_nums]
            dataset = tf.data.experimental.sample_from_datasets(
                class_wise_datasets, class_weights)
        else:
            dataset = tf.data.Dataset.from_tensor_slices((files, classes, bboxes))

        # Shuffle and repeat in training
        if is_training:
            dataset = dataset.shuffle(buffer_size=self.shuffle_buffer)
            dataset = dataset.repeat()
        dataset = dataset.apply(tf.contrib.data.map_and_batch(
            preprocess_fn, batch_size,
            num_parallel_batches=4,  # cpu cores
            drop_remainder=True if is_training else False))
        # Create and apply map function
        load_and_process = self._get_load_and_process_fn(is_training)

        dataset = dataset.map(load_and_process)
        dataset = dataset.batch(batch_size)
        num_iters = ceil(count / batch_size)
        with tf.Session() as sess:
            next=dataset.make_one_shot_iterator().get_next()
            for i in range(num_iters):
                data=sess.run(next)
                x=data['image']
                y=tf.keras.utils.to_categorical(data['label'],3)
                # yield(x,y)
                ## for colab
                print('Creating file', (i+1))
                if is_training:
                    dest_file_path =  "output/train_"+str(i + 1) + ".h5"
                else:
                    dest_file_path =  "output/val_"+str(i + 1) + ".h5"
                with h5py.File(dest_file_path, 'w') as f:
                    f.create_dataset("input_data", data=x)
                    f.create_dataset("input_labels", data=y)
                    

    def _get_load_and_process_fn(self, is_training):
        """Creates map function for TF dataset"""
        def load_and_process(path, label, bbox):
            # Load image
            image = tf.image.decode_png(tf.io.read_file(path), channels=1)

            # Apply augmentations and/or crop to bbox
            if is_training:
                image, bbox = self._augment_image_and_bbox(image, bbox)
            else:
                image = tf.image.crop_to_bounding_box(image, bbox[1], bbox[0], bbox[3] - bbox[1], bbox[2] - bbox[0])

            # Stack to 3-channel, scale to [0, 1] and resize
            image = tf.image.grayscale_to_rgb(image)
            image = tf.cast(image, tf.float32)
            image = image / 255.0
            image = tf.image.resize(image, [self.image_height, self.image_width])
            label = tf.cast(label, dtype=tf.int32)
            return {'image': image, 'label': label}

        return load_and_process

    def _augment_image_and_bbox(self, image, bbox):
        """Apply augmentations to image and bbox"""
        image_shape = tf.cast(tf.shape(image), tf.float32)
        image_height, image_width = image_shape[0], image_shape[1]
        image = augmentations.random_exterior_exclusion(image)
        bbox = augmentations.random_bbox_jitter(bbox, image_height, image_width, self.max_bbox_jitter)
        image, bbox = augmentations.random_rotation(image, self.max_rotation, bbox)
        image, bbox = augmentations.random_shear(image, self.max_shear, bbox)
        image = tf.image.crop_to_bounding_box(image, bbox[1], bbox[0], bbox[3] - bbox[1], bbox[2] - bbox[0])
        image = augmentations.random_shift_and_scale(image, self.max_pixel_shift, self.max_pixel_scale_change)
        image = tf.image.random_flip_left_right(image)
        return image, bbox

    def _get_files(self, split_file):
        """Gets image filenames and classes"""
        files, classes, bboxes = [], [], []
        with open(split_file, 'r') as f:
            for line in f.readlines():
                fname, cls, xmin, ymin, xmax, ymax = line.strip('\n').split()
                files.append(os.path.join(self.data_dir, fname))
                classes.append(int(cls))
                bboxes.append([int(xmin), int(ymin), int(xmax), int(ymax)])
        return files, classes, bboxes
