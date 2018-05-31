import zipfile
import numpy as np
from tensorflow.contrib.learn.python.learn.datasets import base
import cv2
from six.moves import xrange

IMAGE_HEIGHT = 224
IMAGE_WIDTH = 224
IMAGE_DEPTH = 3

class DataSet(object):

  def __init__(self,
               images,
               labels,
               # fake_data=False,
               one_hot=False,
               # dtype=dtypes.float32,
               reshape=True):
    """Construct a DataSet.
    one_hot arg is used only if fake_data is true.  `dtype` can be either
    `uint8` to leave the input as `[0, 255]`, or `float32` to rescale into
    `[0, 1]`.
    """
    self._images = images
    self._labels = labels
    self._epochs_completed = 0
    self._index_in_epoch = 0
    self._init_feed = 0
    self._num_examples = len(images)

  @property
  def images(self):
    return self._images

  @property
  def labels(self):
    return self._labels

  @property
  def num_examples(self):
    return self._num_examples

  @property
  def epochs_completed(self):
    return self._epochs_completed

  def next_batch(self, batch_size, fake_data=False):
    """Return the next `batch_size` examples from this data set."""
    if fake_data:
      fake_image = [1] * 784
      if self.one_hot:
        fake_label = [1] + [0] * 9
      else:
        fake_label = 0
      return [fake_image for _ in xrange(batch_size)], [
          fake_label for _ in xrange(batch_size)
      ]
    start = self._index_in_epoch
    self._index_in_epoch += batch_size

    if self._index_in_epoch > self._num_examples or self._init_feed == 0:
      if self._init_feed == 0:
        self._init_feed = 1
      # Finished epoch
      if self._index_in_epoch > self._num_examples:
        self._epochs_completed += 1
      # Shuffle the data
      perm = np.arange(self._num_examples)
      np.random.shuffle(perm)
      self._images = self._images[perm]
      self._labels = self._labels[perm]
      # Start next epoch
      start = 0
      self._index_in_epoch = batch_size
      assert batch_size <= self._num_examples
    end = self._index_in_epoch
    return self._images[start:end], self._labels[start:end]

def read_data_sets(train_zip,
                   # fake_data=False,
                   one_hot=False,
                   # dtype=dtypes.float32,
                   reshape=True,
                   validation_size=5000):
    zf = zipfile.ZipFile(train_zip, mode="r")
    file_list = zf.filelist[1:]

    # for debug
    s = [x for x in range(1, 251)] + [x for x in range(12501, 12751)]
    _list = [file_list[f] for f in s]
    file_list = _list

    num_files = len(file_list)
    images = np.zeros((num_files, 224, 224, 3), dtype=np.float32)
    labels = np.zeros((num_files, ), dtype=np.int32)

    for i, _zf in enumerate(file_list):
        name = _zf.filename
        raw = zf.read(name)
        im = cv2.imdecode(np.frombuffer(raw, np.uint8), 1)
        im = cv2.resize(im, (IMAGE_WIDTH, IMAGE_HEIGHT))
        im = im.astype(np.float32)
        im = im - np.mean(im, axis=(0, 1), keepdims=True)
        images[i] = im
        labels[i] = 0 if "cat" in name else 1
        if i % 1000 == 0:
            print('Load the %d image of 25000' % i)

    if one_hot:
        # convert to one_hot format
        one_hot_labels = np.zeros((num_files, 2))
        for i in range(num_files):
            one_hot_labels[i, labels[i]] = 1

    # split train & validation
    c0_idx = np.where(labels==0)[0]
    c1_idx = np.where(labels==1)[0]
    val_sz = int(validation_size / 2)
    c0_idx = list(c0_idx)
    c1_idx = list(c1_idx)

    train_index = c0_idx[:-val_sz] + c1_idx[:-val_sz]
    train_images = images[train_index]
    train_labels = labels[train_index]

    validation_index = c0_idx[-val_sz:] + c1_idx[-val_sz:]
    validation_images = images[validation_index]
    validation_labels = labels[validation_index]

    train = DataSet(images=train_images, labels=train_labels)
    validation = DataSet(images=validation_images, labels=validation_labels)
    test = DataSet(images=validation_images, labels=validation_labels)

    return base.Datasets(train=train, validation=validation, test=test)
