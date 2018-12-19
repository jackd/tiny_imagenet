from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
DOWNLOAD_URL = 'http://cs231n.stanford.edu/tiny-imagenet-200.zip'
DOWNLOAD_FILENAME = 'tiny-imagenet-200.zip'
TRAIN_EXAMPLES_PER_CLASS = 500
N_CLASSES = 200
N_VAL_EXAMPLES = 10000
N_TEST_EXAMPLES = 10000

_n_examples = {
    'train': 100000,
    'val': 10000,
    'test': 10000,
}


def n_examples(mode):
    return _n_examples[mode.lower()]


class TinyImagenetManager(object):
    def __init__(self, data_dir=None):
        if data_dir is None:
            data_dir = os.path.join(
                os.path.realpath(os.path.dirname(__file__)), 'data')
        if not os.path.isdir(data_dir):
            os.makedirs(data_dir)
        self._data_dir = data_dir
        self._extract_dir = None

    @property
    def data_dir(self):
        return self._data_dir

    def download(self):
        data_dir = self.data_dir
        download_path = os.path.join(data_dir, DOWNLOAD_FILENAME)
        if not os.path.isfile(download_path):
            import wget
            wget.download(DOWNLOAD_URL, out=data_dir)
        assert(os.path.isfile(download_path))
        return download_path

    def extract(self):
        extract_dir = os.path.join(
            self.data_dir, os.path.splitext(DOWNLOAD_FILENAME)[0])
        if not os.path.isdir(extract_dir):
            import zipfile
            download_path = self.download()
            with zipfile.ZipFile(download_path) as zf:
                zf.extractall(self.data_dir)
        assert(os.path.isdir(extract_dir))
        return extract_dir

    @property
    def extract_dir(self):
        if self._extract_dir is None:
            self._extract_dir = self.extract()
        return self._extract_dir

    def train_image_path(self, wordnet_id, example_index):
        return os.path.join(
            self.extract_dir, 'train', wordnet_id, 'images',
            '%s_%d.JPEG' % (wordnet_id, example_index))

    def val_image_path(self, example_index):
        return os.path.join(
            self.extract_dir, 'val', 'images', 'val_%d.JPEG' % example_index)

    def test_image_path(self, example_index):
        return os.path.join(
            self.extract_dir, 'test', 'images', 'test_%d.JPEG' % example_index)

    def load_val_annotations(self):
        path = os.path.join(self.extract_dir, 'val', 'val_annotations.txt')
        wordnet_ids = []
        bboxes = []
        with open(path, 'r') as fp:
            for line in fp.readlines():
                line = line.rstrip()
                if line != '':
                    line = line.split('\t')
                    wordnet_id = line[1]
                    bbox = tuple(int(b) for b in line[2:])
                    wordnet_ids.append(wordnet_id)
                    bboxes.append(bbox)
        return wordnet_ids, bboxes

    def load_wordnet_ids(self):
        path = os.path.join(self.extract_dir, 'wnids.txt')
        with open(path, 'r') as fp:
            lines = [line.rstrip() for line in fp.readlines()]
            if lines[-1] == '':
                lines = lines[:-1]
        return lines

    def load_words(self):
        path = os.path.join(self.extract_dir, 'words.txt')
        out = {}
        with open(path, 'r') as fp:
            for line in fp.readlines():
                line = line.rstrip()
                if line != '':
                    wordnet_id, desc = line.split('\t')
                    out[wordnet_id] = desc
        return out
