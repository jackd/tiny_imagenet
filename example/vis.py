#!/usr/bin/python
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from tiny_imagenet.manager import TinyImagenetManager, N_VAL_EXAMPLES

man = TinyImagenetManager()

print(man.extract_dir)

ids = man.load_wordnet_ids()
indices = {wnid: i for i, wnid in enumerate(ids)}

val_ids, bboxes = man.load_val_annotations()
wnids = man.load_wordnet_ids()
words = man.load_words()
class_words = [words[wnid] for wnid in wnids]
val_indices = [indices[i] for i in val_ids]

for i in range(N_VAL_EXAMPLES):
    image = Image.open(man.val_image_path(i))
    class_index = val_indices[i]
    plt.imshow(np.array(image))
    plt.title('%d: %s' % (class_index, class_words[class_index]))
    plt.show()
