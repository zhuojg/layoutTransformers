import os
from PIL import Image
from scipy import ndimage
import numpy as np
import tensorflow as tf
import json
import random
import math


class MagazineDataloader:
    def __init__(self, json_files_path='./data/magazine'):
        dataset = tf.data.Dataset.list_files(os.path.join(json_files_path, '*.json'))
