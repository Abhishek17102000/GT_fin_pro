import sys
sys.path.append('.')

import torch
import numpy as np
import skimage.io as io
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

from lib.dataset import *
from lib.utils.vis import batch_draw_skeleton, batch_visualize_preds


def visualize_2d_data(dataset_name, debug_mode=True):
    is_train = True
    sequence_length = 32
    batch_size = 1
    data_loader = DataLoader(
        dataset=eval(dataset_name)(seqlen=sequence_length, debug=debug_mode),
        batch_size=batch_size,
        shuffle=True,
        num_workers=1,
    )

    for i, target in enumerate(data_loader):
        for key, value in target.items():
            print(key, value.shape)

        if debug_mode:
            if dataset_name == 'Insta':
                input_data = torch.ones(batch_size, sequence_length, 3, 224, 224)[0]
            else:
                input_data = target['video'][0]
            single_target = {key: value[0] for key, value in target.items()}

            dataset_name = 'spin'
            plt.figure(figsize=(19.2, 10.8))
            images = batch_draw_skeleton(input_data, single_target, dataset=dataset_name, max_images=4)
            plt.imshow(images)
            plt.show()

        if i == 20:
            break


if __name__ == '__main__':
    visualize_2d_data('Insta', debug_mode=True)
