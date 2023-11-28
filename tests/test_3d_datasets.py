import sys
sys.path.append('.')
import time
from lib.dataset import *
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from lib.models.smpl import SMPL, SMPL_MODEL_DIR
from lib.utils.vis import batch_draw_skeleton, batch_visualize_preds

# Dataset and sequence parameters
selected_dataset = 'MPII3D'
sequence_length = 16
enable_debug = True

# Initialize the dataset
dataset_instance = eval(selected_dataset)(set='val', seqlen=sequence_length, debug=enable_debug)

# Create a DataLoader
data_loader = DataLoader(
    dataset=dataset_instance,
    batch_size=4,
    shuffle=True,
    num_workers=1,
)

# Initialize the SMPL model
smpl_model = SMPL(SMPL_MODEL_DIR)

start_time = time.time()
for iteration, target_data in enumerate(data_loader):
    # Measure data loading time
    data_loading_time = time.time() - start_time
    start_time = time.time()
    print(f'Data loading time: {data_loading_time:.4f}')

    # Print shapes of loaded data
    for key, value in target_data.items():
        print(key, value.shape)

    if enable_debug:
        input_data = target_data['video'][0]
        single_target_data = {key: value[0] for key, value in target_data.items()}

        if selected_dataset == 'MPII3D':
            # Draw skeleton for MPII3D dataset
            images = batch_draw_skeleton(input_data, single_target_data, dataset='spin', max_images=4)
            plt.imshow(images)
            plt.show()
        else:
            # For other datasets, visualize SMPL predictions
            theta = single_target_data['theta']
            pose, shape = theta[:, 3:75], theta[:, 75:]

            # Uncomment the following lines if 'smpl' function is used
            # verts, j3d, smpl_j3d = smpl_model(pose, shape)

            # Use 'smpl' function with SMPL parameters
            pred_output = smpl_model(betas=shape, body_pose=pose[:, 3:], global_orient=pose[:, :3], pose2rot=True)

            # Update target data with SMPL vertices
            single_target_data['verts'] = pred_output.vertices

            # Visualize predictions
            images = batch_visualize_preds(input_data, single_target_data, single_target_data, max_images=4, dataset='spin')
            plt.imshow(images)
            plt.show()

    if iteration == 100:
        break
