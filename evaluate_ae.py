import os.path as osp
import os
import sys
import argparse
import numpy as np

# add paths
parent_dir = osp.dirname(osp.abspath(__file__))
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

from src.autoencoder import Configuration as Conf
from src.point_net_ae import PointNetAutoEncoder

from src.in_out import snc_category_to_synth_id, create_dir, PointCloudDataSet, \
                                        load_all_point_clouds_under_folder, \
                                        load_all_point_clouds_from_filenames

from src.tf_utils import reset_tf_graph
from src.general_utils import plot_3d_point_cloud

from src.evaluation_metrics import minimum_mathing_distance, \
                        jsd_between_point_cloud_sets, coverage



# command line arguments
# fmt: off
parser = argparse.ArgumentParser()
parser.add_argument('--class_name', type=str, default='chair', help='Single class name (for example: chair) [default: chair]')
parser.add_argument('--experiment_name', type=str, default='single_class_ae', help='Folder for saving data form the training [default: single_class_ae]')
parser.add_argument('--batch_size', type=int, default=50, help='Batch size [default: 50]')
parser.add_argument('--restore_epoch', type=int, default=500, help='Take the checkpoint from this epoch [default: 500]')
parser.add_argument('--dont_use_splits', action='store_false', help='Use pre-split data from data/data_splits folder')
flags = parser.parse_args()
# fmt: on

print(("Evaluation flags:", flags))



# ##### MANUAL FLAGS
# flags.experiment_name = "train_ae_jar"
# flags.batch_size = 10 
# flags.class_name = "jar"



## Define Basic Parameters
experiment_name = flags.experiment_name
class_name = flags.class_name
# class_name = raw_input('Give me the class name (e.g. "chair"): ').lower()


## Paths 
project_dir = osp.dirname(osp.abspath(__file__))
top_in_dir = osp.join(project_dir, "data", "shape_net_core_uniform_samples_2048")
train_dir = osp.join(project_dir, "log", experiment_name)
if not osp.exists(train_dir): os.mkdir(train_dir)
eval_dir = osp.join(train_dir, 'eval')
if not osp.exists(eval_dir): os.mkdir(eval_dir)


## Load Point-Clouds - Test
if flags.dont_use_splits: # use predefined train/val/test splits
    with open(osp.join(project_dir, "data", "data_splits", "test.txt"), "r") as f_test:
        filenames_test = f_test.read().split('\n')[:-1]
    pc_data_test = load_all_point_clouds_from_filenames(
        file_names=filenames_test, n_threads=8, file_ending=".ply", verbose=True)
else: 
    syn_id = snc_category_to_synth_id()[class_name]
    class_dir = osp.join(top_in_dir , syn_id)
    pc_data_test = load_all_point_clouds_under_folder(
        class_dir, n_threads=8, file_ending='.ply', verbose=True)


## Load/restore pretrained model
try:
    conf = Conf.load(train_dir + '/configuration')
except:
    print("Configuration cannot be loaded, check paths. Exiting...")
    exit()
reset_tf_graph()
ae = PointNetAutoEncoder(conf.experiment_name, conf)
ae.restore_model(conf.train_dir, epoch=flags.restore_epoch)


## Evaluate AE
reconstructions, losses, input_pointcloud, _, _ = ae.evaluate(pc_data_test, conf)


# Compare reconstructions with input data - evaluation metrics
mmd, matched_dists = minimum_mathing_distance(reconstructions, input_pointcloud, flags.batch_size, normalize=True, use_EMD=False)
cov, matched_ids = coverage(reconstructions, input_pointcloud, flags.batch_size, normalize=True, use_EMD=False)
jsd = jsd_between_point_cloud_sets(reconstructions, input_pointcloud, resolution=28)


## Save outputs
# Reconstructions
file_name_reconstructions = ("_".join(["reconstructions", class_name, experiment_name]) + ".npy")
file_path_reconstructions = osp.join(eval_dir, file_name_reconstructions)
np.save(file_path_reconstructions, reconstructions)

# Losses
file_name_losses = "_".join(["ae_loss", class_name, experiment_name]) + ".npy"
file_path_losses = osp.join(eval_dir, file_name_losses)
np.save(file_path_losses, losses)

# save log file
log_file_name = "_".join(["eval_stats", class_name, experiment_name]) + ".txt"
log_file = open(osp.join(eval_dir, log_file_name), "w", 1)
log_file.write("Mean ae loss: %.9f\n" % losses.mean())

log_file.write("Minimum Mathing Distance (MMD) score: %.9f\n" % mmd)
log_file.write("Coverage score: %.9f\n" % cov)
log_file.write("Jensen-Shannon Divergence (JSD) score: %.9f\n" % jsd)


log_file.close()


## Exapmle visualization
feed_pc, feed_model_names, _ = pc_data_test.next_batch(10)
reconstructions = ae.reconstruct(feed_pc)[0]
latent_codes = ae.transform(feed_pc)

test_id = 4
plot_3d_point_cloud(reconstructions[test_id][:, 0], 
                    reconstructions[test_id][:, 1], 
                    reconstructions[test_id][:, 2], in_u_sphere=True);




print("-------  THE END of EVALUATION  ----------")
