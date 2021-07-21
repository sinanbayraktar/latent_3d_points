import os.path as osp
import os
import sys
import argparse

# add paths
parent_dir = osp.dirname(osp.abspath(__file__))
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

from src.ae_templates import mlp_architecture_ala_iclr_18, default_train_params
from src.autoencoder import Configuration as Conf
from src.point_net_ae import PointNetAutoEncoder

from src.in_out import snc_category_to_synth_id, create_dir, PointCloudDataSet, \
                                        load_all_point_clouds_under_folder, \
                                        load_all_point_clouds_from_filenames

from src.tf_utils import reset_tf_graph



# command line arguments
# fmt: off
parser = argparse.ArgumentParser()
parser.add_argument('--class_name', type=str, default='chair', help='Single class name (for example: chair) [default: chair]')
parser.add_argument('--experiment_name', type=str, default='single_class_ae', help='Folder for saving data form the training [default: single_class_ae]')
parser.add_argument('--training_epochs', type=int, default=500, help='Number of training epochs [default: 500]')
parser.add_argument('--batch_size', type=int, default=50, help='Batch size [default: 50]')
parser.add_argument('--restore_epoch', type=int, default=0, help='Continue training from this epoch [default: 0]')
parser.add_argument('--dont_use_splits', action='store_false', help='Use pre-split data from data/data_splits folder')
flags = parser.parse_args()
# fmt: on

print(("Train flags:", flags))



## Define Basic Parameters
experiment_name = flags.experiment_name
n_pc_points = 2048                # Number of points per model.
bneck_size = 128                  # Bottleneck-AE size
ae_loss = 'chamfer'                   # Loss to optimize: 'emd' or 'chamfer'
# class_name = raw_input('Give me the class name (e.g. "chair"): ').lower()
class_name = flags.class_name


## Paths 
project_dir = osp.dirname(osp.abspath(__file__))
top_in_dir = osp.join(project_dir, "data", "shape_net_core_uniform_samples_2048")
train_dir = osp.join(project_dir, "log", experiment_name)
if not osp.exists(train_dir):
    os.mkdir(train_dir)


## Load Point-Clouds 
if flags.dont_use_splits: # use predefined train/val/test splits
    with open(osp.join(project_dir, "data", "data_splits", "train.txt"), "r") as f_train:
        filenames_train = f_train.read().split('\n')[:-1]
    with open(osp.join(project_dir, "data", "data_splits", "val.txt"), "r") as f_val:
        filenames_val = f_val.read().split('\n')[:-1]
    pc_data_train = load_all_point_clouds_from_filenames(
        file_names=filenames_train, n_threads=8, file_ending=".ply", verbose=True)
    pc_data_val = load_all_point_clouds_from_filenames(
        file_names=filenames_val, n_threads=8, file_ending=".ply", verbose=True)
else: 
    syn_id = snc_category_to_synth_id()[class_name]
    class_dir = osp.join(top_in_dir , syn_id)
    pc_data_train = load_all_point_clouds_under_folder(
        class_dir, n_threads=8, file_ending='.ply', verbose=True)
    pc_data_val = None


## Load/restore pretrained model
if flags.restore_epoch != 0:
    conf = Conf.load(train_dir + '/configuration')
    reset_tf_graph()
    ae = PointNetAutoEncoder(conf.experiment_name, conf)
    ae.restore_model(conf.train_dir, epoch=flags.restore_epoch)


## Create a new model
else: 
    ## Training parameters
    train_params = default_train_params()

    ## AutoEncoder templates 
    encoder, decoder, enc_args, dec_args = mlp_architecture_ala_iclr_18(n_pc_points, bneck_size)

    ## Configuration parameters 
    conf = Conf(n_input = [n_pc_points, 3],
                loss = ae_loss,
                training_epochs = flags.training_epochs,
                batch_size = flags.batch_size,
                denoising = train_params['denoising'],
                learning_rate = train_params['learning_rate'],
                train_dir = train_dir,
                loss_display_step = train_params['loss_display_step'],
                saver_step = train_params['saver_step'],
                z_rotate = train_params['z_rotate'],
                encoder = encoder,
                decoder = decoder,
                encoder_args = enc_args,
                decoder_args = dec_args
            )
    conf.experiment_name = experiment_name
    conf.held_out_step = 5   # How often to evaluate/print out loss on 
                            # held_out data (if they are provided in ae.train() ).
    conf.save(osp.join(train_dir, 'configuration'))

    ## Build AE Model 
    reset_tf_graph()
    ae = PointNetAutoEncoder(conf.experiment_name, conf)


## Train the AE (save output to train_stats.txt)  
buf_size = 1 # Make 'training_stats' file to flush each output line regarding training.
fout = open(osp.join(conf.train_dir, 'train_stats.txt'), 'a', buf_size)
train_stats = ae.train(pc_data_train, conf, log_file=fout, held_out_data=pc_data_val)
fout.close()


print("-------  THE END of TRAINING  ----------")
