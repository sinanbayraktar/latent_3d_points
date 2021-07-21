# Data splits 
# sinanbayraktar 
# 19.07.2021 

from __future__ import print_function

# import system modules
from builtins import str
from builtins import range
import os.path as osp
import os 
import sys
import argparse
import numpy as np

# add paths
parent_dir = osp.dirname(osp.dirname(osp.dirname(osp.abspath(__file__))))
if parent_dir not in sys.path:
    sys.path.append(parent_dir)
parent_dir2 = osp.dirname(osp.dirname(osp.abspath(__file__)))
if parent_dir2 not in sys.path:
    sys.path.append(parent_dir2)

from reconstruction.src.in_out import *


# command line arguments
# fmt: off
parser = argparse.ArgumentParser()
parser.add_argument('--object_classes', type=str, default='multi', help='Single class name (for example: chair) or multi [default: multi]')
parser.add_argument('--split_folder', type=str, default='data/data_splits', help='Folder for reading the train/val/test splits [default: data/data_splits]')
flags = parser.parse_args()
# fmt: on


def arrange_split(labels, top_in_dir):
    path_list = []

    # Append parent paths 
    for i in range(labels.shape[0]):
        class_name, model_name = labels[i].split("_")
        curr_path = osp.join(top_in_dir, class_name, model_name) + ".ply"
        path_list.append(curr_path)

    path_array = np.array(path_list, dtype=object)
    return path_array


def create_splits():
    # define basic parameters
    project_dir = osp.dirname(osp.abspath(__file__))
    top_in_dir = osp.join(project_dir, "data", "shape_net_core_uniform_samples_2048")  
    data_splits_dir = osp.join(project_dir, str(flags.split_folder))
    if not os.path.exists(data_splits_dir):
        os.mkdir(data_splits_dir)

    # Create txt file paths 
    train_txt_path = osp.join(data_splits_dir, "train.txt")
    val_txt_path = osp.join(data_splits_dir, "val.txt")
    test_txt_path = osp.join(data_splits_dir, "test.txt")
    # f_train = open(train_txt_path, "w")
    # f_val = open(val_txt_path, "w")
    # f_test = open(test_txt_path, "w")

    # Check if the files are there
    assert not os.path.exists(train_txt_path), "train.txt exists, please remove it and rerun! Exiting..."
    assert not os.path.exists(val_txt_path), "val.txt exists, please remove it and rerun! Exiting..."
    assert not os.path.exists(test_txt_path), "test.txt exists, please remove it and rerun! Exiting..."

    if flags.object_classes == "multi":
        class_name = ["chair", "table", "car", "airplane"]
    else:
        class_name = [str(flags.object_classes)]

    # load Point-Clouds - first
    syn_id = snc_category_to_synth_id()[class_name[0]]
    class_dir = osp.join(top_in_dir, syn_id)
    pc_data_train, pc_data_val, pc_data_test = load_and_split_all_point_clouds_under_folder(
        class_dir, n_threads=8, file_ending=".ply", verbose=True)

    # load Point-Clouds - the rest
    for i in range(1, len(class_name)): 
        syn_id = snc_category_to_synth_id()[class_name[i]]
        class_dir = osp.join(top_in_dir, syn_id)
        pc_data_train_curr, pc_data_val_curr, pc_data_test_curr = load_and_split_all_point_clouds_under_folder(
            class_dir, n_threads=8, file_ending=".ply", verbose=True)
        pc_data_train.merge(pc_data_train_curr)
        pc_data_val.merge(pc_data_val_curr)
        pc_data_test.merge(pc_data_test_curr)

    # Get labels 
    train_paths = arrange_split(pc_data_train.labels, top_in_dir)
    val_paths = arrange_split(pc_data_val.labels, top_in_dir)
    test_paths = arrange_split(pc_data_test.labels, top_in_dir)

    # Save numpy arrays to txt files 
    np.savetxt(train_txt_path, train_paths, fmt='%s')
    np.savetxt(val_txt_path, val_paths, fmt='%s')
    np.savetxt(test_txt_path, test_paths, fmt='%s')


    print("DATA SPLIT IS DONE!")


# def read_splits():
#     # You can use: 
#     # reconstruction/src/in_out.py 
#     # load_all_point_clouds_under_folder() 
#     # function with filenames you get from 
#     # txt files 



if __name__ == "__main__": 
    create_splits()
    print("")


