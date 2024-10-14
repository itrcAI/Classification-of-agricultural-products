

import torch
import torch.utils.data as data
import torchnet as tnt
import numpy as np
import pandas as pd
import datetime as dt
import matplotlib.pyplot as plt
import shutil
from scipy.stats import norm
import random
from collections import Counter
from mpl_toolkits.mplot3d import Axes3D



from sklearn.model_selection import KFold
from sklearn.metrics import confusion_matrix
from torch import Tensor
import os
import json
import pickle as pkl
import argparse
import pprint



def extract_all_pixle(path: str,
                      save_fig: str) -> None:
    
    """"
Plot the number of pixel for each sample
    """""
    all_npy_file = get_all_npy(path)
    # step 1 -> initialize dictionary for save number of pixle for each file
    all_pixle_for_each_file = dict()
    
    # step 2 -> iterate in files and extract data
    for name , path in all_npy_file.items():
        pixle = np.load(path).shape[2]
        all_pixle_for_each_file.update({int(name) : pixle})
        
        
        # step 2.1 -> sort data based names
        all_pixle_for_each_file = dict(
            sorted(all_pixle_for_each_file.items(), key=lambda item:item[0])
        )
    
    point_plot(all_pixle_for_each_file,save_fig,"Sample","Number Of Pixel","Pixel counts" )


def extract_sample_in_pixel(path: str,
                      save_fig: str,
                      threshold: int ) -> None:
    
    """"
Plot the Number of sample in each pixel
threshold: for clustering close groups of number of pix
    """""
    all_npy_file = get_all_npy(path)
    # step 1 -> initialize dictionary for save number of pixle for each file
    all_pixle_for_each_file = dict() 
    # step 2 -> iterate in files and extract data
    for name , path in all_npy_file.items():
        pixle = np.load(path).shape[2]
        all_pixle_for_each_file.update({int(name) : pixle})               
        # step 2.1 -> sort data based names
        all_pixle_for_each_file = dict(
            sorted(all_pixle_for_each_file.items(), key=lambda item:item[0])
        )
  
    pixel_counts = list(all_pixle_for_each_file.values())
    pixel_frequency = Counter(pixel_counts)
    thresholdd = threshold  
    grouped_counter = {}
    for key, value in pixel_frequency.items():
        grouped = False
        for group_key in grouped_counter:
            if abs(key - group_key) <= thresholdd:
                grouped_counter[group_key] += value
                grouped = True
                break
        if not grouped:
            grouped_counter[key] = value
    grouped_counter = dict(sorted(grouped_counter.items()))  
    point_plot(grouped_counter,save_fig,"Number Of Pixel","Number Of Sample","Number of sample in each pixel" )



def data_len(path: str) -> int:
  """""
  It returns the number of sample
  path: path to the dataset
  """""
  data_folder = path
  l = [f for f in os.listdir(data_folder) if f.endswith('.npy')]
  pid = [int(f.split('.')[0]) for f in l]
  pid = list(np.sort(pid))
  pid = list(map(str, pid))
  len_data = len(pid) 
  return len_data


def data_check(path: str) -> None:
  """""
  It checks the shape of each sample (it must be 3 dimension),
   dimension 0 must be 24 and dimension 1 must be 10 
  path: path to the dataset befor DATA folder
  """""
  data_folder = os.path.join(path, 'DATA')
  l = [f for f in os.listdir(data_folder) if f.endswith('.npy')]
  pid = [int(f.split('.')[0]) for f in l]
  pid = list(np.sort(pid))
  pid = list(map(str, pid))
  len_data = len(pid) 
  for item in range(len_data):
    #print("pid[item]: ", pid[item])
    x0 = np.load(os.path.join(path, 'DATA', '{}.npy'.format(pid[item])))  ##it returns Sample
    if x0.ndim != 3:
      print("Bpixel: ", pid[item], ", x0.shape: ", x0.shape, ", x0.ndim: ", x0.ndim)
    if x0.shape[0] != 55 or x0.shape[1] != 4:
      print("Bpixel: ", pid[item], ", x0.shape[0]: ", x0.shape[0], ", x0.shape[1]: ", x0.shape[1])

def check_geo(data_path: str,
              geo_path: str):
  """""
  It returns the samples which are not in geo file
  data_path: path to the .npy
  geo_path: path to the geo geo file
  """""
  l = [f for f in os.listdir(data_path) if f.endswith('.npy')]
  pid = [int(f.split('.')[0]) for f in l]
  pid = list(np.sort(pid))
  print("len_Samples:",len(pid))
  
  with open(geo_path, 'r') as f:
   data = json.load(f)
  sample_numbers = list(data.keys())
  sample_numbers = list(map(int, sample_numbers))
  sample_numbers = list(np.sort(sample_numbers))
  print("len_geo:",  len(sample_numbers))

  missing_numbers = [num for num in pid if num not in sample_numbers]
  print("len_different:",  len(missing_numbers))

  return missing_numbers

def get_all_npy(path : str) -> dict:
    """""
Load All data from the path  
path: path to the source of data (get .npy in this path)
    """""   
    files = dict()
    for root, dir , _files in os.walk(path):
        for file in _files:
            if file.endswith(".npy"):
                name = file.split(".npy")[0]
                files[name] = os.path.join(root, file) 
    return files


def check_samples_existence(path: str, sample_names: list) -> dict:
    """
    Check if samples with the given names exist among the loaded .npy samples
    path: path to the source of data
    sample_names: list of sample names to check for existence
    """
    loaded_samples = get_all_npy(path)
    results = {}
    for sample_name in sample_names:
        results[sample_name] = sample_name in loaded_samples

    for sample_name, exists in results.items():
        if exists:
            print(f"Sample with name '{sample_name}' exists.")
        else:
            print(f"Sample with name '{sample_name}' does not exist.")
    
    return results



def remove_dim(dim: int,
                npy_files : list,
                list_spec : list) -> np.array:
    """""
dim: which dimension to remove(0:date,  1:spec, 2:pixel)
It takes the list of spectrum then remove them
npy_files: Source data which comes from def get_all_npy 
list_spec:  list of spectrum which removes
    """""
    files_result = {}
    for name , npy_files in npy_files.items():
        data = np.load(npy_files)
        delsp = np.delete(data, list_spec, axis=dim)  # Delete along axis 1 (columns)
        files_result[name] = delsp
    return files_result

def elminate_date(npy_files : list,
                 list_dates : list) -> np.array:
    """""
It takes the list of data and then zeros out the required dates
npy_files: Source data which comes from def get_all_npy 
list_dates:  list of dates which turns to zeros
    """""
    files_result = {}
    for name , npy_file in npy_files.items():
        data = np.load(npy_file)
        for date in list_dates:
            matrix_date = data[date]
            data[date] = np.zeros(matrix_date.shape)
        files_result[name] = data
    return files_result


def remove_dimension(dim:int,
                    list_spec: list,
                    path_source: str,
                    path_Rsave: str) -> None:
    """""
dim: which dimension to remove(0:date,  1:spec, 2:pixel)
It takes the list of spectrum then remove them with save the result
list_spec:  list of spectrum which removes
path_source:  path to the source of data (get .npy in this path)
path_Rsave: path to create and save results (save .npy in this path)
    """""
    os.makedirs(path_Rsave, exist_ok=True)
    npy_files = get_all_npy(path_source)
    x = remove_dim(dim,npy_files,list_spec)
    for file in x:
        path = os.path.join(path_Rsave, '{}.npy'.format(file))
        np.save(path, x[file])

def remove_dimension_pkl(dim:int,
                    list_spec,
                    path_source: str,
                    path_Rsave: str) -> None:
    """""
dim: which dimension to remove(0:date,  1:spec)
It takes the list of dates and spectrum then remove them with save the result
list_spec:  list of spectrum or dates which removes
path_source:  path to the source of data (get .pkl in this path)
path_Rsave: path to create and save results (save .pkl in this path)
    """""
    os.makedirs(path_Rsave, exist_ok=True)
    load_pkl=  pkl.load(open( path_source, 'rb'))
    load_pkl_li = list(load_pkl)
    del_0 = np.delete(load_pkl_li[0], list_spec, axis=dim)
    load_pkl_li[0] = del_0
    del_1 = np.delete(load_pkl_li[1], list_spec, axis=dim)
    load_pkl_li[1] = del_1
    load_pkl_tu = tuple(load_pkl_li)
    pkl.dump(load_pkl_tu, open(os.path.join(path_Rsave,'s2_pkl.pkl'), 'wb'))


def zero_pad(list_dates: list,
             path_source: str,
             path_Rsave: str) -> None:
    """""
For zero padding of specific dates with save the result
list_dates:  list of dates which turns to zeros
path_source:  path to the source of data (get .npy in this path)
path_Rsave: path to create and save results (save .npy in this path)
    """""
    os.makedirs(path_Rsave, exist_ok=True)
    npy_files = get_all_npy(path_source)
    x = elminate_date(npy_files, list_dates)
    for file in x:
        path = os.path.join(path_Rsave, '{}.npy'.format(file))
        np.save(path, x[file])

def point_plot(data,
               path : str,
               x_lable : str,
               y_lable : str,
               title : str):
    """"""""""
It is used for general Plot
    """""""""  
    plt.figure(figsize=(15, 10))

    x = list(data.keys())
    y = list(data.values())
    x = list(map(lambda x : str(x), x))

    for _x, _y in zip(x, y):
        plt.text(_x, _y, f'{_y:.0f}', fontsize=9, ha='center', va='bottom')
    plt.plot(x, y, marker='o', linestyle='--')
    plt.xlabel(x_lable)
    plt.ylabel(y_lable)
    plt.title(title)
    plt.grid(True)
    plt.savefig(path)
    plt.close()


def split_data_labels(path_source: str,
                      path_Rsave: str,
                      labels: str) -> None:
    """""
Split sample in diffrent folder by they labels name
labels:  label_19class  or  label_44class
path_source:  path to the source of data (get befor DATA in this path)
path_Rsave: path to create and save results (get befor DATA in this path))
    """""                      
    data_folder = os.path.join(path_source, 'DATA')
    meta_folder = os.path.join(path_source, 'META')
    data_folder1 = os.path.join(path_Rsave, 'DATA')
    sub_classes = None

    npy_files = [f for f in os.listdir(data_folder) if f.endswith('.npy')]
    npy_names = [f.split(".")[0] for f in npy_files]
    npy_names.sort(key=lambda x: int(x))
    # Load labels from JSON file
    with open(os.path.join(meta_folder, 'labels.json'), 'r') as f:
        labels_data = json.load(f)
        labels_dict = labels_data[labels]
    # Create a dictionary to store npy data based on labels
    grouped_npy = {}
    # Iterate through npy files and their corresponding labels
    for npy_name in npy_names:
        label = labels_dict[npy_name]
        # Filter based on sub_classes (optional)
        if sub_classes is not None and label not in sub_classes:
            continue
        # Group npy data by label
        if label not in grouped_npy:
            grouped_npy[label] = []
        grouped_npy[label].append(npy_name)
    # Create folders for each label group and save npy files
    for label, npy_names in grouped_npy.items():
        label_folder = os.path.join(data_folder1, str(label))
        if not os.path.exists(label_folder):
            os.makedirs(label_folder)
        for npy_name in npy_names:
        # Construct original and new paths
            npy_path = os.path.join(data_folder, npy_name + ".npy")
            new_npy_path = os.path.join(label_folder, npy_name + ".npy")
        # Copy the npy file
            with open(npy_path, 'rb') as src, open(new_npy_path, 'wb') as dst:
                shutil.copyfileobj(src, dst)
    print("Successfully grouped and saved npy files based on labels.")

def labels_count(path: str,
                 classes: str):
  """""
  It returns the number of sample in each classes
  path: path to the .JSON
  classes: 'label_51class'
  """""
  with open( path , 'r') as file:
    data = json.load(file)

  label_counts = {}
  for label, count in data[classes].items():
    if count in label_counts:
        label_counts[count] += 1
    else:
        label_counts[count] = 1

  label_counts = dict(sorted(label_counts.items(), key=lambda item:item[0]))
  for count, num_instances in label_counts.items():
    print(f"label {count}: {num_instances}")


####for labels count. with DATA folder and labels.json### use "Data_distribution"#####
def point_plot2(data,
               D_list,  ##List of deleted classes
               path : str,
               x_labels_list: list, 
               x_lable : str,
               y_lable : str,
               title : str):
    
    for i in D_list:
     del data[i]
    
    plt.figure(figsize=(15, 10))

    x = list(data.keys())
    y = list(data.values())
    x = list(map(lambda x : str(x), x))

    for _x, _y in zip(x, y):
        plt.text(_x, _y, f'{_y:.0f}', fontsize=9, ha='center', va='bottom')
    plt.plot(x, y, marker='o', linestyle='--')
    plt.xticks(x, x_labels_list)
    plt.xlabel(x_lable)
    plt.ylabel(y_lable)
    plt.title(title)
    plt.grid(True)
    plt.savefig(path)
    print("data.keys():", data.keys())
    print("len(data.keys()):", len(data.keys()))
    #plt.show()
    plt.close()

def Data_distribution(dataset_folder, label_class, res_dir, Delet_label_class, x_labels_list ):

    """""
    It returns the number of sample in each classes. sample must be in DATA folder
    dataset_folder: path dataset before DATA and META
    label_class: 'label_51class'
    res_dir: save result direction
    Delet_label_class: do not count deleted labels
    x_labels_list: real name of each labels
    """""

    if not os.path.exists(res_dir):
        os.makedirs(res_dir)

    data_folder = os.path.join(dataset_folder , 'DATA')
    l = [f for f in os.listdir(data_folder) if f.endswith('.npy')]
    pid = [int(f.split('.')[0]) for f in l]
    pid = list(np.sort(pid))

    with open(os.path.join(dataset_folder, 'META', 'labels.json'), 'r') as file:
        data = json.load(file)
    Dic = data[label_class]
    converted_Dic = {int(key): value for key, value in Dic.items()}
    Final_dic = {key: converted_Dic[key] for key in pid if key in converted_Dic}

    class_19_44 = list(Final_dic.values())
    counter = {}
    for _class in class_19_44:
        if _class in counter:
            counter[_class] += 1
        elif _class not in counter:
            counter[_class] = 0
    counter = dict(sorted(counter.items(), key=lambda item:item[0]))
    save_path_cn = os.path.join(res_dir, "number_of_classes.png")
    point_plot2(counter,Delet_label_class, save_path_cn,x_labels_list, "Classes", "Number", "Number of each class")
    for count, num_instances in counter.items():
        print(f"label {count}: {num_instances}")

########################################
def split_data_test_train_flat(path_source: str, path_train_save: str, path_test_save: str, labels: str, test_ratio: float, seed: int) -> None:

    """""
    split data from each classes
    path_source: path dataset before DATA and META
    path_train_save: save train result direction
    path_test_save: save test result direction
    labels: 'label_51class'
    test_ratio: ratio of split
    seed: seed
    """""
    random.seed(seed)
    data_folder = os.path.join(path_source, 'DATA')
    meta_folder = os.path.join(path_source, 'META')
    sub_classes = None

    npy_files = [f for f in os.listdir(data_folder) if f.endswith('.npy')]
    npy_names = [f.split(".")[0] for f in npy_files]
    npy_names.sort(key=lambda x: int(x))

    with open(os.path.join(meta_folder, 'labels.json'), 'r') as f:
        labels_data = json.load(f)
        labels_dict = labels_data[labels]

    for npy_name in npy_names:
        label = labels_dict[npy_name]
        if sub_classes is not None and label not in sub_classes:
            continue

        if random.random() < test_ratio:
            dest_folder = path_test_save
        else:
            dest_folder = path_train_save

        if not os.path.exists(dest_folder):
            os.makedirs(dest_folder)

        npy_path = os.path.join(data_folder, npy_name + ".npy")
        new_npy_path = os.path.join(dest_folder, npy_name + ".npy")
        with open(npy_path, 'rb') as src, open(new_npy_path, 'wb') as dst:
            shutil.copyfileobj(src, dst)

    print("Successfully split and saved npy files into test and train folders without subdirectories based on labels and specified ratio.")



def extract_ND_3D(path_source: str,
                       path_Rsave: str,
                       spectrum_index: int) -> None:

    """""
    Normal distributionT (ND) of all DATA for mean of all time
    path_source: path dataset before DATA and META
    path_Rsave: save result direction
    spectrum_index: which spectrum
    """""
    
    all_npy_file = get_all_npy(path_source)
    # step 1 -> initialize dictionary for save number of pixle for each file
    all_pixle_for_each_file = dict()
    all_result = []
    # step 2 -> iterate in files and extract data
    for name , path in all_npy_file.items():
        data = np.load(path)
        data  = data[:,spectrum_index,:]
        
        #means = np.mean(data, axis=1)
        n =data.shape[0]
        means = np.zeros(n)

        stds = np.std(data, axis=1)
        result = np.column_stack((means, stds))
        mean_col1 = np.mean(result[:, 0])
        mean_col2 = np.mean(result[:, 1])
        result = np.array([[mean_col1, mean_col2]])
        all_result.append(result)
  
    final_result = np.vstack(all_result)
    datan, pixeln = final_result.shape
    fig = plt.figure(figsize=(20, 16))
    ax = fig.add_subplot(111, projection='3d')

    for i in range(datan):
        sample_mean = final_result[i, 0]
        std = final_result[i, 1]
        x = np.linspace(sample_mean - 4*std, sample_mean + 4*std, 100)
        y = np.full_like(x, i)  
        z = norm.pdf(x, sample_mean, std)
        ax.plot(x, y, z)

    ax.set_xlabel('mean value ')
    ax.set_ylabel('time')
    ax.set_zlabel('Probability density')
    plt.title('3D graph of standard deviation')
    save_path_cn = os.path.join(path_Rsave, "std_var_3D.png")
    plt.savefig(save_path_cn)
    plt.close()
    


