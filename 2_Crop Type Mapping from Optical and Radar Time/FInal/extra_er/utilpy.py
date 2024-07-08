# %%
import torch
import torch.utils.data as data
import torchnet as tnt
import numpy as np
import pandas as pd
import datetime as dt

from sklearn.model_selection import KFold
from sklearn.metrics import confusion_matrix
from torch import Tensor
import os
import json
import pickle as pkl
import argparse
import pprint

# %%
def data_len(path: str) -> int:
  """""
  It returns the number of sample
  path: path to the dataset befor DATA folder
  """""
  data_folder = os.path.join(path, 'DATA')
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
    print("pid[item]: ", pid[item])
    x0 = np.load(os.path.join(path, 'DATA', '{}.npy'.format(pid[item])))  ##it returns Sample
    if x0.ndim != 3:
      print("Bpixel: ", pid[item], ", x0.shape: ", x0.shape, ", x0.ndim: ", x0.ndim)
    if x0.shape[0] != 24 or x0.shape[1] != 10:
      print("Bpixel: ", pid[item], ", x0.shape[0]: ", x0.shape[0], ", x0.shape[1]: ", x0.shape[1])


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
    pkl.dump(load_pkl_tu, open(os.path.join(path_Rsave,'S2-2017-T31TFM-meanstd.pkl'), 'wb'))


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




# %%
#aaa = data_len("/home/mhbokaei/shakouri/CropTypeMappinp/multi_sensor/DATA_split")
#print(aaa)

# %%
#data_check("/home/mhbokaei/shakouri/CropTypeMappinp/multi_sensor/DATA_split")

# %%
#listt =[0,1]
#source = "/home/mhbokaei/shakouri/test/Satellite/Alldata/dataset_test_01/DATA"
#save = "/home/mhbokaei/shakouri/test/Satellite/Alldata/dataset_test_01_zero/DATA"
#zero_pad(listt,source,save)

# %%
#data1 = np.load("/home/mhbokaei/shakouri/CropTypeMappinp/multi_sensor/All_dataset/dataset_100/test_folder/s2_data/DATA/10.npy")
#print("data1.shape: ",data1.shape)
#print("data1[15]: ",data1[5])

# %%
#listt =[1,5,9,14,21]
#dim = 0
## validation  test   /home/mhbokaei/shakouri/CropTypeMappinp/multi_sensor/All_dataset/dataset_10000/dataset_folder/s1_data/  val_folder  test_folder
#source = "/home/mhbokaei/shakouri/CropTypeMappinp/multi_sensor/DATA_split/train/"
#save = "/home/mhbokaei/shakouri/CropTypeMappinp/multi_sensor/All_dataset/dataset_10000/dataset_folder/s2_data/DATA"
#remove_dimension(dim,listt,source,save)

# %%
#data1 = np.load("/home/mhbokaei/shakouri/CropTypeMappinp/multi_sensor/All_dataset/dataset_10000/dataset_folder/s2_data/DATA/9999.npy")
#print("data1.shape: ",data1.shape)
#print("data1[15]: ",data1[10])

# %%
#listt =[1,5,9,14,21]
#dim = 0
#source = "/home/mhbokaei/shakouri/CropTypeMappinp/multi_sensor/S2-2017-T31TFM-meanstd.pkl"
#save = "/home/mhbokaei/shakouri/CropTypeMappinp/multi_sensor/remoo/s2_data"
#remove_dimension_pkl(dim,listt,source,save)



# %%
preLoad_pkl = pkl.load(open("/home/mhbokaei/shakouri/CropTypeMappinp/multi_sensor/remoo/s1_data/S2-2017-T31TFM-meanstd.pkl", 'rb'))
print("preLoad_pkl[0].shape:", preLoad_pkl[0].shape)
print("preLoad_pkl[1].shape:", preLoad_pkl[1].shape)
#print("preLoad_pkl:", preLoad_pkl)

