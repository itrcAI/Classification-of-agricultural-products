import os 
import time
import numpy as np
from data_proccessing import TrainProcessor, TestProcessor
from models import model_creation
from evaluation import evaluate
from sklearn.preprocessing import StandardScaler, LabelEncoder
import random
import tensorflow as tf
from tensorflow.keras.utils import plot_model
from tensorflow.keras.models import load_model
import joblib
import config

def set_seeds(seed=42):
    np.random.seed(seed)
    tf.random.set_seed(seed)
    random.seed(seed)

set_seeds(42)
from tensorflow.keras.mixed_precision import set_global_policy
set_global_policy('mixed_float16')

def train_model(model, X_train, y_train, batch_size=32, epochs=30):
    model.fit(
        X_train, y_train, 
        batch_size=batch_size, 
        epochs=epochs,
        validation_split=0.2
    )
    return model

def train_custom_model(model, X_train, y_train, additional_features, batch_size=32, epochs=30):
    model.fit(
        [X_train, additional_features], y_train, 
        batch_size=batch_size, 
        epochs=epochs,
        validation_split=0.2
    )
    return model

def file_name_extraction(directory, file_format):
    """
    Extract file names from a directory with a specific format.
    
    Parameters:
    - directory (str): Directory path.
    - file_format (str): File format to filter.
    
    Returns:
    - list: Sorted list of file names.
    """
    items = os.listdir(directory)
    file_list = [name for name in items if name.endswith(file_format)]
    return np.sort(file_list)

def folder_creation(path, folder_name):
    if not os.path.exists(os.path.join(path , folder_name + '/')):
        os.mkdir(os.path.join(path , folder_name + '/'))




def main_func(filepath, direction, train_df_file_name, model_names, shape_dim, epochs):

    train_df = os.path.join(filepath, train_df_file_name)


    num_temporal_steps = 55
    main_label_encoder = LabelEncoder()
    main_scaler = StandardScaler()

    for model_name in model_names:
        folder_creation(direction, model_name)
        save_dir = os.path.join(direction, model_name + '/')
        dates = np.arange(16, 54, 4)  
        for i in dates[4:]:
            

            train_process  = TrainProcessor(filepath_train = train_df, num_temporal_steps=num_temporal_steps, shape_dim=shape_dim, label_encoder=main_label_encoder
                                            , scaler=main_scaler, dates=i)
            X_train, y_train, additional_features_train, valid_classes, num_classes, fit_label_encoder, fit_scaler = train_process.train_processor()
            np.save(os.path.join(save_dir,'valid_classes.npy'), valid_classes)
            np.save(os.path.join(save_dir,'num_classes.npy'), num_classes)
        

            joblib.dump(fit_label_encoder, os.path.join(save_dir, 'label_encoder_model.pkl'))
            joblib.dump(fit_scaler, os.path.join(save_dir, 'scaler_model.pkl'))
            num_spectral_bands = X_train.shape[2]
            ############## Build model ###############
            
            if shape_dim == 3:
                input_shape = (num_temporal_steps, num_spectral_bands)
            elif shape_dim == 4:
                input_shape = (num_temporal_steps, num_spectral_bands, 1)
            '''
            available models are :[  conv_lstm1d,
                                    cbam_resnet,
                                    lstm,
                                    bilstm
                                    tempcnn1d,
                                    tempcnn2d,
                                    cnn_lstm,
                                    lstm_cnn,
                                    custom_model,
                                    autoencoder_lstm,
                                    autoencoder_cnn,
                                    unet_lstm
                                            ]
            '''

            model_builder = model_creation(model_name=model_name, input_shape=input_shape, num_classes=num_classes)
            model = model_builder.build_model()
            print(model.summary())
            print(len(model.layers))

            ######## training the model ###########
            start = time.time()

            model = train_model(model, X_train, y_train, batch_size=32, epochs=epochs)

            model.save(os.path.join(save_dir, f'{model_name}_in_month{i*7/30:.2f}model.keras'))
            end = time.time()
            
    

            print('Training time: {:.2f} minutes'.format((end - start) / 60))
            


main_func(utils.filepath, utils.direction, utils.train_df_file_name, utils.model_names, utils.shape_dim, utils.epochs)
