
import os 
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
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
from tensorflow.keras import models
from tensorflow.keras.layers import Layer
import joblib
import config

class CustomCast(Layer):
    def __init__(self, dtype=None, **kwargs):
        super(CustomCast, self).__init__(**kwargs)
        self.dtype = dtype

    def call(self, inputs):
        # Cast the inputs to the desired dtype if specified
        if self.dtype is not None:
            return tf.cast(inputs, self.dtype)
        return inputs

# Register the custom object
custom_objects = {"CustomCast": CustomCast}


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



def main_func(filepath, direction, keep_labels, model_names, shape_dim):
        
        test_dir = os.path.join(filepath, 'test_data/')
        test_list = file_name_extraction(test_dir, '.csv')



        for model_name in model_names:

            save_dir = os.path.join(direction, model_name + '/')
            fit_label_encoder = joblib.load(os.path.join(save_dir, 'label_encoder_model.pkl'))

            # for class_label, encoded_value in zip(fit_label_encoder.classes_, range(len(fit_label_encoder.classes_))):
            #     print(f"Label: {class_label}, Value: {encoded_value}")

            fit_scaler = joblib.load(os.path.join(save_dir, 'scaler_model.pkl'))
            dates = np.arange(16, 54, 4)

            valid_classes = np.load(os.path.join(save_dir,'valid_classes.npy'),allow_pickle=True)
            num_classes = np.load(os.path.join(save_dir,'num_classes.npy'),allow_pickle=True)

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
            for i in dates[4:]:

                model = load_model(os.path.join(save_dir, f'{model_name}_in_month{i*7/30:.2f}model.keras'), custom_objects=custom_objects)  
                plot_model(model, to_file=os.path.join(save_dir, f'{model_name}_model_architecture.png'), show_shapes=True, show_layer_names=True)


                for name in test_list:
                    print(f'{name} is running')
                    path = os.path.join(test_dir, name)

                    Test_processor = TestProcessor(filepath_test=path, valid_classes=valid_classes, num_classes=num_classes, 
                                                    num_temporal_steps=55, shape_dim=shape_dim, label_encoder=fit_label_encoder,
                                                    scaler=fit_scaler, dates=i)
                    X_test, y_test, X_cord, Y_cord, additional_features_test = Test_processor.test_processor()

                    save_path = os.path.join(save_dir, f'robust_{model_name}_accuracy_{name}_in_{i*7/30:.2f}_month.xlsx')

                    evaluation = evaluate(model=model, X_test=X_test, y_test=y_test,additional_features = additional_features_test, label_encoder=fit_label_encoder,
                                            file_path=save_path, X_cord=X_cord, Y_cord=Y_cord, keep_labels=keep_labels)
                    # evaluation.evaluate_model()
                    evaluation.evaluate_model_reclassified()


main_func(utils.filepath, utils.direction, utils.keep_labels, utils.model_names, utils.shape_dim)