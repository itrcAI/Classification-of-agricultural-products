import tensorflow as tf
from tensorflow.keras import layers, models,Model, Input
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    GlobalAveragePooling1D,
    MaxPooling1D,
    UpSampling1D,
    Conv2D, 
    Conv1D,
    Bidirectional, 
    Dense, 
    Flatten, 
    Add, 
    GlobalAveragePooling2D, 
    GlobalMaxPooling2D, 
    Reshape, 
    Activation, 
    Concatenate, 
    multiply, 
    BatchNormalization,
    Dropout,
    ZeroPadding1D,
    Cropping1D
)
from tensorflow.keras.regularizers import l2
from tensorflow.keras.optimizers import Adam, Nadam
from functools import partial
import numpy as np
import random
from tensorflow.keras.mixed_precision import set_global_policy
set_global_policy('mixed_float16')

def set_seeds(seed=42):
    np.random.seed(seed)
    tf.random.set_seed(seed)
    random.seed(seed)

set_seeds(42)




class model_creation:
    def __init__(self, model_name, input_shape, num_classes):
        self.model_name = model_name
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.num_additional_features = 5

        # Define DefaultConv2D
        self.DefaultConv2D = partial(tf.keras.layers.Conv2D, kernel_size=3, strides=1,
                                     padding="same", kernel_initializer="he_normal", use_bias=False)

        # Define shared layers for channel attention
        self.Shared_Layer_One = Dense(self.input_shape[-1] // 8, activation='relu')
        self.Shared_Layer_Two = Dense(self.input_shape[-1])

    def build_model(self):
        # Mapping model names to methods
        model_dict = {
            'conv_lstm1d': self.build_conv_lstm1d_model,
            'cbam_resnet': self.build_cbam_resnet_model,
            'lstm': self.build_lstm_model,
            'tempcnn1d': self.build_tempcnn_model,
            'tempcnn2d': self.build_tempcnn2d_model,
            'bilstm' : self.build_bi_lstm_model,
            'cnn_lstm': self.build_cnn_lstm_model,
            'lstm_cnn': self.build_lstm_cnn_model,
            'custom_model': self.build_custom_model,
            'autoencoder_lstm': self.build_autoencoder_lstm_model,
            'autoencoder_cnn': self.build_autoencoder_cnn_model,
            'unet_lstm': self. build_unet_lstm_model
        }

        # Check if model name is valid
        if self.model_name not in model_dict:
            raise ValueError(f"Model name {self.model_name} is not recognized. Available models: {list(model_dict.keys())}")

        # Return the corresponding model
        return model_dict[self.model_name]()
    

    def build_custom_model(self):

        # Time series input
        time_series_input = Input(shape=self.input_shape, name="time_series_input")
        
        # Simplified feature extraction with Conv1D and pooling
        x = Conv1D(64, kernel_size=3, activation='relu', kernel_regularizer=l2(0.001))(time_series_input)
        x = BatchNormalization()(x)
        x = Dropout(0.2)(x)
        x = Conv1D(64, kernel_size=3, activation='relu', kernel_regularizer=l2(0.001))(time_series_input)
        x = BatchNormalization()(x)
        x = Dropout(0.2)(x)
        x = Conv1D(32, kernel_size=3, activation='relu', kernel_regularizer=l2(0.001))(x)
        x = BatchNormalization()(x)
        x = Dropout(0.2)(x)
        x = GlobalAveragePooling1D()(x)  
        
        # Additional features input
        additional_features_input = Input(shape=(self.num_additional_features,), name="additional_features_input")
        
        # Combine features
        combined_features = Concatenate()([x, additional_features_input])
        
        # Classification layers
        y = Dense(256, activation='relu', kernel_regularizer=l2(0.001))(combined_features)
        y = BatchNormalization()(y)
        y = Dropout(0.3)(y)
        y = Dense(64, activation='relu', kernel_regularizer=l2(0.001))(y)
        output = Dense(self.num_classes, activation='softmax', name="output_layer")(y)
        
        # Model
        model = Model(inputs=[time_series_input, additional_features_input], outputs=output)
        
        # Compile with gradient clipping and lower learning rate
        model.compile(optimizer=Adam(learning_rate=0.0001, clipnorm=1.0), loss='categorical_crossentropy', metrics=['accuracy'])

        return model

    def build_cnn_lstm_model(self):
        model = models.Sequential()
        
        # CNN layers
        model.add(Conv1D(64, kernel_size=7, activation='elu', input_shape=self.input_shape))
        model.add(layers.BatchNormalization())
        model.add(layers.Dropout(0.2))
        model.add(Conv1D(64, kernel_size=7, activation='elu'))
        model.add(layers.BatchNormalization())
        model.add(layers.Dropout(0.2))
        model.add(Conv1D(32, kernel_size=7, activation='elu'))
        model.add(layers.BatchNormalization())
        model.add(layers.Dropout(0.2))
        # model.add(Flatten())
        # model.add(Dense(256, activation='elu'))
        # model.add(Dense(self.num_classes, activation='softmax'))
        
        # model.add(Conv1D(128, kernel_size=3, activation='relu', padding='same'))
        # model.add(BatchNormalization())
        # model.add(layers.Dropout(0.3))
        
        # LSTM layers
        model.add(layers.LSTM(32, return_sequences=False, dropout=0.3))
        model.add(BatchNormalization())
        
        # BiLSTM layer
        # model.add(Bidirectional(layers.LSTM(32, return_sequences=False, dropout=0.3)))
        # model.add(BatchNormalization())
        
        # # Flatten the output of LSTM
        model.add(Flatten())
        model.add(Dense(256, activation='relu'))
        model.add(BatchNormalization())
        model.add(layers.Dropout(0.2))
        # Dense layers
        model.add(Dense(64, activation='relu'))
        # model.add(layers.Dropout(0.3))
        model.add(Dense(self.num_classes, activation='softmax'))
        
        # Compile the model
        model.compile(optimizer=Nadam(learning_rate=0.001), 
                    loss='categorical_crossentropy', 
                    metrics=['accuracy'])
        
        return model
    
    def build_lstm_cnn_model(self):
        model = models.Sequential()
        
        # LSTM layers
        model.add(layers.Input(shape=self.input_shape))
        model.add(layers.LSTM(128, return_sequences=True, dropout=0.3))
        # model.add(layers.LSTM(128, return_sequences=True, dropout=0.3))
        
        # CNN layers after LSTM output
        model.add(Conv1D(64, kernel_size=3, activation='relu', padding='same'))
        model.add(BatchNormalization())
        model.add(layers.Dropout(0.3))
        
        model.add(Conv1D(128, kernel_size=3, activation='relu', padding='same'))
        model.add(BatchNormalization())
        model.add(layers.Dropout(0.3))
        
        # Flatten the output
        model.add(Flatten())
        
        # Dense layers
        model.add(Dense(64, activation='relu'))
        model.add(layers.Dropout(0.3))
        model.add(Dense(self.num_classes, activation='softmax'))
        
        # Compile the model
        model.compile(optimizer=Adam(learning_rate=0.001), 
                    loss='categorical_crossentropy', 
                    metrics=['accuracy'])
        
        return model

    
    def build_bi_lstm_model(self):
        model = models.Sequential()
        
        # Input layer
        model.add(layers.Input(shape=self.input_shape))
        
        # LSTM layer
        model.add(layers.LSTM(128, return_sequences=True))
        model.add(layers.BatchNormalization())
        model.add(layers.Dropout(0.2))
        
        # BiLSTM layer
        model.add(Bidirectional(layers.LSTM(128, return_sequences=False)))
        model.add(layers.BatchNormalization())
        model.add(layers.Dropout(0.2))
        
        # Flatten the output for Dense layers
        model.add(layers.Flatten())
        
        # Dense layers
        model.add(layers.Dense(64, activation='relu'))
        model.add(layers.Dense(self.num_classes, activation='softmax'))  # Adjust activation for the number of classes
        model.compile(optimizer=Adam(learning_rate=1e-3),
                loss='categorical_crossentropy',
                metrics=['accuracy'])
        return model

    def build_conv_lstm1d_model(self):
        model = models.Sequential()
        model.add(layers.Input(shape=self.input_shape))

        # ConvLSTM1D Layers
        model.add(layers.ConvLSTM1D(filters=64, kernel_size=3, strides=1, padding='same', return_sequences=True,
                                    activation='relu', dropout=0.3))
        model.add(layers.BatchNormalization())
    

        # model.add(layers.ConvLSTM1D(filters=128, kernel_size=3, padding='same', return_sequences=True,
        #                             activation='relu'))
        # model.add(layers.BatchNormalization())
        # model.add(layers.Dropout(0.2))

        # model.add(layers.ConvLSTM1D(filters=128, kernel_size=3, padding='same', return_sequences=False,
        #                             activation='relu'))
        # model.add(layers.BatchNormalization())
        # model.add(layers.Dropout(0.2))

        # Dense layer to classify the time-series
        model.add(layers.Flatten())
        model.add(layers.Dense(32, activation='relu'))
        model.add(layers.Dense(self.num_classes, activation='softmax'))

        # Compile the model
        model.compile(optimizer=Adam(learning_rate=0.001), loss='categorical_crossentropy', metrics=['accuracy'])

        return model

    class ResidualUnit(tf.keras.layers.Layer):
        def __init__(self, filters, strides=1, activation="relu", **kwargs):
            super().__init__(**kwargs)
            self.activation = tf.keras.activations.get(activation)
            self.main_layers = [
                layers.Conv2D(filters, kernel_size=3, strides=strides, padding='same'),
                layers.BatchNormalization(),
                self.activation,
                layers.Conv2D(filters, kernel_size=3, padding='same'),
                layers.BatchNormalization()
            ]
            self.skip_layers = [layers.Conv2D(filters, kernel_size=1, strides=strides, padding='same')]

        def call(self, inputs):
            Z = inputs
            for layer in self.main_layers:
                Z = layer(Z)
            skip_Z = inputs
            for layer in self.skip_layers:
                skip_Z = layer(skip_Z)
            return self.activation(Z + skip_Z)

    def _cbam_block(self, cbam_feature, ratio=8):
        cbam_feature = self._channel_attention(cbam_feature, ratio)
        cbam_feature = self._spatial_attention(cbam_feature)
        return cbam_feature

    def _channel_attention(self, input_feature, ratio=8):
        channel_axis = -1  # TensorFlow uses channels_last format by default
        channel = input_feature.shape[channel_axis]

        avg_pool = GlobalAveragePooling2D()(input_feature)
        avg_pool = Reshape((1, 1, channel))(avg_pool)
        avg_pool = self.Shared_Layer_One(avg_pool)
        avg_pool = self.Shared_Layer_Two(avg_pool)

        max_pool = GlobalMaxPooling2D()(input_feature)
        max_pool = Reshape((1, 1, channel))(max_pool)
        max_pool = self.Shared_Layer_One(max_pool)
        max_pool = self.Shared_Layer_Two(max_pool)

        cbam_feature = Add()([avg_pool, max_pool])
        cbam_feature = Activation('sigmoid')(cbam_feature)

        return multiply([input_feature, cbam_feature])

    def _spatial_attention(self, input_feature):
        avg_pool = tf.reduce_mean(input_feature, axis=-1, keepdims=True)
        max_pool = tf.reduce_max(input_feature, axis=-1, keepdims=True)
        concat = Concatenate(axis=-1)([avg_pool, max_pool])
        cbam_feature = Conv2D(filters=1, kernel_size=7, strides=1, padding='same', activation='sigmoid')(concat)
        return multiply([input_feature, cbam_feature])

    def build_cbam_resnet_model(self):
        input1 = layers.Input(shape=self.input_shape)

        # Initial Conv Layer
        conv1 = layers.Conv2D(64, (3, 3), padding='same')(input1)
        bn1 = layers.BatchNormalization()(conv1)
        relu1 = layers.Activation('relu')(bn1)

        # Residual Block + CBAM
        resblock1 = self.ResidualUnit(filters=64)(relu1)
        cbam1 = self._cbam_block(resblock1, ratio=8)

        # Global Average Pooling and Output Layer
        global_average_pooling = GlobalAveragePooling2D()(cbam1)
        output = layers.Dense(self.num_classes, activation="softmax")(global_average_pooling)

        model = models.Model(inputs=input1, outputs=output)
        model.compile(optimizer=Adam(learning_rate=0.001), loss='categorical_crossentropy', metrics=['accuracy'])
        
        return model

    def build_lstm_model(self):
        model = models.Sequential()

        # LSTM layer
        model.add(layers.Input(shape=self.input_shape))
        model.add(layers.LSTM(128, return_sequences=True))
        model.add(layers.BatchNormalization())
        model.add(layers.Dropout(0.2))
        model.add(layers.Flatten())

        # Dense layers
        model.add(layers.Dense(64, activation='relu'))
        model.add(layers.BatchNormalization())
        model.add(layers.Dropout(0.2))
        model.add(layers.Dense(64, activation='relu'))

        # Output layer
        model.add(layers.Dense(self.num_classes, activation='softmax'))

        # Compile the model
        model.compile(optimizer=Nadam(learning_rate=0.001), 
                    loss='categorical_focal_crossentropy', 
                    metrics=['accuracy'])

        return model

    def build_tempcnn_model(self):
        # Define the TempCNN model
        model = Sequential([
            Conv1D(64, kernel_size=9, activation='elu', input_shape=self.input_shape),
            layers.BatchNormalization(),
            layers.Dropout(0.2),
            Conv1D(64, kernel_size=9, activation='elu'),
            layers.BatchNormalization(),
            layers.Dropout(0.2),
            Conv1D(32, kernel_size=9, activation='elu'),
            layers.BatchNormalization(),
            layers.Dropout(0.2),
            Flatten(),
            Dense(256, activation='elu'),
            Dense(self.num_classes, activation='softmax')
        ])
        
        # Compile the model
        model.compile(optimizer=Adam(learning_rate=0.001), loss='categorical_crossentropy', metrics=['accuracy'])
        
        return model

    def build_tempcnn2d_model(self):
        # Define the TempCNN model
        model = Sequential([
            Conv2D(64, kernel_size=(5, 5), activation='relu', input_shape=self.input_shape),
            layers.BatchNormalization(),
            layers.Dropout(0.2),
            layers.MaxPooling2D(pool_size=(2, 2)),
            Conv2D(128, kernel_size=(3, 3), activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.2),
            layers.MaxPooling2D(pool_size=(2, 2)),
            Flatten(),
            Dense(128, activation='relu'),
            Dense(self.num_classes, activation='softmax')
        ])

        # Compile the model
        model.compile(optimizer=Adam(learning_rate=0.001), loss='categorical_crossentropy', metrics=['accuracy'])
        
        return model
    
    def build_autoencoder_lstm_model(self):
        # Encoder part
        input_layer = Input(shape=self.input_shape, name="encoder_input")
        encoded = Conv1D(64, kernel_size=3, activation="relu", padding="same")(input_layer)
        encoded = BatchNormalization()(encoded)
        encoded = Conv1D(32, kernel_size=3, activation="relu", padding="same")(encoded)
        encoded = BatchNormalization()(encoded)
        encoded = GlobalAveragePooling1D()(encoded)

        # Repeat encoded output for LSTM
        encoded_expanded = Reshape((1, 32))(encoded)

        # LSTM layer
        lstm_out = layers.LSTM(32, return_sequences=True, dropout=0.2)(encoded_expanded)
        lstm_out = Flatten()(lstm_out)

        # Decoder part
        decoder = Dense(self.input_shape[0] * 32, activation="relu")(lstm_out)
        decoder = Reshape((self.input_shape[0], 32))(decoder)
        decoder = Conv1D(32, kernel_size=3, activation="relu", padding="same")(decoder)
        decoder = BatchNormalization()(decoder)
        decoder = Conv1D(64, kernel_size=3, activation="relu", padding="same")(decoder)
        decoder = BatchNormalization()(decoder)
        decoder_output = Conv1D(self.input_shape[-1], kernel_size=3, activation="sigmoid", padding="same")(decoder)

        # Classification layers
        x = GlobalAveragePooling1D()(decoder_output)
        output_layer = Dense(self.num_classes, activation="softmax", name="output_layer")(x)

        # Model
        model = Model(inputs=input_layer, outputs=output_layer)
        model.compile(optimizer=Nadam(learning_rate=0.001), loss="categorical_crossentropy", metrics=["accuracy"])
        return model

    def build_autoencoder_cnn_model(self):

        # Encoder part
        input_layer = Input(shape=self.input_shape, name="encoder_input")
        encoded = Conv1D(64, kernel_size=3, activation="relu", padding="same")(input_layer)
        encoded = BatchNormalization()(encoded)
        encoded = Conv1D(32, kernel_size=3, activation="relu", padding="same")(encoded)
        encoded = BatchNormalization()(encoded)
        encoded = GlobalAveragePooling1D()(encoded)

        # CNN layer for feature extraction
        cnn_out = Dense(32, activation="relu")(encoded)

        # Decoder part
        decoder = Dense(self.input_shape[0] * 32, activation="relu")(cnn_out)
        decoder = Reshape((self.input_shape[0], 32))(decoder)
        decoder = Conv1D(32, kernel_size=3, activation="relu", padding="same")(decoder)
        decoder = BatchNormalization()(decoder)
        decoder = Conv1D(64, kernel_size=3, activation="relu", padding="same")(decoder)
        decoder = BatchNormalization()(decoder)
        decoder_output = Conv1D(self.input_shape[-1], kernel_size=3, activation="sigmoid", padding="same")(decoder)

        # Classification layers
        x = GlobalAveragePooling1D()(decoder_output)
        output_layer = Dense(self.num_classes, activation="softmax", name="output_layer")(x)

        # Model
        model = Model(inputs=input_layer, outputs=output_layer)
        model.compile(optimizer=Nadam(learning_rate=0.001), loss="categorical_crossentropy", metrics=["accuracy"])
        return model
    
    def build_unet_lstm_model(self):
        # Encoder part (Contracting Path)
        input_layer = Input(shape=self.input_shape, name="input_layer")
        conv1 = Conv1D(64, kernel_size=3, activation="relu", padding="same")(input_layer)
        conv1 = BatchNormalization()(conv1)
        pool1 = MaxPooling1D(pool_size=2, padding="same")(conv1)

        conv2 = Conv1D(128, kernel_size=3, activation="relu", padding="same")(pool1)
        conv2 = BatchNormalization()(conv2)
        pool2 = MaxPooling1D(pool_size=2, padding="same")(conv2)

        # Decoder part (Expanding Path)
        up1 = UpSampling1D(size=2)(pool2)  # Upsample
        concat1 = Concatenate()([up1, conv2])  # Concatenate with skip connection
        conv3 = Conv1D(128, kernel_size=3, activation="relu", padding="same")(concat1)
        conv3 = BatchNormalization()(conv3)

        up2 = UpSampling1D(size=2)(conv2)  # Upsample
        crop_layer = Cropping1D(cropping=(0, 1))(up2)
        concat2 = Concatenate()([crop_layer, conv1])  # Concatenate with skip connection
        conv4 = Conv1D(64, kernel_size=3, activation="relu", padding="same")(concat2)
        conv4 = BatchNormalization()(conv4)

        # LSTM part
        lstm_input = layers.LSTM(64, return_sequences=False)(conv4)  # LSTM layer
        output_layer = Dense(self.num_classes, activation="softmax")(lstm_input)  # Final classification layer

        # Create model
        model = Model(inputs=input_layer, outputs=output_layer)
        model.compile(optimizer=Nadam(learning_rate=0.001), loss="categorical_crossentropy", metrics=["accuracy"])

        return model