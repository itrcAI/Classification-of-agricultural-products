import tensorflow as tf
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
import natsort
from sklearn.model_selection import train_test_split

# Base class to handle common methods for both Train and Test data
class BaseProcessor:

    def __init__(self, num_temporal_steps, shape_dim, label_encoder, scaler, dates=None):
        self.num_temporal_steps = num_temporal_steps
        self.shape_dim = shape_dim
        self.label_encoder = label_encoder
        self.scaler = scaler
        self.dates = dates



    def extract_data_columns(self, df, column_name):
        columns = [col for col in df.columns if column_name in col.lower()]
        data_frame = pd.DataFrame()
        data_frame[columns] = df[columns]

        return data_frame

    def reshape_data(self, X):
        # Number of spectral bands
        num_spectral_bands = X.shape[1] // self.num_temporal_steps

        # Reshape the dataset to (samples, num_temporal_steps, num_spectral_bands)
        if self.shape_dim == 3:
            X = X.reshape(X.shape[0], self.num_temporal_steps, num_spectral_bands)
        elif self.shape_dim == 4:
            X = X.reshape(X.shape[0], self.num_temporal_steps, num_spectral_bands, 1)

        # Normalize the features along the temporal and spectral axes
        


        if self.shape_dim == 3:
            X = X.reshape(X.shape[0], self.num_temporal_steps, num_spectral_bands)
        elif self.shape_dim == 4:
            X = X.reshape(X.shape[0], self.num_temporal_steps, num_spectral_bands, 1)

        return X

# Train data processor subclass
class TrainProcessor(BaseProcessor):

    def __init__(self, filepath_train, num_temporal_steps, shape_dim, label_encoder, scaler, dates=None):
        super().__init__(num_temporal_steps, shape_dim, label_encoder, scaler, dates)
        self.filepath_train = filepath_train


    # def load_train_data(self):
    #     # Load and preprocess the training data
    #     data_train = pd.read_csv(self.filepath_train)
    #     data_train = data_train[data_train['class'] != 'p']
    #     data_train = data_train.reindex(columns=natsort.humansorted(data_train.columns)).fillna(-1)
        

    #     # Count the occurrences of each class in the training data
    #     class_counts = data_train['class'].value_counts()
    #     valid_classes = class_counts[class_counts >= 2].index
    #     data_train_filtered = data_train[data_train['class'].isin(valid_classes)]

    #     ndvi_columns = [col for col in data_train_filtered.columns if 'ndvi' in col.lower()]
    #     ndvi_df = pd.DataFrame()
    #     ndvi_df[ndvi_columns] = data_train_filtered[ndvi_columns]
    #     ndvi_df['max_ndvi'] = ndvi_df.max(axis=1)
    #     ndvi_df['min_ndvi'] = ndvi_df.min(axis=1)
    #     ndvi_df[['SA', 'SF', 'SP']] = data_train_filtered[['SA', 'SF', 'SP']]
    #     additional_features_train = ndvi_df[['max_ndvi','min_ndvi','SA', 'SF', 'SP']]

    #     # Extract features and target from the filtered data
    #     X_train = data_train_filtered.iloc[:, 0:-11].values
    #     y_train = data_train_filtered['class'].values

    #     return X_train, y_train, valid_classes, additional_features_train
    
    def load_train_data(self):
        # Load and preprocess the training data
        data_train = pd.read_csv(self.filepath_train)
        print(data_train)
        data_train = data_train[data_train['class'] != 'p']
        data_train = data_train.reindex(columns=natsort.humansorted(data_train.columns)).fillna(-1)
        

        # Count the occurrences of each class in the training data
        class_counts = data_train['class'].value_counts()
        valid_classes = class_counts[class_counts >= 2].index
        data_train_filtered = data_train[data_train['class'].isin(valid_classes)]

        ndvi_train_df = self.extract_data_columns(data_train_filtered, 'ndvi')
        green_train_df = self.extract_data_columns(data_train_filtered, 'b4')
        red_train_df = self.extract_data_columns(data_train_filtered, 'b3')
        blue_train_df = self.extract_data_columns(data_train_filtered, 'b2')

        

        cfi_index = ndvi_train_df.values * ((red_train_df.values + green_train_df.values) + (green_train_df.values - blue_train_df.values))
        cfi_column_names = [f"{i}_CFI" for i in range(55)]
        cfi_train_df = pd.DataFrame(columns=cfi_column_names)
        cfi_train_df[cfi_column_names] = cfi_index

        SWIR1 = self.extract_data_columns(data_train_filtered, 'b11')
        SWIR2 = self.extract_data_columns(data_train_filtered, 'b12')
        nbr_train_index = (SWIR1.values -  SWIR2.values) / (SWIR1.values +  SWIR2.values)
        nbr_column_names = [f"{i}_nbr2" for i in range(55)]
        nbr_train_df = pd.DataFrame(columns=nbr_column_names)
        nbr_train_df[nbr_column_names] = nbr_train_index

        b8a = self.extract_data_columns(data_train_filtered, 'b8a')
        b11 = self.extract_data_columns(data_train_filtered, 'b11')
        ndwi_train_index = (b8a.values -  b11.values) / (b8a.values +  b11.values)
        ndwi_column_names = [f"{i}_ndwi2" for i in range(55)]
        ndwi_train_df = pd.DataFrame(columns=ndwi_column_names)
        ndwi_train_df[ndwi_column_names] = ndwi_train_index

        lai_index = 2.5 * ndvi_train_df.values + 0.1
        lai_column_names = [f"{i}_LAI" for i in range(55)]
        lai_train_df = pd.DataFrame(columns=lai_column_names)
        lai_train_df[lai_column_names] = lai_index
        # print('cfi_index data frame is : ', cfi_train_df)

        ndvi_train_df['max_ndvi'] = ndvi_train_df.max(axis=1)
        ndvi_train_df['min_ndvi'] = ndvi_train_df.min(axis=1)

        additional_features_train = ndvi_train_df[['max_ndvi','min_ndvi']].reset_index(drop=True)
        # print('additional_features_train data frame is : ', additional_features_train)

        # Extract features and target from the filtered data
        X_train = data_train_filtered.iloc[:, 0:-10].reset_index(drop=True)
        # X_train_ndvi = self.extract_data_columns(data_train_filtered, 'ndvi').reset_index(drop=True)
        
        # X_train_vv = self.extract_data_columns(data_train_filtered, 'vv').reset_index(drop=True)
        # X_train_vh = self.extract_data_columns(data_train_filtered, 'vh').reset_index(drop=True)
        # X_train_savi = self.extract_data_columns(data_train_filtered, 'savi').reset_index(drop=True)
        # X_train_bsi = self.extract_data_columns(data_train_filtered, 'bsi').reset_index(drop=True)
        # X_train_slope = self.extract_data_columns(data_train_filtered, 'slope').reset_index(drop=True)
        # X_train_evi = self.extract_data_columns(data_train_filtered, 'evi').reset_index(drop=True)
        # X_train_nir = self.extract_data_columns(data_train_filtered, 'b8').reset_index(drop=True)



        X_train_evi = self.extract_data_columns(data_train_filtered, 'evi').reset_index(drop=True)
        X_train = pd.concat([X_train, additional_features_train],axis=1)
        # print('X_train : ', X_train)
        X_train = X_train.reindex(columns=natsort.humansorted(X_train.columns)).values
        y_train = data_train_filtered['class'].values

        return X_train, y_train, valid_classes, additional_features_train
        


    
    def normalize_train(self,x):
        x = x.reshape(x.shape[0], -1)
        X = self.scaler.fit_transform(x)
        return X
    
    def encode_train(self, y):
        # Transform class labels from strings to integers
        y = self.label_encoder.fit_transform(y)
        return y

    def robust_train_data(self, X_train, y_train):
        # Split training data into sub-sample
        X_train_all_dates, X_train_sub_sample, y_train_all_dates, y_train_sub_sample = train_test_split(
            X_train, y_train, test_size=0.2, random_state=42, stratify=y_train)

        if self.dates is not None:
            X_train_sub_sample[:, self.dates:, :] = 0

        X_train = np.concatenate((X_train_all_dates, X_train_sub_sample), axis=0)
        y_train = np.concatenate((y_train_all_dates, y_train_sub_sample), axis=0)

        # Convert labels to categorical (one-hot encoding)
        num_classes = len(np.unique(y_train))
        y_train = tf.keras.utils.to_categorical(y_train, num_classes)

        return X_train, y_train, num_classes

    def train_processor(self):
        X_train, y_train, valid_classes, additional_features_train = self.load_train_data()
        y_train = self.encode_train(y_train)
        X_train = self.normalize_train(X_train)
        print(X_train.shape, y_train.shape)
        additional_features_train_numpy = X_train[:,-2:]

        # costum_data_scaler = StandardScaler()
        # additional_features_train_numpy = additional_features_train.to_numpy()
        # additional_features_train_numpy = additional_features_train_numpy.reshape(additional_features_train_numpy.shape[0], -1)
        # additional_features_train_numpy = costum_data_scaler.fit_transform(additional_features_train_numpy)
        # additional_features_train = self.normalize_train(additional_features_train.to_numpy())
        X_train = self.reshape_data(X_train[:,:-2])
        X_train, y_train, num_classes = self.robust_train_data(X_train, y_train)
        fit_label_encoder = self.label_encoder
        fit_scaler = self.scaler
        del self.label_encoder
        del self.scaler
        # print(additional_features_train)


        return X_train, y_train, additional_features_train_numpy, valid_classes, num_classes, fit_label_encoder, fit_scaler

# Test data processor subclass
class TestProcessor(BaseProcessor):

    def __init__(self, filepath_test,valid_classes,num_classes, num_temporal_steps, shape_dim, label_encoder, scaler, dates=None):
        super().__init__(num_temporal_steps, shape_dim, label_encoder, scaler, dates)
        self.filepath_test = filepath_test
        self.valid_classes = valid_classes
        self.num_classes = num_classes
        
    
    # def load_test_data(self):
    #     # Load and preprocess the test data
    #     data_test = pd.read_csv(self.filepath_test)
    #     data_test = data_test[data_test['class'] != 'p']
    #     data_test = data_test.reindex(columns=natsort.humansorted(data_test.columns)).fillna(-1)
    #     data_test_filtered = data_test[data_test['class'].isin(self.valid_classes)]

    #     ndvi_columns = [col for col in data_test_filtered.columns if 'ndvi' in col.lower()]
    #     ndvi_df = pd.DataFrame()
    #     ndvi_df[ndvi_columns] = data_test_filtered[ndvi_columns]
    #     ndvi_df['max_ndvi'] = ndvi_df.max(axis=1)
    #     ndvi_df['min_ndvi'] = ndvi_df.min(axis=1)
    #     ndvi_df[['SA', 'SF', 'SP']] = data_test_filtered[['SA', 'SF', 'SP']]
    #     additional_features_test = ndvi_df[['max_ndvi','min_ndvi','SA', 'SF', 'SP']]

    #     # Extract features and target from the filtered data
    #     self.X_test = data_test_filtered.iloc[:, 0:-11].values
    #     self.y_test = data_test_filtered['class'].values
    #     X_cord = data_test_filtered['X'].values
    #     Y_cord = data_test_filtered['Y'].values

    #     return self.X_test, self.y_test, X_cord, Y_cord, additional_features_test


    
    def load_test_data(self):
        # Load and preprocess the test data
        data_test = pd.read_csv(self.filepath_test)
        data_test = data_test[data_test['class'] != 'p']
        data_test = data_test.reindex(columns=natsort.humansorted(data_test.columns)).fillna(-1)
        data_test_filtered = data_test[data_test['class'].isin(self.valid_classes)]

        #### extractiong cfi index for canola #############
        ndvi_test_df = self.extract_data_columns(data_test_filtered, 'ndvi')
        green_test_df = self.extract_data_columns(data_test_filtered, 'b4')
        red_test_df = self.extract_data_columns(data_test_filtered, 'b3')
        blue_test_df = self.extract_data_columns(data_test_filtered, 'b2')
        cfi_test_index = ndvi_test_df.values * ((red_test_df.values + green_test_df.values) + (green_test_df.values - blue_test_df.values))
        cfi_column_names = [f"{i}_CFI" for i in range(55)]
        cfi_test_df = pd.DataFrame(columns=cfi_column_names)
        cfi_test_df[cfi_column_names] = cfi_test_index


        SWIR1 = self.extract_data_columns(data_test_filtered, 'b11')
        SWIR2 = self.extract_data_columns(data_test_filtered, 'b12')
        nbr_test_index = (SWIR1.values -  SWIR2.values) / (SWIR1.values +  SWIR2.values)
        nbr_column_names = [f"{i}_nbr2" for i in range(55)]
        nbr_test_df = pd.DataFrame(columns=nbr_column_names)
        nbr_test_df[nbr_column_names] = nbr_test_index

        b8a = self.extract_data_columns(data_test_filtered, 'b8a')
        b11 = self.extract_data_columns(data_test_filtered, 'b11')
        ndwi_test_index = (b8a.values -  b11.values) / (b8a.values +  b11.values)
        ndwi_column_names = [f"{i}_ndwi2" for i in range(55)]
        ndwi_test_df = pd.DataFrame(columns=ndwi_column_names)
        ndwi_test_df[ndwi_column_names] = ndwi_test_index

        lai_index_test = 2.5 * ndvi_test_df.values + 0.1
        lai_column_names = [f"{i}_LAI" for i in range(55)]
        lai_test_df = pd.DataFrame(columns=lai_column_names)
        lai_test_df[lai_column_names] = lai_index_test


        ndvi_test_df['max_ndvi'] = ndvi_test_df.max(axis=1)
        ndvi_test_df['min_ndvi'] = ndvi_test_df.min(axis=1)

        additional_features_test = ndvi_test_df[['max_ndvi','min_ndvi']].reset_index(drop=True)

        # Extract features and target from the filtered data
        X_test = data_test_filtered.iloc[:, 0:-11].reset_index(drop=True)
        
        # X_test_ndvi = self.extract_data_columns(data_test_filtered, 'ndvi').reset_index(drop=True)
        # X_test_vv = self.extract_data_columns(data_test_filtered, 'vv').reset_index(drop=True)
        # X_test_vh = self.extract_data_columns(data_test_filtered, 'vh').reset_index(drop=True)
        # X_test_savi = self.extract_data_columns(data_test_filtered, 'savi').reset_index(drop=True)
        # X_test_bsi = self.extract_data_columns(data_test_filtered, 'bsi').reset_index(drop=True)
        # X_test_slope = self.extract_data_columns(data_test_filtered, 'slope').reset_index(drop=True)
        # X_test_evi = self.extract_data_columns(data_test_filtered, 'evi').reset_index(drop=True)
        # X_test_nir = self.extract_data_columns(data_test_filtered, 'b8').reset_index(drop=True)
        
        X_test = pd.concat([X_test, additional_features_test],axis=1)
        X_test = X_test.reindex(columns=natsort.humansorted(X_test.columns)).values
        y_test = data_test_filtered['class'].values
        X_cord = data_test_filtered['X'].values
        Y_cord = data_test_filtered['Y'].values

        return X_test, y_test, X_cord, Y_cord, additional_features_test
    
    def normalize_test(self,x):
        x = x.reshape(x.shape[0], -1)
        X = self.scaler.transform(x)
        return X
    
    def encode_test(self, y):
        # Transform class labels from strings to integers
        y = self.label_encoder.transform(y)
        return y

    def test_data_early_classification(self,X_test,y_test):
        if self.dates is not None:
            X_test[:, self.dates:, :] = 0

        y_test = tf.keras.utils.to_categorical(y_test, self.num_classes)
        return X_test, y_test

    def test_processor(self):
        X_test, y_test, X_cord, Y_cord, additional_features_test = self.load_test_data()
        y_test = self.encode_test(y_test)
        X_test = self.normalize_test(X_test)


        # additional_features_test_numpy = additional_features_test.to_numpy()
        # additional_features_test_numpy = additional_features_test_numpy.reshape(additional_features_test_numpy.shape[0], -1)
        # additional_features_test_numpy = self.costum_data_scaler.transform(additional_features_test_numpy)
        additional_features_test_numpy = X_test[:,-2:]

        X_test = self.reshape_data(X_test[:,:-2])
        X_test, y_test = self.test_data_early_classification(X_test,y_test)

        del self.label_encoder
        del self.scaler

        return X_test, y_test, X_cord, Y_cord, additional_features_test_numpy
