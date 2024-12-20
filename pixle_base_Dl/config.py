filepath = '/root/dl/bahar'
direction = f'/root/dl/result_bahar'
train_df_file_name = 'train.csv'

keep_labels = ['wi-bi-wr-br', 'po','to', 'a', 'sb', 'c'] 
num_temporal_steps = 55

pexel_base_1d_models_name = [ 'lstm',
                            'bilstm',
                            'tempcnn1d',
                            'cnn_lstm',
                            'lstm_cnn',
                             'autoencoder_lstm',
                             'autoencoder_cnn',
                             'unet_lstm'
                                        ]
pexel_base_2d_models_name = [ 'conv_lstm1d',
                            'cbam_resnet',
                            'tempcnn2d',
                                        ]

model_names = pexel_base_1d_models_name

shape_dim = 3
epochs = 20








