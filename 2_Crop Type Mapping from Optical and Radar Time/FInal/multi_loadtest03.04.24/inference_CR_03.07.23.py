
import torch
import torch.utils.data as data
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import json
import os
import pickle as pkl
import argparse
import pprint
from tqdm import tqdm
import seaborn as sns
from learning.focal_loss import FocalLoss
from learning.weight_init import weight_init
from learning.metrics import mIou, confusion_matrix_analysis


from models.stclassifier_fusion import PseTae_pretrained
from dataset_fusion import PixelSetData

# %%
torch.__version__


# %%
def prepare_model_and_loader(args):
    mean_std1 = pkl.load(open(args['dataset_folder_meanstd1'] + '/S1-meanstd.pkl', 'rb'))
    mean_std2 = pkl.load(open(args['dataset_folder_meanstd2'] + '/S2-meanstd.pkl', 'rb'))
    extra = 'geomfeat' if args['geomfeat'] else None
    dt = PixelSetData(args['dataset_folder'],pms_extra=args['past_res'], labels=args['label_class'], npixel=args['npixel'],
                          sub_classes = args['sub_class'],
                          norm_s1=mean_std1,
                          norm_s2=mean_std2,
                          minimum_sampling=args['minimum_sampling'],
                          fusion_type = args['fusion_type'], interpolate_method = args['interpolate_method'],
                          extra_feature='geomfeat' if args['geomfeat'] else None,  
                          jitter=None,return_id=True)

    dl =data.DataLoader(dt, batch_size=args['batch_size'], num_workers=args['num_workers']) 


    model_config = dict(input_dim_s1=args['input_dim_s1'], input_dim_s2=args['input_dim_s2'], mlp1=args['mlp1'], pooling=args['pooling'],
                            mlp2=args['mlp2'], n_head=args['n_head'], d_k=args['d_k'], mlp3=args['mlp3'],
                            dropout=args['dropout'], T=args['T'], len_max_seq=args['lms'],
                            positions=None, fusion_type = args['fusion_type'],
                            mlp4=args['mlp4'],hidden_dim= args['hidden_dim'], kernel_size=args['kernel_size'], input_neuron = args['mlp2'][1], output_dim=args['mlp4'][0])

    if args['geomfeat']:
        model_config.update(with_extra=True, extra_size=7)
    else:
        model_config.update(with_extra=False, extra_size=None)

    model = PseTae_pretrained(args['weight_dir'], model_config, device=args['device'])

    return model, dl


def recursive_todevice(x, device):
    if isinstance(x, torch.Tensor):
        return x.to(device)
    else:
        return [recursive_todevice(c, device) for c in x]


def predict(model, loader, args):
    record = []
    device = torch.device(args['device'])
    

    for (x,x2, y,dates, ids) in tqdm(loader):
        y_true = (list(map(int, y)))
        ids = list(ids)

        x = recursive_todevice(x, device)
        x2 = recursive_todevice(x2, device)
        with torch.no_grad():
            prediction = model(x,x2,dates)
        y_p = list(prediction.argmax(dim=1).cpu().numpy())

        record.append(np.stack([ids, y_true, y_p], axis=1))

    record = np.concatenate(record, axis=0)

    os.makedirs(args['output_dir'], exist_ok=True)
    np.save(os.path.join(args['output_dir'], 'Predictions_id_ytrue_y_pred.npy'), record)


def test_evaluation(args):

    data =np.load(os.path.join(args['output_dir'], 'Predictions_id_ytrue_y_pred.npy'))
    data_num=data.shape[0]

    y_true=[]
    y_pred=[]
    for i in range(0, data_num):
        y_true1 = data[i][1]
        y_pred1 = data[i][2]
        y_true.append(y_true1)
        y_pred.append(y_pred1)
    
    
    y_true = [int(x) for x in y_true]
    y_pred = [int(x) for x in y_pred]
    y_true = [x if x in args['main_classes'] else args['others_classes'] for x in y_true]
    y_pred = [x if x in args['main_classes'] else args['others_classes'] for x in y_pred]
    
    return confusion_matrix(y_true, y_pred, labels=list(range(args['num_classes']))), y_true, y_pred
    

def save_results(conf_mat, args, y_true, y_pred):

    pkl.dump(conf_mat, open(os.path.join(args['output_dir'], 'conf_mat.pkl'), 'wb'))


    # save y_true, y_pred
    pkl.dump(y_true, open(os.path.join(args['output_dir'], 'y_true_test_data.pkl'), 'wb'))
    pkl.dump(y_pred, open(os.path.join(args['output_dir'], 'y_pred_test_data.pkl'), 'wb'))
    

    # ----> save confusion matrix
    #just test classes
    true_labels =  args['x_labels_list_test']
    predicted_labels =  args['x_labels_list_test']
    plt.figure(figsize=(15,10))
    # eleminate Classes
    conf_mat = conf_mat[np.ix_(args['cm_test_classes'], args['cm_test_classes'])]
    img = sns.heatmap(conf_mat, annot = True, fmt='d',linewidths=0.5, cmap='Blues',xticklabels=predicted_labels, yticklabels=true_labels)
    img.tick_params(top=False, labeltop=False, bottom=True, labelbottom=True)
    img.set(ylabel="True Label", xlabel="Predicted Label")
    img.figure.savefig(os.path.join(args['output_dir'], 'conf_mat_picture.png'))
    img.get_figure().clf()
    ########
    mat1 = conf_mat
    col_totals = mat1.sum(axis=0)  # Sum of each column
    normalized_mat1 = mat1 / col_totals[np.newaxis, :]  # Normalize each column separately
    
    # Plotting
    plt.figure(figsize=(15, 10))
    img = sns.heatmap(normalized_mat1, annot=True, fmt='.2f', linewidths=0.5, cmap='Blues', cbar=True,xticklabels=predicted_labels, yticklabels=true_labels)
    img.tick_params(top=False, labeltop=False, bottom=True, labelbottom=True)
    img.set(ylabel="True Label", xlabel="Predicted Label")
    img.figure.savefig(os.path.join(args['output_dir'], 'conf_mat_picture_perclass.png'))
    img.get_figure().clf()



def overall_performance(args):
    cm = np.zeros((args['num_classes'], args['num_classes']))
    cm += pkl.load(open(os.path.join(args['output_dir'], 'conf_mat.pkl'), 'rb'))
    per_class, perf = confusion_matrix_analysis(cm)

    print('Overall performance:')
    print('Acc: {},  IoU: {}'.format(perf['Accuracy'], perf['MACRO_IoU']))

    with open(os.path.join(args['output_dir'], 'overall.json'), 'w') as file:
        file.write(json.dumps(perf, indent=4))
    with open(os.path.join(args['output_dir'], 'per_class.json'), 'w') as file:
        file.write(json.dumps(per_class, indent=4))
    
    
def main(args):
    print('Preparation . . . ')
    model, loader = prepare_model_and_loader(args)
    print('Inference . . .')
    predict(model, loader, args)
    print('Results stored in directory {}'.format(args['output_dir']))
    conf_mat, y_true, y_pred= test_evaluation(args)
    save_results(conf_mat, args, y_true, y_pred)
    overall_performance(args)


# %%
if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    # /home/mhbokaei/shakouri/test/Satellite/dataset_folder100


    # Set-up parameters
    parser.add_argument('--dataset_folder', default='/home/mhbokaei/shakouri/ALLDATA/block/iran/ghazvin_se1/run_alltime/test_folder/s1_data', type=str,
                        help='Path to the Test Folder.')
    parser.add_argument('--dataset_folder_meanstd1', default='/home/mhbokaei/shakouri/ALLDATA/block/iran/ghazvin_se1/run_alltime/test_folder/s1_data', type=str,
                        help='Path to mean-std1 in test folder.')
    parser.add_argument('--dataset_folder_meanstd2', default='/home/mhbokaei/shakouri/ALLDATA/block/iran/ghazvin_se1/run_alltime/test_folder/s2_data', type=str,
                        help='Path to mean-std2 in test folder.')                                      
    parser.add_argument('--weight_dir', default='/home/mhbokaei/shakouri/CropTypeMappinp/multi_sensor/results_pse_nadam', type=str,
                        help='Path to the folder containing the model weights')
    parser.add_argument('--output_dir', default='./output_pse_ghazvin_new',
                        help='Path to the folder where the predictions should be stored')
    parser.add_argument('--past_res', default='/home/mhbokaei/shakouri/CropTypeMappinp/multi_sensor/results_pse_nadam', help='Path to the folder where the past results stored')
                        
    # ---------------------------add sensor argument to test s1/s2
    parser.add_argument('--minimum_sampling', default=None, type=int,
                        help='minimum time series length to sample')      
    parser.add_argument('--fusion_type', default='pse', type=str,
                        help='level of multi-sensor fusion e.g. early, pse, tsa, softmax_avg, softmax_norm')
    parser.add_argument('--interpolate_method', default='nn', type=str,
                        help='type of interpolation for early and pse fusion. eg. "nn","linear"') 
                        
    parser.add_argument('--num_workers', default=8, type=int, help='Number of data loading workers')
    parser.add_argument('--rdm_seed', default=1, type=int, help='Random seed')
    
    parser.add_argument('--device', default='cuda', type=str,
                        help='Name of device to use for tensor computations (cuda/cpu)')
    parser.add_argument('--label_class', default='label_51class', type=str, help='it can be label_19class or label_44class')
    parser.add_argument('--sub_class', default=None, type=list, help='Identify the subclass of the class')
    
    parser.add_argument('--main_classes', default=[0,2,9,16,17,18], type=list, help='Main classes we want do not change')
    parser.add_argument('--others_classes', default=6, type=int, help='the class of others')
    parser.add_argument('--cm_test_classes', default=[0,2,6,9,16,17,18], type=list, help='Main classes we want to show in confusion matrix')
    parser.add_argument('--x_labels_list_test', default=["wi-bi-wr-br","po","others","a","c","to","sb"] , type=list, help='The name of classes for test confusion matrix')
    # Dataset parameters
    parser.add_argument('--batch_size', default=256, type=int, help='Batch size')
    parser.add_argument('--npixel', default=40, type=int, help='Number of pixels to sample from the input images')

    # Architecture Hyperparameters
    ## PSE
    parser.add_argument('--input_dim_s1', default=4, type=int, help='Number of channels of input images_s1')
    parser.add_argument('--input_dim_s2', default=17, type=int, help='Number of channels of input images_s2')
    parser.add_argument('--mlp1', default='[17,32,64]', type=str, help='Number of neurons in the layers of MLP1')
    parser.add_argument('--pooling', default='mean_std', type=str, help='Pixel-embeddings pooling strategy')
    parser.add_argument('--mlp2', default='[135,128]', type=str, help='Number of neurons in the layers of MLP2')
    parser.add_argument('--geomfeat', default=1, type=int,
                        help='If 1 the precomputed geometrical features (f) are used in the PSE.')

    ## TAE
    parser.add_argument('--n_head', default=4, type=int, help='Number of attention heads')
    parser.add_argument('--d_k', default=32, type=int, help='Dimension of the key and query vectors')
    parser.add_argument('--mlp3', default='[512,128,128]', type=str, help='Number of neurons in the layers of MLP3')
    parser.add_argument('--T', default=1000, type=int, help='Maximum period for the positional encoding')
    parser.add_argument('--positions', default='bespoke', type=str,
                        help='Positions to use for the positional encoding (bespoke / order)')
    parser.add_argument('--lms', default=55, type=int,
                        help='Maximum sequence length for positional encoding (only necessary if positions == order)')
    parser.add_argument('--dropout', default=0.2, type=float, help='Dropout probability')

    ##ConvLSTM
    parser.add_argument('--hidden_dim', default=32, type=int, help='number of filtter. it must be power of 2 and same or biger than 16')
    parser.add_argument('--kernel_size', default=3, type=int, help='Size of kernel')

    ## Classifier
    parser.add_argument('--num_classes', default=21, type=int, help='Number of classes')
    parser.add_argument('--mlp4', default='[256, 64, 32, 21]', type=str, help='Number of neurons in the layers of MLP4')

# %%
args= parser.parse_args(args=[])
args= vars(args)
for k, v in args.items():
        if 'mlp' in k:
            v = v.replace('[', '')
            v = v.replace(']', '')
            args[k] = list(map(int, v.split(',')))

pprint.pprint(args)
main(args)


