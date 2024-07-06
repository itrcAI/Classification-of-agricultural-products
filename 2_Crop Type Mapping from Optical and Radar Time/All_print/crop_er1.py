# %%
import torch
from torch import nn
import torch.utils.data as data
import numpy as np
import torchnet as tnt
# from sklearn.model_selection import KFold
from sklearn.metrics import confusion_matrix
import os
import json
import pickle as pkl
import argparse
import pprint
from datetime import datetime

from models.stclassifier_fusion import PseTae
from dataset_fusion import PixelSetData, PixelSetData_preloaded
from learning.focal_loss import FocalLoss
from learning.weight_init import weight_init
from learning.metrics import mIou, confusion_matrix_analysis

import seaborn as sns
import matplotlib.pyplot as plt



# %%
torch.__version__


# %%

def train_epoch(model, optimizer, criterion, data_loader, device, args):
    start = datetime.now()
    acc_meter = tnt.meter.ClassErrorMeter(accuracy=True)
    loss_meter = tnt.meter.AverageValueMeter()
    y_true = []
    y_pred = []

    for i, (x, x2, y, dates) in enumerate(data_loader): 
                
        y_true.extend(list(map(int, y)))

        x = recursive_todevice(x, device)
        x2 = recursive_todevice(x2, device) 
        y = y.to(device)

        optimizer.zero_grad()
        out = model(x, x2, dates)
        loss = criterion(out, y.long())
        loss.backward()
        optimizer.step()

        pred = out.detach()
        y_p = pred.argmax(dim=1).cpu().numpy()
        y_pred.extend(list(y_p))
        acc_meter.add(pred, y)
        loss_meter.add(loss.item())

        if (i + 1) % args['display_step'] == 0:
            print('Step [{}/{}], Loss: {:.4f}, Acc : {:.2f}'.format(i + 1, len(data_loader), loss_meter.value()[0],
                                                                    acc_meter.value()[0]))

    epoch_metrics = {'train_loss': loss_meter.value()[0],
                     'train_accuracy': acc_meter.value()[0],
                     'train_IoU': mIou(y_true, y_pred, n_classes=args['num_classes'])}
    print('train epoch complete in ----------------------->', datetime.now()-start)
    return epoch_metrics


def evaluation(model, criterion, loader, device, args, mode='val'):
    y_true = []
    y_pred = []

    acc_meter = tnt.meter.ClassErrorMeter(accuracy=True)
    loss_meter = tnt.meter.AverageValueMeter()

    for (x, x2, y, dates) in loader: 

        y_true.extend(list(map(int, y)))

        x = recursive_todevice(x, device)
        x2 = recursive_todevice(x2, device) #add x2 to device
        y = y.to(device)


        with torch.no_grad():
            prediction = model(x, x2, dates)  
            loss = criterion(prediction, y)

        acc_meter.add(prediction, y)
        loss_meter.add(loss.item())

        y_p = prediction.argmax(dim=1).cpu().numpy()
        y_pred.extend(list(y_p))

    metrics = {'{}_accuracy'.format(mode): acc_meter.value()[0],
               '{}_loss'.format(mode): loss_meter.value()[0],
               '{}_IoU'.format(mode): mIou(y_true, y_pred, args['num_classes'])}

    if mode == 'val':
        return metrics
    elif mode == 'test':
        return metrics, confusion_matrix(y_true, y_pred, labels=list(range(args['num_classes']))), y_true, y_pred 


def get_pse(folder, args):
    if args['preload']:
        dt = PixelSetData_preloaded(args[folder], labels='label_19class', npixel=args['npixel'],
                          sub_classes = None,
                          norm=None,
                          minimum_sampling=args['minimum_sampling'],
                          fusion_type = args['fusion_type'], interpolate_method = args['interpolate_method'],
                          extra_feature='geomfeat' if args['geomfeat'] else None,  
                          jitter=None)
    else:
        dt = PixelSetData(args[folder], labels='label_19class', npixel=args['npixel'],
                          sub_classes = None,
                          norm=None,
                          minimum_sampling=args['minimum_sampling'],
                          fusion_type = args['fusion_type'], interpolate_method = args['interpolate_method'],
                          extra_feature='geomfeat' if args['geomfeat'] else None,  
                          jitter=None)
    return dt

def get_loaders(args):
    loader_seq =[]
    train_dataset = get_pse('dataset_folder', args)
    val_dataset = get_pse('val_folder', args)
    test_dataset = get_pse('test_folder', args)

    if args['dataset_folder2'] is not None:
        train_dataset2 = get_pse('dataset_folder2', args)
        train_dataset = data.ConcatDataset([train_dataset, train_dataset2])

        
    train_loader = data.DataLoader(train_dataset, batch_size=args['batch_size'],
                                        num_workers=args['num_workers'], shuffle = True, pin_memory =True) 

    validation_loader = data.DataLoader(val_dataset, batch_size=args['batch_size'],
                                        num_workers=args['num_workers'], shuffle = False, pin_memory = True)

    test_loader = data.DataLoader(test_dataset, batch_size=args['batch_size'],
                                    num_workers=args['num_workers'], shuffle = False, pin_memory =True)

    loader_seq.append((train_loader, validation_loader, test_loader))
    return loader_seq

def recursive_todevice(x, device):
    if isinstance(x, torch.Tensor):
        return x.to(device)
    else:
        return [recursive_todevice(c, device) for c in x]


def prepare_output(args):
    os.makedirs(args['res_dir'], exist_ok=True)


def checkpoint(log, args):
    with open(os.path.join(args['res_dir'], 'trainlog.json'), 'w') as outfile:
        json.dump(log, outfile, indent=4)


def save_results(metrics, conf_mat, args, y_true, y_pred):
    with open(os.path.join(args['res_dir'], 'test_metrics.json'), 'w') as outfile:
        json.dump(metrics, outfile, indent=4)
    pkl.dump(conf_mat, open(os.path.join(args['res_dir'], 'conf_mat.pkl'), 'wb'))


    # save y_true, y_pred
    pkl.dump(y_true, open(os.path.join(args['res_dir'], 'y_true_test_data.pkl'), 'wb'))
    pkl.dump(y_pred, open(os.path.join(args['res_dir'], 'y_pred_test_data.pkl'), 'wb'))

    # ----> save confusion matrix 
    plt.figure(figsize=(15,10))
    img = sns.heatmap(conf_mat, annot = True, fmt='d',linewidths=0.5, cmap='OrRd')
    img.tick_params(top=True, labeltop=True, bottom=False, labelbottom=False)
    img.set(ylabel="True Label", xlabel="Predicted Label")
    img.figure.savefig(os.path.join(args['res_dir'], 'conf_mat_picture.png'))
    img.get_figure().clf()


def overall_performance(args):
    cm = np.zeros((args['num_classes'], args['num_classes']))
    cm += pkl.load(open(os.path.join(args['res_dir'], 'conf_mat.pkl'), 'rb'))
    per_class, perf = confusion_matrix_analysis(cm)

    print('Overall performance:')
    print('Acc: {},  IoU: {}'.format(perf['Accuracy'], perf['MACRO_IoU']))

    with open(os.path.join(args['res_dir'], 'overall.json'), 'w') as file:
        file.write(json.dumps(perf, indent=4))
    with open(os.path.join(args['res_dir'], 'per_class.json'), 'w') as file:
        file.write(json.dumps(per_class, indent=4))


def plot_metrics(args):
    with open(os.path.join(args['res_dir'], 'trainlog.json'), 'r') as file:
        d = json.loads(file.read())
    
    epoch = [i+1 for i in range(len(d))]
    train_loss = [d[str(i+1)]["train_loss"] for i in range(len(d))]
    val_loss = [d[str(i+1)]["val_loss"] for i in range(len(d))]
    train_acc = [d[str(i+1)]["train_accuracy"] for i in range(len(d))]
    val_acc = [d[str(i+1)]["val_accuracy"] for i in range(len(d))]

    # plot loss/accuracy
    plt.title('monitoring metrics - loss')
    plt.plot(epoch, train_loss, label = "train_loss")
    plt.plot(epoch,val_loss, label = "val_loss")
    plt.xlabel('epoch')
    plt.legend()
    plt.savefig(os.path.join(args['res_dir'], 'loss_graph.png'))
    plt.clf()

    plt.title('monitoring metrics - accuracy')
    plt.plot(epoch, train_acc, label = "train_accuracy")
    plt.plot(epoch,val_acc, label = "val_accuracy")
    plt.xlabel('epoch')
    plt.legend()
    plt.savefig(os.path.join(args['res_dir'], 'accuracy_graph.png'))
    plt.clf()
    #---------------------------------------------

def main(args):
    np.random.seed(args['rdm_seed'])
    torch.manual_seed(args['rdm_seed'])
    prepare_output(args)

    extra = 'geomfeat' if args['geomfeat'] else None

    device = torch.device(args['device'])

    loaders = get_loaders(args)
    for _, (train_loader, val_loader, test_loader) in enumerate(loaders):
        print('Train {}, Val {}, Test {}'.format(len(train_loader), len(val_loader), len(test_loader)))

        model_args = dict(input_dim=args['input_dim'], mlp1=args['mlp1'], pooling=args['pooling'],
                            mlp2=args['mlp2'], n_head=args['n_head'], d_k=args['d_k'], mlp3=args['mlp3'],
                            dropout=args['dropout'], T=args['T'], len_max_seq=args['lms'],
                            positions=None, fusion_type = args['fusion_type'],
                            mlp4=args['mlp4'])

        if args['geomfeat']:
            model_args.update(with_extra=True, extra_size=4) 
        else:
            model_args.update(with_extra=False, extra_size=None)

        model = PseTae(**model_args)

        print(model.param_ratio())


        model = model.to(device)
        model.apply(weight_init)
        optimizer = torch.optim.Adam(model.parameters())
        criterion = FocalLoss(args['gamma'])

        trainlog = {}


        best_mIoU = 0
        for epoch in range(1, args['epochs'] + 1):
            print('EPOCH {}/{}'.format(epoch, args['epochs']))

            model.train()
            train_metrics = train_epoch(model, optimizer, criterion, train_loader, device=device, args=args)

            print('Validation . . . ')
            model.eval()
            val_metrics = evaluation(model, criterion, val_loader, device=device, args=args, mode='val')

            print('Loss {:.4f},  Acc {:.2f},  IoU {:.4f}'.format(val_metrics['val_loss'], val_metrics['val_accuracy'],
                                                                 val_metrics['val_IoU']))

            trainlog[epoch] = {**train_metrics, **val_metrics}
            checkpoint(trainlog, args)

            if val_metrics['val_IoU'] >= best_mIoU:
                best_mIoU = val_metrics['val_IoU']
                torch.save({'epoch': epoch, 'state_dict': model.state_dict(),
                            'optimizer': optimizer.state_dict()},
                           os.path.join(args['res_dir'], 'model.pth.tar'))

        print('Testing best epoch . . .')
        model.load_state_dict(
            torch.load(os.path.join(args['res_dir'],  'model.pth.tar'))['state_dict'])
        model.eval()

        test_metrics, conf_mat, y_true, y_pred = evaluation(model, criterion, test_loader, device=device, mode='test', args=args) 

        print('Loss {:.4f},  Acc {:.2f},  IoU {:.4f}'.format(test_metrics['test_loss'], test_metrics['test_accuracy'],
                                                             test_metrics['test_IoU']))
                                                             
        save_results(test_metrics, conf_mat, args, y_true, y_pred) 

    overall_performance(args)
    plot_metrics(args)



# %%

if __name__ == '__main__':
    start = datetime.now()

    parser = argparse.ArgumentParser()

    #/home/mhbokaei/shakouri/CropTypeMappinp/multi_sensor/All_dataset/dataset_100/dataset_folder/s1_data
    #/home/mhbokaei/shakouri/CropTypeMappinp/multi_sensor/All_dataset/dataset_100/test_folder/s1_data
    #/home/mhbokaei/shakouri/CropTypeMappinp/multi_sensor/All_dataset/dataset_100/val_folder/s1_data
    


    # Set-up parameters
    parser.add_argument('--dataset_folder', default='/home/mhbokaei/shakouri/CropTypeMappinp/multi_sensor/All_dataset/dataset_100/dataset_folder/s1_data', type=str,
                        help='Path to the folder where the results are saved.')

    # set-up data loader folders -----------------------------
    parser.add_argument('--dataset_folder2', default=None, type=str,
                        help='Path to second train folder to concat with first initial loader.')
    parser.add_argument('--val_folder', default='/home/mhbokaei/shakouri/CropTypeMappinp/multi_sensor/All_dataset/dataset_100/val_folder/s1_data', type=str,
                        help='Path to the validation folder.')
    parser.add_argument('--test_folder', default='/home/mhbokaei/shakouri/CropTypeMappinp/multi_sensor/All_dataset/dataset_100/test_folder/s1_data', type=str,
                        help='Path to the test folder.')

    # ---------------------------add sensor argument to test s1/s2
    parser.add_argument('--minimum_sampling', default=None, type=int,
                        help='minimum time series length to sample')      
    parser.add_argument('--fusion_type', default='tsa', type=str,
                        help='level of multi-sensor fusion e.g. early, pse, tsa, softmax_avg, softmax_norm')
    parser.add_argument('--interpolate_method', default='nn', type=str,
                        help='type of interpolation for early and pse fusion. eg. "nn","linear"')    
    
    parser.add_argument('--res_dir', default='./results', help='Path to the folder where the results should be stored')
    parser.add_argument('--num_workers', default=8, type=int, help='Number of data loading workers')
    parser.add_argument('--rdm_seed', default=1, type=int, help='Random seed')
    parser.add_argument('--device', default='cuda', type=str,
                        help='Name of device to use for tensor computations (cuda/cpu)')
    parser.add_argument('--display_step', default=50, type=int,
                        help='Interval in batches between display of training metrics')
    parser.add_argument('--preload', dest='preload', action='store_true',
                        help='If specified, the whole dataset is loaded to RAM at initialization')
    parser.set_defaults(preload=False)

    # Training parameters
    parser.add_argument('--epochs', default=3, type=int, help='Number of epochs per fold')
    parser.add_argument('--batch_size', default=32, type=int, help='Batch size')
    parser.add_argument('--lr', default=0.001, type=float, help='Learning rate')
    parser.add_argument('--gamma', default=1, type=float, help='Gamma parameter of the focal loss')
    parser.add_argument('--npixel', default=5, type=int, help='Number of pixels to sample from the input images')

    # Architecture Hyperparameters
    ## PSE
    parser.add_argument('--input_dim', default=10, type=int, help='Number of channels of input images')
    parser.add_argument('--mlp1', default='[10,32,64]', type=str, help='Number of neurons in the layers of MLP1')
    parser.add_argument('--pooling', default='mean_std', type=str, help='Pixel-embeddings pooling strategy')
    parser.add_argument('--mlp2', default='[132,128]', type=str, help='Number of neurons in the layers of MLP2')
    parser.add_argument('--geomfeat', default=1, type=int,
                        help='If 1 the precomputed geometrical features (f) are used in the PSE.')

    ## TAE
    parser.add_argument('--n_head', default=4, type=int, help='Number of attention heads')
    parser.add_argument('--d_k', default=32, type=int, help='Dimension of the key and query vectors')
    parser.add_argument('--mlp3', default='[512,128,128]', type=str, help='Number of neurons in the layers of MLP3')
    parser.add_argument('--T', default=1000, type=int, help='Maximum period for the positional encoding')
    parser.add_argument('--positions', default='bespoke', type=str,
                        help='Positions to use for the positional encoding (bespoke / order)')
    parser.add_argument('--lms', default=None, type=int,
                        help='Maximum sequence length for positional encoding (only necessary if positions == order)')
    parser.add_argument('--dropout', default=0.2, type=float, help='Dropout probability')

    ## Classifier
    parser.add_argument('--num_classes', default=19, type=int, help='Number of classes')
    parser.add_argument('--mlp4', default='[256, 64, 32, 19]', type=str, help='Number of neurons in the layers of MLP4- pse and tae nedd 256 except 128')

    args= parser.parse_args(args=[])
    args= vars(args)
    for k, v in args.items():
            if 'mlp' in k:
                v = v.replace('[', '')
                v = v.replace(']', '')
                args[k] = list(map(int, v.split(',')))

    pprint.pprint(args)
    main(args)


    #add processing time
    print('total elapsed time is --->', datetime.now() -start)



