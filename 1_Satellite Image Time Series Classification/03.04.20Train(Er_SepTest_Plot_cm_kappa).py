# %%
import torch
import torch.utils.data as data
import torchnet as tnt
import numpy as np
from sklearn.model_selection import KFold
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import os
import json
import pickle as pkl
import argparse
import pprint
import seaborn as sns

# %%
torch.__version__


# %%
from models.stclassifier import PseTae
from dataset import PixelSetData, PixelSetData_preloaded
from learning.focal_loss import FocalLoss
from learning.weight_init import weight_init
from learning.metrics import mIou, confusion_matrix_analysis

# %%

def train_epoch(model, optimizer, criterion, data_loader, device, args):
    acc_meter = tnt.meter.ClassErrorMeter(accuracy=True)
    loss_meter = tnt.meter.AverageValueMeter()
    y_true = []
    y_pred = []

    for i, (x, y) in enumerate(data_loader):

        y_true.extend(list(map(int, y)))

        x = recursive_todevice(x, device)
        y = y.to(device)

        optimizer.zero_grad()
        out = model(x)
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

    return epoch_metrics


def evaluation(model, criterion, loader, device, args, mode='val'):
    y_true = []
    y_pred = []

    acc_meter = tnt.meter.ClassErrorMeter(accuracy=True)
    loss_meter = tnt.meter.AverageValueMeter()

    for (x, y) in loader:
        y_true.extend(list(map(int, y)))
        x = recursive_todevice(x, device)
        y = y.to(device)

        with torch.no_grad():
            prediction = model(x)
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
        return metrics, confusion_matrix(y_true, y_pred, labels=list(range(args['num_classes'])))


def get_loaders(dt, kfold, args):
    indices = list(range(len(dt)))
    np.random.shuffle(indices)

    kf = KFold(n_splits=kfold, shuffle=False)
    indices_seq = list(kf.split(list(range(len(dt)))))
    ntest = len(indices_seq[0][1])

    loader_seq = []
    for trainval, test_indices in indices_seq:
        trainval = [indices[i] for i in trainval]
        test_indices = [indices[i] for i in test_indices]

        validation_indices = trainval[-ntest:]
        train_indices = trainval[:-ntest]

        train_sampler = data.sampler.SubsetRandomSampler(train_indices)
        validation_sampler = data.sampler.SubsetRandomSampler(validation_indices)
        test_sampler = data.sampler.SubsetRandomSampler(test_indices)

        train_loader = data.DataLoader(dt, batch_size=args['batch_size'],
                                       sampler=train_sampler,
                                       num_workers=args['num_workers'])
        validation_loader = data.DataLoader(dt, batch_size=args['batch_size'],
                                            sampler=validation_sampler,
                                            num_workers=args['num_workers'])
        test_loader = data.DataLoader(dt, batch_size=args['batch_size'],
                                      sampler=test_sampler,
                                      num_workers=args['num_workers'])

        loader_seq.append((train_loader, validation_loader, test_loader))
    return loader_seq


def get_loaders_sepT(dt,dt_sepT, kfold, args):
    indices = list(range(len(dt)))
    indices_T = list(range(len(dt_sepT)))
    np.random.shuffle(indices)
    np.random.shuffle(indices_T)


    kf = KFold(n_splits=kfold, shuffle=False)
    indices_seq = list(kf.split(list(range(len(dt)))))
    indices_seq_T = list(range(len(dt_sepT)))

    def add_element(main_matrix, 
                    temp_matrix = indices_seq_T):
        main_matrix = list(main_matrix)
        main_matrix.append(temp_matrix)
        main_matrix = tuple(main_matrix)
        return main_matrix

    indices_seqq = tuple(map(add_element , indices_seq))

    loader_seq = []
    for train_indices, validation_indices, test_indices in indices_seqq:
        train_indices = [indices[i] for i in train_indices]
        validation_indices = [indices[i] for i in validation_indices]
        test_indices = [indices_T[i] for i in test_indices]
        
        train_sampler = data.sampler.SubsetRandomSampler(train_indices)
        validation_sampler = data.sampler.SubsetRandomSampler(validation_indices)
        test_sampler = data.sampler.SubsetRandomSampler(test_indices)

        train_loader = data.DataLoader(dt, batch_size=args['batch_size'],
                                       sampler=train_sampler,
                                       num_workers=args['num_workers'])
        validation_loader = data.DataLoader(dt, batch_size=args['batch_size'],
                                            sampler=validation_sampler,
                                            num_workers=args['num_workers'])
        test_loader = data.DataLoader(dt_sepT, batch_size=args['batch_size'],
                                      sampler=test_sampler,
                                      num_workers=args['num_workers'])

        loader_seq.append((train_loader, validation_loader, test_loader))
    return loader_seq

def recursive_todevice(x, device):
    if isinstance(x, torch.Tensor):
        return x.to(device)
    else:
        return [recursive_todevice(c, device) for c in x]


def prepare_output(args):
    os.makedirs(args['res_dir'], exist_ok=True)
    for fold in range(1, args['kfold'] + 1):
        os.makedirs(os.path.join(args['res_dir'], 'Fold_{}'.format(fold)), exist_ok=True)


def checkpoint(fold, log, args):
    with open(os.path.join(args['res_dir'], 'Fold_{}'.format(fold), 'trainlog.json'), 'w') as outfile:
        json.dump(log, outfile, indent=4)


def save_results(fold, metrics, conf_mat, args):
    with open(os.path.join(args['res_dir'], 'Fold_{}'.format(fold), 'test_metrics.json'), 'w') as outfile:
        json.dump(metrics, outfile, indent=4)
    pkl.dump(conf_mat, open(os.path.join(args['res_dir'], 'Fold_{}'.format(fold), 'conf_mat.pkl'), 'wb'))
    ########Confusion Matrix Display#######errrrrrrrr
    true_labels =  ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's']
    predicted_labels =  ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's']
    plt.figure(figsize=(15,10))
    img = sns.heatmap(conf_mat, annot = True, fmt='d',linewidths=0.5, cmap='OrRd',xticklabels=predicted_labels, yticklabels=true_labels)
    img.tick_params(top=True, labeltop=True, bottom=False, labelbottom=False)
    img.set(ylabel="True Label", xlabel="Predicted Label")
    img.figure.savefig(os.path.join(args['res_dir'], 'conf_mat_picture.png'))
    img.get_figure().clf()

    mat1 = conf_mat
    col_totals = mat1.sum(axis=0)  # Sum of each column
    normalized_mat1 = mat1 / col_totals[np.newaxis, :]  # Normalize each column separately
    plt.figure(figsize=(15, 10))
    img = sns.heatmap(normalized_mat1, annot=True, fmt='.2f', linewidths=0.5, cmap='OrRd', cbar=True,xticklabels=predicted_labels, yticklabels=true_labels)
    img.tick_params(top=True, labeltop=True, bottom=False, labelbottom=False)
    img.set(ylabel="True Label", xlabel="Predicted Label")
    img.figure.savefig(os.path.join(args['res_dir'], 'conf_mat_picture_perclass.png'))
    img.get_figure().clf()

########Number of classes#######er
def point_plot(data,
               D_list,  ##List of deleted classes
               path : str,
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
    plt.xlabel(x_lable)
    plt.ylabel(y_lable)
    plt.title(title)
    plt.grid(True)
    plt.savefig(path)
    print("data.keys():", data.keys())
    print("len(data.keys()):", len(data.keys()))
    #plt.show()
    plt.close()

def Data_distribution(args):

    data_folder = os.path.join(args['dataset_folder'] , 'DATA')
    l = [f for f in os.listdir(data_folder) if f.endswith('.npy')]
    pid = [int(f.split('.')[0]) for f in l]
    pid = list(np.sort(pid))

    with open(os.path.join(args['dataset_folder'], 'META', 'labels.json'), 'r') as file:
        data = json.load(file)
    Dic = data[args['label_class']]
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
    save_path_cn = os.path.join(args['res_dir'], "number_of_classes.png")
    point_plot(counter,args['Delet_label_class'], save_path_cn, "Classes", "Number", "Number of each class")

###########################################################################################

def overall_performance(args):
    cm = np.zeros((args['num_classes'], args['num_classes']))
    for fold in range(1, args['kfold'] + 1):
        cm += pkl.load(open(os.path.join(args['res_dir'], 'Fold_{}'.format(fold), 'conf_mat.pkl'), 'rb'))

    _, perf = confusion_matrix_analysis(cm)

    print('Overall performance:')
    print('Acc: {},  IoU: {}'.format(perf['Accuracy'], perf['MACRO_IoU']))

    with open(os.path.join(args['res_dir'], 'overall.json'), 'w') as file:
        file.write(json.dumps(perf, indent=4))


def main(args):
    np.random.seed(args['rdm_seed'])
    torch.manual_seed(args['rdm_seed'])
    prepare_output(args)
    mean_std = pkl.load(open(args['dataset_folder'] + '/S2-2017-T31TFM-meanstd.pkl', 'rb'))
    mean_std_sepT = pkl.load(open(args['dataset_folder_sepT'] + '/S2-2017-T31TFM-meanstd.pkl', 'rb'))

    # 19 - 44 classes None label_class sub_class

    extra = 'geomfeat' if args['geomfeat'] else None

    if args['preload']:
        dt = PixelSetData_preloaded(args['dataset_folder'], labels=args['label_class'], npixel=args['npixel'],
                          sub_classes=args['sub_class'],
                          norm=mean_std,
                          extra_feature=extra)
        dt_sepT = PixelSetData_preloaded(args['dataset_folder_sepT'], labels=args['label_class'], npixel=args['npixel'],
                          sub_classes= args['sub_class'],
                          norm=mean_std_sepT,
                          extra_feature=extra)        
    else:
        dt = PixelSetData(args['dataset_folder'], labels=args['label_class'], npixel=args['npixel'],
                          sub_classes= args['sub_class'],
                          norm=mean_std,
                          extra_feature=extra)
        dt_sepT = PixelSetData(args['dataset_folder_sepT'], labels=args['label_class'], npixel=args['npixel'],
                          sub_classes= args['sub_class'],
                          norm=mean_std_sepT,
                          extra_feature=extra)        
        
    device = torch.device(args['device'])
    print("len(dt)1: ",len(dt))

    if args['separate_test'] == 'yes' :
        loaders = get_loaders_sepT(dt, dt_sepT, args['kfold'], args)
    elif args['separate_test'] == 'no':
        loaders = get_loaders(dt, args['kfold'], args)

    Data_distribution(args)

    for fold, (train_loader, val_loader, test_loader) in enumerate(loaders):
        print("len(dt)0: ",len(dt))
        print("len(dt_sepT)0: ",len(dt_sepT))
        print('Starting Fold {}'.format(fold + 1))
        print('Train {}, Val {}, Test {}'.format(len(train_loader), len(val_loader), len(test_loader)))
        print("test_loader: ", test_loader)

        model_args= dict(input_dim=args['input_dim'], mlp1=args['mlp1'], pooling=args['pooling'],
                            mlp2=args['mlp2'], n_head=args['n_head'], d_k=args['d_k'], mlp3=args['mlp3'],
                            dropout=args['dropout'], T=args['T'], len_max_seq=args['lms'],
                            positions=dt.date_positions if args['positions'] == 'bespoke' else None,
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

        training_history = {'train_accuracy': [], 'train_loss': [], 'val_accuracy': [], 'val_loss': []} #Er

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
            checkpoint(fold + 1, trainlog, args)

            ###############Prepare Accuracy and Loss Data for Plot##er 
            training_history['train_accuracy'].append(train_metrics['train_accuracy'])           
            training_history['train_loss'].append(train_metrics['train_loss'])   
            training_history['val_accuracy'].append(val_metrics['val_accuracy'])          
            training_history['val_loss'].append(val_metrics['val_loss'])   

            if val_metrics['val_IoU'] >= best_mIoU:
                best_mIoU = val_metrics['val_IoU']
                torch.save({'epoch': epoch, 'state_dict': model.state_dict(),
                            'optimizer': optimizer.state_dict()},
                           os.path.join(args['res_dir'], 'Fold_{}'.format(fold + 1), 'model.pth.tar'))

        
        #############Plot the Accuracy and Loss###########er
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 7))  # Adjust figsize as needed
        ax1.plot(training_history["train_accuracy"], label="train_accuracy", lw = 3, linestyle ='-')
        ax1.plot(training_history["val_accuracy"], label="val_accuracy", lw = 3, linestyle ='--')
        ax1.set_ylabel('Accuracy (%)', fontsize=15)
        ax1.set_xlabel('Epoch', fontsize=15)
        ax1.legend()
        ax1.tick_params(axis='both', which='major', labelsize=15)
        ax2.plot(training_history["train_loss"], label="train_loss", lw = 3, linestyle ='-')
        ax2.plot(training_history["val_loss"], label="val_loss", lw = 3, linestyle ='--')
        ax2.set_ylabel('Loss ', fontsize=15)
        ax2.set_xlabel('Epoch', fontsize=15)
        ax2.legend()
        ax2.tick_params(axis='both', which='major', labelsize=15)
        plt.tight_layout()
        save_path = os.path.join(args['res_dir'],'Fold_{}'.format(fold+1), "evplot.png")
        plt.savefig(save_path)

               
        print('Testing best epoch . . .')
        model.load_state_dict(
            torch.load(os.path.join(args['res_dir'], 'Fold_{}'.format(fold + 1), 'model.pth.tar'))['state_dict'])
        model.eval()

        test_metrics, conf_mat = evaluation(model, criterion, test_loader, device=device, mode='test', args=args)

        print('Loss {:.4f},  Acc {:.2f},  IoU {:.4f}'.format(test_metrics['test_loss'], test_metrics['test_accuracy'],
                                                             test_metrics['test_IoU']))
        save_results(fold + 1, test_metrics, conf_mat, args)

    overall_performance(args)



# %%

if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    # Set-up parameters
    # /home/mhbokaei/shakouri/test/Satellite/Alldata/dataset_folder100
    # /home/mhbokaei/shakouri/test/Satellite/Alldata/dataset_folder10000
    # /home/mhbokaei/shakouri/test/Satellite/Alldata/dataset_folder50000
    # /home/mhbokaei/shakouri/test/Satellite/Alldata/dataset_folder90000
    # /home/mhbokaei/shakouri/SatelliteImage_TimeSeries_Classification/dataset_folder
    # /home/mhbokaei/shakouri/test/Satellite/Alldata/dataset_folder_train
   
    
    parser.add_argument('--dataset_folder', default='/home/mhbokaei/shakouri/test/Satellite/Alldata/dataset_folder10000', type=str,
                        help='Path to the folder where the results are saved.')
    parser.add_argument('--dataset_folder_sepT', default='/home/mhbokaei/shakouri/test/Satellite/Alldata/dataset_folder10000', type=str,
                        help='Path to the folder where the results are saved.')
    parser.add_argument('--separate_test', default='no', type=str, help='Shows the dataset for Test is different(no  yes)')

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
    parser.add_argument('--label_class', default='label_19class', type=str, help='it can be label_19class or label_44class')
    parser.add_argument('--sub_class', default=None, type=list, help='Identify the subclass of the class')
    parser.add_argument('--Delet_label_class', default=[], type=list, help='Plot the number of data per classes')


    # Training parameters
    parser.add_argument('--kfold', default=5, type=int, help='Number of folds for cross validation')
    parser.add_argument('--epochs', default=5, type=int, help='Number of epochs per fold')
    parser.add_argument('--batch_size', default=64, type=int, help='Batch size')
    parser.add_argument('--lr', default=0.001, type=float, help='Learning rate')
    parser.add_argument('--gamma', default=1, type=float, help='Gamma parameter of the focal loss')
    parser.add_argument('--npixel', default=64, type=int, help='Number of pixels to sample from the input images')

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
    parser.add_argument('--mlp4', default='[128, 64, 32, 19]', type=str, help='Number of neurons in the layers of MLP4')



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


