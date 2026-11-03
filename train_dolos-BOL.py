import argparse
import os
import numpy as np
import time
import random
import csv

import torch
import torch.nn as nn
#from dataloader_v2 import DataLoader
from dataloader_mix import DataLoader
import torch.utils.data as data
# from torch.optim.lr_scheduler import ReduceLROnPlateau
from timesformer.models.vit7 import TimeSformer
from sklearn.metrics import classification_report, accuracy_score, f1_score, roc_curve, auc
import timesformer.models.optimizer as optim
torch.multiprocessing.set_sharing_strategy('file_system')
#nn.CrossEntropyLoss()

#os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID" # 按照PCI_BUS_ID顺序从0开始排列GPU设备
#os.environ["CUDA_VISIBLE_DEVICES"] = "1"
torch.backends.cudnn.enabled = True
def parse_options():
    parser = argparse.ArgumentParser(description="WILTY baseline repo")
    ##### TRAINING DYNAMICS

    parser.add_argument('--device', type=str, default="cuda", help='the gpu id used for predict')
    parser.add_argument('--device_ID', type=str, default="cuda:1", help='the gpu id used for predict')
    parser.add_argument('--lr', type=float, default=1e-4, help='initial learning rate')  # default=0.001
    parser.add_argument('--when', type=int, default=15,
                        help='when to decay learning rate (default: 10)')
    parser.add_argument('--batch_size', type=int, default=2, help='initial batchsize')
    parser.add_argument('--num_epochs', type=int, default=51, help='total training epochs')
    parser.add_argument('--seed', type=int, default=1111,
                        help='random seed')

    parser.add_argument('--len', type=int, default=64, help='lenght of frames')
    parser.add_argument('--size', type=int, default=224, help='block of image size')

    ##### UNIMODAL ARCHITECTURE

    #parser.add_argument('--num_encoders', type=int, default=4, help="number of transformer encoders for each modality")
    #parser.add_argument('--adapter', action='store_true', help="indicator of using adapter")
    #parser.add_argument('--adapter_type', type=str, default='efficient_conv',
    #                    help='indicate which type of adapter to use - full_conv or efficient_conv or nlp')
    ##### DATA

    #parser.add_argument('--data_root', type=str, default='F:\\ADD\\dateset\\DOLOS\\')
    #parser.add_argument('--audio_path', type=str, default='F:\\ADD\\dateset\\DOLOS\\audio_files\\',
     #                   help='audio dataset root dir')
    #parser.add_argument('--visual_path', type=str, default='F:\\ADD\\dateset\\DOLOS\\face_frames\\',
     #                   help='visual (FACE) dataset root dir')

    #### TRAINING LOGS

    #parser.add_argument('--log', type=str, default="logs", help='log and save model name')
   # parser.add_argument('--protocols', type=list,default=[['train_fold3.csv', 'test_fold3.csv']],help='protocols for train/test')

    """
    parser.add_argument('--protocols', type=list,
                        default=[['train_file_1.csv', 'test_file_1.csv'], ['train_file_2.csv', 'test_file_2.csv'],
                                 ['train_file_3.csv', 'test_file_3.csv'], ['train_file_4.csv', 'test_file_4.csv'],
                                 ['train_file_5.csv', 'test_file_5.csv']],
                        help='protocols for train/test')                                 
    ['train_fold1.csv', 'test_fold1.csv'], ['train_fold2.csv', 'test_fold2.csv'],
                                 ['train_fold3.csv', 'test_fold3.csv'], ['long.csv', 'short.csv'],
                                 ['short.csv', 'long.csv'],
                                 ['male.csv', 'female.csv'], ['female.csv', 'male.csv']

    ['WILTY.csv', 'bag_of_lies.csv'], ['WILTY.csv', 'RLT_annotations.csv'], 
    ['bol_audio_visual.csv', 'bag_of_lies.csv'], ['bol_audio_visual.csv', 'RLT_annotations.csv']
    """
    #parser.add_argument('--full_annotation', type=str,
    #                    default='/DOLOS/data/protocols/WILTY_binary.csv')

    #parser.add_argument('--model_name', type=str, default='DOLOS_')
   # parser.add_argument('--model_to_train', type=str, default='fusion',
    #                    help='specify vision or audio or fusion- indiciates the model to be trained')
    #parser.add_argument('--fusion_type', type=str, default='cross2', help='modality fusion type')

    opts = parser.parse_args()
    torch.manual_seed(opts.seed)
    opts.device = torch.device(opts.device_ID)
    # if "fusion" in opts.model_to_train:
    #     opts.model_to_train = opts.model_to_train + "_" + opts.fusion_type
    return opts

def evaluation(labels, preds):
    acc = accuracy_score(labels, preds)
    f1 = f1_score(labels, preds)
    fpr, tpr, thresholds = roc_curve(labels, preds, pos_label=1)
    auc_score = auc(fpr, tpr)
    return acc, f1, auc_score

def train_one_epoch(args, train_data_loader, model,optimizer,loss_fn):
    ### Local Parameters
    epoch_loss = []
    epoch_predictions = []
    epoch_labels = []
    start_time = time.time()
    # model.cuda()
    model.train()
    tn = 0
    ln = 0
    # data_size = len(train_data_loader)
    for cur_iter, (img, AUs, label) in enumerate(train_data_loader):
        # for i in range(args.batch_size):

        # Loading waves and labels to device
        img = img.to(args.device)
        AUs = AUs.to(args.device)
        label = label.to(args.device)

        #lr = optim.get_epoch_lr(cur_epoch + float(cur_iter) / data_size, cfg)
        #optim.set_lr(optimizer, lr)

        # Reseting Gradients
        optimizer.zero_grad()

        # Forward
        preds = model((img,AUs))
        #preds = torch.sigmoid(preds)
        
        # pred = F.softmax(pred, dim=1)
        '''
        if cur_iter %100==0:
          tn = 0
          ln = 0
        if (label[1] == 0) & (tn==0):
            print('127',preds,label)
            tn += 1
        if (label[1] == 1) & (ln == 0):
            print('130',preds,label)
            ln += 1
        '''

        #print(preds.shape)
        #print(label.shape)
        # Calculating Loss
        _loss = loss_fn(preds, label)
        loss = _loss.item()
        epoch_loss.append(loss)
        # Backward
        _loss.backward()
        optimizer.step()
        epoch_labels.append(label)
        epoch_predictions.append(torch.argmax(preds, dim=1))

        if cur_iter % 500 == 0:
            print("iter {}, loss {:.20f}".format(str(cur_iter), loss))

    # CALCULATE ACCURACY AND CONFUSION MATRIX
    #epoch_predictions = torch.cat(epoch_predictions)
    #epoch_labels = torch.cat(epoch_labels)

    epoch_predictions = torch.cat(epoch_predictions)
    epoch_labels = torch.cat(epoch_labels)
    epoch_loss = np.mean(epoch_loss)

    # Overall Epoch Results
    end_time = time.time()
    total_time = end_time - start_time

    # Acc and Loss
    #    epoch_loss = .mean(epoch_loss)
    print("total_time:", total_time)
    # return epoch_loss, acc, cfm, f1
    return epoch_loss, epoch_predictions, epoch_labels


"""def evaluation(labels, preds):
    acc = accuracy_score(labels, preds)
    f1 = f1_score(labels, preds)
    fpr, tpr, thresholds = roc_curve(labels, preds, pos_label=1)
    auc_score = auc(fpr, tpr)
    return acc, f1, auc_score"""

def val_one_epoch(args, val_data_loader, model, loss_fn):
    #  Local Parameters
    epoch_loss = []
    epoch_predictions = []
    epoch_labels = []
    start_time = time.time()

    model.eval()
    
    with torch.no_grad():
      for cur_iter, (img, AUs, labels) in enumerate(val_data_loader):
        # for i in range(args.batch_size):

        # Loading waves and labels to device
        img = img.to(args.device)
        AUs = AUs.to(args.device)
        labels = labels.to(args.device)
        
        preds = model((img, AUs))
        preds = torch.sigmoid(preds)
        
        _loss = loss_fn(preds, labels)
        loss = _loss.item()
        epoch_loss.append(loss)
        epoch_labels.append(labels)
        epoch_predictions.append(torch.argmax(preds, dim=1))
    
    epoch_predictions = torch.cat(epoch_predictions)
    epoch_labels = torch.cat(epoch_labels)
    epoch_loss = np.mean(epoch_loss)
    end_time = time.time()
    total_time = end_time - start_time
    return epoch_loss, epoch_predictions, epoch_labels

def split_list(lst, split_ratio):
    random.shuffle(lst)
    split_point = int(len(lst) * split_ratio)
    train_set = lst[:split_point]
    test_set = lst[split_point:]
    return train_set, test_set

def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True

def read_file(file_path):
    column_values = []
    with open(file_path,'r') as file:
        reader = csv.reader(file)
        for row in reader:
            column_values.append(row[0])
    return column_values

def train_test(train_sets, test_sets, args):
    root_folder = "/data01/behavior_group/d21_qiaoyy/DOLOs/image/"
    AUs_root = '/data01/behavior_group/d21_qiaoyy/DOLOs/20AUs/'
    best_acc = 0.0
    
    random.shuffle(train_sets)
    empt_file = ['trial_lie_017', 'trial_truth_004', 'trial_truth_007', 'trial_truth_010']
    train_set = []
    test_set = []
    
    for d in train_sets:
        if d not in empt_file:
            train_set.append(d)
    print(len(train_set))
    test_sets = np.load('/data01/behavior_group/d21_qiaoyy/RL/test_RL.npy')
    for d in test_sets:
        if d not in empt_file:
            test_set.append(d)
    print(len(test_set))
    
    
    train_dataset = DataLoader(video_folder=root_folder, img_size=args.size, frame_len=args.len, video_names=train_set, dataset = 'DOLOs')
    train_batch = data.DataLoader(train_dataset, batch_size=args.batch_size,
                                  shuffle=False, num_workers=1, drop_last=True)

    root_folder = "/data01/behavior_group/d21_qiaoyy/RL/imgs/"
    AUs_root = '/data01/behavior_group/d21_qiaoyy/RL/AU/AUs/'
    test_dataset = DataLoader(root_folder, args.size, args.len, test_set, 'RL')
    test_batch = data.DataLoader(test_dataset, batch_size=args.batch_size,
                                  shuffle=False, num_workers=1, drop_last=True)

    model = TimeSformer(img_size=args.size, num_classes=2, num_frames=args.len, attention_type='divided_space_time')
    model.to(args.device)
    # data_size = len(train_batch)
    # Optimizer
    optimizer = torch.optim.Adam(model.parameters())
    # optimizer = torch.optim.SGD(lr=0.01,momentum=0.9,weight_decay=1e-4)
    # Loss Function
    loss_fn = nn.CrossEntropyLoss()
    print("\t Started Training")
    with open('Outputs/results.txt','w') as f:
        for epoch in range(args.num_epochs):
            cur_epoch = epoch
            f.write('Epoch....' + str(cur_epoch) + ':\n')
            print('\t\t Epoch....', epoch)
            # Training
            lr = args.lr * (0.1 ** (cur_epoch // 10))
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr
                
            loss, preds, lables  = train_one_epoch(args, train_batch, model, optimizer, loss_fn)
            train_acc, train_f1, train_auc = evaluation(lables.detach().cpu().numpy(), preds.detach().cpu().numpy())
            cur_results = "epoch {}, loss {:.5f}, train_acc {:.5f}, train_f1: {:.5f}, train_auc:{:.5f} ".format(epoch, loss, train_acc, train_f1, train_auc)
            f.write(cur_results + '\n')
            print("epoch {}, loss {:.5f}, train_acc {:.5f}, train_f1: {:.5f}, train_auc:{:.5f} ".format(epoch, loss, train_acc, train_f1, train_auc))
            
            if (cur_epoch % 5 == 0):
                torch.save(model, "checkpoints/" + str(cur_epoch) + ".pth")
            
                val_loss, val_preds, val_lables = val_one_epoch(args, test_batch, model, loss_fn)
                val_acc, val_f1, val_auc = evaluation(val_lables.detach().cpu().numpy(),
                                                      val_preds.detach().cpu().numpy())
                cur_results = "test_acc {:.5f}, test_f1: {:.5f}, test_auc:{:.5f}".format(val_acc, val_f1, val_auc)
                f.write(cur_results + '\n')
                print('\t\t Epoch....', epoch)
                print("test_acc {:.5f}, test_f1: {:.5f}, test_auc:{:.5f}".format(val_acc, val_f1, val_auc))
                if val_acc > best_acc:
                    best_acc = val_acc
                    results = "best results are acc {:.5f}, f1: {:.5f}, auc:{:.5f} ".format(val_acc, val_f1, val_auc)
                    #f.write(results + '\n')
        f.write(results + '\n')
    print("results:\n\n")
    print(results)
    return results



if __name__ == "__main__":
    seed = np.random.randint(0, 10000)
    set_random_seed(seed)
    print(f"Using seed: {seed}")
    opts = parse_options()
    root_path = '/data01/behavior_group/d21_qiaoyy/DOLOs/protocols/'
    train_roots = ['train_fold2.csv']
    test_roots = ['train_fold2.csv']
    n_t = len(test_roots)
    with open('Outputs/results.txt','w') as f:
      for n in range(n_t):
          train_file = train_roots[n]
          test_file = test_roots[n]
          train_path = root_path + train_file
          test_path = root_path + test_file
          train_fs = read_file(train_path)
          test_fs = read_file(test_path)
          results = train_test(train_fs,test_fs, args=opts)
          

