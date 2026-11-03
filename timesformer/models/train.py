import argparse
import os
import numpy as np
import time
import random

import torch
import torch.nn as nn
from dataloader import DataLoader
import torch.utils.data as data
# from torch.optim.lr_scheduler import ReduceLROnPlateau
from timesformer.models.vit import TimeSformer
from sklearn.metrics import classification_report, accuracy_score, f1_score, roc_curve, auc
import timesformer.models.optimizer as optim
#nn.CrossEntropyLoss()

#os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID" # 按照PCI_BUS_ID顺序从0开始排列GPU设备
#os.environ["CUDA_VISIBLE_DEVICES"] = "1"
torch.backends.cudnn.enabled = True
def parse_options():
    parser = argparse.ArgumentParser(description="WILTY baseline repo")
    ##### TRAINING DYNAMICS

    parser.add_argument('--device', type=str, default="cuda:0", help='the gpu id used for predict')
    parser.add_argument('--device_ID', type=str, default="cuda:0", help='the gpu id used for predict')
    parser.add_argument('--lr', type=float, default=1e-4, help='initial learning rate')  # default=0.001
    parser.add_argument('--when', type=int, default=15,
                        help='when to decay learning rate (default: 20)')
    parser.add_argument('--batch_size', type=int, default=2, help='initial batchsize')
    parser.add_argument('--num_epochs', type=int, default=100, help='total training epochs')
    parser.add_argument('--seed', type=int, default=1111,
                        help='random seed')

    parser.add_argument('--len', type=int, default=8, help='lenght of frames')
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

def train_one_epoch(args, train_data_loader, model,optimizer,loss_fn):
    ### Local Parameters
    epoch_loss = []
    epoch_predictions = []
    epoch_predictions_multi = []
    epoch_labels = []
    epoch_labels_multi = []
    start_time = time.time()
    #model.cuda()
    model.train()
    # data_size = len(train_data_loader)
    for cur_iter, (img, AUs, label，videos) in enumerate(train_data_loader):
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
        pred = model((img,AUs))
        # pred = F.softmax(pred, dim=1)

        #print(preds.shape)
        #print(label.shape)
        # Calculating Loss

        loss = loss_fn(pred, label)
        epoch_loss.append(loss)
        # Backward
        loss.backward()
        optimizer.step()

        if cur_iter % 500 == 0:
            print("iter {}, loss {:.20f}".format(str(cur_iter), loss))

    # CALCULATE ACCURACY AND CONFUSION MATRIX
    #epoch_predictions = torch.cat(epoch_predictions)
    #epoch_labels = torch.cat(epoch_labels)

    # Overall Epoch Results
    end_time = time.time()
    total_time = end_time - start_time

    # Acc and Loss
#    epoch_loss = .mean(epoch_loss)
    print("total_time:" , total_time)
    # return epoch_loss, acc, cfm, f1
    return epoch_loss


"""def evaluation(labels, preds):
    acc = accuracy_score(labels, preds)
    f1 = f1_score(labels, preds)
    fpr, tpr, thresholds = roc_curve(labels, preds, pos_label=1)
    auc_score = auc(fpr, tpr)
    return acc, f1, auc_score"""

def evaluation(args, val_batch, model):
    model.eval()
    labels = []
    preds = []
    for cur_iter, (img, AUs, label) in enumerate(val_batch):
        img = img.to(args.device)
        AUs = AUs.to(args.device)
        labels.append(label.detach().numpy())
        pred = model((img, AUs)).cpu().detach().numpy()
        idx = np.argmax(pred,axis=1)
        preds.append(idx)
    acc = accuracy_score(np.asarray(labels).flatten(), np.asarray(preds).flatten())
    # f1 = f1_score(labels, preds)
    # fpr, tpr, thresholds = roc_curve(labels, preds, pos_label=1)
    # auc_score = auc(fpr, tpr)
    return acc#, f1, auc_score

def split_list(lst, split_ratio):
    random.shuffle(lst)
    split_point = int(len(lst) * split_ratio)
    train_set = lst[:split_point]
    test_set = lst[split_point:]
    return train_set, test_set

def train_test(args):
    train_folder = "E:/dataset/ZDL/DOLOS/image/"
    video_names = np.load('train_video.npy')
    train_set, val_set = split_list(video_names, 0.8)
    train_dataset = DataLoader(video_folder=train_folder, img_size=args.size, frame_len=args.len, video_names=train_set)

    train_batch = data.DataLoader(train_dataset, batch_size=args.batch_size,
                                  shuffle=False, num_workers=1, drop_last=True)

    val_dataset = DataLoader(video_folder=train_folder, img_size=args.size, frame_len=args.len, video_names=val_set)

    val_batch = data.DataLoader(val_dataset, batch_size=args.batch_size,
                                  shuffle=False, num_workers=1, drop_last=True)

    model = TimeSformer(img_size=args.size, num_classes=2, num_frames=args.len, attention_type='divided_space_time')
    model.to(args.device)
    # data_size = len(train_batch)
        # Optimizer
    optimizer = torch.optim.Adam(model.parameters())
    #optimizer = torch.optim.SGD(lr=0.01,momentum=0.9,weight_decay=1e-4)
    # Loss Function
    loss_fn = nn.CrossEntropyLoss()
    print("\t Started Training")
    for epoch in range(args.num_epochs):
        cur_epoch = epoch + 1
        print('\t\t Epoch....', epoch + 1)
        # Training
        lr = args.lr * (0.1 ** (cur_epoch // 20))
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        #lr = optimizer.get_epoch_lr(cur_epoch + float(cur_iter) / data_size, cfg)
        #optimizer.set_lr(optimizer, lr)
        loss= train_one_epoch(args, train_batch, model, optimizer,loss_fn)
        acc = evaluation(args, val_batch, model)
        # acc = test(args, val_batch, model)
        print('acc:',acc)
        #print("loss:" ,loss)
        # if(cur_epoch % 10 == 0):
        #     torch.save(model,"Outputs/"+str(cur_epoch)+".pth")



if __name__ == "__main__":
    opts = parse_options()
    # For every run create an new entry and register the hyperparameters
    # num_existing_files = len(os.listdir(opts.log))
    #log_name = os.path.join(opts.log, str(time.time()) + '_' + opts.model_name + '.txt')
    """with open(log_name, 'w') as f:
        f.write("\nOptimizer and LR = Adam, " + str(opts.lr))
        f.write("\nBatch Size = " + str(opts.batch_size))
        f.write("\nEpochs = " + str(opts.num_epochs))
        f.write("\nNum Encoders = " + str(opts.num_encoders))
        f.write("\nAdapter = " + str(opts.adapter))
        if opts.adapter == True:
            f.write("\nAdapter Type = " + opts.adapter_type)
        f.write("\n------------------------------------------")
        f.write("\n------------------------------------------")
    f.close()"""

    train_test(args=opts)

