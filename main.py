"""Function introduction
This py is the main program for training module of scene classification.

    train_and_val        -- training and evaluation process of scene classification
***********************************************************************************
Modification Date:2021-03-13  By: Yunhao Cheng
"""
import torch
from torch import optim
from .shufflenetv2 import shufflenetv2
import os
import numpy as np
import random
from torchvision.datasets import ImageFolder
from scene_classification.utils import confusion_matrix, get_acc, train_tf, LabelSmoothingCrossEntropy, plot_confusion_matrix
from utils.common_utils import create_dir, write_log
import datetime
from tensorboardX import SummaryWriter
from .config import *


def train_and_val(fpath):
    """Training and evaluation process of scene classification
    Parameters:
        fpath          -- training set path(str)
    **********************************************************
    Modification Date:2021-03-13  By: Yunhao Cheng
    """
    # set random seed
    torch.manual_seed(999)
    torch.cuda.manual_seed(999)
    np.random.seed(999)
    random.seed(999)

    # choose GPU or CPU
    os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(str(e) for e in GPU)
    if torch.cuda.is_available():
        DEVICE = torch.device("cuda")
        print('\nGPU IS AVAILABLE')
        torch.backends.cudnn.benchmark = True
    else:
        DEVICE = torch.device("cpu")

    # load train set and test set
    total_datasets = ImageFolder(fpath, transform=train_tf)
    image_name = list(total_datasets.classes)
    train_size = int(TRAIN_RATIO * len(total_datasets))
    test_size = len(total_datasets) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(total_datasets, [train_size, test_size])
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=1, shuffle=False)
    length_train = len(train_dataset)
    length_val = len(val_dataset)

    # instantiation network
    net = shufflenetv2(class_num=len(os.listdir(fpath))).to(DEVICE)
    print('The Model is shufflenetv2\n')

    # optimizer and loss function
    optimizer = optim.SGD(net.parameters(), lr=LR, momentum=0.9, weight_decay=5e-4)
    loss_function = LabelSmoothingCrossEntropy()

    # warmup
    train_scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=MILESTONES, gamma=GAMMA)

    # create folder to save information
    date_now = datetime.datetime.now().strftime('%Y%m')
    data_now_tb = datetime.datetime.now().strftime('%Y%m%d%H%M%S')
    sup_fpath = os.path.dirname(fpath)
    model_path = os.path.join(sup_fpath, 'scene_model_' + date_now)
    create_dir(model_path)
    checkpoint_path = os.path.join(model_path, '{type}.pth')
    log_file = os.path.join(model_path, 'log.txt')
    tb_path = os.path.join(model_path, 'runs')
    np.save(os.path.join(model_path, 'classes.npy'), image_name)
    create_dir(tb_path)
    writer = SummaryWriter(log_dir=os.path.join(tb_path, data_now_tb))

    # training and evaluation process
    best_acc = 0.0
    iters = 0
    for epoch in range(1, EPOCH):
        net.train()
        train_loss, train_correct_num, train_num = 0.0, 0, 0.0

        for i, (image, label) in enumerate(train_loader):
            image, label = image.to(DEVICE), label.to(DEVICE)
            optimizer.zero_grad()
            output = net(image)
            train_correct_num += get_acc(output, label)
            train_num += image.shape[0]
            loss = loss_function(output, label)
            train_loss += loss.item()

            # backward
            loss.backward()
            optimizer.step()

            train_acc = train_correct_num / train_num

            iters += 1
            # training information
            logs = 'Training Epoch: {epoch} [{trained_samples}/{total_samples}]\tLoss: {:0.4f}\tAcc: {:0.4f}\tLR: {:0.6f}'.format(
                train_loss / (i+1),
                train_acc,
                optimizer.param_groups[0]['lr'],
                epoch=epoch,
                trained_samples=int(train_num),
                total_samples=length_train)
            print(logs)
            # save information to tensorboard
            write_log(log_file, logs, iters)
            writer.add_scalar('Train/lr', optimizer.param_groups[0]['lr'], iters)
            writer.add_scalar('Train/loss', (train_loss/(i+1)), iters)
            writer.add_scalar('Train/acc', train_acc, iters)

        train_scheduler.step()

        # start to save best performance model
        if epoch % 2 == 0 and epoch > MILESTONES[1]:
            net.eval()
            correct_pred_dict = dict()
            count_class_dict = dict()
            count_acc_dict = dict()
            conf_matrix = torch.zeros(len(os.listdir(fpath)), len(os.listdir(fpath)))

            for i, data in enumerate(val_loader):
                images, labels = data
                images, labels = images.to(DEVICE), labels.to(DEVICE)
                with torch.no_grad():
                    outputs = net(images)

                _, pred = outputs.topk(1, 1, largest=True, sorted=True)
                conf_matrix = confusion_matrix(pred, labels, conf_matrix)
                labels = labels.view(labels.size(0), -1).expand_as(pred)
                correct = pred.eq(labels).float()

                if correct == 1:
                    if image_name[pred] not in correct_pred_dict:
                        correct_pred_dict[image_name[pred]] = 0
                    correct_pred_dict[image_name[pred]] += 1

                if image_name[labels] not in count_class_dict:
                    count_class_dict[image_name[labels]] = 0
                count_class_dict[image_name[labels]] += 1

                val_correct_num = sum(correct_pred_dict.values())
                val_num = sum(count_class_dict.values())
                val_acc = val_correct_num / val_num

                logs = 'Testing: [{val_samples}/{total_samples}]\tAccuracy: {:.4f}'.format(
                                                                            val_acc,
                                                                            val_samples=val_num,
                                                                            total_samples=length_val)
                print(logs)
            # save information
            write_log(log_file, logs, iters)
            writer.add_scalar('Test/acc', val_acc, epoch)

            for m in range(len(image_name)):
                if image_name[m] not in correct_pred_dict:
                    correct_pred_dict[image_name[m]] = 0
                if image_name[m] in count_class_dict:
                    count_acc_dict[image_name[m]] = correct_pred_dict[image_name[m]] / count_class_dict[image_name[m]]
                else:
                    count_acc_dict[image_name[m]] = correct_pred_dict[image_name[m]]

            logs = 'Accuracy per category:' + str(count_acc_dict)
            write_log(log_file, logs, iters)
            print()
            if best_acc < val_acc:
                torch.save(net.state_dict(), checkpoint_path.format(type='best'))
                best_acc = val_acc

    if best_acc > 0.9:
        logs = '模型准确率为：' + str(best_acc) + ', 训练已完成！'
    else:
        logs = '模型准确率为：' + str(best_acc) + ', 准确率较低，建议重新考虑场景类别！'
    write_log(log_file, logs, iters)
    writer.close()

    # 画混淆矩阵
    plot_confusion_matrix(cm=conf_matrix, classes=image_name, save_path=model_path)