import os
import torch
import time
import copy
import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from torchvision import transforms
from stage3.Dataset import data_transforms
torch.set_default_tensor_type(torch.FloatTensor)


def train(args, data_loaders, model, criterion_cls, criterion_pts, optim_adam, optim_sgd, scheduler, device):
    # set arguments
    since = time.time()
    epoch = args.epochs
    lest_loss = 100000.0
    best_acc = 0.0
    best_model = copy.deepcopy(model.state_dict())
    loss_list = {'train': [], 'val': []}
    acc_list = {'train': {'pos': [], 'neg': [], 'total': []},
                'val': {'pos': [], 'neg': [], 'total': []}}

    # loop train process for "epoch" times
    for epoch_id in range(epoch):
        print('=' * 20 + ' Epoch {}/{} '.format(epoch_id + 1, epoch) + '=' * 20)

        if epoch_id < args.adam_epoch:
            optimizer = optim_adam
        else:
            optimizer = optim_sgd

        for phase in ['train', 'val']:
            running_loss = 0.0
            corrects_cls = {'pos': 0, 'neg': 0}
            cls_count = {'pos': 0, 'neg': 0}
            # set model for different mode in different phases
            if phase == 'train':
                model.train()  # set model to training mode
            else:
                model.eval()  # set model to evaluate mode

            for batch_idx, batch in enumerate(data_loaders[phase]):
                # get data
                img = batch['image']
                landmark = batch['landmarks']
                cls = batch['cls']
                mask = batch['cls'].to(device)
                # ground truth
                input_img = img.to(device)
                target_pts = landmark.to(device)
                target_cls = cls.to(device)
                # clear the gradients in optimizer
                optimizer.zero_grad()

                # forward propagation
                with torch.set_grad_enabled(phase == 'train'):
                    # get output
                    output_cls, output_pts = model(input_img)
                    # get loss
                    cls_loss = criterion_cls(output_cls, target_cls)
                    pts_loss = criterion_pts(output_pts * mask.reshape(-1, 1), target_pts)
                    loss = args.cls_loss_weight * cls_loss + pts_loss
                    # backward propagation and optimize in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                running_loss += loss.item() * input_img.size(0)
                cls_count['pos'] += torch.sum(target_cls == 1)
                cls_count['neg'] += torch.sum(target_cls == 0)
                corrects_cls['pos'] += torch.sum((torch.max(output_cls, 1)[1] == target_cls) * target_cls)
                corrects_cls['neg'] += torch.sum((torch.max(output_cls, 1)[1] == target_cls) * (1 - target_cls))

            # calculate loss and accurate for current epoch
            epoch_loss = running_loss / len(data_loaders[phase].dataset)
            pos_acc = float(corrects_cls['pos']) / float(cls_count['pos'])
            neg_acc = float(corrects_cls['neg']) / float(cls_count['neg'])
            epoch_acc = float(corrects_cls['pos'] + corrects_cls['neg']) / len(data_loaders[phase].dataset)
            print('{} loss: {:.4f}, pos acc: {:.2%}, neg acc: {:.2%}, acc: {:.2%}. '
                  .format(phase, epoch_loss, pos_acc, neg_acc, epoch_acc))

            # record loss and accurate in list
            loss_list[phase].append(epoch_loss)
            acc_list[phase]['pos'].append(pos_acc)
            acc_list[phase]['neg'].append(neg_acc)
            acc_list[phase]['neg'].append(neg_acc)
            acc_list[phase]['total'].append(epoch_acc)

            # scheduler move a step in training phase
            if phase == 'train':
                scheduler.step(epoch_loss)

            # evaluate model in validation phase
            if phase == 'val' and epoch_loss < lest_loss and epoch_acc > best_acc:
                lest_loss = epoch_loss
                best_acc = epoch_acc
                best_model = copy.deepcopy(model.state_dict())
                print('* Current best model loss is {:.4f}, accuracy is {:.2%}. '
                      .format(lest_loss, best_acc))

        # save model
        if args.save_model:
            if not os.path.exists(args.save_directory):
                os.makedirs(args.save_directory)
            saved_model_name = os.path.join(args.save_directory, 'detector_epoch' + '_' + str(epoch_id) + '.pt')
            torch.save(model.state_dict(), saved_model_name)

    # get final results
    print('Best model val loss is {:.4f}, accuracy is {:.2%}. '.format(lest_loss, best_acc))
    model.load_state_dict(best_model)
    torch.save(model.state_dict(), os.path.join(args.save_directory, 'best_model.pt'))
    print('Best model saved as best_model.pt! ')
    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))

    # draw training plot
    x = [i for i in range(epoch)]
    y1 = loss_list['train']
    y2 = loss_list['val']
    y3 = acc_list['train']['total']
    y4 = acc_list['val']['total']
    plt.plot(x, y1, color="r", linestyle="-", marker="o", linewidth=1, label="train loss")
    plt.plot(x, y2, color="b", linestyle="-", marker="o", linewidth=1, label="val loss")
    plt.plot(x, y3, color="g", linestyle="-", marker="x", linewidth=1, label="train acc")
    plt.plot(x, y4, color="y", linestyle="-", marker="x", linewidth=1, label="val acc")
    plt.legend()
    plt.title('train and val vs. epoches')
    plt.xlabel('epochs')
    plt.ylabel('loss')
    plt.savefig("train and val vs epoches.jpg")
    plt.close('all')
    return


def test(data_loader, device, criterion_pts, model):
    corr_count = 0
    total_loss = 0.0
    # set model to evaluate mode
    model.eval()
    # forward propagation
    with torch.no_grad():
        # iterate over data batch
        for data in data_loader:
            # get data
            img = data['image']
            landmark = data['landmarks']
            cls = data['cls']
            # ground truth
            input_img = img.to(device)
            target_pts = landmark.to(device).float()
            output_pts, output_pts = model(input_img)
            loss = criterion_pts(output_pts, target_pts)
            total_loss += loss.item() * input_img.size(0)
            if cls == output_pts:
                corr_count += 1
    avg_acc = corr_count / len(data_loader.dataset)
    avg_loss = total_loss / len(data_loader.dataset)
    return avg_acc, avg_loss


def draw_box(event, x, y, flag, param):
    global test_img, x1, y1, x2, y2
    if event == cv2.EVENT_LBUTTONDOWN:
        x1, y1 = x, y
        cv2.rectangle(test_img, (x1, y1), (x, y), (0, 0, 255), 2)
    elif event == cv2.EVENT_LBUTTONUP:
        cv2.rectangle(test_img, (x1, y1), (x, y), (0, 0, 255), 2)
        x2, y2 = x, y


def predict(img_dir_name, model, device):
    # get roi
    global test_img, x1, y1, x2, y2
    test_img = cv2.imread(img_dir_name)
    x1, y1, x2, y2 = 0, 0, 0, 0
    msg = 'draw roi, press enter to confirm'
    cv2.namedWindow(msg)
    cv2.setMouseCallback(msg, draw_box)
    while True:
        cv2.imshow(msg, test_img)
        if cv2.waitKey(1) & 0xFF == 13:
            break
    rect = (x1, y1, x2, y2)
    # get test data
    image = Image.open(img_dir_name).convert('L')
    img_crop = image.crop(tuple(rect))
    sample = {'image': img_crop, 'landmarks': np.array([]), 'cls': 0}
    sample = data_transforms['test'](sample)
    # set model to evaluate mode
    model.eval()
    # forward propagation
    with torch.no_grad():
        output_cls, output_pts = model(sample['image'].to(device).reshape(-1, 1, 112, 112))
        plt.title('Predicted as {} example. '.format('Positive' if output_cls else 'Negative'))
        if output_cls:
            output_pts = output_pts.reshape(-1, 2).tolist()
            x, y = [l[0] for l in output_pts], [l[1] for l in output_pts]
            plt.scatter(x, y, c='r')
        plt.imshow(transforms.ToPILImage()(sample['image'].to(device)))
        plt.savefig("predict result")
        plt.close('all')
    return


