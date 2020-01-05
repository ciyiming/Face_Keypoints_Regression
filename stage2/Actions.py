import os
import torch
import time
import copy
import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from torchvision import transforms
# from stage1.Dataset import data_transforms
from Dataset import data_transforms


def train(args, data_loaders, model, criterion, optimizer, device):
    # set arguments
    since = time.time()
    epoch = args.epochs
    lest_loss = 100000.0
    best_model = copy.deepcopy(model.state_dict())
    loss_list = {'train': [], 'val': []}

    for epoch_id in range(epoch):
        print('=' * 10 + ' Epoch {}/{} '.format(epoch_id + 1, epoch) + '=' * 10)
        for phase in ['train', 'val']:
            running_loss = 0.0

            # set model for different mode in different phases
            if phase == 'train':
                model.train()  # set model to training mode
            else:
                model.eval()  # set model to evaluate mode

            for batch_idx, batch in enumerate(data_loaders[phase]):
                # get data
                img = batch['image']
                landmark = batch['landmarks']
                # ground truth
                input_img = img.to(device)
                target_pts = landmark.to(device).float()
                # clear the gradients in optimizer
                optimizer.zero_grad()

                # forward propagation
                with torch.set_grad_enabled(phase == 'train'):
                    # get output
                    output_pts = model(input_img)
                    # get loss
                    loss = criterion(output_pts, target_pts)
                    # backward propagation and optimize in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                running_loss += loss.item() * input_img.size(0)

            # calculate loss and accurate for current epoch
            epoch_loss = running_loss / len(data_loaders[phase].dataset)
            print('{} loss: {:.4f}. '.format(phase, epoch_loss))

            # record loss and accurate in list
            loss_list[phase].append(epoch_loss)

            if phase == 'val' and epoch_loss < lest_loss:
                lest_loss = epoch_loss
                best_model = copy.deepcopy(model.state_dict())
                print('* Current best model loss is {:.4f}. '.format(lest_loss))
        # save model
        if args.save_model:
            if not os.path.exists(args.save_directory):
                os.makedirs(args.save_directory)
            saved_model_name = os.path.join(args.save_directory, 'detector_epoch' + '_' + str(epoch_id) + '.pt')
            torch.save(model.state_dict(), saved_model_name)

    print('Best model val loss is {:.4f}. '.format(lest_loss))
    model.load_state_dict(best_model)
    torch.save(model.state_dict(), os.path.join(args.save_directory,
                                                'best_model_' + str(lest_loss) + '.pt'))
    print('Best model saved as best_model.pt! ')
    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))

    x = [i for i in range(epoch)]
    y1 = loss_list['train']
    y2 = loss_list['val']
    plt.plot(x, y1, color="r", linestyle="-", marker="o", linewidth=1, label="val")
    plt.plot(x, y2, color="b", linestyle="-", marker="o", linewidth=1, label="train")
    plt.legend()
    plt.title('train and val loss vs. epoches')
    plt.xlabel('epochs')
    plt.ylabel('loss')
    plt.savefig("train and val loss vs epoches.jpg")
    plt.close('all')
    return lest_loss


def test(data_loader, device, criterion, model):
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
            # ground truth
            input_img = img.to(device)
            target_pts = landmark.to(device).float()
            output_pts = model(input_img)
            loss = criterion(output_pts, target_pts)
            total_loss += loss.item() * input_img.size(0)
    avg_loss = total_loss / len(data_loader.dataset)
    return avg_loss


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
    sample = {'image': img_crop, 'landmarks': np.array([])}
    sample = data_transforms['test'](sample)
    # set model to evaluate mode
    model.eval()
    # forward propagation
    with torch.no_grad():
        output_pts = model(sample['image'].to(device).reshape(-1, 1, 112, 112))
        output_pts = output_pts.reshape(-1, 2).tolist()
        x, y = [l[0] for l in output_pts], [l[1] for l in output_pts]
        plt.imshow(transforms.ToPILImage()(sample['image'].to(device)))
        plt.scatter(x, y, c='r')
        plt.savefig("predict result")
        plt.close('all')
    return


