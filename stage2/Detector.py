import argparse
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
# from stage1.Network import Net
# from stage1.Dataset import FaceLandmarksDataset
# from stage1.Actions import *
from Network import Net
from Dataset import FaceLandmarksDataset
from Actions import *


DATAPATH = './'
LABLE_FILE_NAME = {'train': 'train.txt', 'test': 'test.txt'}

torch.set_default_tensor_type(torch.FloatTensor)


def main():
    # set arguments
    parser = argparse.ArgumentParser(description='Detector')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')

    parser.add_argument('--train-batch-size', type=int, default=8, metavar='N',
                       help='input batch size for training (default: 128)')
    parser.add_argument('--test-batch-size', type=int, default=8, metavar='N',
                       help='input batch size for testing (default: 128)')
    parser.add_argument('--epochs', type=int, default=2000, metavar='N',
                        help='input the number of epochs to train (default: 1000)')
    parser.add_argument('--learning-rate', type=float, default=0.00006, metavar='LR',
                        help='input the learning rate of optimizer (default: 0.00008)')
    parser.add_argument('--momentum', type=float, default=0.6, metavar='M',
                        help='input the momentum of optimizer (default: 0.25)')
    parser.add_argument('--log-interval', type=int, default=20, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--save-model', action='store_true', default=True,
                        help='save the current model')
    parser.add_argument('--save-directory', type=str, default='trained_models',
                        help='the directory to save learnt models')

    parser.add_argument('--phase', type=str, default='train',
                        help='choose from Train/train, Test/test, Predict/predict, Finetune/finetune')

    args = parser.parse_args()

    # process control
    torch.manual_seed(args.seed)
    is_use_cuda = not args.no_cuda and torch.cuda.is_available()
    device = torch.device("cuda" if is_use_cuda else "cpu")
    datasets = {x: FaceLandmarksDataset(DATAPATH, LABLE_FILE_NAME[x], data_transforms[x])
                for x in ['train', 'test']}
    train_loader = DataLoader(datasets['train'], batch_size=args.train_batch_size, shuffle=True)
    test_loader = DataLoader(datasets['test'], batch_size=args.test_batch_size)
    data_loaders = {'train': train_loader, 'val': test_loader}
    model = Net().to(device)

    # training control
    criterion = nn.MSELoss(reduction='mean')
    optimizer = optim.RMSprop(model.parameters(), lr=args.learning_rate, momentum=args.momentum)

    # phase control
    if args.phase == 'Train' or args.phase == 'train':
        print('===> Start Training')
        lest_loss = train(args, data_loaders, model, criterion, optimizer, device)
        f = open('lest_loss_' + str(lest_loss) + '.txt', 'a+')
        [print(k, file=f) for k in args.__dict__.items()]
        [print(optimizer, file=f)]
        f.close()

    elif args.phase == 'Test' or args.phase == 'test':
        print('===> Start Testing')
        test_model = model
        test_model.load_state_dict(torch.load('./trained_models/best_model.pt'))
        avg_loss = test(test_loader, device, criterion, test_model)
        print('Average loss for test set is {:.2f}. '.format(avg_loss))

    elif args.phase == 'Finetune' or args.phase == 'finetune':
        print('===> Start Finetuning')
        pretrained_model = model
        pretrained_model.load_state_dict(torch.load('./trained_models/best_model.pt'))
        trainable_vars = [pretrained_model.ip3.bias, pretrained_model.ip3.weight]
        optimizer = optim.SGD(trainable_vars, lr=args.learning_rate, momentum=args.momentum)
        train(args, data_loaders, pretrained_model, criterion, optimizer, device)

    elif args.phase == 'Predict' or args.phase == 'predict':
        print('===> Predict')
        img_dir_name = './data/I/001063.jpg'
        test_model = model
        test_model.load_state_dict(torch.load('./trained_models/best_model.pt'))
        predict(img_dir_name, test_model, device)


if __name__ == '__main__':
    main()
