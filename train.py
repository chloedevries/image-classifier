import utils
import argparse

parser = argparse.ArgumentParser()

parser.add_argument('--data_dir', dest='dir')
parser.add_argument('--arch', dest = 'arch', default = 'densenet169')
parser.add_argument('--hidden_layers', dest = 'hidden_layers', default = '1664, 832, 350, 175, 102')
parser.add_argument('--learning_rate', dest = 'learning_rate', default = 0.001)
parser.add_argument('--gpu', dest = 'gpu', default = True)
parser.add_argument('--epochs', dest = 'epochs', default = 3)

args = parser.parse_args()

data_dir = args.dir
arch = args.arch
hidden_layers = [int(x) for x in args.hidden_layers.split(',')]
learning_rate = args.learning_rate
gpu = args.gpu
epochs = int(args.epochs)

trainloader, testloader, validationloader, class_to_idx = utils.load_data(data_dir)

model, criterion, optimizer = utils.instantiate_model(arch = arch, hidden_layers = hidden_layers, 
                                                      learning_rate = learning_rate)
model.class_to_idx = class_to_idx

utils.train_network(model, criterion, optimizer, trainloader, validationloader, gpu, epochs)

utils.save_checkpoint(model, optimizer, arch, hidden_layers, epochs, learning_rate)

print('All done!')