import torch
from torch import nn, optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models
import numpy as np
from PIL import Image

def load_data(data_dir = 'flowers'):
    train_dir = data_dir + '/train'
    valid_dir = data_dir + '/valid'
    test_dir = data_dir + '/test'

    # Define transforms
    train_transforms = transforms.Compose([transforms.RandomRotation(60),
                                              transforms.RandomHorizontalFlip(),
                                              transforms.RandomResizedCrop(224),
                                              transforms.ToTensor(),
                                              transforms.Normalize([0.485, 0.456, 0.406],
                                                                   [0.229, 0.224, 0.225])])

    test_transforms = transforms.Compose([transforms.Resize(255),
                                          transforms.CenterCrop(224),
                                          transforms.ToTensor(),
                                          transforms.Normalize([0.485, 0.456, 0.406],
                                                               [0.229, 0.224, 0.225])])

    # Load datasets
    train_data = datasets.ImageFolder(train_dir, transform = train_transforms)
    test_data = datasets.ImageFolder(test_dir, transform = train_transforms)
    validation_data = datasets.ImageFolder(valid_dir, transform = train_transforms)

    # Define data loaders
    trainloader = torch.utils.data.DataLoader(train_data, batch_size = 64, shuffle = True)
    testloader = torch.utils.data.DataLoader(test_data, batch_size = 64)
    validationloader = torch.utils.data.DataLoader(validation_data, batch_size = 64)
    
    return trainloader, testloader, validationloader, train_data.class_to_idx

class NeuralNetwork(nn.Module):
    # Source: fc_model.py function from Udacity Pytorch lesson
    def __init__(self, input_size, output_size, hidden_layers, drop_p=0.5):
        super().__init__()
        
        self.hidden_layers = nn.ModuleList([nn.Linear(input_size, hidden_layers[0])])
        
        layer_sizes = zip(hidden_layers[:-1], hidden_layers[1:])
        self.hidden_layers.extend(
            [nn.Linear(s1, s2) for s1, s2 in layer_sizes])
        
        self.output = nn.Linear(hidden_layers[-1], output_size)
        self.dropout = nn.Dropout(p=drop_p)
        
    def forward(self, x):
        for layer in self.hidden_layers:
            x = F.relu(layer(x))
            x = self.dropout(x)
        x = self.output(x)
        x = F.log_softmax(x, dim=1)
        
        return x    
    
def instantiate_model(arch = 'densenet169',
                      hidden_layers = [832, 350, 175],
                      learning_rate = 0.001):
    if arch == 'resnet50':
        model = models.resnet50(pretrained=True)
    elif arch == 'densenet169':
        model = models.densenet169(pretrained=True)
    elif arch == 'vgg13':
        model = models.vgg13(pretrained = True)
    else:
        print("Architecture options are: resnet50, densenet169, vgg13")

    arch_inputs = {'resnet50':2048,
                 'densenet169':1664,
                 'vgg13':25088}

    classifier = NeuralNetwork(input_size = arch_inputs[arch],
                               output_size = 102,
                               hidden_layers = hidden_layers)
    model.classifier = classifier
    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.classifier.parameters(), 
                           lr = learning_rate)
    
    return model, criterion, optimizer

def train_network(model, criterion, optimizer, 
                  trainloader, validationloader,
                  gpu,
                  epochs = 3, 
                  print_step = 50):
    device = torch.device("cuda" if gpu == True else 'cpu')
    steps = 0
    running_loss = 0
    model.to(device)
    for epoch in range(epochs):
        for images, labels in trainloader:
            steps += 1
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()

            output = model(images)
            loss = criterion(output, labels)
            running_loss += loss
            loss.backward()
            optimizer.step()

            if steps % print_step == 0:
                test_loss = 0
                accuracy = 0
                model.eval()

                with torch.no_grad():
                    for images, labels in validationloader:
                        images, labels = images.to(device), labels.to(device)
                        output = model(images)
                        loss = criterion(output, labels)
                        test_loss += loss.item()

                        ps = torch.exp(output)
                        top_prob, top_class = ps.topk(1, dim=1)
                        equals = top_class == labels.view(top_class.shape)
                        accuracy += torch.mean(equals.type(torch.FloatTensor))

                print(f'Epoch: {epoch + 1}/{epochs}'),
                print(f'Training loss: {running_loss / print_step:.3f} ',
                      f'Testing loss: {test_loss / len(validationloader):.3f} ',
                      f'Accuracy: {accuracy / len(validationloader):.3f}'),
                print('--')
                model.train()
                running_loss = 0
    print('Training complete.')

def save_checkpoint(model, optimizer, arch, hidden_layers, epochs, learning_rate,
                    filepath = 'script_checkpoint.pth'):
    checkpoint = {'arch': arch,
                  'hidden_layers': hidden_layers,
                  'epochs': epochs,
                  'learning_rate': learning_rate,
                  'class_to_idx': model.class_to_idx,
                  'optimizer_state_dict': optimizer.state_dict(),
                  'model_state_dict': model.state_dict()}

    torch.save(checkpoint, filepath)

def load_checkpoint(filepath):
    checkpoint = torch.load(filepath)
    model, criterion, optimizer = instantiate_model(checkpoint['arch'],
                                                    checkpoint['hidden_layers'],
                                                    checkpoint['learning_rate'])
    
    model.load_state_dict(checkpoint['model_state_dict'])
    model.class_to_idx = checkpoint['class_to_idx']
    
    return model
    
    
def process_image(image):
    im = Image.open(image)
    
    # resize to 256 pixels
    scale = min(im.width/256, im.height/256)
    width, height = int(im.width/scale), int(im.height/scale)
    im = im.resize((width, height))
    
    # crop to center 224x224 pixels
    left = (width - 224)/2
    top = (height - 224)/2
    right = (width + 224)/2
    bottom = (height + 224)/2

    im = im.crop((left, top, right, bottom))
    
    # normalize colors
    to_tensor = transforms.ToTensor()
    normalize = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    im = normalize(to_tensor(im))
    
    return im    
   
def predict(image_path, model, topk=5, gpu=True):
    device = torch.device('cuda' if gpu == True else 'cpu')
    model.to(device)
    
    im = process_image(image_path)
    im = im.unsqueeze_(0)
    
    model.eval()
    output = model(im.to(device))
    ps = torch.exp(output)
    
    top_prob, top_class = ps.topk(topk, dim=1)
    class_dict = dict((y, x) for x, y in model.class_to_idx.items())
    
    classes = []
    for k in top_class.squeeze().tolist():
        classes.append(class_dict[k])
    probs = top_prob.squeeze().tolist()
    
    return probs, classes