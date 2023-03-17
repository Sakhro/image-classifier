import numpy as np
import torch
from torchvision import datasets, transforms, models
from collections import OrderedDict
import os
import argparse
from workspace_utils import active_session

def is_savefile_exist(save_path):
  return os.path.isfile(save_path)

def parse_args():
    parser = argparse.ArgumentParser(prog='Image classifier training')

    parser.add_argument("data_dir", default = "./flowers", type = str)
    parser.add_argument("--arch", default = "vgg13", choices = ["vgg13", "vgg16"])
    parser.add_argument("--learning_rate", default = 0.003, type = float)
    parser.add_argument("--epochs", default = 1, type = int)
    parser.add_argument("--save_path", default="trained_models/checkpoint.pth", type = str)
    parser.add_argument("--gpu", action='store_const', default = False, const = True)
    parser.add_argument("--hidden_unit", default = 512, type = int)
    
    args = parser.parse_args()
    
    return args

def get_model(arch):
    if arch == "vgg13":
        model = models.vgg13(pretrained=True)
    elif arch == "vgg16":
        model = models.vgg16(pretrained=True)
    else:
        raise ValueError('Only vgg16 and vgg13 are posible')
        
    for param in model.parameters():
        param.requires_grad = False
        
    return model

def get_classifier(input_size, hidden_unit, output_size):
    classifier = torch.nn.Sequential(OrderedDict([
                          ('fc1', torch.nn.Linear(input_size, hidden_unit)),
                          ('relu', torch.nn.ReLU()),
                          ('drop', torch.nn.Dropout(0.2)),
                          ('fc2', torch.nn.Linear(hidden_unit, output_size)),
                          ('output', torch.nn.LogSoftmax(dim=1))
                          ]))
    
    return classifier

def get_dataloaders(data_dir):
    train_dir = data_dir + '/train'
    valid_dir = data_dir + '/valid'
    test_dir = data_dir + '/test'

    train_transforms = transforms.Compose([transforms.RandomRotation(30),
                                           transforms.RandomResizedCrop(224),
                                           transforms.RandomHorizontalFlip(),
                                           transforms.ToTensor(),
                                           transforms.Normalize([0.485, 0.456, 0.406],
                                                                [0.229, 0.224, 0.225])])

    test_valid_transforms = transforms.Compose([transforms.Resize(255),
                                                transforms.CenterCrop(224),
                                                transforms.ToTensor(),
                                                transforms.Normalize([0.485, 0.456, 0.406],
                                                                     [0.229, 0.224, 0.225])])

    train_data = datasets.ImageFolder(train_dir, transform=train_transforms)
    test_data = datasets.ImageFolder(test_dir, transform=test_valid_transforms)
    valid_data = datasets.ImageFolder(valid_dir, transform=test_valid_transforms)

    trainloader = torch.utils.data.DataLoader(train_data, batch_size=32, shuffle=True)
    testloader = torch.utils.data.DataLoader(test_data, batch_size=32)
    validloader = torch.utils.data.DataLoader(valid_data, batch_size=32)
    
    dataloaders = {
        "train": torch.utils.data.DataLoader(train_data, batch_size=32, shuffle=True),
        "test": torch.utils.data.DataLoader(test_data, batch_size=32),
        "valid": torch.utils.data.DataLoader(valid_data, batch_size=32),
    }
    
    return dataloaders, train_data.class_to_idx

def validation(model, dataloaders, criterion, device, data_type):
    test_loss = 0
    accuracy = 0

    model.eval()

    with torch.no_grad():
        for inputs, labels in dataloaders[data_type]:
            inputs, labels = inputs.to(device), labels.to(device)

            logps = model.forward(inputs)
            loss = criterion(logps, labels)

            test_loss += loss.item()

            ps = torch.exp(logps)
            top_p, top_class = ps.topk(1, dim=1)
            equals = top_class == labels.view(*top_class.shape)
            accuracy += torch.mean(equals.type(torch.FloatTensor)).item()
            
    model.train()

    return test_loss, accuracy

def train(model, dataloaders, optimizer, device, epochs):
    criterion = torch.nn.NLLLoss()
    model.to(device);
    
    steps = 0
    print_every = 5
    running_loss = 0
    
    model.train()
    
    print("Start training")
        
    for epoch in range(int(epochs)):
        for inputs, labels in dataloaders['train']:
            steps += 1
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()

            logps = model.forward(inputs)
            loss = criterion(logps, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            if steps % print_every == 0:
                test_loss, accuracy = validation(model, dataloaders, criterion, device, "valid")
                validloader = dataloaders["valid"]

                print(f"Epoch {epoch+1}/{epochs}.. "
                      f"Train loss: {running_loss/print_every:.3f}.. "
                      f"Test loss: {test_loss/len(validloader):.3f}.. "
                      f"Test accuracy: {accuracy/len(validloader):.3f}")

                running_loss = 0
    
    print("End training")

def save_model(save_path, model, optimizer, arch):
    checkpoint = {
        'arch': arch,
        'class_to_idx': model.class_to_idx,
        'classifier': model.classifier,
        'optimizer_state_dict': optimizer.state_dict(),
        'model_state_dict': model.state_dict()
    }
    
    torch.save(checkpoint, save_path)
    
def main():
    args = parse_args()
    
    model = get_model(args.arch)
    
    output_size = 102
    input_size = model.classifier[0].in_features
    
    classifier = get_classifier(input_size, args.hidden_unit, output_size)
    model.classifier = classifier
    
    dataloaders, class_to_idx = get_dataloaders(args.data_dir)
    model.class_to_idx = class_to_idx
    
    device = torch.device('cuda' if args.gpu else 'cpu')
    
    optimizer = torch.optim.Adam(model.classifier.parameters(), lr=args.learning_rate)
    
    if is_savefile_exist(args.save_path):
        loaded_file = torch.load(args.save_path)
        model.load_state_dict(loaded_file["model_state_dict"])
        optimizer.load_state_dict(loaded_file["optimizer_state_dict"])

    train(model, dataloaders, optimizer, device, args.epochs)
    
    save_model(args.save_path, model, optimizer, args.arch)
    
if __name__ == '__main__':
    with active_session():
        main()

# python train.py ./flowers --arch vgg13 --learning_rate 0.003 --hidden_unit 512 --epochs 1 --gpu