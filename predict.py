import numpy as np
import torch
from torchvision import models
import os
import json
import argparse
from workspace_utils import active_session
from PIL import Image

def is_savefile_exist(save_path):
  return os.path.isfile(save_path)

def process_image(image):
    width, height = image.size
    shortest_side = 256
    
    ratio = max(shortest_side/width, shortest_side/height)
    new_size = int(width*ratio), int(height*ratio)
    image.thumbnail(new_size)
    
    input_size = 224
    left = (new_size[0] - input_size)/2
    top = (new_size[1] - input_size)/2
    right = (new_size[0] + input_size)/2
    bottom = (new_size[1] + input_size)/2
    crop_image = image.crop((left, top, right, bottom))
    
    np_image = np.array(crop_image)/255
    means = [0.485, 0.456, 0.406]
    std_dev = [0.229, 0.224, 0.225]
    np_image = (np_image - means)/std_dev
    
    np_image = np_image.transpose((2,0,1))

    return np_image

def parse_args():
    parser = argparse.ArgumentParser(prog='Image classifier prediction')

    parser.add_argument("image_path", type = str)
    parser.add_argument("--topk", default = 1, type = int)
    parser.add_argument("--gpu", action='store_const', default = False, const = True)

    args = parser.parse_args()
    
    return args

def get_model(arch = "vgg13"):
    if arch == "vgg13":
        model = models.vgg13(pretrained=True)
    elif arch == "vgg16":
        model = models.vgg16(pretrained=True)
    else:
        raise ValueError('Only vgg16 and vgg13 are posible')
        
    for param in model.parameters():
        param.requires_grad = False
        
    return model

def get_cat_name():
    with open('cat_to_name.json', 'r') as f:
        cat_to_name = json.load(f)
        
    return cat_to_name

def predict(np_image, model, device, topk=5):
    tensor_image = torch.from_numpy(np_image).type(torch.FloatTensor).to(device)
    
    tensor_image.unsqueeze_(0)
    
    model.eval()
    
    with torch.no_grad():
        logps = model.forward(tensor_image)
        
    ps = torch.exp(logps)
    top_p, top_class = ps.topk(topk)
    
    return top_p.cpu().numpy().tolist()[0], top_class.cpu().numpy().tolist()[0]

def get_model(arch):
    if arch == "vgg13":
        model = models.vgg13(pretrained=True)
    elif arch == "vgg16":
        model = models.vgg16(pretrained=True)
    else:
        raise ValueError('Only vgg16 and vgg13 are posible')
        
    return model

def get_classes_labels(class_to_idx, classes):
    cat_to_name = get_cat_name()
    
    classes_to_idx = {val: key for key, val in class_to_idx.items()}
    top_k_idx = [classes_to_idx[classe] for classe in classes]
    labels = [cat_to_name[idx] for idx in top_k_idx]
    
    return labels

def main():
    args = parse_args()
    checkpoint_path = "./trained_models/checkpoint.pth"
    
    if not is_savefile_exist(args.image_path):
        raise Exception('Wrong file path')

    if not is_savefile_exist(checkpoint_path):
        raise Exception('Missing checkpoint file')

    device = torch.device("cuda" if args.gpu else "cpu")
    
    image = Image.open(args.image_path)
    image = process_image(image)
    
    checkpoint = torch.load(checkpoint_path)

    model = get_model(checkpoint["arch"])
    model.classifier = checkpoint["classifier"]
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    
    probs, classes = predict(image, model, device, args.topk)

    labels = get_classes_labels(checkpoint["class_to_idx"], classes)
    
    for prob, label in zip(probs, labels):
        print(f"Result: {label} - {int(float(prob) * 100)}%")
    
  
if __name__ == '__main__':
    with active_session():
        main()
        
# python predict.py flower.jpeg --gpu