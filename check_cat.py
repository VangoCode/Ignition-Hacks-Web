import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import numpy as np
import PIL
from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True


from torch.autograd import Variable

import torch.nn as nn
import torch.nn.functional as F

import torch.optim as optim
#
class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = torchvision.models.resnet18(pretrained = True)
        for i in self.model.parameters():
          i.requires_grad = False

        self.model.fc = nn.Linear(512, len(classes))
    def forward(self, x):
        return self.model(x)

classes = ('Abyssinian', 'American Bobtail', 'American Curl', 'American Shorthair', 'American Wirehair', 'Applehead Siamese', 'Balinese', 'Bengal', 'Birman', 'Bombay', 'British Shorthair', 'Burmese', 'Burmilla', 'Calico', 'Canadian Hairless', 'Chartreux', 'Chausie', 'Chinchilla', 'Cornish Rex', 'Cymric', 'Devon Rex', 'Dilute Calico', 'Dilute Tortoiseshell', 'Domestic Long Hair', 'Domestic Medium Hair', 'Domestic Short Hair', 'Egyptian Mau', 'Exotic Shorthair', 'Extra-Toes Cat - Hemingway Polydactyl', 'Havana', 'Himalayan', 'Japanese Bobtail', 'Javanese', 'Korat', 'LaPerm', 'Maine Coon', 'Manx', 'Munchkin', 'Nebelung', 'Norwegian Forest Cat', 'Ocicat', 'Oriental Long Hair', 'Oriental Short Hair', 'Oriental Tabby', 'Persian', 'Pixiebob', 'Ragamuffin', 'Ragdoll', 'Russian Blue', 'Scottish Fold', 'Selkirk Rex', 'Siamese', 'Siberian', 'Silver', 'Singapura', 'Snowshoe', 'Somali', 'Sphynx - Hairless Cat', 'Tabby', 'Tiger', 'Tonkinese', 'Torbie', 'Tortoiseshell', 'Turkish Angora', 'Turkish Van', 'Tuxedo', 'York Chocolate')  # Defining the classes we have

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize((64, 64))

])

device = torch.device('cpu')
map_location=torch.device('cpu')
model = Net().to(device)
PATH = './cat_breeds1.pth'

# Loading the trained network
model.load_state_dict(torch.load(PATH, map_location=torch.device('cpu')))

def image_loader(image_name):
    """load image, returns cuda tensor"""
    image = Image.open(image_name)
    image = transform(image)
    image = image.unsqueeze(0)
    return image

def callImage(path):
    image = image_loader(path)

    pred = model(image)
    print(pred)
    predIdx = torch.argmax(pred)
    print(predIdx)
    return classes[predIdx]