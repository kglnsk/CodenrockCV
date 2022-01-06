#темплейт для запуска решения



import torch
from torch import nn
import torchvision
from torchvision import transforms,models
from torch.utils.data import Dataset
from torch.autograd import Variable
from PIL import Image
import os
import pandas as pd
import timm
import time 
from tqdm import tqdm


def predict_image(image):
    image_tensor = test_transforms(image).float()
    image_tensor = image_tensor.unsqueeze_(0)
    inputs = Variable(image_tensor)
    inputs = inputs.to(device)
    output_1 = inference_model(inputs)
    output_2 = effnet_model(inputs)
    output = (torch.nn.functional.softmax(output_1,dim = -1) + torch.nn.functional.softmax(output_2,dim = -1))/2.0
    print(output)
    index = output.detach().cpu().numpy().argmax()
    return index

if __name__ == "__main__":
  print("Ok")
  img_size = 299
  device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
  inference_model = torch.load("model.pt")
  effnet_model = torch.load('eff_net.pt')
  inference_model.eval().to(device)
  effnet_model.eval().to(device)
  test_transforms = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
  submission = pd.DataFrame(columns = ['image_name','class_id'])
  idxs = []
  files = os.listdir('data/test') 
  for filename in files:
    image = Image.open(os.path.join('./data/test',filename))
    index = predict_image(image)
    idxs.append(index)
  submission['image_name'] = files
  submission['class_id'] = idxs
  submission.to_csv('./data/out/submission.csv',index = False,sep = '\t')




