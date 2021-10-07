import os
from Model.resnet18 import ResNet, train_model
from LoadData.LoadData import InputImg

btch = 32
epoch = 85
width = 128
num_task = 2

os.environ['CUDA_VISIBLE_DEVICES'] = '/gpu:0'

datasets = InputImg('./Data', width)
train_x, train_y, test_x, test_y = datasets.load_data()
train_model(train_x, train_y, test_x, test_y, num_task, epoch, btch, width)