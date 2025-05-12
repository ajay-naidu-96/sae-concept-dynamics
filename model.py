import torch
import torch.nn as nn
import torch.nn.functional as F

class MnistCNN(nn.Module):
    
    def __init__(self):
        super(MnistCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 10)
        
        
    def forward(self, x):

        x = self.pool(F.relu(self.conv1(x)))  
        x = self.pool(F.relu(self.conv2(x)))  

        x = x.view(x.size(0), -1)             

        fc1_logits = self.fc1(x)
        fc1_activations = F.relu(fc1_logits)
       
        logits = self.fc2(fc1_activations)         

        return logits, fc1_logits, fc1_activations
