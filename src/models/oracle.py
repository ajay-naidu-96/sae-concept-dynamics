import torch
import torch.nn.functional as F
import torch.nn as nn
import torchvision.models as models
import timm

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


class ResNet18CIFAR10(nn.Module):
    
    def __init__(self):
        super().__init__()
        self.backbone = models.resnet18(weights='IMAGENET1K_V1')
        self.backbone.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.backbone.maxpool = nn.Identity()
        
        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Identity()
        
        self.dropout = nn.Dropout(0.1)
        self.fc_linear = nn.Linear(in_features, 128) 
        self.fc_relu = nn.ReLU()
        self.fc_final = nn.Linear(128, 10)
    
    def forward(self, x):
        backbone_features = self.backbone(x)
        
        x = self.dropout(backbone_features)
        
        pre_relu = self.fc_linear(x)
        
        post_relu = self.fc_relu(pre_relu)
        
        logits = self.fc_final(post_relu)
        
        return logits, pre_relu, post_relu


class ViTWithActivations(nn.Module):
    """
    A wrapper for the timm ViT model to extract intermediate activations
    in a way that is compatible with the ActivationLogger.
    """
    def __init__(self, num_classes=10):
        super().__init__()
        # Initialize the ViT model from timm
        self.vit = timm.create_model("vit_base_patch16_224.orig_in21k_ft_in1k", pretrained=False)
        
        # Replace the final classification head for CIFAR-10
        self.vit.head = nn.Linear(self.vit.head.in_features, num_classes)

    # Delegate state_dict and load_state_dict to the underlying vit model
    # This is crucial for saving and loading weights in your TrainLoop
    def state_dict(self):
        return self.vit.state_dict()

    def load_state_dict(self, state_dict):
        self.vit.load_state_dict(state_dict)

    def forward(self, x):
        """
        This custom forward pass extracts features before the final classification head.
        """
        # The 'forward_features' method gives us the activations we want to log
        features = self.vit.forward_features(x)

        # The 'forward_head' method takes these features to produce the final logits
        logits = self.vit.forward_head(features)

        # We return the features in the second position to match what ActivationLogger expects
        # This is the same return format as your ResNet18 and VGG16 models
        return logits, features, None
        




