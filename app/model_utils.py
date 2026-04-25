import torch
import torch.nn as nn
from torchvision.models import resnet18
from torchvision import transforms

CLASSES = ['airplane', 'automobile', 'bird',
           'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu") 

# This function builds a ResNet-18 model and modifies the final fully connected 
# layer to output the specified number of classes (default is 10 for CIFAR-10).
def build_model(num_classes=10):
    model = resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model

# This function loads the model weights from the specified path,
# moves the model to the appropriate device (CPU or GPU), and sets it to evaluation mode.
def load_model(weights_path: str):
    device = get_device()
    model = build_model(num_classes=10)

    state_dict = torch.load(weights_path, map_location=device, weights_only=True)
    model.load_state_dict(state_dict)

    model.to(device)
    model.eval()
    return model, device

def get_transform():
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465),
                             (0.2023, 0.1994, 0.2010))
    ])