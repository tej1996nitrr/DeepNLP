import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.tensorboard import SummaryWriter
from utils import save_checkpoint, load_checkpoint
from loader import get_loader
from models import CNNtoRNN

def train():
    transform = transforms.Compose(
        [
            transforms.Resize([356,356]),
            transforms.RandomCrop((299,299)) ,#inception takes input 299,299
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5),(0.5, 0.5, 0.5))
        ] 
    )
    train_loader, dataset = get_loader(
        root_folder="",
        annotation_file="...",
        transform=transform,
        num_workers=2
    )
    torch.backends.cudnn.benchmark = True # performance boosting
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    load_model = False
    save_model = True

    #Hyperparameters

