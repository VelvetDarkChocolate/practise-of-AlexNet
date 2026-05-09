import torch
from torch import nn
from torchsummary import summary
import torch.nn.functional as F

class AlexNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.features=nn.Sequential(
            nn.Conv2d(in_channels=1,out_channels=96,kernel_size=11,stride=4),
            nn.ReLU(),

            nn.MaxPool2d(kernel_size=3,stride=2),

            nn.Conv2d(in_channels=96,out_channels=256,kernel_size=5,padding=2),
            nn.ReLU(),

            nn.MaxPool2d(kernel_size=3,stride=2),

            nn.Conv2d(in_channels=256,out_channels=384,kernel_size=3,padding=1),
            nn.ReLU(),

            nn.Conv2d(in_channels=384,out_channels=384,kernel_size=3,padding=1),
            nn.ReLU(),

            nn.Conv2d(in_channels=384,out_channels=256,kernel_size=3,padding=1),
            nn.ReLU(),

            nn.MaxPool2d(kernel_size=3,stride=2)
        )

        self.flatten=nn.Flatten()

        self.classifier=nn.Sequential(
            nn.Linear(6*6*256,4096),
            nn.ReLU(),
            nn.Dropout(p=0.5),

            nn.Linear(4096,4096),
            nn.ReLU(),
            nn.Dropout(p=0.5),

            nn.Linear(4096,10)
        )

    def forward(self,x):
        x=self.features(x)
        x=self.flatten(x)
        x=self.classifier(x)
        return x 
    
if __name__=="__main__":
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model=AlexNet().to(device)
    print(summary(model,(1,227,227)))
