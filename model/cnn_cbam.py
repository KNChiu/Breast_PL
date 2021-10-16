import torch
import torch.nn as nn
import torch.nn.functional as F

class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc1   = nn.Conv2d(in_planes, in_planes // 16, 1, bias=False)
        self.relu1 = nn.LeakyReLU()
        self.fc2   = nn.Conv2d(in_planes // 16, in_planes, 1, bias=False)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return self.sigmoid(out)
    
class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()

        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1

        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)

class CnnCbam(nn.Module):
    def __init__(self):
        super(CnnCbam, self).__init__()
        self.conv1 = nn.Sequential(
                nn.Conv2d(in_channels = 3, out_channels = 16, kernel_size = 7,
                          stride = 1, padding = 3),
                # nn.BatchNorm2d(16),
                nn.LeakyReLU(),
                nn.MaxPool2d(kernel_size = 2),
                # nn.Dropout2d(0.2)
                )

        self.conv2 = nn.Sequential(
                nn.Conv2d(in_channels = 16, out_channels = 32, kernel_size = 5,
                          stride = 1, padding = 2),
                # nn.BatchNorm2d(32),
                nn.LeakyReLU(),
                nn.MaxPool2d(kernel_size = 2),
                # nn.Dropout2d(0.2)
                )

        self.conv3 = nn.Sequential(
                nn.Conv2d(in_channels = 32, out_channels = 64, kernel_size = 3,             
                          stride = 1, padding = 1),
                # nn.BatchNorm2d(64),
                nn.LeakyReLU(),
                nn.MaxPool2d(kernel_size = 2),
                # nn.Dropout2d(0.2)
                )
        
        self.fc1 = nn.Linear(in_features = 64 * 80 * 80, out_features = 128)
        self.fc2 = nn.Linear(in_features = 128, out_features = 32)
        self.fc3 = nn.Linear(in_features = 32, out_features = 8)
        self.fc4 = nn.Linear(in_features = 8, out_features = 2)
        
        self.ca1 = ChannelAttention(16)#64
        self.sa1 = SpatialAttention()
        
        self.ca2 = ChannelAttention(32)#64
        self.sa2 = SpatialAttention()
        
        self.ca3 = ChannelAttention(64)#64
        self.sa3 = SpatialAttention()
        
        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(64, 2)

    def forward(self, x):
        x = self.conv1(x)
        x = self.ca1(x) * x
        x = self.sa1(x) * x
        
        x = self.conv2(x)
        x = self.ca2(x) * x
        x = self.sa2(x) * x
        
        x = self.conv3(x)
        # x = self.ca3(x) * x
        # x = self.sa3(x) * x
        
        
        # x = self.gap(x)
        x = x.view(x.size(0), -1)#flatten
        # output = F.softmax(self.fc(x))
        
        
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        output = self.fc4(x)
        # output = F.softmax(self.fc4(x), dim=1)
        return output

# def CnnCbam(pretrain = False, **kwargs):
#     model = CNN()
#     return model