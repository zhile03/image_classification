import torch
import torch.nn as nn

class ResidualBlock(nn.Module):
  def __init__(self, in_channels, out_channels, stride=1):
    super(ResidualBlock, self).__init__()
    
    # first 3x3 convolution layer
    self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False) 
    self.bn1 = nn.BatchNorm2d(out_channels)
    self.relu = nn.ReLU(inplace=True)
    
    # second 3x3 convolution layer
    self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
    self.bn2 = nn.BatchNorm2d(out_channels)
    
    # shortcut connection
    self.shortcut = nn.Sequential()
    if in_channels != out_channels or stride != 1:
      self.shortcut = nn.Sequential(
          nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False), # 1x1 conv
          nn.BatchNorm2d(out_channels)
      )
    
  def forward(self, x):
    out = self.relu(self.bn1(self.conv1(x)))
    out = self.bn2(self.conv2(out)) 
    out += self.shortcut(x) # residual learning
    out = self.relu(out)
    return out

class ResNet(nn.Module):
  def __init__(self, n=3):
    super(ResNet, self).__init__()
    
    self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1, bias=False)
    self.bn1 = nn.BatchNorm2d(32)
    
    self.layer1 = self.make_layer(32, 32, n, stride=1) # 32 filters, 32x32 feature maps
    self.layer2 = self.make_layer(32, 64, n, stride=2) # 64 filters, 16x16 feature maps
    self.layer3 = self.make_layer(64, 128, n, stride=2) # 128 filters, 8x8 feature maps
    
    self.avgpool = nn.AdaptiveAvgPool2d((1,1))
    self.fc = nn.Linear(128, 10)
    
  def make_layer(self, in_channels, out_channels, n, stride):
    layers = [ResidualBlock(in_channels, out_channels, stride)]
    for _ in range(1, n):
      layers.append(ResidualBlock(out_channels, out_channels, 1)) # stride=1 for all but the first block
    return nn.Sequential(*layers)
  
  def forward(self, x):
    out = torch.relu(self.bn1(self.conv1(x)))
    out = self.layer1(out)
    out = self.layer2(out)
    out = self.layer3(out)
    
    out = self.avgpool(out)
    out = out.view(out.size(0), -1)
    out = self.fc(out)
    return out
