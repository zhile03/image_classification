import os
import cv2
import torch
import random
import numpy as np
from torch.utils.data import Dataset, DataLoader

def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict

def augmentation(img): 
    # pad to 40x40 (4 pixels on each side)
    img = np.pad(img, ((0, 0), (4, 4), (4, 4)), mode='reflect') # [3, 40, 40]
    
    # random crop to 32x32
    h = random.randint(0, 8)
    w = random.randint(0, 8)
    img = img[:, h:h + 32, w:w + 32]

    # geometric transformation
    if random.random() < 0.5:  # hflip
        img = img[:, :, ::-1]

    return np.ascontiguousarray(img)

class CIFAR10(Dataset):
    def __init__(self, data_dir, phase='train'):
        self.data_dir = data_dir # directory containing the CIFAR-10 dataset
        self.phase = phase # train or test phase
        
        # load data
        self.data = []
        self.labels = []
        
        if self.phase == 'train':
            # load training batches 1-5
            for i in range(1, 6):
                batch_file = os.path.join(self.data_dir, f'data_batch_{i}')
                batch = unpickle(batch_file)
                self.data.append(batch[b'data'])
                self.labels.extend(batch[b'labels'])
                
        else:
            # load test batch
            batch_file = os.path.join(self.data_dir, 'test_batch')
            batch = unpickle(batch_file)
            self.data.append(batch[b'data'])
            self.labels.extend(batch[b'labels'])
            
        # concatenate all data
        self.data = np.concatenate(self.data, axis=0)
        self.data = self.data.reshape(-1, 3, 32, 32) # reshape to [C, H, W]
        self.labels = np.array(self.labels)
        
        # precompute normalization parameters
        self.mean = np.array([0.4914, 0.4822, 0.4465]).reshape(3, 1, 1)
        self.std = np.array([0.2023, 0.1994, 0.2010]).reshape(3, 1, 1)
    
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, index):
        # get image and label
        img = self.data[index].astype(np.float32) / 255.0 # [3, 32, 32] normalized to [0, 1]
        label = self.labels[index]
        
        if self.phase == 'train':
            # data augmentation
            img = augmentation(img)
            
        img = torch.from_numpy(img).float()
        img = (img - torch.from_numpy(self.mean).float()) / torch.from_numpy(self.std).float() # normalize
        
        return img, label
    

if __name__ == '__main__':
    os.makedirs('./test_dataloader/', exist_ok=True)
    
    data_dir = './cifar-10-batches-py'
    trainset = CIFAR10(data_dir=data_dir, phase='train')
    trainloader = DataLoader(trainset, batch_size=128, shuffle=True, num_workers=8, pin_memory=True)

    testset = CIFAR10(data_dir=data_dir, phase='test')
    testloader = DataLoader(testset, batch_size=128, shuffle=False, num_workers=8, pin_memory=True)
    
    # check a batch from the trainloader
    images, labels = next(iter(trainloader))
    print(f'Images shape: {images.shape}')  # should be [128, 3, 32, 32]
    print(f'Labels shape: {labels.shape}')  # should be [128]

    # save an example image (denormalize for visualization)
    img = images[0].numpy() # [3, 32, 32]
    mean = np.array([0.4914, 0.4822, 0.4465]).reshape(3, 1, 1)
    std = np.array([0.2023, 0.1994, 0.2010]).reshape(3, 1, 1)
    img = (img * std + mean)
    img = np.clip(img * 255, 0, 255).astype(np.uint8).transpose(1, 2, 0) # [32, 32, 3]
    cv2.imwrite('./test_dataloader/cifar10_example.png', cv2.cvtColor(img, cv2.COLOR_RGB2BGR))