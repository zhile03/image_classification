import os
import yaml
import argparse
import torch
import time
import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
import glob
import matplotlib.pyplot as plt
import cv2
import numpy as np
from model import ResNet
from dataset import CIFAR10
from torch.utils.data import DataLoader

# Apply He initialization
def weights_init(m):
  if isinstance(m, nn.Conv2d):
    nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')


def train(model, dataloader, criterion, device, optimizer, scheduler, iteration, max_iterations):
    model.train()
    running_loss = 0.0
    total_samples = 0
    
    for batch_idx, (images, labels) in enumerate(dataloader):
        if iteration >= max_iterations:
            break
        
        images = images.float().to(device)
        labels = labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        scheduler.step() # step per iteration based on the paper
        
        running_loss += loss.item() * images.size(0)
        total_samples += images.size(0)
        iteration += 1
        
        if batch_idx % 100 == 0:
            print(f"Iteration {iteration}/{max_iterations} | Batch {batch_idx}/{len(dataloader)} | Loss: {loss.item():.6f}")
            
    epoch_loss = running_loss / total_samples if total_samples > 0 else 0
    return epoch_loss, iteration


def test(model, dataloader, criteria, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for images, labels in dataloader:
            images = images.float().to(device)
            labels = labels.to(device)
            outputs = model(images)
            loss = criteria(outputs, labels)
            
            running_loss += loss.item() * images.size(0)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
    avg_loss = running_loss / total
    accuracy = 100 * correct / total
    return avg_loss, accuracy


def save_checkpoints(iteration, save_dir, model, optimizer):
    checkpoints_dir = os.path.join(save_dir, 'checkpoints')
    os.makedirs(checkpoints_dir, exist_ok=True)
    checkpoint_path = os.path.join(checkpoints_dir, f'checkpoint_iter_{iteration}.pth')
    torch.save({
        'iteration': iteration,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }, checkpoint_path)
    print(f"Checkpoint saved: {checkpoint_path}")


def plot_metrics(train_losses, valid_accuracies, learning_rates, save_dir):
    iterations = range(1, len(train_losses)+1)
    plt.figure(figsize=(10, 6))

    plt.subplot(1, 2, 1)
    plt.plot(iterations, train_losses, label='Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(iterations, valid_accuracies, label='Validation Accuracy', color='orange')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.title('Validation Accuracy')
    plt.legend()

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'results.png'), dpi=200)
    plt.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--save-dir', type=str, default='exp/first-try', help='')
    parser.add_argument('--batch-size', type=int, default=128, help='batch size for training')
    parser.add_argument('--num-workers', type=int, default=8, help='the number of dataloader workers')
    parser.add_argument('--n', type=int, default=3, help='number of residual blocks per stage')
    parser.add_argument('--total-iterations', type=int, default=64000, help='total training iterations')
    parser.add_argument('--data-dir', type=str, default='./cifar-10-batches-py', help='directory containing CIFAR-10 dataset')
    parser.add_argument('--resume', action='store_true', help='resume training from the latest checkpoint')
    parser.add_argument('--resume_weight', type=str, default='', help='path to resume weight file if needed')
    opt = parser.parse_args()

    # device
    if torch.cuda.is_available():
        device = 'cuda'
    elif torch.backends.mps.is_available():
        device = 'mps'
    else:
        device = 'cpu'
    device = torch.device(device)

    # create folders to save results
    if os.path.exists(opt.save_dir):
        print(f"Warning: {opt.save_dir} exists, please delete it manually if it is useless.")
    os.makedirs(opt.save_dir, exist_ok=True)

    # save hyp-parameter
    with open(os.path.join(opt.save_dir, 'hyp.yaml'), 'w') as f:
        yaml.dump(vars(opt), f, sort_keys=False)

    # create model
    model = ResNet(n=opt.n)
    model.apply(weights_init)
    model.to(device)

    # loss and optimizer
    criteria = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=0.0001)
    scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=[32000, 48000], gamma=0.1)

    # check for the latest checkpoint
    start_iteration = 0
    if opt.resume and os.path.isfile(opt.resume_weight):
        checkpoint = torch.load(opt.resume_weight, map_location=device)
        start_iteration = checkpoint['iteration']
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        for _ in range(start_iteration):
            scheduler.step() 
        print(f"Resuming training from iteration {start_iteration}.")
    else:
        start_epoch = 0
        print("No checkpoint found, starting training from scratch.")

    
    # dataloader
    train_dataset = CIFAR10(data_dir=opt.data_dir, phase='train')
    train_dataloader = DataLoader(train_dataset, batch_size=opt.batch_size, shuffle=True, num_workers=opt.num_workers, pin_memory=True)

    test_dataset = CIFAR10(data_dir=opt.data_dir, phase='test')
    test_dataloader = DataLoader(test_dataset, batch_size=opt.batch_size, shuffle=False, num_workers=opt.num_workers, pin_memory=True)

    
    # list to store metrics
    iteration = start_iteration
    train_losses = []
    valid_accuracies = []
    learning_rates = []
    
    
    results_file = os.path.join(opt.save_dir, 'results.txt')
    with open(results_file, 'w') as f:
        f.write('Iteration | LR | Training Loss | Validation Loss | Accuracy | Time\n')
    
    
    while iteration < opt.total_iterations:
        t0 = time.time()
        train_loss, iteration = train(model=model, dataloader=train_dataloader, criterion=criteria, device=device,
                                    optimizer=optimizer, scheduler=scheduler, iteration=iteration, max_iterations=opt.total_iterations) 
        t1 = time.time()
        valid_loss, accuracy = test(model=model, dataloader=test_dataloader, criteria=criteria, device=device)
        t2 = time.time()
        current_lr = optimizer.param_groups[0]['lr'] 
        print(f"Iteration {iteration}/{opt.total_iterations} | LR: {current_lr:.5f} | Training Loss: {train_loss:.5f} |"
              f"Validation Loss: {valid_loss:.5f} | Accuracy: {accuracy:.2f}% | Time: {t2 - t0:.1f} seconds")

        # store metrics
        train_losses.append(train_loss)
        valid_accuracies.append(accuracy)
        learning_rates.append(current_lr)
        
        # save checkpoints
        if iteration % 5000 == 0 or iteration == opt.total_iterations:
            save_checkpoints(iteration, opt.save_dir, model, optimizer)
          
        with open(results_file, 'a') as f:
            f.write(f"{iteration} | {current_lr:.5f} | {train_loss:.5f} | {valid_loss:.5f} | {accuracy:.5f} | {t2-t0:.1f}\n")
        
    # plot metrics after training
    plot_metrics(train_losses, valid_accuracies, learning_rates, opt.save_dir)
    print("Training finished.")