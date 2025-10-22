import os
import torch
import argparse
import json
import csv
import time
from datetime import datetime

from util import HSIDataset
from tqdm import tqdm
from torchinfo import summary
from main_net import RPCA_Net


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--data_path',
        required=True,
        help='Path to training data.',
    )
    parser.add_argument(
        '--save_path',
        default='./Weight',
        help='Path for checkpointing (weights).',
    )
    parser.add_argument(
        '--metrics_path',
        default='./Metrics',
        help='Path for saving metrics.',
    )
    parser.add_argument(
        '--resume',
        help='Resume training from saved checkpoint(s).',
    )
    parser.add_argument(
        '--checkpoint_freq',
        type=int,
        default=2,
        help='Checkpoint model every x epochs.',
    )
    parser.add_argument(
        '--loss_freq',
        type=int,
        default=20,
        help='Report (average) loss every x iterations.',
    )
    parser.add_argument(
        '--N_iter',
        type=int,
        default=10,
        help='Number of unrolled iterations.',
    )
    parser.add_argument(
        '--input_dim',
        type=int,
        default=8,
        help='Number of channels of the input tensor.',
    )
    parser.add_argument(
        '--set_lr',
        type=float,
        default=-1,
        help='Set new learning rate.',
    )
    return parser.parse_args()


def train(opt):

    torch.backends.cudnn.benchmark = True

    train_path = opt.data_path
    data_train = HSIDataset(train_path)
    data_train_loader = torch.utils.data.DataLoader(data_train, batch_size=1, shuffle=True, num_workers=4)

    model = RPCA_Net(N_iter=opt.N_iter, tensor_num_channels=opt.input_dim)
    model.cuda()
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-5)
    loss = torch.nn.L1Loss()
    summary(model, input_size=[(1, opt.input_dim, 64, 64), (1, opt.input_dim, 64, 64)])

    if opt.resume is not None:
        print('Resume training from' + opt.resume)
        checkpoint = torch.load(opt.resume)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        epoch_0 = checkpoint['epoch'] + 1
        model.train()
    else:
        print('Start training from scratch.')
        epoch_0 = 1

    if not opt.set_lr == -1:
        for groups in optimizer.param_groups: groups['lr'] = opt.set_lr; break
        print('New learning rate:', end=" ")
        for groups in optimizer.param_groups: print(groups['lr']); break

    if not os.path.exists(opt.save_path):
        os.makedirs(opt.save_path, exist_ok=True)
    else:
        print('WARNING: save_path already exists. Checkpoints may be overwritten.')
    
    if not os.path.exists(opt.metrics_path):
        os.makedirs(opt.metrics_path, exist_ok=True)
    
    # Initialize metrics logging
    metrics_log = []
    csv_filename = os.path.join(opt.metrics_path, f'training_metrics_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv')
    json_filename = os.path.join(opt.metrics_path, f'training_metrics_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json')
    
    with open(csv_filename, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Epoch', 'Iteration', 'Loss', 'Time'])

    avg_loss = 0
    epoch_losses = []
    start_time = time.time()
    
    for epoch in tqdm(range(epoch_0, 101), desc='Training'):
        epoch_start = time.time()
        for i, (data, omega, target) in enumerate(tqdm(data_train_loader, desc=f'Epoch {epoch}')):

            data, omega, target = data.cuda(), omega.cuda(), target.cuda()

            img = model(data, omega)
            total_loss = loss(img, target)

            optimizer.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 2)
            optimizer.step()
            avg_loss += total_loss.item()
            epoch_losses.append(total_loss.item())
            
            if ((i + 1) % opt.loss_freq) == 0:
                avg_loss_val = avg_loss/opt.loss_freq
                rep = (
                    f'Epoch: {epoch:>5d}, '
                    f'Iter: {i+1:>6d}, '
                    f'Loss: {avg_loss_val:>6.2e}'
                )
                tqdm.write(rep)
                
                # Log to CSV
                with open(csv_filename, 'a', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow([epoch, i+1, avg_loss_val, time.time() - start_time])
                
                avg_loss = 0

        # Calculate epoch metrics
        epoch_time = time.time() - epoch_start
        epoch_avg_loss = sum(epoch_losses) / len(epoch_losses) if epoch_losses else 0
        
        # Log epoch summary
        metrics_log.append({
            'epoch': epoch,
            'avg_loss': epoch_avg_loss,
            'epoch_time': epoch_time,
            'timestamp': datetime.now().isoformat()
        })
        
        if (epoch % opt.checkpoint_freq) == 0:
            torch.save(
                {'epoch': epoch, 'model_state_dict': model.state_dict(),
                 'optimizer_state_dict': optimizer.state_dict(),
                 'loss': epoch_avg_loss},
                os.path.join(opt.save_path, f'AGTC-Landsat_epoch_{epoch}.pth')
            )
            print(f'Checkpoint saved: epoch_{epoch}.pth')
        
        epoch_losses = []  # Reset for next epoch
    
    # Save final model
    torch.save(
        {'epoch': 100, 'model_state_dict': model.state_dict(),
         'optimizer_state_dict': optimizer.state_dict()},
        os.path.join(opt.save_path, 'AGTC-Landsat.pth')
    )
    print(f'Final model saved to {opt.save_path}/AGTC-Landsat.pth')
    
    # Save metrics JSON
    with open(json_filename, 'w') as f:
        json.dump(metrics_log, f, indent=4)
    
    print(f'Training complete! Metrics saved to {opt.metrics_path}')


if __name__ == '__main__':
    opt_args = parse_args()
    train(opt_args)
