import copy
import numpy as np
from pathlib import Path
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F


class BaselineNet(nn.Module):
    def __init__(self, in_shape, hidden_1, out_heads, out_len):
        super().__init__()
        self.in_shape = in_shape
        self.fc1 = nn.Linear(in_shape, hidden_1)
        self.heads = nn.ModuleList([nn.Linear(hidden_1, out_len) for _ in range(out_heads)])
        self.relu = nn.ReLU()

    def forward(self, x):
        x = x.view(-1, self.in_shape)
        hidden = self.relu(self.fc1(x))
        # [head(hidden) for head in self.heads]
        # y = torch.sigmoid(self.fc2(hidden))
        return [self.relu(head(hidden)) for head in self.heads]


class MaskedBCELoss(nn.Module):
    def __init__(self, masked_with=-1):
        super().__init__()
        self.masked_with = masked_with

    def forward(self, input, org_image, target):
        target = target.view(input.shape)
        org_image = org_image.view(input.shape)
        loss = F.binary_cross_entropy(input, org_image, reduction='none')
        loss[target == self.masked_with] = 0
        return loss.sum()


def train_BaselineNet(in_shape, out_heads, out_len, results_file, device, dataloaders, dataset_sizes, learning_rate, num_epochs,
          early_stop_patience, model_path):

    # Train baseline
    baseline_net = BaselineNet(in_shape, 20, out_heads, out_len)
    baseline_net.to(device)
    optimizer = torch.optim.Adam(baseline_net.parameters(), lr=learning_rate)
    # criterion = MaskedBCELoss()
    criterion = nn.CrossEntropyLoss()
    best_loss = np.inf
    early_stop_count = 0

    for epoch in range(num_epochs):
        for phase in ['train', 'val']:
            if phase == 'train':
                baseline_net.train()
            else:
                baseline_net.eval()

            running_loss = 0.0
            num_preds = 0

            bar = tqdm(dataloaders[phase],
                       desc='NN Epoch {} {}'.format(epoch, phase).ljust(20))
            for i, batch in enumerate(bar):
                
                inputs = torch.as_tensor(batch['input'], dtype=torch.float32)
                label = torch.as_tensor(batch['label'], dtype=torch.float32)

                inputs = inputs.to(device)
                targets = label.to(device)

                
                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    preds = baseline_net(inputs)
                    ##
                    losses = [criterion(pred, targets[:, k]) for k, pred in enumerate(preds)]
                    # losses = [F.binary_cross_entropy (F.softmax(pred, dim = 1), targets[:, k], reduce='sum') for k, pred in enumerate(preds)]
                    
                    total_loss = sum(losses) 
                    ##

                    if phase == 'train':
                        total_loss.backward()
                        optimizer.step()

                running_loss += total_loss.item()
                num_preds += 1
                if i % 10 == 0:
                    bar.set_postfix(loss='{:.2f}'.format(running_loss / num_preds),
                                    early_stop_count=early_stop_count)
            
            

            epoch_loss = running_loss / dataset_sizes[phase]
            mem = '%.3gG' % (torch.cuda.memory_reserved() / 1E9 if torch.cuda.is_available() else 0)  # (GB)
            s = ('%10s' * 3 + '%10.4g' ) % (
                '%g/%g' % (epoch + 1, num_epochs ), mem, phase, epoch_loss)
            with open(results_file, 'a') as f:
                f.write(s + '\n')  # append metrics, val_loss  
            results_file
            results_file
            results_file
            # deep copy the model
            if phase == 'val':
                if epoch_loss < best_loss:
                    best_loss = epoch_loss
                    best_model_wts = copy.deepcopy(baseline_net.state_dict())
                    early_stop_count = 0
                else:
                    early_stop_count += 1

        if early_stop_count >= early_stop_patience:
            break

    baseline_net.load_state_dict(best_model_wts)
    baseline_net.eval()

    # Save model weights
    Path(model_path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(baseline_net.state_dict(), model_path)

    return baseline_net
