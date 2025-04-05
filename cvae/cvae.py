import numpy as np
import pandas as pd
from pathlib import Path
import pyro
import pyro.distributions as dist
from pyro.infer import SVI, Trace_ELBO, Predictive, RenyiELBO
from torch.distributions.transforms import SigmoidTransform
import torch
import torch.nn as nn
from tqdm import tqdm
# from data import get_val_images
from Dataloader.dataloader import CVAEBITS, get_val_images
import torch.nn.functional as F
import copy



class Classifier(nn.Module):
    def __init__(self, in_shape, hidden_1):
        super().__init__()
        self.in_shape = in_shape
        self.fc1 = nn.Linear(in_shape, hidden_1)
        self.fc2 = nn.Linear(hidden_1, 1)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = x.view(-1, self.in_shape)
        hidden = self.relu(self.fc1(x))
        return torch.sigmoid(self.fc2(hidden))

class Encoder(nn.Module):
    def __init__(self, in_shape, hidden_1, z_dim):
        super().__init__()
        self.in_shape = in_shape
        self.fc1 = nn.Linear(in_shape, hidden_1)
        # self.fc2 = nn.Linear(hidden_1, hidden_2)
        self.fc21 = nn.Linear(hidden_1, z_dim)
        self.fc22 = nn.Linear(hidden_1, z_dim)
        self.relu = nn.ReLU()
        self.sp = nn.Softplus()

    def forward(self, x, y = None):
        # put x and y together in the same image for simplification
        if (y ==None):
            x = x.view(-1, self.in_shape)
        if(y != None):
            x = torch.cat((x, y.view(y.size(0), -1)), dim=-1)
        # then compute the hidden units
        hidden = self.relu(self.fc1(x))
        # then return a mean vector and a (positive) square root covariance
        # each of size batch_size x z_dim
        z_loc = self.fc21(hidden)
        # z_scale = torch.exp(self.fc22(hidden))
        z_scale = self.sp(self.fc22(hidden)) + 1e-2
        if torch.isnan(z_loc[0][0] ):
            z = 2
        return z_loc, z_scale

    # def __init__(self, in_shape, hidden_1, out_heads, out_len):
    #     super().__init__()
    #     self.in_shape = in_shape
    #     self.fc1 = nn.Linear(in_shape, hidden_1)
    #     self.heads = nn.ModuleList([nn.Linear(hidden_1, out_len) for _ in range(out_heads)])
    #     self.relu = nn.ReLU()

    # def forward(self, x):
    #     x = x.view(-1, self.in_shape)
    #     hidden = self.relu(self.fc1(x))
    #     # [head(hidden) for head in self.heads]
    #     # y = torch.sigmoid(self.fc2(hidden))
    #     return [head(hidden) for head in self.heads]

class Decoder(nn.Module):
    def __init__(self, z_dim, hidden_1, out_heads, out_len):
        super().__init__()
        self.z_dim = z_dim
        self.fc1 = nn.Linear(z_dim, hidden_1)
        self.heads = nn.ModuleList([nn.Linear(hidden_1, out_len) for _ in range(out_heads)])
        self.relu = nn.ReLU()

    def forward(self, z):     
        # z = z.view(-1, self.in_shape)
        hidden = self.relu(self.fc1(z))
        # [head(hidden) for head in self.heads]
        # y = torch.sigmoid(self.fc2(hidden))
        return [F.softmax(head(hidden), dim = -1) for head in self.heads]
        # return [torch.sigmoid(head(hidden)) for head in self.heads]
    
    




class CVAE(nn.Module):
    def __init__(self, in_shape, out_heads, out_len, z_dim, pre_trained_baseline_net):
        super().__init__()
        # The CVAE is composed of multiple MLPs, such as recognition network
        # qφ(z|x, y), (conditional) prior network pθ(z|x), and generation
        # network pθ(y|x, z). Also, CVAE is built on top of the NN: not only
        # the direct input x, but also the initial guess y_hat made by the NN
        # are fed into the prior network.
        hidden_1 = 10
        self.baseline_net = pre_trained_baseline_net
        self.prior_net = Encoder(in_shape + out_len*out_heads, hidden_1, z_dim)
        self.generation_net = Decoder(z_dim, hidden_1, out_heads, out_len)
        self.recognition_net = Encoder(in_shape + out_len*out_heads, hidden_1, z_dim)

    def model(self, xs, ys=None):
        # register this pytorch module and all of its sub-modules with pyro
        pyro.module("generation_net", self)
        batch_size = xs.shape[0]
        with pyro.plate("data"):

            # Prior network uses the baseline predictions as initial guess.
            # This is the generative process with recurrent connection
            with torch.no_grad():
                # this ensures the training process does not change the
                # baseline network
                xs = torch.as_tensor(xs, dtype=torch.float32)
                y_hat = torch.stack(self.baseline_net(xs), dim = 1)

            # sample 
            prior_loc, prior_scale = self.prior_net(xs + 1e-8, y_hat)
            ## the next line is for the time we dont want "z" be considered as a part of variational likelihood, i.e. p_{\theta}(z)
            zs = dist.Normal(prior_loc, prior_scale).sample()
            ## the next line takes "z" into account for lower bount optimization, i.e. p_{\theta}(z)
            # zs = pyro.sample('z', dist.Normal(prior_loc, prior_scale).to_event(1))
            # zs = pyro.sample('z', dist.Normal(prior_loc, prior_scale).to_event(2))
            # zs = pyro.sample('z', dist.TransformedDistribution(dist.Normal(prior_loc, prior_scale), SigmoidTransform()).to_event(1))

            # the output y is generated from the distribution pθ(y|x, z)
            loc = torch.stack(self.generation_net(zs), dim = 1)
            loc = torch.as_tensor(loc, dtype=torch.float32)
            if ys is not None:
                
                mask_loc = loc.view(batch_size, -1)
                mask_ys = ys.view(batch_size, -1)
                # y = pyro.sample('y',dist.Categorical(probs= loc, validate_args=False).to_event(1), obs = torch.as_tensor(torch.argmax(ys, dim=-1), dtype=torch.float32))
                y = pyro.sample('y', dist.Bernoulli(probs= loc, validate_args=False).to_event(2), obs=ys)
                # y = pyro.sample('y',dist.Categorical(probs= loc, validate_args=False).to_event(2), obs = torch.as_tensor(torch.argmax(ys, dim=-1), dtype=torch.float32))
            else:
                # In testing, no need to sample: the output is already a
                # probability in [0, 1] range.
                pyro.deterministic('y', loc.detach())

            # return the loc so we can debug it later
            return loc

    def guide(self, xs, ys=None):
        with pyro.plate("data"):
            if ys is None:
                # at inference time, ys is not provided. In that case,
                # the model uses the prior network
                # y_hat = self.baseline_net(xs).view(xs.shape)
                xs = torch.as_tensor(xs, dtype=torch.float32)
                y_hat = torch.stack(self.baseline_net(xs), dim = 1)
                loc, scale = self.prior_net(xs + 1e-8, y_hat)
            else:
                # at training time, uses the variational distribution
                # q(z|x,y) = normal(loc(x,y),scale(x,y))
                loc, scale = self.recognition_net(xs + 1e-8, ys)

            ## the next line is for the time we dont want "z" be considered as a part of variational likelihood, i.e. q_{\phi}(z)
            h = dist.Normal(loc, scale).sample()
            ## the next line takes "z" into account for lower bount optimization, i.e. q_{\phi}(z)
            # h = pyro.sample("z", dist.Normal(loc, scale).to_event(1))
            # h = pyro.sample("z", dist.Normal(loc, scale).to_event(1), infer={"is_auxiliary": True})
            # h = pyro.sample("z", dist.TransformedDistribution(dist.Normal(loc + 1e-8, scale + 1e-8), SigmoidTransform()).to_event(1))

    def save(self, model_path):
        torch.save({
            'prior': self.prior_net.state_dict(),
            'generation': self.generation_net.state_dict(),
            'recognition': self.recognition_net.state_dict()
        }, model_path)

    def load(self, model_path, map_location=None):
        net_weights = torch.load(model_path, map_location=map_location)
        self.prior_net.load_state_dict(net_weights['prior'])
        self.generation_net.load_state_dict(net_weights['generation'])
        self.recognition_net.load_state_dict(net_weights['recognition'])
        self.prior_net.eval()
        self.generation_net.eval()
        self.recognition_net.eval()


def train(in_shape, out_heads, out_len, z_dim, results_file, device, dataloaders, dataset_sizes, learning_rate, num_epochs,
          early_stop_patience, model_path, pre_trained_baseline_net):

    # clear param store
    pyro.clear_param_store()

    cvae_net = CVAE(in_shape, out_heads, out_len, z_dim, pre_trained_baseline_net)
    cvae_net.to(device)
    optimizer = pyro.optim.Adam({"lr": learning_rate, "weight_decay": 0.0001})
    svi = SVI(cvae_net.model, cvae_net.guide, optimizer, loss=RenyiELBO(num_particles = 5))
    

    # svi = SVI(cvae_net.model, cvae_net.guide, optimizer, loss=Trace_ELBO(num_particles = 5))
    best_loss = np.inf
    early_stop_count = 0
    Path(model_path).parent.mkdir(parents=True, exist_ok=True)

    # to track evolution
    val_inp, digits = get_val_images(dataloaders['val'])
    val_inp = val_inp.to(device)
    samples = []
    losses = []

    for epoch in range(num_epochs):
        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            running_loss = 0.0

            # Iterate over data.
            bar = tqdm(dataloaders[phase],
                       desc='CVAE Epoch {} {}'.format(epoch, phase).ljust(20))
            for i, batch in enumerate(bar):
                                
                inputs = torch.as_tensor(batch['input'], dtype=torch.float32)
                outputs = torch.as_tensor(batch['label'], dtype=torch.float32)
                inputs = inputs.to(device)
                outputs = outputs.to(device)

                if phase == 'train':
                    loss = svi.step(inputs, outputs) / inputs.size(0)
                else:
                    loss = svi.evaluate_loss(inputs, outputs) / inputs.size(0)

                # statistics
                running_loss += loss
                if i % 10 == 0:
                    bar.set_postfix(loss='{:.2f}'.format(loss),
                                    early_stop_count=early_stop_count)

                # track evolution
                if phase == 'train':
                    df = pd.DataFrame(columns=['epoch', 'loss'])
                    df.loc[0] = [epoch + float(i) / len(dataloaders[phase]), loss]
                    losses.append(df)
                    if i % 47 == 0:  
                        dfs = predict_samples(
                            val_inp, digits, cvae_net,
                            epoch + float(i) / len(dataloaders[phase]),
                        )
                        samples.append(dfs)

            epoch_loss = running_loss / dataset_sizes[phase]
            mem = '%.3gG' % (torch.cuda.memory_reserved() / 1E9 if torch.cuda.is_available() else 0)  # (GB)
            s = ('%10s' * 3 + '%10.4g' ) % (
                '%g/%g' % (epoch + 1, num_epochs ), mem, phase, epoch_loss)
            with open(results_file, 'a') as f:
                f.write(s + '\n')  # append metrics, val_loss
            # deep copy the model
            if phase == 'val':
                if epoch_loss < best_loss:
                    best_loss = epoch_loss
                    cvae_net.save(model_path)
                    early_stop_count = 0
                else:
                    early_stop_count += 1

        if early_stop_count >= early_stop_patience:
            break

    # Save model weights
    cvae_net.load(model_path)

    # record evolution
    # samples = pd.concat(samples, axis=0, ignore_index=True)
    # samples.to_csv('samples.csv', index=False)

    losses = pd.concat(losses, axis=0, ignore_index=True)
    losses.to_csv('losses.csv', index=False)

    return cvae_net


def predict_samples(inputs, digits, pre_trained_cvae, epoch_frac):
    predictive = Predictive(pre_trained_cvae.model,
                            guide=pre_trained_cvae.guide,
                            num_samples=1)
    preds = predictive(inputs)
    y_loc = preds['y'].squeeze().detach().cpu()
    y_loc = y_loc.view(y_loc.shape[0], -1)
    y_loc = y_loc.numpy()
    dfs = pd.DataFrame(data=y_loc)
    dfs['digit'] = digits.numpy()
    dfs['epoch'] = epoch_frac
    return dfs

def cvae_classifier(z_dim, dataloaders, dataset_sizes, pre_trained_cvae, results_file, num_epochs, device, learning_rate, 
                    early_stop_patience, model_path ):
    
    
    predictive = Predictive(pre_trained_cvae.model,
                            guide=pre_trained_cvae.guide,
                            num_samples=1)
    
    # Train Classifier
    classifier = Classifier(z_dim, 10)
    classifier.to(device)
    optimizer = torch.optim.Adam(classifier.parameters(), lr=learning_rate)
    best_loss = np.inf
    early_stop_count = 0

    for epoch in range(num_epochs):
        for phase in ['cvae_cls_train', 'cvae_cls_val']:
            if phase == 'cvae_cls_train':
                classifier.train()
            else:
                classifier.eval()

            running_loss = 0.0
            num_preds = 0

            bar = tqdm(dataloaders[phase],
                       desc='NN Epoch {} {}'.format(epoch, phase).ljust(20))
            for i, batch in enumerate(bar):
                
                # TODO: repalce batch['input'] with batch['z']
                inputs = torch.as_tensor(batch['input'], dtype=torch.float32)
                label = torch.as_tensor(batch['label'], dtype=torch.float32)

                inputs = inputs.to(device)
                targets = label.to(device)

                
                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'cvae_cls_train'):
                    
                    cvae_preds = predictive(inputs)
                    xs = torch.as_tensor(inputs, dtype=torch.float32)
                    y_hat = torch.stack(pre_trained_cvae.baseline_net(xs), dim = 1)
                    loc, scale = pre_trained_cvae.prior_net(xs + 1e-8, y_hat)
                    z_samples = dist.Normal(loc, scale).sample()
                    
                    z_loc = z_samples.squeeze().detach()
                    z_loc = z_loc.view(z_loc.shape[0], -1)
                    preds = classifier(z_loc)
                    ##
                    loss = F.binary_cross_entropy(preds.squeeze(), targets, reduction='mean')
                    
                    ##

                    if phase == 'cvae_cls_train':
                        loss.backward()
                        optimizer.step()
                running_loss += loss
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
            # deep copy the model
            if phase == 'cvae_cls_val':
                if epoch_loss < best_loss:
                    best_loss = epoch_loss
                    best_model_wts = copy.deepcopy(classifier.state_dict())
                    early_stop_count = 0
                else:
                    early_stop_count += 1

        if early_stop_count >= early_stop_patience:
            break

    classifier.load_state_dict(best_model_wts)
    classifier.eval()

    # Save model weights
    Path(model_path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(classifier.state_dict(), model_path)

    return classifier

