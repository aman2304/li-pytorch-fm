import math
# Check the train step against the logger steps to determine whether or not to save the model

import torch
import tqdm
from sklearn.metrics import roc_auc_score
from torch.optim.lr_scheduler import OneCycleLR
from torch.utils.data import DataLoader

from torchfm.dataset.avazu import AvazuDataset
from torchfm.dataset.criteo import CriteoDataset
from torchfm.dataset.movielens import MovieLens1MDataset, MovieLens20MDataset
from torchfm.model.afi import AutomaticFeatureInteractionModel
from torchfm.model.afm import AttentionalFactorizationMachineModel
from torchfm.model.dcn import DeepCrossNetworkModel
from torchfm.model.dfm import DeepFactorizationMachineModel
from torchfm.model.ffm import FieldAwareFactorizationMachineModel
from torchfm.model.fm import FactorizationMachineModel
from torchfm.model.fnfm import FieldAwareNeuralFactorizationMachineModel
from torchfm.model.fnn import FactorizationSupportedNeuralNetworkModel
from torchfm.model.hofm import HighOrderFactorizationMachineModel
from torchfm.model.lr import LogisticRegressionModel
from torchfm.model.ncf import NeuralCollaborativeFiltering
from torchfm.model.nfm import NeuralFactorizationMachineModel
from torchfm.model.pnn import ProductNeuralNetworkModel
from torchfm.model.wd import WideAndDeepModel
from torchfm.model.xdfm import ExtremeDeepFactorizationMachineModel
from torchfm.model.afn import AdaptiveFactorizationNetwork


def get_dataset(name, path):
    if name == 'movielens1M':
        return MovieLens1MDataset(path)
    elif name == 'movielens20M':
        return MovieLens20MDataset(path)
    elif name == 'criteo':
        return CriteoDataset(path)
    elif name == 'avazu':
        return AvazuDataset(path)
    else:
        raise ValueError('unknown dataset name: ' + name)


def get_model(name, dataset):
    """
    Hyperparameters are empirically determined, not opitmized.
    """
    field_dims = dataset.field_dims
    if name == 'lr':
        return LogisticRegressionModel(field_dims)
    elif name == 'fm':
        return FactorizationMachineModel(field_dims, embed_dim=16)
    elif name == 'hofm':
        return HighOrderFactorizationMachineModel(field_dims, order=3, embed_dim=16)
    elif name == 'ffm':
        return FieldAwareFactorizationMachineModel(field_dims, embed_dim=4)
    elif name == 'fnn':
        return FactorizationSupportedNeuralNetworkModel(field_dims, embed_dim=16, mlp_dims=(16, 16), dropout=0.2)
    elif name == 'wd':
        return WideAndDeepModel(field_dims, embed_dim=16, mlp_dims=(16, 16), dropout=0.2)
    elif name == 'ipnn':
        return ProductNeuralNetworkModel(field_dims, embed_dim=16, mlp_dims=(16,), method='inner', dropout=0.2)
    elif name == 'opnn':
        return ProductNeuralNetworkModel(field_dims, embed_dim=16, mlp_dims=(16,), method='outer', dropout=0.2)
    elif name == 'dcn':
        return DeepCrossNetworkModel(field_dims, embed_dim=16, num_layers=3, mlp_dims=(16, 16), dropout=0.2)
    elif name == 'nfm':
        return NeuralFactorizationMachineModel(field_dims, embed_dim=64, mlp_dims=(64,), dropouts=(0.2, 0.2))
    elif name == 'ncf':
        # only supports MovieLens dataset because for other datasets user/item colums are indistinguishable
        assert isinstance(dataset, MovieLens20MDataset) or isinstance(dataset, MovieLens1MDataset)
        return NeuralCollaborativeFiltering(field_dims, embed_dim=16, mlp_dims=(16, 16), dropout=0.2,
                                            user_field_idx=dataset.user_field_idx,
                                            item_field_idx=dataset.item_field_idx)
    elif name == 'fnfm':
        return FieldAwareNeuralFactorizationMachineModel(field_dims, embed_dim=4, mlp_dims=(64,), dropouts=(0.2, 0.2))
    elif name == 'dfm':
        return DeepFactorizationMachineModel(field_dims, embed_dim=16, mlp_dims=(16, 16), dropout=0.2)
    elif name == 'xdfm':
        return ExtremeDeepFactorizationMachineModel(
            field_dims, embed_dim=16, cross_layer_sizes=(16, 16), split_half=False, mlp_dims=(16, 16), dropout=0.2)
    elif name == 'afm':
        return AttentionalFactorizationMachineModel(field_dims, embed_dim=16, attn_size=16, dropouts=(0.2, 0.2))
    elif name == 'afi':
        return AutomaticFeatureInteractionModel(
            field_dims, embed_dim=16, atten_embed_dim=64, num_heads=2, num_layers=3, mlp_dims=(400, 400),
            dropouts=(0, 0, 0))
    elif name == 'afn':
        print("Model:AFN")
        return AdaptiveFactorizationNetwork(
            field_dims, embed_dim=16, LNN_dim=1500, mlp_dims=(400, 400, 400), dropouts=(0, 0, 0))
    else:
        raise ValueError('unknown model name: ' + name)


def save_metrics(
        metrics_dict,
        train_loss,
        train_error,
        generalization_loss,
        generalization_error
):
    metrics_dict["train_loss"].append(train_loss)
    metrics_dict["train_error"].append(train_error)
    metrics_dict["generalization_loss"].append(generalization_loss)
    metrics_dict["generalization_error"].append(generalization_error)


"""
* Get a per-step schedule for saving the model
*
* @param initial_steps: the first steps to log per mini-batch
* @param post_interval: log every k steps after initial_steps
* @param total_steps: total steps in the pipeline
"""


def get_log_schedule(
        initial_steps,
        post_interval,
        total_steps,
):
    # If we use 50, we will get logs for steps 0 -> 50, technically 51 steps
    intervals = list(range(initial_steps))
    end_intervals = list(range(initial_steps, total_steps, post_interval))

    intervals.extend(end_intervals)

    log_schedule = [0] * total_steps

    for step in intervals:
        log_schedule[step] = 1

    return log_schedule


class EarlyStopper(object):

    def __init__(self, num_trials, save_path):
        self.num_trials = num_trials
        self.trial_counter = 0
        self.best_accuracy = 0
        self.save_path = save_path

    def is_continuable(self, model, accuracy):
        if accuracy > self.best_accuracy:
            self.best_accuracy = accuracy
            self.trial_counter = 0
            torch.save(model, self.save_path)
            return True
        elif self.trial_counter + 1 < self.num_trials:
            self.trial_counter += 1
            return True
        else:
            return False


def generalization_metrics(model, data_loader, device, criterion):
    model.eval()

    generalization_loss = 0
    correct = 0
    total = 0

    with torch.no_grad():
        for fields, targets in tqdm.tqdm(data_loader, smoothing=0, mininterval=1.0):
            fields, targets = fields.to(device), targets.to(device)
            outputs = model(fields)
            loss = criterion(outputs, targets).mean()
            generalization_loss += loss.item()
            predicted = torch.round(outputs)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

    generalization_error = 1 - (correct / total)

    return generalization_loss, generalization_error


def train_single_step(
        model,
        optimizer,
        inputs,
        targets,
        criterion,
        learning_rate,
):
    outputs = model(inputs)
    loss = criterion(outputs, targets)
    loss.backward()
    optimizer.step()

    if type(learning_rate) != float:
        learning_rate.step()

    return outputs, loss


def train_single_epoch(
        model,
        optimizer,
        train_loader,
        device,
        criterion,
        learning_rate,
        log_schedule,
        train_step,
        logging_path
):
    model.train()
    train_loss = 0
    correct = 0
    total = 0

    optimizer.zero_grad()

    for batch_idx, (inputs, targets) in enumerate(train_loader):
        inputs, targets = inputs.to(device), targets.to(device)

        outputs, loss = train_single_step(model=model, optimizer=optimizer, inputs=inputs, targets=targets,
                                          criterion=criterion, learning_rate=learning_rate)

        if log_schedule[train_step] == 1:
            print(f'Logging @ Step: {train_step}')
            torch.save({
                'step': train_step,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict()
            }, f'{logging_path}_step_{train_step}.pt')

        train_step += 1

        train_loss += loss.item()
        predicted = torch.round(outputs)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

    train_error = 1 - (correct / total)

    return train_loss, train_error, train_step


def test(model, data_loader, device):
    model.eval()
    targets, predicts = list(), list()
    with torch.no_grad():
        for fields, target in tqdm.tqdm(data_loader, smoothing=0, mininterval=1.0):
            fields, target = fields.to(device), target.to(device)
            y = model(fields)
            targets.extend(target.tolist())
            predicts.extend(y.tolist())
    return roc_auc_score(targets, predicts)


def main(dataset_name,
         dataset_path,
         model_name,
         epoch,
         learning_rate,
         warmup_fraction,
         decay,
         batch_size,
         weight_decay,
         initial_log_steps,
         log_k_steps,
         device,
         save_dir):
    device = torch.device(device)
    dataset = get_dataset(dataset_name, dataset_path)
    train_length = int(len(dataset) * 0.8)
    valid_length = int(len(dataset) * 0.1)
    test_length = len(dataset) - train_length - valid_length
    train_dataset, valid_dataset, test_dataset = torch.utils.data.random_split(
        dataset, (train_length, valid_length, test_length))
    train_data_loader = DataLoader(train_dataset, batch_size=batch_size, num_workers=8)
    valid_data_loader = DataLoader(valid_dataset, batch_size=batch_size, num_workers=8)
    test_data_loader = DataLoader(test_dataset, batch_size=batch_size, num_workers=8)
    model = get_model(model_name, dataset).to(device)
    criterion = torch.nn.BCELoss()
    optimizer = torch.optim.Adam(params=model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    train_step = 0

    total_steps = int(len(train_data_loader) * epoch)

    log_schedule = get_log_schedule(
        initial_steps=initial_log_steps,
        post_interval=log_k_steps,
        total_steps=total_steps
    )

    if warmup_fraction is not None or decay is True:

        if warmup_fraction is None:
            warmup_fraction = 0.0
            div_factor = 1.0
        else:
            warmup_fraction = warmup_fraction
            div_factor = 100.0

        if decay is False:
            final_div_factor = 1.0
        else:
            final_div_factor = 10.0

        learning_rate = OneCycleLR(
            optimizer=optimizer,
            max_lr=learning_rate,
            total_steps=total_steps,
            pct_start=warmup_fraction,
            anneal_strategy='linear',
            div_factor=div_factor,
            final_div_factor=final_div_factor
        )
    else:
        learning_rate = learning_rate

    print(learning_rate)

    logging_path = f'logging/{dataset_name}/{model_name}'

    # Do we still want to have this early stopper save?
    early_stopper = EarlyStopper(num_trials=2, save_path=f'{save_dir}/{model_name}_early_stopper.pt')
    for epoch_i in range(epoch):
        train_loss, train_error, train_step = train_single_epoch(model=model, optimizer=optimizer,
                                                                 train_loader=train_data_loader,
                                                                 device=device, criterion=criterion,
                                                                 learning_rate=learning_rate,
                                                                 log_schedule=log_schedule,
                                                                 train_step=train_step,
                                                                 logging_path=logging_path)
        # print(f"Step: {global_step}")

        auc = test(model, valid_data_loader, device)
        print('epoch:', epoch_i, 'validation: auc:', auc)
        if not early_stopper.is_continuable(model, auc):
            print(f'validation: best auc: {early_stopper.best_accuracy}')
            break
    auc = test(model, test_data_loader, device)
    print(f'test auc: {auc}')


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_name', default='criteo')
    parser.add_argument('--dataset_path', help='criteo/train.txt, avazu/train, or ml-1m/ratings.dat')
    parser.add_argument('--model_name', default='afi')
    parser.add_argument('--epoch', type=int, default=100)
    parser.add_argument('--learning_rate', type=float, default=0.001)
    parser.add_argument('--warmup_fraction', type=float, default=None)
    parser.add_argument('--decay', type=bool, default=False)
    parser.add_argument('--batch_size', type=int, default=2048)
    parser.add_argument('--weight_decay', type=float, default=1e-6)
    parser.add_argument('--initial_log_steps', type=int, default=50)
    parser.add_argument('--log_k_steps', type=int, default=50)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--save_dir', default='chkpt')
    args = parser.parse_args()

    main(args.dataset_name,
         args.dataset_path,
         args.model_name,
         args.epoch,
         args.learning_rate,
         args.warmup_fraction,
         args.decay,
         args.batch_size,
         args.weight_decay,
         args.initial_log_steps,
         args.log_k_steps,
         args.device,
         args.save_dir)
