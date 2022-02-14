import os
import sys
import time
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import plotly.graph_objects as go
import torch
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder


def make_dirs(path):
    """Make Directory If not Exists"""
    if not os.path.exists(path):
        os.makedirs(path)


def load_data(data):
    """Data Loader"""
    data_dir = os.path.join(data)

    data = pd.read_csv(data_dir,
                       # infer_datetime_format=True,
                       parse_dates=['date']
                       )

    data.index = data['date']
    data = data.drop('date', axis=1)

    return data


def split_sequence_classification(sequence, n_steps, hop):
    X = list()
    sequence = np.transpose(sequence)
    for i in range(0, sequence.shape[1], hop):
        end_ix = i + n_steps

        if end_ix > sequence.shape[1]-1:
            break

        seq_x = sequence[:, i:end_ix]

        X.append(seq_x)

    return np.array(X)


def split_sequence(sequence, n_steps_in, n_steps_out, hop):
    X, y = list(), list()
    sequence = np.transpose(sequence)
    sequence_len = sequence.shape[1]

    for i in range(0, sequence_len, hop):
        end_idx = i + n_steps_in
        out_end_idx = end_idx + n_steps_out

        if out_end_idx > sequence_len - 1:
            break

        seq_x = sequence[..., i:end_idx]
        seq_context = sequence[:1, end_idx:out_end_idx]
        seq_y = sequence[1:, end_idx:out_end_idx]

        empty = np.empty((1, seq_context.shape[1]))
        empty[:] = np.NaN
        seq_context = np.concatenate((seq_context, empty), axis=0)
        seq_x = np.concatenate((seq_x, seq_context), axis=1)

        X.append(seq_x)
        y.append(seq_y)

    return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)


def split_sequence_step(sequence, n_steps_in, n_steps_out, n_steps, hop):
    X, y = list(), list()
    sequence = np.transpose(sequence)
    sequence_len = sequence.shape[1]

    for i in range(0, sequence_len, hop):
        steps_x = []

        if i + n_steps_in + n_steps_out + n_steps < sequence_len - 1:
            for j in range(n_steps):
                start_idx = i + j
                end_idx = start_idx + n_steps_in
                out_end_idx = end_idx + n_steps_out

                seq_x = sequence[..., start_idx:end_idx]
                seq_context = sequence[:1, end_idx:out_end_idx]

                empty = np.empty((1, seq_context.shape[1]))
                empty[:] = np.NaN
                seq_context = np.concatenate((seq_context, empty), axis=0)
                seq_x = np.concatenate((seq_x, seq_context), axis=1)

                steps_x.append(seq_x)
                if j == n_steps-1:
                    seq_y = sequence[1:, end_idx:out_end_idx]
            X.append(np.array(steps_x, dtype=np.float32))
            y.append(seq_y)

    return np.array(X), np.array(y)


def data_loader(x, y, train_split, batch_size, mode, classes=None):
    """Prepare data by applying sliding windows and return data loader"""

    if mode == 'train':
        # Split to Train, Validation and Test Set #
        train_seq, test_seq, train_label, test_label = train_test_split(x, y, train_size=train_split, shuffle=True, random_state=77)
        # Convert to Tensor #
        # for k, v in classes.items():
        #     print(f"The Number of labels in Train set: {k} {v} ~~> {np.where(train_label == v)[0].shape}")
        # for k, v in classes.items():
        #     print(f"The Number of labels in Test set: {k} {v} ~~> {np.where(test_label == v)[0].shape}")

        train_set = TensorDataset(torch.from_numpy(train_seq), torch.from_numpy(train_label))
        test_set = TensorDataset(torch.from_numpy(test_seq), torch.from_numpy(test_label))

        # Data Loader #
        train_loader = DataLoader(train_set, batch_size=batch_size, drop_last=False, shuffle=True)
        test_loader = DataLoader(test_set, batch_size=batch_size, drop_last=False, shuffle=True)

        return train_loader, test_loader

    elif mode == 'test':
        # Convert to Tensor #
        inference_set = TensorDataset(torch.from_numpy(x), torch.from_numpy(y))
        return DataLoader(inference_set, batch_size=batch_size, drop_last=False, shuffle=False)


def get_lr_scheduler(lr_scheduler, optimizer):
    """Learning Rate Scheduler"""
    if lr_scheduler == 'step':
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)
    elif lr_scheduler == 'plateau':
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.2, threshold=0.01, patience=5)
    elif lr_scheduler == 'cosine':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10, eta_min=0)
    else:
        raise NotImplementedError
    return scheduler

# def train(model, dataloader, parameters, loss_function, args, device):
#     # Optimizer #
#     optim = torch.optim.Adam(model.parameters(), lr=parameters.get("lr", args.lr), betas=(0.5, 0.999))
#     optim_scheduler = get_lr_scheduler(args.lr_scheduler, optim)

def percentage_error(actual, predicted):
    """Percentage Error"""
    res = np.empty(actual.shape)
    for j in range(actual.shape[0]):
        if actual[j] != 0:
            res[j] = (actual[j] - predicted[j]) / actual[j]
        else:
            res[j] = predicted[j] / np.mean(actual)
    return res


def mean_percentage_error(y_true, y_pred):
    """Mean Percentage Error"""
    mpe = np.mean(percentage_error(np.asarray(y_true), np.asarray(y_pred))) * 100
    return mpe


def mean_absolute_percentage_error(y_true, y_pred):
    """Mean Absolute Percentage Error"""
    mape = np.mean(np.abs(percentage_error(np.asarray(y_true), np.asarray(y_pred)))) * 100
    return mape


def plot_pred_test(pred, actual, path, feature, model, output_size, seq_length, iteration):
    """Plot Test set Prediction"""
    plt.figure(figsize=(10, 6))

    plt.plot(np.arange(len(pred)), pred, label='Pred', color="red", alpha=0.8)
    plt.plot(np.arange(len(actual)), actual, label='Actual', color="blue", alpha=0.5)

    plt.xlabel('Time', fontsize=18)
    plt.ylabel('{}'.format(feature), fontsize=18)
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)

    plt.legend(loc='best', fontsize=18)
    plt.grid()
    # plt.tight_layout()

    plt.title('Using {} predict {} with before {}'.format(model.__class__.__name__, output_size, seq_length), fontsize=18)
    plt.savefig(os.path.join(path, 'Using {} predict {} with before {} at {}.png'.format(model.__class__.__name__, output_size, seq_length, iteration)), dpi=100)


def plotly_pred_test(pred, label, model, output_size, seq_length, path, iteration):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=[i for i in range(len(pred))], y=pred, name='pred'))
    fig.add_trace(go.Scatter(x=[i for i in range(len(label))], y=label, name='label'))
    fig.update_layout(title_text='Using {} predict {} with before {}'.format(model.__class__.__name__, output_size, seq_length))

    fig.write_html(os.path.join(path, f'Using {model.__class__.__name__} predict {output_size} with before {seq_length} at {iteration}.html'))
