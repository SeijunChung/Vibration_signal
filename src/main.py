import os
import random
import glob
import argparse
import numpy as np
import zipfile
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error, f1_score

import torch
import torchvision
import torchvision.transforms as transforms

import drnn
from models import AttentionalResNet, CNN, DRNN_Classifier, resnet50
from utils import *
from captum.attr import LRP, LayerLRP

# Reproducibility #
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# Device Configuration #
os.environ["CUDA_VISIBLE_DEVICES"] = "7"
device = 'cuda:7' if torch.cuda.is_available() else 'cpu'


def classification(args):
    # Fix Seed #
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    pixel_wise = True

    # Weights and Plots Path #
    paths = [args.weights_path, args.plots_path]
    for path in paths:
        path = path + args.task
        make_dirs(path)

    # Prepare Data #
    classes = {"1st_Normal": 0, "1st_Unbalance": 1, "1st_Looseness": 2, "1st_high": 3, "1st_Bearing": 4,
               "2nd_Unbalance": 5, "2nd_Looseness": 6, "2nd_Bearing": 7, "3rd_Normal": 8, "3rd_Unbalance": 9}

    # Constants #
    best_val_loss = 100
    best_val_improv = 0

    # Prepare Network #
    if args.model == 'cnn':
        model = CNN(args.input_size, args.hidden_size).to(device)
    elif args.model == 'drnn':
        model = DRNN_Classifier(n_inputs=2, n_hidden=256, n_layers=7, n_classes=10).to(device)
    else:
        raise NotImplementedError

    # Lists #
    train_losses, val_losses = list(), list()
    pred_vals, labels = list(), list()

    # Loss Function #
    criterion = torch.nn.CrossEntropyLoss()

    # Optimizer #
    optim = torch.optim.Adam(model.parameters(), lr=args.lr, betas=(0.5, 0.999))
    optim_scheduler = get_lr_scheduler(args.lr_scheduler, optim)

    conv1_channel, conv2_channel, conv3_channel = 32, 64, 128

    conv1_kernelsize, conv2_kernelsize, conv3_kernelsize = 17, 21, 17
    conv1_pad, conv2_pad, conv3_pad = int((conv1_kernelsize - 1) / 2), int((conv2_kernelsize - 1) / 2), int((conv3_kernelsize - 1) / 2)
    conv1_stride, conv2_stride, conv3_stride = 1, 1, 1

    pool1_size, pool2_size, pool3_size = 2, 2, 2
    pool1_pad, pool2_pad, pool3_pad = 0, 0, 0
    pool1_stride, pool2_stride, pool3_stride = 2, 2, 2

    def _next(length, size, pad, stride):
        return int((length + 2 * pad - size + stride) / stride)

    input_len = 128
    l1_conv_len = _next(input_len, conv1_kernelsize, conv1_pad, conv1_stride)
    l1_pool_len = _next(l1_conv_len, pool1_size, pool1_pad, pool1_stride)
    l2_conv_len = _next(l1_pool_len, conv2_kernelsize, conv2_pad, conv2_stride)
    l2_pool_len = _next(l2_conv_len, pool2_size, pool2_pad, pool2_stride)
    l3_conv_len = _next(l2_pool_len, conv3_kernelsize, conv3_pad, conv3_stride)
    l3_pool_len = _next(l3_conv_len, pool3_size, pool3_pad, pool3_stride)
    print(l1_conv_len, l1_pool_len, l2_conv_len, l2_pool_len, l3_conv_len, l3_pool_len)

    Receptive_Field_index3 = []
    for node in range(l3_pool_len):
        l3_conv_start = node * pool3_stride - pool3_pad
        l3_conv_end = node * pool3_stride - pool3_pad + pool3_size - 1
        l2_result_start = l3_conv_start * conv3_stride - conv3_pad
        l2_result_end = l3_conv_end * conv3_stride - conv3_pad + conv3_kernelsize - 1
        l2_conv_start = l2_result_start * pool3_stride - pool2_pad
        l2_conv_end = l2_result_end * pool2_stride - pool2_pad + pool2_size - 1
        l1_result_start = l2_conv_start * conv2_stride - conv2_pad
        l1_result_end = l2_conv_end * conv2_stride - conv2_pad + conv2_kernelsize - 1
        l1_conv_start = l1_result_start * pool1_stride - pool1_pad
        l1_conv_end = l1_result_end * pool1_stride - pool1_pad + pool1_size - 1
        input_start = l1_conv_start * conv1_stride - conv1_pad
        input_end = l1_conv_end * conv1_stride - conv1_pad + conv1_kernelsize - 1
        Receptive_Field_index3.append(np.arange(input_start, input_end + 1))
    print(Receptive_Field_index3)

    # Train and Validation #
    if args.mode == 'train':
        for i, fp in enumerate(glob.glob(args.data_path + "train/*.csv")):
            label = fp.split("/")[-1].split(".")[0][6:]
            df = pd.read_csv(fp)
            data = df[df.columns[1:3]].values
            # print(f"{fp.split('/')[-1].split('.')[0]} length :  {data.shape[0]}")
            hop = int(data.shape[0] // 50000)

            x = split_sequence_classification(data, args.input_size, hop)
            y = np.empty(x.shape[0])
            y[:] = int(classes[label])
            # print(f"The Number of Samples: {label} ~> {x.shape[0]}")
            train_x = x if i == 0 else np.append(train_x, x, axis=0)
            train_y = y if i == 0 else np.append(train_y, y, axis=0)

        print("Data Loading Completed!")
        print(f"X shape: {train_x.shape}, y shape: {train_y.shape}")
        for k, v in classes.items():
            print(f"The Number of labels: {k, v} ~~> {np.where(train_y == v)[0].shape}")

        train_loader, val_loader = data_loader(train_x, train_y, args.train_split, args.batch_size, args.mode, classes)

        # Train #
        print(f"Training {model.__class__.__name__} started with total epoch of {args.num_epochs}.")

        for epoch in range(args.num_epochs):
            model.train()
            for i, (data, label) in enumerate(train_loader):

                # Prepare Data #
                data = data.to(device, dtype=torch.float32)
                label = label.to(device, dtype=torch.long)
                if args.model == "drnn":
                    data = data.transpose(0, 2)
                    data = data.transpose(1, 2)
                # Forward Data #
                pred = model(data)
                # Calculate Loss #
                train_loss = criterion(pred, label)

                # Initialize Optimizer, Back Propagation and Update #
                optim.zero_grad()
                train_loss.backward()
                optim.step()

                # Add item to Lists #
                train_losses.append(train_loss.item())

            # Print Statistics #
            if (epoch+1) % args.print_every == 0:
                print("Epoch [{}/{}]".format(epoch+1, args.num_epochs))
                print("Train Loss {:.4f}".format(np.average(train_losses)))

            # Learning Rate Scheduler #
            optim_scheduler.step()

            # Validation #

            model.eval()
            correct = 0
            total = 0
            with torch.no_grad():
                for i, (data, label) in enumerate(val_loader):

                    # Prepare Data #
                    data = data.to(device, dtype=torch.float32)
                    label = label.to(device, dtype=torch.long)
                    if args.model == "drnn":
                        data = data.transpose(0, 2)
                        data = data.transpose(1, 2)

                    # Forward Data #
                    output = model(data)

                    # Calculate Loss #
                    val_loss = criterion(output, label)
                    val_losses.append(val_loss.item())

                    # Select the max value #
                    _, pred_val = torch.max(output, 1)
                    pred_vals += pred_val.detach().cpu().numpy().tolist()
                    labels += label.detach().cpu().numpy().tolist()
                    total += label.size(0)
                    correct += pred_val.eq(label).sum().item()

            if (epoch + 1) % args.print_every == 0:

                # Print Statistic evaluation #
                print('Loss: %.4f | ACC.: %.3f%% (%d/%d) | F1-Score(Macro): %.6f%%' % (np.average(val_losses), 100. * correct / total, correct, total, f1_score(labels, pred_vals, average="macro")))

                # Save the model only if validation loss decreased #
                curr_val_loss = np.average(val_losses)

                if curr_val_loss < best_val_loss:
                    best_val_loss = min(curr_val_loss, best_val_loss)
                    torch.save(model.state_dict(), os.path.join(args.weights_path, f'Best_{model.__class__.__name__}_model.pkl'))

                    print("Best model is saved!\n")
                    best_val_improv = 0

                elif curr_val_loss >= best_val_loss:
                    best_val_improv += 1
                    print("Best Validation has not improved for {} epochs.\n".format(best_val_improv))

    elif args.mode == 'test':
        pred_tests = []
        # Load the Model Weight #
        model.load_state_dict(torch.load(os.path.join(args.weights_path, f'Best_{model.__class__.__name__}_model.pkl')))
        label = pd.read_csv(args.data_path + "result_test_classification.csv", index_col=False)

        inferenceset = np.load(args.data_path + "test/inference_set_classification.npy")
        # print(inferenceset.shape, label.loc[:, "label"].values.shape)

        test_losses, inf_maes, inf_mses, inf_rmses, labels = [], [], [], [], []
        test_loader = data_loader(inferenceset, label.loc[:, "label"].values, args.train_split, args.batch_size, args.mode)

        # Test #
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for i, (data, label) in enumerate(test_loader):

                # Prepare Data #
                data = data.to(device, dtype=torch.float32)
                label = label.to(device, dtype=torch.long)
                if args.model == "drnn":
                    data = data.transpose(0, 2)
                    data = data.transpose(1, 2)

                # Forward Data #
                output = model(data)

                # Calculate Loss #
                test_loss = criterion(output, label)
                test_losses.append(test_loss.item())

                # Select the max value #
                _, pred_test = torch.max(output, 1)
                pred_test[pred_test == 5] = 1
                pred_test[pred_test == 6] = 2
                pred_test[pred_test == 7] = 4
                pred_test[pred_test == 8] = 0
                pred_test[pred_test == 9] = 1

                pred_tests += pred_test.detach().cpu().numpy().tolist()
                labels += label.detach().cpu().numpy().tolist()

                total += label.size(0)
                correct += pred_test.eq(label).sum().item()

            print('F1 Score %.6f%% | Acc: %.3f%% (%d/%d)' % (f1_score(labels, pred_tests, average="macro"), 100. * correct / total, correct, total))

            # Plot Figure #
            # plot_pred_test(pred_tests[0], labels[0], args.plots_path, args.feature, model, step)

    elif args.mode == 'lrp':
        # Load the Model Weight #
        model.load_state_dict(torch.load(os.path.join(args.weights_path, f'Best_{model.__class__.__name__}_model.pkl')))
        label = pd.read_csv(args.data_path + "result_test_classification.csv", index_col=False)

        inferenceset = np.load(args.data_path + "test/inference_set_classification.npy")
        print(inferenceset.shape, label.loc[:, "label"].values.shape)

        # LRP
        model = model.cpu()

        convlrp = LayerLRP(model, model.conv1)
        convlrp_attr_test = convlrp.attribute(torch.Tensor(inferenceset), target=0, attribute_to_layer_input=True)
        # print(convlrp_attr_test.size())
        plt.figure(figsize=(10, 4))
        x_axis_data = np.arange(convlrp_attr_test[:, 0, :].shape[1])

        y_axis_deconv_lrp_attr_test = convlrp_attr_test[:, 1, :].mean(0).detach().numpy()
        y_axis_deconv_lrp_attr_test = y_axis_deconv_lrp_attr_test / np.linalg.norm(y_axis_deconv_lrp_attr_test, ord=1)
        x_axis_labels = ['{}'.format(i + 1) for i in range(len(y_axis_deconv_lrp_attr_test))]

        ax = plt.subplot()
        ax.set_title('Aggregated neuron importances in the CNN linear layer of the model')
        ax.bar(x_axis_data, y_axis_deconv_lrp_attr_test, align='center', alpha=0.5, color='red')
        ax.autoscale_view()
        plt.tight_layout()

        ax.set_xticks(x_axis_data)
        ax.set_xticklabels(x_axis_labels)
        plt.show()

    else:
        raise NotImplementedError


def regression(args):
    # Fix Seed #
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    # Weights and Plots Path #
    paths = [args.weights_path + args.task, args.plots_path + args.task + "/" + args.model]
    for path in paths:
        make_dirs(path)

    # Prepare Data #
    # fps = [fp for fp in glob.glob(args.data_path + "train/*.csv")]
    # for i, fp in enumerate(fps):
    #     data_temp = pd.read_csv(fp, index_col=False, float_precision="round_trip", dtype=object)
    #     data_temp = data_temp[[f"{fp.split('/')[-1][6:-4]}_c1", f"{fp.split('/')[-1][6:-4]}_c2"]].values.astype(np.float32)
    #     data = data_temp if i == 0 else np.append(data, data_temp, axis=0)
    # print("All Data loaded")
    # with open('data/data.npy', 'wb') as f:
    #     np.save(f, data)

    with open('data/train/data.npy', 'rb') as f:
        data = np.load(f)

    X, y = split_sequence(data, args.input_size, args.output_size, args.hop_length)

    # Split the Dataset #
    print(f"X shape: {X.shape}, y shape: {y.shape}")

    # Make a Multiscale Input
    n_input = 2
    n_hidden = 2
    n_layers = 4
    dropout = 0
    cell_type = 'GRU'

    preprocessing_model = drnn.DRNN(n_input, n_hidden, n_layers, dropout=dropout, cell_type=cell_type).to(device)

    # Lists #
    train_losses, val_losses = list(), list()
    val_maes, val_mses, val_rmses = list(), list(), list()
    pred_tests, labels = list(), list()

    # Constants #
    best_val_loss = 100
    best_val_improv = 0

    # Prepare Network #
    if args.model == 'attn_resnet':
        model = AttentionalResNet(args.input_size, args.output_size).to(device)
    elif args.model == 'resnet50_4th':
        model = resnet50(args.input_size, args.output_size).to(device)
    else:
        raise NotImplementedError

    # Loss Function #
    criterion = torch.nn.MSELoss()

    # Optimizer #
    optim = torch.optim.Adam(model.parameters(), lr=args.lr, betas=(0.5, 0.999))
    optim_scheduler = get_lr_scheduler(args.lr_scheduler, optim)

    # Train and Validation #
    if args.mode == 'train':
        train_loader, val_loader = data_loader(X, y, args.train_split, args.batch_size, args.mode)
        # Train #
        print(f"Training {model.__class__.__name__} started with total epoch of {args.num_epochs}.")

        for epoch in range(args.num_epochs):
            model.train()
            for i, (data, label) in enumerate(train_loader):
                # Prepare Data #
                data = data.to(device, dtype=torch.float32)
                label = label.to(device, dtype=torch.float32)
                data = data.transpose(0, 2)
                data = data.transpose(1, 2)
                _, hidden1 = preprocessing_model(data)
                data, _ = preprocessing_model(data, hidden1)
                data = data.transpose(2, 3)
                data = data.transpose(1, 3)

                # Forward Data #
                pred = model(data)

                # Calculate Loss #
                train_loss = criterion(pred.squeeze(), label.squeeze())

                # Initialize Optimizer, Back Propagation and Update #
                optim.zero_grad()
                train_loss.backward()
                optim.step()

                # Add item to Lists #
                train_losses.append(train_loss.item())

            # Print Statistics #
            if (epoch+1) % args.print_every == 0:
                print("Epoch [{}/{}]".format(epoch+1, args.num_epochs))
                print("Train Loss {:.4f}".format(np.average(train_losses)))

            # Learning Rate Scheduler #
            optim_scheduler.step()

            # Validation #
            model.eval()
            with torch.no_grad():
                for i, (data, label) in enumerate(val_loader):
                    # Prepare Data #
                    data = data.to(device, dtype=torch.float32)
                    label = label.to(device, dtype=torch.float32)
                    data = data.transpose(0, 2)
                    data = data.transpose(1, 2)
                    _, hidden1 = preprocessing_model(data)
                    data, _ = preprocessing_model(data, hidden1)

                    data = data.transpose(1, 2)
                    data = data.transpose(2, 3)

                    # Forward Data #
                    pred_val = model(data)

                    # Calculate Loss #
                    val_loss = criterion(pred_val.squeeze(), label.squeeze())

                    pred_val = pred_val.squeeze().detach().cpu().numpy()
                    label = label.squeeze().detach().cpu().numpy()

                    # Calculate Metrics #
                    val_mae = mean_absolute_error(label, pred_val)
                    val_mse = mean_squared_error(label, pred_val, squared=True)
                    val_rmse = mean_squared_error(label, pred_val, squared=False)

                    # Add item to Lists #
                    val_losses.append(val_loss.item())
                    val_maes.append(val_mae.item())
                    val_mses.append(val_mse.item())
                    val_rmses.append(val_rmse.item())

            if (epoch + 1) % args.print_every == 0:

                # Print Statistic evaluation #
                print("Val Loss {:.4f}".format(np.average(val_losses)))
                print(" MAE : {:.4f}".format(np.average(val_maes)))
                print(" MSE : {:.4f}".format(np.average(val_mses)))
                print(" RMSE : {:.4f}".format(np.average(val_rmses)))

                # Save the model only if validation loss decreased #
                curr_val_loss = np.average(val_losses)

                if curr_val_loss < best_val_loss:
                    best_val_loss = min(curr_val_loss, best_val_loss)
                    torch.save(model.state_dict(), os.path.join(args.weights_path + args.task, f'Best_{model.__class__.__name__}_version_{args.version}_trial_{args.trial}.pkl'))

                    print("Best model is saved!\n")
                    best_val_improv = 0

                elif curr_val_loss >= best_val_loss:
                    best_val_improv += 1
                    print("Best Validation has not improved for {} epochs.\n".format(best_val_improv))

    elif args.mode == 'test':
        submission = pd.read_csv(args.data_path + "sample_submission_regression.csv", index_col=False)
        model.load_state_dict(torch.load(os.path.join(args.weights_path + args.task,
                                                      f'Best_{model.__class__.__name__}_version_{args.version}_trial_{args.trial}.pkl')))
        label = pd.read_csv(args.data_path + "result_test_regression_revised.csv", index_col=False)

        # Forward Data #
        # fps = [fp.split("/")[-1] for fp in glob.glob(args.data_path + "test/*.csv")]
        # one = [fp for fp in fps if int(fp[:-4]) < 10]
        # two = [fp for fp in fps if 10 <= int(fp[:-4]) < 100]
        # three = [fp for fp in fps if 100 <= int(fp[:-4]) < 1000]
        # four = [fp for fp in fps if 1000 <= int(fp[:-4]) < 10000]
        # five = [fp for fp in fps if 10000 <= int(fp[:-4]) < 100000]
        # fps = sorted(one) + sorted(two) + sorted(three) + sorted(four) + sorted(five)
        # print(label["name"].tolist() == fps)

        # c1, c2 = [], []
        # for i, fp in enumerate(fps):
        #     if i % 500 == 0: print(i)
        #     data_temp = pd.read_csv(args.data_path + "test/" + fp, index_col=False, dtype=object)
        #     c1.append(data_temp["c1"].tolist())
        #     c2.append(data_temp["c2"].tolist())
        # test_c1 = np.expand_dims(np.array(c1, dtype=np.float64), 1)
        # test_c2 = np.expand_dims(np.array(c2, dtype=np.float64), 1)
        # testset = np.concatenate((test_c1, test_c2), axis=1)
        # print(test_c1.shape, test_c2.shape, testset.shape)
        # np.save(args.data_path + "test/inference_set.npy", testset)

        inferenceset = np.load(args.data_path + "test/inference_set.npy")

        pred_infers, inf_maes, inf_mses, inf_rmses, labels = [], [], [], [], []

        inf_loader = data_loader(inferenceset, label.loc[:, "t+1":"t+1054"].values, args.train_split, args.test_split, args.batch_size, args.task, args.mode)

        model.eval()
        with torch.no_grad():
            for i, (data, label) in enumerate(inf_loader):
                # Prepare Data #
                data = data.to(device, dtype=torch.float32)
                label = label.to(device, dtype=torch.float32)

                # Forward Data #
                pred_test = model(data)

                pred_tests += pred_test[:, 0, :].detach().cpu().numpy().tolist()
                labels += label.detach().cpu().numpy().tolist()

        # submission = pd.DataFrame(columns=submission.columns, data=np.concatenate((np.array(fps).reshape(-1, 1), np.array(pred_infers)), axis=1))
        # submission.to_csv("./submission_unsorted.csv")
        # submission = submission.sort_values(by=["name"])
        # submission.to_csv("./submission_sorted.csv")

        # print(fps)
        # for fp in submission["name"]:
        #     print(fps.index(fp))
        #     submission.loc[submission["name"] == fp, "t+1":"t+1054"] = np.array(pred_infers)[fps.index(fp), :]
                if i == 1:
                    for j in range(pred_test.shape[0]):
                        plotly_pred_test(pred_test[j], labels[j], model, args.output_size, args.seq_length, args.plots_path + args.task + "/" + args.model, j)
                        plot_pred_test(pred_test[j], labels[j], args.plots_path, model, args.output_size, args.seq_length, j)

        print("Experimental Results of Inference")
        print("RMSE: {:.4f}".format(mean_squared_error(np.array(labels), np.array(pred_tests), squared=False)),"\t",
              "MSE: {:.4f}".format(mean_squared_error(np.array(labels), np.array(pred_tests), squared=True)),"\t",
              "MAE: {:.4f}".format(mean_absolute_error(np.array(labels), np.array(pred_tests))),"\t",
              "MAPE: {:.4f}".format(mean_absolute_percentage_error(np.array(labels), np.array(pred_tests))))

    else:
        raise NotImplementedError


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--trial', type=int, default=1, help='tracking')
    parser.add_argument('--version', type=int, default=1, help='tracking')
    parser.add_argument('--seed', type=int, default=7777, help='seed for reproducibility')
    parser.add_argument('--seq_length', type=int, default=128, help='window size')
    parser.add_argument('--batch_size', type=int, default=256, help='mini-batch size')
    parser.add_argument('--hop_length', type=int, default=128, help='hop length')
    parser.add_argument('--task', type=str, default='classification', choices=['classification', 'regression'])
    parser.add_argument('--mode', type=str, default="train", choices=['train', 'test', "lrp"])
    parser.add_argument('--model', type=str, default='cnn', choices=['dnn', 'cnn', 'drnn', 'resnet50', 'resnet50_2nd', 'resnet50_3rd', 'resnet50_4th', 'attn_resnet'])
    parser.add_argument('--input_size', type=int, default=128, help='input_size')
    parser.add_argument('--hidden_size', type=int, default=2048, help='hidden_size')
    parser.add_argument('--num_layers', type=int, default=4, help='num_layers')
    parser.add_argument('--output_size', type=int, default=10, help='output_size')
    parser.add_argument('--bidirectional', type=bool, default=False, help='use bidirectional or not')
    parser.add_argument('--qkv', type=int, default=5, help='dimension for query, key and value')
    parser.add_argument('--weight_history', type=float, default=0.2)
    parser.add_argument('--weight_context', type=float, default=0.8)
    parser.add_argument('--data_path', type=str, default='../data/', help='which data to use')
    parser.add_argument('--weights_path', type=str, default='../results/weights/', help='weights path')
    parser.add_argument('--plots_path', type=str, default='../results/plots/', help='plots path')
    parser.add_argument('--train_split', type=float, default=0.9, help='train_split')
    parser.add_argument('--test_split', type=float, default=0.1, help='test_split')
    parser.add_argument('--num_epochs', type=int, default=100, help='total epoch')
    parser.add_argument('--print_every', type=int, default=1, help='print statistics for every default epoch')
    parser.add_argument('--lr', type=float, default=1e-3, help='learning rate')
    parser.add_argument('--lr_scheduler', type=str, default='cosine', help='learning rate scheduler', choices=['step', 'plateau', 'cosine'])

    config = parser.parse_args()
    print(config)

    torch.cuda.empty_cache()

    if config.task == "regression":
        regression(config)
    elif config.task == "classification":
        classification(config)
    else:
        raise NotImplementedError