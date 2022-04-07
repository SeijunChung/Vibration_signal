import os, glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.gridspec as gridspec
from matplotlib.pyplot import cm
from PIL import Image
from PIL import ImageDraw
from PIL import ImageFont
from sklearn.metrics import f1_score
from utils import make_dirs, get_lr_scheduler, split_sequence_classification, data_loader
from models import CNN, DRNN_Classifier, GuidedBackpropRelu, GuidedReluModel, CAM
import torch
import torch.nn as nn
from captum.attr import LayerLRP
from sklearn.preprocessing import normalize

# Weights and Plots Path #
paths = ["../results/weights/", "../results/plots/"]

data_path = "../data/"
for path in paths:
    path = path + "task3"
    make_dirs(path)

weights_path = paths[0]
# Prepare Data #
classes = {"1st_Normal": 0, "1st_Unbalance": 1, "1st_Looseness": 2, "1st_high": 3, "1st_Bearing": 4,
           "2nd_Unbalance": 5, "2nd_Looseness": 6, "2nd_Bearing": 7, "3rd_Normal": 8, "3rd_Unbalance": 9}

# Device Configuration #
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
device = 'cuda:7' if torch.cuda.is_available() else 'cpu'

# Constants #
best_val_loss = 100
best_f1_score = 0
best_val_improv = 0
model = "cnn"
lr_scheduler = "cosine"
mode = "gradcam"
train_split, batch_size = 0.9, 256
input_size = 128
conv_channels = [256, 1024, 64]
conv_kernelsizes = [9, 5, 3]

# Prepare Network #
if model == 'cnn':
    model = CNN(input_size, conv_channels, conv_kernelsizes).to(device)
elif model == 'drnn':
    model = DRNN_Classifier(n_inputs=2, n_hidden=256, n_layers=7, n_classes=10).to(device)
else:
    raise NotImplementedError

# Lists #
train_losses, val_losses = list(), list()
pred_vals, labels = list(), list()

# Loss Function #
criterion = torch.nn.CrossEntropyLoss()

# Optimizer #
optim = torch.optim.Adam(model.parameters(), lr=1e-3, betas=(0.5, 0.999))
optim_scheduler = get_lr_scheduler(lr_scheduler, optim)


def select_greedy_protos(K, m):
    ''' selected: an array of selected prototypes '''
    ''' obj_list: a list of the objective values '''

    n = np.shape(K)[0]
    selected = np.array([], dtype=int)
    obj_list = []
    nsk = 0

    colsum = 2 / n * np.sum(K, axis=0)

    for i in range(m):
        argmax = -1
        candidates = np.setdiff1d(range(n), selected)
        vec1 = colsum[candidates]
        lenS = len(selected)

        if lenS > 0:
            temp = K[selected, :][:, candidates]
            vec2 = np.sum(temp, axis=0) * 2 + np.diagonal(K)[candidates]
            vec2 = vec2 / (lenS + 1)
            vec3 = vec1 - vec2
        else:
            vec3 = vec1 - (np.abs(np.diagonal(K)[candidates]))

        ''' vec3: {J(selected U {new})-J(selected)}*(lenS + 1) '''
        ''' increase of the objective value'''
        max_idx = np.argmax(vec3)

        if lenS > 0:
            ''' j: J(selected U {new})'''
            sk = np.sum(K[selected, :][:, selected])
            j = vec3[max_idx] / (lenS + 1) - nsk / (lenS * (lenS + 1)) + (1 / (lenS ** 2) - 1 / ((lenS + 1) ** 2)) * sk
            obj_list.append(j)
        else:
            obj_list.append(vec3[max_idx])

        argmax = candidates[max_idx]
        selected = np.append(selected, argmax)

        ''' nsk: (2/n)*\sum{k([n],S)} '''
        nsk += vec1[max_idx]

    return selected, obj_list


if mode == 'lrp':
    # Load the Model Weight #
    model.load_state_dict(torch.load(os.path.join(weights_path, f'Best_{model.__class__.__name__}_model_2th.pkl')))
    label = pd.read_csv(data_path + "result_test_classification.csv", index_col=False)

    inferenceset = np.load(data_path + "test/inference_set_classification.npy")

    # LRP
    model = model.cpu()

    convlrp = LayerLRP(model, model.conv1)
    convlrp_attr_test = convlrp.attribute(torch.Tensor(inferenceset), target=0, attribute_to_layer_input=True)

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

elif mode == "gradcam":
    data_temp = pd.read_csv("../data/train/train_1st_Normal.csv", index_col=False, float_precision="round_trip", dtype=object)
    data_temp = data_temp[["1st_Normal_c1", "1st_Normal_c2"]].values.astype(np.float32)

    for i, fp in enumerate(glob.glob(data_path + "train/*.csv")):
        label = fp.split("/")[-1].split(".")[0][6:]
        df = pd.read_csv(fp)
        data = df[df.columns[1:3]].values
        # print(f"{fp.split('/')[-1].split('.')[0]} length :  {data.shape[0]}")
        hop = int(data.shape[0] // 1000)

        x = split_sequence_classification(data, input_size, hop)
        y = np.empty(x.shape[0])
        y[:] = int(classes[label])
        print(f"The Number of Samples: {label} ~> {x.shape[0]}")
        train_x = x if i == 0 else np.append(train_x, x, axis=0)
        train_y = y if i == 0 else np.append(train_y, y, axis=0)

    train_loader, val_loader = data_loader(train_x, train_y, train_split, batch_size, "train", classes)

    # Load weights saved #
    model.load_state_dict(torch.load(os.path.join(weights_path, f'Best_{model.__class__.__name__}_model_5th.pkl')))

    # for m in model.modules():
    #     print(m)

    # Validation #
    guided_relu = GuidedBackpropRelu.apply
    guide = GuidedReluModel(model, nn.ReLU, guided_relu)
    cam = CAM(guide)
    guide.reset_output()

    # Validation #
    for i, (data, label) in enumerate(val_loader):
        # Prepare Data #
        data = data.to(device, dtype=torch.float32)
        data.requires_grad = True
        label = label.to(device, dtype=torch.long)
        # Forward Data #
        output = guide.forward(data)
        output = torch.index_select(output, dim=1, index=label)
        output = torch.sum(output)
        output.backward(retain_graph=True)

        for j in range(20):
            out = cam.get_cam(j)
            # print(out.size())
            guided_backprop = guide.get_visual(j)
            # print(guided_img.size())
            cam.visualize(out, guided_backprop, data[j], label[j])

        break


elif mode == 'ptap':
    data_temp = pd.read_csv("../data/train/train_1st_Normal.csv", index_col=False, float_precision="round_trip", dtype=object)
    data_temp = data_temp[["1st_Normal_c1", "1st_Normal_c2"]].values.astype(np.float32)

    # Regression Figure
    # start = 10544*10
    # end = 10544*10 + 1054*4
    # fig, ax = plt.subplots(2, 1, figsize=(7, 3), sharex=True)
    # ax[0].plot(data_temp[start:end, 0])
    # ax[0].set_ylabel("Channel 1", fontsize=12, weight="bold")
    # ax[1].plot(data_temp[start:end, 1])
    # ax[1].set_ylabel("Channel 2", fontsize=12, weight="bold")
    # ax[0].xaxis.set_visible(False)
    # ax[0].set_yticks([])
    # ax[1].set_yticks([])

    # Create a Rectangle patch
    # rect1 = patches.Rectangle((0, -1), 1054*3, 2, linestyle="--", linewidth=3, edgecolor='darkorange', facecolor='none')
    # rect2 = patches.Rectangle((0, -8), 1054*3, 16, linestyle="--", linewidth=3, edgecolor='darkorange', facecolor='none')
    # rect3 = patches.Rectangle((1054 * 3, -8), 1054, 16, linestyle="--", linewidth=3, edgecolor='red', facecolor='lightcoral', fill=True)
    # rect4 = patches.Rectangle((1054 * 3, -1), 1054, 2, linestyle="--", linewidth=2, edgecolor='dodgerblue', facecolor='none')

    # Add the patch to the Axes
    # ax[0].add_patch(rect1)
    # ax[0].add_patch(rect4)
    # ax[1].add_patch(rect2)
    # ax[1].add_patch(rect3)
    # ax[1].text(1054 * 3.45, -2, '?', fontsize=30, weight='bold')
    # plt.tight_layout()
    # plt.savefig("../results/regression.png", dpi=100)

    # Classification Figure
    # start = 128*13
    # end = 128*13 + 128
    # fig, ax = plt.subplots(2, 1, figsize=(7, 3), sharex=True)
    # ax[0].plot(data_temp[start:end, 0])
    # ax[0].set_ylabel("Channel 1", fontsize=12, weight="bold")
    # ax[1].plot(data_temp[start:end, 1])
    # ax[1].set_ylabel("Channel 2", fontsize=12, weight="bold")
    # ax[0].xaxis.set_visible(False)
    # ax[0].set_yticks([])
    # ax[1].set_yticks([])
    # plt.tight_layout()
    # plt.savefig("../results/classification.png", dpi=100)

    # Open an Image
    # img = Image.open('../results/classification.png')
    # Call draw Method to add 2D graphics in an image
    # I1 = ImageDraw.Draw(img)

    # Custom font style and font size
    # myFont = ImageFont.truetype("../Arial.ttf", 150)
    # Add Text to an image
    # I1.text((320, 60), "?", font=myFont, fill=(255, 0, 0), align="center")
    # Save the edited image
    # img.save("../results/classification2.png")
    # quit()

    conv_channels = [256, 1024, 64]
    conv_kernelsizes = [9, 5, 3]
    conv_paddings = [int((conv_kernelsize - 1) / 2) for conv_kernelsize in conv_kernelsizes]
    conv_strides = [1, 1, 1]
    pool_sizes = [2, 2, 2]
    pool_pads = [0, 0, 0]
    pool_strides = [2, 2, 2]


    def _next(length, size, pad, stride):
        return int((length + 2 * pad - size + stride) / stride)

    input_len = 128
    l1_conv_len = _next(input_len, conv_kernelsizes[0], conv_paddings[0], conv_strides[0])
    l1_pool_len = _next(l1_conv_len, pool_sizes[0], pool_pads[0], pool_strides[0])
    l2_conv_len = _next(l1_pool_len, conv_kernelsizes[1], conv_paddings[1], conv_strides[1])
    l2_pool_len = _next(l2_conv_len, pool_sizes[1], pool_pads[1], pool_strides[1])
    l3_conv_len = _next(l2_pool_len, conv_kernelsizes[2], conv_paddings[2], conv_strides[2])
    l3_pool_len = _next(l3_conv_len, pool_sizes[2], pool_pads[2], pool_strides[2])
    # print(l1_conv_len, l1_pool_len, l2_conv_len, l2_pool_len, l3_conv_len, l3_pool_len)

    Receptive_Field_index3 = []
    for node in range(l3_pool_len):
        l3_conv_start = node * pool_strides[2] - pool_pads[2]
        l3_conv_end = node * pool_strides[2] - pool_pads[2] + pool_sizes[2] - 1
        l2_result_start = l3_conv_start * conv_strides[2] - conv_paddings[2]
        l2_result_end = l3_conv_end * conv_strides[2] - conv_paddings[2] + conv_kernelsizes[2] - 1
        l2_conv_start = l2_result_start * pool_strides[2] - pool_pads[1]
        l2_conv_end = l2_result_end * pool_strides[1] - pool_pads[1] + pool_sizes[1] - 1
        l1_result_start = l2_conv_start * conv_strides[1] - conv_paddings[1]
        l1_result_end = l2_conv_end * conv_strides[1] - conv_paddings[1] + conv_kernelsizes[1] - 1
        l1_conv_start = l1_result_start * pool_strides[0] - pool_pads[0]
        l1_conv_end = l1_result_end * pool_strides[0] - pool_pads[0] + pool_sizes[0] - 1
        input_start = l1_conv_start * conv_strides[0] - conv_paddings[0]
        input_end = l1_conv_end * conv_strides[0] - conv_paddings[0] + conv_kernelsizes[0] - 1
        Receptive_Field_index3.append(np.arange(input_start, input_end + 1))

    for i, fp in enumerate(glob.glob(data_path + "train/*.csv")):
        label = fp.split("/")[-1].split(".")[0][6:]
        df = pd.read_csv(fp)
        data = df[df.columns[1:3]].values
        # print(f"{fp.split('/')[-1].split('.')[0]} length :  {data.shape[0]}")
        hop = int(data.shape[0] // 1000)

        x = split_sequence_classification(data, input_size, hop)
        y = np.empty(x.shape[0])
        y[:] = int(classes[label])
        print(f"The Number of Samples: {label} ~> {x.shape[0]}")
        train_x = x if i == 0 else np.append(train_x, x, axis=0)
        train_y = y if i == 0 else np.append(train_y, y, axis=0)

    # train_x = train_x[:6000, :, :]
    # train_y = train_y[:6000]

    train_loader, val_loader = data_loader(train_x, train_y, train_split, batch_size, "train", classes)

    # Load weights saved #
    model.load_state_dict(torch.load(os.path.join(weights_path, f'Best_{model.__class__.__name__}_model.pkl')))

    # Validation #
    model.eval()
    correct = 0
    total = 0
    activation = {}


    def get_activation(name):
        def hook(model, input, output):
            activation[name] = output.detach()

        return hook


    model.max_pool3.register_forward_hook(get_activation('max_pool3'))

    with torch.no_grad():
        for i, (data, label) in enumerate(train_loader):
            # Prepare Data #
            data = data.to(device, dtype=torch.float32)
            label = label.to(device, dtype=torch.long)

            # Forward Data #
            output = model(data)

            intermediate_output = activation['max_pool3'] if i == 0 else torch.cat(
                (intermediate_output, activation['max_pool3']), dim=0)

            # Calculate Loss #
            val_loss = criterion(output, label)
            val_losses.append(val_loss.item())

            # Select the max value #
            _, pred_val = torch.max(output, 1)
            pred_vals += pred_val.detach().cpu().numpy().tolist()
            labels += label.detach().cpu().numpy().tolist()
            total += label.size(0)
            correct += pred_val.eq(label).sum().item()
        curr_f1_score = f1_score(labels, pred_vals, average="macro")
        print(intermediate_output.size())

        # Print Statistic evaluation #
        print('Loss: %.4f | ACC.: %.3f%% (%d/%d) | F1-Score(Macro): %.6f%%' % (
            np.average(val_losses), 100. * correct / total, correct, total,
            f1_score(labels, pred_vals, average="macro")))

    percent = 1  ### top 5%

    threshold = np.percentile(intermediate_output.detach().cpu().numpy(), 100 - percent, [0, 1])
    threshold_bool = (intermediate_output.detach().cpu().numpy() > threshold)

    print(threshold_bool.shape)
    print(train_x.shape)

    pattern_length = np.max([len(x) for x in Receptive_Field_index3])  # 277

    reindex_threshold_bool = []
    for data_idx in range(threshold_bool.shape[0]):  # 1000
        for output_c in range(conv_channels[2]):  # 1
            if len([x for x in threshold_bool[data_idx, output_c, :] if x]):
                index = []
                for idx in [i for i, x in enumerate(threshold_bool[data_idx, output_c, :]) if x]:
                    reindex_threshold_bool.append([data_idx, output_c, idx])

    pattern_idx_df = pd.DataFrame(reindex_threshold_bool, columns=["data_idx", "output_channel", "pattern_xs"])
    groups = pattern_idx_df.groupby(["data_idx", "pattern_xs"])
    pattern_repetitive_idx_df = groups["output_channel"].apply(list).reset_index(name='output_channel')

    print(pattern_repetitive_idx_df)

    TAP = []
    pattern_id = 0

    data_idx = pattern_repetitive_idx_df.loc[:, "data_idx"].values.tolist()
    output_channel = pattern_repetitive_idx_df.loc[:, "output_channel"].values.tolist()
    pattern_xs = pattern_repetitive_idx_df.loc[:, "pattern_xs"].values.tolist()

    for d_idx, output_c, p_xs in zip(data_idx, output_channel, pattern_xs):
        if (Receptive_Field_index3[p_xs][0] >= 0) and (Receptive_Field_index3[p_xs][-1] < train_x.shape[2]):
            pattern_dict = {}
            pattern_dict["pattern_id"] = pattern_id
            pattern_dict["data_idx"] = d_idx
            pattern_dict["output_channel"] = output_c
            pattern_dict["pattern_xs"] = Receptive_Field_index3[p_xs]
            pattern_dict["pattern_ys"] = train_x[d_idx, :, Receptive_Field_index3[p_xs]].T
            pattern_dict["features"] = intermediate_output[d_idx, :, p_xs].detach().cpu().numpy()
            pattern_dict["activations"] = threshold_bool[d_idx, :, p_xs]
            TAP.append(pattern_dict)
            pattern_id += 1

    for i, x in enumerate(TAP):
        subsequences = np.expand_dims(x['pattern_ys'], 0) if i == 0 else np.append(subsequences,
                                                                                   np.expand_dims(x['pattern_ys'], 0),
                                                                                   axis=0)

    # display_n = 32
    # column = 4
    # row = int((display_n - 1) / column) + 1

    # fig = plt.figure(figsize=(10, row * 1.5))
    # gs = gridspec.GridSpec(row, column)
    # gs.update(wspace=0, hspace=0)

    # samples = np.random.choice(len(subsequences), display_n, replace=False)
    # for i, n in enumerate(samples):
    #     ax = plt.subplot(gs[i])
    #     ax.plot(subsequences[n, 0, :], color='black', alpha=2.5)
    #     ax.plot(subsequences[n, 1, :], color='blue', alpha=2.5)
    #     ax.set_ylim(-5, 5)
    #     ax.set_xticks([])
    #     ax.set_yticks([])
    #     plt.setp(ax.spines.values(), linewidth=2)
    # plt.suptitle("Patterns Before Clustering", y=1.05, fontsize=18)
    # plt.tight_layout()
    # plt.savefig('../results/plots/' + 'subsequence.png', dpi=100, bbox_inches='tight')  # 그림 저장
    # plt.show()

    data_indices = np.array([x['data_idx'] for x in TAP])
    # subsequences = np.array([x['pattern_ys'] for x in TAP])
    features = np.array([x['features'] for x in TAP])

    powers = 20
    pat = np.array(features)
    pat = pat / (np.linalg.norm(pat, axis=1).reshape(-1, 1))
    pat = normalize(pat, norm='l2')

    gram_kernel = np.power(np.inner(pat, pat), powers)

    # Gram kernel
    m = 10  ## the number of prototype
    selected, obj_list = select_greedy_protos(gram_kernel, m)
    # print(selected.shape)
    print(gram_kernel[:, selected].shape)

    color = cm.rainbow(np.linspace(0, 1, m))[::-1]
    classified = np.argmax(gram_kernel[:, selected], axis=1)
    (unique, counts) = np.unique(classified, return_counts=True)
    print(unique, counts)

    yrange = (-3.5, 3.5)
    protos = subsequences[selected[:m]]

    # ro = int((m - 1) / 12) + 1
    # column = 16
    # # fig = plt.figure(figsize=(25,3*row))
    # figs, axes = plt.subplots(4, 3, figsize=(7, ro * 6))
    # gs = gridspec.GridSpec(ro, column)
    # gs.update(wspace=0, hspace=0)
    #
    # for n in range(m):
    #     group_idx = [i for i, x in enumerate(classified) if x == n]
    #     members = subsequences[classified == n]
    #     # proto_std = members.std(axis=0)
    #     print("n:", n, "members:", members.shape, "protos", protos[n].shape)
    #
    #     if n < 3:
    #         row = 0
    #         col = n % 3
    #     elif n < 6:
    #         row = 1
    #         col = n % 3
    #     elif n < 9:
    #         row = 2
    #         col = n % 3
    #     elif n < 12:
    #         row = 3
    #         col = n % 3
    #
    #     for d in range(2):
    #         axes[row, col].plot(members[:, d, :].mean(axis=0), color='black', alpha=0.6)
    #         axes[row, col].plot(protos[n][d, :], color=color[n], alpha=1, linewidth=2)
    #     axes[row, col].set_ylim(yrange)
    #
    #     axes[row, col].set_xticks([])
    #     axes[row, col].set_yticks([])
    #     plt.setp(axes[row, col].spines.values(), linewidth=1.5)
    #
    # plt.tight_layout()
    # plt.savefig('../results/plots/' + 'Prototypes.png', dpi=100, bbox_inches='tight')  # 그림 저장
    # plt.show()

    # proto_idx = 1
    # display_n = 9
    #
    # group_idx = [i for i, x in enumerate(classified) if x == proto_idx]
    # members = subsequences[classified == proto_idx]
    # proto_std = members.std(axis=0)
    # proto_mean = members.mean(axis=0)
    #
    # plt.figure(figsize=(2, 2))
    # for d in range(2):
    #     plt.plot(protos[proto_idx][d], color=color[proto_idx], alpha=1, linewidth=2)
    # plt.ylim(-5, 5)
    # plt.xticks([])
    # plt.yticks([])
    # plt.title("Prototype {}".format(proto_idx))
    # plt.show()
    #
    # fig, ax= plt.subplots(1, display_n, figsize=(1.7 * display_n, 2))
    # gs = gridspec.GridSpec(1, display_n)
    # gs.update(wspace=0, hspace=0)
    #
    # indices = np.random.choice(len(members), display_n, replace=False)
    # for t, member in enumerate(members[indices]):
    #     # ax = plt.subplot(gs[t])
    #     for d in range(2):
    #         ax[t].plot(member[d], linewidth=1.5, color='black')
    #
    #     ax[t].set_ylim(yrange)
    #     ax[t].set_xticks([])
    #     ax[t].set_yticks([])
    #     plt.setp(ax[t].spines.values(), linewidth=2)
    #
    # fig.suptitle("PTAPs in Prototype {}".format(proto_idx), y=1.1, fontsize=18)
    # plt.tight_layout()
    # plt.show()

    data_indices = np.array([x['data_idx'] for x in TAP])
    subsequences = np.array([x['pattern_ys'] for x in TAP])
    subseq_x = np.array([x['pattern_xs'] for x in TAP])
    features = np.array([x['features'] for x in TAP])

    tar_classes = range(10)
    for tar_class in tar_classes:
        tar_set = np.where(train_y == tar_class)[0]

        ####################### This part is to choose patterns randomly without overlapping. #############
        idxes = np.random.choice(tar_set, 70)

        # idx = 199
        for idx in idxes:
            pattern_list = np.where(data_indices == idx)[0]
            pattern_list = pattern_list[np.argsort(-np.max(gram_kernel[pattern_list, :], axis=1))]

            pattern_list2 = []
            for i, p in enumerate(pattern_list):
                if i == 0:
                    pattern_list2.append(p)
                else:
                    add = True
                    for p2 in pattern_list2:
                        if np.abs(subseq_x[p][0] - subseq_x[p2][0]) < 150:
                            add = False
                            break
                    if add:
                        pattern_list2.append(p)

            # print(subsequences.shape)
            # print(subseq_x.shape)
            ########################################################################################################

            fig, ax = plt.subplots(2, 1, figsize=(8, 3), sharex=True)
            for k in range(2):
                ax[k].plot(train_x[idx, k, :], c='black')

                corres_p_list = []
                for p_idx in pattern_list2:
                    proto_idx = classified[p_idx]
                    corres_p_list.append(proto_idx)

                    members = subsequences[classified == proto_idx]
                    # proto_dim1_std = members[:, 0, :].std(axis=0) * 1.5
                    # proto_dim2_std = members[:, 0, :].std(axis=0) * 1.5
                    # proto_dim1_mean = members[:, 1, :].mean(axis=0)
                    # proto_dim2_mean = members[:, 1, :].mean(axis=0)

                    ax[k].fill_between(subseq_x[p_idx, :],
                                       members[:, k, :].mean(axis=0) - members[:, k, :].std(axis=0) * 1.5,
                                       members[:, k, :].mean(axis=0) + members[:, k, :].std(axis=0) * 1.5,
                                       color=color[proto_idx], alpha=0.2)
                    ax[k].plot(subseq_x[p_idx, :], protos[proto_idx][k], c=color[proto_idx], linewidth=5, alpha=0.75)
            fig.suptitle("Data {} | Class {}".format(idx, tar_class), fontsize=15)
            plt.xticks([])
            # plt.show()
            plt.savefig(f"../results/plots/classification/target_{tar_class}/Prototypeswithts_{idx}.png", dpi=100,
                        bbox_inches='tight')  # 그림 저장
            # plt.ylim(-3, 3)

else:
    raise NotImplementedError