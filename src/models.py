import torch
import torch.nn as nn
import drnn
import torchvision
import torch.autograd as autograd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import torch.optim as optim
import torch.utils.data as Data
from convolution_lstm import ConvLSTM
from torch.autograd import Function
from skimage.transform import resize
from sklearn.preprocessing import MinMaxScaler


device = 'cuda:7' if torch.cuda.is_available() else 'cpu'


def minmax_norm(x):
    return (x - x.min()) / x.max() - x.min()


class GuidedBackpropRelu(Function):
    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        return input.clamp(min=0)

    @staticmethod
    def backward(ctx, grad_output):
        input = ctx.saved_tensors[0]
        grad_input = grad_output.clone()
        grad_input[grad_input < 0] = 0
        grad_input[input < 0] = 0
        return grad_input


class GuidedReluModel(nn.Module):
    def __init__(self, model, to_be_replaced, replace_to):
        super(GuidedReluModel, self).__init__()
        self.model = model
        self.to_be_replaced = to_be_replaced
        self.replace_to = replace_to
        self.layers = []
        self.output = []

        for i, m in enumerate(self.model.modules()):
            if isinstance(m, self.to_be_replaced):
                self.layers.append(self.replace_to)
            elif isinstance(m, nn.Conv1d):
                self.layers.append(m)
            elif isinstance(m, nn.BatchNorm1d):
                self.layers.append(m)
            elif isinstance(m, nn.Linear):
                self.layers.append(m)
            elif isinstance(m, nn.AvgPool2d):
                self.layers.append(m)
            elif isinstance(m, nn.MaxPool1d):
                self.layers.append(m)

    def reset_output(self):
        self.output = []

    def hook(self, grad):
        out = grad.cpu().data
        self.output.append(out)

    def get_visual(self, idx):
        print("gradient:", self.output[0].size())
        grad = self.output[0][idx]
        return grad

    def forward(self, x):
        out = x
        print("self.hook:", self.hook)
        out.register_hook(self.hook)
        for i, m in enumerate(self.layers):
            print(f"layer {i+1}:", m)
        for ind, i in enumerate(self.layers[:-1]):
            out = i(out)
        out = out.view(out.size()[0], -1)
        out = self.layers[-1](out)
        return out


class CAM():
    def __init__(self, model):
        self.gradient = []
        self.model = model
        self.h = self.model.model.layer[-1].register_backward_hook(self.save_gradient)

    def save_gradient(self, *args):
        grad_input = args[1]
        grad_output = args[2]
        self.gradient.append(grad_output[0])

    def get_gradient(self):
        return self.gradient[0]

    def remove_hook(self):
        self.h.remove()

    def normalize_cam(self, x):
        x = 2 * (x - torch.min(x)) / (torch.max(x) - torch.min(x) + 1e-8) - 1
        # x[x<torch.max(x)]=-1
        return x

    def normalize_gradcam(self, x):
        x = 2 * (x - np.min(x)) / (np.max(x) - np.min(x) + 1e-8) - 1
        # x[x<torch.max(x)]=-1
        return x

    def visualize(self, cam_img, guided_img, time_series, label):
        guided_img = guided_img.numpy()
        cam_img = resize(cam_img.cpu().data.numpy(), output_shape=(128,))
        x = time_series.cpu().data.numpy()
        y = label.cpu().numpy()
        # print("cam_img:", np.min(cam_img), np.max(cam_img))
        # print("guided_img:", guided_img[0], np.min(guided_img), np.max(guided_img))
        # print("cam X Guided_img.shape:", (guided_img * cam_img).shape)

        fig, ax = plt.subplots(7, 1, figsize=(12, 10), sharex=True)
        ax[0].set_title(f"Original time-series, Label {y}")
        ax[0].plot(x[0])
        ax[0].set_ylabel("Channel 1", fontsize=12, weight="bold")
        ax[1].plot(x[1])
        ax[1].set_ylabel("Channel 2", fontsize=12, weight="bold")
        ax[0].xaxis.set_visible(False)
        ax[0].set_yticks([])
        ax[1].set_yticks([])

        norm = matplotlib.colors.Normalize(vmin=np.min(cam_img), vmax=np.max(cam_img))
        ax[2].set_title("Class Activation Map")
        ax[2].scatter(np.arange(0, len(cam_img)), cam_img, color=plt.cm.bwr(norm(cam_img)), edgecolor='none')
        ax[2].set_ylabel("CAM", fontsize=12, weight="bold")
        ax[2].xaxis.set_visible(False)
        ax[2].set_yticks([])

        norm1 = matplotlib.colors.Normalize(vmin=np.min(guided_img[0]), vmax=np.max(guided_img[0]))
        norm2 = matplotlib.colors.Normalize(vmin=np.min(guided_img[1]), vmax=np.max(guided_img[1]))
        ax[3].set_title("Guided Backpropagation")
        ax[3].scatter(np.arange(0, len(guided_img[0])), guided_img[0], color=plt.cm.Greys(norm1(guided_img[0])), edgecolor='none')
        ax[3].set_ylabel("Channel 1", fontsize=12, weight="bold")
        ax[4].scatter(np.arange(0, len(guided_img[1])), guided_img[1], color=plt.cm.Greys(norm2(guided_img[1])), edgecolor='none')
        ax[4].set_ylabel("Channel 2", fontsize=12, weight="bold")
        ax[3].xaxis.set_visible(False)
        ax[3].set_yticks([])
        ax[4].set_yticks([])

        gradcam = cam_img * guided_img

        gradcam_norm1 = self.normalize_gradcam(gradcam[0])
        gradcam_norm2 = self.normalize_gradcam(gradcam[1])

        norm1 = matplotlib.colors.Normalize(vmin=np.min(gradcam_norm1), vmax=np.max(gradcam_norm1))
        norm2 = matplotlib.colors.Normalize(vmin=np.min(gradcam_norm2), vmax=np.max(gradcam_norm2))

        ax[5].set_title("Guided x CAM")
        ax[5].plot(np.arange(0, len(x[0])), x[0])
        # ax[5].scatter(np.arange(0, len(x[0])), x[0], color=plt.cm.OrRd(norm1(x[0])), edgecolor='none', alpha=0.5)
        x_space = np.linspace(0, 128, 257)
        print(x_space)

        for i in range(128):
            if i == 0:
                ax[5].fill_between((x_space[2*i], x_space[2*i+1]), (-1, 1), color=plt.cm.OrRd(norm1(x[0]))[i])
            elif i == 127:
                ax[5].fill_between((x_space[2*i], x_space[2*i+1]), (-1, 1), color=plt.cm.OrRd(norm1(x[0]))[i])
            else:
                print((x_space[2*i-1], x_space[2*i+2]))
                ax[5].fill_between((x_space[2*i-1], x_space[2*i+2]), (-1, 1), color=plt.cm.OrRd(norm1(x[0]))[i])
        ax[5].set_ylabel("Channel 1", fontsize=12, weight="bold")
        ax[6].plot(np.arange(0, len(x[1])), x[1])
        # ax[6].scatter(np.arange(0, len(x[1])), x[1], color=plt.cm.OrRd(norm2(x[1])), edgecolor='none', alpha=0.5)
        ax[6].set_ylabel("Channel 2", fontsize=12, weight="bold")
        ax[5].xaxis.set_visible(False)
        ax[5].set_yticks([])
        ax[6].set_yticks([])
        plt.tight_layout()
        plt.show()

    def get_cam(self, idx):
        grad = self.get_gradient()
        # print("grad size:", grad.size())
        alpha = torch.sum(grad, dim=2, keepdim=True)
        # print("alpha size:", alpha.size())
        # alpha = torch.sum(alpha, dim=1, keepdim=True)

        cam = alpha[idx] * grad[idx]
        cam = torch.sum(cam, dim=0)
        cam = self.normalize_cam(cam)
        # print("cam size:", cam)
        self.remove_hook()

        return cam


class DRNN_Classifier(nn.Module):

    def __init__(self, n_inputs, n_hidden, n_layers, n_classes, cell_type="GRU"):
        super(DRNN_Classifier, self).__init__()

        self.drnn = drnn.DRNN(n_inputs, n_hidden, n_layers, dropout=0, cell_type=cell_type)
        self.linear = nn.Linear(n_hidden, n_classes)

    def forward(self, inputs):
        _, _, layer_outputs = self.drnn(inputs)
        print(layer_outputs[-1].size())
        pred = self.linear(layer_outputs[-1])

        return pred


def attention(ConvLstm_out):
    attention_w = []
    for k in range(5):
        attention_w.append(torch.sum(torch.mul(ConvLstm_out[k], ConvLstm_out[-1]))/5)
    m = nn.Softmax()
    attention_w = torch.reshape(m(torch.stack(attention_w)), (-1, 5))

    cl_out_shape = ConvLstm_out.shape
    ConvLstm_out = torch.reshape(ConvLstm_out, (5, -1))
    convLstmOut = torch.matmul(attention_w, ConvLstm_out)
    convLstmOut = torch.reshape(convLstmOut, (cl_out_shape[1], cl_out_shape[2]))
    return convLstmOut


class LinearRegression(nn.Module):
    def __init__(self, input_size, output_size, w1, w2):
        super(LinearRegression, self).__init__()
        self.w1 = w1
        self.w2 = w2
        self.output_size = output_size
        self.flatten = nn.Flatten(start_dim=1, end_dim=2)
        self.linear1 = nn.Linear(2*input_size, output_size)
        self.linear2 = nn.Linear(output_size, output_size)

    def forward(self, x):
        x1 = x[:, :2, :]
        x2 = x[:, 2:, -self.output_size:]
        x1 = self.flatten(x1)
        x2 = self.flatten(x2)

        out1 = self.linear1(x1)
        out2 = self.linear2(x2)
        out = torch.mul(out1, self.w1) + torch.mul(out2, self.w2)
        return out.unsqueeze(1)


class DNN(nn.Module):
    """Deep Neural Network"""
    def __init__(self, input_size, hidden_dim, output_size, w1, w2):
        super(DNN, self).__init__()
        self.w1 = w1
        self.w2 = w2
        self.input_size = input_size
        self.hidden_size = hidden_dim
        self.output_size = output_size
        self.num_channels = 2

        self.layer1 = nn.Sequential(
            nn.Flatten(start_dim=1, end_dim=2),
            nn.Linear(self.num_channels*input_size, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, output_size)
        )

        self.layer2 = nn.Sequential(
            nn.Flatten(start_dim=1, end_dim=2),
            nn.Linear(output_size, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, output_size)
        )

    def forward(self, x):
        x1 = x[:, :2, :]
        x2 = x[:, 2:, -self.output_size:]

        out1 = self.layer1(x1)
        out2 = self.layer2(x2)
        out = torch.mul(out1, self.w1) + torch.mul(out2, self.w2)

        return out.unsqueeze(1)


class CNN(nn.Module):
    """Convolutional Neural Networks"""
    def __init__(self, input_size, hidden_dims, kernel_sizes):
        super(CNN, self).__init__()
        self.input_size = input_size
        self.hidden_dims = hidden_dims
        self.layer = nn.Sequential(
            nn.Conv1d(in_channels=2, out_channels=self.hidden_dims[0], kernel_size=kernel_sizes[0], padding=int((kernel_sizes[0]-1)/2)),
            nn.BatchNorm1d(self.hidden_dims[0]),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(in_channels=self.hidden_dims[0], out_channels=self.hidden_dims[1], kernel_size=kernel_sizes[1], padding=int((kernel_sizes[1]-1)/2)),
            nn.BatchNorm1d(self.hidden_dims[1]),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(in_channels=self.hidden_dims[1], out_channels=self.hidden_dims[2], kernel_size=kernel_sizes[2], padding=int((kernel_sizes[2]-1)/2)),
            nn.BatchNorm1d(self.hidden_dims[2]),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.AvgPool2d((64, 1))
        )
        self.fc_layer = nn.Linear(16, 10, bias=True)

    def forward(self, x):
        out = self.layer(x)
        out = self.fc_layer(out.squeeze(1))
        return out


# class CNN(nn.Module):
#     """Convolutional Neural Networks"""
#     def __init__(self, input_size, hidden_dims, kernel_sizes):
#         super(CNN, self).__init__()
#         self.input_size = input_size
#         self.hidden_dims = hidden_dims
#         self.conv1 = nn.Conv1d(in_channels=2, out_channels=self.hidden_dims[0], kernel_size=kernel_sizes[0], padding=int((kernel_sizes[0]-1)/2))
#         self.bn1 = nn.BatchNorm1d(self.hidden_dims[0])
#         self.max_pool1 = nn.MaxPool1d(2)
#         self.conv2 = nn.Conv1d(in_channels=self.hidden_dims[0], out_channels=self.hidden_dims[1], kernel_size=kernel_sizes[1], padding=int((kernel_sizes[1]-1)/2))
#         self.bn2 = nn.BatchNorm1d(self.hidden_dims[1])
#         self.max_pool2 = nn.MaxPool1d(2)
#         self.conv3 = nn.Conv1d(in_channels=self.hidden_dims[1], out_channels=self.hidden_dims[2], kernel_size=kernel_sizes[2], padding=int((kernel_sizes[2]-1)/2))
#         self.bn3 = nn.BatchNorm1d(self.hidden_dims[2])
#         self.max_pool3 = nn.MaxPool1d(2)
#         self.avg_pool = nn.AvgPool1d(64)
#         self.flatten = nn.Flatten()
#         self.fc = nn.Linear(16, 10, bias=True)
#         self.relu = nn.ReLU()
#
#     def forward(self, x):
#         out = self.conv1(x)
#         out = self.bn1(out)
#         out = self.relu(out)
#         out = self.max_pool1(out)
#         out = self.conv2(out)
#         out = self.bn2(out)
#         out = self.relu(out)
#         out = self.max_pool2(out)
#         out = self.conv3(out)
#         out = self.bn3(out)
#         out = self.relu(out)
#         out = self.max_pool3(out)
#         # print(out.size())
#         out = self.avg_pool(out.transpose(1,2))
#         out = self.flatten(out)
#         out = self.fc(out.squeeze(1))
#         return out


class Conv_LSTM(nn.Module):
    def __init__(self):
        super(Conv_LSTM, self).__init__()
        self.conv1_lstm = ConvLSTM(input_channels=128, hidden_channels=[128],
                                   kernel_size=3, step=5, effective_step=[4])
        self.conv2_lstm = ConvLSTM(input_channels=512, hidden_channels=[512],
                                   kernel_size=3, step=5, effective_step=[4])
        self.conv3_lstm = ConvLSTM(input_channels=1024, hidden_channels=[1024],
                                   kernel_size=3, step=5, effective_step=[4])
        # self.conv4_lstm = ConvLSTM(input_channels=256, hidden_channels=[256],
        #                            kernel_size=3, step=5, effective_step=[4])

    def forward(self, conv1_out, conv2_out, conv3_out):
        conv1_lstm_out = torch.stack([attention(self.conv1_lstm(conv1_out[:, i, :, :])[0][0]) for i in range(conv1_out.size(1))])
        conv2_lstm_out = torch.stack([attention(self.conv2_lstm(conv2_out[:, i, :, :])[0][0]) for i in range(conv2_out.size(1))])
        conv3_lstm_out = torch.stack([attention(self.conv3_lstm(conv3_out[:, i, :, :])[0][0]) for i in range(conv3_out.size(1))])
        # print("\n")
        # print(conv1_lstm_out.size(), conv2_lstm_out.size(), conv3_lstm_out.size())
        return conv1_lstm_out.unsqueeze(0), conv2_lstm_out.unsqueeze(0), conv3_lstm_out.unsqueeze(0)


class CnnDecoder(nn.Module):
    def __init__(self, in_channels):
        super(CnnDecoder, self).__init__()
        self.deconv3 = nn.Sequential(
            nn.ConvTranspose1d(in_channels, 512, 2, 2),
            nn.SELU()
        )
        self.deconv2 = nn.Sequential(
            nn.ConvTranspose1d(1024, 128, 4, 4),
            nn.SELU()
        )
        self.deconv1 = nn.Sequential(
            nn.ConvTranspose1d(256, 32, 4, 4, 1),
            nn.SELU()
        )
        self.deconv0 = nn.Sequential(
            nn.ConvTranspose1d(32, 1, 3, 1, 1),
            nn.SELU()
        )

    def forward(self, conv1_out, conv2_out, conv3_out):
        deconv3 = self.deconv3(conv3_out)
        # print("deconv3 output in Decorder:", deconv3.size())
        deconv3_concat = torch.cat((deconv3, conv2_out), dim=1)
        # print("concat (deconv3, conv2_out) in Decorder:", deconv3_concat.size())
        deconv2 = self.deconv2(deconv3_concat)
        # print("deconv2 output in Decorder:", deconv2.size())
        deconv2_concat = torch.cat((deconv2, conv1_out), dim=1)
        # print("concat (deconv2, conv1_out) in Decorder:", deconv2_concat.size())
        deconv1 = self.deconv1(deconv2_concat)
        # print("deconv1 output in Decorder:", deconv1.size())
        output = self.deconv0(deconv1)
        return output


class BottleNeck(nn.Module):
    """Residual block for resnet over 50 layers
    """
    expansion = 4

    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()

        self.residual_function = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv1d(out_channels, out_channels, stride=stride, kernel_size=9, padding=4, bias=False),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv1d(out_channels, out_channels * BottleNeck.expansion, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm1d(out_channels * BottleNeck.expansion),
        )

        self.shortcut = nn.Sequential()

        if stride != 1 or in_channels != out_channels * BottleNeck.expansion:
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_channels, out_channels * BottleNeck.expansion, stride=stride, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm1d(out_channels * BottleNeck.expansion)
            )

    def forward(self, x):
        return nn.ReLU(inplace=True)(self.residual_function(x) + self.shortcut(x))


class ResNet1d_Encoder(nn.Module):

    def __init__(self, block, num_block, input_size, output_size):
        super(ResNet1d_Encoder, self).__init__()

        self.in_channels = 32
        self.input_size = input_size
        self.output_size = output_size
        self.conv1 = Conv(2, 32, 3, 1)
        self.conv2 = Conv(32, 128, 3, 1)
        self.conv3 = Conv(128, 512, 3, 1)
        self.conv4 = Conv(512, 2048, 3, 1)
        self.conv1_x = self._make_layer(block, 32, num_block[0], 3)
        self.conv2_x = self._make_layer(block, 128, num_block[1], 4)
        self.conv3_x = self._make_layer(block, 512, num_block[2], 4)
        self.conv4_x = self._make_layer(block, 1024, num_block[3], 2)
        self.avg_pool = nn.AvgPool2d((128, 1))
        self.max_pool = nn.MaxPool2d((4, 1))
        self.layers = nn.Sequential(
            nn.BatchNorm1d(2),
            nn.Conv1d(in_channels=2, out_channels=32, kernel_size=3, padding=1, bias=False)
        )

    def _make_layer(self, block, out_channels, num_blocks, stride):
        """
        make resnet layers(by layer i didnt mean this 'layer' was the
        same as a neuron network layer, ex. conv layer), one layer may
        contain more than one residual block

        Args:
            block: block type, basic block or bottle neck block
            out_channels: output depth channel number of this layer
            num_blocks: how many blocks per layer
            stride: the stride of the first block of this layer

        Return:
            return a resnet layer
        """
        # we have num_block blocks per layer, the first block could be 1 or 2, other blocks would always be 1
        # print("num_blocks:", num_blocks)

        strides = [stride] + [1] * (num_blocks - 1)
        # print(f"{block.__name__} ~ strides:", strides)
        layers = []
        for stride in strides:
            # print(f"{block.__name__} ~ in_channel:", self.in_channels, "out_channels:", out_channels, "stride:", stride)
            layers.append(block(self.in_channels, out_channels, stride))
            self.in_channels = out_channels * block.expansion

        return nn.Sequential(*layers)

    def forward(self, x):
        x1 = x[..., :self.input_size]
        # print("input1 size:", x1.size())
        x2 = x[..., :1, self.input_size:]
        # print("input2 size:", x2.size())
        output = torch.stack([self.conv1(x1[n]) for n in range(x1.size(0))])
        # output = self.conv1(x1)
        # print("conv1 output:", output.size())
        output = torch.stack([self.conv1_x(output[n]) for n in range(output.size(0))])
        # output = self.conv1_x(output)
        # print("Resnet Module 1 output:", output.size())
        output = torch.stack([self.avg_pool(output[n]) for n in range(output.size(0))])
        # output = self.avg_pool(output)
        # print("avg_pool output:", output.size())
        output = torch.cat((output, x2), dim=2)
        # print("concat output:", output.size())
        output = torch.stack([self.layers(output[n]) for n in range(output.size(0))])
        # output = self.layers(output)
        # print("layers output:", output.size())
        output = torch.stack([self.conv2(output[n]) for n in range(output.size(0))])
        # output = self.conv2(output)
        # print("conv2 output:", output.size())
        output1 = torch.stack([self.conv2_x(output[n]) for n in range(output.size(0))])
        # output1 = self.conv2_x(output)
        # print("Resnet Module 2 output:", output1.size())
        output1 = torch.stack([self.max_pool(output1[n]) for n in range(output1.size(0))])
        # output1 = self.max_pool(output1)
        # print("max_pool output:", output1.size())
        output2 = torch.stack([self.conv3(output1[n]) for n in range(output1.size(0))])
        # output2 = self.conv3(output1)
        # print("conv3 output:", output2.size())
        output2 = torch.stack([self.conv3_x(output2[n]) for n in range(output2.size(0))])
        # output2 = self.conv3_x(output2)
        # print("Resnet Module 3 output:", output2.size())
        output2 = torch.stack([self.max_pool(output2[n]) for n in range(output2.size(0))])
        # output2 = self.max_pool(output2)
        # print("max_pool output:", output2.size())
        output3 = torch.stack([self.conv4(output2[n]) for n in range(output2.size(0))])
        # output3 = self.conv4(output2)
        # print("conv4 output:", output3.size())
        output3 = torch.stack([self.conv4_x(output3[n]) for n in range(output3.size(0))])
        # output3 = self.conv4_x(output3)
        # print("Resnet Module 4 output:", output3.size())
        output3 = torch.stack([self.max_pool(output3[n]) for n in range(output3.size(0))])
        # output3 = self.max_pool(output3)
        # print("Outputs", output1.size(), output2.size(), output3.size())

        return output1, output2, output3


class AttentionalResNet(nn.Module):
    def __init__(self, input_size, output_size):
        super(AttentionalResNet, self).__init__()
        self.resnet_encoder = ResNet1d_Encoder(BottleNeck, [3, 4, 6, 3], input_size, output_size)
        self.conv_lstm = Conv_LSTM()
        self.cnn_decoder = CnnDecoder(1024)

    def forward(self, x):
        conv1_out, conv2_out, conv3_out = self.resnet_encoder(x)
        # print("Outputs", conv1_out.size(), conv2_out.size(), conv3_out.size())
        conv1_lstm_out, conv2_lstm_out, conv3_lstm_out = self.conv_lstm(conv1_out, conv2_out, conv3_out)
        # print("Outputs", conv1_lstm_out.squeeze(0).size(),  conv2_lstm_out.squeeze(0).size(), conv3_lstm_out.squeeze(0).size())
        gen_x = self.cnn_decoder(conv1_lstm_out.squeeze(0), conv2_lstm_out.squeeze(0), conv3_lstm_out.squeeze(0))
        # print(gen_x.size())

        return gen_x


class ResNet1d_4th(nn.Module):

    def __init__(self, block, num_block, input_size, output_size):
        super(ResNet1d_4th, self).__init__()

        self.in_channels = 32
        self.input_size = input_size
        self.output_size = output_size
        self.conv1 = Conv(2, 32, 3, 1)
        self.conv2 = Conv(64, 128, 3, 1)
        self.conv3 = Conv(128, 256, 3, 1)
        self.conv4 = Conv(256, 512, 3, 1)
        self.conv5 = Conv(512, 1024, 3, 1)
        self.conv1_x = self._make_layer(block, 32, num_block[0], 3)
        self.conv2_x = self._make_layer(block, 64, num_block[1], 2)
        self.conv3_x = self._make_layer(block, 128, num_block[2], 2)
        self.conv4_x = self._make_layer(block, 256, num_block[3], 1)
        self.avg_pool = nn.AvgPool2d((128, 1))
        self.max_pool = nn.MaxPool2d((2, 1))
        self.max_pool_2 = nn.MaxPool1d(2, ceil_mode=True)
        self.layers = nn.Sequential(
            nn.BatchNorm1d(2),
            nn.Conv1d(in_channels=2, out_channels=64, kernel_size=3, padding=1, bias=False)
        )

        self.decoder = self._make_layer(block=Deconvolution, out_channels=128, num_blocks=4, stride=1)
        self.conv5 = nn.Sequential(
            nn.Conv1d(in_channels=512, out_channels=128, kernel_size=1, bias=False),
            nn.BatchNorm1d(128),
            nn.Conv1d(in_channels=128, out_channels=32, kernel_size=3, bias=False),
            nn.BatchNorm1d(32),
            nn.Conv1d(in_channels=32, out_channels=8, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm1d(8),
            nn.Conv1d(in_channels=8, out_channels=1, kernel_size=3, stride=2, bias=False)
        )

    def _make_layer(self, block, out_channels, num_blocks, stride):
        """
        make resnet layers(by layer i didnt mean this 'layer' was the
        same as a neuron network layer, ex. conv layer), one layer may
        contain more than one residual block

        Args:
            block: block type, basic block or bottle neck block
            out_channels: output depth channel number of this layer
            num_blocks: how many blocks per layer
            stride: the stride of the first block of this layer

        Return:
            return a resnet layer
        """
        # we have num_block blocks per layer, the first block could be 1 or 2, other blocks would always be 1
        # print("in_channel:", self.in_channels)
        # print("num_blocks:", num_blocks)
        if block.__name__ == "Deconvolution":
            self.in_channels = 896
        strides = [stride] + [1] * (num_blocks - 1)
        # print(f"{block.__name__} ~ strides:", strides)
        layers = []
        for stride in strides:
            # print(f"{block.__name__} ~ in_channel:", self.in_channels, "out_channels:", out_channels, "stride:", stride)
            layers.append(block(self.in_channels, out_channels, stride))
            self.in_channels = out_channels * block.expansion

        return nn.Sequential(*layers)

    def forward(self, x):
        x1 = x[..., :self.input_size]
        # print("input1 size:", x1.size())
        x2 = x[:, :1, self.input_size:]
        # print("input2 size:", x2.size())
        output = self.conv1(x1)
        # print("conv1 output:", output.size())
        output = self.conv1_x(output)
        # print("Resnet Module 1 output:", output.size())
        output = self.avg_pool(output)
        # print("avg_pool output:", output.size())
        output = torch.cat((output, x2), dim=1)
        # print("concat output:", output.size())
        output = self.layers(output)
        # print("layers output:", output.size())
        output = self.conv2(output)
        output1 = self.conv2_x(output)
        # print("Resnet Module 2 output:", output1.size())
        output1 = self.max_pool(output1)
        # print("max_pool output:", output1.size())
        output2 = self.conv3(output1)
        # print("output2:", output2.size())
        output2 = self.conv3_x(output2)
        # print("Resnet Module 3 output:", output2.size())
        output1 = self.max_pool_2(output1)
        # print("max_pool_2 output:", output1.size())
        output2 = self.max_pool(output2)
        # print("max_pool output:", output2.size())
        output3 = self.conv4(output2)
        output3 = self.conv4_x(output3)
        # print("Resnet Module 4 output:", output3.size())
        # output2 = self.max_pool_2(output2)
        # print("max_pool_2 output:", output2.size())
        output3 = self.max_pool(output3)
        # output3 = self.max_pool_2(output3)
        # print("Before concatenate", output1.size(), output2.size(), output3.size())
        output = torch.cat((output1, output2, output3), dim=1)
        # print("After concatenate", output.size())
        output = self.max_pool_2(output)
        output = self.decoder(output)
        # print("decoder output", output.size())
        output = self.conv5(output)
        # print("conv5 output", output.size())
        return output


class ResNet1d_3rd(nn.Module):

    def __init__(self, block, num_block, input_size, output_size):
        super(ResNet1d_3rd, self).__init__()

        self.in_channels = 64
        self.input_size = input_size
        self.output_size = output_size
        self.conv1 = Conv(2, 64, 3, 1)
        self.conv2 = Conv(64, 256, 3, 1)
        self.conv3 = Conv(128, 512, 3, 1)
        self.conv4 = Conv(256, 1024, 3, 1)
        self.conv1_x = self._make_layer(block, 64, num_block[0], 3)
        self.conv2_x = self._make_layer(block, 128, num_block[1], 2)
        self.conv3_x = self._make_layer(block, 256, num_block[2], 1)
        self.conv4_x = self._make_layer(block, 512, num_block[3], 1)
        self.avg_pool = nn.AvgPool2d((256, 1))
        self.max_pool = nn.MaxPool2d((4, 1))
        self.max_pool_2 = nn.MaxPool1d(4, ceil_mode=True)
        self.layers = nn.Sequential(
            nn.BatchNorm1d(2),
            nn.Conv1d(in_channels=2, out_channels=64, kernel_size=3, padding=1, bias=False),
            # nn.MaxPool2d((16, 1))
        )

        self.decoder = self._make_layer(block=Deconvolution, out_channels=64, num_blocks=4, stride=1)
        self.conv5 = nn.Sequential(
            nn.Conv1d(in_channels=256, out_channels=128, kernel_size=1, bias=False),
            nn.BatchNorm1d(128),
            nn.Conv1d(in_channels=128, out_channels=32, kernel_size=3, bias=False),
            nn.BatchNorm1d(32),
            nn.Conv1d(in_channels=32, out_channels=8, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm1d(8),
            nn.Conv1d(in_channels=8, out_channels=1, kernel_size=3, stride=2, bias=False)
        )

    def _make_layer(self, block, out_channels, num_blocks, stride):
        """
        make resnet layers(by layer i didnt mean this 'layer' was the
        same as a neuron network layer, ex. conv layer), one layer may
        contain more than one residual block

        Args:
            block: block type, basic block or bottle neck block
            out_channels: output depth channel number of this layer
            num_blocks: how many blocks per layer
            stride: the stride of the first block of this layer

        Return:
            return a resnet layer
        """
        # we have num_block blocks per layer, the first block could be 1 or 2, other blocks would always be 1
        # print("num_blocks:", num_blocks)
        if block.__name__ == "Deconvolution":
            self.in_channels = 896
        strides = [stride] + [1] * (num_blocks - 1)
        # print(f"{block.__name__} ~ strides:", strides)
        layers = []
        for stride in strides:
            # print(f"{block.__name__} ~ in_channel:", self.in_channels, "out_channels:", out_channels, "stride:", stride)
            layers.append(block(self.in_channels, out_channels, stride))
            self.in_channels = out_channels * block.expansion

        return nn.Sequential(*layers)

    def forward(self, x):
        x1 = x[..., :self.input_size]
        # print("input1 size:", x1.size())
        x2 = x[:, :1, self.input_size:]
        # print("input2 size:", x2.size())
        output = self.conv1(x1)
        # print("conv1 output:", output.size())
        output = self.conv1_x(output)
        # print("Resnet Module 1 output:", output.size())
        output = self.avg_pool(output)
        # print("avg_pool output:", output.size())
        output = torch.cat((output, x2), dim=1)
        # print("concat output:", output.size())
        output = self.layers(output)
        # print("layers output:", output.size())
        output = self.conv2(output)
        output1 = self.conv2_x(output)
        # print("Resnet Module 2 output:", output1.size())
        output1 = self.max_pool(output1)
        # print("max_pool output:", output1.size())
        output2 = self.conv3(output1)
        output2 = self.conv3_x(output2)
        # print("Resnet Module 3 output:", output2.size())
        output1 = self.max_pool_2(output1)
        # print("max_pool_2 output:", output1.size())
        output2 = self.max_pool(output2)
        # print("max_pool output:", output2.size())
        output3 = self.conv4(output2)
        output3 = self.conv4_x(output3)
        # print("Resnet Module 4 output:", output3.size())
        output2 = self.max_pool_2(output2)
        # print("max_pool_2 output:", output2.size())
        output3 = self.max_pool(output3)
        output3 = self.max_pool_2(output3)
        # print("Before concatenate", output1.size(), output2.size(), output3.size())
        output = torch.cat((output1, output2, output3), dim=1)
        # print("After concatenate", output.size())
        output = self.decoder(output)
        # print("decoder output", output.size())
        output = self.conv5(output)
        # print("conv5 output", output.size())
        return output


class ResNet1d_2nd(nn.Module):

    def __init__(self, block, num_block, input_size, output_size):
        super(ResNet1d_2nd, self).__init__()

        self.in_channels = 64
        self.input_size = input_size
        self.output_size = output_size
        self.conv1 = Conv(2, 64, 3, 1)
        self.conv2 = Conv(64, 192, 3, 1)
        self.conv3 = Conv(128, 384, 3, 1)
        self.conv4 = Conv(128, 384, 3, 1)
        self.conv1_x = self._make_layer(block, 64, num_block[0], 3)
        self.conv2_x = self._make_layer(block, 128, num_block[1], 2)
        self.conv3_x = self._make_layer(block, 128, num_block[2], 1)
        self.conv4_x = self._make_layer(block, 128, num_block[3], 1)
        self.avg_pool = nn.AvgPool2d((192, 1))
        self.max_pool = nn.MaxPool2d((3, 1))
        self.max_pool_2 = nn.MaxPool1d(4, ceil_mode=True)
        self.layers = nn.Sequential(
            nn.BatchNorm1d(2),
            nn.Conv1d(in_channels=2, out_channels=64, kernel_size=3, padding=1, bias=False),
            # nn.MaxPool2d((16, 1))
        )

        self.decoder = self._make_layer(block=Deconvolution, out_channels=64, num_blocks=4, stride=1)
        self.conv5 = nn.Sequential(
            nn.Conv1d(in_channels=256, out_channels=128, kernel_size=1, bias=False),
            nn.BatchNorm1d(128),
            nn.Conv1d(in_channels=128, out_channels=32, kernel_size=3, bias=False),
            nn.BatchNorm1d(32),
            nn.Conv1d(in_channels=32, out_channels=8, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm1d(8),
            nn.Conv1d(in_channels=8, out_channels=1, kernel_size=3, stride=2, bias=False)
        )

    def _make_layer(self, block, out_channels, num_blocks, stride):
        """
        make resnet layers(by layer i didnt mean this 'layer' was the
        same as a neuron network layer, ex. conv layer), one layer may
        contain more than one residual block

        Args:
            block: block type, basic block or bottle neck block
            out_channels: output depth channel number of this layer
            num_blocks: how many blocks per layer
            stride: the stride of the first block of this layer

        Return:
            return a resnet layer
        """
        # we have num_block blocks per layer, the first block could be 1 or 2, other blocks would always be 1
        # print("num_blocks:", num_blocks)

        strides = [stride] + [1] * (num_blocks - 1)
        # print(f"{block.__name__} ~ strides:", strides)
        layers = []
        for stride in strides:
            # print(f"{block.__name__} ~ in_channel:", self.in_channels, "out_channels:", out_channels, "stride:", stride)
            layers.append(block(self.in_channels, out_channels, stride))
            self.in_channels = out_channels * block.expansion

        return nn.Sequential(*layers)

    def forward(self, x):
        x1 = x[..., :self.input_size]
        # print("input1 size:", x1.size())
        x2 = x[:, :1, self.input_size:]
        # print("input2 size:", x2.size())
        output = self.conv1(x1)
        # print("conv1 output:", output.size())
        output = self.conv1_x(output)
        # print("Resnet Module 1 output:", output.size())
        output = self.avg_pool(output)
        # print("avg_pool output:", output.size())
        output = torch.cat((output, x2), dim=1)
        # print("concat output:", output.size())
        output = self.layers(output)
        # print("layers output:", output.size())
        output = self.conv2(output)
        output1 = self.conv2_x(output)
        # print("Resnet Module 2 output:", output1.size())
        output1 = self.max_pool(output1)
        # print("max_pool output:", output1.size())
        output2 = self.conv3(output1)
        output2 = self.conv3_x(output2)
        # print("Resnet Module 3 output:", output2.size())
        output1 = self.max_pool_2(output1)
        # print("max_pool_2 output:", output1.size())
        output2 = self.max_pool(output2)
        # print("max_pool output:", output1.size())
        output3 = self.conv4(output2)
        output3 = self.conv4_x(output3)
        # print("Resnet Module 4 output:", output3.size())
        output2 = self.max_pool_2(output2)
        # print("max_pool_2 output:", output2.size())
        output3 = self.max_pool(output3)
        output3 = self.max_pool_2(output3)
        # print("Before concatenate", output1.size(), output2.size(), output3.size())
        output = torch.cat((output1, output2, output3), dim=1)
        # print("After concatenate", output.size())
        output = self.decoder(output)
        # print("decoder output", output.size())
        output = self.conv5(output)
        # print("conv5 output", output.size())
        return output


class ResNet1d(nn.Module):

    def __init__(self, block, num_block, input_size, output_size):
        super(ResNet1d, self).__init__()

        self.in_channels = 128
        self.input_size = input_size
        self.output_size = output_size
        self.conv1 = Conv(2, 128, 3, 1)
        self.conv2 = Conv(2, 384, 3, 1)
        self.conv3 = Conv(2, 384, 3, 1)
        self.conv4 = Conv(2, 384, 3, 1)
        self.conv1_x = self._make_layer(block, 128, num_block[0], 3)
        self.conv2_x = self._make_layer(block, 128, num_block[1], 2)
        self.conv3_x = self._make_layer(block, 128, num_block[2], 1)
        self.conv4_x = self._make_layer(block, 128, num_block[3], 1)
        self.max_pool_1 = nn.MaxPool2d((384, 1))
        self.max_pool = nn.MaxPool2d((192, 1))
        self.max_pool_2 = nn.MaxPool1d(4, ceil_mode=True)
        self.layers = nn.Sequential(
            nn.BatchNorm1d(2),
            nn.Conv1d(in_channels=2, out_channels=32, kernel_size=3, padding=1, bias=False),
            nn.MaxPool2d((16, 1))
        )
        self.decoder = self._make_layer(block=Deconvolution, out_channels=3, num_blocks=4, stride=1)
        self.conv5 = nn.Sequential(
            nn.Conv1d(in_channels=12, out_channels=4, kernel_size=3, bias=False),
            nn.BatchNorm1d(4),
            nn.Conv1d(in_channels=4, out_channels=1, kernel_size=3, stride=2, bias=False)
        )

    def _make_layer(self, block, out_channels, num_blocks, stride):
        """
        make resnet layers(by layer i didnt mean this 'layer' was the
        same as a neuron network layer, ex. conv layer), one layer may
        contain more than one residual block

        Args:
            block: block type, basic block or bottle neck block
            out_channels: output depth channel number of this layer
            num_blocks: how many blocks per layer
            stride: the stride of the first block of this layer

        Return:
            return a resnet layer
        """
        # we have num_block blocks per layer, the first block
        # could be 1 or 2, other blocks would always be 1
        # print("num_blocks:", num_blocks)
        if block.__name__ == "Deconvolution":
            self.in_channels = 6
        strides = [stride] + [1] * (num_blocks - 1)
        # print(f"{block.__name__} ~ strides:", strides)
        layers = []
        for stride in strides:
            # print(f"{block.__name__} ~ in_channel:", self.in_channels, "out_channels:", out_channels, "stride:", stride)
            layers.append(block(self.in_channels, out_channels, stride))
            self.in_channels = out_channels * block.expansion

        return nn.Sequential(*layers)

    def forward(self, x):
        x1 = x[..., :self.input_size]
        x2 = x[:, :1, self.input_size:]
        output = self.conv1(x1)
        output = self.conv1_x(output)
        output = self.max_pool_1(output)
        output = torch.cat((output, x2), dim=1)
        output = self.layers(output)
        output = self.conv2(output)
        output1 = self.conv2_x(output)
        output1 = self.max_pool(output1)
        output2 = self.conv3(output1)
        output2 = self.conv3_x(output2)
        output1 = self.max_pool_2(output1)
        output2 = self.max_pool(output2)
        output3 = self.conv4(output2)
        output3 = self.conv4_x(output3)
        output2 = self.max_pool_2(output2)
        output3 = self.max_pool(output3)
        output3 = self.max_pool_2(output3)
        output = torch.cat((output1, output2, output3), dim=1)
        output = self.decoder(output)
        output = self.conv5(output)
        return output


class Deconvolution(nn.Module):
    """Deconvolution block"""
    expansion = 4

    def __init__(self, in_channels, out_channels, stride=1):
        super(Deconvolution, self).__init__()
        # print(in_channels, out_channels, "in deconvolution")

        self.deconvolution = nn.Sequential(
            nn.ConvTranspose1d(in_channels, out_channels, 2, 2, 0, 0),
            nn.BatchNorm1d(out_channels),
            nn.SELU(inplace=True),
            nn.Conv1d(out_channels, out_channels * Deconvolution.expansion, stride=stride, kernel_size=9, padding=4, bias=False),
            nn.BatchNorm1d(out_channels * Deconvolution.expansion),
        )

    def forward(self, x):
        return nn.SELU(inplace=True)(self.deconvolution(x))


class Conv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, padding):
        super(Conv, self).__init__()
        self.residual_function = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm1d(out_channels),
            nn.Conv1d(out_channels, out_channels, kernel_size=9, padding=4, bias=False),
            nn.BatchNorm1d(out_channels),
            nn.Conv1d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm1d(out_channels)
        )

        self.shortcut = nn.Sequential(
            nn.Conv1d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, padding=padding),
            nn.BatchNorm1d(out_channels))

    def forward(self, x):
        return self.residual_function(x) + self.shortcut(x)


class BasicBlock(nn.Module):
    """Basic Block for resnet 18 and resnet 34
    """
    # BasicBlock and BottleNeck block have different output size
    # we use class attribute expansion to distinct
    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()

        # residual function
        self.residual_function = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels * BasicBlock.expansion, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels * BasicBlock.expansion)
        )

        # shortcut
        self.shortcut = nn.Sequential()

        # the shortcut output dimension is not the same with residual function
        # use 1*1 convolution to match the dimension
        if stride != 1 or in_channels != BasicBlock.expansion * out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels * BasicBlock.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels * BasicBlock.expansion)
            )

    def forward(self, x):
        return nn.ReLU(inplace=True)(self.residual_function(x) + self.shortcut(x))


def resnet18(input_size, output_size):
    """ return a ResNet 18 object
    """
    return ResNet1d(BasicBlock, [2, 2, 2, 2], input_size, output_size)


def resnet34(input_size, output_size):
    """ return a ResNet 34 object
    """
    return ResNet1d(BasicBlock, [3, 4, 6, 3], input_size, output_size)


def resnet50(input_size, output_size):
    """ return a ResNet 50 object
    """
    return ResNet1d_4th(BottleNeck, [3, 4, 6, 3], input_size, output_size)


def resnet101(input_size, output_size):
    """ return a ResNet 101 object
    """
    return ResNet1d(BottleNeck, [3, 4, 23, 3], input_size, output_size)


def resnet152(input_size, output_size):
    """ return a ResNet 152 object
    """
    return ResNet1d(BottleNeck, [3, 8, 36, 3], input_size, output_size)


if __name__ == '__main__':

    pixel_wise = True
    data_dir = './MNIST_data'
    n_classes = 10

    cell_type = "GRU"
    n_hidden = 20
    if pixel_wise:
        n_layers = 9
    else:
        n_layers = 4

    batch_size = 256
    learning_rate = 1.0e-3
    training_iters = 10
    display_step = 25

    train_data = torchvision.datasets.MNIST(root=data_dir,
                                            train=True,
                                            transform=torchvision.transforms.ToTensor(),
                                            download=True)

    test_data = torchvision.datasets.MNIST(root=data_dir, train = False)
    test_x = test_data.data.type(torch.FloatTensor)[:2000] / 255.0
    if pixel_wise:
        test_x = test_x.view(test_x.size(0), 784).unsqueeze(2).transpose(1, 0)
    else:
        test_x = test_x.transpose(1, 0)

    test_y = test_data.targets[:2000]

    test_x = test_x.to(device)
    test_y = test_y.to(device)

    train_loader = Data.DataLoader(train_data, batch_size, shuffle=False, num_workers=1)

    print("==> Building a dRNN with %s cells" %cell_type)

    if pixel_wise:
        model = DRNN_Classifier(1, n_hidden, n_layers, n_classes, cell_type=cell_type)
    else:
        model = DRNN_Classifier(28, n_hidden, n_layers, n_classes, cell_type=cell_type)

    model.to(device)

    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()

    for iter in range(training_iters):
        for step, (batch_x, batch_y) in enumerate(train_loader):
            batch_x = batch_x.view(-1, 28, 28)

            if pixel_wise:
                batch_x = batch_x.view(batch_size, 784).unsqueeze(2).transpose(1, 0)
            else:
                batch_x = batch_x.transpose(1, 0)

            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)

            optimizer.zero_grad()
            pred = model(batch_x)

            loss = criterion(pred, batch_y)

            loss.backward()
            optimizer.step()

            if (step + 1) % display_step == 0:
                print("Iter " + str(iter + 1) + ", Step " + str(step+1) +", Average Loss: " + "{:.6f}".format(loss.item()))

        test_output = model(test_x)
        pred_y = torch.max(test_output, 1)[1].data.squeeze()
        accuracy = sum(pred_y == test_y) / float(test_y.size(0))

        print("========> Validation Accuracy: {:.6f}".format(accuracy))

    print("end")
