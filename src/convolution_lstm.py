import torch
import torch.nn as nn
from torch.autograd import Variable

device = "cuda:7" if torch.cuda.is_available else "cpu"


class ConvLSTMCell(nn.Module):
    def __init__(self, input_channels, hidden_channels, kernel_size):
        super(ConvLSTMCell, self).__init__()

        assert hidden_channels % 2 == 0

        self.input_channels = input_channels
        self.hidden_channels = hidden_channels
        self.kernel_size = kernel_size
        self.num_features = 4

        self.padding = int((kernel_size - 1) / 2)

        self.Wxi = nn.Conv1d(self.input_channels, self.hidden_channels, self.kernel_size, 1, self.padding, bias=True)
        self.Whi = nn.Conv1d(self.hidden_channels, self.hidden_channels, self.kernel_size, 1, self.padding, bias=False)
        self.Wxf = nn.Conv1d(self.input_channels, self.hidden_channels, self.kernel_size, 1, self.padding, bias=True)
        self.Whf = nn.Conv1d(self.hidden_channels, self.hidden_channels, self.kernel_size, 1, self.padding, bias=False)
        self.Wxc = nn.Conv1d(self.input_channels, self.hidden_channels, self.kernel_size, 1, self.padding, bias=True)
        self.Whc = nn.Conv1d(self.hidden_channels, self.hidden_channels, self.kernel_size, 1, self.padding, bias=False)
        self.Wxo = nn.Conv1d(self.input_channels, self.hidden_channels, self.kernel_size, 1, self.padding, bias=True)
        self.Who = nn.Conv1d(self.hidden_channels, self.hidden_channels, self.kernel_size, 1, self.padding, bias=False)

        self.Wci = None
        self.Wcf = None
        self.Wco = None

    def forward(self, x, h, c):
        # print("Wci type ============================================>", self.Wci.type())
        h = h.to(device)
        c = c.to(device)
        # print("===============================Forward Conv_LSTM==========================================")
        # print("hidden(h) dimension size in ConvLSTMcell ============================================>", h.size())
        # print("cell(c) state dimension size in ConvLSTMcell ============================================>", c.size())
        self.Wci = self.Wci.to(device)
        self.Wcf = self.Wcf.to(device)
        self.Wco = self.Wco.to(device)
        # print("Wxi(x) in ConvLSTMcell ============================================>", self.Wxi(x).size())
        # print("Whi(h) in ConvLSTMcell ============================================>", self.Whi(h).size())
        # print("Wxf(x) in ConvLSTMcell ============================================>", self.Wxf(x).size())
        # print("Whf(h) in ConvLSTMcell ============================================>", self.Whf(h).size())
        ci = torch.sigmoid(self.Wxi(x) + self.Whi(h) + c * self.Wci)
        cf = torch.sigmoid(self.Wxf(x) + self.Whf(h) + c * self.Wcf)
        # print("ci in ConvLSTMcell ============================================>", ci.size())
        # print("cf in ConvLSTMcell ============================================>", cf.size())
        # print("Wxc(h) in ConvLSTMcell ============================================>", self.Wxc(x).size())
        # print("Whc(h) in ConvLSTMcell ============================================>", self.Whc(h).size())

        cc = cf * c + ci * torch.tanh(self.Wxc(x) + self.Whc(h))
        # print("ci in ConvLSTMcell ============================================>", ci.size())
        # print("cf in ConvLSTMcell ============================================>", cf.size())
        # print("Wxo(h) in ConvLSTMcell ============================================>", self.Wxo(x).size())
        # print("Who(h) in ConvLSTMcell ============================================>", self.Who(h).size())
        # print("Wco in ConvLSTMcell ============================================>", self.Wco.size())
        # print("cc in ConvLSTMcell ============================================>", cc.size())
        co = torch.sigmoid(self.Wxo(x) + self.Who(h) + cc * self.Wco)
        # print("co in ConvLSTMcell ============================================>", co.size())
        ch = co * torch.tanh(cc)
        # print("ch in ConvLSTMcell ============================================>", ch.size())
        return ch, cc

    def init_hidden(self, batch_size, hidden, shape):
        if self.Wci is None:
            self.Wci = Variable(torch.zeros(1, hidden, shape)).to("cpu")
            self.Wcf = Variable(torch.zeros(1, hidden, shape)).to("cpu")
            self.Wco = Variable(torch.zeros(1, hidden, shape)).to("cpu")
        else:
            assert shape == self.Wci.size()[2], 'Input Width Mismatched!'
            # assert shape[1] == self.Wci.size()[3], 'Input Width Mismatched!'
        return (Variable(torch.zeros(batch_size, hidden, shape)).to("cpu"),
                Variable(torch.zeros(batch_size, hidden, shape)).to("cpu"))


class ConvLSTM(nn.Module):
    # input_channels corresponds to the first input feature map
    # hidden state is a list of succeeding lstm layers.
    def __init__(self, input_channels, hidden_channels, kernel_size, step=1, effective_step=[1]):
        super(ConvLSTM, self).__init__()
        # print("input_channels:", input_channels)
        # print("hidden_channels:", hidden_channels)
        self.input_channels = [input_channels] + hidden_channels
        # print("self.input_channels:", self.input_channels)
        self.hidden_channels = hidden_channels
        self.kernel_size = kernel_size  # 3
        self.num_layers = len(hidden_channels)  # 1
        self.step = step  # 5
        self.effective_step = effective_step  # 4
        self._all_layers = []

        for i in range(self.num_layers):
            name = 'cell{}'.format(i)
            cell = ConvLSTMCell(self.input_channels[i], self.hidden_channels[i], self.kernel_size)
            setattr(self, name, cell)
            self._all_layers.append(cell)

    def forward(self, input):
        internal_state = []
        outputs = []
        for step in range(self.step):  # step = 5
            x = input
            for i in range(self.num_layers):   # step = 1
                # all cells are initialized in the first step
                name = 'cell{}'.format(i)
                if step == 0:
                    bsize, _, width = x.size()
                    # print(f"step {step} ~~> x.size():", bsize, self.hidden_channels[i], height, width)
                    (h, c) = getattr(self, name).init_hidden(batch_size=bsize, hidden=self.hidden_channels[i],
                                                             shape=width)
                    # print(f"step {step} init hidden:", h.size(), c.size())
                    internal_state.append((h, c))

                # do forward
                (h, c) = internal_state[i]
                x, new_c = getattr(self, name)(x, h, c)
                internal_state[i] = (x, new_c)

            # print(f"Effective step {step}:", step in self.effective_step)
            # only record effective steps
            if step in self.effective_step:
                outputs.append(x)

        return outputs, (x, new_c)


if __name__ == '__main__':
    # gradient check
    convlstm = ConvLSTM(input_channels=512, hidden_channels=[128, 64, 64, 32, 32], kernel_size=3, step=5,
                        effective_step=[4]).to("cpu")
    loss_fn = nn.MSELoss()

    input = Variable(torch.randn(5, 512, 64, 32)).to("cpu")
    print(input.type())
    target = Variable(torch.randn(1, 32, 64, 32)).double().to("cpu")

    output = convlstm(input)
    output = output[0][0].double()
    print(output.shape)
    # res = torch.autograd.gradcheck(loss_fn, (output, target), eps=1e-6, raise_exception=True)
    # print(res)
