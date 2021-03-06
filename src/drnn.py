import torch
import torch.nn as nn

use_cuda = torch.cuda.is_available()
device = 'cuda:7' if torch.cuda.is_available() else 'cpu'


class DRNN(nn.Module):

    def __init__(self, n_input, n_hidden, n_layers, dropout=0, cell_type='GRU', batch_first=False):
        super(DRNN, self).__init__()

        self.dilations = [2 ** i for i in range(n_layers)]
        # print(f"dilations: {self.dilations} as the Num of layers is {n_layers}")

        self.cell_type = cell_type
        self.batch_first = batch_first

        layers = []
        if self.cell_type == "GRU":
            cell = nn.GRU
        elif self.cell_type == "RNN":
            cell = nn.RNN
        elif self.cell_type == "LSTM":
            cell = nn.LSTM
        else:
            raise NotImplementedError

        for i in range(n_layers):
            if i == 0:
                c = cell(n_input, n_hidden, dropout=dropout)
            else:
                c = cell(n_hidden, n_hidden, dropout=dropout)
            layers.append(c)
        self.cells = nn.Sequential(*layers)
        # print("cells:", self.cells)

    def forward(self, inputs, hidden=None):
        output, output2 = [inputs], []
        if self.batch_first:
            inputs = inputs.transpose(0, 1)

        for i, (cell, dilation) in enumerate(zip(self.cells, self.dilations)):
            if hidden is None:
                # print("hidden is None")
                inputs, _ = self.drnn_layer(cell, inputs, dilation)
            else:
                inputs, hidden[i] = self.drnn_layer(cell, inputs, dilation, hidden[i])
            # print("final output size size:", inputs.size())

            if self.batch_first:
                inputs = inputs.transpose(0, 1)
            output.append(inputs)
            output2.append(inputs[-dilation:])
        # output = torch.stack(output, dim=0)
        return output, output2, inputs

    def drnn_layer(self, cell, inputs, rate, hidden=None):
        # print(f"=============================== dilation rate : {rate} ====================================")
        n_steps = len(inputs)
        batch_size = inputs[0].size(0)
        hidden_size = cell.hidden_size
        # print("n_steps:", n_steps, "batch_size:", batch_size, "hidden_size:", hidden_size)

        inputs, _ = self._pad_inputs(inputs, n_steps, rate)
        dilated_inputs = self._prepare_inputs(inputs, rate)
        # print("_pad_inputs:", inputs.size(), "dilated_inputs:", dilated_inputs.size())

        if hidden is None:
            dilated_outputs, hidden = self._apply_cell(dilated_inputs, cell, batch_size, rate, hidden_size)
        else:
            hidden = self._prepare_inputs(hidden, rate)
            dilated_outputs, hidden = self._apply_cell(dilated_inputs, cell, batch_size, rate, hidden_size, hidden=hidden)
        # print("dilated_output after _apply_cell:", dilated_outputs.size(), "hidden after _apply_cell:", hidden.size())
        splitted_outputs = self._split_outputs(dilated_outputs, rate)
        # print("output after split:", splitted_outputs.size())
        outputs = self._unpad_outputs(splitted_outputs, n_steps)
        # print("output after unpad:", outputs.size())
        return outputs, hidden

    def _apply_cell(self, dilated_inputs, cell, batch_size, rate, hidden_size, hidden=None):
        if hidden is None:
            if self.cell_type == 'LSTM':
                c, m = self.init_hidden(batch_size * rate, hidden_size)
                hidden = (c.unsqueeze(0), m.unsqueeze(0))
            else:
                hidden = self.init_hidden(batch_size * rate, hidden_size).unsqueeze(0)
        # print("inputs in apply cell function:", dilated_inputs.size())
        # print("hidden in apply cell function:", hidden.size())
        # print("cell:", cell)

        hidden = hidden.to(device)
        dilated_outputs, hidden = cell(dilated_inputs, hidden)

        return dilated_outputs, hidden

    def _unpad_outputs(self, splitted_outputs, n_steps):
        return splitted_outputs[:n_steps]

    def _split_outputs(self, dilated_outputs, rate):
        # print("dilated_outputs:", dilated_outputs.size(1))
        batchsize = dilated_outputs.size(1) // rate

        blocks = [dilated_outputs[:, i * batchsize: (i + 1) * batchsize, :] for i in range(rate)]

        interleaved = torch.stack(blocks).transpose(1, 0).contiguous()
        interleaved = interleaved.view(dilated_outputs.size(0) * rate, batchsize, dilated_outputs.size(2))
        return interleaved

    def _pad_inputs(self, inputs, n_steps, rate):
        is_even = (n_steps % rate) == 0

        if not is_even:
            dilated_steps = n_steps // rate + 1

            zeros_ = torch.zeros(dilated_steps * rate - inputs.size(0),
                                 inputs.size(1),
                                 inputs.size(2))
            zeros_ = zeros_.to(device)

            inputs = torch.cat((inputs, zeros_))
        else:
            dilated_steps = n_steps // rate

        return inputs, dilated_steps

    def _prepare_inputs(self, inputs, rate):
        dilated_inputs = torch.cat([inputs[j::rate, :, :] for j in range(rate)], 1)
        return dilated_inputs

    def init_hidden(self, batch_size, hidden_dim):
        hidden = torch.zeros(batch_size, hidden_dim)
        hidden = hidden.to(device)
        if self.cell_type == "LSTM":
            memory = torch.zeros(batch_size, hidden_dim)
            memory = memory.to(device)
            return hidden, memory
        else:
            return hidden
