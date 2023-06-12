import torch
import torch.nn as nn
import torch.nn.functional as F


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1 or classname.find('Conv') != -1:
        nn.init.constant_(m.weight, 0)
        nn.init.normal_(m.bias, 0, 0.01)


class HyperLinear(nn.Module):
    def __init__(self, dim_in, dim_out, dim_context, hypernet_dim=8, n_hidden=1, activation=nn.Tanh):
        super(HyperLinear, self).__init__()
        self.dim_in = dim_in
        self.dim_context = dim_context
        self.dim_out = dim_out
        self.params_dim = self.dim_in * self.dim_out + self.dim_out

        layers = []
        dims = [1+dim_context] + [hypernet_dim] * n_hidden + [self.params_dim]
        for i in range(1, len(dims)):
            layers.append(nn.Linear(dims[i - 1], dims[i]))
            if i < len(dims) - 1:
                layers.append(activation())
        self._hypernet = nn.Sequential(*layers)
        self._hypernet.apply(weights_init)

    def forward(self, t, context, x):
        #TODO
        params = self._hypernet(t.view(1, 1)).view(-1)
        b = params[:self.dim_out].view(self.dim_out)
        w = params[self.dim_out:].view(self.dim_out, self.dim_in)
        return F.linear(x, w, b)


class IgnoreLinear(nn.Module):
    def __init__(self, dim_in, dim_out, dim_context):
        super(IgnoreLinear, self).__init__()
        self._layer = nn.Linear(dim_in, dim_out)

    def forward(self, t, context, x):
        return self._layer(x)


class ConcatLinear(nn.Module):
    def __init__(self, dim_in, dim_out, dim_context):
        super(ConcatLinear, self).__init__()
        self._layer = nn.Linear(dim_in + 1 + dim_context, dim_out)

    def forward(self, t, context, x):
        #TODO
        if x.dim() == 3:
            context = context.unsqueeze(1).expand(-1, x.size(1), -1)
        tt = torch.ones_like(x[:, :1]) * t
        # ttx = torch.cat([tt, x], 1)
        x_context = torch.cat((tt, x, context), dim=2)
        return self._layer(x_context)


class ConcatLinear_v2(nn.Module):
    def __init__(self, dim_in, dim_out, dim_context):
        super(ConcatLinear_v2, self).__init__()
        self._layer = nn.Linear(dim_in, dim_out)
        self._hyper_bias = nn.Linear(1+dim_context, dim_out, bias=False)

    # TODO
    def forward(self, t, context, x):
        bias = self._hyper_bias(torch.cat([t, context], 1))
        if x.dim() == 3:
            bias = bias.unsqueeze(1)
        return self._layer(x) + bias
    


class SquashLinear(nn.Module):
    def __init__(self, dim_in, dim_out, dim_context):
        super(SquashLinear, self).__init__()
        self._layer = nn.Linear(dim_in, dim_out)
        self._hyper = nn.Linear(1+dim_context, dim_out)

    # TODO
    def forward(self, t, context, x):
        return self._layer(x) * torch.sigmoid(self._hyper(torch.cat([t, context], 1)))


class ConcatSquashLinear(nn.Module):
    def __init__(self, dim_in, dim_out, dim_context):
        super(ConcatSquashLinear, self).__init__()
        self._layer = nn.Linear(dim_in+dim_context, dim_out)
        self._hyper_bias = nn.Linear(1, dim_out, bias=False)
        self._hyper_gate = nn.Linear(1, dim_out)

    # TODO
    def forward(self, t, context, x):
        return self._layer(torch.cat([x, context], 1)) * torch.sigmoid(self._hyper_gate(t.view(-1))) \
            + self._hyper_bias(t.view(-1))

class ConcatLinear_v3(nn.Module):
    def __init__(self, dim_in, dim_out, dim_context):
        super(ConcatLinear_v3, self).__init__()
        self._layer = nn.Linear(dim_in+dim_context+1, dim_out)

    def forward(self, t, context, x):
        return self._layer(torch.cat([x, context, t.repeat(x.shape[0], 1)], 1))