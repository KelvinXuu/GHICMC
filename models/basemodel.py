import torch.optim
import torch.nn as nn
from torch.nn.parameter import Parameter
import torch


class MlpDecoder(nn.Module):
    def __init__(self, decoder_dim):
        super(MlpDecoder, self).__init__()
        self._dim = len(decoder_dim)-1
        decoder_layers = []
        decoder_dim = [i for i in reversed(decoder_dim)]
        for i in range(self._dim):
            decoder_layers.append(nn.Linear(decoder_dim[i], decoder_dim[i + 1]))
            decoder_layers.append(nn.BatchNorm1d(decoder_dim[i + 1]))
            decoder_layers.append(nn.ReLU())

        self._decoder = nn.Sequential(*decoder_layers)

    def forward(self, x):
        return self._decoder(x)


class GNNLayer(nn.Module):
    def __init__(self,
                 in_features_dim, out_features_dim,
                 activation='relu', use_bias=True):
        super(GNNLayer, self).__init__()
        self.in_features = in_features_dim
        self.out_features = out_features_dim
        self.use_bias = use_bias
        self.weight = Parameter(torch.FloatTensor(in_features_dim, out_features_dim))
        if self.use_bias:
            self.bias = Parameter(torch.FloatTensor(out_features_dim))
        self.init_parameters()

        self._bn1d = nn.BatchNorm1d(out_features_dim)
        if activation == 'sigmoid':
            self._activation = nn.Sigmoid()
        elif activation == 'leakyrelu':
            self._activation = nn.LeakyReLU(0.2, inplace=True)
        elif activation == 'tanh':
            self._activation = nn.Tanh()
        elif activation == 'relu':
            self._activation = nn.ReLU()
        else:
            raise ValueError('Unknown activation type %s' % self._activation)

    def init_parameters(self):
        """Initialize weights"""
        torch.nn.init.xavier_uniform_(self.weight)
        if self.use_bias:
            torch.nn.init.zeros_(self.bias)
        else:
            self.register_parameter('bias', None)

    def forward(self, features, adj, active=True, batchnorm=True):
        support = torch.mm(features, self.weight)
        output = torch.spmm(adj.to_dense(), support)

        if self.use_bias:
            output += self.bias
        if batchnorm:
            output = self._bn1d(output)
        if active:
            output = self._activation(output)

        return output


class GraphEncoder(nn.Module):
    def __init__(self, encoder_dim, activation='relu', batchnorm=True):
        super(GraphEncoder, self).__init__()
        self._dim = len(encoder_dim) - 1
        self._activation = activation
        self._batchnorm = batchnorm

        encoder_layers = []
        for i in range(self._dim):
            encoder_layers.append(GNNLayer(encoder_dim[i], encoder_dim[i + 1], activation=self._activation))

        self._encoder = nn.Sequential(*encoder_layers)

    def forward(self, x, adj, skip_connect=False):
        lst = [x]

        z = self._encoder[0](x, adj)
        lst.append(z)
        for layer in self._encoder[1:-1]:
            if skip_connect:
                z = layer(z, adj) + z
                lst.append(z)
            else:
                z = layer(z, adj)
                lst.append(z)
        z = self._encoder[-1](z, adj, True, False)

        return z, lst


class GNNLayer_t(nn.Module):
    def __init__(self,
                 in_features_dim, out_features_dim,
                 activation='relu', use_bias=True):
        super(GNNLayer_t, self).__init__()
        self.in_features = in_features_dim
        self.out_features = out_features_dim
        self.use_bias = use_bias
        self.weight = Parameter(torch.FloatTensor(in_features_dim, out_features_dim))
        self.weight2 = Parameter(torch.FloatTensor(in_features_dim, out_features_dim))
        if self.use_bias:
            self.bias = Parameter(torch.FloatTensor(out_features_dim))
        self.init_parameters()

        self._bn1d = nn.BatchNorm1d(out_features_dim)
        if activation == 'sigmoid':
            self._activation = nn.Sigmoid()
        elif activation == 'leakyrelu':
            self._activation = nn.LeakyReLU(0.2, inplace=True)
        elif activation == 'tanh':
            self._activation = nn.Tanh()
        elif activation == 'relu':
            self._activation = nn.ReLU()
        else:
            raise ValueError('Unknown activation type %s' % self._activation)

    def init_parameters(self):
        """Initialize weights"""
        torch.nn.init.xavier_uniform_(self.weight)
        torch.nn.init.xavier_uniform_(self.weight2)
        if self.use_bias:
            torch.nn.init.zeros_(self.bias)
        else:
            self.register_parameter('bias', None)

    def forward(self, features, rest_features, adj, active=True, batchnorm=True):
        support1 = torch.mm(features, self.weight)
        support2 = torch.mm(rest_features, self.weight2)
        output1 = torch.spmm(adj.to_dense(), (support1+support2)/2)

        if self.use_bias:
            output1 += self.bias
        if batchnorm:
            output1 = self._bn1d(output1)
        if active:
            output1 = self._activation(output1)

        return output1


class GraphEncoder_t(nn.Module):
    def __init__(self, encoder_dim, activation='relu', batchnorm=True):
        super(GraphEncoder_t, self).__init__()
        self._dim = len(encoder_dim) - 1
        self._activation = activation
        self._batchnorm = batchnorm

        encoder_layers = []
        for i in range(self._dim):
            encoder_layers.append(GNNLayer_t(encoder_dim[i], encoder_dim[i + 1], activation=self._activation))

        self._encoder = nn.Sequential(*encoder_layers)

    def forward(self, x, xs, adj, skip_connect=False):
        z = self._encoder[0](x, xs[0], adj,)
        i = 1
        for layer in self._encoder[1:-1]:
            if skip_connect:
                z = layer(z, xs[i], adj,) + z
            else:
                z = layer(z, xs[i], adj,)
            i = i+1
        z = self._encoder[-1](z, xs[-1], adj, True, False)

        return z


class InnerProductDecoder(nn.Module):
    """Decoder for using inner product for prediction."""

    def __init__(self, activation=torch.sigmoid):
        super(InnerProductDecoder, self).__init__()
        self.activation = activation

    def forward(self, z):
        # z = F.dropout(z, 0.5, training=self.training)
        adj = torch.mm(z, z.t())
        adj = self.activation(adj)
        return adj


class InnerProductDecoderW(nn.Module):
    """Decoder for using inner product for prediction."""

    def __init__(self, z_dim, activation=torch.sigmoid):
        super(InnerProductDecoderW, self).__init__()
        self.activation = activation
        self.W = Parameter(torch.Tensor(z_dim, z_dim))
        torch.nn.init.xavier_normal_(self.W)

    def forward(self, z):
        adj = z @ self.W @ torch.t(z)
        adj = self.activation(adj)
        return adj


class ClusterProject(nn.Module):
    def __init__(self, latent_dim, n_clusters):
        super(ClusterProject, self).__init__()
        self._latent_dim = latent_dim
        self._n_clusters = n_clusters

        self.cluster_projector = nn.Sequential(
            nn.Linear(self._latent_dim, self._latent_dim),
            nn.ReLU(),
        )

        self.cluster = nn.Sequential(
            nn.Linear(self._latent_dim, self._n_clusters),
            nn.Softmax(dim=1)
        )

    def forward(self, x):
        z = self.cluster_projector(x)
        y = self.cluster(z)
        return y, z


class FusionAdjLayer(nn.Module):

    def __init__(self, num_views, fusion_type='weighted'):
        """
        :param fusion_type: include concatenate/average
        """
        super(FusionAdjLayer, self).__init__()
        self.fusion_type = fusion_type
        self.num_views = num_views

        self.pai_adj = nn.Parameter(torch.ones(self.num_views) / self.num_views, requires_grad=True)

    def forward(self, adjs):
        # combine the adjacent matrix
        exp_sum_pai_adj = 0
        for i in range(self.num_views):
            exp_sum_pai_adj += torch.exp(self.pai_adj[i])

        combined_adjacent = (torch.exp(self.pai_adj[0]) / exp_sum_pai_adj) * adjs[0]
        for i in range(1, self.num_views):
            combined_adjacent = combined_adjacent + (torch.exp(self.pai_adj[i]) / exp_sum_pai_adj) * adjs[i]

        return combined_adjacent, self.pai_adj


class InstanceProject(nn.Module):
    def __init__(self, latent_dim):
        super(InstanceProject, self).__init__()
        self._latent_dim = latent_dim
        self.instance_projector = nn.Sequential(
            nn.Linear(self._latent_dim, self._latent_dim),
            nn.BatchNorm1d(self._latent_dim),
            nn.ReLU(),
            nn.Linear(self._latent_dim, self._latent_dim),
            nn.BatchNorm1d(self._latent_dim),
            nn.ReLU(),
        )

    def forward(self, x):
        return self.instance_projector(x)