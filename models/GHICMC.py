import torch.optim
from .basemodel import *
from utils.loss import *
from utils.visualization import *
from utils.evaluation import *
from utils.graph_adjacency import *


class GHICMC(nn.Module):
    def __init__(self, config):
        super(GHICMC, self).__init__()
        self.config = config
        self.device = config["device"]
        self.v_num = config["v_num"]
        self.num_epochs = self.config["training"]["epoch"]
        self._latent_dim = config['Autoencoder']['gcnEncoder1'][-1]
        self._n_clusters = config['n_clusters']

        self.criterion_AE = nn.MSELoss()

        self.cluster = ClusterProject(self._latent_dim, self._n_clusters).to(self.device)
        self.criterion_cluster = MyClusterLoss(self._n_clusters, 0.5, self.device).to(self.device)

        self.fusion_adj = FusionAdjLayer(self.config["v_num"]).to(self.device)

        self.epoch_list = []
        self.loss_list = {'loss_ins': [], 'loss_clu': [], 'loss_hc': [], 'loss': []}
        self.eva_list = {'ARI': [], 'NMI': [], 'ACC': []}

        self.gcnEncoders = list()
        self.gcnDecoders = list()
        self.graphEncoders = list()    
        self.instance_projector = list()

        for i in range(self.v_num):
            self.gcnEncoders.append(GraphEncoder(config['Autoencoder'][f'gcnEncoder{i + 1}'], activation=config['Autoencoder'][f'activations{i + 1}'], batchnorm=config['Autoencoder']['batchnorm']).to(self.device))
            self.gcnDecoders.append(MlpDecoder(config['Autoencoder'][f'gcnEncoder{i + 1}']).to(self.device))
            self.instance_projector.append(InstanceProject(self._latent_dim).to(self.device))
            self.graphEncoders.append(GraphEncoder_t(config['Autoencoder'][f'graphEncoder{i + 1}'], activation=config['Autoencoder'][f'activations{i + 1}'], batchnorm=config['Autoencoder']['batchnorm']).to(self.device))

        self.graphEncoderf = GraphEncoder(config['Autoencoder'][f'graphEncoderf'], activation=config['Autoencoder'][f'activationsf'], batchnorm=config['Autoencoder']['batchnorm']).to(self.device)
        self.gcnEncoders = nn.ModuleList(self.gcnEncoders)
        self.gcnDecoders = nn.ModuleList(self.gcnDecoders)
        self.instance_projector = nn.ModuleList(self.instance_projector)
        self.graphEncoders = nn.ModuleList(self.graphEncoders)

        self.lambda_AE = 1
        self.lambda_clu = 10
        self.lambda_hc = 10

        self.loss_AE = torch.tensor(0.0).to(self.device)
        self.loss_clu = torch.tensor(0.0).to(self.device)
        self.loss_sc = torch.tensor(0.0).to(self.device)
        self.loss = torch.tensor(0.0).to(self.device)

        self.optimizer = torch.optim.Adam(self.parameters(), lr=self.config['training']['lr'])

    def run_train(self, train_data, train_labels, adj, adj_add, mask, accumulated_metrics, logger):

        features = list()
        features_add = list()
        self.mask = mask.to(self.device)
        self.adj = adj
        self.adj_add = adj_add

        for i in range(self.v_num):
            features_add.append(train_data[i].clone().to(self.device))
            features.append(features_add[i][mask[:, i].bool()])
            self.adj[i] = self.adj[i].to(self.device)
            self.adj_add[i] = self.adj_add[i].to(self.device)

        # #####################################################
        for epoch in range(self.num_epochs):
            self.train()

            self.loss_AE = torch.tensor(0.0).to(self.device)
            self.loss_clu = torch.tensor(0.0).to(self.device)
            self.loss_sc = torch.tensor(0.0).to(self.device)

            x_rec = list()
            P = list()

            self.adj_f, _ = self.fusion_adj(self.adj_add)
            h, h_expand, h_pg, z, y = self(features, self.adj)
            h_f = sum(h_expand) / torch.sum(self.mask, dim=1).view(-1, 1)

            for i in range(self.v_num):
                x_rec.append(self.gcnDecoders[i](h[i]))
                self.loss_AE += self.criterion_AE(features[i], x_rec[i])
            self.loss_AE = self.lambda_AE * self.loss_AE

            for i in range(self.v_num):
                sum_Y = torch.sum(h_pg[i] ** 2, dim=1)
                num = -2. * torch.matmul(h_pg[i], h_pg[i].t())
                num = 1. / (1. + (num + sum_Y.unsqueeze(1) + sum_Y).t())
                torch.diagonal(num).fill_(0.)
                p = num / torch.sum(num)
                P.append(torch.maximum(p, torch.tensor(1e-12)))

            sum_Y = torch.sum(h_f ** 2, dim=1)
            num = -2. * torch.matmul(h_f, h_f.t())
            num = 1. / (1. + (num + sum_Y.unsqueeze(1) + sum_Y).t())
            torch.diagonal(num).fill_(0.)
            q = num / torch.sum(num)
            Q = torch.maximum(q, torch.tensor(1e-12))

            kl = nn.KLDivLoss(reduction='batchmean')
            for i in range(self.v_num):
                hc_loss = kl(P[i].log(), Q.detach())
                self.loss_sc += self.lambda_hc * hc_loss

            cluster_loss = self.criterion_cluster(y)
            self.loss_clu = self.lambda_clu * cluster_loss

            self.loss = self.loss_AE + self.loss_sc + self.loss_clu

            total_loss = self.loss.item()

            self.optimizer.zero_grad()
            self.loss.backward()
            self.optimizer.step()

            if True:
                self.eval()
                with torch.no_grad():
                    h, h_expand, h_pg, z, y = self(features, self.adj)
                    y_max = sum(y)
                    y = y_max.data.cpu().numpy().argmax(1)

                    scores = evaluation(y_pred=y, y_true=train_labels.cpu().numpy(),
                                        accumulated_metrics=accumulated_metrics)

            if (epoch + 1) % 10 == 0:
                output = ("Epoch:{:.0f}/{:.0f}===>loss={:.4f}  ACC={:.4f}  NMI={:.4f}  ARI={:.4f}".format(
                    (epoch + 1), self.num_epochs, total_loss, scores["accuracy"], scores["NMI"], scores["ARI"]))
                logger.info(output)

            self.loss_list['loss'].append(total_loss)
            self.eva_list['ACC'].append(scores['accuracy'])
            self.eva_list['ARI'].append(scores['ARI'])
            self.eva_list['NMI'].append(scores['NMI'])

            loss_plot(self.loss_list['loss'], self.eva_list['ACC'], self.eva_list['NMI'], self.eva_list['ARI'],
                      self.config["dataset"])

        return accumulated_metrics['acc'][-1], accumulated_metrics['nmi'][-1], accumulated_metrics['ARI'][-1]

    def forward(self, x, adj):
        h = list()
        h_expand = list()
        h_ag = list()
        z = list()
        y = list()

        for i in range(self.v_num):
            mask_indices = torch.nonzero(self.mask[:, i]).squeeze()
            h_feature, _ = self.gcnEncoders[i](x[i], adj[i])  # Nv * d

            h_expand_feature = torch.zeros(self.mask.shape[0], h_feature.shape[1]).to(self.device)
            h_expand_feature[mask_indices] = h_feature

            h.append(h_feature)
            h_expand.append(h_expand_feature)

            # z_pro_t = nn.functional.normalize(self.instance_projector[i](h_feature), dim=1)
            # z_pro = torch.zeros(self.mask.shape[0], h_feature.shape[1]).to(self.device)
            # z_pro[mask_indices] = z_pro_t
            # z.append(z_pro)  # n*d

        for i in range(self.v_num):
            h_f = sum(h_expand) / torch.sum(self.mask, dim=1).view(-1, 1)
            feature_f, feature_lst = self.graphEncoderf(h_f, self.adj_f)
            feature = self.graphEncoders[i](h_expand[i], feature_lst, self.adj_f)

            h_ag.append(feature)
            y_label, _ = self.cluster(feature)
            y.append(y_label)

        return h, h_expand, h_ag, z, y