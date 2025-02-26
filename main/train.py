import collections
import warnings
from models.GHICMC import GHICMC
from config.config import get_config
from utils.util import *
from utils.graph_adjacency import *
from utils.dataloader import *
from utils.visualization import *
warnings.simplefilter("ignore")

data_dict = {
    0: 'Scene-15',
    1: 'LandUse-21',
    2: 'MSRC-v1',
    3: 'handwritten',
    4: '100leaves',
}

if __name__ == '__main__':

    test_time = 5

    for flag in [3]:
        config = get_config(flag)
        init_seed = config["seed"]
        mask_seed = config["mask_seed"]
        train_data, train_labels = load_train_data(config)

        for rate in [0.5]:
            fold_acc, fold_nmi, fold_ari = [], [], []
            config["missing_rate"] = rate
            logger = get_logger(config)

            for tt in range(1, test_time+1):
                logger.info("{}___{:.1f}___the {} test time....".format(config['dataset'], config['missing_rate'], tt))
                setup_seed(mask_seed)
                mask = get_mask(train_data[0].shape[0], config["missing_rate"], config["v_num"])

                train_miss = list()
                adj = list()
                adj_add = list()
                for i in range(config["v_num"]):
                    train_miss_data = train_data[i] * mask[:, i][:, np.newaxis]
                    train_miss.append(train_miss_data.to(torch.float32))

                mask = torch.from_numpy(mask).long()

                # ===============get_adjacency===============
                for i in range(config["v_num"]):
                    features = train_miss[i][mask[:, i].bool()]
                    adj_i, _ = get_adjacency(features, features.shape[0], config["topk"])
                    adj.append(adj_i)

                    mask_idx = mask[:, i].view(-1, 1) * mask[:, i]
                    result = torch.zeros(mask.shape[0], mask.shape[0])
                    result[mask_idx.bool()] = adj_i.to_dense().view(-1)
                    indices = torch.nonzero(result)
                    values = result[indices[:, 0], indices[:, 1]]
                    result = torch.sparse_coo_tensor(indices.t(), values, result.size())
                    adj_add.append(result)
                print("make adj done!")

                model_seed = init_seed+tt

                setup_seed(model_seed)
                accumulated_metrics = collections.defaultdict(list)  # Accumulated metrics

                model = GHICMC(config)
                model = model.to(config["device"])

                acc, nmi, ari = model.run_train(train_miss, train_labels, adj, adj_add, mask, accumulated_metrics, logger)

                fold_acc.append(acc)
                fold_nmi.append(nmi)
                fold_ari.append(ari)

                print("{}___{:.1f}___the {} test time RESULT: ACC:{:.2f}  NMI:{:.2f}  ARI:{:.2f}".format(
                    config['dataset'], config['missing_rate'], tt, 
                    round(np.mean(acc) * 100, 2), round(np.mean(nmi) * 100, 2), round(np.mean(ari) * 100, 2)))

            logger.info('--------------------Training over--------------------')
            acc, nmi, ari = cal_std(logger, fold_acc, fold_nmi, fold_ari)

            filepath = f'../result/{config["dataset"]}.txt'
            write_eva(filepath, rate, fold_acc, fold_nmi, fold_ari)

            logger.handlers.clear()
