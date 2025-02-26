
import scipy
import torch
from sklearn.preprocessing import MinMaxScaler


def load_train_data(config):
    data_name = config['dataset']

    if "Scene-15" in data_name:
        print("Loading Scene-15 for Training....")
        data_set = scipy.io.loadmat(f'../data/Scene-15.mat')

        datas = data_set["X"][0]
        labels = data_set["Y"]

        Y_train = torch.tensor(labels).view(-1)

        scaler = MinMaxScaler()
        X_train_v1 = scaler.fit_transform(datas[0])
        X_train_v2 = scaler.fit_transform(datas[1])
        X_train_v1 = torch.tensor(X_train_v1).to(torch.float32)
        X_train_v2 = torch.tensor(X_train_v2).to(torch.float32)

        print("Loading Scene-15 over!!!")
        return [X_train_v1, X_train_v2], Y_train

    elif "LandUse-21" in data_name:
        print("Loading LandUse-21 for Training....")
        data_set = scipy.io.loadmat(f'../data/LandUse-21.mat')

        datas = data_set["X"][0]
        labels = data_set["Y"]

        Y_train = torch.tensor(labels).view(-1)

        scaler = MinMaxScaler()
        X_train_v2 = scaler.fit_transform(datas[1])
        X_train_v3 = scaler.fit_transform(datas[2])
        X_train_v2 = torch.tensor(X_train_v2).to(torch.float32)
        X_train_v3 = torch.tensor(X_train_v3).to(torch.float32)

        print("Loading LandUse-21 over!!!")
        return [X_train_v2, X_train_v3], Y_train

    elif "MSRC-v1" in data_name:
        print("Loading MSRC-v1 for Training....")
        data_set = scipy.io.loadmat(f'../data/MSRC-v1.mat')

        datas = data_set["X"][0]
        labels = data_set["Y"]

        Y_train = torch.tensor(labels).view(-1)

        scaler = MinMaxScaler()
        X_train_v1 = scaler.fit_transform(datas[0])
        X_train_v2 = scaler.fit_transform(datas[1])
        X_train_v3 = scaler.fit_transform(datas[2])
        X_train_v4 = scaler.fit_transform(datas[3])
        X_train_v5 = scaler.fit_transform(datas[4])
        X_train_v6 = scaler.fit_transform(datas[5])
        X_train_v1 = torch.tensor(X_train_v1).to(torch.float32)
        X_train_v2 = torch.tensor(X_train_v2).to(torch.float32)
        X_train_v3 = torch.tensor(X_train_v3).to(torch.float32)
        X_train_v4 = torch.tensor(X_train_v4).to(torch.float32)
        X_train_v5 = torch.tensor(X_train_v5).to(torch.float32)
        X_train_v6 = torch.tensor(X_train_v6).to(torch.float32)

        print("Loading MSRC-v1 over!!!")
        return [X_train_v1,X_train_v2,X_train_v3,X_train_v4,X_train_v5,X_train_v6], Y_train

    elif "handwritten" in data_name:
        print("Loading handwritten for Training....")
        data_set = scipy.io.loadmat(f'../data/handwritten.mat')

        datas = data_set["X"][0]
        labels = data_set["Y"]
        Y_train = torch.tensor(labels).view(-1)

        scaler = MinMaxScaler()
        X_train_v1 = scaler.fit_transform(datas[0])
        X_train_v2 = scaler.fit_transform(datas[1])
        X_train_v3 = scaler.fit_transform(datas[2])
        X_train_v5 = scaler.fit_transform(datas[4])
        X_train_v1 = torch.tensor(X_train_v1).to(torch.float32)
        X_train_v2 = torch.tensor(X_train_v2).to(torch.float32)
        X_train_v3 = torch.tensor(X_train_v3).to(torch.float32)
        X_train_v5 = torch.tensor(X_train_v5).to(torch.float32)

        print("Loading handwritten over!!!")
        return [X_train_v1,X_train_v2,X_train_v3, X_train_v5,], Y_train

    elif "100leaves" in data_name:
        print("Loading 100leaves for Training....")
        data_set = scipy.io.loadmat(f'../data/100leaves.mat')

        datas = data_set["X"][0]
        labels = data_set["Y"]

        Y_train = torch.tensor(labels).view(-1)

        scaler = MinMaxScaler()
        X_train_v1 = scaler.fit_transform(datas[0])
        X_train_v2 = scaler.fit_transform(datas[1])
        X_train_v3 = scaler.fit_transform(datas[2])
        X_train_v1 = torch.tensor(X_train_v1).to(torch.float32)
        X_train_v2 = torch.tensor(X_train_v2).to(torch.float32)
        X_train_v3 = torch.tensor(X_train_v3).to(torch.float32)

        print("Loading 100leaves over!!!")
        return [X_train_v1,  X_train_v2, X_train_v3], Y_train
