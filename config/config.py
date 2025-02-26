import torch


def get_config(flag=0):

    if flag == 0:
        config = dict(
            dataset="Scene-15",
            seed=24,
            mask_seed=1,
            v_num=2,
            topk=10,
            missing_rate=0.5,
            n_clusters=15,
            training=dict(
                epoch=200,
                lr=1e-4,
            ),
            Autoencoder=dict(
                gcnEncoder1=[20, 1024, 1024, 1024, 1024 // 2],
                gcnEncoder2=[59, 1024, 1024, 1024, 1024 // 2],

                graphEncoder1=[1024 // 2, 1024, 1024, 1024, 1024 // 2],
                graphEncoder2=[1024 // 2, 1024, 1024, 1024, 1024 // 2],
                graphEncoderf=[1024 // 2, 1024, 1024, 1024, 1024 // 2],

                activations1='relu',
                activations2='relu',
                activationsf='relu',
                batchnorm=True,
            )
        )
    elif flag == 1:
        config = dict(
            dataset="LandUse-21",
            seed=30,
            mask_seed=5,
            v_num=2,
            topk=10,
            missing_rate=0.5,
            n_clusters=21,
            training=dict(
                epoch=300,
                lr=2e-5,
            ),
            Autoencoder=dict(
                gcnEncoder1=[59, 1024, 1024, 1024, 1024 // 4],
                gcnEncoder2=[40, 1024, 1024, 1024, 1024 // 4],

                graphEncoder1=[1024 // 4, 1024, 1024, 1024, 1024 // 4],
                graphEncoder2=[1024 // 4, 1024, 1024, 1024, 1024 // 4],
                graphEncoderf=[1024 // 4, 1024, 1024, 1024, 1024 // 4],

                activations1='relu',
                activations2='relu',
                activationsf='relu',
                batchnorm=True,
            )
        )
    elif flag == 2:
        config = dict(
            dataset="MSRC-v1",
            seed=35,
            mask_seed=1,
            v_num=6,
            topk=10,
            missing_rate=0.5,
            n_clusters=7,
            training=dict(
                epoch=200,
                lr=1e-4,
            ),
            Autoencoder=dict(
                gcnEncoder1=[1302, 1024, 1024, 1024, 1024 // 8],
                gcnEncoder2=[48, 1024, 1024, 1024, 1024 // 8],
                gcnEncoder3=[512, 1024, 1024, 1024, 1024 // 8],
                gcnEncoder4=[100, 1024, 1024, 1024, 1024 // 8],
                gcnEncoder5=[256, 1024, 1024, 1024, 1024 // 8],
                gcnEncoder6=[210, 1024, 1024, 1024, 1024 // 8],

                graphEncoder1=[1024 // 8, 1024, 1024, 1024, 1024 // 8],
                graphEncoder2=[1024 // 8, 1024, 1024, 1024, 1024 // 8],
                graphEncoder3=[1024 // 8, 1024, 1024, 1024, 1024 // 8],
                graphEncoder4=[1024 // 8, 1024, 1024, 1024, 1024 // 8],
                graphEncoder5=[1024 // 8, 1024, 1024, 1024, 1024 // 8],
                graphEncoder6=[1024 // 8, 1024, 1024, 1024, 1024 // 8],
                graphEncoderf=[1024 // 8, 1024, 1024, 1024, 1024 // 8],

                activations1='relu',
                activations2='relu',
                activations3='relu',
                activations4='relu',
                activations5='relu',
                activations6='relu',
                activationsf='relu',
                batchnorm=True,
            )
        )
    elif flag == 3:
        config = dict(
            dataset="handwritten",  # 2 300
            seed=34,
            mask_seed=5,
            v_num=4,
            topk=10,
            missing_rate=0.5,
            n_clusters=10,
            training=dict(
                epoch=200,
                lr=1e-3,
            ),
            Autoencoder=dict(

                gcnEncoder1=[240, 1024, 1024, 1024, 1024 // 8],
                gcnEncoder2=[76, 1024, 1024, 1024, 1024 // 8],
                gcnEncoder3=[216, 1024, 1024, 1024, 1024 // 8],
                gcnEncoder4=[64, 1024, 1024, 1024, 1024 // 8],

                graphEncoder1=[1024 // 8, 1024, 1024, 1024, 1024 // 8],
                graphEncoder2=[1024 // 8, 1024, 1024, 1024, 1024 // 8],
                graphEncoder3=[1024 // 8, 1024, 1024, 1024, 1024 // 8],
                graphEncoder4=[1024 // 8, 1024, 1024, 1024, 1024 // 8],
                graphEncoderf=[1024 // 8, 1024, 1024, 1024, 1024 // 8],

                activations1='relu',
                activations2='relu',
                activations3='relu',
                activations4='relu',
                activationsf='relu',
                batchnorm=True,
            )
        )
    elif flag == 4:
        config = dict(
            dataset="100leaves",
            seed=21,
            mask_seed=5,
            v_num=3,
            topk=10,
            missing_rate=0.5,
            n_clusters=100,
            training=dict(
                epoch=1000,
                lr=1e-4,
            ),
            Autoencoder=dict(
                gcnEncoder1=[64, 1024, 1024, 1024, 1024 // 2],
                gcnEncoder2=[64, 1024, 1024, 1024, 1024 // 2],
                gcnEncoder3=[64, 1024, 1024, 1024, 1024 // 2],

                graphEncoder1=[1024 // 2, 1024, 1024, 1024, 1024 // 2],
                graphEncoder2=[1024 // 2, 1024, 1024, 1024, 1024 // 2],
                graphEncoder3=[1024 // 2, 1024, 1024, 1024, 1024 // 2],
                graphEncoderf=[1024 // 2, 1024, 1024, 1024, 1024 // 2],

                activations1='relu',
                activations2='relu',
                activations3='relu',
                activationsf='relu',
                batchnorm=True,
            )
        )

    config["device"] = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    return config



