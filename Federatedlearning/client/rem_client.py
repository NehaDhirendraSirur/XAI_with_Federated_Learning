import pickle
from pathlib import Path

import hydra
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf

import flwr as fl

from dataset import prepare_dataset
from client_class import generate_client_fn,FlowerClient


# A decorator for Hydra. This tells hydra to by default load the config in conf/base.yaml
@hydra.main(config_path="../conf", config_name="base", version_base=None)

def main(cfg: DictConfig):
    ## 1. Parse config & get experiment output dir
    print(OmegaConf.to_yaml(cfg))
    save_path = HydraConfig.get().runtime.output_dir

    ## 2. Prepare your dataset

    trainloaders, validationloaders, testloader = prepare_dataset(
        cfg.num_clients, cfg.batch_size
    )

    ## 3. Define your clients
    client_fn = generate_client_fn(trainloaders, validationloaders, cfg.num_classes)


    fl.client.start_numpy_client(
            server_address="127.0.0.1:9000",
            client=FlowerClient(trainloader=trainloaders[0],
            valloader=validationloaders[0],
            num_classes=cfg.num_classes,),
            grpc_max_message_length=-1
    )


if __name__ == "__main__":
    main()