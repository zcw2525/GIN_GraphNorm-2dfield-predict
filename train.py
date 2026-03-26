import pytorch_lightning as ptl
import time

from torch_geometric.loader import DataLoader
from dataset_dataloader import CFDDataset
from config import SAVE_DIR, TRAIN_PHI, TEST_PHI,num_layers
from model import GIN_Module
from utils import get_device,cal_time
from pytorch_lightning.callbacks import ModelCheckpoint



def main():

    train_dataset = CFDDataset(SAVE_DIR, TRAIN_PHI)
    # test_dataset = CFDDataset(SAVE_DIR, TEST_PHI)
    sample = train_dataset[0]

    train_loader = DataLoader(
        train_dataset,
        batch_size=1,
        num_workers=0,
        shuffle=True,
        persistent_workers=False,
    )
    # for batch in train_loader:
    #     print(batch)
    #     print(batch.x.shape)
    #     print(batch.edge_index.shape)
    #     break
    #
    base_config = {
        "input_dim": sample.x.shape[1],
        "output_dim": sample.y.shape[1],
        "num_evs": 4,
        "hidden_dim": 64,
        "num_layers": num_layers,
        "lr": 1e-3,
        "dropout":0.1,
    }

    expriments = [
        {"name": 'GIN', "normalization": None, "residual": False},
        {"name": 'GIN_graphnorm', "normalization": 'graphnorm', "residual": False},
        {"name": 'GIN_res', "normalization": None, "residual": True},
        {"name": 'GIN_graphnorm_res', "normalization": 'graphnorm', "residual": True},
        {"name": 'GIN_graphnormv2_res', "normalization": 'graphnormv2', "residual": True},
    ]

    for exp in expriments:
        ptl.seed_everything(42)
        name = exp["name"]

        #拆分不同的字典再拼接
        config = {**base_config,
                  **exp}
        print("Running experiment", name)
        start_time = time.perf_counter()

        model = GIN_Module(config)

        checkpoint = ModelCheckpoint(
            monitor="train_loss",
            mode="min",
            save_top_k=1
        )

        trainer = ptl.Trainer(
            max_epochs=50,
            accelerator=get_device().type,
            devices=1,
            default_root_dir=f"logs/{name}",
            log_every_n_steps=1,
            callbacks=[checkpoint],
        )

        trainer.fit(model, train_loader)

        end_time = time.perf_counter()
        runtime = end_time - start_time
        minutes, seconds = cal_time(runtime)
        print(f'{name} runtime: {minutes}m {seconds:.2f}s')

        with open("running_log.txt", "a") as f:
            f.write(f'{name} runtime: {minutes}m {seconds:.2f}s\n')

if __name__ == '__main__':
    main()



