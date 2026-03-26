import pytorch_lightning as ptl
from torch_geometric.loader import DataLoader

from dataset_dataloader import CFDDataset
from config import SAVE_DIR, TEST_PHI
from model import GIN_Module
from utils import get_device
import os


def main():

    # dataset
    test_dataset = CFDDataset(SAVE_DIR, TEST_PHI)

    test_loader = DataLoader(
        test_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=0
    )

    base_log_dir = "logs"

    model_names = [
        "GIN",
        "GIN_graphnorm",
        "GIN_graphnorm_res",
        "GIN_res",
        "GIN_graphnormv2_res",
    ]

    for name in model_names:

        checkpoints_dir = os.path.join(
            base_log_dir,
            name,
            "lightning_logs",
            "version_1",
            "checkpoints"
        )

        ckpt_file = [f for f in os.listdir(checkpoints_dir) if f.endswith(".ckpt")][0]
        checkpoint_path = os.path.join(checkpoints_dir, ckpt_file)

        print(f"\n===== Testing {name} | {ckpt_file} =====")

        model = GIN_Module.load_from_checkpoint(checkpoint_path)

        trainer = ptl.Trainer(
            accelerator=get_device().type,
            devices=1
        )

        trainer.test(model, test_loader)


if __name__ == "__main__":
    main()