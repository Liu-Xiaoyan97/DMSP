import argparse

import pytorch_lightning as pl
from omegaconf import OmegaConf

from Modules.data_lightning import DataModuleForSentenceClassification
# from Modules.MSP_lightning import ModelModule
from Modules.DMSP_lightning import ModelModule

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--max_len', type=int, default=64)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--max_epoch', type=int, default=100)
    parser.add_argument('--ckpt', type=str, default=None)
    parser.add_argument('--is_train', type=str, default='Y')
    args = parser.parse_args()
    pl.seed_everything(1)
    """load config.yml"""
    config = OmegaConf.load("config.yml")
    dataset_conf, model_conf = config.dataset, config.model
    dm = DataModuleForSentenceClassification(file_path_dir=[dataset_conf.filename, dataset_conf.subset], max_len=args.max_len,
                                             batch_size=args.batch_size, label_map=dataset_conf.label_map)
    model = ModelModule(**model_conf, epsilon=0.5, patience=2)
    trainer = pl.Trainer(
        callbacks=[
            pl.callbacks.ModelCheckpoint(
                monitor="validation_step_accuracy",
                save_top_k=2,
                filename='{epoch}-{validation_step_loss:.4f}-{validation_step_accuracy:.4f}',
                mode='max'
            )
        ],
        accelerator="auto",
        devices='auto',
        max_epochs=args.max_epoch,
        check_val_every_n_epoch=1,
        log_every_n_steps=1
    )

    if args.is_train == "Y":
        dm.setup('fit')
        train_loader = dm.train_dataloader()
        val_loader = dm.val_dataloader()
        if args.ckpt:
            trainer.fit(model, train_loader, val_loader, ckpt_path=args.ckpt)
        else:
            trainer.fit(model, train_loader, val_loader)
    else:
        dm.setup('test')
        test_loader = dm.test_dataloader()
        trainer.test(model, test_loader, args.ckpt)
