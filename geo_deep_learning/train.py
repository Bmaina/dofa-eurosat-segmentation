"""Train model with Lightning CLI."""
import logging
from typing import Any

from lightning.pytorch import Trainer, seed_everything
from lightning.pytorch.cli import ArgsType, LightningCLI

from geo_deep_learning.config import logging_config  # noqa: F401
from geo_deep_learning.tools.mlflow_logger import LoggerSaveConfigCallback

logger = logging.getLogger(__name__)


class GeoDeepLearningCLI(LightningCLI):
    """Custom LightningCLI."""

    def after_fit(self) -> None:
        """Log test metrics."""
        if self.trainer.is_global_zero:
            try:
                test_dataloader = self.datamodule.test_dataloader()
                if test_dataloader is None:
                    logger.warning("No test dataloader found.")
                    return

                best_model_path = self.trainer.checkpoint_callback.best_model_path
                logger.info("Best model path: %s", best_model_path)

                test_trainer = Trainer(
                    devices=1,
                    accelerator="cpu",
                    strategy="auto",
                    logger=self.trainer.logger,
                )

                best_model = self.model.__class__.load_from_checkpoint(
                    best_model_path,
                    weights_from_checkpoint_path=None,
                    strict=True,
                )

                test_trainer.test(
                    model=best_model,
                    dataloaders=test_dataloader,
                )

                logger.info("Test metrics logged successfully.")

            except Exception as e:
                logger.warning("after_fit hook failed: %s", str(e))
                print(f"\nTraining complete. Note: post-training test step skipped ({e})")

        self.trainer.strategy.barrier()


def main(args: ArgsType = None) -> None:
    """Run the main training pipeline."""
    seed_everything(42, workers=True)
    cli = GeoDeepLearningCLI(
        save_config_callback=LoggerSaveConfigCallback,
        save_config_kwargs={"overwrite": True},
        parser_kwargs={"parser_mode": "omegaconf"},
        auto_configure_optimizers=False,
        args=args,
    )
    if cli.trainer.is_global_zero:
        logger.info("Done!")


if __name__ == "__main__":
    main()