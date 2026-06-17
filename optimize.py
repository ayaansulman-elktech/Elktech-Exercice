import optuna
import os
import argparse
from src.utils.config_parser import load_config
from src.models.model_factory import ModelFactory
from src.data.dataset import CocoDetectionDataset
from src.training.trainer import Trainer

def objective(trial, base_config):
    # Suggest hyperparameters
    architecture = trial.suggest_categorical("architecture", ["detr", "yolos"])
    
    if architecture == "detr":
        model_name = "facebook/detr-resnet-50"
    else:
        model_name = "hustvl/yolos-tiny"

    lr = trial.suggest_float("learning_rate", 1e-5, 1e-4, log=True)
    weight_decay = trial.suggest_float("weight_decay", 1e-5, 1e-3, log=True)
    batch_size = trial.suggest_categorical("batch_size", [2, 4])
    
    # Update config
    config = base_config.copy()
    config["model"] = config.get("model", {}).copy()
    config["model"]["architecture"] = architecture
    config["model"]["name_or_path"] = model_name
    config["model"]["freeze_backbone"] = True

    config["training"] = config.get("training", {}).copy()
    config["training"]["learning_rate"] = lr
    config["training"]["weight_decay"] = weight_decay
    config["training"]["batch_size"] = batch_size
    config["training"]["epochs"] = 1 # Keep epochs low for optimization speed
    
    config["experiment"] = config.get("experiment", {}).copy()
    config["experiment"]["name"] = f"optuna_{architecture}_trial_{trial.number}"
    
    print(f"\n--- Starting Trial {trial.number} ---")
    print(f"Params: arch={architecture}, lr={lr:.6f}, wd={weight_decay:.6f}, bs={batch_size}")
    
    # Setup model, data and trainer
    processor, model = ModelFactory.create_model_and_processor(config)
    
    dataset_config = config.get("dataset", {})
    train_path = dataset_config.get("train_path")
    val_path = dataset_config.get("val_path")
    
    def collate_fn(batch):
        images = [item[0] for item in batch]
        annotations = [item[1] for item in batch]
        encoding = processor(images=images, annotations=annotations, return_tensors="pt")
        batch_dict = {}
        batch_dict['pixel_values'] = encoding['pixel_values']
        batch_dict['pixel_mask'] = encoding['pixel_mask']
        batch_dict['labels'] = encoding['labels']
        return batch_dict

    train_dataset = CocoDetectionDataset(
        img_folder=os.path.join(train_path, "images"),
        ann_file=os.path.join(train_path, "_annotations.coco.json"),
        processor=processor
    )
    
    val_dataset = CocoDetectionDataset(
        img_folder=os.path.join(val_path, "images"),
        ann_file=os.path.join(val_path, "_annotations.coco.json"),
        processor=processor
    )
    
    trainer = Trainer(
        model=model,
        processor=processor,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        config=config,
        collate_fn=collate_fn
    )
    
    val_loss = trainer.train()
    return val_loss

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/default_config.yaml")
    parser.add_argument("--n_trials", type=int, default=3)
    args = parser.parse_args()
    
    base_config = load_config(args.config)
    
    study = optuna.create_study(direction="minimize")
    study.optimize(lambda trial: objective(trial, base_config), n_trials=args.n_trials)
    
    print("\n" + "="*40)
    print("OPTIMIZATION COMPLETE")
    print(f"Best trial: {study.best_trial.number}")
    print(f"Best value (val_loss): {study.best_value:.4f}")
    print(f"Best params: {study.best_params}")
    print("="*40)

if __name__ == "__main__":
    main()
