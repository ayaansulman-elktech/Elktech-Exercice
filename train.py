import argparse
import os
from src.utils.config_parser import load_config
from src.models.model_factory import ModelFactory
from src.data.dataset import CocoDetectionDataset
from src.training.trainer import Trainer

def main(config_path):
    print(f"Loading configuration from {config_path}...")
    config = load_config(config_path)
    
    print("Initializing model and processor...")
    processor, model = ModelFactory.create_model_and_processor(config)
    
    print("Preparing datasets...")
    dataset_config = config.get("dataset", {})
    train_path = dataset_config.get("train_path")
    val_path = dataset_config.get("val_path")
    
    # We need a collate_fn for the DataLoader. The processor handles padding.
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
    
    print("Initializing trainer...")
    trainer = Trainer(
        model=model,
        processor=processor,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        config=config,
        collate_fn=collate_fn
    )
    
    print("Starting training...")
    trainer.train()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train an object detection model.")
    parser.add_argument("--config", type=str, default="configs/default_config.yaml", help="Path to the YAML configuration file.")
    args = parser.parse_args()
    
    main(args.config)
