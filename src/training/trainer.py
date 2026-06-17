import torch
import mlflow
import os
import yaml
from torch.utils.data import DataLoader
from tqdm import tqdm

class Trainer:
    def __init__(self, model, processor, train_dataset, val_dataset, config, collate_fn):
        self.model = model
        self.processor = processor
        self.config = config
        self.training_config = config.get("training", {})
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        
        self.train_dataloader = DataLoader(
            train_dataset, 
            batch_size=self.training_config.get("batch_size", 4), 
            shuffle=True, 
            collate_fn=collate_fn
        )
        self.val_dataloader = DataLoader(
            val_dataset, 
            batch_size=self.training_config.get("batch_size", 4), 
            shuffle=False, 
            collate_fn=collate_fn
        )
        
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(), 
            lr=self.training_config.get("learning_rate", 1e-4),
            weight_decay=self.training_config.get("weight_decay", 1e-4)
        )
        self.epochs = self.training_config.get("epochs", 5)

    def train(self):
        experiment_name = self.config.get("experiment", {}).get("name", "default_experiment")
        # Use sqlite instead of file backend to avoid MLflow exception
        mlflow.set_tracking_uri(f"sqlite:///{os.path.abspath('mlruns.db').replace(os.sep, '/')}")
        mlflow.set_experiment(experiment_name)
        
        best_val_loss = float('inf')
        
        with mlflow.start_run():
            # Log all config parameters
            mlflow.log_dict(self.config, "config.yaml")
            
            # Flatten config for easier viewing in MLflow UI
            flat_config = {}
            for k, v in self.config.items():
                if isinstance(v, dict):
                    for sub_k, sub_v in v.items():
                        flat_config[f"{k}.{sub_k}"] = sub_v
                else:
                    flat_config[k] = v
            mlflow.log_params(flat_config)
            
            for epoch in range(self.epochs):
                print(f"Epoch {epoch+1}/{self.epochs}")
                train_loss = self._train_one_epoch()
                val_loss = self._validate()
                
                print(f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
                mlflow.log_metrics({"train_loss": train_loss, "val_loss": val_loss}, step=epoch)
                
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
            
            # Save the final model (just the weights for now)
            os.makedirs("models_output", exist_ok=True)
            model_path = os.path.join("models_output", "final_model.pt")
            torch.save(self.model.state_dict(), model_path)
            mlflow.log_artifact(model_path, "models")
            
            print("Training complete. Model saved and logged to MLflow.")
            
        return best_val_loss

    def _train_one_epoch(self):
        self.model.train()
        total_loss = 0
        
        # Use tqdm for a progress bar
        progress_bar = tqdm(self.train_dataloader, desc="Training")
        
        for batch in progress_bar:
            pixel_values = batch["pixel_values"].to(self.device)
            pixel_mask = batch["pixel_mask"].to(self.device)
            labels = [{k: v.to(self.device) for k, v in t.items()} for t in batch["labels"]]

            self.optimizer.zero_grad()
            
            outputs = self.model(pixel_values=pixel_values, pixel_mask=pixel_mask, labels=labels)
            loss = outputs.loss
            
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
            progress_bar.set_postfix({"loss": loss.item()})
            
        return total_loss / len(self.train_dataloader)

    def _validate(self):
        self.model.eval()
        total_loss = 0
        
        progress_bar = tqdm(self.val_dataloader, desc="Validating")
        
        with torch.no_grad():
            for batch in progress_bar:
                pixel_values = batch["pixel_values"].to(self.device)
                pixel_mask = batch["pixel_mask"].to(self.device)
                labels = [{k: v.to(self.device) for k, v in t.items()} for t in batch["labels"]]

                outputs = self.model(pixel_values=pixel_values, pixel_mask=pixel_mask, labels=labels)
                loss = outputs.loss
                
                total_loss += loss.item()
                
        return total_loss / len(self.val_dataloader)
