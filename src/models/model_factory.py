from transformers import AutoImageProcessor, AutoModelForObjectDetection

class ModelFactory:
    """Factory class to load Hugging Face object detection models based on config."""
    
    @staticmethod
    def create_model_and_processor(config: dict):
        model_config = config.get("model", {})
        architecture = model_config.get("architecture", "").lower()
        name_or_path = model_config.get("name_or_path")
        num_labels = model_config.get("num_labels", 91)
        
        if not name_or_path:
            raise ValueError("Model 'name_or_path' must be specified in the configuration.")

        print(f"Loading {architecture} model from {name_or_path}...")
        
        # Load processor
        processor = AutoImageProcessor.from_pretrained(name_or_path)
        
        # Load model. We use ignore_mismatched_sizes=True in case we are 
        # initializing a pre-trained model with a different number of classes.
        model = AutoModelForObjectDetection.from_pretrained(
            name_or_path,
            num_labels=num_labels,
            ignore_mismatched_sizes=True
        )

        # Optimization for CPU: Freeze the backbone to speed up training
        if model_config.get("freeze_backbone", False):
            print("Freezing model backbone...")
            # For DETR and YOLOS, the backbone is usually accessible via model.model.backbone or model.vit
            if hasattr(model, "model") and hasattr(model.model, "backbone"):
                for param in model.model.backbone.parameters():
                    param.requires_grad = False
            elif hasattr(model, "vit"):
                for param in model.vit.parameters():
                    param.requires_grad = False
        
        return processor, model
