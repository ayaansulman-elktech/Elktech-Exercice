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
        
        return processor, model
