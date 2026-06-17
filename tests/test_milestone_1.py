import sys
import os

# Add the root directory to the python path so we can import src
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils.config_parser import load_config
from src.models.model_factory import ModelFactory

def test_config_and_factory():
    config_path = "configs/default_config.yaml"
    
    print(f"Loading configuration from {config_path}...")
    config = load_config(config_path)
    print("Configuration loaded successfully:")
    print(config)
    print("-" * 40)
    
    print("Instantiating model from factory...")
    processor, model = ModelFactory.create_model_and_processor(config)
    
    print("Model and processor instantiated successfully!")
    print(f"Model class: {type(model).__name__}")
    print(f"Processor class: {type(processor).__name__}")

if __name__ == "__main__":
    test_config_and_factory()
