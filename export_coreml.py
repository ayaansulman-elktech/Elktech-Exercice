import torch
import coremltools as ct
from transformers import AutoImageProcessor, AutoModelForObjectDetection
import os

# Wrapper class to make the Hugging Face model compatible with Core ML
class YOLOSWrapper(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
        
    def forward(self, pixel_values):
        outputs = self.model(pixel_values=pixel_values)
        # Core ML works best with fixed tensor outputs rather than dictionaries
        return outputs.logits, outputs.pred_boxes

def export():
    model_name = "hustvl/yolos-tiny"
    weights_path = "models_output/final_model.pt"
    
    print(f"Loading model {model_name}...")
    # Use the original resolution (often 224x224 for YOLOS-tiny) to avoid interpolation issues
    processor = AutoImageProcessor.from_pretrained(model_name)
    model = AutoModelForObjectDetection.from_pretrained(model_name, num_labels=91, ignore_mismatched_sizes=True)
    
    if os.path.exists(weights_path):
        print(f"Loading optimized weights from {weights_path}...")
        model.load_state_dict(torch.load(weights_path, map_location="cpu"))
    
    model.eval()
    wrapper = YOLOSWrapper(model)

    # Use a standard resolution for YOLOS-Tiny (224x224) to avoid the interpolation bug
    input_size = 224
    example_input = torch.rand(1, 3, input_size, input_size)
    
    print(f"Tracing model with input size {input_size}x{input_size}...")
    # strict=False is often necessary for Hugging Face models
    traced_model = torch.jit.trace(wrapper, example_input, strict=False)
    
    print("Converting to Core ML...")
    mlmodel = ct.convert(
        traced_model,
        inputs=[ct.TensorType(name="image", shape=example_input.shape)],
        # Add labels if available
        classifier_config=ct.ClassifierConfig(list(model.config.id2label.values()))
    )
    
    # Save the model
    output_path = "ObjectDetector.mlpackage"
    mlmodel.save(output_path)
    print(f"Core ML model exported successfully to: {output_path}")

if __name__ == "__main__":
    export()
