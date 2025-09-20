import torch
from torchvision import transforms
from PIL import Image
from typing import Union, List, Dict, Any
import numpy as np

class CustomImageProcessor:
    def __init__(
        self,
        image_size: int = 224,
        mean: List[float] = [0.485, 0.456, 0.406],
        std: List[float] = [0.229, 0.224, 0.225],
        device = None
    ):
        self.image_size = image_size
        self.transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std)
        ])

        self.device = device

    def preprocess(
        self, 
        images: Union[Image.Image, List[Image.Image]], 
        text: Union[str, List[str]] = None,
        return_tensors: str = "pt"
    ) -> Dict[str, torch.Tensor]:
        """Process images and text into model inputs.
        
        Args:
            images: Single image or list of PIL Images
            text: Optional text or list of texts
            return_tensors: Format of tensors to return ("pt" for PyTorch)
            
        Returns:
            dict: Contains processed inputs
        """
        if not isinstance(images, list):
            images = [images]
            
        # Process images
        processed_images = []
        for image in images:
            if isinstance(image, str):
                image = Image.open(image).convert('RGB')
            processed_images.append(self.transform(image))
            
        # Stack images into batch
        image_tensor = torch.stack(processed_images)
        
        outputs = {"pixel_values": image_tensor.to(self.device)}
        
        # Process text if provided
        if text is not None:
            if not isinstance(text, list):
                text = [text]
            # Add text processing logic here if needed
            outputs["text"] = text
            
        return outputs

    def __call__(
        self, 
        images: Union[Image.Image, List[Image.Image]], 
        text: Union[str, List[str]] = None,
        return_tensors: str = "pt"
    ) -> Dict[str, torch.Tensor]:
        """Callable interface matching HuggingFace style."""
        return self.preprocess(images, text, return_tensors)

    def batch_decode(
        self,
        outputs: torch.Tensor,
        skip_special_tokens: bool = True
    ) -> List[str]:
        """Decode model outputs back to human readable format.
        
        Args:
            outputs: Model output tensors
            skip_special_tokens: Whether to remove special tokens
            
        Returns:
            list: Decoded outputs
        """
        # Add decoding logic here if needed
        return outputs.tolist()

# Example usage:
if __name__ == "__main__":
    # Initialize processor
    processor = CustomImageProcessor(image_size=224)
    
    # Load and process image
    image = Image.open("/path/to/image.jpg")
    inputs = processor(
        images=image,
        text="optional text input",
        return_tensors="pt"
    )
    
    print("Processed inputs:", inputs["pixel_values"].shape)