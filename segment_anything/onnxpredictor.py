from segment_anything import SamPredictor
import onnxruntime as ort
# from wonnx import Session
import torch
import cv2
from torch.nn import functional as F
import logging

class OnnxImageEncoder():
    def __init__(self, file_path: str) -> None:
        self.ort_session = ort.InferenceSession(file_path)
        self.img_size = 1024
        
    def __call__(self, image: torch.Tensor) -> torch.Tensor:
        return self.ort_session.run(None, {'input': image.cpu().numpy()})

class SamOnnxModel():
    mask_threshold: float = 0.0
    image_format: str = "RGB"
    
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
        
    pixel_mean = torch.Tensor([123.675, 116.28, 103.53]).unsqueeze(-1).unsqueeze(-1)
    pixel_std  = torch.Tensor([58.395, 57.12, 57.375]).unsqueeze(-1).unsqueeze(-1)
    def __init__(self, 
                file_path: str):
        
        self.image_encoder = OnnxImageEncoder(file_path)
        # self.register_buffer("pixel_mean", torch.Tensor(pixel_mean).view(-1, 1, 1), False)
        # self.register_buffer("pixel_std", torch.Tensor(pixel_std).view(-1, 1, 1), False)
        
    def preprocess(self, x: torch.Tensor) -> torch.Tensor:
        """Normalize pixel values and pad to a square input."""
        # print(x.shape)
        # print(f"pixel_mean: {self.pixel_mean}")
        # print(f"pixel_std: {self.pixel_std}")
        # print(x.device)
        logging.info(f"device: {x.device}")
        pixel_mean = self.pixel_mean.to(x.device)
        pixel_std = self.pixel_std.to(x.device)
        # Normalize colors
        x = (x - pixel_mean) / pixel_std

        # Pad
        h, w = x.shape[-2:]
        padh = self.image_encoder.img_size - h
        padw = self.image_encoder.img_size - w
        x = F.pad(x, (0, padw, 0, padh))
        return x

class SamOnnxPredictor(SamPredictor):
    def __init__(self, file_path: str) -> None:
        model = SamOnnxModel(file_path)
        super().__init__(model)
        

if __name__ == "__main__":
    import time
    logging.basicConfig(level=logging.INFO)
    print("start processing")
    tstart = time.time()
    predictor = SamOnnxPredictor("/Users/desjajja/Projects/sam-app/models/vit_quantized.onnx")
    image = cv2.imread('/Users/desjajja/Projects/sam-app/dog.jpg')
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    predictor.set_image(image)
    embedding = predictor.get_image_embedding()[0]
    assert embedding.shape == (1, 256, 64, 64)
    tend = time.time()
    print(f"process finished, time cost: {tend - tstart:.3f}s")
    
    import numpy as np
    np.save("images/embedding_onnx.npy", embedding)