import torch

class Config:
    device='cuda' if torch.cuda.is_available else 'cpu'
    

    