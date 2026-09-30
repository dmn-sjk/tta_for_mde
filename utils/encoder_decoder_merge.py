import torch
import torch.nn as nn

class EncoderDecoderMerge(nn.Module):
    def __init__(self, encoder: nn.Module, decoder: nn.Module):
        """
        Convenience class that merges encoder and decoder into a single module.
        
        Args:
            encoder (nn.Module): The encoder network
            decoder (nn.Module): The decoder network
        """
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through both encoder and decoder sequentially.
        
        Args:
            x (torch.Tensor): Input tensor
            
        Returns:
            torch.Tensor: Output after passing through encoder and decoder
        """
        # Pass through encoder
        encoded = self.encoder(x)
        # Pass through decoder
        decoded = self.decoder(encoded)
        
        return decoded
    
    @property
    def device(self) -> torch.device:
        """
        Returns the device of the model (assumes both encoder and decoder are on same device)
        """
        return next(self.parameters()).device
