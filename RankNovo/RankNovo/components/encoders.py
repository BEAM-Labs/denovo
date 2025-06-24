"""Simple encoders for input into Transformers and the like."""
import torch
import einops
import numpy as np


class MassEncoder(torch.nn.Module):
    """Encode mass values using sine and cosine waves.

    Parameters
    ----------
    dim_model : int
        The number of features to output.
    min_wavelength : float
        The minimum wavelength to use.
    max_wavelength : float
        The maximum wavelength to use.
    """

    def __init__(self, dim_model, min_wavelength=0.001, max_wavelength=10000):
        """Initialize the MassEncoder"""
        super().__init__()

        n_sin = int(dim_model / 2)
        n_cos = dim_model - n_sin

        if min_wavelength:
            base = min_wavelength / (2 * np.pi)
            scale = max_wavelength / min_wavelength
        else:
            base = 1
            scale = max_wavelength / (2 * np.pi)

        sin_term = base * scale ** (
            torch.arange(0, n_sin).float() / (n_sin - 1)
        )
        cos_term = base * scale ** (
            torch.arange(0, n_cos).float() / (n_cos - 1)
        )

        self.register_buffer("sin_term", sin_term)
        self.register_buffer("cos_term", cos_term)

    def forward(self, X):
        """Encode m/z values.

        Parameters
        ----------
        X : torch.Tensor of shape (n_masses)
            The masses to embed.

        Returns
        -------
        torch.Tensor of shape (n_masses, dim_model)
            The encoded features for the mass spectra.
        """
        sin_mz = torch.sin(X / self.sin_term)
        cos_mz = torch.cos(X / self.cos_term)
        return torch.cat([sin_mz, cos_mz], axis=-1)


class PeakEncoder(MassEncoder):
    """Encode m/z values in a mass spectrum using sine and cosine waves.

    Parameters
    ----------
    dim_model : int
        The number of features to output.
    dim_intensity : int, optional
        The number of features to use for intensity. The remaining features
        will be used to encode the m/z values.
    min_wavelength : float, optional
        The minimum wavelength to use.
    max_wavelength : float, optional
        The maximum wavelength to use.
    """

    def __init__(
        self,
        dim_model,
        dim_intensity=None,
        min_wavelength=0.001,
        max_wavelength=10000,
    ):
        """Initialize the MzEncoder"""
        self.dim_intensity = dim_intensity
        self.dim_mz = dim_model

        if self.dim_intensity is not None:
            self.dim_mz -= self.dim_intensity

        super().__init__(
            dim_model=self.dim_mz,
            min_wavelength=min_wavelength,
            max_wavelength=max_wavelength,
        )

        if self.dim_intensity is None:
            self.int_encoder = torch.nn.Linear(1, dim_model, bias=False)
        else:
            self.int_encoder = MassEncoder(
                dim_model=dim_intensity,
                min_wavelength=0,
                max_wavelength=1,
            )

    def forward(self, X):
        """Encode m/z values and intensities.

        Note that we expect intensities to fall within the interval [0, 1].

        Parameters
        ----------
        X : torch.Tensor of shape (n_spectra, n_peaks, 2)
            The spectra to embed. Axis 0 represents a mass spectrum, axis 1
            contains the peaks in the mass spectrum, and axis 2 is essentially
            a 2-tuple specifying the m/z-intensity pair for each peak. These
            should be zero-padded, such that all of the spectra in the batch
            are the same length.

        Returns
        -------
        torch.Tensor of shape (n_spectr, n_peaks, dim_model)
            The encoded features for the mass spectra.
        """
        # mass encoder
        m_over_z = X[:, :, [0]]
        encoded = super().forward(m_over_z)
        
        # intensity encoder
        intensity = self.int_encoder(X[:, :, [1]])
        

        if self.dim_intensity is None:
            return encoded + intensity
        
        return torch.cat([encoded, intensity], dim=2)

class PositionalEncoder(torch.nn.Module):
    """The positional encoder for sequences.

    Parameters
    ----------
    dim_model : int
        The number of features to output.
    """

    def __init__(self, dim_model, max_wavelength=10000):
        """Initialize the MzEncoder"""
        super().__init__()

        n_sin = int(dim_model / 2)
        n_cos = dim_model - n_sin
        scale = max_wavelength / (2 * np.pi)

        sin_term = scale ** (torch.arange(0, n_sin).float() / (n_sin - 1))
        cos_term = scale ** (torch.arange(0, n_cos).float() / (n_cos - 1))
        self.register_buffer("sin_term", sin_term)
        self.register_buffer("cos_term", cos_term)

    def forward(self, X):
        """Encode positions in a sequence.

        Parameters
        ----------
        X : torch.Tensor of shape (batch_size, n_sequence, n_features)
            The first dimension should be the batch size (i.e. each is one
            peptide) and the second dimension should be the sequence (i.e.
            each should be an amino acid representation).

        Returns
        -------
        torch.Tensor of shape (batch_size, n_sequence, n_features)
            The encoded features for the mass spectra.
        """
        pos = torch.arange(X.shape[1]).type_as(self.sin_term)
        pos = einops.repeat(pos, "n -> b n", b=X.shape[0])
        sin_in = einops.repeat(pos, "b n -> b n f", f=len(self.sin_term))
        cos_in = einops.repeat(pos, "b n -> b n f", f=len(self.cos_term))

        sin_pos = torch.sin(sin_in / self.sin_term)
        cos_pos = torch.cos(cos_in / self.cos_term)
        encoded = torch.cat([sin_pos, cos_pos], axis=2)
        return encoded + X

class TrainablePositionEncoding(torch.nn.Module):
    def __init__(self, d_model, max_sequence_length=150):
        super().__init__()
        self.max_sequence_length = max_sequence_length
        self.d_model = d_model
        self.pe = torch.nn.Embedding(self.max_sequence_length, self.d_model)
        torch.nn.init.constant_(self.pe.weight, 0.)

    def forward(self, x):
        sequence_length = x.size(1)
        positions = torch.arange(sequence_length, dtype=torch.long, device=x.device)
        positions = positions.unsqueeze(0).expand(x.size(0), sequence_length)
        position_embeddings = self.pe(positions)
        return x + position_embeddings
    
class RotaryPositionalEmbeddings(torch.nn.Module):

    def __init__(self, d: int, base: int = 10_000):
        super().__init__()
        self.base = base
        self.d = d
        self.cos_cached = None
        self.sin_cached = None

    def _build_cache(self, seq_len: int, device: torch.device):
        if self.cos_cached is not None and seq_len <= self.cos_cached.shape[0]:
            return

        theta = 1. / (self.base ** (torch.arange(0, self.d, 2).float() / self.d)).to(device)
        seq_idx = torch.arange(seq_len, device=device).float()
        idx_theta = torch.einsum('n,d->nd', seq_idx, theta)
        idx_theta2 = torch.stack([idx_theta, idx_theta], dim=-1).reshape(seq_len, -1)

        self.cos_cached = idx_theta2.cos().to(device)  # [seq_len, d]
        self.sin_cached = idx_theta2.sin().to(device)  # [seq_len, d]

    def _neg_half(self, x: torch.Tensor):
        d_2 = self.d // 2
        return torch.cat([-x[..., d_2:], x[..., :d_2]], dim=-1)

    def forward(self, x: torch.Tensor):
        seq_len, device = x.shape[1], x.device
        self._build_cache(seq_len, device)

        batch_size = x.shape[0]
        cos_cached = self.cos_cached[:seq_len, :].unsqueeze(0).expand(batch_size, -1, -1)
        sin_cached = self.sin_cached[:seq_len, :].unsqueeze(0).expand(batch_size, -1, -1)

        neg_half_x = self._neg_half(x)
        x_rotated = (x * cos_cached) - (neg_half_x * sin_cached)  # [batch_size, seq_len, d]

        return x_rotated