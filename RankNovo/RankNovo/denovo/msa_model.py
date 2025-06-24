"""A de novo peptide sequencing model."""
import logging
from typing import Any, Dict, List, Optional, Set, Tuple, Union

from .. import masses
import numpy as np
import pytorch_lightning as pl
import torch
from torch.utils.tensorboard import SummaryWriter
from ..components.transformers import SpectrumEncoder
from ..components import ModelMixin
from . import evaluate

from ..msa.tokenizer import Alphabet, agent
from ..msa.constants import proteinseq_toks
from ..msa.msa_transformer import MSADecoder
import argparse
import os

logger = logging.getLogger("RankNovo")

class Spec2Pep(pl.LightningModule, ModelMixin):

    def __init__(
        self,
        dim_model: int = 512,
        n_head: int = 8,
        dim_feedforward: int = 1024,
        n_layers: int = 9,
        dropout: float = 0.0,
        dim_intensity: Optional[int] = None,
        custom_encoder: Optional[SpectrumEncoder] = None,
        max_length: int = 100,
        residues: Union[Dict[str, float], str] = "canonical",
        max_charge: int = 5,
        precursor_mass_tol: float = 50,
        isotope_error_range: Tuple[int, int] = (0, 1),
        n_beams: int = 5,
        n_log: int = 10,
        tb_summarywriter: Optional[
            torch.utils.tensorboard.SummaryWriter] = None,
        train_label_smoothing: float = 0.01,
        warmup_iters: int = 100_000,
        max_iters: int = 600_000,
        out_writer = None,
        species = None,
        save_folder = "./",
        use_col = True,
        **kwargs: Dict,
    ):
        super().__init__()
        self.save_hyperparameters()
        
        
        self.encoder = SpectrumEncoder(
                dim_model=dim_model,
                n_head=n_head,
                dim_feedforward=dim_feedforward,
                n_layers=n_layers,
                dropout=dropout,
                dim_intensity=dim_intensity,
            )
        self.msa_alphabet = Alphabet(proteinseq_toks)
        self.msa_transformer_batch_converter = self.msa_alphabet.get_batch_converter()
        decoder_args = argparse.Namespace(
            num_layers=n_layers,
            embed_dim=dim_model,
            logit_bias=False,
            ffn_embed_dim=dim_feedforward,
            attention_heads=n_head,
            dropout=dropout,
            attention_dropout=dropout,
            activation_dropout=dropout,
            max_tokens_per_msa=2 ** 14,
            embed_positions_msa=False,
            max_positions=max_length,
            max_charge=max_charge
        )
        self.msa_decoder = MSADecoder(decoder_args, self.msa_alphabet, use_col)
        self.global_final = torch.nn.Linear(dim_model, 1)
        torch.nn.init.xavier_uniform_(self.global_final.weight)
        torch.nn.init.zeros_(self.global_final.bias)
        self.local_final = torch.nn.Linear(dim_model, 1)
        torch.nn.init.xavier_uniform_(self.local_final.weight)
        torch.nn.init.zeros_(self.local_final.bias)

        # Optimizer settings.
        self.warmup_iters = warmup_iters
        self.max_iters = max_iters
        self.opt_kwargs = kwargs
        self.best_pep_recall = 0
        self.save_folder = save_folder
        if not os.path.exists(self.save_folder):
            os.makedirs(self.save_folder, exist_ok=True)

        # Data properties.
        self.max_length = max_length
        self.residues = residues
        self.precursor_mass_tol = precursor_mass_tol
        self.isotope_error_range = isotope_error_range
        self.n_beams = n_beams
        self.peptide_mass_calculator = masses.PeptideMass(
            self.residues)
        
        # Logging.
        self.n_log = n_log
        self._history = []
        if tb_summarywriter is not None:
            self.tb_summarywriter = SummaryWriter(tb_summarywriter)
        else:
            self.tb_summarywriter = tb_summarywriter

        # Output writer during predicting.
        self.out_writer = out_writer

        torch.autograd.set_detect_anomaly(True)
    
    
    def _forward_step(
        self,
        spectra: torch.Tensor, # torch.Size([48, 300, 2])
        precursors: torch.Tensor, # torch.Size([48, 3])
        sequences: List[str],
    ):
        """
        the returns are all in shape (B, N)
        """
        
        spectrum, masks = self.encoder(spectra,precursors) # (B, L, D), (B, L)
        formatted_sequences = agent.MSAFormater(sequences)
        labels, _, tokens, fine_labels, fine_mask = self.msa_transformer_batch_converter(formatted_sequences)

        decoder_output = self.msa_decoder(tokens, precursors, spectrum, masks)
        logits = decoder_output["logits"]
        first_logits = logits[:, :, 0, :]
        global_prediction = self.global_final(first_logits).squeeze(-1)
        token_mask = decoder_output["mask"]

        seq_mask = token_mask.all(dim=2)

        local_prediction = self.local_final(logits).squeeze(-1)
        
        return global_prediction, local_prediction, ~seq_mask, labels, fine_labels, fine_mask
    
    def my_pure_valid_forward(self, batch):
        """
        A single validation step.

        Parameters
        ----------
        batch : Tuple[torch.Tensor, torch.Tensor, List[str]]
            A batch of (i) MS/MS spectra, (ii) precursor information, (iii)
            peptide sequences.

        Returns
        -------
        torch.Tensor
            The loss of the validation step.
        """
        spectra, precursors, sequences, _ = batch
        spectra = spectra.to(self.device)
        precursors = precursors.to(self.device)

        peptides_true = [s[0] for s in sequences]
        sequences = [s[1:] for s in sequences]

        global_prediction, local_prediction, mask, _, _, fine_mask = self._forward_step(spectra, precursors, sequences)
        mask = mask.to(self.device)
        global_prediction[mask == 0] = float('inf')
        
        B, N, _ = local_prediction.shape
        local_score = torch.zeros((B, N)).to(local_prediction.device)
        for i in range(B):
            for j in range(N):
                score = torch.max(torch.abs(local_prediction[i, j, :][fine_mask[i, j, :]]))
                local_score[i, j] = score

        total_score = global_prediction
        min_indices = torch.argmin(total_score, dim=1).cpu().tolist()

        for i, val in enumerate(min_indices):
            sequences[i].append(sequences[i][val])

        sequences = np.array(sequences)
        aa_precision_list = []
        pep_recall_list = []
        for i in range(sequences.shape[1]):
            model_sequences = sequences[:, i].copy().tolist()
            aa_precision, _, pep_recall = evaluate.aa_match_metrics(
                *evaluate.aa_match_batch(model_sequences, peptides_true,
                                        self.peptide_mass_calculator.masses))
            if i == 0:
                _, n_aa, _ = evaluate.aa_match_batch(model_sequences, peptides_true,
                                            self.peptide_mass_calculator.masses)
            aa_precision_list.append(aa_precision)
            pep_recall_list.append(pep_recall)
        
      
        return aa_precision_list, n_aa, pep_recall_list, np.concatenate([np.array(peptides_true).reshape(-1,1), sequences], axis=-1), global_prediction.cpu().numpy()


    def configure_optimizers(
        self, ) -> Tuple[torch.optim.Optimizer, Dict[str, Any]]:
        """
        Initialize the optimizer.

        This is used by pytorch-lightning when preparing the model for training.

        Returns
        -------
        Tuple[torch.optim.Optimizer, Dict[str, Any]]
            The initialized Adam optimizer and its learning rate scheduler.
        """
        optimizer = torch.optim.AdamW(self.parameters(), **self.opt_kwargs)

        # Apply learning rate scheduler per step.
        lr_scheduler = CosineWarmupScheduler(optimizer,
                                             warmup=self.warmup_iters,
                                             max_iters=self.max_iters)
        return [optimizer], {"scheduler": lr_scheduler, "interval": "step"}


    def on_before_optimizer_step(self, optimizer, optimizer_idx):
        total_norm = 0.0
        for p in self.parameters():
            if p.grad is not None:
                param_norm = p.grad.detach().data.norm(2)
                total_norm += param_norm.item() ** 2
        total_norm = total_norm ** (1. / 2)
        # if(total_norm > 1000):
        #     self.zero_grad()
        if(not self.logger==None):
            self.logger.experiment["/grad_norm_before_clip"].append(total_norm)
    
    def on_train_batch_end(self, outputs, batch, batch_idx):
        total_norm = 0.0
        for p in self.parameters():
            if p.grad is not None:
                param_norm = p.grad.detach().data.norm(2) # L2 norm
                total_norm += param_norm.item() ** 2
        total_norm = total_norm ** (1. / 2)
        if(not self.logger==None):
            self.logger.experiment["/grad_norm_after_clip"].append(total_norm)
            
    def on_after_backward(self) -> None:
        valid_gradients = True
        for name, param in self.named_parameters():
            if param.grad is not None:
                valid_gradients = not (torch.isnan(param.grad).any() or torch.isinf(param.grad).any())
                if not valid_gradients:
                    break
        if not valid_gradients:
            logger.warning(f'detected inf or nan values in gradients. not updating model parameters')
            self.zero_grad()

class CosineWarmupScheduler(torch.optim.lr_scheduler._LRScheduler):
    """
    Learning rate scheduler with linear warm up followed by cosine shaped decay.

    Parameters
    ----------
    optimizer : torch.optim.Optimizer
        Optimizer object.
    warmup : int
        The number of warm up iterations.
    max_iters : torch.optim
        The total number of iterations.
    """

    def __init__(self, optimizer: torch.optim.Optimizer, warmup: int,
                 max_iters: int):
        self.warmup, self.max_iters = warmup, max_iters
        super().__init__(optimizer)

    def get_lr(self):
        lr_factor = self.get_lr_factor(epoch=self.last_epoch)
        return [base_lr * lr_factor for base_lr in self.base_lrs]

    def get_lr_factor(self, epoch):

        decay = self.warmup/self.max_iters
        if epoch <= self.warmup:
            lr_factor = 1 * (epoch / self.warmup)
        else:
            lr_factor = 0.5 * (1 + np.cos(np.pi * (
                (epoch - (decay * self.max_iters)) / ((1-decay) * self.max_iters))))


        return lr_factor
