"""Spec2Pep: Transformer model for de novo peptide sequencing with CTC and PMC decoding."""
from loguru import logger
import re
import os
import torch
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import pytorch_lightning as pl
import torch.nn.functional as F

from . import mass_con
from . import mass_con_rules_v2
from ..components.transformers_0509 import PeptideDecoder, SpectrumEncoder
from ..components.mixins import ModelMixin
from .utils import PeptideMass_PTM
from .utils import mass_cal, ctc_post_processing
from .utils import find_ptm_variants, compute_ptm_probabilities, merge_tokens

INPUT_LENGTH = 40


class Config:
    """Simple configuration object that accepts keyword arguments as attributes."""

    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)

    def __repr__(self):
        class_name = self.__class__.__name__
        attrs = ', '.join(f"{key}={repr(value)}" for key, value in self.__dict__.items())
        return f"{class_name}({attrs})"


def mask_illegal_tokens(preds, illegal_idx):
    """Mask illegal token indices by setting their logits to negative infinity.

    Args:
        preds: Tensor of shape [batch_size, token_len, vocab_size].
        illegal_idx: List of token indices to mask out.

    Returns:
        Masked prediction tensor.
    """
    mask = torch.zeros_like(preds)
    mask[:, :, illegal_idx] = float('-inf')
    return preds + mask


class Spec2Pep(pl.LightningModule, ModelMixin):
    """A Transformer model for de novo peptide sequencing.

    Uses a Qwen3-based encoder-decoder architecture with CTC scoring and
    optional precursor mass control (PMC) knapsack decoding for inference.

    Parameters
    ----------
    dim_model : int
        The latent dimensionality used by the transformer model.
    n_head : int
        The number of attention heads in each layer.
    dim_feedforward : int
        The dimensionality of the fully connected layers.
    n_layers : int
        The number of transformer layers.
    dropout : float
        The dropout probability for all layers.
    dim_intensity : Optional[int]
        The number of features for encoding peak intensity.
    custom_encoder : Optional[SpectrumEncoder]
        A pretrained encoder to use.
    max_length : int
        The maximum peptide length to decode.
    residues : Union[Dict[str, float], str]
        The amino acid dictionary and their masses.
    max_charge : int
        The maximum precursor charge to consider.
    precursor_mass_tol : float
        The maximum allowable precursor mass tolerance in ppm.
    isotope_error_range : Tuple[int, int]
        The isotope error range to consider.
    n_beams : int
        Number of beams used during beam search decoding.
    mass_control_tol : float
        Tolerance in Da for precursor mass control decoding.
    PMC_enable : bool
        Whether to enable precursor mass control knapsack decoding.
    legal_ptms : List[str], optional
        List of allowed PTM tokens for constrained decoding.
    rules : dict, optional
        PTM transition rules mapping PTM tokens to allowed preceding residues.
    flash_attn : bool
        Whether to use Flash Attention.
    use_bf16 : bool
        Whether to use bfloat16 precision.
    """

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
        mass_control_tol: float = 0.1,
        out_writer=None,
        PMC_enable=True,
        legal_ptms: List[str] = None,
        num_key_value_heads: int = None,
        head_dim: int = None,
        flash_attn: bool = False,
        use_bf16: bool = False,
        rules=None,
        output_site_probs: bool = False,
        log_filename=None,
        **kwargs: Dict,
    ):
        super().__init__()
        self.mass_control_tol = mass_control_tol
        self.save_hyperparameters()
        self.PMC_enable = PMC_enable
        self.log_filename = log_filename
        self.output_site_probs = output_site_probs
        self.raw_rules = rules if rules else {}

        config = Config(
            head_dim=head_dim,
            hidden_size=dim_model,
            intermediate_size=dim_feedforward,
            num_attention_heads=n_head,
            num_hidden_layers=n_layers,
            num_key_value_heads=num_key_value_heads,
            max_position_embeddings=1000,
            attention_dropout=dropout,
            flash_attn=flash_attn,
            use_bf16=use_bf16,
        )
        if flash_attn:
            logger.info("Using Flash Attention")
        if use_bf16:
            logger.info("Using BFloat16")

        # Build the model.
        if custom_encoder is not None:
            self.encoder = custom_encoder
        else:
            self.encoder = SpectrumEncoder(
                dim_model=dim_model,
                n_head=n_head,
                dim_feedforward=dim_feedforward,
                n_layers=n_layers,
                dropout=dropout,
                dim_intensity=dim_intensity,
                config=config,
            )
        self.decoder = PeptideDecoder(
            dim_model=dim_model,
            n_head=n_head,
            dim_feedforward=dim_feedforward,
            n_layers=n_layers,
            dropout=dropout,
            residues=residues,
            max_charge=max_charge,
            max_pep_len=max_length,
            config=config,
        )

        self.legal_ptms = legal_ptms
        if legal_ptms is not None:
            logger.info("legal_ptms enabled")
            self.legal_idx = (
                [self.decoder._aa2idx[ptm] for ptm in legal_ptms]
                + [i for i in range(20)]
                + [len(self.decoder._amino_acids) - 1]
            )
            self.illegal_idx = [i for i in range(len(self.decoder._amino_acids)) if i not in self.legal_idx]

        self.n_term_allowed_ptms = set()
        self.rules = {}

        if rules is not None:
            logger.info("PTM rules enabled")
            for ptm, allowed_residues in rules.items():
                ptm_idx = self.decoder._aa2idx[ptm]
                valid_prev_ids = []
                for residue in allowed_residues:
                    if residue == "n-terminal":
                        self.n_term_allowed_ptms.add(ptm_idx)
                    else:
                        valid_prev_ids.append(self.decoder._aa2idx[residue])
                self.rules[ptm_idx] = valid_prev_ids

        self.trans_mask, self.start_mask = self._build_transition_matrix()

        if len(self.rules) > 0:
            logger.info("Using Mass Control Rules")
            logger.info(f"Rules: {self.rules}")

        self.n_layers = n_layers
        self.class_head = torch.nn.Linear(dim_model, 2)
        self.calctime = 0.0

        self.softmax = torch.nn.Softmax(2)
        self.ctcloss = torch.nn.CTCLoss(blank=self.decoder.get_blank_idx(), zero_infinity=True)
        self.confidence = torch.nn.CTCLoss(blank=self.decoder.get_blank_idx(), zero_infinity=False)

        # Data properties.
        self.max_charge = max_charge
        self.max_length = max_length
        self.residues = residues
        self.precursor_mass_tol = precursor_mass_tol
        self.isotope_error_range = isotope_error_range
        self.n_beams = n_beams
        self.peptide_mass_calculator = PeptideMass_PTM(self.residues)

        self.n_log = n_log

        # Output writer during prediction.
        self.out_writer = out_writer

    def _build_transition_matrix(self):
        """Build the transition and start mask matrices for constrained decoding.

        Returns
        -------
        trans_mask : torch.Tensor of shape (vocab_size, vocab_size)
            Transition mask where -inf blocks invalid transitions.
        start_mask : torch.Tensor of shape (vocab_size,)
            Start mask where -inf blocks tokens that cannot begin a sequence.
        """
        vocab_size = len(self.decoder._amino_acids)
        blank_id = self.decoder.get_blank_idx()

        # Build illegal_idx only when legal_ptms whitelist is provided
        illegal_indices = []
        if self.legal_ptms is not None:
            base_legal = list(range(20)) + [len(self.decoder._amino_acids) - 1]
            ptm_legal = [self.decoder._aa2idx[ptm] for ptm in self.legal_ptms]
            legal_set = set(base_legal + ptm_legal)
            legal_set.add(blank_id)
            all_indices = set(range(vocab_size))
            illegal_indices = list(all_indices - legal_set)

        # Initialize matrices
        trans_mask = torch.zeros((vocab_size, vocab_size), dtype=torch.float)
        start_mask = torch.zeros(vocab_size, dtype=torch.float)

        # Apply global illegal token mask
        if illegal_indices:
            trans_mask[:, illegal_indices] = float('-inf')
            start_mask[illegal_indices] = float('-inf')

        # Apply PTM transition rules
        if self.rules:
            for ptm_idx, allowed_prevs in self.rules.items():
                # Start mask: block PTMs that cannot appear at N-terminus
                if ptm_idx not in self.n_term_allowed_ptms:
                    start_mask[ptm_idx] = float('-inf')

                # Transition mask: block all incoming edges then open allowed ones
                trans_mask[:, ptm_idx] = float('-inf')
                if allowed_prevs:
                    trans_mask[allowed_prevs, ptm_idx] = 0.0

        return trans_mask, start_mask

    def constrained_argmax(self, logits):
        """Perform constrained greedy decoding using transition and start masks.

        Applies CTC merge logic (collapse repeated tokens separated by blanks)
        while respecting PTM placement rules defined in the transition matrix.

        Args:
            logits: Softmax probabilities of shape (batch_size, seq_len, vocab_size).

        Returns:
            List of lists of token indices for each batch element.
        """
        logits = torch.flip(logits.clone(), dims=[1])
        bs, seq_len, vocab_size = logits.shape
        blank_id = self.decoder.get_blank_idx()
        preds = []

        if self.trans_mask.device != logits.device:
            self.trans_mask = self.trans_mask.to(logits.device)
            self.start_mask = self.start_mask.to(logits.device)
        trans_mask, start_mask = self.trans_mask, self.start_mask

        for i in range(bs):
            logits_i = logits[i, :, :]
            pred = []
            prev_token = None
            last_was_blank = False

            for j in range(seq_len):
                logit = logits_i[j, :]

                # Apply appropriate mask
                if len(pred) == 0:
                    logit = logit + start_mask
                else:
                    logit = logit + trans_mask[prev_token]

                tmp = torch.argmax(logit).item()

                if tmp == blank_id:
                    last_was_blank = True
                    continue

                # CTC merge: skip consecutive duplicates unless separated by blank
                if prev_token is not None and tmp == prev_token and not last_was_blank:
                    last_was_blank = False
                    continue

                pred.append(tmp)
                prev_token = tmp
                last_was_blank = False

            preds.append(pred)

        return preds

    def forward_new(self, spectra, precursors, true_peps, batch_idx=None):
        """Forward pass for inference: decode peptide sequences from spectra.

        Uses constrained argmax decoding, optionally followed by PMC knapsack
        decoding when the argmax result does not satisfy the mass constraint.

        Args:
            spectra: Input spectra tensor of shape (batch, n_peaks, 2).
            precursors: Precursor info tensor of shape (batch, 3).
            true_peps: Ground truth peptide sequences (unused during inference).
            batch_idx: Current batch index (optional).

        Returns:
            top_tokens: List of predicted peptide strings.
            inference_scores: List of confidence scores.
            decode_methods: List of decoding method labels ("argmax" or "PMC").
        """
        encoder_output, mask = self.encoder(spectra)
        last_logits, all_logits, self_attns = encoder_output
        output_logits, _, output_list, _ = self.decoder(None, precursors, last_logits, mask)

        if self.legal_ptms is not None:
            output_logits = mask_illegal_tokens(output_logits, self.illegal_idx)

        batch_size = output_logits.shape[0]

        top_tokens = [[] for _ in range(batch_size)]
        decode_methods = ["" for _ in range(batch_size)]
        inference_scores = [0 for _ in range(batch_size)]
        all_site_probs = [[] for _ in range(batch_size)]

        # Constrained argmax decoding
        forward_probs = F.softmax(output_logits, -1)
        preds_ids = self.constrained_argmax(forward_probs)
        preds_aas = [''.join(self.decoder._idx2aa[x] for x in preds_ids[i]) for i in range(batch_size)]

        forward_log_probs = F.log_softmax(output_logits, -1)
        forward_log_probs = torch.flip(forward_log_probs, dims=[1])

        for i in range(batch_size):
            if not self.PMC_enable:
                top_tokens[i] = preds_aas[i]
                decode_methods[i] = "argmax"
                if top_tokens[i] == "":
                    inference_scores[i] = 0.0
                else:
                    pmc_ids = torch.LongTensor(
                        [self.decoder._aa2idx[aa] for aa in re.split(r"(?<=.)(?=[A-Z(])", preds_aas[i])]
                    ).to(forward_log_probs.device)
                    pred_logits = forward_log_probs[i, :, :]
                    input_length = torch.LongTensor([INPUT_LENGTH]).to(forward_log_probs.device)
                    target_length = torch.LongTensor([len(pmc_ids)]).to(forward_log_probs.device)
                    pmc_score = self.ctcloss(pred_logits.unsqueeze(1), pmc_ids.unsqueeze(0), input_length, target_length)
                    inference_scores[i] = torch.exp(-pmc_score)
            else:
                mass_true = precursors[i, 0].item() - 18.01
                forward_pred_mass, forward_seq = mass_cal(preds_aas[i], self.residues.copy())
                forward_mass_diff = abs(mass_true - forward_pred_mass)
                forward_valid = forward_mass_diff < self.mass_control_tol

                if forward_valid:
                    # Argmax result satisfies mass constraint
                    top_tokens[i] = preds_aas[i]
                    decode_methods[i] = "argmax"
                    pmc_ids = torch.LongTensor(
                        [self.decoder._aa2idx[aa] for aa in re.split(r"(?<=.)(?=[A-Z(])", preds_aas[i])]
                    ).to(forward_log_probs.device)
                    pred_logits = forward_log_probs[i, :, :]
                    input_length = torch.LongTensor([INPUT_LENGTH]).to(forward_log_probs.device)
                    target_length = torch.LongTensor([len(pmc_ids)]).to(forward_log_probs.device)
                    pmc_score = self.ctcloss(pred_logits.unsqueeze(1), pmc_ids.unsqueeze(0), input_length, target_length)
                    inference_scores[i] = torch.exp(-pmc_score)
                else:
                    # Use knapsack algorithm for mass-constrained decoding
                    if len(self.rules) > 0:
                        forward_knap = mass_con_rules_v2.knapDecode(
                            forward_log_probs[[i], :, :],
                            precursors[[i], 0],
                            self.mass_control_tol,
                            self.residues.copy(),
                            trans_mask=self.trans_mask.clone(),
                            start_mask=self.start_mask.clone(),
                        )
                    else:
                        forward_knap = mass_con.knapDecode(
                            forward_log_probs[[i], :, :],
                            precursors[[i], 0],
                            self.mass_control_tol,
                            self.residues.copy(),
                        )
                    forward_knap = ctc_post_processing(forward_knap)

                    if forward_knap:
                        decode_methods[i] = "PMC"
                        pmc_seq = "".join(self.decoder.detokenize_truth(forward_knap, True)[::-1])
                        top_tokens[i] = pmc_seq
                        pmc_ids = torch.LongTensor(
                            [self.decoder._aa2idx[aa] for aa in re.split(r"(?<=.)(?=[A-Z(])", pmc_seq)]
                        ).to(forward_log_probs.device)
                        pred_logits = forward_log_probs[i, :, :]
                        input_length = torch.LongTensor([INPUT_LENGTH]).to(forward_log_probs.device)
                        target_length = torch.LongTensor([len(pmc_ids)]).to(forward_log_probs.device)
                        pmc_score = self.ctcloss(pred_logits.unsqueeze(1), pmc_ids.unsqueeze(0), input_length, target_length)
                        inference_scores[i] = torch.exp(-pmc_score)
                    else:
                        # Knapsack failed; fall back to argmax result
                        top_tokens[i] = preds_aas[i]
                        decode_methods[i] = "return to argmax"
                        if top_tokens[i] == "":
                            inference_scores[i] = 0.0
                        else:
                            pmc_ids = torch.LongTensor(
                                [self.decoder._aa2idx[aa] for aa in re.split(r"(?<=.)(?=[A-Z(])", preds_aas[i])]
                            ).to(forward_log_probs.device)
                            pred_logits = forward_log_probs[i, :, :]
                            input_length = torch.LongTensor([INPUT_LENGTH]).to(forward_log_probs.device)
                            target_length = torch.LongTensor([len(pmc_ids)]).to(forward_log_probs.device)
                            pmc_score = self.ctcloss(pred_logits.unsqueeze(1), pmc_ids.unsqueeze(0), input_length, target_length)
                            inference_scores[i] = torch.exp(-pmc_score)

            # Compute PTM site probabilities if enabled
            if self.output_site_probs and self.raw_rules and top_tokens[i]:
                pred_logits_i = forward_log_probs[i, :, :]
                input_length_i = torch.LongTensor([INPUT_LENGTH]).to(forward_log_probs.device)
                all_variations = find_ptm_variants(top_tokens[i], self.raw_rules)
                variation_scores = []
                for variation in all_variations:
                    variation_ids = torch.LongTensor(
                        [self.decoder._aa2idx[aa] for aa in re.split(r"(?<=.)(?=[A-Z(])", variation)]
                    ).to(forward_log_probs.device)
                    target_len = torch.LongTensor([len(variation_ids)]).to(forward_log_probs.device)
                    v_score = self.confidence(pred_logits_i.unsqueeze(1), variation_ids.unsqueeze(0), input_length_i, target_len)
                    variation_scores.append(torch.exp(-v_score))
                all_site_probs[i] = compute_ptm_probabilities(
                    top_tokens[i], inference_scores[i], all_variations, variation_scores
                )

        return top_tokens, inference_scores, decode_methods, all_site_probs

    def predict_step(self, batch, batch_idx, dataloader_idx=None):
        """A single prediction step that writes results to a TSV file.

        Handles charge overflow by skipping spectra with charge > max_charge.
        Outputs detailed per-spectrum results including mass errors and metadata.
        """
        MAX_ALLOWED_CHARGE = self.max_charge

        original_precursors = batch[1]
        charges = original_precursors[:, 1]

        # Handle charge overflow
        bad_indices = torch.where(charges > MAX_ALLOWED_CHARGE)[0]
        safe_precursors = original_precursors.clone()
        if len(bad_indices) > 0:
            safe_precursors[bad_indices, 1] = 1.0

        peptides, inferscores, decode_methods, site_probs = self.forward_new(batch[0], safe_precursors, batch[2], batch_idx)

        # Invalidate results for out-of-range charges
        if len(bad_indices) > 0:
            for idx in bad_indices:
                peptides[idx] = ""
                inferscores[idx] = 0.0
                decode_methods[idx] = "skipped_charge_limit"
                site_probs[idx] = []

        metadata_list = batch[3]

        # Determine output file path
        import torch.distributed as dist
        if dist.is_initialized():
            rank = dist.get_rank()
        else:
            rank = 0

        file_path = self.log_filename.replace(".json", f"_rank{rank}.tsv")

        headers = [
            "title", "prediction", "label", "confidence_score", "decode_method",
            "label_mass", "prediction_mass", "label_mass_diff(ppm)",
            "prediction_mass_diff(ppm)", "label_mass_diff(Da)",
            "prediction_mass_diff(Da)", "precursor_mass_neutral",
            "precursor_mass_residue", "precursor_mz", "precursor_charge",
            "precursor_intensity", "retention_time",
        ]
        if self.output_site_probs:
            headers.append("modification_site_probs")
        header_str = "\t".join(headers) + "\n"

        if not os.path.exists(file_path):
            with open(file_path, 'a') as f:
                f.write(header_str)

        MASS_H2O = 18.0105647

        with open(file_path, 'a') as f:
            for i in range(len(peptides)):
                title = metadata_list[i].get('title', "")
                prediction = peptides[i]
                label = batch[2][i]
                confidence_score = float(inferscores[i])
                decode_method = decode_methods[i]

                # Use original batch data for logging true charge
                precursor_mass_neutral = float(batch[1][i][0])
                precursor_mass_residue = precursor_mass_neutral - MASS_H2O
                precursor_mz = float(batch[1][i][2])
                precursor_charge = int(batch[1][i][1])
                precursor_intensity = float(metadata_list[i].get('precursor_intensity', -1.0))

                rt_str = str(metadata_list[i].get('retention_time', -1.0))
                try:
                    retention_time = float(rt_str)
                except ValueError:
                    parts = [float(x) for x in rt_str.split('-') if x.strip()]
                    retention_time = sum(parts) / len(parts) if parts else -1.0

                # Compute theoretical masses and errors
                label_mass = 0.0
                if label and len(label) > 0:
                    try:
                        l_res_mass, _ = mass_cal(label, self.residues.copy())
                        label_mass = l_res_mass + MASS_H2O
                    except KeyError:
                        label_mass = 0.0

                prediction_mass = 0.0
                if len(prediction) > 0:
                    p_res_mass, _ = mass_cal(prediction, self.residues.copy())
                    prediction_mass = p_res_mass + MASS_H2O

                if label_mass > 0:
                    label_diff_da = precursor_mass_neutral - label_mass
                    label_diff_ppm = (label_diff_da / label_mass) * 1_000_000
                else:
                    label_diff_da = 0.0
                    label_diff_ppm = 0.0

                if prediction_mass > 0:
                    pred_diff_da = precursor_mass_neutral - prediction_mass
                    pred_diff_ppm = (pred_diff_da / prediction_mass) * 1_000_000
                else:
                    pred_diff_da = 0.0
                    pred_diff_ppm = 0.0

                row = [
                    str(title), str(prediction), str(label),
                    f"{confidence_score:.6f}", str(decode_method),
                    f"{label_mass:.4f}", f"{prediction_mass:.4f}",
                    f"{label_diff_ppm:.4f}", f"{pred_diff_ppm:.4f}",
                    f"{label_diff_da:.4f}", f"{pred_diff_da:.4f}",
                    f"{precursor_mass_neutral:.4f}", f"{precursor_mass_residue:.4f}",
                    f"{precursor_mz:.4f}", str(precursor_charge),
                    f"{precursor_intensity:.2f}", f"{retention_time:.4f}",
                ]
                if self.output_site_probs:
                    probs = site_probs[i]
                    if probs:
                        prob_strs = []
                        tokens = merge_tokens(re.split(r"(?<=.)(?=[A-Z(])", prediction))
                        for pos, token in enumerate(tokens):
                            if pos < len(probs) and probs[pos]:
                                ptm_prob_parts = [f"{ptm}:{float(prob):.4f}" for ptm, prob in probs[pos].items()]
                                prob_strs.append(";".join(ptm_prob_parts) if ptm_prob_parts else "")
                            else:
                                prob_strs.append("")
                        row.append(",".join(prob_strs))
                    else:
                        row.append("")
                f.write("\t".join(row) + "\n")

        return None

    def on_predict_epoch_end(self, results) -> None:
        """Synchronize processes at the end of prediction."""
        import torch.distributed as dist
        if dist.is_initialized():
            dist.barrier()

    def _get_output_peptide_and_scores(
        self, aa_tokens: List[str], aa_scores: torch.Tensor
    ) -> Tuple[str, List[str], float, str]:
        """Get peptide output with amino acid and peptide-level confidence scores.

        Parameters
        ----------
        aa_tokens : List[str]
            Amino acid tokens of the peptide sequence.
        aa_scores : torch.Tensor
            Amino acid-level confidence scores.

        Returns
        -------
        peptide : str
            Peptide sequence string.
        aa_tokens : List[str]
            Amino acid tokens.
        peptide_score : float
            Peptide-level confidence score.
        aa_scores : str
            Comma-separated amino acid scores.
        """
        aa_tokens = aa_tokens[1:] if self.decoder.reverse else aa_tokens[:-1]
        peptide = "".join(aa_tokens)

        if len(peptide) == 0:
            aa_tokens = []

        step = -1 if self.decoder.reverse else 1
        top_aa_scores = [
            aa_score[self.decoder._aa2idx[aa_token]].item()
            for aa_score, aa_token in zip(aa_scores, aa_tokens[::step])
        ][::step]

        peptide_score = _aa_to_pep_score(top_aa_scores)
        aa_scores = ",".join(list(map("{:.5f}".format, top_aa_scores)))
        return peptide, aa_tokens, peptide_score, aa_scores


def _aa_to_pep_score(aa_scores: List[float]) -> float:
    """Calculate peptide-level confidence score from amino acid level scores.

    Parameters
    ----------
    aa_scores : List[float]
        Amino acid level confidence scores.

    Returns
    -------
    float
        Peptide confidence score (mean of amino acid scores).
    """
    return np.mean(aa_scores)


def _calc_mass_error(calc_mz: float, obs_mz: float, charge: int, isotope: int = 0) -> float:
    """Calculate the mass error in ppm between theoretical and observed m/z.

    Parameters
    ----------
    calc_mz : float
        The theoretical m/z.
    obs_mz : float
        The observed m/z.
    charge : int
        The charge.
    isotope : int
        Number of C13 isotopes to correct for (default: 0).

    Returns
    -------
    float
        The mass error in ppm.
    """
    return (calc_mz - (obs_mz - isotope * 1.00335 / charge)) / obs_mz * 10**6
