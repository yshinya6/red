from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Tuple, Union

import torch
from transformers.generation.utils import GenerateDecoderOnlyOutput, ModelOutput


@dataclass
class GenerateSpeculativeDecodingOutput(GenerateDecoderOnlyOutput):
    sequences: torch.LongTensor = None
    scores: Optional[Tuple[torch.FloatTensor]] = None
    logits: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[Tuple[torch.FloatTensor]]] = None
    hidden_states: Optional[Tuple[Tuple[torch.FloatTensor]]] = None
    past_key_values: Optional[Tuple[Tuple[Tuple[torch.FloatTensor]]]] = None
    avg_accepted_length: Optional[float] = None
    avg_acceptance_rate: Optional[float] = None
    avg_kl_divergence: Optional[float] = None
    avg_tv_distance: Optional[float] = None
    avg_assistant_decoding_time: Optional[float] = None
    avg_assistant_decoding_time_per_iter: Optional[float] = None
    avg_verification_time: Optional[float] = None
    avg_verification_time_per_iter: Optional[float] = None
    avg_prefill_time: Optional[float] = None


@dataclass
class GenerateDecodingOutput(GenerateDecoderOnlyOutput):
    sequences: torch.LongTensor = None
    scores: Optional[Tuple[torch.FloatTensor]] = None
    logits: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[Tuple[torch.FloatTensor]]] = None
    hidden_states: Optional[Tuple[Tuple[torch.FloatTensor]]] = None
    past_key_values: Optional[Tuple[Tuple[Tuple[torch.FloatTensor]]]] = None
    prefill_time: Optional[float] = None
    decoding_time: Optional[float] = None
    decoding_time_per_iter: Optional[float] = None


@dataclass
class GenerateDecodingDistOutput(GenerateDecoderOnlyOutput):
    sequences: torch.LongTensor = None
    scores: Optional[Tuple[torch.FloatTensor]] = None
    logits: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[Tuple[torch.FloatTensor]]] = None
    hidden_states: Optional[Tuple[Tuple[torch.FloatTensor]]] = None
    past_key_values: Optional[Tuple[Tuple[Tuple[torch.FloatTensor]]]] = None
    prefill_time: Optional[float] = None
    decoding_time: Optional[float] = None
    decoding_time_per_iter: Optional[float] = None
    avg_kl_divergence: Optional[float] = None
    avg_tv_distance: Optional[float] = None
    avg_diversity: Optional[float] = None
    avg_relevance: Optional[float] = None


@dataclass
class GenerateTokenReductionDecodingOutput(GenerateDecoderOnlyOutput):
    sequences: torch.LongTensor = None
    scores: Optional[Tuple[torch.FloatTensor]] = None
    logits: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[Tuple[torch.FloatTensor]]] = None
    hidden_states: Optional[Tuple[Tuple[torch.FloatTensor]]] = None
    past_key_values: Optional[Tuple[Tuple[Tuple[torch.FloatTensor]]]] = None
    new_input_length: Optional[int] = None
    prefil_time: Optional[float] = None
    visual_emb_time: Optional[float] = None
    decoding_time: Optional[float] = None


@dataclass
class Qwen2_5_VLCausalLMOutputWithPast(ModelOutput):
    loss: Optional[torch.FloatTensor] = None
    logits: torch.FloatTensor = None
    past_key_values: Optional[List[torch.FloatTensor]] = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None
    rope_deltas: Optional[torch.LongTensor] = None
    position_ids: Optional[torch.LongTensor] = None
