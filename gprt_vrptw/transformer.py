from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn

from .config import GPRTConfig
from .grammar import Grammar, Tree


@dataclass(slots=True)
class GeneratedExpression:
    tree: Tree
    token_ids: tuple[int, ...]
    log_probability: Tensor
    entropy: Tensor


class CausalTransformer(nn.Module):
    """Small decoder-only causal language model over prefix programs."""

    def __init__(self, grammar: Grammar, config: GPRTConfig):
        super().__init__()
        self.grammar = grammar
        self.config = config
        # BOS + max expression tokens + EOS.
        self.max_sequence_length = config.max_tokens + 2
        self.token_embedding = nn.Embedding(len(grammar.vocab), config.d_model, padding_idx=grammar.pad_id)
        self.position_embedding = nn.Embedding(self.max_sequence_length, config.d_model)
        layer = nn.TransformerEncoderLayer(
            d_model=config.d_model,
            nhead=config.nhead,
            dim_feedforward=config.dim_feedforward,
            dropout=config.dropout,
            batch_first=True,
            norm_first=True,
            activation="gelu",
        )
        self.layers = nn.TransformerEncoder(layer, config.num_layers, norm=nn.LayerNorm(config.d_model))
        self.output = nn.Linear(config.d_model, len(grammar.vocab))

    @staticmethod
    def causal_mask(length: int, device: torch.device | None = None) -> Tensor:
        return torch.triu(torch.ones(length, length, dtype=torch.bool, device=device), diagonal=1)

    def forward(self, input_ids: Tensor, padding_mask: Tensor | None = None) -> Tensor:
        if input_ids.ndim != 2:
            raise ValueError("input_ids must have shape [batch, sequence]")
        if input_ids.shape[1] > self.max_sequence_length:
            raise ValueError("sequence exceeds learned positional embedding")
        positions = torch.arange(input_ids.shape[1], device=input_ids.device).unsqueeze(0)
        hidden = self.token_embedding(input_ids) + self.position_embedding(positions)
        if padding_mask is None:
            padding_mask = input_ids.eq(self.grammar.pad_id)
        encoded = self.layers(
            hidden,
            mask=self.causal_mask(input_ids.shape[1], input_ids.device),
            src_key_padding_mask=padding_mask,
        )
        return self.output(encoded)

    def generate(
        self,
        count: int,
        *,
        temperature: float | None = None,
        greedy: bool = False,
    ) -> list[GeneratedExpression]:
        if count < 1:
            return []
        temperature = self.config.temperature if temperature is None else temperature
        if temperature <= 0:
            raise ValueError("temperature must be positive")
        device = next(self.parameters()).device
        sequences: list[list[int]] = [[self.grammar.bos_id] for _ in range(count)]
        expression_ids: list[list[int]] = [[] for _ in range(count)]
        states = [self.grammar.initial_state() for _ in range(count)]
        active = [True] * count
        log_probs: list[list[Tensor]] = [[] for _ in range(count)]
        entropies: list[list[Tensor]] = [[] for _ in range(count)]

        for _ in range(self.config.max_tokens + 1):
            if not any(active):
                break
            ids = torch.tensor(sequences, dtype=torch.long, device=device)
            padding = ids.eq(self.grammar.pad_id)
            logits = self(ids, padding)[:, -1, :] / temperature
            for row in range(count):
                if not active[row]:
                    sequences[row].append(self.grammar.pad_id)
                    continue
                valid_ids = self.grammar.valid_token_ids(states[row])
                mask = torch.full_like(logits[row], float("-inf"))
                mask[valid_ids] = 0.0
                distribution = torch.distributions.Categorical(logits=logits[row] + mask)
                action = torch.argmax(distribution.logits) if greedy else distribution.sample()
                token_id = int(action.item())
                token = self.grammar.id_to_token[token_id]
                log_probs[row].append(distribution.log_prob(action))
                probabilities = distribution.probs
                safe_terms = torch.where(
                    probabilities > 0,
                    -probabilities * torch.log(probabilities.clamp_min(torch.finfo(probabilities.dtype).tiny)),
                    torch.zeros_like(probabilities),
                )
                entropies[row].append(safe_terms.sum())
                sequences[row].append(token_id)
                if token == "EOS":
                    active[row] = False
                else:
                    self.grammar.step(states[row], token)
                    expression_ids[row].append(token_id)
        if any(active):
            raise RuntimeError("Grammar-constrained decoding failed to terminate")

        generated: list[GeneratedExpression] = []
        for row in range(count):
            tree = self.grammar.decode_expression(expression_ids[row])
            generated.append(
                GeneratedExpression(
                    tree=tree,
                    token_ids=tuple(expression_ids[row]),
                    log_probability=torch.stack(log_probs[row]).sum(),
                    entropy=torch.stack(entropies[row]).mean(),
                )
            )
        return generated

    def teacher_forced(
        self,
        trees: list[Tree],
        *,
        temperature: float | None = None,
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Return likelihood under the same temperature-scaled policy as sampling."""
        if not trees:
            zero = self.output.weight.sum() * 0.0
            return zero.unsqueeze(0)[:0], zero, zero
        temperature = self.config.temperature if temperature is None else temperature
        if temperature <= 0:
            raise ValueError("temperature must be positive")
        sequences = [self.grammar.encode_expression(tree) for tree in trees]
        max_len = max(len(sequence) for sequence in sequences)
        padded = [sequence + [self.grammar.pad_id] * (max_len - len(sequence)) for sequence in sequences]
        tensor = torch.tensor(padded, dtype=torch.long, device=next(self.parameters()).device)
        inputs, targets = tensor[:, :-1], tensor[:, 1:]
        logits = self(inputs, inputs.eq(self.grammar.pad_id)) / temperature
        grammar_mask = torch.full_like(logits, float("-inf"))
        for row, tree in enumerate(trees):
            state = self.grammar.initial_state()
            target_tokens = [self.grammar.id_to_token[int(value)] for value in targets[row] if int(value) != self.grammar.pad_id]
            for column, token in enumerate(target_tokens):
                valid_ids = self.grammar.valid_token_ids(state)
                grammar_mask[row, column, valid_ids] = 0.0
                if token != "EOS":
                    self.grammar.step(state, token)
            if len(target_tokens) < targets.shape[1]:
                grammar_mask[row, len(target_tokens) :, :] = 0.0
        logits = logits + grammar_mask
        log_distribution = torch.log_softmax(logits, dim=-1)
        target_mask = targets.ne(self.grammar.pad_id)
        selected = log_distribution.gather(-1, targets.unsqueeze(-1)).squeeze(-1)
        selected = selected * target_mask
        sequence_log_prob = selected.sum(dim=1)
        token_count = target_mask.sum().clamp_min(1)
        nll = -selected.sum() / token_count
        probabilities = log_distribution.exp()
        token_entropy = -(
            probabilities
            * torch.log(probabilities.clamp_min(torch.finfo(probabilities.dtype).tiny))
        ).sum(dim=-1)
        entropy = (token_entropy * target_mask).sum() / token_count
        return sequence_log_prob, nll, entropy
