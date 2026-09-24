from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass(slots=True)
class GPRTConfig:
    # Transformer (project adaptation; not claimed as paper-original hyperparameters).
    d_model: int = 64
    nhead: int = 4
    num_layers: int = 2
    dim_feedforward: int = 128
    dropout: float = 0.1
    max_tokens: int = 63
    max_depth: int = 6
    learning_rate: float = 1e-3
    gradient_clip: float = 1.0
    temperature: float = 1.0

    # Neural-GP loop.
    population_size: int = 64
    transformer_samples: int = 32
    standalone_rl_samples_per_cycle: int = 352
    tournament_size: int = 5
    elite_count: int = 2
    crossover_rate: float = 0.6
    mutation_rate: float = 0.3
    reproduction_rate: float = 0.1
    gp_generations: int = 3
    cycles: int = 32
    batch_size_instances: int = 4
    validation_interval: int = 4
    validation_candidates: int = 16
    screening_train_size: int = 8
    elite_screening_candidates: int = 8
    behavior_probe_size: int = 1
    archive_seed_count: int = 0
    complexity_penalty: float = 1e-4

    # Policy optimization.
    alpha: float = 0.1
    beta: float = 0.001
    elite_fraction: float = 0.10
    baseline_decay: float = 0.9
    advantage_mode: str = "batch_standardize"

    # Optional structural and feature ablations. New feature vocabulary makes
    # pre-improvement checkpoints intentionally incompatible with this protocol.
    typed_grammar: bool = False
    use_lookahead_features: bool = True
    require_full_train_coverage: bool = True

    # Ablations.
    use_transformer_seeding: bool = True
    use_gp: bool = True
    use_elite_imitation: bool = True
    use_reinforce: bool = True

    def validate(self) -> None:
        if self.d_model % self.nhead:
            raise ValueError("d_model must be divisible by nhead")
        if self.transformer_samples > self.population_size:
            raise ValueError("transformer_samples cannot exceed population_size")
        rates = self.crossover_rate + self.mutation_rate + self.reproduction_rate
        if abs(rates - 1.0) > 1e-9:
            raise ValueError("GP crossover/mutation/reproduction rates must sum to 1")
        if min(self.max_tokens, self.population_size, self.cycles, self.standalone_rl_samples_per_cycle) < 1:
            raise ValueError("positive token, population, and cycle limits are required")
        if min(
            self.batch_size_instances,
            self.validation_candidates,
            self.screening_train_size,
            self.elite_screening_candidates,
            self.behavior_probe_size,
        ) < 1:
            raise ValueError("screening, validation, batch, and probe sizes must be positive")
        if self.archive_seed_count < 0:
            raise ValueError("archive_seed_count cannot be negative")
        random_seed_slots = self.population_size - (
            self.transformer_samples if self.use_transformer_seeding else 0
        )
        if self.archive_seed_count > random_seed_slots:
            raise ValueError("archive seeds must replace random seeds, not increase the population")
        if self.archive_seed_count and not self.use_gp:
            raise ValueError("archive seeding requires GP")
        if self.advantage_mode not in {"ewma_scale", "batch_standardize", "none"}:
            raise ValueError("advantage_mode must be ewma_scale, batch_standardize, or none")

    def to_dict(self) -> dict[str, object]:
        return asdict(self)
