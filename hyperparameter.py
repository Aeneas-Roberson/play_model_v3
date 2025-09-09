# hyperparameter_configuration.py
"""
Hyperparameter Configuration System for CFB Hierarchical Model
6-phase training optimized for 12-hour sessions with Monte Carlo objectives
Target: <13 RMSE to beat Vegas (14+ RMSE)
"""

import jax
import jax.numpy as jnp
import optax
import flax.linen as nn
from typing import Dict, List, Tuple, Optional, Any, Callable
from dataclasses import dataclass, field
from enum import Enum
import numpy as np
import logging
import json
from pathlib import Path
import time
from datetime import datetime, timedelta


class TrainingPhase(Enum):
    """Training phase enumeration"""
    WARMUP = "phase_1_warmup"
    PLAY_FOCUS = "phase_2_play_focus"
    DRIVE_INTEGRATION = "phase_3_drive_integration"
    HIERARCHY_BALANCE = "phase_4_hierarchy_balance"
    GAME_EMPHASIS = "phase_5_game_emphasis"
    MONTE_CARLO_OPT = "phase_6_monte_carlo_optimization"


@dataclass
class ModelArchitectureConfig:
    """LSTM architecture configuration"""
    
    # Input dimensions
    embedding_features: int = 512
    dynamic_state_features: int = 20
    total_input_dims: int = 532
    
    # Bidirectional LSTM configuration
    lstm_layer_1_units: int = 256  # Per direction
    lstm_layer_2_units: int = 128  # Per direction
    lstm_dropout_1: float = 0.15
    lstm_dropout_2: float = 0.20
    lstm_recurrent_dropout_1: float = 0.15
    lstm_recurrent_dropout_2: float = 0.20
    
    # Prediction head dimensions
    play_head_hidden: List[int] = field(default_factory=lambda: [512, 256])
    drive_head_hidden: List[int] = field(default_factory=lambda: [384, 192])
    game_head_hidden: List[int] = field(default_factory=lambda: [512, 384, 256])
    
    # Output dimensions
    play_type_outputs: int = 8
    success_flags_outputs: int = 11
    yards_gained_outputs: int = 1
    drive_outcome_outputs: int = 10
    drive_metrics_outputs: int = 5
    game_final_scores_outputs: int = 3
    game_volume_stats_outputs: int = 8
    game_efficiency_stats_outputs: int = 10
    game_explosiveness_outputs: int = 12
    
    def get_total_parameters(self) -> int:
        """Estimate total model parameters"""
        # Simplified calculation
        lstm_params = (
            2 * self.lstm_layer_1_units * self.total_input_dims * 4 +  # Layer 1
            2 * self.lstm_layer_2_units * (2 * self.lstm_layer_1_units) * 4  # Layer 2
        )
        
        play_params = sum([
            self.play_head_hidden[0] * (2 * self.lstm_layer_2_units),
            self.play_head_hidden[1] * self.play_head_hidden[0],
            (self.play_type_outputs + self.success_flags_outputs + 
             self.yards_gained_outputs) * self.play_head_hidden[1]
        ])
        
        drive_params = sum([
            self.drive_head_hidden[0] * (2 * self.lstm_layer_2_units),
            self.drive_head_hidden[1] * self.drive_head_hidden[0],
            (self.drive_outcome_outputs + self.drive_metrics_outputs) * 
            self.drive_head_hidden[1]
        ])
        
        game_params = sum([
            self.game_head_hidden[0] * (2 * self.lstm_layer_2_units),
            self.game_head_hidden[1] * self.game_head_hidden[0],
            self.game_head_hidden[2] * self.game_head_hidden[1],
            33 * self.game_head_hidden[2]  # Total game outputs
        ])
        
        return lstm_params + play_params + drive_params + game_params


@dataclass
class PhaseConfig:
    """Configuration for a single training phase"""
    name: str
    duration_hours: float
    focus: str
    loss_weights: Dict[str, float]
    learning_rate: float
    batch_size: int
    description: str
    warmup_steps: int = 0
    decay_steps: int = 15000
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for serialization"""
        return {
            'name': self.name,
            'duration_hours': self.duration_hours,
            'focus': self.focus,
            'loss_weights': self.loss_weights,
            'learning_rate': self.learning_rate,
            'batch_size': self.batch_size,
            'description': self.description,
            'warmup_steps': self.warmup_steps,
            'decay_steps': self.decay_steps
        }


@dataclass
class TrainingPhasesConfig:
    """6-phase hierarchical training configuration"""
    
    def __init__(self):
        self.phases = self._create_phases()
        self.total_duration_hours = 48
        self.sessions_count = 4
        self.hours_per_session = 12
        
    def _create_phases(self) -> Dict[TrainingPhase, PhaseConfig]:
        """Create all 6 training phases"""
        return {
            TrainingPhase.WARMUP: PhaseConfig(
                name="phase_1_warmup",
                duration_hours=8.0,
                focus="Embedding stabilization + Play prediction foundation",
                loss_weights={
                    'play_level': 0.8,
                    'drive_level': 0.15,
                    'game_level': 0.05,
                    'state_consistency': 0.1
                },
                learning_rate=1e-4,
                batch_size=1536,
                description="Stabilize embeddings, learn basic play patterns",
                warmup_steps=1000,
                decay_steps=15000
            ),
            
            TrainingPhase.PLAY_FOCUS: PhaseConfig(
                name="phase_2_play_focus",
                duration_hours=8.0,
                focus="Play-level mastery + State transition learning",
                loss_weights={
                    'play_level': 0.7,
                    'drive_level': 0.2,
                    'game_level': 0.1,
                    'state_consistency': 0.2
                },
                learning_rate=2e-4,
                batch_size=1536,
                description="Master play predictions, learn state dynamics",
                warmup_steps=500,
                decay_steps=15000
            ),
            
            TrainingPhase.DRIVE_INTEGRATION: PhaseConfig(
                name="phase_3_drive_integration",
                duration_hours=8.0,
                focus="Drive-level prediction + Hierarchical consistency",
                loss_weights={
                    'play_level': 0.5,
                    'drive_level': 0.35,
                    'game_level': 0.15,
                    'state_consistency': 0.25
                },
                learning_rate=2e-4,
                batch_size=1536,
                description="Integrate drive outcomes, maintain play accuracy",
                warmup_steps=500,
                decay_steps=15000
            ),
            
            TrainingPhase.HIERARCHY_BALANCE: PhaseConfig(
                name="phase_4_hierarchy_balance",
                duration_hours=8.0,
                focus="Balanced hierarchical learning",
                loss_weights={
                    'play_level': 0.4,
                    'drive_level': 0.35,
                    'game_level': 0.25,
                    'state_consistency': 0.3
                },
                learning_rate=1.8e-4,
                batch_size=1536,
                description="Balance all three prediction levels",
                warmup_steps=300,
                decay_steps=15000
            ),
            
            TrainingPhase.GAME_EMPHASIS: PhaseConfig(
                name="phase_5_game_emphasis",
                duration_hours=8.0,
                focus="Game-level prediction mastery for Vegas performance",
                loss_weights={
                    'play_level': 0.3,
                    'drive_level': 0.3,
                    'game_level': 0.4,
                    'state_consistency': 0.35
                },
                learning_rate=1.5e-4,
                batch_size=1536,
                description="Focus on game statistics for Monte Carlo accuracy",
                warmup_steps=300,
                decay_steps=15000
            ),
            
            TrainingPhase.MONTE_CARLO_OPT: PhaseConfig(
                name="phase_6_monte_carlo_optimization",
                duration_hours=8.0,
                focus="State consistency + Simulation fidelity",
                loss_weights={
                    'play_level': 0.25,
                    'drive_level': 0.25,
                    'game_level': 0.5,
                    'state_consistency': 0.4
                },
                learning_rate=1e-4,
                batch_size=1536,
                description="Optimize for Monte Carlo simulation quality",
                warmup_steps=200,
                decay_steps=15000
            )
        }
    
    def get_phase(self, phase: TrainingPhase) -> PhaseConfig:
        """Get configuration for specific phase"""
        return self.phases[phase]
    
    def get_session_phases(self, session_num: int) -> List[TrainingPhase]:
        """Get phases for a specific session (1-4)"""
        session_mapping = {
            1: [TrainingPhase.WARMUP, TrainingPhase.PLAY_FOCUS],
            2: [TrainingPhase.DRIVE_INTEGRATION, TrainingPhase.HIERARCHY_BALANCE],
            3: [TrainingPhase.GAME_EMPHASIS, TrainingPhase.MONTE_CARLO_OPT],
            4: []  # Buffer/validation session
        }
        return session_mapping.get(session_num, [])


@dataclass
class LearningRateConfig:
    """Learning rate scheduling configuration"""
    
    strategy: str = "cosine_decay_with_warmup"
    base_learning_rate: float = 2e-4
    warmup_epochs: int = 2
    decay_strategy: str = "cosine"
    
    # Adaptive adjustments
    patience: int = 3
    factor: float = 0.8
    min_lr: float = 1e-6
    monitor_metric: str = "val_game_rmse"
    
    def create_schedule(self, phase_config: PhaseConfig) -> optax.Schedule:
        """Create Optax learning rate schedule for a phase"""
        
        if phase_config.warmup_steps > 0:
            # Warmup + cosine decay
            schedules = [
                optax.linear_schedule(
                    init_value=phase_config.learning_rate * 0.1,
                    end_value=phase_config.learning_rate,
                    transition_steps=phase_config.warmup_steps
                ),
                optax.cosine_decay_schedule(
                    init_value=phase_config.learning_rate,
                    decay_steps=phase_config.decay_steps - phase_config.warmup_steps,
                    alpha=0.1
                )
            ]
            
            boundaries = [phase_config.warmup_steps]
            schedule = optax.join_schedules(schedules, boundaries)
        else:
            # Just cosine decay
            schedule = optax.cosine_decay_schedule(
                init_value=phase_config.learning_rate,
                decay_steps=phase_config.decay_steps,
                alpha=0.1
            )
        
        return schedule


@dataclass
class LossFunctionConfig:
    """Multi-task loss configuration"""
    
    # Play-level losses
    play_type_loss: str = "categorical_crossentropy"
    play_type_weight: float = 0.3
    play_type_label_smoothing: float = 0.1
    
    success_flags_loss: str = "binary_crossentropy"
    success_flags_weight: float = 0.4
    success_flags_class_weight: str = "balanced"
    
    yards_gained_loss: str = "huber"
    yards_gained_weight: float = 0.3
    yards_gained_delta: float = 1.5
    
    # Drive-level losses
    drive_outcome_loss: str = "categorical_crossentropy"
    drive_outcome_weight: float = 0.6
    drive_outcome_label_smoothing: float = 0.05
    
    drive_metrics_loss: str = "mse"
    drive_metrics_weight: float = 0.4
    
    # Game-level losses (critical for Monte Carlo)
    home_points_weight: float = 0.33
    away_points_weight: float = 0.33
    point_margin_weight: float = 0.34
    target_game_rmse: float = 12.5
    
    volume_stats_weight: float = 0.25
    efficiency_stats_weight: float = 0.25
    explosiveness_stats_weight: float = 0.25
    
    # State consistency loss
    state_transition_validity_weight: float = 0.4
    hierarchical_consistency_weight: float = 0.4
    temporal_coherence_weight: float = 0.2
    
    # Penalty weights
    impossible_state_penalty: float = 10.0
    inconsistent_prediction_penalty: float = 5.0
    temporal_violation_penalty: float = 3.0
    
    def get_total_loss_function(self) -> Callable:
        """Create combined loss function"""
        # This would be implemented with actual JAX/Flax loss functions
        pass


@dataclass
class TPUOptimizationConfig:
    """TPU v2-8 specific optimizations"""
    
    hardware_target: str = "TPU v2-8"
    memory_capacity: str = "512GB HBM"
    
    # Batch configuration
    batch_size: int = 1536
    max_drives_per_game: int = 32
    max_plays_per_drive: int = 18
    total_sequence_length: int = 576
    gradient_accumulation_steps: int = 1
    drop_remainder: bool = True
    
    # Memory allocation estimates
    model_parameters_gb: float = 15.0
    embedding_cache_gb: float = 25.0
    sequence_buffers_gb: float = 180.0
    gradient_buffers_gb: float = 15.0
    state_management_gb: float = 12.0
    overhead_gb: float = 20.0
    total_estimated_gb: float = 267.0
    utilization_target: float = 0.52
    
    # Compilation optimizations
    xla_compilation: bool = True
    mixed_precision: bool = False  # Full fp32 for stability
    gradient_checkpointing: bool = True
    fused_ops: List[str] = field(default_factory=lambda: ['batch_norm', 'activation', 'dropout'])
    layout_optimization: bool = True
    
    # Data pipeline
    prefetch_buffer: str = "AUTOTUNE"
    parallel_reads: int = 16
    cache_dataset: bool = True
    shuffle_buffer: int = 10000
    interleave_cycle_length: int = 8
    
    def create_optimizer(self, learning_rate_schedule) -> optax.GradientTransformation:
        """Create TPU-optimized optimizer"""
        return optax.chain(
            optax.clip_by_global_norm(1.0),  # Gradient clipping
            optax.adamw(
                learning_rate=learning_rate_schedule,
                weight_decay=1e-5
            )
        )


@dataclass
class ValidationConfig:
    """Validation and early stopping configuration"""
    
    validation_frequency: str = "every_2_hours"
    validations_per_phase: int = 4
    
    # Early stopping
    monitor_metric: str = "val_game_rmse_combined"
    patience: int = 8  # 4 hours of no improvement
    min_delta: float = 0.1
    restore_best_weights: bool = True
    
    # Key metrics and targets
    target_game_rmse: float = 13.0  # Beat Vegas 14+
    target_state_consistency: float = 0.95
    target_hierarchical_alignment: float = 0.90
    target_simulation_fidelity: float = 0.88
    
    # Metric weights for combined score
    game_rmse_weight: float = 1.0
    state_consistency_weight: float = 0.3
    hierarchical_alignment_weight: float = 0.2
    simulation_fidelity_weight: float = 0.5
    
    # Data splits
    train_years: List[int] = field(default_factory=lambda: list(range(2015, 2022)))
    validation_years: List[int] = field(default_factory=lambda: [2022])
    test_years: List[int] = field(default_factory=lambda: [2023, 2024])
    temporal_weighting: str = "equal"
    conference_balance: str = "proportional"
    
    def should_stop_early(self, metrics_history: List[Dict], 
                          current_phase: int) -> bool:
        """Determine if early stopping should trigger"""
        if len(metrics_history) < self.patience:
            return False
        
        recent_metrics = [m[self.monitor_metric] for m in metrics_history[-self.patience:]]
        best_recent = min(recent_metrics)
        
        # Check if improvement is less than min_delta
        if len(metrics_history) > self.patience:
            previous_best = min([m[self.monitor_metric] 
                                for m in metrics_history[:-self.patience]])
            improvement = previous_best - best_recent
            
            if improvement < self.min_delta:
                return True
        
        return False


@dataclass
class RegularizationConfig:
    """Regularization configuration"""
    
    # Dropout rates
    embedding_dropout: float = 0.1
    lstm_dropout_1: float = 0.15
    lstm_recurrent_dropout_1: float = 0.15
    lstm_dropout_2: float = 0.20
    lstm_recurrent_dropout_2: float = 0.20
    play_head_dropout: List[float] = field(default_factory=lambda: [0.3, 0.25, 0.2])
    drive_head_dropout: List[float] = field(default_factory=lambda: [0.25, 0.2])
    game_head_dropout: List[float] = field(default_factory=lambda: [0.3, 0.25, 0.2])
    
    # Weight regularization
    l1_regularization: float = 0.0
    l2_regularization: float = 1e-5
    embedding_regularization: float = 1e-6
    
    # Gradient optimization
    gradient_clip_value: float = 1.0
    gradient_clip_method: str = "global_norm"
    gradient_noise_enabled: bool = False
    gradient_noise_stddev: float = 0.01
    
    # Batch normalization
    lstm_layer_norm: bool = True
    prediction_head_bn: bool = True
    bn_momentum: float = 0.99
    bn_epsilon: float = 1e-5


@dataclass
class CheckpointConfig:
    """Checkpointing and session management configuration"""
    
    max_session_duration_hours: float = 12.0
    checkpoint_frequency: str = "end_of_phase"
    auto_save_frequency_minutes: int = 30
    validation_checkpoint: bool = True
    
    # What to save
    save_model_weights: bool = True
    save_optimizer_state: bool = True
    save_training_state: bool = True
    save_validation_metrics: bool = True
    save_game_state_cache: bool = True
    
    # Session restart
    automatic_phase_detection: bool = True
    loss_weight_restoration: bool = True
    learning_rate_continuation: bool = True
    validation_state_restoration: bool = True
    warm_restart_epochs: int = 1
    
    checkpoint_dir: str = "/content/drive/MyDrive/cfb_model/checkpoints/"
    
    def get_checkpoint_path(self, phase_name: str, 
                           timestamp: Optional[str] = None) -> str:
        """Generate checkpoint file path"""
        if timestamp is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        return f"{self.checkpoint_dir}/{phase_name}_{timestamp}.ckpt"


@dataclass
class MonitoringConfig:
    """Performance monitoring configuration"""
    
    # Logging frequencies
    training_metrics_frequency: int = 100  # steps
    validation_metrics_hours: float = 2.0
    system_metrics_minutes: int = 10
    game_simulation_hours: float = 4.0
    
    # Key performance indicators
    target_game_rmse: float = 13.0
    vegas_rmse_baseline: float = 14.2
    improvement_target_percent: float = 8.5
    
    target_state_consistency: float = 0.95
    target_simulation_fidelity: float = 0.88
    target_hierarchical_alignment: float = 0.90
    
    target_tpu_utilization: float = 0.85
    target_memory_utilization: float = 0.52
    
    # Alert thresholds
    game_rmse_increase_alert: float = 0.5
    consistency_drop_alert: float = 0.02
    training_stall_minutes: int = 30
    
    memory_usage_alert: float = 0.90
    tpu_utilization_alert: float = 0.70
    gradient_explosion_threshold: float = 10.0
    
    # Logging configuration
    use_wandb: bool = True
    use_tensorboard: bool = True
    log_dir: str = "/content/drive/MyDrive/cfb_model/logs/"
    
    def should_alert(self, metric_name: str, value: float) -> bool:
        """Check if metric value should trigger alert"""
        alert_conditions = {
            'game_rmse': value > self.target_game_rmse + self.game_rmse_increase_alert,
            'state_consistency': value < self.target_state_consistency - self.consistency_drop_alert,
            'memory_usage': value > self.memory_usage_alert,
            'tpu_utilization': value < self.tpu_utilization_alert,
            'gradient_norm': value > self.gradient_explosion_threshold
        }
        
        return alert_conditions.get(metric_name, False)


@dataclass
class ExperimentConfig:
    """Hyperparameter experimentation configuration"""
    
    tuning_philosophy: str = "conservative_with_monitoring"
    
    # Phase 1 adjustments
    phase1_slow_convergence_lr_increase: float = 1.2e-4
    phase1_slow_convergence_dropout_reduce: float = 0.12
    phase1_unstable_lr_decrease: float = 8e-5
    phase1_unstable_gradient_clip: float = 0.8
    phase1_unstable_warmup_increase: int = 500
    
    # Game-level focus adjustments
    game_rmse_not_improving_loss_weight_increase: float = 0.05
    game_rmse_not_improving_neurons_add: int = 64
    game_overfitting_dropout_increase: float = 0.05
    game_overfitting_l2_increase: float = 1e-6
    game_overfitting_neurons_reduce: int = 32
    
    # Monte Carlo optimization
    poor_consistency_state_loss_increase: float = 0.05
    poor_consistency_add_regularization: bool = True
    poor_consistency_extend_phase6_hours: float = 2.0
    
    low_simulation_quality_hierarchical_weight_increase: float = 0.1
    low_simulation_quality_consistency_checks: str = "more_frequent"
    low_simulation_quality_bottom_up_favor: str = "play_level"
    
    def get_adjustment(self, issue: str, phase: int) -> Dict[str, Any]:
        """Get recommended adjustment for specific issue"""
        adjustments = {
            'slow_convergence': {
                'lr': self.phase1_slow_convergence_lr_increase,
                'dropout': self.phase1_slow_convergence_dropout_reduce
            },
            'unstable_training': {
                'lr': self.phase1_unstable_lr_decrease,
                'gradient_clip': self.phase1_unstable_gradient_clip
            },
            'game_rmse_stuck': {
                'loss_weight_increase': self.game_rmse_not_improving_loss_weight_increase,
                'add_neurons': self.game_rmse_not_improving_neurons_add
            },
            'overfitting': {
                'dropout_increase': self.game_overfitting_dropout_increase,
                'l2_increase': self.game_overfitting_l2_increase
            }
        }
        
        return adjustments.get(issue, {})


class HyperparameterConfiguration:
    """
    Complete hyperparameter configuration manager
    """
    
    def __init__(self, config_path: Optional[str] = None):
        self.architecture = ModelArchitectureConfig()
        self.training_phases = TrainingPhasesConfig()
        self.learning_rate = LearningRateConfig()
        self.loss_functions = LossFunctionConfig()
        self.tpu_optimization = TPUOptimizationConfig()
        self.validation = ValidationConfig()
        self.regularization = RegularizationConfig()
        self.checkpoint = CheckpointConfig()
        self.monitoring = MonitoringConfig()
        self.experiment = ExperimentConfig()
        
        self.logger = logging.getLogger('HyperparameterConfig')
        logging.basicConfig(level=logging.INFO)
        
        if config_path:
            self.load_config(config_path)
        
        self._validate_configuration()
    
    def _validate_configuration(self):
        """Validate configuration consistency"""
        # Check total training time
        total_phase_hours = sum(
            phase.duration_hours 
            for phase in self.training_phases.phases.values()
        )
        
        if abs(total_phase_hours - 48.0) > 0.1:
            self.logger.warning(
                f"⚠️ Total phase duration ({total_phase_hours}h) "
                f"doesn't match expected 48h"
            )
        
        # Check batch size consistency
        if self.tpu_optimization.batch_size != 1536:
            self.logger.warning(
                f"⚠️ Batch size {self.tpu_optimization.batch_size} "
                f"may not be optimal for TPU v2-8"
            )
        
        # Check memory usage
        if self.tpu_optimization.utilization_target > 0.6:
            self.logger.warning(
                f"⚠️ Memory utilization target {self.tpu_optimization.utilization_target} "
                f"may be too high for stability"
            )
        
        self.logger.info("✅ Configuration validated")
    
    def get_phase_config(self, phase: TrainingPhase) -> Dict[str, Any]:
        """Get complete configuration for a training phase"""
        phase_config = self.training_phases.get_phase(phase)
        
        return {
            'phase_config': phase_config.to_dict(),
            'optimizer': self._create_phase_optimizer(phase_config),
            'loss_weights': phase_config.loss_weights,
            'batch_size': phase_config.batch_size,
            'learning_rate_schedule': self.learning_rate.create_schedule(phase_config),
            'regularization': {
                'dropout': self._get_phase_dropout(phase),
                'l2': self.regularization.l2_regularization
            },
            'validation': {
                'frequency_hours': 2.0,
                'metrics': self._get_phase_metrics(phase)
            }
        }
    
    def _create_phase_optimizer(self, phase_config: PhaseConfig) -> optax.GradientTransformation:
        """Create optimizer for specific phase"""
        lr_schedule = self.learning_rate.create_schedule(phase_config)
        return self.tpu_optimization.create_optimizer(lr_schedule)
    
    def _get_phase_dropout(self, phase: TrainingPhase) -> Dict[str, float]:
        """Get dropout rates for specific phase"""
        # Adjust dropout based on phase
        base_dropout = {
            'embedding': self.regularization.embedding_dropout,
            'lstm_1': self.regularization.lstm_dropout_1,
            'lstm_2': self.regularization.lstm_dropout_2,
            'play_head': self.regularization.play_head_dropout,
            'drive_head': self.regularization.drive_head_dropout,
            'game_head': self.regularization.game_head_dropout
        }
        
        # Reduce dropout in early phases, increase in later phases
        if phase in [TrainingPhase.WARMUP, TrainingPhase.PLAY_FOCUS]:
            multiplier = 0.8
        elif phase in [TrainingPhase.GAME_EMPHASIS, TrainingPhase.MONTE_CARLO_OPT]:
            multiplier = 1.2
        else:
            multiplier = 1.0
        
        adjusted_dropout = {}
        for key, value in base_dropout.items():
            if isinstance(value, list):
                adjusted_dropout[key] = [v * multiplier for v in value]
            else:
                adjusted_dropout[key] = value * multiplier
        
        return adjusted_dropout
    
    def _get_phase_metrics(self, phase: TrainingPhase) -> List[str]:
        """Get metrics to track for specific phase"""
        base_metrics = [
            'loss', 'val_loss',
            'play_accuracy', 'val_play_accuracy',
            'game_rmse', 'val_game_rmse'
        ]
        
        if phase in [TrainingPhase.DRIVE_INTEGRATION, TrainingPhase.HIERARCHY_BALANCE]:
            base_metrics.extend(['drive_accuracy', 'val_drive_accuracy'])
        
        if phase in [TrainingPhase.GAME_EMPHASIS, TrainingPhase.MONTE_CARLO_OPT]:
            base_metrics.extend([
                'state_consistency', 'val_state_consistency',
                'simulation_fidelity', 'val_simulation_fidelity'
            ])
        
        return base_metrics
    
    def get_session_schedule(self, session_num: int) -> List[Dict[str, Any]]:
        """Get training schedule for a 12-hour session"""
        if session_num < 1 or session_num > 4:
            raise ValueError(f"Invalid session number: {session_num}")
        
        phases = self.training_phases.get_session_phases(session_num)
        schedule = []
        
        for phase in phases:
            phase_info = self.get_phase_config(phase)
            phase_info['session'] = session_num
            phase_info['estimated_duration'] = self.training_phases.get_phase(phase).duration_hours
            schedule.append(phase_info)
        
        return schedule
    
    def save_config(self, path: str):
        """Save configuration to JSON file"""
        config_dict = {
            'architecture': self.architecture.__dict__,
            'training_phases': {
                phase.name: config.to_dict() 
                for phase, config in self.training_phases.phases.items()
            },
            'learning_rate': self.learning_rate.__dict__,
            'loss_functions': self.loss_functions.__dict__,
            'tpu_optimization': self.tpu_optimization.__dict__,
            'validation': {
                k: v if not isinstance(v, list) else list(v)
                for k, v in self.validation.__dict__.items()
            },
            'regularization': {
                k: v if not isinstance(v, list) else list(v)
                for k, v in self.regularization.__dict__.items()
            },
            'checkpoint': self.checkpoint.__dict__,
            'monitoring': self.monitoring.__dict__,
            'experiment': self.experiment.__dict__
        }
        
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(path, 'w') as f:
            json.dump(config_dict, f, indent=2, default=str)
        
        self.logger.info(f"💾 Configuration saved to {path}")
    
    def load_config(self, path: str):
        """Load configuration from JSON file"""
        path = Path(path)
        
        if not path.exists():
            self.logger.warning(f"⚠️ Config file not found: {path}")
            return
        
        with open(path, 'r') as f:
            config_dict = json.load(f)
        
        # Update configurations
        for key, value in config_dict.get('architecture', {}).items():
            if hasattr(self.architecture, key):
                setattr(self.architecture, key, value)
        
        # Load other configurations similarly...
        
        self.logger.info(f"✅ Configuration loaded from {path}")
    
    def estimate_training_time(self) -> Dict[str, Any]:
        """Estimate total training time and checkpoints"""
        estimates = {
            'total_hours': 48,
            'sessions': 4,
            'hours_per_session': 12,
            'phases': 6,
            'hours_per_phase': 8,
            'checkpoints_per_phase': 16,  # Every 30 minutes
            'total_checkpoints': 96,
            'validation_runs': 24,  # Every 2 hours
            'estimated_completion': datetime.now() + timedelta(hours=48)
        }
        
        # Estimate steps per phase
        batch_size = self.tpu_optimization.batch_size
        steps_per_hour = 3600 / 2  # Assume 2 seconds per step
        
        for phase in TrainingPhase:
            phase_config = self.training_phases.get_phase(phase)
            steps = int(steps_per_hour * phase_config.duration_hours)
            estimates[f'{phase.name}_steps'] = steps
        
        return estimates
    
    def get_expected_outcomes(self) -> Dict[str, Any]:
        """Get expected training outcomes by phase"""
        return {
            'phase_1_completion': {
                'hours': 8,
                'expected_play_accuracy': 0.75,
                'expected_game_rmse': 18.0,
                'state_consistency': 0.85
            },
            'phase_3_completion': {
                'hours': 24,
                'expected_play_accuracy': 0.82,
                'expected_game_rmse': 16.0,
                'state_consistency': 0.90
            },
            'phase_6_completion': {
                'hours': 48,
                'expected_play_accuracy': 0.85,
                'expected_game_rmse': 12.5,  # Beat Vegas target
                'state_consistency': 0.95,
                'monte_carlo_ready': True
            },
            'final_model': {
                'total_parameters': f"{self.architecture.get_total_parameters() / 1e6:.1f}M",
                'inference_speed_ms': 100,
                'memory_footprint_gb': 8,
                'monte_carlo_throughput': 1000,  # games/hour
                'vegas_performance': '8.5% better RMSE'
            }
        }


class AdaptiveHyperparameterTuner:
    """
    Adaptive hyperparameter tuning during training
    """
    
    def __init__(self, base_config: HyperparameterConfiguration):
        self.base_config = base_config
        self.adjustment_history = []
        self.performance_history = []
        self.logger = logging.getLogger('AdaptiveTuner')
    
    def analyze_performance(self, metrics: Dict[str, float], 
                           phase: TrainingPhase,
                           elapsed_hours: float) -> Optional[Dict[str, Any]]:
        """
        Analyze performance and suggest adjustments
        """
        self.performance_history.append({
            'phase': phase.name,
            'elapsed_hours': elapsed_hours,
            'metrics': metrics
        })
        
        # Check for issues
        issues = self._detect_issues(metrics, phase, elapsed_hours)
        
        if issues:
            adjustments = self._recommend_adjustments(issues, phase)
            
            if adjustments:
                self.adjustment_history.append({
                    'phase': phase.name,
                    'elapsed_hours': elapsed_hours,
                    'issues': issues,
                    'adjustments': adjustments
                })
                
                return adjustments
        
        return None
    
    def _detect_issues(self, metrics: Dict[str, float],
                       phase: TrainingPhase,
                       elapsed_hours: float) -> List[str]:
        """Detect training issues"""
        issues = []
        
        # Check convergence speed
        if elapsed_hours > 2 and metrics.get('loss', float('inf')) > 1.0:
            issues.append('slow_convergence')
        
        # Check stability
        if len(self.performance_history) > 5:
            recent_losses = [h['metrics'].get('loss', 0) 
                           for h in self.performance_history[-5:]]
            if np.std(recent_losses) > 0.5:
                issues.append('unstable_training')
        
        # Check game RMSE
        if phase in [TrainingPhase.GAME_EMPHASIS, TrainingPhase.MONTE_CARLO_OPT]:
            game_rmse = metrics.get('val_game_rmse', float('inf'))
            if game_rmse > self.base_config.validation.target_game_rmse:
                issues.append('game_rmse_stuck')
        
        # Check overfitting
        train_loss = metrics.get('loss', 0)
        val_loss = metrics.get('val_loss', float('inf'))
        if val_loss > train_loss * 1.5:
            issues.append('overfitting')
        
        # Check state consistency
        if phase == TrainingPhase.MONTE_CARLO_OPT:
            consistency = metrics.get('state_consistency', 0)
            if consistency < self.base_config.validation.target_state_consistency:
                issues.append('poor_consistency')
        
        return issues
    
    def _recommend_adjustments(self, issues: List[str],
                              phase: TrainingPhase) -> Dict[str, Any]:
        """Recommend hyperparameter adjustments"""
        adjustments = {}
        
        for issue in issues:
            adjustment = self.base_config.experiment.get_adjustment(issue, phase.value)
            adjustments.update(adjustment)
        
        return adjustments if adjustments else None
    
    def apply_adjustments(self, adjustments: Dict[str, Any],
                         current_config: Dict[str, Any]) -> Dict[str, Any]:
        """Apply recommended adjustments to current configuration"""
        updated_config = current_config.copy()
        
        for key, value in adjustments.items():
            if key in updated_config:
                if isinstance(value, (int, float)):
                    # Numerical adjustment
                    if 'increase' in key:
                        updated_config[key] = updated_config[key] * (1 + value)
                    elif 'decrease' in key:
                        updated_config[key] = updated_config[key] * (1 - value)
                    else:
                        updated_config[key] = value
                else:
                    updated_config[key] = value
        
        self.logger.info(f"📊 Applied adjustments: {adjustments}")
        
        return updated_config


# Utility functions
def create_hyperparameter_config(config_path: Optional[str] = None) -> HyperparameterConfiguration:
    """Factory function to create hyperparameter configuration"""
    return HyperparameterConfiguration(config_path)

def create_adaptive_tuner(base_config: HyperparameterConfiguration) -> AdaptiveHyperparameterTuner:
    """Factory function to create adaptive tuner"""
    return AdaptiveHyperparameterTuner(base_config)

def get_phase_schedule() -> Dict[int, List[str]]:
    """Get phase schedule for training sessions"""
    return {
        1: ["phase_1_warmup", "phase_2_play_focus"],           # Session 1: 0-12h
        2: ["phase_3_drive_integration", "phase_4_hierarchy_balance"],  # Session 2: 12-24h
        3: ["phase_5_game_emphasis", "phase_6_monte_carlo_optimization"],  # Session 3: 24-36h
        4: []  # Session 4: Buffer/validation (36-48h)
    }