# ⚙️ Hyperparameter Configuration Design Document

## Executive Summary

This document provides complete hyperparameter specifications for the CFB hierarchical model optimized for TPU v2-8 training with Monte Carlo simulation objectives. The configuration implements 6-phase hierarchical training designed for 12-hour session constraints, targeting <13 RMSE for game-level predictions to outperform Vegas accuracy (14+ RMSE).

**🎯 Key Design Goals:**
- **Monte Carlo Optimization**: State consistency prioritized for simulation fidelity
- **12-Hour Session Compatibility**: 6 phases of ~8 hours each with seamless checkpointing
- **Game Stats Focus**: Home/away scores and margin predictions weighted equally for betting insights
- **TPU v2-8 Efficiency**: 1536 batch size for optimal memory/compute balance
- **Sub-Vegas Accuracy**: Target <13 RMSE vs Vegas 14+ RMSE on game predictions
- **Hierarchical Learning**: Joint training with curriculum learning (Option B approach)

---

## 🏗️ Model Architecture Configuration

### LSTM Architecture Specification

```python
HIERARCHICAL_LSTM_CONFIG = {
    # INPUT LAYER
    'input_dimensions': {
        'embedding_features': 512,  # From 5 embedding containers
        'dynamic_state_features': 20,  # From game state manager
        'total_input_dims': 532
    },
    
    # BIDIRECTIONAL LSTM STACK
    'lstm_layers': {
        'layer_1': {
            'units_per_direction': 256,
            'total_units': 512,  # Bidirectional
            'return_sequences': True,
            'dropout': 0.15,
            'recurrent_dropout': 0.15
        },
        'layer_2': {
            'units_per_direction': 128,
            'total_units': 256,  # Bidirectional
            'return_sequences': False,  # Final state for prediction heads
            'dropout': 0.20,
            'recurrent_dropout': 0.20
        }
    },
    
    # HIERARCHICAL PREDICTION HEADS
    'prediction_heads': {
        'play_head': {
            'input_dims': 256,
            'hidden_layers': [512, 256],
            'output_tasks': {
                'play_type': 8,      # Classification
                'success_flags': 11,  # Multi-binary
                'yards_gained': 1     # Regression
            },
            'total_outputs': 20
        },
        'drive_head': {
            'input_dims': 256,
            'hidden_layers': [384, 192],
            'output_tasks': {
                'drive_outcome': 10,  # Classification (TD, FG, Punt, etc.)
                'drive_metrics': 5    # Regression (yards, plays, time, etc.)
            },
            'total_outputs': 15
        },
        'game_head': {
            'input_dims': 256,
            'hidden_layers': [512, 384, 256],  # Deeper for complex game stats
            'output_tasks': {
                'final_scores': 3,        # home_points, away_points, margin
                'volume_stats': 8,        # rushing/passing yards and attempts
                'efficiency_stats': 10,   # YPR, YPA, completion %, etc.
                'explosiveness': 12       # Explosive play metrics
            },
            'total_outputs': 33  # Primary focus for Monte Carlo
        }
    }
}
```

---

## 📚 6-Phase Hierarchical Training Strategy

### Phase Structure for 12-Hour Session Compatibility

```python
HIERARCHICAL_TRAINING_PHASES = {
    'total_duration': '48 hours (6 phases × 8 hours)',
    'session_structure': '4 sessions × 12 hours each',
    'checkpoint_strategy': 'save_every_phase',
    
    'phase_1_warmup': {
        'duration': '8 hours',
        'session': 'Session 1 (0-12h)',
        'focus': 'Embedding stabilization + Play prediction foundation',
        'loss_weights': {
            'play_level': 0.8,
            'drive_level': 0.15,
            'game_level': 0.05,
            'state_consistency': 0.1
        },
        'learning_rate': 1e-4,  # Conservative start
        'batch_size': 1536,
        'description': 'Stabilize embeddings, learn basic play patterns'
    },
    
    'phase_2_play_focus': {
        'duration': '8 hours', 
        'session': 'Session 1 (0-12h)',
        'focus': 'Play-level mastery + State transition learning',
        'loss_weights': {
            'play_level': 0.7,
            'drive_level': 0.2,
            'game_level': 0.1,
            'state_consistency': 0.2
        },
        'learning_rate': 2e-4,  # Increase for active learning
        'batch_size': 1536,
        'description': 'Master play predictions, learn state dynamics'
    },
    
    'phase_3_drive_integration': {
        'duration': '8 hours',
        'session': 'Session 2 (12-24h)', 
        'focus': 'Drive-level prediction + Hierarchical consistency',
        'loss_weights': {
            'play_level': 0.5,
            'drive_level': 0.35,
            'game_level': 0.15,
            'state_consistency': 0.25
        },
        'learning_rate': 2e-4,
        'batch_size': 1536,
        'description': 'Integrate drive outcomes, maintain play accuracy'
    },
    
    'phase_4_hierarchy_balance': {
        'duration': '8 hours',
        'session': 'Session 2 (12-24h)',
        'focus': 'Balanced hierarchical learning',
        'loss_weights': {
            'play_level': 0.4,
            'drive_level': 0.35, 
            'game_level': 0.25,
            'state_consistency': 0.3
        },
        'learning_rate': 1.8e-4,  # Slight decay for stability
        'batch_size': 1536,
        'description': 'Balance all three prediction levels'
    },
    
    'phase_5_game_emphasis': {
        'duration': '8 hours',
        'session': 'Session 3 (24-36h)',
        'focus': 'Game-level prediction mastery for Vegas performance',
        'loss_weights': {
            'play_level': 0.3,
            'drive_level': 0.3,
            'game_level': 0.4,  # Emphasize game stats
            'state_consistency': 0.35
        },
        'learning_rate': 1.5e-4,
        'batch_size': 1536,
        'description': 'Focus on game statistics for Monte Carlo accuracy'
    },
    
    'phase_6_monte_carlo_optimization': {
        'duration': '8 hours',
        'session': 'Session 3 (24-36h)',
        'focus': 'State consistency + Simulation fidelity',
        'loss_weights': {
            'play_level': 0.25,
            'drive_level': 0.25, 
            'game_level': 0.5,   # Maximum game focus
            'state_consistency': 0.4  # Critical for Monte Carlo
        },
        'learning_rate': 1e-4,  # Conservative for fine-tuning
        'batch_size': 1536,
        'description': 'Optimize for Monte Carlo simulation quality'
    }
}
```

---

## 🎯 Learning Rate Schedule Configuration

### Adaptive Learning Rate Strategy

```python
LEARNING_RATE_CONFIG = {
    'strategy': 'cosine_decay_with_warmup',
    'base_learning_rate': 2e-4,  # Optimal for large sequence models
    'warmup_epochs': 2,          # Per phase
    'decay_strategy': 'cosine',
    
    'phase_specific_rates': {
        'phase_1': {
            'initial_lr': 1e-4,
            'peak_lr': 1.5e-4,
            'final_lr': 1.2e-4,
            'warmup_steps': 1000,
            'decay_steps': 15000
        },
        'phase_2': {
            'initial_lr': 1.2e-4,
            'peak_lr': 2e-4, 
            'final_lr': 1.8e-4,
            'warmup_steps': 500,
            'decay_steps': 15000
        },
        'phase_3': {
            'initial_lr': 1.8e-4,
            'peak_lr': 2e-4,
            'final_lr': 1.8e-4,
            'warmup_steps': 500,
            'decay_steps': 15000
        },
        'phase_4': {
            'initial_lr': 1.8e-4,
            'peak_lr': 1.8e-4,
            'final_lr': 1.6e-4,
            'warmup_steps': 300,
            'decay_steps': 15000
        },
        'phase_5': {
            'initial_lr': 1.6e-4,
            'peak_lr': 1.5e-4,
            'final_lr': 1.2e-4,
            'warmup_steps': 300,
            'decay_steps': 15000
        },
        'phase_6': {
            'initial_lr': 1.2e-4,
            'peak_lr': 1e-4,
            'final_lr': 5e-5,
            'warmup_steps': 200,
            'decay_steps': 15000
        }
    },
    
    'adaptive_adjustments': {
        'patience': 3,  # Epochs before LR adjustment
        'factor': 0.8,  # Reduction factor
        'min_lr': 1e-6,
        'monitor_metric': 'val_game_rmse'  # Key metric for Vegas performance
    }
}
```

---

## ⚖️ Loss Function Configuration

### Multi-Task Loss Weighting Strategy

```python
LOSS_FUNCTION_CONFIG = {
    'total_loss_components': 4,
    
    # PLAY-LEVEL LOSSES
    'play_level_losses': {
        'play_type_classification': {
            'loss_function': 'categorical_crossentropy',
            'weight': 0.3,
            'label_smoothing': 0.1
        },
        'success_flags': {
            'loss_function': 'binary_crossentropy', 
            'weight': 0.4,
            'class_weight': 'balanced'  # Handle imbalanced success rates
        },
        'yards_gained': {
            'loss_function': 'huber',  # Robust to outliers
            'weight': 0.3,
            'delta': 1.5  # Huber parameter
        }
    },
    
    # DRIVE-LEVEL LOSSES
    'drive_level_losses': {
        'drive_outcome': {
            'loss_function': 'categorical_crossentropy',
            'weight': 0.6,
            'label_smoothing': 0.05
        },
        'drive_metrics': {
            'loss_function': 'mse',
            'weight': 0.4
        }
    },
    
    # GAME-LEVEL LOSSES (CRITICAL FOR MONTE CARLO)
    'game_level_losses': {
        'final_scores': {
            'home_points': {
                'loss_function': 'mse',
                'weight': 0.33,  # Equal weighting as requested
                'target_rmse': 12.5  # Sub-Vegas target
            },
            'away_points': {
                'loss_function': 'mse', 
                'weight': 0.33,  # Equal weighting
                'target_rmse': 12.5
            },
            'point_margin': {
                'loss_function': 'mse',
                'weight': 0.34,  # Slightly higher for betting insights
                'target_rmse': 12.0  # Most critical for betting
            }
        },
        'volume_stats': {
            'loss_function': 'mse',
            'weight': 0.25,
            'normalization': 'z_score'  # Normalize for different stat scales
        },
        'efficiency_stats': {
            'loss_function': 'mse',
            'weight': 0.25,
            'normalization': 'min_max'
        },
        'explosiveness_stats': {
            'loss_function': 'mse',
            'weight': 0.25,
            'normalization': 'robust_scale'
        }
    },
    
    # STATE CONSISTENCY LOSS (CRITICAL FOR MONTE CARLO)
    'state_consistency_loss': {
        'loss_function': 'custom_state_consistency',
        'components': {
            'state_transition_validity': 0.4,  # Valid game state changes
            'hierarchical_consistency': 0.4,   # Play→Drive→Game alignment
            'temporal_coherence': 0.2          # Time progression validity
        },
        'penalty_weights': {
            'impossible_state': 10.0,  # Heavy penalty for invalid states
            'inconsistent_prediction': 5.0,
            'temporal_violation': 3.0
        }
    }
}
```

---

## 🚀 TPU v2-8 Optimization Configuration

### Memory and Compute Optimization

```python
TPU_OPTIMIZATION_CONFIG = {
    'hardware_target': 'TPU v2-8',
    'memory_capacity': '512GB HBM',
    
    'batch_configuration': {
        'batch_size': 1536,  # Optimal TPU utilization
        'sequence_length': {
            'max_drives_per_game': 32,
            'max_plays_per_drive': 18,
            'total_sequence_length': 576  # 32 × 18
        },
        'gradient_accumulation_steps': 1,  # Not needed with large batches
        'drop_remainder': True  # Required for TPU
    },
    
    'memory_allocation': {
        'model_parameters': '~15GB',
        'embedding_cache': '~25GB', 
        'sequence_buffers': '~180GB',  # Large sequences
        'gradient_buffers': '~15GB',
        'state_management': '~12GB',
        'overhead': '~20GB',
        'total_estimated': '~267GB',
        'utilization_target': '~52%'  # Conservative for stability
    },
    
    'compilation_optimizations': {
        'xla_compilation': True,
        'mixed_precision': False,  # Full fp32 for numerical stability
        'gradient_checkpointing': True,  # Memory efficiency
        'fused_ops': ['batch_norm', 'activation', 'dropout'],
        'layout_optimization': True
    },
    
    'data_pipeline_optimization': {
        'prefetch_buffer': 'AUTOTUNE',
        'parallel_reads': 16,
        'cache_dataset': True,  # 512GB allows full caching
        'shuffle_buffer': 10000,
        'interleave_cycle_length': 8
    }
}
```

---

## 📊 Validation and Early Stopping Configuration

### Monte Carlo Focused Validation Strategy

```python
VALIDATION_CONFIG = {
    'validation_frequency': 'every_2_hours',  # 4 validations per phase
    'early_stopping': {
        'monitor_metric': 'val_game_rmse_combined',
        'patience': 8,  # 4 hours of no improvement
        'min_delta': 0.1,  # Minimum RMSE improvement
        'restore_best_weights': True
    },
    
    'key_metrics': {
        'primary_target': {
            'game_rmse_combined': {
                'home_score_rmse': 0.33,
                'away_score_rmse': 0.33, 
                'margin_rmse': 0.34,
                'target_threshold': 13.0,  # Beat Vegas 14+
                'weight': 1.0
            }
        },
        'secondary_targets': {
            'state_consistency_rate': {
                'target_threshold': 0.95,  # 95% valid state transitions
                'weight': 0.3
            },
            'hierarchical_alignment': {
                'target_threshold': 0.90,  # 90% consistent predictions
                'weight': 0.2
            },
            'simulation_fidelity': {
                'target_threshold': 0.88,  # Monte Carlo quality
                'weight': 0.5
            }
        }
    },
    
    'validation_splits': {
        'training': '2015-2021 (7 seasons)',
        'validation': '2022 (1 season)',
        'test': '2023-2024 (2 seasons)',
        'temporal_weighting': 'equal',  # Start equal, can adjust later
        'conference_balance': 'proportional'
    }
}
```

---

## 🔧 Regularization Configuration

### Overfitting Prevention for Large Model

```python
REGULARIZATION_CONFIG = {
    'dropout_strategy': {
        'embedding_dropout': 0.1,   # Light dropout for embeddings
        'lstm_dropout': 0.15,       # Layer 1 LSTM
        'lstm_recurrent_dropout': 0.15,
        'lstm_layer2_dropout': 0.20,
        'lstm_layer2_recurrent_dropout': 0.20,
        'prediction_head_dropout': [0.3, 0.25, 0.2],  # Play, Drive, Game heads
    },
    
    'weight_regularization': {
        'l1_regularization': 0.0,   # No L1 for sparse features
        'l2_regularization': 1e-5,  # Light L2 for stability
        'embedding_regularization': 1e-6  # Very light for embeddings
    },
    
    'gradient_optimization': {
        'gradient_clipping': {
            'method': 'global_norm',
            'clip_value': 1.0
        },
        'gradient_noise': {
            'enabled': False  # Start without, can enable if needed
        }
    },
    
    'batch_normalization': {
        'lstm_layer_norm': True,    # Layer normalization in LSTM
        'prediction_head_bn': True, # Batch norm in heads
        'momentum': 0.99,
        'epsilon': 1e-5
    }
}
```

---

## 💾 Checkpointing and Session Management

### 12-Hour Session Compatibility

```python
CHECKPOINT_CONFIG = {
    'session_management': {
        'max_session_duration': '12 hours',
        'checkpoint_frequency': 'end_of_phase',  # Every 8 hours
        'auto_save_frequency': '30 minutes',     # Safety saves
        'validation_checkpoint': True            # Save after validation
    },
    
    'checkpoint_structure': {
        'model_state': {
            'lstm_weights': True,
            'embedding_weights': True, 
            'prediction_head_weights': True,
            'optimizer_state': True
        },
        'training_state': {
            'current_phase': True,
            'epoch_within_phase': True,
            'global_step': True,
            'loss_history': True,
            'learning_rate_state': True
        },
        'validation_metrics': {
            'best_game_rmse': True,
            'validation_history': True,
            'early_stopping_counter': True
        },
        'game_state_management': {
            'state_cache_status': True,
            'consistency_violation_stats': True,
            'simulation_quality_metrics': True
        }
    },
    
    'session_restart_protocol': {
        'automatic_phase_detection': True,
        'loss_weight_restoration': True,
        'learning_rate_continuation': True,
        'validation_state_restoration': True,
        'warm_restart_epochs': 1  # Brief warm-up after restart
    }
}
```

---

## 📈 Performance Monitoring Configuration

### Real-Time Training Insights

```python
MONITORING_CONFIG = {
    'logging_frequency': {
        'training_metrics': 'every_100_steps',
        'validation_metrics': 'every_2_hours',  
        'system_metrics': 'every_10_minutes',
        'game_simulation_quality': 'every_4_hours'
    },
    
    'key_performance_indicators': {
        'vegas_comparison': {
            'target_game_rmse': '<13.0',
            'current_vegas_rmse': '~14.2',
            'improvement_target': '8.5% better than Vegas'
        },
        'monte_carlo_readiness': {
            'state_consistency_rate': '>95%',
            'simulation_fidelity_score': '>88%',
            'hierarchical_alignment': '>90%'
        },
        'training_efficiency': {
            'tpu_utilization': '>85%',
            'memory_utilization': '~52%',
            'training_speed': 'steps_per_second'
        }
    },
    
    'alert_thresholds': {
        'performance_degradation': {
            'game_rmse_increase': '+0.5',  # Alert if RMSE jumps
            'consistency_drop': '-2%',     # Alert if state consistency drops
            'training_stall': '30_minutes' # Alert if no progress
        },
        'system_issues': {
            'memory_usage': '>90%',
            'tpu_utilization': '<70%',
            'gradient_explosion': 'norm > 10.0'
        }
    }
}
```

---

## 🧪 Hyperparameter Experimentation Framework

### Adaptive Tuning Strategy

```python
HYPERPARAMETER_TUNING_CONFIG = {
    'tuning_philosophy': 'conservative_with_monitoring',
    
    'phase_1_adjustments': {
        'if_slow_convergence': {
            'increase_lr': 'to 1.2e-4',
            'reduce_dropout': 'to 0.12',
            'increase_batch_size': 'if memory allows'
        },
        'if_unstable_training': {
            'decrease_lr': 'to 8e-5', 
            'increase_gradient_clipping': 'to 0.8',
            'add_warmup_steps': '+500'
        }
    },
    
    'game_level_focus_adjustments': {
        'if_rmse_not_improving': {
            'increase_game_loss_weight': '+0.05',
            'add_game_head_capacity': '+64_neurons',
            'adjust_lr_schedule': 'plateau_detection'
        },
        'if_overfitting_game_stats': {
            'increase_game_head_dropout': '+0.05',
            'add_l2_regularization': '+1e-6',
            'reduce_game_head_capacity': '-32_neurons'
        }
    },
    
    'monte_carlo_optimization': {
        'if_poor_state_consistency': {
            'increase_state_loss_weight': '+0.05',
            'add_state_regularization': True,
            'extend_phase_6_duration': '+2_hours'
        },
        'if_simulation_quality_low': {
            'increase_hierarchical_loss_weight': '+0.1',
            'add_consistency_checks': 'more_frequent',
            'adjust_bottom_up_weighting': 'favor_play_level'
        }
    }
}
```

---

## 🎯 Expected Training Outcomes

### Performance Targets and Timeline

```python
EXPECTED_OUTCOMES = {
    'training_timeline': {
        'phase_1_completion': {
            'hours': 8,
            'expected_play_accuracy': '~75%',
            'expected_game_rmse': '~18-20',
            'state_consistency': '~85%'
        },
        'phase_3_completion': {
            'hours': 24, 
            'expected_play_accuracy': '~82%',
            'expected_game_rmse': '~16-17',
            'state_consistency': '~90%'
        },
        'phase_6_completion': {
            'hours': 48,
            'expected_play_accuracy': '~85%',
            'expected_game_rmse': '<13',  # Beat Vegas target
            'state_consistency': '~95%',
            'monte_carlo_ready': True
        }
    },
    
    'final_model_specifications': {
        'total_parameters': '~42M',
        'inference_speed': '<100ms per game simulation',
        'memory_footprint': '~8GB for inference',
        'monte_carlo_throughput': '1000+ game simulations/hour',
        'vegas_performance_comparison': '8.5% better RMSE'
    },
    
    'model_capabilities': {
        'real_time_prediction': 'Play-by-play during live games',
        'game_simulation': 'Complete game Monte Carlo',
        'betting_insights': 'Live stat tracking vs projections', 
        'scenario_analysis': 'What-if game situations',
        'team_performance_modeling': 'Season-long projections'
    }
}
```

---

## 🚀 Implementation Example

### Complete Training Pipeline

```python
import tensorflow as tf
from typing import Dict, Any
import numpy as np

class CFBHierarchicalTrainer:
    """
    Complete CFB hierarchical model trainer with 6-phase strategy
    """
    
    def __init__(self, 
                 config: Dict[str, Any],
                 data_pipeline,
                 model_architecture):
        
        self.config = config
        self.data_pipeline = data_pipeline 
        self.model = model_architecture
        self.current_phase = 1
        self.session_hours = 0
        
        # Initialize optimizers for each phase
        self.optimizers = self._create_phase_optimizers()
        self.loss_functions = self._create_loss_functions()
        self.metrics = self._create_metrics()
        
    def _create_phase_optimizers(self) -> Dict[int, tf.keras.optimizers.Optimizer]:
        """Create optimizers for each training phase"""
        optimizers = {}
        
        for phase in range(1, 7):
            lr_config = self.config['LEARNING_RATE_CONFIG']['phase_specific_rates'][f'phase_{phase}']
            
            # Cosine decay schedule
            lr_schedule = tf.keras.optimizers.schedules.CosineDecay(
                initial_learning_rate=lr_config['peak_lr'],
                decay_steps=lr_config['decay_steps'],
                alpha=lr_config['final_lr'] / lr_config['peak_lr']
            )
            
            # Warmup wrapper
            if lr_config['warmup_steps'] > 0:
                lr_schedule = WarmupCosineDecay(
                    base_schedule=lr_schedule,
                    warmup_steps=lr_config['warmup_steps'],
                    warmup_lr=lr_config['initial_lr']
                )
            
            optimizers[phase] = tf.keras.optimizers.AdamW(
                learning_rate=lr_schedule,
                weight_decay=1e-5,
                clipnorm=1.0  # Gradient clipping
            )
            
        return optimizers
    
    def _create_loss_functions(self) -> Dict[str, tf.keras.losses.Loss]:
        """Create loss functions for all prediction tasks"""
        return {
            # Play level losses
            'play_type': tf.keras.losses.CategoricalCrossentropy(
                label_smoothing=0.1,
                name='play_type_loss'
            ),
            'success_flags': tf.keras.losses.BinaryCrossentropy(
                name='success_flags_loss'
            ),
            'yards_gained': tf.keras.losses.Huber(
                delta=1.5,
                name='yards_gained_loss'
            ),
            
            # Drive level losses  
            'drive_outcome': tf.keras.losses.CategoricalCrossentropy(
                label_smoothing=0.05,
                name='drive_outcome_loss'
            ),
            'drive_metrics': tf.keras.losses.MeanSquaredError(
                name='drive_metrics_loss'
            ),
            
            # Game level losses (CRITICAL)
            'home_points': tf.keras.losses.MeanSquaredError(name='home_points_loss'),
            'away_points': tf.keras.losses.MeanSquaredError(name='away_points_loss'),
            'point_margin': tf.keras.losses.MeanSquaredError(name='point_margin_loss'),
            'volume_stats': tf.keras.losses.MeanSquaredError(name='volume_stats_loss'),
            'efficiency_stats': tf.keras.losses.MeanSquaredError(name='efficiency_stats_loss'),
            'explosiveness_stats': tf.keras.losses.MeanSquaredError(name='explosiveness_stats_loss'),
            
            # State consistency loss
            'state_consistency': StateConsistencyLoss(name='state_consistency_loss')
        }
    
    def _create_metrics(self) -> Dict[str, tf.keras.metrics.Metric]:
        """Create metrics for monitoring"""
        return {
            'game_rmse_combined': CombinedGameRMSE(name='game_rmse_combined'),
            'vegas_performance_ratio': VegasPerformanceRatio(name='vegas_ratio'),
            'state_consistency_rate': StateConsistencyRate(name='state_consistency'),
            'monte_carlo_fidelity': MonteCarloFidelity(name='mc_fidelity')
        }
    
    def train_phase(self, phase_num: int, duration_hours: int = 8):
        """Train a single phase with proper checkpointing"""
        print(f"🚀 Starting Phase {phase_num}/{6}")
        
        # Get phase configuration
        phase_config = self.config['HIERARCHICAL_TRAINING_PHASES'][f'phase_{phase_num}']
        loss_weights = phase_config['loss_weights']
        
        # Set current optimizer
        current_optimizer = self.optimizers[phase_num]
        
        # Configure loss weights for this phase
        self._configure_phase_loss_weights(loss_weights)
        
        # Training loop
        start_time = time.time()
        step = 0
        
        while (time.time() - start_time) < (duration_hours * 3600):
            # Training step
            batch = next(self.data_pipeline.train_iterator)
            loss_dict = self._train_step(batch, current_optimizer, loss_weights)
            
            # Logging
            if step % 100 == 0:
                self._log_training_metrics(step, loss_dict, phase_num)
            
            # Validation every 2 hours
            if step % 7200 == 0:  # Approximate steps per 2 hours
                val_metrics = self._validate_model()
                self._log_validation_metrics(val_metrics, phase_num)
                
                # Early stopping check
                if self._check_early_stopping(val_metrics):
                    print(f"Early stopping triggered in Phase {phase_num}")
                    break
            
            # Auto-save every 30 minutes
            if step % 1800 == 0:
                self._save_checkpoint(phase_num, step, temporary=True)
            
            step += 1
        
        # Save phase completion checkpoint
        self._save_checkpoint(phase_num, step, temporary=False)
        print(f"✅ Phase {phase_num} completed")
    
    @tf.function
    def _train_step(self, batch, optimizer, loss_weights):
        """Single training step with hierarchical loss"""
        with tf.GradientTape() as tape:
            # Forward pass
            predictions = self.model(batch['inputs'], training=True)
            
            # Calculate losses
            losses = {}
            total_loss = 0
            
            # Play level losses
            play_loss = (
                self.loss_functions['play_type'](batch['play_type'], predictions['play_type']) * 0.3 +
                self.loss_functions['success_flags'](batch['success_flags'], predictions['success_flags']) * 0.4 +
                self.loss_functions['yards_gained'](batch['yards_gained'], predictions['yards_gained']) * 0.3
            )
            losses['play_level'] = play_loss
            total_loss += play_loss * loss_weights['play_level']
            
            # Drive level losses  
            drive_loss = (
                self.loss_functions['drive_outcome'](batch['drive_outcome'], predictions['drive_outcome']) * 0.6 +
                self.loss_functions['drive_metrics'](batch['drive_metrics'], predictions['drive_metrics']) * 0.4
            )
            losses['drive_level'] = drive_loss
            total_loss += drive_loss * loss_weights['drive_level']
            
            # Game level losses (CRITICAL FOR VEGAS PERFORMANCE)
            game_loss = (
                self.loss_functions['home_points'](batch['home_points'], predictions['home_points']) * 0.33 +
                self.loss_functions['away_points'](batch['away_points'], predictions['away_points']) * 0.33 +
                self.loss_functions['point_margin'](batch['point_margin'], predictions['point_margin']) * 0.34 +
                self.loss_functions['volume_stats'](batch['volume_stats'], predictions['volume_stats']) * 0.25 +
                self.loss_functions['efficiency_stats'](batch['efficiency_stats'], predictions['efficiency_stats']) * 0.25 +
                self.loss_functions['explosiveness_stats'](batch['explosiveness_stats'], predictions['explosiveness_stats']) * 0.25
            )
            losses['game_level'] = game_loss
            total_loss += game_loss * loss_weights['game_level']
            
            # State consistency loss (CRITICAL FOR MONTE CARLO)
            state_loss = self.loss_functions['state_consistency'](batch['state_transitions'], predictions)
            losses['state_consistency'] = state_loss
            total_loss += state_loss * loss_weights['state_consistency']
            
            losses['total'] = total_loss
        
        # Backward pass
        gradients = tape.gradient(total_loss, self.model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))
        
        return losses
    
    def train_complete_pipeline(self):
        """Run complete 6-phase training pipeline"""
        print("🎯 Starting Complete CFB Hierarchical Training Pipeline")
        print("📅 Total Duration: 48 hours across 4 sessions of 12 hours each")
        
        session_schedule = [
            (1, 2),  # Session 1: Phases 1-2 (0-12h)
            (3, 4),  # Session 2: Phases 3-4 (12-24h) 
            (5, 6),  # Session 3: Phases 5-6 (24-36h)
            (None, None)  # Session 4: Buffer/validation (36-48h)
        ]
        
        for session_idx, (phase_start, phase_end) in enumerate(session_schedule[:-1]):
            print(f"\n🏗️ SESSION {session_idx + 1} (Hours {session_idx * 12}-{(session_idx + 1) * 12})")
            
            for phase in range(phase_start, phase_end + 1):
                # Check for existing checkpoint
                if self._load_phase_checkpoint(phase):
                    print(f"📂 Resumed Phase {phase} from checkpoint")
                else:
                    print(f"🆕 Starting Phase {phase} from scratch")
                
                # Train phase
                self.train_phase(phase, duration_hours=8)
                
                # Validate phase completion
                val_metrics = self._validate_model()
                self._assess_phase_completion(phase, val_metrics)
                
                # Break for session if time limit reached
                if self.session_hours >= 12:
                    print(f"⏰ Session {session_idx + 1} time limit reached")
                    self.session_hours = 0
                    break
        
        # Final validation and model export
        print("\n🏁 Training Complete! Running Final Validation...")
        final_metrics = self._comprehensive_final_validation()
        self._export_production_model(final_metrics)
        
        return final_metrics

# Usage Example
def main():
    """Complete training pipeline execution"""
    
    # Initialize trainer
    trainer = CFBHierarchicalTrainer(
        config={
            'HIERARCHICAL_TRAINING_PHASES': HIERARCHICAL_TRAINING_PHASES,
            'LEARNING_RATE_CONFIG': LEARNING_RATE_CONFIG,
            'LOSS_FUNCTION_CONFIG': LOSS_FUNCTION_CONFIG,
            'TPU_OPTIMIZATION_CONFIG': TPU_OPTIMIZATION_CONFIG,
            'VALIDATION_CONFIG': VALIDATION_CONFIG
        },
        data_pipeline=cfb_data_pipeline,  # From previous documents
        model_architecture=cfb_hierarchical_model
    )
    
    # Execute complete training
    final_metrics = trainer.train_complete_pipeline()
    
    # Results summary
    print(f"\n🎯 FINAL RESULTS:")
    print(f"📊 Game RMSE: {final_metrics['game_rmse_combined']:.2f}")
    print(f"🎰 Vegas Performance: {final_metrics['vegas_performance_ratio']:.1%} better")
    print(f"🔄 State Consistency: {final_metrics['state_consistency_rate']:.1%}")
    print(f"🎮 Monte Carlo Ready: {final_metrics['monte_carlo_fidelity']:.1%}")

if __name__ == "__main__":
    main()
```

---

## 🎯 Next Steps Integration

This hyperparameter configuration provides:

✅ **6-Phase Hierarchical Training**: Joint learning with curriculum progression  
✅ **12-Hour Session Compatibility**: 8-hour phases with seamless checkpointing  
✅ **Sub-Vegas Performance**: <13 RMSE target vs Vegas 14+ RMSE  
✅ **Monte Carlo Optimization**: State consistency prioritized for simulation fidelity  
✅ **TPU v2-8 Efficiency**: 1536 batch size for optimal memory/compute balance  
✅ **Equal Game Stats Weighting**: Home/away scores and margin weighted equally  
✅ **Adaptive Learning Rates**: Phase-specific 2e-4 base with cosine decay  
✅ **Comprehensive Monitoring**: Real-time performance tracking and alert systems  

**Ready for integration with:**
- Document #6: Data joining logic across the 4 parquet tables
- Document #7: Vegas data integrations  
- Production deployment pipeline
- Monte Carlo simulation system

The hyperparameter configuration transforms the architectural foundations from Documents #1-4 into a production-ready training system optimized for beating Vegas accuracy while maintaining simulation fidelity for Monte Carlo applications.