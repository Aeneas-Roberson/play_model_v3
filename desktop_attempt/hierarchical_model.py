"""
Hierarchical CFB Model with LSTM and Prediction Heads
Complete implementation for play → drive → game predictions
"""

import jax
import jax.numpy as jnp
import flax.linen as nn
from typing import Dict, Tuple, Optional, Any
import numpy as np


class PlayLSTM(nn.Module):
    """LSTM for processing play sequences within drives"""
    hidden_size: int = 256
    dropout_rate: float = 0.1
    
    @nn.compact
    def __call__(self, x, training: bool = False):
        # x shape: (batch, seq_len, features)
        lstm_cell = nn.LSTMCell(features=self.hidden_size)
        
        # Initialize carry (hidden state, cell state)
        batch_size = x.shape[0]
        carry = lstm_cell.initialize_carry(
            jax.random.PRNGKey(0), (batch_size,)
        )
        
        # Process sequence
        outputs = []
        for t in range(x.shape[1]):
            carry, y = lstm_cell(carry, x[:, t, :])
            y = nn.Dropout(rate=self.dropout_rate)(y, deterministic=not training)
            outputs.append(y)
        
        # Stack outputs: (batch, seq_len, hidden_size)
        return jnp.stack(outputs, axis=1), carry


class DriveLSTM(nn.Module):
    """LSTM for processing drive sequences within games"""
    hidden_size: int = 384
    dropout_rate: float = 0.1
    
    @nn.compact
    def __call__(self, x, training: bool = False):
        # x shape: (batch, num_drives, drive_features)
        lstm_cell = nn.LSTMCell(features=self.hidden_size)
        
        batch_size = x.shape[0]
        carry = lstm_cell.initialize_carry(
            jax.random.PRNGKey(0), (batch_size,)
        )
        
        outputs = []
        for t in range(x.shape[1]):
            carry, y = lstm_cell(carry, x[:, t, :])
            y = nn.Dropout(rate=self.dropout_rate)(y, deterministic=not training)
            outputs.append(y)
        
        return jnp.stack(outputs, axis=1), carry


class GameLSTM(nn.Module):
    """LSTM for game-level predictions"""
    hidden_size: int = 512
    dropout_rate: float = 0.1
    
    @nn.compact
    def __call__(self, x, training: bool = False):
        # x shape: (batch, game_features)
        x = nn.Dense(features=self.hidden_size)(x)
        x = nn.relu(x)
        x = nn.Dropout(rate=self.dropout_rate)(x, deterministic=not training)
        x = nn.Dense(features=self.hidden_size // 2)(x)
        x = nn.relu(x)
        x = nn.Dropout(rate=self.dropout_rate)(x, deterministic=not training)
        return x


class PredictionHeads(nn.Module):
    """Multi-level prediction heads for hierarchical outputs"""
    
    play_output_dim: int = 7  # Play type prediction
    drive_output_dim: int = 3  # Drive outcome (TD, FG, Punt, etc.)
    game_output_dim: int = 2   # Final score predictions
    dropout_rate: float = 0.1
    
    @nn.compact
    def __call__(self, play_features, drive_features, game_features, training: bool = False):
        predictions = {}
        
        # Play-level predictions
        play_head = nn.Dense(features=128)(play_features)
        play_head = nn.relu(play_head)
        play_head = nn.Dropout(rate=self.dropout_rate)(play_head, deterministic=not training)
        predictions['play_outcomes'] = nn.Dense(features=self.play_output_dim)(play_head)
        predictions['play_yards'] = nn.Dense(features=1)(play_head)  # Yards gained
        
        # Drive-level predictions
        drive_head = nn.Dense(features=128)(drive_features)
        drive_head = nn.relu(drive_head)
        drive_head = nn.Dropout(rate=self.dropout_rate)(drive_head, deterministic=not training)
        predictions['drive_outcomes'] = nn.Dense(features=self.drive_output_dim)(drive_head)
        predictions['drive_points'] = nn.Dense(features=1)(drive_head)  # Points scored
        
        # Game-level predictions
        game_head = nn.Dense(features=256)(game_features)
        game_head = nn.relu(game_head)
        game_head = nn.Dropout(rate=self.dropout_rate)(game_head, deterministic=not training)
        predictions['home_score'] = nn.Dense(features=1)(game_head)
        predictions['away_score'] = nn.Dense(features=1)(game_head)
        predictions['total_points'] = nn.Dense(features=1)(game_head)
        predictions['spread'] = nn.Dense(features=1)(game_head)
        
        return predictions


class HierarchicalCFBModel(nn.Module):
    """Complete hierarchical model: embeddings → LSTM → predictions"""
    
    embedding_model: nn.Module
    max_plays_per_drive: int = 18
    max_drives_per_game: int = 32
    play_lstm_hidden: int = 256
    drive_lstm_hidden: int = 384
    game_lstm_hidden: int = 512
    dropout_rate: float = 0.1
    
    def setup(self):
        self.play_lstm = PlayLSTM(
            hidden_size=self.play_lstm_hidden,
            dropout_rate=self.dropout_rate
        )
        self.drive_lstm = DriveLSTM(
            hidden_size=self.drive_lstm_hidden,
            dropout_rate=self.dropout_rate
        )
        self.game_lstm = GameLSTM(
            hidden_size=self.game_lstm_hidden,
            dropout_rate=self.dropout_rate
        )
        self.prediction_heads = PredictionHeads(
            dropout_rate=self.dropout_rate
        )
    
    @nn.compact
    def __call__(self, batch: Dict[str, jnp.ndarray], training: bool = False) -> Dict[str, jnp.ndarray]:
        # Get embeddings from embedding model
        embeddings = self.embedding_model(batch['features'], training=training)
        
        # Reshape for hierarchical processing
        batch_size = embeddings.shape[0]
        total_seq_len = embeddings.shape[1]
        embed_dim = embeddings.shape[2]
        
        # Reshape to (batch, drives, plays, features)
        hierarchical_shape = (
            batch_size,
            self.max_drives_per_game,
            self.max_plays_per_drive,
            embed_dim
        )
        
        # Handle variable length sequences
        if total_seq_len != self.max_drives_per_game * self.max_plays_per_drive:
            # Pad or truncate as needed
            target_len = self.max_drives_per_game * self.max_plays_per_drive
            if total_seq_len < target_len:
                # Pad
                pad_width = ((0, 0), (0, target_len - total_seq_len), (0, 0))
                embeddings = jnp.pad(embeddings, pad_width)
            else:
                # Truncate
                embeddings = embeddings[:, :target_len, :]
        
        embeddings_reshaped = embeddings.reshape(hierarchical_shape)
        
        # Process plays within each drive
        play_outputs_list = []
        play_final_states = []
        
        for drive_idx in range(self.max_drives_per_game):
            drive_plays = embeddings_reshaped[:, drive_idx, :, :]
            play_outputs, play_final = self.play_lstm(drive_plays, training=training)
            play_outputs_list.append(play_outputs)
            play_final_states.append(play_final[0])  # Hidden state only
        
        # Stack play outputs: (batch, drives, plays, hidden)
        play_outputs_stacked = jnp.stack(play_outputs_list, axis=1)
        
        # Aggregate play features to drive level
        # Use mean pooling over plays for drive representation
        drive_features = jnp.mean(play_outputs_stacked, axis=2)
        
        # Add play final states as additional drive features
        play_final_stacked = jnp.stack(play_final_states, axis=1)
        drive_features = jnp.concatenate([drive_features, play_final_stacked], axis=-1)
        
        # Process drives within game
        drive_outputs, drive_final = self.drive_lstm(drive_features, training=training)
        
        # Aggregate drive features to game level
        game_features = jnp.mean(drive_outputs, axis=1)
        game_features = jnp.concatenate([game_features, drive_final[0]], axis=-1)
        
        # Process game-level features
        game_outputs = self.game_lstm(game_features, training=training)
        
        # Get predictions at all levels
        predictions = self.prediction_heads(
            play_outputs_stacked.reshape(-1, play_outputs_stacked.shape[-1]),
            drive_outputs.reshape(-1, drive_outputs.shape[-1]),
            game_outputs,
            training=training
        )
        
        return predictions


class HierarchicalLoss(nn.Module):
    """Multi-level loss function for hierarchical model"""
    
    play_weight: float = 0.2
    drive_weight: float = 0.3
    game_weight: float = 0.5
    consistency_weight: float = 0.1
    
    def __call__(self, predictions: Dict, targets: Dict) -> Tuple[jnp.ndarray, Dict]:
        losses = {}
        
        # Play-level losses
        if 'play_outcomes' in predictions and 'play_labels' in targets:
            play_ce = optax.softmax_cross_entropy_with_integer_labels(
                predictions['play_outcomes'],
                targets['play_labels']
            )
            losses['play_classification'] = jnp.mean(play_ce)
        
        if 'play_yards' in predictions and 'play_yards_actual' in targets:
            losses['play_regression'] = jnp.mean(
                (predictions['play_yards'] - targets['play_yards_actual']) ** 2
            )
        
        # Drive-level losses
        if 'drive_outcomes' in predictions and 'drive_labels' in targets:
            drive_ce = optax.softmax_cross_entropy_with_integer_labels(
                predictions['drive_outcomes'],
                targets['drive_labels']
            )
            losses['drive_classification'] = jnp.mean(drive_ce)
        
        if 'drive_points' in predictions and 'drive_points_actual' in targets:
            losses['drive_regression'] = jnp.mean(
                (predictions['drive_points'] - targets['drive_points_actual']) ** 2
            )
        
        # Game-level losses
        if 'home_score' in predictions and 'home_score_actual' in targets:
            losses['home_score_mse'] = jnp.mean(
                (predictions['home_score'] - targets['home_score_actual']) ** 2
            )
        
        if 'away_score' in predictions and 'away_score_actual' in targets:
            losses['away_score_mse'] = jnp.mean(
                (predictions['away_score'] - targets['away_score_actual']) ** 2
            )
        
        if 'spread' in predictions and 'spread_actual' in targets:
            losses['spread_mse'] = jnp.mean(
                (predictions['spread'] - targets['spread_actual']) ** 2
            )
        
        # Hierarchical consistency loss
        # Ensure drive points sum approximately to game scores
        if 'drive_points' in predictions and 'home_score' in predictions:
            # This is simplified - in practice would need proper aggregation
            consistency_loss = jnp.abs(
                jnp.sum(predictions['drive_points']) - 
                (predictions['home_score'] + predictions['away_score'])
            )
            losses['consistency'] = consistency_loss * self.consistency_weight
        
        # Weighted total loss
        total_loss = 0.0
        
        # Play level
        if 'play_classification' in losses:
            total_loss += losses['play_classification'] * self.play_weight * 0.5
        if 'play_regression' in losses:
            total_loss += losses['play_regression'] * self.play_weight * 0.5
        
        # Drive level
        if 'drive_classification' in losses:
            total_loss += losses['drive_classification'] * self.drive_weight * 0.5
        if 'drive_regression' in losses:
            total_loss += losses['drive_regression'] * self.drive_weight * 0.5
        
        # Game level
        if 'home_score_mse' in losses:
            total_loss += losses['home_score_mse'] * self.game_weight * 0.3
        if 'away_score_mse' in losses:
            total_loss += losses['away_score_mse'] * self.game_weight * 0.3
        if 'spread_mse' in losses:
            total_loss += losses['spread_mse'] * self.game_weight * 0.4
        
        # Add consistency
        if 'consistency' in losses:
            total_loss += losses['consistency']
        
        return total_loss, losses


def create_hierarchical_model(embedding_model, config):
    """Factory function to create hierarchical model"""
    return HierarchicalCFBModel(
        embedding_model=embedding_model,
        max_plays_per_drive=config.get('max_plays_per_drive', 18),
        max_drives_per_game=config.get('max_drives_per_game', 32),
        play_lstm_hidden=config.get('play_lstm_hidden', 256),
        drive_lstm_hidden=config.get('drive_lstm_hidden', 384),
        game_lstm_hidden=config.get('game_lstm_hidden', 512),
        dropout_rate=config.get('dropout_rate', 0.1)
    )