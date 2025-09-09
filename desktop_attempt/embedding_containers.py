# embedding_container.py
"""
Embedding Container System for JAX/Flax CFB Model
Optimized for TPU v2-8 with functional transformations
"""

import jax
import jax.numpy as jnp
import flax.linen as nn
from typing import Dict, List, Tuple, Optional, Any, NamedTuple
from dataclasses import dataclass
import numpy as np
import os
import sys

# Handle both Colab and local environments
if 'google.colab' in sys.modules:
    BASE_PATH = "/content/drive/MyDrive/cfb_model/"
else:
    BASE_PATH = os.path.expanduser("~/cfb_model/")

@dataclass
class EmbeddingConfig:
    """Configuration for embedding containers"""
    offense_output_dim: int = 128
    defense_output_dim: int = 128
    game_state_output_dim: int = 64
    play_context_output_dim: int = 64
    interaction_output_dim: int = 128
    dropout_rate: float = 0.1
    use_layer_norm: bool = True
    
    # Vocabulary sizes from data analysis
    vocab_sizes: Dict[str, int] = None
    
    def __post_init__(self):
        if self.vocab_sizes is None:
            self.vocab_sizes = {
                'offense_conference': 37,
                'coach_offense': 277,
                'defense_conference': 37,
                'coach_defense': 277,
                'venue_id': 226,
                'down': 5,
                'period': 5
            }

class OffenseEmbeddingContainer(nn.Module):
    """Offense team embedding container"""
    
    config: EmbeddingConfig
    training: bool = True
    
    @nn.compact
    def __call__(self, inputs: Dict[str, jnp.ndarray]) -> jnp.ndarray:
        """
        Process offense features into embeddings
        
        Args:
            inputs: Dictionary with 'categorical' and 'numerical' keys
            
        Returns:
            Embedded features of shape [batch, seq_len, output_dim]
        """
        
        # Extract inputs
        categorical = inputs['categorical']  # [batch, seq, 4]
        numerical = inputs['numerical']      # [batch, seq, 43]
        
        batch_size, seq_len = categorical.shape[:2]
        
        # Categorical embeddings
        conference_embed = nn.Embed(
            num_embeddings=self.config.vocab_sizes['offense_conference'],
            features=12,
            name='offense_conference_embed'
        )(categorical[..., 0])
        
        coach_embed = nn.Embed(
            num_embeddings=self.config.vocab_sizes['coach_offense'],
            features=24,
            name='coach_offense_embed'
        )(categorical[..., 1])
        
        home_away_embed = nn.Embed(
            num_embeddings=2,
            features=4,
            name='home_away_embed'
        )(categorical[..., 2])
        
        new_coach_embed = nn.Embed(
            num_embeddings=2,
            features=4,
            name='new_coach_embed'
        )(categorical[..., 3])
        
        # Combine categorical embeddings
        categorical_combined = jnp.concatenate([
            conference_embed, coach_embed, home_away_embed, new_coach_embed
        ], axis=-1)  # [batch, seq, 44]
        
        # Process numerical features with layer norm
        if self.config.use_layer_norm:
            numerical = nn.LayerNorm()(numerical)
        
        # Ensure numerical has correct shape
        batch_size, seq_len = categorical_combined.shape[:2]
        if numerical.ndim == 2:
            numerical = numerical.reshape(batch_size, seq_len, -1)
        
        # Combine all features
        combined = jnp.concatenate([categorical_combined, numerical], axis=-1)
        
        # Reshape for dense layers
        combined_flat = combined.reshape(-1, combined.shape[-1])
        
        # Projection layers
        x = nn.Dense(256, name='projection_1')(combined_flat)
        x = nn.relu(x)
        x = nn.Dropout(self.config.dropout_rate, deterministic=not self.training)(x)
        
        x = nn.Dense(self.config.offense_output_dim, name='projection_2')(x)
        
        # Final layer norm
        if self.config.use_layer_norm:
            x = nn.LayerNorm()(x)
        
        # Reshape back to sequence
        output = x.reshape(batch_size, seq_len, self.config.offense_output_dim)
        
        return output

class DefenseEmbeddingContainer(nn.Module):
    """Defense team embedding container"""
    
    config: EmbeddingConfig
    training: bool = True
    
    @nn.compact
    def __call__(self, inputs: Dict[str, jnp.ndarray]) -> jnp.ndarray:
        """Process defense features into embeddings"""
        
        categorical = inputs['categorical']  # [batch, seq, 3]
        numerical = inputs['numerical']      # [batch, seq, 40]
        
        batch_size, seq_len = categorical.shape[:2]
        
        # Categorical embeddings
        conference_embed = nn.Embed(
            num_embeddings=self.config.vocab_sizes['defense_conference'],
            features=12,
            name='defense_conference_embed'
        )(categorical[..., 0])
        
        coach_embed = nn.Embed(
            num_embeddings=self.config.vocab_sizes['coach_defense'],
            features=24,
            name='coach_defense_embed'
        )(categorical[..., 1])
        
        new_coach_embed = nn.Embed(
            num_embeddings=2,
            features=4,
            name='defense_new_coach_embed'
        )(categorical[..., 2])
        
        # Combine categorical
        categorical_combined = jnp.concatenate([
            conference_embed, coach_embed, new_coach_embed
        ], axis=-1)  # [batch, seq, 40]
        
        # Process numerical
        if self.config.use_layer_norm:
            numerical = nn.LayerNorm()(numerical)
        
        # Combine
        combined = jnp.concatenate([categorical_combined, numerical], axis=-1)
        combined_flat = combined.reshape(-1, combined.shape[-1])
        
        # Projection
        x = nn.Dense(256, name='projection_1')(combined_flat)
        x = nn.relu(x)
        x = nn.Dropout(self.config.dropout_rate, deterministic=not self.training)(x)
        
        x = nn.Dense(self.config.defense_output_dim, name='projection_2')(x)
        
        if self.config.use_layer_norm:
            x = nn.LayerNorm()(x)
        
        output = x.reshape(batch_size, seq_len, self.config.defense_output_dim)
        
        return output

class GameStateEmbeddingContainer(nn.Module):
    """Game state embedding container"""
    
    config: EmbeddingConfig
    training: bool = True
    
    @nn.compact
    def __call__(self, inputs: Dict[str, jnp.ndarray]) -> jnp.ndarray:
        """Process game state features into embeddings"""
        
        categorical = inputs['categorical']  # [batch, seq, 7]
        numerical = inputs['numerical']      # [batch, seq, 26]
        
        batch_size, seq_len = categorical.shape[:2]
        
        # Venue embedding
        venue_embed = nn.Embed(
            num_embeddings=self.config.vocab_sizes['venue_id'],
            features=16,
            name='venue_embed'
        )(categorical[..., 0])
        
        # Binary embeddings
        binary_embeds = []
        binary_names = ['game_indoors', 'is_field_turf', 'is_offense_home', 
                       'conference_game', 'bowl_game']
        
        for i, name in enumerate(binary_names, start=1):
            embed = nn.Embed(
                num_embeddings=2,
                features=2,
                name=f'{name}_embed'
            )(categorical[..., i])
            binary_embeds.append(embed)
        
        # Wind direction (already binned to 16 directions)
        wind_embed = nn.Embed(
            num_embeddings=16,
            features=8,
            name='wind_direction_embed'
        )(categorical[..., 6])
        
        # Combine categorical
        categorical_combined = jnp.concatenate(
            [venue_embed] + binary_embeds + [wind_embed], axis=-1
        )  # [batch, seq, 34]
        
        # Process numerical
        if self.config.use_layer_norm:
            numerical = nn.LayerNorm()(numerical)
        
        # Combine
        combined = jnp.concatenate([categorical_combined, numerical], axis=-1)
        combined_flat = combined.reshape(-1, combined.shape[-1])
        
        # Projection
        x = nn.Dense(128, name='projection_1')(combined_flat)
        x = nn.relu(x)
        x = nn.Dropout(self.config.dropout_rate, deterministic=not self.training)(x)
        
        x = nn.Dense(self.config.game_state_output_dim, name='projection_2')(x)
        
        if self.config.use_layer_norm:
            x = nn.LayerNorm()(x)
        
        output = x.reshape(batch_size, seq_len, self.config.game_state_output_dim)
        
        return output

class PlayContextEmbeddingContainer(nn.Module):
    """Play context embedding container"""
    
    config: EmbeddingConfig
    training: bool = True
    
    @nn.compact
    def __call__(self, inputs: Dict[str, jnp.ndarray]) -> jnp.ndarray:
        """Process play context features into embeddings"""
        
        categorical = inputs['categorical']  # [batch, seq, 2]
        binary = inputs['binary']           # [batch, seq, 20]
        numerical = inputs['numerical']      # [batch, seq, 8]
        
        batch_size, seq_len = categorical.shape[:2]
        
        # Down embedding
        down_embed = nn.Embed(
            num_embeddings=self.config.vocab_sizes['down'],
            features=8,
            name='down_embed'
        )(categorical[..., 0])
        
        # Period embedding
        period_embed = nn.Embed(
            num_embeddings=self.config.vocab_sizes['period'],
            features=8,
            name='period_embed'
        )(categorical[..., 1])
        
        # Combine categorical
        categorical_combined = jnp.concatenate([down_embed, period_embed], axis=-1)
        
        # Process binary and numerical
        if self.config.use_layer_norm:
            binary = nn.LayerNorm()(binary)
            numerical = nn.LayerNorm()(numerical)
        
        # Combine all
        combined = jnp.concatenate([categorical_combined, binary, numerical], axis=-1)
        combined_flat = combined.reshape(-1, combined.shape[-1])
        
        # Projection
        x = nn.Dense(128, name='projection_1')(combined_flat)
        x = nn.relu(x)
        x = nn.Dropout(self.config.dropout_rate, deterministic=not self.training)(x)
        
        x = nn.Dense(self.config.play_context_output_dim, name='projection_2')(x)
        
        if self.config.use_layer_norm:
            x = nn.LayerNorm()(x)
        
        output = x.reshape(batch_size, seq_len, self.config.play_context_output_dim)
        
        return output

class InteractionFeaturesContainer(nn.Module):
    """Cross-embedding interaction features"""
    
    config: EmbeddingConfig
    training: bool = True
    
    @nn.compact
    def __call__(self, offense_features: jnp.ndarray, 
                defense_features: jnp.ndarray) -> jnp.ndarray:
        """
        Compute interaction features between offense and defense
        
        Args:
            offense_features: [batch, seq, 128]
            defense_features: [batch, seq, 128]
            
        Returns:
            Interaction features [batch, seq, 128]
        """
        
        batch_size, seq_len = offense_features.shape[:2]
        
        # Flatten for processing
        offense_flat = offense_features.reshape(-1, offense_features.shape[-1])
        defense_flat = defense_features.reshape(-1, defense_features.shape[-1])
        
        # Compute interactions
        # 1. Element-wise differences
        diff_features = offense_flat - defense_flat
        
        # 2. Element-wise products
        product_features = offense_flat * defense_flat
        
        # 3. Concatenate and project
        combined = jnp.concatenate([
            diff_features, product_features, offense_flat, defense_flat
        ], axis=-1)
        
        # Interaction layers
        x = nn.Dense(256, name='interaction_1')(combined)
        x = nn.tanh(x)
        x = nn.Dropout(self.config.dropout_rate, deterministic=not self.training)(x)
        
        x = nn.Dense(self.config.interaction_output_dim, name='interaction_2')(x)
        
        if self.config.use_layer_norm:
            x = nn.LayerNorm()(x)
        
        # Reshape back
        output = x.reshape(batch_size, seq_len, self.config.interaction_output_dim)
        
        return output

class CFBEmbeddingModel(nn.Module):
    """Complete embedding model combining all containers"""
    
    config: EmbeddingConfig
    training: bool = True
    
    def setup(self):
        # Initialize all containers
        self.offense_container = OffenseEmbeddingContainer(self.config, self.training)
        self.defense_container = DefenseEmbeddingContainer(self.config, self.training)
        self.game_state_container = GameStateEmbeddingContainer(self.config, self.training)
        self.play_context_container = PlayContextEmbeddingContainer(self.config, self.training)
        self.interaction_container = InteractionFeaturesContainer(self.config, self.training)
    
    def __call__(self, inputs: Dict[str, Dict[str, jnp.ndarray]]) -> jnp.ndarray:
        """
        Process all inputs through embedding containers
        
        Args:
            inputs: Dictionary with keys 'offense', 'defense', 'game_state', 'play_context'
            
        Returns:
            Combined embeddings [batch, seq, 512]
        """
        
        # Process each embedding type
        offense_emb = self.offense_container(inputs['offense'])
        defense_emb = self.defense_container(inputs['defense'])
        game_state_emb = self.game_state_container(inputs['game_state'])
        play_context_emb = self.play_context_container(inputs['play_context'])
        
        # Compute interaction features
        interaction_emb = self.interaction_container(offense_emb, defense_emb)
        
        # Concatenate all embeddings
        combined = jnp.concatenate([
            offense_emb,      # 128
            defense_emb,      # 128
            game_state_emb,   # 64
            play_context_emb, # 64
            interaction_emb   # 128
        ], axis=-1)  # Total: 512 dimensions
        
        return combined

# Utility functions for model initialization
def create_embedding_model(config: EmbeddingConfig = None, training: bool = True):
    """Create and initialize embedding model"""
    
    if config is None:
        config = EmbeddingConfig()
    
    return CFBEmbeddingModel(config=config, training=training)

def init_model_params(model: CFBEmbeddingModel, rng_key: jax.random.PRNGKey,
                     batch_size: int = 32, seq_len: int = 576):
    """Initialize model parameters"""
    
    # Create dummy inputs
    dummy_inputs = {
        'offense': {
            'categorical': jnp.zeros((batch_size, seq_len, 4), dtype=jnp.int32),
            'numerical': jnp.zeros((batch_size, seq_len, 43), dtype=jnp.float32)
        },
        'defense': {
            'categorical': jnp.zeros((batch_size, seq_len, 3), dtype=jnp.int32),
            'numerical': jnp.zeros((batch_size, seq_len, 40), dtype=jnp.float32)
        },
        'game_state': {
            'categorical': jnp.zeros((batch_size, seq_len, 7), dtype=jnp.int32),
            'numerical': jnp.zeros((batch_size, seq_len, 26), dtype=jnp.float32)
        },
        'play_context': {
            'categorical': jnp.zeros((batch_size, seq_len, 2), dtype=jnp.int32),
            'binary': jnp.zeros((batch_size, seq_len, 20), dtype=jnp.float32),
            'numerical': jnp.zeros((batch_size, seq_len, 8), dtype=jnp.float32)
        }
    }
    
    # Initialize parameters
    params = model.init(rng_key, dummy_inputs)
    
    return params