# sequential_batching.py
"""
Sequential Batching System for CFB Hierarchical Model
Optimized for JAX/Flax TPU training with game-centric organization
"""

import jax
import jax.numpy as jnp
# Enable 64-bit precision for large integers
jax.config.update("jax_enable_x64", True)
import numpy as np
from typing import Dict, List, Tuple, Optional, Any, Iterator
from dataclasses import dataclass
import logging
from collections import defaultdict
import pickle
from pathlib import Path
import os
import sys
import gc

# Handle both Colab and local environments
if 'google.colab' in sys.modules:
    BASE_PATH = "/content/drive/MyDrive/cfb_model/"
else:
    BASE_PATH = os.path.expanduser("~/cfb_model/")

@dataclass
class SequentialBatchConfig:
    """Configuration for sequential batching"""
    max_plays_per_drive: int = 18  # Restored to original dimensions
    max_drives_per_game: int = 32  # Restored to original dimensions
    batch_size: int = 256  # Reduced for memory constraints
    prefetch_size: int = 4
    cache_sequences: bool = True
    drive_padding_strategy: str = "adaptive"  # "adaptive" or "fixed"
    sequence_dropout: float = 0.0  # Random sequence dropout for regularization
    
class CFBSequentialBatcher:
    """
    Game-centric sequential batching system for JAX/Flax
    """
    
    def __init__(self, config: SequentialBatchConfig = None):
        self.config = config or SequentialBatchConfig()
        
        # Sequence cache for pre-computed sequences
        self.sequence_cache = {}
        self.cache_stats = {'hits': 0, 'misses': 0}
        
        # Drive boundary markers
        self.DRIVE_START_TOKEN = 999999
        self.DRIVE_END_TOKEN = -999999
        
        # Setup logging
        self.logger = logging.getLogger('CFBSequentialBatcher')
        
        # Configure logging for Colab
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.StreamHandler(),
                logging.FileHandler(os.path.join(BASE_PATH, 'logs/model.log'))
            ] if os.path.exists(os.path.join(BASE_PATH, 'logs')) else [logging.StreamHandler()]
        )
        
    def create_game_sequences(self, 
                            preprocessed_data: Dict[str, jnp.ndarray],
                            game_metadata: Dict[str, np.ndarray], 
                            chunk_size: int = 64) -> Dict[str, jnp.ndarray]:
        """
        Create game-centric sequences with STREAMING processing for memory efficiency
        
        Args:
            preprocessed_data: Output from CFBDataPreprocessor with JAX arrays
            game_metadata: Game, drive, and play IDs
            chunk_size: Number of games to process at once (default: 64)
            
        Returns:
            Dictionary with sequences, masks, and metadata
        """
        self.logger.info("🔄 Creating game-centric sequences with streaming...")
        
        # Group plays by games
        game_groups = self._group_plays_by_game(preprocessed_data, game_metadata)
        
        # Process games in chunks to avoid memory explosion
        game_ids = list(game_groups.keys())
        total_games = len(game_ids)
        
        all_sequences = defaultdict(list)
        all_masks = []
        all_game_info = []
        
        self.logger.info(f"📦 Processing {total_games} games in chunks of {chunk_size}")
        
        for chunk_start in range(0, total_games, chunk_size):
            chunk_end = min(chunk_start + chunk_size, total_games)
            chunk_game_ids = game_ids[chunk_start:chunk_end]
            
            # Process this chunk
            chunk_sequences = defaultdict(list)
            chunk_masks = []
            chunk_game_info = []
            
            for game_id in chunk_game_ids:
                game_data = game_groups[game_id]
                
                # Check cache first
                cache_key = f"game_{game_id}"
                if self.config.cache_sequences and cache_key in self.sequence_cache:
                    cached_seq, cached_mask, cached_info = self.sequence_cache[cache_key]
                    self.cache_stats['hits'] += 1
                else:
                    # Create sequence for this game
                    cached_seq, cached_mask, cached_info = self._process_single_game(
                        game_id, game_data
                    )
                    
                    if self.config.cache_sequences:
                        self.sequence_cache[cache_key] = (cached_seq, cached_mask, cached_info)
                    self.cache_stats['misses'] += 1
                
                # Accumulate in chunk
                for key, value in cached_seq.items():
                    chunk_sequences[key].append(value)
                chunk_masks.append(cached_mask)
                chunk_game_info.append(cached_info)
            
            # Stack this chunk efficiently
            chunk_batched = self._create_batched_tensors(
                chunk_sequences, chunk_masks, chunk_game_info
            )
            
            # Accumulate chunks (this is memory-efficient since chunks are small)
            for key, value in chunk_batched['sequences'].items():
                all_sequences[key].append(value)
            all_masks.append(chunk_batched['masks'])
            all_game_info.extend(chunk_batched['metadata'])
            
            self.logger.info(f"  ✅ Processed chunk {chunk_start//chunk_size + 1}/{(total_games-1)//chunk_size + 1}")
        
        # Final concatenation of chunks
        final_sequences = {}
        for key, chunk_list in all_sequences.items():
            final_sequences[key] = jnp.concatenate(chunk_list, axis=0)
        
        final_masks = jnp.concatenate(all_masks, axis=0)
        
        final_result = {
            'sequences': final_sequences,
            'masks': final_masks,
            'metadata': all_game_info,
            'shape_info': {
                'batch_size': len(all_game_info),
                'max_sequence_length': self.config.max_plays_per_drive * self.config.max_drives_per_game,
                'max_drives': self.config.max_drives_per_game,
                'max_plays_per_drive': self.config.max_plays_per_drive
            }
        }
        
        self.logger.info(f"✅ Created {len(all_game_info)} game sequences via streaming")
        self.logger.info(f"📊 Cache stats - Hits: {self.cache_stats['hits']}, "
                        f"Misses: {self.cache_stats['misses']}")
        
        return final_result
    
    def _group_plays_by_game(self, 
                           preprocessed_data: Dict[str, jnp.ndarray],
                           game_metadata: Dict[str, np.ndarray]) -> Dict[int, Dict]:
        """Group all preprocessed plays by game_id"""
        
        game_groups = defaultdict(lambda: defaultdict(list))
        
        game_ids = game_metadata['game_ids']
        drive_ids = game_metadata['drive_ids']
        play_ids = game_metadata['play_ids']
        
        # Get unique games
        unique_games = np.unique(game_ids)
        
        for game_id in unique_games:
            # Get indices for this game
            game_mask = game_ids == game_id
            game_indices = np.where(game_mask)[0]
            
            # Get drives for this game
            game_drives = drive_ids[game_indices]
            unique_drives = np.unique(game_drives)
            
            # Organize by drives
            drives_data = []
            for drive_id in unique_drives:
                drive_mask = (game_ids == game_id) & (drive_ids == drive_id)
                drive_indices = np.where(drive_mask)[0]
                
                # Sort by play order (assuming play_ids are sequential)
                sorted_indices = drive_indices[np.argsort(play_ids[drive_indices])]
                
                drive_data = {
                    'drive_id': drive_id,
                    'indices': sorted_indices,
                    'play_count': len(sorted_indices)
                }
                drives_data.append(drive_data)
            
            game_groups[game_id] = {
                'game_id': game_id,
                'drives': drives_data,
                'total_plays': len(game_indices),
                'indices': game_indices
            }
        
        return game_groups
    
    def _process_single_game(self, 
                           game_id: int,
                           game_data: Dict) -> Tuple[Dict, jnp.ndarray, Dict]:
        """Process a single game into padded sequences with drive structure"""
        
        drives = game_data['drives']
        total_plays = game_data['total_plays']
        
        # Initialize sequence containers
        max_seq_length = self.config.max_plays_per_drive * self.config.max_drives_per_game
        
        sequences = {
            'plays': np.zeros(max_seq_length, dtype=np.int64),  # Play indices
            'drives': np.zeros(max_seq_length, dtype=np.int64),  # Drive indices
            'drive_boundaries': np.zeros(max_seq_length, dtype=np.int64),  # Boundary markers
            'temporal_positions': np.zeros(max_seq_length, dtype=np.float32)  # Temporal encoding
        }
        
        # Process each drive
        current_position = 0
        drive_start_positions = []
        drive_end_positions = []
        
        for drive_idx, drive in enumerate(drives[:self.config.max_drives_per_game]):
            if current_position >= max_seq_length:
                break
                
            drive_start_positions.append(current_position)
            
            # Add drive start marker only if there's space
            if current_position < max_seq_length:
                sequences['drive_boundaries'][current_position] = self.DRIVE_START_TOKEN
                current_position += 1
            
            # Process plays in drive
            play_indices = drive['indices']
            num_plays = min(len(play_indices), self.config.max_plays_per_drive)
            
            plays_added = 0
            for play_idx in range(num_plays):
                if current_position >= max_seq_length:
                    break
                    
                sequences['plays'][current_position] = int(play_indices[play_idx])
                sequences['drives'][current_position] = drive['drive_id']
                sequences['temporal_positions'][current_position] = current_position / max_seq_length
                current_position += 1
                plays_added += 1
            
            # Add drive end marker only if there's space and we added plays
            if current_position < max_seq_length and plays_added > 0:
                sequences['drive_boundaries'][current_position] = self.DRIVE_END_TOKEN
                drive_end_positions.append(current_position)
                current_position += 1
        
        # Create mask
        mask = np.zeros(max_seq_length, dtype=np.float32)
        mask[:current_position] = 1.0
        
        # Apply sequence dropout if configured (training only)
        if self.config.sequence_dropout > 0:
            dropout_mask = np.random.random(max_seq_length) > self.config.sequence_dropout
            mask = mask * dropout_mask.astype(np.float32)
        
        # Metadata
        metadata = {
            'game_id': game_id,
            'total_plays': total_plays,
            'sequence_length': current_position,
            'num_drives': len(drives),
            'drive_starts': drive_start_positions,
            'drive_ends': drive_end_positions,
            'padding_ratio': 1.0 - (current_position / max_seq_length)
        }
        
        # Convert to JAX arrays
        sequences_jax = {}
        for k, v in sequences.items():
            if k in ['plays', 'drives', 'drive_boundaries']:
                sequences_jax[k] = jnp.array(v, dtype=jnp.int64)
            else:
                sequences_jax[k] = jnp.array(v, dtype=jnp.float32)
        mask_jax = jnp.array(mask, dtype=jnp.float32)
        
        return sequences_jax, mask_jax, metadata
    
    def _create_batched_tensors(self, 
                              sequences: Dict[str, List],
                              masks: List[jnp.ndarray],
                              metadata: List[Dict]) -> Dict[str, Any]:
        """Create batched JAX tensors optimized for TPU training"""
        
        # Stack sequences
        batched_sequences = {}
        for key, seq_list in sequences.items():
            if seq_list:
                batched_sequences[key] = jnp.stack(seq_list, axis=0)
        
        # Stack masks
        batched_masks = jnp.stack(masks, axis=0) if masks else jnp.array([], dtype=jnp.float32)
        
        return {
            'sequences': batched_sequences,
            'masks': batched_masks,
            'metadata': metadata,
            'shape_info': {
                'batch_size': len(masks),
                'max_sequence_length': self.config.max_plays_per_drive * self.config.max_drives_per_game,
                'max_drives': self.config.max_drives_per_game,
                'max_plays_per_drive': self.config.max_plays_per_drive
            }
        }
    
    def create_training_iterator(self, 
                               data: Dict[str, jnp.ndarray],
                               game_sequences: Dict[str, Any],
                               shuffle: bool = True,
                               drop_remainder: bool = True) -> Iterator[Dict]:
        """
        Create iterator that yields batches ready for training
        
        Args:
            data: Preprocessed features from CFBDataPreprocessor
            game_sequences: Output from create_game_sequences
            shuffle: Whether to shuffle data
            drop_remainder: Drop incomplete batches for TPU
            
        Yields:
            Batches with features and sequences properly aligned
        """
        
        num_games = game_sequences['shape_info']['batch_size']
        batch_size = self.config.batch_size
        
        # Calculate number of batches
        if drop_remainder:
            num_batches = num_games // batch_size
        else:
            num_batches = (num_games + batch_size - 1) // batch_size
        
        # Create indices
        indices = np.arange(num_games)
        
        if shuffle:
            rng = jax.random.PRNGKey(42)
            indices = jax.random.permutation(rng, indices)
        
        for batch_idx in range(num_batches):
            start_idx = batch_idx * batch_size
            end_idx = min(start_idx + batch_size, num_games)
            
            if drop_remainder and (end_idx - start_idx) < batch_size:
                continue
            
            batch_indices = indices[start_idx:end_idx]
            
            # Extract batch from sequences
            batch_sequences = jax.tree_map(
                lambda x: x[batch_indices] if isinstance(x, jnp.ndarray) else [x[i] for i in batch_indices],
                game_sequences['sequences']
            )
            
            batch_masks = game_sequences['masks'][batch_indices]
            
            # Extract corresponding features from preprocessed data
            play_indices = []
            for idx in batch_indices:
                game_meta = game_sequences['metadata'][idx]
                seq_length = game_meta['sequence_length']
                plays = batch_sequences['plays'][idx][:seq_length]
                play_indices.extend(plays)
            
            # Get features for these plays
            batch_features = self._extract_play_features(data, play_indices)
            
            yield {
                'features': batch_features,
                'sequences': batch_sequences,
                'masks': batch_masks,
                'metadata': [game_sequences['metadata'][i] for i in batch_indices]
            }
    
    def _extract_play_features(self, 
                             data: Dict[str, jnp.ndarray],
                             play_indices: List[int]) -> Dict[str, jnp.ndarray]:
        """Extract features for specific play indices"""
        
        # Add proper handling for empty indices:
        if len(play_indices) == 0:
            return {
                'offense': jnp.array([], dtype=jnp.float32),
                'defense': jnp.array([], dtype=jnp.float32),
                'game_state': jnp.array([], dtype=jnp.float32),
                'play_context': jnp.array([], dtype=jnp.float32),
                'targets': jnp.array([], dtype=jnp.float32)
            }
        
        features = {}
        
        # Extract each feature type
        for feature_type in ['offense', 'defense', 'game_state', 'play_context']:
            if feature_type in data:
                features[feature_type] = data[feature_type][play_indices]
        
        # Extract targets
        if 'targets' in data:
            features['targets'] = data['targets'][play_indices]
        
        return features
    
    def apply_drive_aware_padding(self, 
                                sequences: Dict[str, jnp.ndarray],
                                masks: jnp.ndarray) -> Tuple[Dict, jnp.ndarray]:
        """
        Apply drive-aware padding strategy for better sequence modeling
        
        This ensures drives are properly padded and separated
        """
        
        batch_size = sequences['plays'].shape[0]
        max_seq_length = sequences['plays'].shape[1]
        
        # Identify drive boundaries
        drive_boundaries = sequences.get('drive_boundaries', 
                                        jnp.zeros_like(sequences['plays']))
        
        # Create drive-aware attention mask
        attention_mask = jnp.zeros((batch_size, max_seq_length, max_seq_length))
        
        for batch_idx in range(batch_size):
            # Find drive start/end positions
            starts = jnp.where(drive_boundaries[batch_idx] == self.DRIVE_START_TOKEN)[0]
            ends = jnp.where(drive_boundaries[batch_idx] == self.DRIVE_END_TOKEN)[0]
            
            # Create attention blocks for each drive
            for start, end in zip(starts, ends):
                if end > start:
                    # Allow attention within drive
                    attention_mask = attention_mask.at[batch_idx, start:end, start:end].set(1.0)
        
        # Apply base mask
        attention_mask = attention_mask * masks[:, None, :] * masks[:, :, None]
        
        return sequences, attention_mask
    
    def compute_sequence_statistics(self, 
                                  game_sequences: Dict[str, Any]) -> Dict[str, Any]:
        """Compute statistics about created sequences"""
        
        metadata = game_sequences['metadata']
        
        stats = {
            'total_games': len(metadata),
            'avg_plays_per_game': np.mean([m['total_plays'] for m in metadata]),
            'avg_drives_per_game': np.mean([m['num_drives'] for m in metadata]),
            'avg_sequence_length': np.mean([m['sequence_length'] for m in metadata]),
            'avg_padding_ratio': np.mean([m['padding_ratio'] for m in metadata]),
            'max_sequence_used': max([m['sequence_length'] for m in metadata]),
            'sequence_coverage': {
                '90_percentile': np.percentile([m['sequence_length'] for m in metadata], 90),
                '95_percentile': np.percentile([m['sequence_length'] for m in metadata], 95),
                '99_percentile': np.percentile([m['sequence_length'] for m in metadata], 99)
            }
        }
        
        return stats
    
    def save_sequences(self, 
                      sequences: Dict[str, Any],
                      path: str):
        """Save processed sequences to disk"""
        
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        # Convert JAX arrays to numpy for pickling
        numpy_sequences = jax.tree_map(
            lambda x: np.array(x) if isinstance(x, jnp.ndarray) else x,
            sequences
        )
        
        with open(path, 'wb') as f:
            pickle.dump(numpy_sequences, f)
        
        self.logger.info(f"💾 Saved sequences to {path}")
    
    def load_sequences(self, path: str) -> Dict[str, Any]:
        """Load processed sequences from disk"""
        
        path = Path(path)
        
        with open(path, 'rb') as f:
            numpy_sequences = pickle.load(f)
        
        # Convert back to JAX arrays
        jax_sequences = jax.tree_map(
            lambda x: jnp.array(x, dtype=jnp.float32) if isinstance(x, np.ndarray) else x,
            numpy_sequences
        )
        
        self.logger.info(f"✅ Loaded sequences from {path}")
        return jax_sequences

class DriveAwareSequentialBatcher(CFBSequentialBatcher):
    """
    Enhanced sequential batcher with drive-level awareness and hierarchical organization
    """
    
    def __init__(self, config: SequentialBatchConfig = None):
        super().__init__(config)
        self.drive_statistics = {}
    
    def create_hierarchical_sequences(self,
                                    preprocessed_data: Dict[str, jnp.ndarray],
                                    game_metadata: Dict[str, np.ndarray],
                                    drive_targets: Optional[Dict] = None,
                                    game_targets: Optional[Dict] = None) -> Dict[str, Any]:
        """
        Create sequences with hierarchical organization for multi-level training
        
        Args:
            preprocessed_data: Preprocessed features
            game_metadata: Game/drive/play IDs
            drive_targets: Optional drive-level targets
            game_targets: Optional game-level targets
            
        Returns:
            Hierarchically organized sequences
        """
        
        # Create base game sequences
        game_sequences = self.create_game_sequences(preprocessed_data, game_metadata)
        
        # Organize hierarchically
        hierarchical_sequences = {
            'play_level': self._organize_play_level(game_sequences, preprocessed_data),
            'drive_level': self._organize_drive_level(game_sequences, drive_targets),
            'game_level': self._organize_game_level(game_sequences, game_targets),
            'cross_level_mappings': self._create_cross_level_mappings(game_sequences)
        }
        
        # Compute hierarchical statistics
        hierarchical_sequences['statistics'] = self._compute_hierarchical_stats(
            hierarchical_sequences
        )
        
        return hierarchical_sequences
    
    def _organize_play_level(self, 
                           game_sequences: Dict,
                           preprocessed_data: Dict) -> Dict[str, jnp.ndarray]:
        """Organize sequences for play-level predictions"""
        
        play_sequences = {}
        
        # Extract play indices from sequences
        play_indices = game_sequences['sequences']['plays'].ravel()
        valid_mask = play_indices > 0  # Filter out padding
        valid_plays = play_indices[valid_mask]
        
        # Get features for valid plays
        for feature_type in ['offense', 'defense', 'game_state', 'play_context']:
            if feature_type in preprocessed_data:
                play_sequences[feature_type] = preprocessed_data[feature_type][valid_plays]
        
        # Get targets
        if 'targets' in preprocessed_data:
            play_sequences['targets'] = preprocessed_data['targets'][valid_plays]
        
        # Add play-to-drive mapping
        drive_indices = game_sequences['sequences']['drives'].ravel()[valid_mask]
        play_sequences['drive_ids'] = drive_indices
        
        return play_sequences
    
    def _organize_drive_level(self,
                            game_sequences: Dict,
                            drive_targets: Optional[Dict]) -> Dict[str, jnp.ndarray]:
        """Organize sequences for drive-level predictions"""
        
        drive_sequences = {}
        
        # Aggregate plays by drives
        unique_drives = []
        drive_features = []
        
        for game_meta in game_sequences['metadata']:
            drive_starts = game_meta['drive_starts']
            drive_ends = game_meta['drive_ends']
            
            for start, end in zip(drive_starts, drive_ends):
                if end > start:
                    # Extract drive sequence
                    drive_slice = slice(start, end)
                    drive_features.append({
                        'start': start,
                        'end': end,
                        'length': end - start,
                        'game_id': game_meta['game_id']
                    })
        
        drive_sequences['drive_features'] = drive_features
        
        # Add drive targets if provided
        if drive_targets:
            drive_sequences['targets'] = drive_targets
        
        return drive_sequences
    
    def _organize_game_level(self,
                           game_sequences: Dict,
                           game_targets: Optional[Dict]) -> Dict[str, jnp.ndarray]:
        """Organize sequences for game-level predictions"""
        
        game_level = {
            'game_ids': [m['game_id'] for m in game_sequences['metadata']],
            'total_plays': [m['total_plays'] for m in game_sequences['metadata']],
            'num_drives': [m['num_drives'] for m in game_sequences['metadata']]
        }
        
        # Add game targets if provided
        if game_targets:
            game_level['targets'] = game_targets
        
        return game_level
    
    def _create_cross_level_mappings(self, game_sequences: Dict) -> Dict[str, Any]:
        """Create mappings between hierarchical levels"""
        
        mappings = {
            'play_to_drive': {},
            'drive_to_game': {},
            'play_to_game': {}
        }
        
        # Build mappings from metadata
        for game_meta in game_sequences['metadata']:
            game_id = game_meta['game_id']
            
            # Track drives in this game
            for drive_idx, (start, end) in enumerate(zip(
                game_meta['drive_starts'], 
                game_meta['drive_ends']
            )):
                drive_key = f"{game_id}_{drive_idx}"
                mappings['drive_to_game'][drive_key] = game_id
                
                # Track plays in this drive
                for play_pos in range(start, end):
                    play_key = f"{game_id}_{play_pos}"
                    mappings['play_to_drive'][play_key] = drive_key
                    mappings['play_to_game'][play_key] = game_id
        
        return mappings
    
    def _compute_hierarchical_stats(self, 
                                  hierarchical_sequences: Dict) -> Dict[str, Any]:
        """Compute statistics for hierarchical sequences"""
        
        stats = {
            'play_level': {
                'total_plays': len(hierarchical_sequences['play_level'].get('drive_ids', [])),
                'unique_drives': len(np.unique(
                    hierarchical_sequences['play_level'].get('drive_ids', [])
                ))
            },
            'drive_level': {
                'total_drives': len(
                    hierarchical_sequences['drive_level'].get('drive_features', [])
                ),
                'avg_drive_length': np.mean([
                    d['length'] for d in 
                    hierarchical_sequences['drive_level'].get('drive_features', [])
                ]) if hierarchical_sequences['drive_level'].get('drive_features') else 0
            },
            'game_level': {
                'total_games': len(hierarchical_sequences['game_level']['game_ids']),
                'avg_plays_per_game': np.mean(
                    hierarchical_sequences['game_level']['total_plays']
                ) if hierarchical_sequences['game_level']['total_plays'] else 0
            }
        }
        
        return stats

# Utility functions
def create_sequential_batcher(config: SequentialBatchConfig = None) -> CFBSequentialBatcher:
    """Factory function to create sequential batcher"""
    return CFBSequentialBatcher(config or SequentialBatchConfig())

def create_hierarchical_batcher(config: SequentialBatchConfig = None) -> DriveAwareSequentialBatcher:
    """Factory function to create hierarchical batcher"""
    return DriveAwareSequentialBatcher(config or SequentialBatchConfig())