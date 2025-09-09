# data_preprocessing.py
"""
CFB Data Preprocessing Pipeline for JAX/Flax TPU Training
Optimized for Google Colab TPU v2-8 with 512GB HBM
"""

import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
import pickle
from dataclasses import dataclass
import gc
from functools import partial
import logging

# Configure JAX for TPU
jax.config.update('jax_platform_name', 'tpu')

@dataclass
class DataConfig:
    """Configuration for data preprocessing"""
    base_path: str = "/content/drive/MyDrive/cfb_model/parquet_files/"
    cache_dir: str = "/tmp/cfb_cache/"
    max_plays_per_drive: int = 18  # 99.1% coverage
    max_drives_per_game: int = 32
    batch_size: int = 1536  # Optimized for TPU v2-8
    prefetch_buffer: int = 4
    shuffle_buffer: int = 10000
    
class CFBDataPreprocessor:
    """
    JAX-optimized data preprocessing for CFB hierarchical model
    """
    
    def __init__(self, config: DataConfig = None):
        self.config = config or DataConfig()
        self.base_path = Path(self.config.base_path)
        self.cache_dir = Path(self.config.cache_dir)
        self.cache_dir.mkdir(exist_ok=True, parents=True)
        
        # Feature dimensions from analysis
        self.feature_dims = {
            'offense_numerical': 43,
            'offense_categorical': 4,
            'defense_numerical': 40,
            'defense_categorical': 3,
            'game_state_numerical': 26,
            'game_state_categorical': 7,
            'play_context_numerical': 28,
            'play_context_categorical': 2
        }
        
        # Label encoders for categorical features
        self.label_encoders = {}
        self.is_fitted = False
        
        # Setup logging
        self.logger = logging.getLogger('CFBDataPreprocessor')
        logging.basicConfig(level=logging.INFO)
        
    def load_and_preprocess(self, years: List[int], 
                          weeks: Optional[List[int]] = None,
                          split: str = 'train') -> Dict[str, jnp.ndarray]:
        """
        Main preprocessing pipeline
        
        Returns JAX arrays ready for TPU training
        """
        self.logger.info(f"📂 Loading data for years: {years}, split: {split}")
        
        # Load raw parquet data
        raw_df = self._load_parquet_files(years, weeks)
        
        # Fit encoders on training data only
        if split == 'train' and not self.is_fitted:
            self._fit_encoders(raw_df)
            self.is_fitted = True
            
        # Process features
        processed = self._process_features(raw_df)
        
        # Create sequences
        sequences = self._create_game_sequences(processed, raw_df)
        
        # Convert to JAX arrays
        jax_data = self._convert_to_jax(sequences)
        
        # Cache processed data
        self._cache_processed_data(jax_data, split)
        
        self.logger.info(f"✅ Preprocessing complete for {split}")
        return jax_data
    
    def _load_parquet_files(self, years: List[int], 
                          weeks: Optional[List[int]] = None) -> pd.DataFrame:
        """Load and join parquet files"""
        
        all_dfs = []
        
        for year in years:
            year_weeks = weeks if weeks else range(1, 18)
            
            for week in year_weeks:
                try:
                    # Load all 4 tables for this week
                    tables = {}
                    table_names = ['offense_embedding', 'defense_embedding', 
                                 'game_state_embedding', 'play_targets']
                    
                    for table in table_names:
                        path = self.base_path / f"{table}/{year}/week_{week}.parquet"
                        if path.exists():
                            tables[table] = pd.read_parquet(path)
                    
                    if len(tables) == 4:
                        # Join on play_id
                        joined = tables['offense_embedding']
                        for table_name in ['defense_embedding', 'game_state_embedding', 'play_targets']:
                            joined = joined.merge(tables[table_name], on='play_id', 
                                                 how='inner', suffixes=('', f'_{table_name}'))
                        all_dfs.append(joined)
                        
                except Exception as e:
                    self.logger.warning(f"Missing data for {year} week {week}: {e}")
                    
        if all_dfs:
            combined_df = pd.concat(all_dfs, ignore_index=True)
            self.logger.info(f"Loaded {len(combined_df):,} plays")
            return self._clean_columns(combined_df)
        else:
            raise ValueError("No data loaded")
    
    def _clean_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Remove duplicate columns from joins"""
        
        # Remove duplicate columns
        duplicate_patterns = ['_defense_embedding', '_game_state_embedding', '_play_targets']
        for pattern in duplicate_patterns:
            cols_to_drop = [col for col in df.columns if pattern in col]
            df = df.drop(columns=cols_to_drop, errors='ignore')
            
        # Remove duplicate game_id columns
        game_id_cols = [col for col in df.columns if 'game_id' in col and col != 'game_id']
        df = df.drop(columns=game_id_cols, errors='ignore')
        
        return df
    
    def _fit_encoders(self, df: pd.DataFrame):
        """Fit label encoders for categorical features"""
        
        # Offense categorical
        self.label_encoders['offense_conference'] = self._create_encoder(df['offense_conference'])
        df['coach_full'] = df['coach_first_name'].fillna('') + '_' + df['coach_last_name'].fillna('')
        self.label_encoders['coach_offense'] = self._create_encoder(df['coach_full'])
        
        # Defense categorical
        self.label_encoders['defense_conference'] = self._create_encoder(df['defense_conference'])
        df['defense_coach_full'] = df['defense_coach_first_name'].fillna('') + '_' + df['defense_coach_last_name'].fillna('')
        self.label_encoders['coach_defense'] = self._create_encoder(df['defense_coach_full'])
        
        # Game state categorical
        self.label_encoders['venue_id'] = self._create_encoder(df['venue_id'])
        
        # Play context categorical
        self.label_encoders['down'] = self._create_encoder(df['down'].fillna(1))
        self.label_encoders['period'] = self._create_encoder(df['period'].fillna(1))
        
        self.logger.info(f"✅ Fitted {len(self.label_encoders)} label encoders")
    
    def _create_encoder(self, series: pd.Series) -> Dict[Any, int]:
        """Create label encoder for categorical feature"""
        unique_vals = series.dropna().unique()
        return {val: idx for idx, val in enumerate(unique_vals)}
    
    def _process_features(self, df: pd.DataFrame) -> Dict[str, np.ndarray]:
        """Process all features into numerical arrays"""
        
        processed = {}
        
        # OFFENSE FEATURES
        processed['offense'] = self._process_offense_features(df)
        
        # DEFENSE FEATURES  
        processed['defense'] = self._process_defense_features(df)
        
        # GAME STATE FEATURES
        processed['game_state'] = self._process_game_state_features(df)
        
        # PLAY CONTEXT FEATURES
        processed['play_context'] = self._process_play_context_features(df)
        
        # TARGET VARIABLES
        processed['targets'] = self._process_targets(df)
        
        # METADATA
        processed['metadata'] = {
            'game_ids': df['game_id'].values,
            'play_ids': df['play_id'].values,
            'drive_ids': df['driveId'].values if 'driveId' in df else df['drive_id'].values
        }
        
        return processed
    
    def _process_offense_features(self, df: pd.DataFrame) -> Dict[str, np.ndarray]:
        """Process offense embedding features"""
        
        # Categorical encoding
        df['coach_full'] = df['coach_first_name'].fillna('') + '_' + df['coach_last_name'].fillna('')
        
        categorical = np.column_stack([
            self._encode_categorical(df['offense_conference'], 'offense_conference'),
            self._encode_categorical(df['coach_full'], 'coach_offense'),
            df['home_away_indicator'].fillna(0).values,
            df['new_coach_indicator'].fillna(0).values
        ])
        
        # Numerical features
        numerical_cols = [
            'years_at_school', 'coach_total_experience', 'talent_zscore',
            'run_rate_1st_down', 'run_rate_2nd_short', 'run_rate_2nd_medium',
            'run_rate_2nd_long', 'run_rate_3rd_short', 'run_rate_3rd_medium',
            'run_rate_3rd_long', 'punt_rate_4th_short', 'punt_rate_4th_medium',
            'punt_rate_4th_long', 'fg_attempt_rate_by_field_position',
            'go_for_it_rate_4th_down', 'go_for_2_rate', 'onside_kick_rate',
            'fake_punt_rate', 'avg_seconds_per_play', 'plays_per_game',
            'penalty_rate', 'penalty_yards_per_game', 'recent_avg_seconds_per_play',
            'recent_plays_per_game', 'recent_penalty_rate',
            'recent_run_rate_by_down_distance', 'opponent_wins', 'opponent_losses',
            'home_wins', 'home_losses', 'away_wins', 'away_losses',
            'conference_wins', 'conference_losses', 'avg_opponent_talent_rating',
            'avg_opponent_talent_rating_of_wins', 'avg_opponent_talent_rating_of_losses',
            'strength_of_schedule', 'wins_vs_favored_opponents',
            'losses_vs_weaker_opponents', 'point_differential_vs_talent_expectation'
        ]
        
        numerical = np.column_stack([df[col].fillna(0).values for col in numerical_cols])
        
        return {'categorical': categorical.astype(np.int32), 
                'numerical': numerical.astype(np.float32)}
    
    def _process_defense_features(self, df: pd.DataFrame) -> Dict[str, np.ndarray]:
        """Process defense embedding features"""
        
        df['defense_coach_full'] = df['defense_coach_first_name'].fillna('') + '_' + df['defense_coach_last_name'].fillna('')
        
        categorical = np.column_stack([
            self._encode_categorical(df['defense_conference'], 'defense_conference'),
            self._encode_categorical(df['defense_coach_full'], 'coach_defense'),
            df['defense_new_coach_indicator'].fillna(0).values
        ])
        
        numerical_cols = [
            'defense_years_at_school', 'defense_coach_total_experience',
            'defense_talent_zscore', 'defense_run_stop_rate_1st_down',
            'defense_run_stop_rate_2nd_short', 'defense_run_stop_rate_2nd_medium',
            'defense_run_stop_rate_2nd_long', 'defense_run_stop_rate_3rd_short',
            'defense_run_stop_rate_3rd_medium', 'defense_run_stop_rate_3rd_long',
            'defense_red_zone_fg_rate', 'defense_red_zone_stop_rate',
            'defense_avg_seconds_allowed_per_play', 'defense_plays_allowed_per_game',
            'defense_penalty_rate', 'defense_penalty_yards_per_game',
            'defense_recent_avg_seconds_allowed_per_play',
            'defense_recent_plays_allowed_per_game', 'defense_recent_penalty_rate',
            'defense_recent_stop_rate_by_down_distance', 'defense_opponent_wins',
            'defense_opponent_losses', 'defense_home_wins', 'defense_home_losses',
            'defense_away_wins', 'defense_away_losses', 'defense_conference_wins',
            'defense_conference_losses', 'defense_avg_opponent_talent_rating',
            'defense_avg_opponent_talent_rating_of_wins',
            'defense_avg_opponent_talent_rating_of_losses',
            'defense_strength_of_schedule', 'defense_wins_vs_favored_opponents',
            'defense_losses_vs_weaker_opponents',
            'defense_point_differential_vs_talent_expectation'
        ]
        
        numerical = np.column_stack([df[col].fillna(0).values for col in numerical_cols])
        
        return {'categorical': categorical.astype(np.int32),
                'numerical': numerical.astype(np.float32)}
    
    def _process_game_state_features(self, df: pd.DataFrame) -> Dict[str, np.ndarray]:
        """Process game state features"""
        
        categorical = np.column_stack([
            self._encode_categorical(df['venue_id'], 'venue_id'),
            df['game_indoors'].fillna(0).values,
            df['is_field_turf'].fillna(0).values,
            df['is_offense_home_team'].fillna(0).values,
            df['conference_game'].fillna(0).values,
            df['bowl_game'].fillna(0).values,
            self._encode_wind_direction(df['wind_direction'].fillna(0).values)
        ])
        
        numerical_cols = [
            'drive_number', 'drive_plays_so_far', 'drive_yards_so_far',
            'drive_start_yardline', 'drive_time_elapsed', 'down', 'distance',
            'yardline', 'yards_to_goal', 'period', 'total_seconds_remaining',
            'offense_score', 'defense_score', 'score_differential',
            'offense_timeouts', 'defense_timeouts', 'temperature', 'humidity',
            'wind_speed', 'precipitation', 'is_red_zone', 'is_goal_line',
            'is_two_minute_warning', 'is_garbage_time', 'is_money_down',
            'is_plus_territory'
        ]
        
        numerical = np.column_stack([df[col].fillna(0).values for col in numerical_cols])
        
        return {'categorical': categorical.astype(np.int32),
                'numerical': numerical.astype(np.float32)}
    
    def _process_play_context_features(self, df: pd.DataFrame) -> Dict[str, np.ndarray]:
        """Process play context features"""
        
        categorical = np.column_stack([
            self._encode_categorical(df['down'].fillna(1), 'down'),
            self._encode_categorical(df['period'].fillna(1), 'period')
        ])
        
        binary_cols = [
            'is_rush', 'is_pass', 'is_punt', 'is_field_goal',
            'is_extra_point', 'is_kickoff', 'is_penalty', 'is_timeout',
            'is_sack', 'is_administrative', 'is_touchdown', 'is_completion',
            'is_interception', 'is_fumble_lost', 'is_fumble_recovered',
            'is_return_td', 'is_safety', 'is_good', 'is_two_point',
            'is_first_down'
        ]
        
        numerical_cols = [
            'distance', 'yardline', 'yards_to_goal', 'clock',
            'offense_score', 'defense_score', 'score_differential', 'yardsGained'
        ]
        
        binary = np.column_stack([df[col].fillna(0).values for col in binary_cols])
        numerical = np.column_stack([df[col].fillna(0).values for col in numerical_cols])
        
        return {'categorical': categorical.astype(np.int32),
                'binary': binary.astype(np.float32),
                'numerical': numerical.astype(np.float32)}
    
    def _process_targets(self, df: pd.DataFrame) -> Dict[str, np.ndarray]:
        """Process target variables"""
        
        # Play-level targets
        play_type_cols = ['is_rush', 'is_pass', 'is_punt', 'is_field_goal',
                         'is_extra_point', 'is_kickoff', 'is_penalty', 'is_timeout']
        play_type = np.column_stack([df[col].fillna(0).values for col in play_type_cols])
        
        success_cols = ['is_sack', 'is_touchdown', 'is_completion', 'is_interception',
                       'is_fumble_lost', 'is_fumble_recovered', 'is_return_td',
                       'is_safety', 'is_good', 'is_two_point', 'is_first_down']
        success_flags = np.column_stack([df[col].fillna(0).values for col in success_cols])
        
        yards_gained = df['yardsGained'].fillna(0).values.reshape(-1, 1)
        
        return {
            'play_type': play_type.astype(np.float32),
            'success_flags': success_flags.astype(np.float32),
            'yards_gained': yards_gained.astype(np.float32)
        }
    
    def _encode_categorical(self, series: pd.Series, encoder_name: str) -> np.ndarray:
        """Encode categorical variable using fitted encoder"""
        
        if encoder_name not in self.label_encoders:
            # Create encoder if not exists (for validation/test sets)
            unique_vals = series.dropna().unique()
            self.label_encoders[encoder_name] = {val: idx for idx, val in enumerate(unique_vals)}
            
        encoder = self.label_encoders[encoder_name]
        return np.array([encoder.get(val, 0) for val in series])
    
    def _encode_wind_direction(self, wind_degrees: np.ndarray) -> np.ndarray:
        """Convert wind direction to 16 bins"""
        bins = (wind_degrees / 22.5).astype(np.int32) % 16
        return bins
    
    def _create_game_sequences(self, processed: Dict, df: pd.DataFrame) -> Dict[str, np.ndarray]:
        """Create padded sequences organized by games"""
        
        sequences = {
            'offense': [],
            'defense': [],
            'game_state': [],
            'play_context': [],
            'targets': [],
            'masks': []
        }
        
        # Group by game_id
        for game_id, game_df in df.groupby('game_id'):
            game_indices = game_df.index.values
            
            # Create game sequence
            game_seq = self._create_single_game_sequence(processed, game_indices)
            
            for key in sequences:
                if key in game_seq:
                    sequences[key].append(game_seq[key])
        
        # Stack all games
        for key in sequences:
            if sequences[key]:
                sequences[key] = np.stack(sequences[key], axis=0)
                
        return sequences
    
    def _create_single_game_sequence(self, processed: Dict, indices: np.ndarray) -> Dict:
        """Create padded sequence for single game"""
        
        max_plays = self.config.max_plays_per_drive * self.config.max_drives_per_game
        
        # Initialize padded arrays
        game_seq = {}
        
        # Process each feature type
        for feature_type in ['offense', 'defense', 'game_state', 'play_context']:
            feat_dict = processed[feature_type]
            
            # Combine categorical and numerical
            if 'categorical' in feat_dict and 'numerical' in feat_dict:
                cat_feats = feat_dict['categorical'][indices]
                num_feats = feat_dict['numerical'][indices]
                combined = np.concatenate([cat_feats, num_feats], axis=1)
            elif 'binary' in feat_dict:
                # Play context special case
                cat_feats = feat_dict['categorical'][indices]
                bin_feats = feat_dict['binary'][indices]
                num_feats = feat_dict['numerical'][indices]
                combined = np.concatenate([cat_feats, bin_feats, num_feats], axis=1)
            else:
                combined = feat_dict['numerical'][indices]
            
            # Pad to max length
            padded = np.zeros((max_plays, combined.shape[1]), dtype=np.float32)
            actual_length = min(len(indices), max_plays)
            padded[:actual_length] = combined[:actual_length]
            
            game_seq[feature_type] = padded
        
        # Process targets
        target_dict = processed['targets']
        target_combined = np.concatenate([
            target_dict['play_type'][indices],
            target_dict['success_flags'][indices],
            target_dict['yards_gained'][indices]
        ], axis=1)
        
        target_padded = np.zeros((max_plays, target_combined.shape[1]), dtype=np.float32)
        target_padded[:actual_length] = target_combined[:actual_length]
        game_seq['targets'] = target_padded
        
        # Create mask
        mask = np.zeros(max_plays, dtype=np.float32)
        mask[:actual_length] = 1.0
        game_seq['masks'] = mask
        
        return game_seq
    
    def _convert_to_jax(self, sequences: Dict[str, np.ndarray]) -> Dict[str, jnp.ndarray]:
        """Convert numpy arrays to JAX arrays"""
        
        jax_data = {}
        for key, value in sequences.items():
            if isinstance(value, np.ndarray):
                jax_data[key] = jnp.array(value)
            else:
                jax_data[key] = value
                
        return jax_data
    
    def _cache_processed_data(self, data: Dict[str, jnp.ndarray], split: str):
        """Cache processed data for faster loading"""
        
        cache_path = self.cache_dir / f"{split}_processed.pkl"
        
        # Convert JAX arrays back to numpy for pickling
        numpy_data = {}
        for key, value in data.items():
            if isinstance(value, jnp.ndarray):
                numpy_data[key] = np.array(value)
            else:
                numpy_data[key] = value
        
        with open(cache_path, 'wb') as f:
            pickle.dump(numpy_data, f)
            
        self.logger.info(f"💾 Cached {split} data to {cache_path}")
    
    def load_cached_data(self, split: str) -> Optional[Dict[str, jnp.ndarray]]:
        """Load cached preprocessed data"""
        
        cache_path = self.cache_dir / f"{split}_processed.pkl"
        
        if cache_path.exists():
            with open(cache_path, 'rb') as f:
                numpy_data = pickle.load(f)
            
            # Convert back to JAX arrays
            jax_data = {}
            for key, value in numpy_data.items():
                if isinstance(value, np.ndarray):
                    jax_data[key] = jnp.array(value)
                else:
                    jax_data[key] = value
                    
            self.logger.info(f"✅ Loaded cached {split} data")
            return jax_data
        
        return None
    
    def create_data_iterator(self, data: Dict[str, jnp.ndarray], 
                           shuffle: bool = True) -> Any:
        """Create data iterator for training"""
        
        num_samples = data['offense'].shape[0]
        
        if shuffle:
            perm = jax.random.permutation(jax.random.PRNGKey(42), num_samples)
            data = jax.tree_map(lambda x: x[perm] if isinstance(x, jnp.ndarray) else x, data)
        
        # Batch data
        num_batches = num_samples // self.config.batch_size
        
        for i in range(num_batches):
            start_idx = i * self.config.batch_size
            end_idx = start_idx + self.config.batch_size
            
            batch = jax.tree_map(
                lambda x: x[start_idx:end_idx] if isinstance(x, jnp.ndarray) else x,
                data
            )
            
            yield batch