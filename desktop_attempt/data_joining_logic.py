# cfb_data_joining_logic.py
"""
Data Joining Logic for CFB Hierarchical Model
Efficiently joins 4 parquet tables with temporal consistency and hierarchical preservation
Optimized for JAX/Flax TPU v2-8 training pipeline
"""

import pandas as pd
import numpy as np
import jax.numpy as jnp
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any, Union
import pyarrow.parquet as pq
import logging
from dataclasses import dataclass
import gc
from datetime import datetime
import yaml
import os
import sys

# Handle both Colab and local environments
if 'google.colab' in sys.modules:
    BASE_PATH = "/content/drive/MyDrive/cfb_model/"
else:
    BASE_PATH = os.path.expanduser("~/cfb_model/")

@dataclass
class JoinConfig:
    """Configuration for data joining operations"""
    base_path: str = "/content/drive/MyDrive/cfb_model/parquet_files/"
    chunk_size_gb: float = 50.0  # Process in 50GB chunks for TPU HBM
    validate_consistency: bool = True
    preserve_hierarchies: bool = True
    drop_duplicates: bool = True
    fill_missing_strategy: str = "smart"  # "smart", "zero", "forward_fill"
    
class CFBDataJoiner:
    """
    Efficient data joining system for CFB hierarchical model
    Handles 4 table joins with minimal memory footprint
    """
    
    def __init__(self, config: JoinConfig = None):
        self.config = config or JoinConfig()
        self.base_path = Path(self.config.base_path)
        
        # Track join statistics
        self.join_stats = {
            'total_plays_processed': 0,
            'successful_joins': 0,
            'validation_failures': 0,
            'memory_peaks_gb': []
        }
        
        # Setup logging  
        self.logger = logging.getLogger('CFBDataJoiner')
        
        # Configure logging for Colab
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.StreamHandler(),
                logging.FileHandler(os.path.join(BASE_PATH, 'logs/model.log'))
            ] if os.path.exists(os.path.join(BASE_PATH, 'logs')) else [logging.StreamHandler()]
        )
        
        # Load football rules for validation
        self.football_rules = self._load_football_rules()
        
    def _load_football_rules(self) -> Dict:
        """Load football rules for validation"""
        return {
            'valid_downs': [1, 2, 3, 4],
            'max_yards_per_play': 100,
            'min_yards_per_play': -20,
            'valid_periods': [1, 2, 3, 4, 5],  # Including OT
            'max_score_change': 8,  # TD + 2pt
            'field_length': 100
        }
    
    def join_yearly_data(self, 
                        years: List[int],
                        weeks: Optional[List[int]] = None) -> Dict[str, Union[np.ndarray, Dict]]:
        """
        Main entry point for joining yearly data
        
        Returns 4-container structure optimized for hierarchical model
        """
        self.logger.info(f"🚀 Starting data join for years: {years}")
        
        all_containers = []
        
        for year in years:
            self.logger.info(f"📂 Processing year {year}...")
            
            # Process year in weekly chunks to manage memory
            year_data = self._process_year_optimized(year, weeks)
            
            if year_data is not None:
                all_containers.append(year_data)
                
            # Memory cleanup
            gc.collect()
        
        # Combine all years
        final_containers = self._combine_year_containers(all_containers)
        
        # Add metadata and validation
        final_containers['metadata'] = self._create_metadata(final_containers)
        
        self.logger.info(f"✅ Join complete: {self.join_stats['successful_joins']:,} plays")
        
        return final_containers
    
    def _process_year_optimized(self, 
                               year: int,
                               weeks: Optional[List[int]]) -> Optional[Dict]:
        """Process single year with memory optimization"""
        
        if weeks is None:
            weeks = list(range(1, 18))  # Standard CFB season
        
        year_containers = {
            'offense_embedding': [],
            'defense_embedding': [],
            'play_embedding': [],
            'game_state_embedding': [],
            'hierarchical_keys': []
        }
        
        for week_idx, week in enumerate(weeks):
            week_data = self._load_and_join_week(year, week)
            
            if week_data is not None:
                # Process into containers
                containers = self._create_embedding_containers(week_data)
                
                for key in year_containers:
                    if key in containers:
                        year_containers[key].append(containers[key])
                
                # Track hierarchical relationships
                hierarchy = self._extract_hierarchy(week_data)
                year_containers['hierarchical_keys'].append(hierarchy)
                
                # Add periodic garbage collection in long-running loops:
                if week_idx % 4 == 0:  # Every 4 weeks
                    gc.collect()
        
        # Concatenate weekly data
        if year_containers['offense_embedding']:
            return self._concatenate_containers(year_containers)
        
        return None
    
    def _load_and_join_week(self, year: int, week: int) -> Optional[pd.DataFrame]:
        """Load and join all 4 tables for a specific week"""
        
        table_names = ['offense_embedding', 'defense_embedding', 
                      'game_state_embedding', 'play_targets']
        
        # Add error handling for missing files
        if not all(Path(self.base_path / f"{table_name}/{year}/week_{week}.parquet").exists() for table_name in table_names):
            self.logger.warning(f"Skipping incomplete week: {year} week {week}")
            return None
        
        tables = {}
        
        # Load each table
        for table_name in table_names:
            path = self.base_path / f"{table_name}/{year}/week_{week}.parquet"
            
            if not path.exists():
                self.logger.debug(f"Missing: {path}")
                return None
            
            try:
                # Load with column selection for memory efficiency
                tables[table_name] = pd.read_parquet(path)
            except Exception as e:
                self.logger.error(f"Error loading {path}: {e}")
                return None
        
        # Perform inner join on play_id
        joined = self._perform_efficient_join(tables)
        
        # Validate if configured
        if self.config.validate_consistency:
            joined = self._validate_joined_data(joined)
        
        return joined
    
    def _perform_efficient_join(self, tables: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """Perform memory-efficient inner join across 4 tables"""
        
        # Start with smallest table for efficiency
        table_sizes = {name: len(df) for name, df in tables.items()}
        start_table = min(table_sizes, key=table_sizes.get)
        
        self.logger.debug(f"Starting join with {start_table} ({table_sizes[start_table]} rows)")
        
        # Initialize with smallest table
        result = tables[start_table].copy()
        
        # Join remaining tables
        for table_name, df in tables.items():
            if table_name == start_table:
                continue
            
            # Clean column names to avoid conflicts
            df = self._prepare_table_for_join(df, table_name)
            
            # Perform join
            result = result.merge(
                df,
                on='play_id',
                how='inner',
                suffixes=('', f'_{table_name}'),
                validate='one_to_one'  # Add validation to catch duplicate issues
            )
            
            # Drop duplicate columns immediately to save memory
            if self.config.drop_duplicates:
                result = self._drop_duplicate_columns(result, table_name)
        
        self.join_stats['successful_joins'] += len(result)
        
        return result
    
    def _prepare_table_for_join(self, df: pd.DataFrame, table_name: str) -> pd.DataFrame:
        """Prepare table for joining by handling column conflicts"""
        
        # Columns that should be preserved from first table only
        preserve_from_first = ['game_id', 'year', 'week', 'created_at', 'updated_at']
        
        # Rename columns that will conflict
        for col in preserve_from_first:
            if col in df.columns and col != 'play_id':
                df = df.drop(columns=[col], errors='ignore')
        
        return df
    
    def _drop_duplicate_columns(self, df: pd.DataFrame, source_table: str) -> pd.DataFrame:
        """Drop duplicate columns after join"""
        
        # Find columns with suffix
        suffix_cols = [col for col in df.columns if f'_{source_table}' in col]
        
        # Drop them
        df = df.drop(columns=suffix_cols, errors='ignore')
        
        return df
    
    def _create_embedding_containers(self, df: pd.DataFrame) -> Dict[str, np.ndarray]:
        """Convert joined DataFrame to 4-container embedding structure"""
        
        containers = {}
        
        # 1. OFFENSE EMBEDDING CONTAINER
        offense_features = self._extract_offense_features(df)
        containers['offense_embedding'] = offense_features
        
        # 2. DEFENSE EMBEDDING CONTAINER  
        defense_features = self._extract_defense_features(df)
        containers['defense_embedding'] = defense_features
        
        # 3. PLAY EMBEDDING CONTAINER
        play_features = self._extract_play_features(df)
        containers['play_embedding'] = play_features
        
        # 4. GAME STATE EMBEDDING CONTAINER
        game_state_features = self._extract_game_state_features(df)
        containers['game_state_embedding'] = game_state_features
        
        return containers
    
    def _extract_offense_features(self, df: pd.DataFrame) -> Dict[str, np.ndarray]:
        """Extract offense embedding features"""
        
        # Categorical features
        categorical = {
            'conference': self._encode_categorical(df['offense_conference']),
            'coach': self._encode_coach_names(df, 'coach_first_name', 'coach_last_name'),
            'home_away': df['home_away_indicator'].fillna(0).values.astype(np.int32),
            'new_coach': df['new_coach_indicator'].fillna(0).values.astype(np.int32)
        }
        
        # Numerical features - all the rate and statistical columns
        numerical_cols = [
            'years_at_school', 'coach_total_experience', 'talent_zscore',
            'run_rate_1st_down', 'run_rate_2nd_short', 'run_rate_2nd_medium',
            'run_rate_2nd_long', 'run_rate_3rd_short', 'run_rate_3rd_medium',
            'run_rate_3rd_long', 'punt_rate_4th_short', 'punt_rate_4th_medium',
            'punt_rate_4th_long', 'fg_attempt_rate_by_field_position',
            'go_for_it_rate_4th_down', 'go_for_2_rate', 'onside_kick_rate',
            'fake_punt_rate', 'avg_seconds_per_play', 'plays_per_game',
            'penalty_rate', 'penalty_yards_per_game',
            'recent_avg_seconds_per_play', 'recent_plays_per_game',
            'recent_penalty_rate', 'recent_run_rate_by_down_distance',
            'opponent_wins', 'opponent_losses', 'home_wins', 'home_losses',
            'away_wins', 'away_losses', 'conference_wins', 'conference_losses',
            'avg_opponent_talent_rating', 'avg_opponent_talent_rating_of_wins',
            'avg_opponent_talent_rating_of_losses', 'strength_of_schedule',
            'wins_vs_favored_opponents', 'losses_vs_weaker_opponents',
            'point_differential_vs_talent_expectation'
        ]
        
        numerical = self._extract_numerical_features(df, numerical_cols)
        
        return {
            'categorical': categorical,
            'numerical': numerical,
            'team_id': df['offense_team_id'].values if 'offense_team_id' in df else None
        }
    
    def _extract_defense_features(self, df: pd.DataFrame) -> Dict[str, np.ndarray]:
        """Extract defense embedding features"""
        
        categorical = {
            'conference': self._encode_categorical(df['defense_conference']),
            'coach': self._encode_coach_names(df, 'defense_coach_first_name', 'defense_coach_last_name'),
            'new_coach': df['defense_new_coach_indicator'].fillna(0).values.astype(np.int32)
        }
        
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
            'defense_recent_stop_rate_by_down_distance',
            'defense_opponent_wins', 'defense_opponent_losses',
            'defense_home_wins', 'defense_home_losses',
            'defense_away_wins', 'defense_away_losses',
            'defense_conference_wins', 'defense_conference_losses',
            'defense_avg_opponent_talent_rating',
            'defense_avg_opponent_talent_rating_of_wins',
            'defense_avg_opponent_talent_rating_of_losses',
            'defense_strength_of_schedule',
            'defense_wins_vs_favored_opponents',
            'defense_losses_vs_weaker_opponents',
            'defense_point_differential_vs_talent_expectation'
        ]
        
        numerical = self._extract_numerical_features(df, numerical_cols)
        
        return {
            'categorical': categorical,
            'numerical': numerical,
            'team_id': df['defense_team_id'].values if 'defense_team_id' in df else None
        }
    
    def _extract_play_features(self, df: pd.DataFrame) -> Dict[str, np.ndarray]:
        """Extract play embedding features (from play_targets)"""
        
        categorical = {
            'down': df['down'].fillna(1).values.astype(np.int32),
            'period': df['period'].fillna(1).values.astype(np.int32)
        }
        
        # Binary play outcome flags
        binary_cols = [
            'is_rush', 'is_pass', 'is_punt', 'is_field_goal',
            'is_extra_point', 'is_kickoff', 'is_penalty', 'is_timeout',
            'is_sack', 'is_administrative', 'is_touchdown', 'is_completion',
            'is_interception', 'is_fumble_lost', 'is_fumble_recovered',
            'is_return_td', 'is_safety', 'is_good', 'is_two_point',
            'is_first_down'
        ]
        
        binary = self._extract_numerical_features(df, binary_cols)
        
        # Continuous features
        numerical_cols = [
            'distance', 'yardline', 'yards_to_goal', 'clock',
            'offense_score', 'defense_score', 'score_differential',
            'yardsGained', 'offense_timeouts', 'defense_timeouts'
        ]
        
        numerical = self._extract_numerical_features(df, numerical_cols)
        
        return {
            'categorical': categorical,
            'binary': binary,
            'numerical': numerical,
            'play_id': df['play_id'].values,
            'drive_id': df['driveId'].values if 'driveId' in df else df['drive_id'].values
        }
    
    def _extract_game_state_features(self, df: pd.DataFrame) -> Dict[str, np.ndarray]:
        """Extract game state embedding features"""
        
        categorical = {
            'venue_id': self._encode_categorical(df['venue_id']) if 'venue_id' in df else np.zeros(len(df)),
            'game_indoors': df['game_indoors'].fillna(0).values.astype(np.int32),
            'is_field_turf': df['is_field_turf'].fillna(0).values.astype(np.int32),
            'is_offense_home_team': df['is_offense_home_team'].fillna(0).values.astype(np.int32),
            'conference_game': df['conference_game'].fillna(0).values.astype(np.int32),
            'bowl_game': df['bowl_game'].fillna(0).values.astype(np.int32)
        }
        
        # Wind direction as circular feature
        if 'wind_direction' in df:
            categorical['wind_direction'] = self._encode_wind_direction(df['wind_direction'].fillna(0))
        
        numerical_cols = [
            'drive_number', 'drive_plays_so_far', 'drive_yards_so_far',
            'drive_start_yardline', 'drive_time_elapsed',
            'total_seconds_remaining', 'temperature', 'humidity',
            'wind_speed', 'precipitation', 'is_red_zone', 'is_goal_line',
            'is_two_minute_warning', 'is_garbage_time', 'is_money_down',
            'is_plus_territory'
        ]
        
        numerical = self._extract_numerical_features(df, numerical_cols)
        
        return {
            'categorical': categorical,
            'numerical': numerical,
            'game_id': df['game_id'].values
        }
    
    def _extract_numerical_features(self, df: pd.DataFrame, columns: List[str]) -> np.ndarray:
        """Extract and handle numerical features with missing value strategy"""
        
        available_cols = [col for col in columns if col in df.columns]
        
        if not available_cols:
            return np.zeros((len(df), len(columns)), dtype=np.float32)
        
        # Extract available columns
        features = []
        for col in columns:
            if col in df.columns:
                if self.config.fill_missing_strategy == "smart":
                    # Smart filling based on feature type
                    if 'rate' in col or 'probability' in col:
                        # Rates default to 0.5 (neutral)
                        features.append(df[col].fillna(0.5).values)
                    elif 'score' in col or 'yards' in col:
                        # Scores/yards default to 0
                        features.append(df[col].fillna(0).values)
                    else:
                        # General default to 0
                        features.append(df[col].fillna(0).values)
                elif self.config.fill_missing_strategy == "forward_fill":
                    features.append(df[col].fillna(method='ffill').fillna(0).values)
                else:  # "zero"
                    features.append(df[col].fillna(0).values)
            else:
                # Column doesn't exist, fill with zeros
                features.append(np.zeros(len(df)))
        
        return np.column_stack(features).astype(np.float32)
    
    def _encode_categorical(self, series: pd.Series) -> np.ndarray:
        """Encode categorical variable as integers"""
        if series.dtype == 'object':
            categories = series.fillna('MISSING').astype('category')
            return categories.cat.codes.values.astype(np.int32)
        return series.fillna(0).values.astype(np.int32)
    
    def _encode_coach_names(self, df: pd.DataFrame, first_col: str, last_col: str) -> np.ndarray:
        """Encode coach names as unique identifiers"""
        if first_col in df.columns and last_col in df.columns:
            full_names = df[first_col].fillna('') + '_' + df[last_col].fillna('')
            return self._encode_categorical(full_names)
        return np.zeros(len(df), dtype=np.int32)
    
    def _encode_wind_direction(self, wind_degrees: pd.Series) -> np.ndarray:
        """Convert wind direction to 16-bin categorical"""
        # Convert degrees to 16 compass directions
        bins = (wind_degrees.fillna(0) / 22.5).astype(int) % 16
        return bins.values.astype(np.int32)
    
    def _extract_hierarchy(self, df: pd.DataFrame) -> Dict[str, np.ndarray]:
        """Extract hierarchical relationships (game -> drive -> play)"""
        
        hierarchy = {
            'game_ids': df['game_id'].values,
            'drive_ids': df['driveId'].values if 'driveId' in df else df['drive_id'].values,
            'play_ids': df['play_id'].values,
            'drive_numbers': df['driveNumber'].values if 'driveNumber' in df else np.zeros(len(df))
        }
        
        # Create mappings
        unique_games = np.unique(hierarchy['game_ids'])
        hierarchy['game_to_drives'] = {}
        
        for game_id in unique_games:
            game_mask = hierarchy['game_ids'] == game_id
            game_drives = np.unique(hierarchy['drive_ids'][game_mask])
            hierarchy['game_to_drives'][game_id] = game_drives
        
        return hierarchy
    
    def _validate_joined_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Validate joined data using football rules"""
        
        violations = []
        
        # Validate downs
        invalid_downs = ~df['down'].isin(self.football_rules['valid_downs'] + [0, -1])
        if invalid_downs.any():
            violations.append(f"Invalid downs: {invalid_downs.sum()} plays")
            df.loc[invalid_downs, 'down'] = 1
        
        # Validate yards gained
        df['yardsGained'] = df['yardsGained'].clip(
            self.football_rules['min_yards_per_play'],
            self.football_rules['max_yards_per_play']
        )
        
        # Validate field position
        df['yardline'] = df['yardline'].clip(0, 100)
        df['yards_to_goal'] = df['yards_to_goal'].clip(0, 100)
        
        # Log violations
        if violations:
            self.join_stats['validation_failures'] += len(violations)
            self.logger.debug(f"Fixed {len(violations)} validation issues")
        
        return df
    
    def _concatenate_containers(self, containers: Dict[str, List]) -> Dict[str, Any]:
        """Concatenate weekly containers into single arrays"""
        
        result = {}
        
        for key in ['offense_embedding', 'defense_embedding', 'play_embedding', 'game_state_embedding']:
            if key in containers and containers[key]:
                # Concatenate all sub-dictionaries
                concatenated = {}
                
                # Get all sub-keys from first element
                if containers[key]:
                    sub_keys = containers[key][0].keys()
                    
                    for sub_key in sub_keys:
                        arrays = [c[sub_key] for c in containers[key] if sub_key in c]
                        
                        if arrays and arrays[0] is not None:
                            if isinstance(arrays[0], dict):
                                # Handle nested dictionaries
                                concatenated[sub_key] = self._concatenate_nested_dict(arrays)
                            else:
                                # Concatenate arrays
                                concatenated[sub_key] = np.concatenate(arrays, axis=0)
                
                result[key] = concatenated
        
        # Handle hierarchical keys
        if 'hierarchical_keys' in containers and containers['hierarchical_keys']:
            result['hierarchy'] = self._merge_hierarchies(containers['hierarchical_keys'])
        
        return result
    
    def _concatenate_nested_dict(self, dict_list: List[Dict]) -> Dict:
        """Concatenate nested dictionaries"""
        result = {}
        
        if dict_list:
            keys = dict_list[0].keys()
            for key in keys:
                arrays = [d[key] for d in dict_list if key in d and d[key] is not None]
                if arrays:
                    result[key] = np.concatenate(arrays, axis=0)
        
        return result
    
    def _merge_hierarchies(self, hierarchy_list: List[Dict]) -> Dict:
        """Merge hierarchical mappings from multiple weeks"""
        
        merged = {
            'game_ids': [],
            'drive_ids': [],
            'play_ids': [],
            'drive_numbers': [],
            'game_to_drives': {}
        }
        
        for hierarchy in hierarchy_list:
            for key in ['game_ids', 'drive_ids', 'play_ids', 'drive_numbers']:
                if key in hierarchy:
                    merged[key].append(hierarchy[key])
            
            if 'game_to_drives' in hierarchy:
                merged['game_to_drives'].update(hierarchy['game_to_drives'])
        
        # Concatenate arrays
        for key in ['game_ids', 'drive_ids', 'play_ids', 'drive_numbers']:
            if merged[key]:
                merged[key] = np.concatenate(merged[key])
        
        return merged
    
    def _combine_year_containers(self, year_containers: List[Dict]) -> Dict[str, Any]:
        """Combine containers from multiple years"""
        
        if not year_containers:
            return {}
        
        if len(year_containers) == 1:
            return year_containers[0]
        
        # Combine all years
        combined = {}
        
        for key in year_containers[0].keys():
            if key == 'metadata':
                continue
            
            combined[key] = self._combine_container_list(
                [yc[key] for yc in year_containers if key in yc]
            )
        
        return combined
    
    def _combine_container_list(self, container_list: List[Dict]) -> Dict:
        """Combine list of containers"""
        
        if not container_list:
            return {}
        
        result = {}
        
        # Get all keys
        all_keys = set()
        for container in container_list:
            all_keys.update(container.keys())
        
        for key in all_keys:
            items = []
            for container in container_list:
                if key in container and container[key] is not None:
                    items.append(container[key])
            
            if items:
                if isinstance(items[0], dict):
                    result[key] = self._combine_container_list(items)
                elif isinstance(items[0], np.ndarray):
                    result[key] = np.concatenate(items, axis=0)
                else:
                    result[key] = items[0]  # Keep first for non-array items
        
        return result
    
    def _create_metadata(self, containers: Dict) -> Dict[str, Any]:
        """Create metadata for joined data"""
        
        total_plays = 0
        if 'play_embedding' in containers and 'play_id' in containers['play_embedding']:
            total_plays = len(containers['play_embedding']['play_id'])
        
        metadata = {
            'total_plays': total_plays,
            'join_statistics': self.join_stats,
            'timestamp': datetime.now().isoformat(),
            'config': {
                'base_path': str(self.config.base_path),
                'validation_enabled': self.config.validate_consistency,
                'fill_strategy': self.config.fill_missing_strategy
            }
        }
        
        # Add hierarchy statistics
        if 'hierarchy' in containers:
            hierarchy = containers['hierarchy']
            metadata['hierarchy_stats'] = {
                'unique_games': len(np.unique(hierarchy['game_ids'])) if 'game_ids' in hierarchy else 0,
                'unique_drives': len(np.unique(hierarchy['drive_ids'])) if 'drive_ids' in hierarchy else 0,
                'avg_plays_per_drive': self._calculate_avg_plays_per_drive(hierarchy)
            }
        
        return metadata
    
    def _calculate_avg_plays_per_drive(self, hierarchy: Dict) -> float:
        """Calculate average plays per drive"""
        
        if 'drive_ids' not in hierarchy or len(hierarchy['drive_ids']) == 0:
            return 0.0
        
        unique_drives, counts = np.unique(hierarchy['drive_ids'], return_counts=True)
        return float(np.mean(counts))

# Utility function for easy usage
def join_cfb_data(years: List[int], 
                  weeks: Optional[List[int]] = None,
                  config: Optional[JoinConfig] = None) -> Dict[str, Any]:
    """
    Convenience function to join CFB data
    
    Returns 4-container structure ready for model training
    """
    joiner = CFBDataJoiner(config)
    return joiner.join_yearly_data(years, weeks)