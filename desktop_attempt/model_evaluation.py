"""
Model Evaluation and Checkpoint Management
Complete implementation for model evaluation, metrics computation, and checkpoint saving/loading
"""

import jax
import jax.numpy as jnp
import flax
from flax.training import checkpoints
import optax
from typing import Dict, Tuple, Optional, Any, List
import numpy as np
import pandas as pd
from pathlib import Path
import json
import pickle


class ModelEvaluator:
    """Comprehensive model evaluation with hierarchical metrics"""
    
    def __init__(self, model, config):
        self.model = model
        self.config = config
        self.metrics_history = []
        
    def evaluate_batch(self, params, batch, rng):
        """Evaluate single batch and return predictions and metrics"""
        # Get model predictions
        predictions = self.model.apply(
            params,
            batch,
            training=False,
            rngs={'dropout': rng}
        )
        
        metrics = {}
        
        # Play-level metrics
        if 'play_outcomes' in predictions and 'play_labels' in batch:
            play_probs = jax.nn.softmax(predictions['play_outcomes'])
            play_preds = jnp.argmax(play_probs, axis=-1)
            play_accuracy = jnp.mean(play_preds == batch['play_labels'])
            metrics['play_accuracy'] = float(play_accuracy)
            
        if 'play_yards' in predictions and 'play_yards_actual' in batch:
            play_yards_error = jnp.sqrt(jnp.mean(
                (predictions['play_yards'] - batch['play_yards_actual']) ** 2
            ))
            metrics['play_yards_rmse'] = float(play_yards_error)
        
        # Drive-level metrics
        if 'drive_outcomes' in predictions and 'drive_labels' in batch:
            drive_probs = jax.nn.softmax(predictions['drive_outcomes'])
            drive_preds = jnp.argmax(drive_probs, axis=-1)
            drive_accuracy = jnp.mean(drive_preds == batch['drive_labels'])
            metrics['drive_accuracy'] = float(drive_accuracy)
            
        if 'drive_points' in predictions and 'drive_points_actual' in batch:
            drive_points_error = jnp.sqrt(jnp.mean(
                (predictions['drive_points'] - batch['drive_points_actual']) ** 2
            ))
            metrics['drive_points_rmse'] = float(drive_points_error)
        
        # Game-level metrics (most important for Vegas comparison)
        if 'home_score' in predictions and 'home_score_actual' in batch:
            home_error = jnp.sqrt(jnp.mean(
                (predictions['home_score'] - batch['home_score_actual']) ** 2
            ))
            metrics['home_score_rmse'] = float(home_error)
            
        if 'away_score' in predictions and 'away_score_actual' in batch:
            away_error = jnp.sqrt(jnp.mean(
                (predictions['away_score'] - batch['away_score_actual']) ** 2
            ))
            metrics['away_score_rmse'] = float(away_error)
            
        # Combined game RMSE (what we compare to Vegas)
        if 'home_score_rmse' in metrics and 'away_score_rmse' in metrics:
            metrics['game_rmse'] = (metrics['home_score_rmse'] + metrics['away_score_rmse']) / 2
            
        # Spread metrics
        if 'spread' in predictions and 'spread_actual' in batch:
            spread_error = jnp.sqrt(jnp.mean(
                (predictions['spread'] - batch['spread_actual']) ** 2
            ))
            metrics['spread_rmse'] = float(spread_error)
            
            # Against the spread accuracy
            spread_sign_correct = jnp.sign(predictions['spread']) == jnp.sign(batch['spread_actual'])
            metrics['ats_accuracy'] = float(jnp.mean(spread_sign_correct))
        
        # Total points over/under
        if 'total_points' in predictions and 'total_points_actual' in batch:
            total_error = jnp.sqrt(jnp.mean(
                (predictions['total_points'] - batch['total_points_actual']) ** 2
            ))
            metrics['total_points_rmse'] = float(total_error)
        
        # Hierarchical consistency metrics
        metrics['state_consistency'] = self.compute_consistency(predictions, batch)
        
        return predictions, metrics
    
    def compute_consistency(self, predictions, batch):
        """Compute hierarchical consistency between levels"""
        consistency_score = 1.0
        
        # Check if drive points sum to approximately game scores
        if 'drive_points' in predictions and 'home_score' in predictions:
            # Simplified check - in practice would need proper aggregation
            total_predicted = predictions['home_score'] + predictions['away_score']
            drive_sum = jnp.sum(predictions['drive_points'])
            
            # Compute consistency as inverse of normalized difference
            diff = jnp.abs(drive_sum - total_predicted) / (total_predicted + 1e-6)
            consistency_score *= jnp.exp(-diff)  # Exponential penalty for inconsistency
        
        return float(consistency_score)
    
    def evaluate_full_dataset(self, params, data_loader, rng):
        """Evaluate on full dataset and return aggregated metrics"""
        all_metrics = []
        all_predictions = []
        
        for batch_idx, batch in enumerate(data_loader):
            batch_rng = jax.random.fold_in(rng, batch_idx)
            predictions, metrics = self.evaluate_batch(params, batch, batch_rng)
            
            all_metrics.append(metrics)
            all_predictions.append(predictions)
        
        # Aggregate metrics
        aggregated = {}
        for key in all_metrics[0].keys():
            values = [m[key] for m in all_metrics if key in m]
            aggregated[key] = np.mean(values)
            aggregated[f'{key}_std'] = np.std(values)
        
        # Add to history
        self.metrics_history.append(aggregated)
        
        return aggregated, all_predictions
    
    def generate_vegas_comparison_data(self, params, test_data, rng):
        """Generate predictions formatted for Vegas comparison"""
        predictions_list = []
        game_ids = []
        
        for batch_idx, batch in enumerate(test_data):
            batch_rng = jax.random.fold_in(rng, batch_idx)
            
            # Get predictions
            predictions = self.model.apply(
                params,
                batch,
                training=False,
                rngs={'dropout': batch_rng}
            )
            
            # Extract game-level predictions
            if 'game_ids' in batch:
                game_ids.extend(batch['game_ids'])
                
            batch_preds = {
                'home_score_pred': predictions['home_score'].squeeze(),
                'away_score_pred': predictions['away_score'].squeeze(),
                'spread_pred': predictions['spread'].squeeze() if 'spread' in predictions else None,
                'total_points_pred': predictions['total_points'].squeeze() if 'total_points' in predictions else None
            }
            
            predictions_list.append(batch_preds)
        
        # Combine all predictions
        combined_predictions = {
            'game_id': np.array(game_ids),
            'home_score_pred': np.concatenate([p['home_score_pred'] for p in predictions_list]),
            'away_score_pred': np.concatenate([p['away_score_pred'] for p in predictions_list])
        }
        
        if predictions_list[0]['spread_pred'] is not None:
            combined_predictions['spread_pred'] = np.concatenate([p['spread_pred'] for p in predictions_list])
            
        if predictions_list[0]['total_points_pred'] is not None:
            combined_predictions['total_points_pred'] = np.concatenate([p['total_points_pred'] for p in predictions_list])
        
        return combined_predictions


class CheckpointManager:
    """Manage model checkpoints with Flax utilities"""
    
    def __init__(self, checkpoint_dir: str, max_to_keep: int = 5):
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.max_to_keep = max_to_keep
        self.checkpoint_history = []
        
    def save_checkpoint(self, 
                       step: int,
                       params: Any,
                       optimizer_state: Any,
                       metrics: Dict,
                       phase_name: str = None):
        """Save checkpoint with associated metadata"""
        
        # Create checkpoint data
        checkpoint_data = {
            'params': params,
            'optimizer_state': optimizer_state,
            'step': step,
            'metrics': metrics,
            'phase': phase_name
        }
        
        # Use simple pickle-based saving to avoid JAX monitoring issues
        checkpoint_path = self.checkpoint_dir / f'checkpoint_{step}.pkl'
        with open(checkpoint_path, 'wb') as f:
            pickle.dump(checkpoint_data, f)
        
        # Clean up old checkpoints if needed
        self._cleanup_old_checkpoints()
        
        # Save metadata separately for easy access
        metadata_path = checkpoint_path.with_suffix('.json')
        def convert_to_serializable(obj):
            """Convert JAX arrays and numpy types to JSON-serializable Python types"""
            if hasattr(obj, 'item'):  # JAX/numpy scalar
                return obj.item()
            elif isinstance(obj, (np.ndarray, jnp.ndarray)):
                return float(obj) if obj.size == 1 else obj.tolist()
            elif isinstance(obj, (np.integer, jnp.integer)):
                return int(obj)
            elif isinstance(obj, (np.floating, jnp.floating)):
                return float(obj)
            elif obj is None:
                return None
            else:
                return obj
        
        metadata = {
            'step': int(step),
            'phase': phase_name,
            'metrics': {k: convert_to_serializable(v) for k, v in metrics.items()},
            'timestamp': pd.Timestamp.now().isoformat()
        }
        
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        self.checkpoint_history.append(metadata)
        # Try different RMSE metric names or use validation loss
        rmse_value = metrics.get('val_rmse') or metrics.get('game_rmse') or metrics.get('val_loss')
        if rmse_value is not None and rmse_value != 'N/A':
            rmse_str = f"{rmse_value:.3f}"
        else:
            rmse_str = 'N/A'
        print(f"💾 Checkpoint saved: step {step}, Loss: {rmse_str}")
        
        return checkpoint_path
    
    def _cleanup_old_checkpoints(self):
        """Remove old checkpoint files to stay within max_to_keep limit"""
        checkpoint_files = sorted(
            [f for f in self.checkpoint_dir.glob('checkpoint_*.pkl')],
            key=lambda x: int(x.stem.split('_')[1])  # Sort by step number
        )
        
        if len(checkpoint_files) > self.max_to_keep:
            files_to_remove = checkpoint_files[:-self.max_to_keep]
            for file_path in files_to_remove:
                file_path.unlink()  # Delete the file
                # Also remove corresponding metadata file if exists
                metadata_path = file_path.with_suffix('.json')
                if metadata_path.exists():
                    metadata_path.unlink()
    
    def load_checkpoint(self, step: Optional[int] = None):
        """Load checkpoint by step or latest if step is None"""
        
        if step is None:
            # Find latest checkpoint
            checkpoint_files = sorted(
                [f for f in self.checkpoint_dir.glob('checkpoint_*.pkl')],
                key=lambda x: int(x.stem.split('_')[1])  # Sort by step number
            )
            if not checkpoint_files:
                print("⚠️ No checkpoint found")
                return None
            checkpoint_path = checkpoint_files[-1]  # Latest
        else:
            # Load specific checkpoint
            checkpoint_path = self.checkpoint_dir / f'checkpoint_{step}.pkl'
            if not checkpoint_path.exists():
                print(f"⚠️ Checkpoint for step {step} not found")
                return None
        
        # Load pickle file
        try:
            with open(checkpoint_path, 'rb') as f:
                checkpoint_data = pickle.load(f)
            
            print(f"✅ Loaded checkpoint from step {checkpoint_data.get('step', 'unknown')}")
            return checkpoint_data
        except Exception as e:
            print(f"❌ Failed to load checkpoint: {e}")
            return None
    
    def get_best_checkpoint(self, metric_name: str = 'game_rmse', lower_is_better: bool = True):
        """Find best checkpoint based on metric"""
        
        if not self.checkpoint_history:
            # Load metadata from disk
            for metadata_file in sorted(self.checkpoint_dir.glob('checkpoint_*.json')):
                with open(metadata_file, 'r') as f:
                    self.checkpoint_history.append(json.load(f))
        
        if not self.checkpoint_history:
            return None
        
        # Find best checkpoint
        best_checkpoint = None
        best_value = float('inf') if lower_is_better else float('-inf')
        
        for checkpoint in self.checkpoint_history:
            if metric_name in checkpoint['metrics']:
                value = checkpoint['metrics'][metric_name]
                if (lower_is_better and value < best_value) or \
                   (not lower_is_better and value > best_value):
                    best_value = value
                    best_checkpoint = checkpoint
        
        if best_checkpoint:
            print(f"🏆 Best checkpoint: step {best_checkpoint['step']}, "
                  f"{metric_name}: {best_value:.3f}")
            return self.load_checkpoint(best_checkpoint['step'])
        
        return None


class TestDataLoader:
    """Load and prepare test data for evaluation"""
    
    def __init__(self, config):
        self.config = config
        self.base_path = Path(config.get('base_path', '/content/drive/MyDrive/cfb_model/parquet_files/'))
        
    def load_test_data(self, years: List[int], weeks: Optional[List[int]] = None):
        """Load test data from parquet files"""
        
        test_data = []
        actual_results = []
        
        for year in years:
            year_path = self.base_path / f'year_{year}'
            
            if weeks:
                week_files = [year_path / f'week_{week}.parquet' for week in weeks]
            else:
                week_files = sorted(year_path.glob('week_*.parquet'))
            
            for week_file in week_files:
                if week_file.exists():
                    df = pd.read_parquet(week_file)
                    
                    # Extract game results
                    game_results = df.groupby('game_id').agg({
                        'home_score': 'last',
                        'away_score': 'last',
                        'spread': 'last' if 'spread' in df.columns else 'first',
                        'total_points': 'last' if 'total_points' in df.columns else 'first'
                    }).reset_index()
                    
                    actual_results.append(game_results)
                    
                    # Prepare data for model input
                    # This would need to match your preprocessing pipeline
                    processed = self.preprocess_for_model(df)
                    test_data.append(processed)
        
        if actual_results:
            combined_results = pd.concat(actual_results, ignore_index=True)
            return test_data, combined_results.to_dict('records')
        
        return None, None
    
    def preprocess_for_model(self, df):
        """Preprocess dataframe for model input"""
        # This should match your preprocessing pipeline
        # Simplified version here
        features = {
            'game_ids': df['game_id'].unique(),
            'features': df.select_dtypes(include=[np.number]).values,
            # Add other required features
        }
        return features


def create_evaluation_pipeline(model, config):
    """Create complete evaluation pipeline"""
    evaluator = ModelEvaluator(model, config)
    checkpoint_manager = CheckpointManager(
        checkpoint_dir=config.get('checkpoint_dir', '/content/drive/MyDrive/cfb_model/checkpoints/')
    )
    test_loader = TestDataLoader(config)
    
    return {
        'evaluator': evaluator,
        'checkpoint_manager': checkpoint_manager,
        'test_loader': test_loader
    }