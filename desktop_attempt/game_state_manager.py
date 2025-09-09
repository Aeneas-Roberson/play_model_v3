# game_state_management.py
"""
Game State Management System for CFB Hierarchical Model
Handles real-time state updates, consistency resolution, and parallel simulation
Optimized for JAX/Flax TPU v2-8 with 512GB HBM
"""

import jax
import jax.numpy as jnp
import numpy as np
from typing import Dict, List, Tuple, Optional, Any, NamedTuple
from dataclasses import dataclass, field
from collections import deque
import copy
import logging
from enum import Enum
import pickle
from pathlib import Path

# Configure JAX for TPU
jax.config.update('jax_platform_name', 'tpu')


class PlayOutcome(Enum):
    """Enumeration of possible play outcomes"""
    RUSHING = "rushing"
    PASSING = "passing"
    TOUCHDOWN = "touchdown"
    FIELD_GOAL = "field_goal"
    PUNT = "punt"
    TURNOVER = "turnover"
    PENALTY = "penalty"
    TIMEOUT = "timeout"
    ADMINISTRATIVE = "administrative"


@dataclass
class CoreGameState:
    """
    Core game state variables for real-time simulation
    """
    # SCORING & POSSESSION
    offense_score: int = 0
    defense_score: int = 0
    score_differential: int = 0
    possessing_team: str = "offense"
    
    # DOWN & DISTANCE CONTEXT
    down: int = 1
    distance: int = 10
    yards_to_goal: int = 80
    field_position: int = 20
    
    # TEMPORAL CONTEXT
    period: int = 1
    total_seconds_remaining: int = 3600
    period_seconds_remaining: int = 900
    
    # POSSESSION CONTEXT
    offense_timeouts: int = 3
    defense_timeouts: int = 3
    possession_start_field_position: int = 20
    plays_this_drive: int = 0
    yards_this_drive: int = 0
    
    # SITUATIONAL FLAGS
    is_red_zone: bool = False
    is_goal_line: bool = False
    is_two_minute_warning: bool = False
    is_garbage_time: bool = False
    is_money_down: bool = False
    is_plus_territory: bool = False
    
    def __post_init__(self):
        """Update derived situational flags after initialization"""
        self.update_situational_flags()
    
    def update_situational_flags(self):
        """Update situational flags based on core state"""
        self.is_red_zone = self.yards_to_goal <= 20
        self.is_goal_line = self.yards_to_goal <= 5
        self.is_two_minute_warning = (
            (self.period in [2, 4] and self.period_seconds_remaining <= 120) or 
            (self.period >= 5)
        )
        self.is_money_down = self.down >= 3
        self.is_plus_territory = self.field_position >= 50
        
        # Garbage time: >21 point lead in 4th quarter OR >28 points any time
        if self.period >= 4:
            self.is_garbage_time = abs(self.score_differential) > 21
        else:
            self.is_garbage_time = abs(self.score_differential) > 28
    
    def to_jax_array(self) -> jnp.ndarray:
        """Convert state to JAX array for model input"""
        state_vector = jnp.array([
            self.offense_score,
            self.defense_score,
            self.score_differential,
            self.down,
            self.distance,
            self.yards_to_goal,
            self.field_position,
            self.period,
            self.total_seconds_remaining,
            self.period_seconds_remaining,
            self.offense_timeouts,
            self.defense_timeouts,
            self.plays_this_drive,
            self.yards_this_drive,
            float(self.is_red_zone),
            float(self.is_goal_line),
            float(self.is_two_minute_warning),
            float(self.is_garbage_time),
            float(self.is_money_down),
            float(self.possession_start_field_position)
        ], dtype=jnp.float32)
        
        return state_vector


class GameStateManager:
    """
    Advanced game state management with cached recent states
    """
    
    def __init__(self, 
                 cache_size: int = 10,
                 tpu_optimized: bool = True,
                 batch_size: int = 2048):
        
        self.cache_size = cache_size
        self.tpu_optimized = tpu_optimized
        self.batch_size = batch_size
        
        # STATE STORAGE
        self.current_state = CoreGameState()
        self.state_cache = deque(maxlen=cache_size)
        self.state_history = []
        
        # CONSISTENCY TRACKING
        self.pending_updates = []
        self.consistency_violations = []
        
        # Setup logging
        self.logger = logging.getLogger('GameStateManager')
        logging.basicConfig(level=logging.INFO)
        
        # TPU OPTIMIZATION
        if self.tpu_optimized:
            self._initialize_tpu_tensors()
    
    def _initialize_tpu_tensors(self):
        """Initialize TPU-optimized tensor storage for states"""
        # Pre-allocate tensors for efficient TPU operations
        self.state_tensor_cache = jnp.zeros(
            (self.batch_size, self.cache_size, 20), dtype=jnp.float32
        )
        self.state_mask = jnp.zeros(
            (self.batch_size, self.cache_size), dtype=jnp.float32
        )
    
    def update_state_from_play_prediction(self, 
                                        play_prediction: Dict, 
                                        actual_outcome: Optional[Dict] = None) -> CoreGameState:
        """
        Real-time state update from individual play prediction
        """
        # Store previous state in cache
        previous_state = copy.deepcopy(self.current_state)
        self.state_cache.append(previous_state)
        
        # Extract play outcome
        play_outcome = actual_outcome if actual_outcome else play_prediction
        
        # Update core state variables
        new_state = copy.deepcopy(self.current_state)
        
        # 1. YARDAGE AND FIELD POSITION
        yards_gained = play_outcome.get('yards_gained', 0)
        new_state.field_position = max(0, min(100, 
            new_state.field_position + yards_gained))
        new_state.yards_to_goal = max(0, new_state.yards_to_goal - yards_gained)
        new_state.yards_this_drive += yards_gained
        new_state.plays_this_drive += 1
        
        # 2. DOWN AND DISTANCE LOGIC
        if yards_gained >= new_state.distance:
            # First down achieved
            new_state.down = 1
            new_state.distance = 10
        else:
            # Advance down, reduce distance
            new_state.down += 1
            new_state.distance = max(0, new_state.distance - yards_gained)
        
        # 3. SCORING UPDATES
        if play_outcome.get('is_touchdown', False):
            if new_state.possessing_team == "offense":
                new_state.offense_score += 7
            else:
                new_state.defense_score += 7
            self._handle_scoring_possession_change(new_state)
            
        elif play_outcome.get('is_field_goal', False) and play_outcome.get('is_good', False):
            if new_state.possessing_team == "offense":
                new_state.offense_score += 3
            self._handle_scoring_possession_change(new_state)
            
        elif play_outcome.get('is_safety', False):
            if new_state.possessing_team == "offense":
                new_state.defense_score += 2
            else:
                new_state.offense_score += 2
            self._handle_safety_possession_change(new_state)
        
        # 4. POSSESSION CHANGES
        if (play_outcome.get('is_interception', False) or 
            play_outcome.get('is_fumble_lost', False)):
            self._handle_turnover(new_state)
            
        elif (play_outcome.get('is_punt', False) or 
              new_state.down > 4 or
              (new_state.down == 4 and yards_gained < new_state.distance)):
            self._handle_possession_change(new_state)
        
        # 5. TIME MANAGEMENT
        seconds_elapsed = self._estimate_play_duration(play_outcome)
        new_state.total_seconds_remaining = max(0, 
            new_state.total_seconds_remaining - seconds_elapsed)
        new_state.period_seconds_remaining = max(0,
            new_state.period_seconds_remaining - seconds_elapsed)
        
        # Handle quarter transitions
        if new_state.period_seconds_remaining <= 0 and new_state.period < 4:
            new_state.period += 1
            new_state.period_seconds_remaining = 900
        
        # 6. UPDATE DERIVED FLAGS
        new_state.score_differential = new_state.offense_score - new_state.defense_score
        new_state.update_situational_flags()
        
        # 7. CONSISTENCY VALIDATION
        self._validate_state_consistency(previous_state, new_state, play_outcome)
        
        # Update current state
        self.current_state = new_state
        self.state_history.append(new_state)
        
        return new_state
    
    def _handle_scoring_possession_change(self, state: CoreGameState):
        """Handle possession change after scoring"""
        state.possessing_team = "defense" if state.possessing_team == "offense" else "offense"
        state.field_position = 25
        state.yards_to_goal = 75
        state.down = 1
        state.distance = 10
        state.plays_this_drive = 0
        state.yards_this_drive = 0
        state.possession_start_field_position = 25
    
    def _handle_turnover(self, state: CoreGameState):
        """Handle turnover possession change"""
        state.possessing_team = "defense" if state.possessing_team == "offense" else "offense"
        state.field_position = 100 - state.field_position
        state.yards_to_goal = state.field_position
        state.down = 1
        state.distance = 10
        state.plays_this_drive = 0
        state.yards_this_drive = 0
        state.possession_start_field_position = state.field_position
    
    def _handle_possession_change(self, state: CoreGameState):
        """Handle normal possession change (punt, turnover on downs)"""
        state.possessing_team = "defense" if state.possessing_team == "offense" else "offense"
        
        if state.yards_to_goal > 40:
            state.field_position = 35
        else:
            state.field_position = max(20, 100 - state.yards_to_goal)
        
        state.yards_to_goal = 100 - state.field_position
        state.down = 1
        state.distance = 10
        state.plays_this_drive = 0
        state.yards_this_drive = 0
        state.possession_start_field_position = state.field_position
    
    def _handle_safety_possession_change(self, state: CoreGameState):
        """Handle possession change after safety"""
        state.field_position = 20
        state.yards_to_goal = 80
        state.down = 1
        state.distance = 10
        state.plays_this_drive = 0
        state.yards_this_drive = 0
        state.possession_start_field_position = 20
    
    def _estimate_play_duration(self, play_outcome: Dict) -> int:
        """Estimate play duration in seconds"""
        if play_outcome.get('is_timeout', False):
            return 0
        elif play_outcome.get('is_pass', False):
            if play_outcome.get('is_completion', False):
                return 6
            else:
                return 3
        elif play_outcome.get('is_rush', False):
            return 5
        elif play_outcome.get('is_punt', False):
            return 15
        elif play_outcome.get('is_field_goal', False):
            return 8
        else:
            return 5
    
    def _validate_state_consistency(self, 
                                  previous_state: CoreGameState, 
                                  new_state: CoreGameState, 
                                  play_outcome: Dict):
        """Validate state transition consistency"""
        violations = []
        
        # Check impossible state transitions
        if new_state.field_position < 0 or new_state.field_position > 100:
            violations.append(f"Invalid field position: {new_state.field_position}")
        
        if new_state.down < 1 or new_state.down > 4:
            violations.append(f"Invalid down: {new_state.down}")
        
        if new_state.distance < 0:
            violations.append(f"Invalid distance: {new_state.distance}")
        
        # Check score consistency
        score_change = (new_state.offense_score - previous_state.offense_score) + \
                      (new_state.defense_score - previous_state.defense_score)
        
        if score_change > 8 or score_change < 0:
            violations.append(f"Impossible score change: {score_change}")
        
        if violations:
            self.consistency_violations.extend(violations)
            self.logger.warning(f"⚠️ State consistency violations: {violations}")
    
    def get_state_context_for_model(self, include_history: bool = True) -> Dict:
        """Generate model input features from current game state"""
        
        context = {
            'offense_score': float(self.current_state.offense_score),
            'defense_score': float(self.current_state.defense_score),
            'score_differential': float(self.current_state.score_differential),
            'down': float(self.current_state.down),
            'distance': float(self.current_state.distance),
            'yards_to_goal': float(self.current_state.yards_to_goal),
            'field_position': float(self.current_state.field_position),
            'period': float(self.current_state.period),
            'total_seconds_remaining': float(self.current_state.total_seconds_remaining),
            'period_seconds_remaining': float(self.current_state.period_seconds_remaining),
            'offense_timeouts': float(self.current_state.offense_timeouts),
            'defense_timeouts': float(self.current_state.defense_timeouts),
            'plays_this_drive': float(self.current_state.plays_this_drive),
            'yards_this_drive': float(self.current_state.yards_this_drive),
            'is_red_zone': float(self.current_state.is_red_zone),
            'is_goal_line': float(self.current_state.is_goal_line),
            'is_two_minute_warning': float(self.current_state.is_two_minute_warning),
            'is_garbage_time': float(self.current_state.is_garbage_time),
            'is_money_down': float(self.current_state.is_money_down),
            'is_offense_home_team': 1.0,
            'possession_start_field_position': float(self.current_state.possession_start_field_position)
        }
        
        # ADD RECENT STATE HISTORY
        if include_history and self.state_cache:
            recent_scores = [state.score_differential for state in self.state_cache]
            recent_field_positions = [state.field_position for state in self.state_cache]
            
            # Pad to cache_size if needed
            while len(recent_scores) < self.cache_size:
                recent_scores.insert(0, 0.0)
                recent_field_positions.insert(0, 50.0)
            
            context.update({
                'recent_score_differentials': recent_scores[-self.cache_size:],
                'recent_field_positions': recent_field_positions[-self.cache_size:],
                'context_length': float(len(self.state_cache))
            })
        
        return context
    
    def reset_for_new_game(self, initial_conditions: Optional[Dict] = None):
        """Reset state manager for new game simulation"""
        self.current_state = CoreGameState()
        self.state_cache.clear()
        self.state_history.clear()
        self.pending_updates.clear()
        self.consistency_violations.clear()
        
        # Apply any initial conditions
        if initial_conditions:
            for key, value in initial_conditions.items():
                if hasattr(self.current_state, key):
                    setattr(self.current_state, key, value)
            self.current_state.update_situational_flags()


class HierarchicalConsistencyManager:
    """
    Manages consistency between play → drive → game level predictions
    """
    
    def __init__(self, hierarchical_model=None):
        self.model = hierarchical_model
        self.resolution_history = []
        self.logger = logging.getLogger('ConsistencyManager')
        
    def resolve_prediction_conflicts(self, 
                                   play_prediction: Dict,
                                   drive_prediction: Dict,
                                   game_prediction: Dict,
                                   current_state: CoreGameState) -> Dict:
        """
        Bottom-up consistency resolution with conflict explanation
        """
        resolution_log = {
            'conflicts_detected': [],
            'resolutions_applied': [],
            'final_predictions': {}
        }
        
        # STEP 1: ACCEPT PLAY-LEVEL PREDICTION (bottom-up priority)
        accepted_play = play_prediction.copy()
        resolution_log['final_predictions']['play'] = accepted_play
        
        # STEP 2: UPDATE DRIVE CONTEXT WITH PLAY OUTCOME
        updated_drive_context = self._update_drive_context_from_play(
            drive_prediction, accepted_play, current_state
        )
        
        # STEP 3: DETECT DRIVE-LEVEL CONFLICTS
        drive_conflicts = self._detect_drive_conflicts(
            drive_prediction, accepted_play, current_state
        )
        
        if drive_conflicts:
            resolution_log['conflicts_detected'].extend(drive_conflicts)
            
            # RE-PREDICT DRIVE OUTCOME WITH UPDATED CONTEXT
            if self.model:
                corrected_drive = self.model.drive_model.predict(updated_drive_context)
            else:
                corrected_drive = drive_prediction
            
            resolution_log['resolutions_applied'].append(
                f"Drive re-prediction: {drive_conflicts} → {corrected_drive.get('outcome', 'N/A')}"
            )
            resolution_log['final_predictions']['drive'] = corrected_drive
        else:
            resolution_log['final_predictions']['drive'] = drive_prediction
        
        # STEP 4: UPDATE GAME CONTEXT WITH DRIVE OUTCOME
        final_drive = resolution_log['final_predictions']['drive']
        updated_game_context = self._update_game_context_from_drive(
            game_prediction, final_drive, current_state
        )
        
        # STEP 5: DETECT GAME-LEVEL CONFLICTS
        game_conflicts = self._detect_game_conflicts(
            game_prediction, final_drive, current_state
        )
        
        if game_conflicts:
            resolution_log['conflicts_detected'].extend(game_conflicts)
            
            # RE-PREDICT GAME STATISTICS WITH UPDATED CONTEXT
            if self.model:
                corrected_game = self.model.game_model.predict(updated_game_context)
            else:
                corrected_game = game_prediction
            
            resolution_log['resolutions_applied'].append(
                f"Game re-prediction: {game_conflicts} → Updated game stats"
            )
            resolution_log['final_predictions']['game'] = corrected_game
        else:
            resolution_log['final_predictions']['game'] = game_prediction
        
        # STORE RESOLUTION HISTORY
        self.resolution_history.append(resolution_log)
        
        return resolution_log
    
    def _detect_drive_conflicts(self, drive_pred: Dict, play_pred: Dict, state: CoreGameState) -> List[str]:
        """Detect conflicts between play and drive predictions"""
        conflicts = []
        
        # EXAMPLE CONFLICT: Play predicts TD, but drive predicts punt
        if (play_pred.get('is_touchdown', False) and 
            drive_pred.get('outcome') == 'Punt'):
            conflicts.append("Play TD vs Drive Punt")
        
        # EXAMPLE CONFLICT: Play gains 15 yards, drive says 0 total yards
        play_yards = play_pred.get('yards_gained', 0)
        drive_remaining_yards = drive_pred.get('total_yards', 0) - state.yards_this_drive
        
        if play_yards > drive_remaining_yards + 5:
            conflicts.append(f"Play yards ({play_yards}) exceed drive remaining ({drive_remaining_yards})")
        
        return conflicts
    
    def _detect_game_conflicts(self, game_pred: Dict, drive_pred: Dict, state: CoreGameState) -> List[str]:
        """Detect conflicts between drive and game predictions"""
        conflicts = []
        
        # EXAMPLE CONFLICT: Drive scores TD, but game total points don't increase
        if (drive_pred.get('outcome') == 'TD' and
            game_pred.get('offense_points', 0) <= state.offense_score):
            conflicts.append("Drive TD vs Game points unchanged")
        
        # EXAMPLE CONFLICT: Drive efficiency vs game averages
        drive_ypp = drive_pred.get('yards_per_play', 0)
        game_ypp = game_pred.get('offensive_yards_per_play', 0)
        
        if abs(drive_ypp - game_ypp) > 3.0:
            conflicts.append(f"Drive YPP ({drive_ypp:.1f}) vs Game YPP ({game_ypp:.1f}) divergence")
        
        return conflicts
    
    def _update_drive_context_from_play(self, drive_pred: Dict, play_pred: Dict, state: CoreGameState) -> Dict:
        """Update drive context features based on accepted play prediction"""
        updated_context = drive_pred.copy()
        
        updated_context['current_yards_this_drive'] = state.yards_this_drive + play_pred.get('yards_gained', 0)
        updated_context['current_plays_this_drive'] = state.plays_this_drive + 1
        updated_context['current_down'] = state.down
        updated_context['current_distance'] = state.distance
        updated_context['current_field_position'] = state.field_position
        
        return updated_context
    
    def _update_game_context_from_drive(self, game_pred: Dict, drive_pred: Dict, state: CoreGameState) -> Dict:
        """Update game context features based on accepted drive prediction"""
        updated_context = game_pred.copy()
        
        if drive_pred.get('outcome') in ['TD', 'FG']:
            updated_context['projected_offensive_scores'] = updated_context.get('projected_offensive_scores', 0) + 1
        
        updated_context['total_offensive_yards'] = updated_context.get('total_offensive_yards', 0) + drive_pred.get('total_yards', 0)
        updated_context['total_offensive_plays'] = updated_context.get('total_offensive_plays', 0) + drive_pred.get('play_count', 0)
        
        return updated_context


class ParallelGameSimulator:
    """
    TPU-optimized parallel simulation of multiple games simultaneously
    """
    
    def __init__(self, 
                 hierarchical_model=None,
                 max_parallel_games: int = 128,
                 tpu_optimized: bool = True):
        
        self.model = hierarchical_model
        self.max_parallel_games = max_parallel_games
        self.tpu_optimized = tpu_optimized
        
        # PARALLEL STATE MANAGEMENT
        self.game_states = {}
        self.active_simulations = set()
        
        # SIMULATION TRACKING
        self.simulation_results = {}
        self.performance_metrics = {
            'games_simulated': 0,
            'total_plays_simulated': 0,
            'avg_simulation_time': 0.0,
            'consistency_violations': 0
        }
        
        self.logger = logging.getLogger('ParallelSimulator')
    
    def simulate_multiple_games(self, 
                               game_setups: List[Dict],
                               max_plays_per_game: int = 200) -> Dict[str, Any]:
        """
        Simulate multiple games in parallel batches
        """
        self.logger.info(f"🎮 Starting parallel simulation of {len(game_setups)} games...")
        
        # INITIALIZE GAME STATES
        for i, setup in enumerate(game_setups):
            game_id = setup.get('game_id', f'sim_game_{i}')
            self.game_states[game_id] = GameStateManager(
                tpu_optimized=self.tpu_optimized,
                batch_size=1
            )
            
            # Apply initial conditions
            self.game_states[game_id].reset_for_new_game(setup.get('initial_conditions', {}))
            self.active_simulations.add(game_id)
        
        # PARALLEL SIMULATION LOOP
        simulation_step = 0
        max_simulation_steps = max_plays_per_game
        
        while self.active_simulations and simulation_step < max_simulation_steps:
            # BATCH PROCESS ACTIVE GAMES
            batch_results = self._simulate_parallel_batch()
            
            # UPDATE STATES AND CHECK COMPLETION
            completed_games = self._process_batch_results(batch_results)
            
            # Remove completed games
            for game_id in completed_games:
                self.active_simulations.discard(game_id)
                self.logger.info(f"✅ Game {game_id} completed")
            
            simulation_step += 1
            
            if simulation_step % 20 == 0:
                self.logger.info(f"🔄 Step {simulation_step}, {len(self.active_simulations)} games active")
        
        # FINALIZE RESULTS
        final_results = self._finalize_simulation_results()
        
        self.logger.info(f"🏁 Simulation complete: {len(final_results['individual_games'])} games")
        return final_results
    
    def _simulate_parallel_batch(self) -> Dict[str, Any]:
        """Simulate one play for all active games in parallel"""
        batch_inputs = {}
        batch_game_ids = list(self.active_simulations)
        
        # PREPARE BATCH INPUTS
        for game_id in batch_game_ids:
            state_manager = self.game_states[game_id]
            game_context = state_manager.get_state_context_for_model()
            
            batch_inputs[game_id] = {
                'current_state': game_context,
                'game_id': game_id
            }
        
        # BATCH PREDICTION (TPU-optimized)
        if self.tpu_optimized and len(batch_game_ids) > 1:
            batch_predictions = self._tpu_batch_predict(batch_inputs)
        else:
            batch_predictions = self._sequential_predict(batch_inputs)
        
        return batch_predictions
    
    @jax.jit
    def _tpu_batch_predict(self, batch_inputs: Dict) -> Dict:
        """TPU-optimized batch prediction for multiple games"""
        game_ids = list(batch_inputs.keys())
        
        # Stack all game contexts into batch tensors
        batch_features = []
        for game_id in game_ids:
            context = batch_inputs[game_id]['current_state']
            feature_vector = self._context_to_feature_vector(context)
            batch_features.append(feature_vector)
        
        # Batch prediction
        batch_tensor = jnp.stack(batch_features)
        
        # Simulate predictions (placeholder - would use actual model)
        batch_predictions_tensor = {
            'play': jnp.ones((len(game_ids), 10)),
            'drive': jnp.ones((len(game_ids), 5)),
            'game': jnp.ones((len(game_ids), 3))
        }
        
        # Convert back to per-game predictions
        batch_results = {}
        for i, game_id in enumerate(game_ids):
            batch_results[game_id] = {
                'play_prediction': batch_predictions_tensor['play'][i],
                'drive_prediction': batch_predictions_tensor['drive'][i] if 'drive' in batch_predictions_tensor else None,
                'game_prediction': batch_predictions_tensor['game'][i] if 'game' in batch_predictions_tensor else None
            }
        
        return batch_results
    
    def _sequential_predict(self, batch_inputs: Dict) -> Dict:
        """Sequential prediction for smaller batches"""
        batch_results = {}
        
        for game_id, inputs in batch_inputs.items():
            # Simulate prediction (placeholder)
            prediction = {
                'play': {'yards_gained': np.random.normal(5, 3), 'is_touchdown': False},
                'drive': {'outcome': 'Continue', 'total_yards': 50},
                'game': {'offense_points': 21, 'defense_points': 14}
            }
            
            batch_results[game_id] = {
                'play_prediction': prediction.get('play', {}),
                'drive_prediction': prediction.get('drive', {}),
                'game_prediction': prediction.get('game', {})
            }
        
        return batch_results
    
    def _process_batch_results(self, batch_results: Dict) -> List[str]:
        """Process batch prediction results and update game states"""
        completed_games = []
        
        for game_id, predictions in batch_results.items():
            if game_id not in self.game_states:
                continue
            
            state_manager = self.game_states[game_id]
            
            # Update game state with play prediction
            new_state = state_manager.update_state_from_play_prediction(
                predictions['play_prediction']
            )
            
            # Check if game is completed
            if self._is_game_completed(new_state):
                completed_games.append(game_id)
                
                # Store final game results
                self.simulation_results[game_id] = {
                    'final_state': new_state,
                    'play_history': state_manager.state_history,
                    'total_plays': len(state_manager.state_history),
                    'final_score': f"{new_state.offense_score}-{new_state.defense_score}",
                    'consistency_violations': len(state_manager.consistency_violations)
                }
        
        return completed_games
    
    def _is_game_completed(self, state: CoreGameState) -> bool:
        """Check if game simulation should end"""
        # Game ends when time expires
        if state.period >= 4 and state.total_seconds_remaining <= 0:
            # Check for tie - would go to overtime
            if state.score_differential == 0 and state.period == 4:
                return False
            else:
                return True
        
        # Game ends in overtime when someone scores
        if state.period >= 5 and state.score_differential != 0:
            return True
        
        # Safety: End if too many periods
        if state.period > 10:
            return True
        
        return False
    
    def _finalize_simulation_results(self) -> Dict[str, Any]:
        """Finalize and return comprehensive simulation results"""
        final_results = {
            'individual_games': self.simulation_results,
            'aggregate_statistics': self._calculate_aggregate_statistics(),
            'performance_metrics': self.performance_metrics,
            'consistency_report': self._generate_consistency_report()
        }
        
        return final_results
    
    def _calculate_aggregate_statistics(self) -> Dict[str, float]:
        """Calculate aggregate statistics across all simulated games"""
        if not self.simulation_results:
            return {}
        
        all_games = list(self.simulation_results.values())
        
        return {
            'avg_total_plays': np.mean([game['total_plays'] for game in all_games]),
            'avg_final_score_differential': np.mean([
                abs(game['final_state'].score_differential) for game in all_games
            ]),
            'overtime_rate': len([
                game for game in all_games if game['final_state'].period > 4
            ]) / len(all_games),
            'avg_consistency_violations': np.mean([
                game['consistency_violations'] for game in all_games
            ])
        }
    
    def _generate_consistency_report(self) -> Dict[str, Any]:
        """Generate report on consistency violations across simulations"""
        all_violations = []
        for game_id, game in self.simulation_results.items():
            if game_id in self.game_states:
                all_violations.extend(self.game_states[game_id].consistency_violations)
        
        return {
            'total_violations': len(all_violations),
            'violation_rate': len(all_violations) / max(1, sum(
                game['total_plays'] for game in self.simulation_results.values()
            )),
            'common_violations': self._count_violation_types(all_violations)
        }
    
    def _count_violation_types(self, violations: List[str]) -> Dict[str, int]:
        """Count types of consistency violations"""
        violation_counts = {}
        for violation in violations:
            violation_type = violation.split(':')[0] if ':' in violation else violation
            violation_counts[violation_type] = violation_counts.get(violation_type, 0) + 1
        
        return dict(sorted(violation_counts.items(), key=lambda x: x[1], reverse=True))
    
    def _context_to_feature_vector(self, context: Dict) -> jnp.ndarray:
        """Convert game context dictionary to tensor feature vector"""
        feature_keys = [
            'offense_score', 'defense_score', 'score_differential',
            'down', 'distance', 'yards_to_goal', 'field_position',
            'period', 'total_seconds_remaining', 'period_seconds_remaining',
            'offense_timeouts', 'defense_timeouts', 'plays_this_drive',
            'yards_this_drive', 'is_red_zone', 'is_goal_line',
            'is_two_minute_warning', 'is_garbage_time', 'is_money_down',
            'possession_start_field_position'
        ]
        
        feature_vector = [context.get(key, 0.0) for key in feature_keys]
        return jnp.array(feature_vector, dtype=jnp.float32)


# TPU-Optimized State Operations
@jax.jit
def tpu_batch_state_update(state_tensor: jnp.ndarray, 
                          play_outcomes_tensor: jnp.ndarray,
                          mask_tensor: jnp.ndarray) -> jnp.ndarray:
    """
    TPU-compiled batch state update operation
    """
    # EXTRACT STATE COMPONENTS
    scores = state_tensor[:, 0:2]
    field_pos = state_tensor[:, 6:7]
    down_distance = state_tensor[:, 3:5]
    
    # EXTRACT PLAY OUTCOMES
    yards_gained = play_outcomes_tensor[:, 0:1]
    is_touchdown = play_outcomes_tensor[:, 1:2]
    is_first_down = play_outcomes_tensor[:, 2:3]
    
    # UPDATE FIELD POSITION
    new_field_pos = jnp.clip(field_pos + yards_gained, 0.0, 100.0)
    
    # UPDATE SCORES
    touchdown_points = is_touchdown * 7.0
    new_offense_score = scores[:, 0:1] + touchdown_points
    new_defense_score = scores[:, 1:2]
    
    # UPDATE DOWN/DISTANCE
    new_down = jnp.where(
        is_first_down > 0.5,
        jnp.ones_like(down_distance[:, 0:1]),
        jnp.clip(down_distance[:, 0:1] + 1.0, 1.0, 4.0)
    )
    
    new_distance = jnp.where(
        is_first_down > 0.5,
        jnp.ones_like(down_distance[:, 1:2]) * 10.0,
        jnp.maximum(down_distance[:, 1:2] - yards_gained, 0.0)
    )
    
    # RECONSTRUCT STATE TENSOR
    updated_state = jnp.concatenate([
        new_offense_score,
        new_defense_score,
        new_offense_score - new_defense_score,
        new_down,
        new_distance,
        state_tensor[:, 5:6],
        new_field_pos,
        state_tensor[:, 7:]
    ], axis=1)
    
    # APPLY MASK
    mask_expanded = jnp.expand_dims(mask_tensor, 1)
    final_state = jnp.where(
        mask_expanded > 0.5,
        updated_state,
        state_tensor
    )
    
    return final_state


class TPUStateOptimizer:
    """
    TPU-specific optimizations for game state management
    """
    
    def __init__(self, max_batch_size: int = 2048):
        self.max_batch_size = max_batch_size
        self.compiled_functions = {}
        
        # Pre-compile frequently used functions
        self._precompile_functions()
    
    def _precompile_functions(self):
        """Pre-compile TPU functions for better performance"""
        print("⚡ Pre-compiling TPU state management functions...")
        
        # Compile batch state update
        dummy_state = jnp.zeros((32, 20), dtype=jnp.float32)
        dummy_outcomes = jnp.zeros((32, 10), dtype=jnp.float32)
        dummy_mask = jnp.ones((32,), dtype=jnp.float32)
        
        # Warm up compilation
        _ = tpu_batch_state_update(dummy_state, dummy_outcomes, dummy_mask)
        
        self.compiled_functions['batch_state_update'] = tpu_batch_state_update
        
        print("✅ TPU functions compiled and ready")
    
    def optimize_state_tensor_layout(self, state_data: Dict) -> jnp.ndarray:
        """Optimize tensor layout for TPU performance"""
        optimized_layout = [
            'offense_score', 'defense_score', 'score_differential',
            'down', 'distance', 'yards_to_goal', 'field_position',
            'period', 'total_seconds_remaining',
            'offense_timeouts', 'defense_timeouts',
            'is_red_zone', 'is_goal_line', 'is_two_minute_warning',
            'plays_this_drive', 'yards_this_drive',
            'is_garbage_time', 'is_money_down'
        ]
        
        feature_tensor = jnp.stack([
            jnp.array([state_data.get(key, 0.0) for key in optimized_layout], dtype=jnp.float32)
        ])
        
        return feature_tensor
    
    def estimate_memory_usage(self, 
                            num_games: int, 
                            cache_size: int = 10, 
                            sequence_length: int = 18) -> Dict[str, str]:
        """Estimate TPU memory usage for game state management"""
        state_features = 20
        state_memory = num_games * state_features * 4
        cache_memory = num_games * cache_size * state_features * 4
        sequence_memory = num_games * sequence_length * state_features * 4
        buffer_memory = max(num_games, self.max_batch_size) * state_features * 4 * 3
        
        total_memory = state_memory + cache_memory + sequence_memory + buffer_memory
        
        return {
            'state_storage': f"{state_memory / (1024**2):.1f} MB",
            'cache_storage': f"{cache_memory / (1024**2):.1f} MB", 
            'sequence_integration': f"{sequence_memory / (1024**2):.1f} MB",
            'batch_buffers': f"{buffer_memory / (1024**2):.1f} MB",
            'total_estimated': f"{total_memory / (1024**2):.1f} MB",
            'tpu_v2_8_utilization': f"{(total_memory / (512 * 1024**3)) * 100:.2f}%"
        }


# Utility functions
def create_game_state_manager(cache_size: int = 10, 
                             tpu_optimized: bool = True) -> GameStateManager:
    """Factory function to create game state manager"""
    return GameStateManager(cache_size=cache_size, tpu_optimized=tpu_optimized)

def create_parallel_simulator(max_games: int = 128,
                             tpu_optimized: bool = True) -> ParallelGameSimulator:
    """Factory function to create parallel simulator"""
    return ParallelGameSimulator(max_parallel_games=max_games, tpu_optimized=tpu_optimized)

def create_consistency_manager(model=None) -> HierarchicalConsistencyManager:
    """Factory function to create consistency manager"""
    return HierarchicalConsistencyManager(hierarchical_model=model)