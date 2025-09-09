# cfb_vegas_comparison.py
"""
Vegas Spread Comparison & Evaluation System for CFB Hierarchical Model
Focuses exclusively on spread performance - the key metric for beating Vegas
Target: Beat Vegas' 14+ RMSE with <13 RMSE
"""

import pandas as pd
import numpy as np
import jax.numpy as jnp
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any, Union
import logging
from dataclasses import dataclass
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import os
import sys
import gc

# Handle both Colab and local environments
if 'google.colab' in sys.modules:
    BASE_PATH = "/content/drive/MyDrive/cfb_model/"
else:
    BASE_PATH = os.path.expanduser("~/cfb_model/")

@dataclass
class VegasConfig:
    """Configuration for Vegas comparison"""
    base_path: str = "/content/drive/MyDrive/cfb_model/parquet_files/"
    lines_folder: str = "lines"
    target_rmse: float = 13.0  # Model target
    vegas_rmse_baseline: float = 14.2  # Vegas typical RMSE
    confidence_threshold: float = 0.6  # For high-confidence bets
    kelly_fraction: float = 0.25  # Conservative Kelly criterion
    
class VegasSpreadEvaluator:
    """
    Evaluates model predictions against Vegas spreads
    Focuses on spread accuracy - the only metric that matters for profitability
    """
    
    def __init__(self, config: VegasConfig = None):
        self.config = config or VegasConfig()
        self.base_path = Path(self.config.base_path)
        self.lines_path = self.base_path / self.config.lines_folder
        
        # Performance tracking
        self.evaluation_results = {}
        self.spread_accuracy_history = []
        
        # Setup logging
        self.logger = logging.getLogger('VegasEvaluator')
        
        # Configure logging for Colab
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.StreamHandler(),
                logging.FileHandler(os.path.join(BASE_PATH, 'logs/model.log'))
            ] if os.path.exists(os.path.join(BASE_PATH, 'logs')) else [logging.StreamHandler()]
        )
    
    def load_vegas_spreads(self, 
                          years: List[int],
                          weeks: Optional[List[int]] = None) -> pd.DataFrame:
        """
        Load Vegas spread data for specified years/weeks
        
        Returns DataFrame with game_id, season, week, spread (home team perspective)
        """
        self.logger.info(f"📊 Loading Vegas spreads for years: {years}")
        
        all_spreads = []
        
        for year in years:
            year_weeks = weeks if weeks else range(1, 18)
            
            for week in year_weeks:
                spread_file = self.lines_path / f"{year}/week_{week}.parquet"
                
                if spread_file.exists():
                    try:
                        week_df = pd.read_parquet(spread_file)
                        # Only keep relevant columns
                        week_df = week_df[['game_id', 'season', 'week', 'spread']].copy()
                        all_spreads.append(week_df)
                    except Exception as e:
                        self.logger.warning(f"Error loading {spread_file}: {e}")
        
        if all_spreads:
            spreads_df = pd.concat(all_spreads, ignore_index=True)
            self.logger.info(f"✅ Loaded {len(spreads_df)} games with Vegas spreads")
            return spreads_df
        else:
            self.logger.warning("⚠️ No Vegas spread data found")
            return pd.DataFrame()
    
    def evaluate_spread_predictions(self,
                                   model_predictions: Dict[str, np.ndarray],
                                   actual_results: Dict[str, np.ndarray],
                                   vegas_spreads: pd.DataFrame) -> Dict[str, Any]:
        """
        Core evaluation function comparing model vs Vegas spread predictions
        
        Args:
            model_predictions: Dict with 'game_id', 'home_score_pred', 'away_score_pred'
            actual_results: Dict with 'game_id', 'home_score', 'away_score'
            vegas_spreads: DataFrame with Vegas spread lines
            
        Returns:
            Comprehensive evaluation metrics
        """
        self.logger.info("🎯 Evaluating model spread predictions vs Vegas...")
        
        # Create unified DataFrame for analysis
        eval_df = self._create_evaluation_dataframe(
            model_predictions, actual_results, vegas_spreads
        )
        
        if len(eval_df) == 0:
            self.logger.error("❌ No games to evaluate")
            return {}
        
        # Calculate spread predictions and actuals
        eval_df = self._calculate_spreads(eval_df)
        
        # Core metrics
        metrics = self._calculate_core_metrics(eval_df)
        
        # ATS (Against The Spread) performance
        ats_metrics = self._calculate_ats_performance(eval_df)
        metrics.update(ats_metrics)
        
        # Statistical analysis
        statistical_metrics = self._calculate_statistical_metrics(eval_df)
        metrics.update(statistical_metrics)
        
        # Betting simulation
        betting_metrics = self._simulate_betting_performance(eval_df)
        metrics.update(betting_metrics)
        
        # Store results
        self.evaluation_results = {
            'metrics': metrics,
            'dataframe': eval_df,
            'timestamp': datetime.now().isoformat()
        }
        
        self._log_results_summary(metrics)
        
        return metrics
    
    def _create_evaluation_dataframe(self,
                                    model_predictions: Dict,
                                    actual_results: Dict,
                                    vegas_spreads: pd.DataFrame) -> pd.DataFrame:
        """Create unified DataFrame for evaluation"""
        
        # Convert model predictions to DataFrame
        model_df = pd.DataFrame({
            'game_id': model_predictions['game_id'],
            'home_score_pred': model_predictions['home_score_pred'],
            'away_score_pred': model_predictions['away_score_pred']
        })
        
        # Convert actual results to DataFrame
        actual_df = pd.DataFrame({
            'game_id': actual_results['game_id'],
            'home_score_actual': actual_results['home_score'],
            'away_score_actual': actual_results['away_score']
        })
        
        # Merge all data
        eval_df = model_df.merge(actual_df, on='game_id', how='inner')
        eval_df = eval_df.merge(vegas_spreads, on='game_id', how='inner')
        
        return eval_df
    
    def _calculate_spreads(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate spread values (away_score - home_score)
        Remember: negative spread means home team favored
        """
        
        # Model predicted spread
        df.loc[:, 'model_spread'] = df['away_score_pred'] - df['home_score_pred']
        # Use .loc to avoid SettingWithCopyWarning
        
        # Actual spread
        df['actual_spread'] = df['away_score_actual'] - df['home_score_actual']
        
        # Vegas spread (already in correct format: away - home)
        df['vegas_spread'] = df['spread']
        
        # Errors
        df['model_spread_error'] = df['model_spread'] - df['actual_spread']
        df['vegas_spread_error'] = df['vegas_spread'] - df['actual_spread']
        
        # Absolute errors
        df['model_abs_error'] = np.abs(df['model_spread_error'])
        df['vegas_abs_error'] = np.abs(df['vegas_spread_error'])
        
        return df
    
    def _calculate_core_metrics(self, df: pd.DataFrame) -> Dict[str, float]:
        """Calculate core performance metrics"""
        
        metrics = {}
        
        # RMSE - The key metric
        metrics['model_spread_rmse'] = np.sqrt(np.mean(df['model_spread_error'] ** 2))
        metrics['vegas_spread_rmse'] = np.sqrt(np.mean(df['vegas_spread_error'] ** 2))
        
        # MAE
        metrics['model_spread_mae'] = df['model_abs_error'].mean()
        metrics['vegas_spread_mae'] = df['vegas_abs_error'].mean()
        
        # Bias
        metrics['model_spread_bias'] = df['model_spread_error'].mean()
        metrics['vegas_spread_bias'] = df['vegas_spread_error'].mean()
        
        # Standard deviation of errors
        metrics['model_error_std'] = df['model_spread_error'].std()
        metrics['vegas_error_std'] = df['vegas_spread_error'].std()
        
        # Correlation with actual spreads
        metrics['model_correlation'] = df['model_spread'].corr(df['actual_spread'])
        metrics['vegas_correlation'] = df['vegas_spread'].corr(df['actual_spread'])
        
        # Performance vs target
        metrics['beats_target_rmse'] = metrics['model_spread_rmse'] < self.config.target_rmse
        metrics['beats_vegas_rmse'] = metrics['model_spread_rmse'] < metrics['vegas_spread_rmse']
        metrics['rmse_improvement_vs_vegas'] = (
            (metrics['vegas_spread_rmse'] - metrics['model_spread_rmse']) / 
            metrics['vegas_spread_rmse'] * 100
        )
        
        return metrics
    
    def _calculate_ats_performance(self, df: pd.DataFrame) -> Dict[str, float]:
        """
        Calculate Against The Spread (ATS) performance
        This is what actually matters for betting
        """
        
        metrics = {}
        
        # Determine ATS winners
        # Model picks home if model_spread < vegas_spread (home covers)
        df['model_picks_home'] = df['model_spread'] < df['vegas_spread']
        
        # Home covers if actual_spread < vegas_spread
        df['home_covers'] = df['actual_spread'] < df['vegas_spread']
        
        # Model correct when pick matches cover result
        df['model_ats_correct'] = df['model_picks_home'] == df['home_covers']
        
        # Calculate ATS win rate
        metrics['model_ats_win_rate'] = df['model_ats_correct'].mean()
        
        # Break-even rate accounting for juice (typically -110)
        juice = 0.1  # 10% juice
        metrics['breakeven_win_rate'] = (1 + juice) / (2 + juice)  # ~52.38%
        metrics['profitable'] = metrics['model_ats_win_rate'] > metrics['breakeven_win_rate']
        
        # ATS performance by spread size
        df['spread_bucket'] = pd.cut(
            np.abs(df['vegas_spread']), 
            bins=[0, 3, 7, 14, 100],
            labels=['Close (0-3)', 'Small (3-7)', 'Medium (7-14)', 'Large (14+)']
        )
        
        ats_by_spread = df.groupby('spread_bucket')['model_ats_correct'].agg(['mean', 'count'])
        
        metrics['ats_by_spread_size'] = ats_by_spread.to_dict()
        
        # Home vs Away performance
        home_games = df[df['vegas_spread'] < 0]  # Home favored
        away_games = df[df['vegas_spread'] > 0]  # Away favored
        
        metrics['ats_home_favored'] = home_games['model_ats_correct'].mean() if len(home_games) > 0 else 0
        metrics['ats_away_favored'] = away_games['model_ats_correct'].mean() if len(away_games) > 0 else 0
        
        # Confidence analysis
        df['model_confidence'] = 1 / (1 + np.exp(-np.abs(df['model_spread'] - df['vegas_spread']) / 3))
        high_confidence = df[df['model_confidence'] > self.config.confidence_threshold]
        
        metrics['high_confidence_ats_rate'] = (
            high_confidence['model_ats_correct'].mean() 
            if len(high_confidence) > 0 else 0
        )
        metrics['high_confidence_games_pct'] = len(high_confidence) / len(df)
        
        return metrics
    
    def _calculate_statistical_metrics(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Calculate statistical significance metrics"""
        
        metrics = {}
        
        # Paired t-test: Model vs Vegas errors
        t_stat, p_value = stats.ttest_rel(
            df['model_abs_error'], 
            df['vegas_abs_error']
        )
        
        metrics['error_ttest_statistic'] = t_stat
        metrics['error_ttest_pvalue'] = p_value
        metrics['model_significantly_better'] = (
            p_value < 0.05 and df['model_abs_error'].mean() < df['vegas_abs_error'].mean()
        )
        
        # Wilcoxon signed-rank test (non-parametric)
        w_stat, w_pvalue = stats.wilcoxon(
            df['model_abs_error'], 
            df['vegas_abs_error']
        )
        
        metrics['wilcoxon_statistic'] = w_stat
        metrics['wilcoxon_pvalue'] = w_pvalue
        
        # Distribution analysis
        metrics['model_error_skewness'] = stats.skew(df['model_spread_error'])
        metrics['model_error_kurtosis'] = stats.kurtosis(df['model_spread_error'])
        
        # Percentile analysis
        percentiles = [10, 25, 50, 75, 90]
        for p in percentiles:
            metrics[f'model_error_p{p}'] = np.percentile(df['model_abs_error'], p)
            metrics[f'vegas_error_p{p}'] = np.percentile(df['vegas_abs_error'], p)
        
        return metrics
    
    def _simulate_betting_performance(self, df: pd.DataFrame) -> Dict[str, float]:
        """
        Simulate betting performance using model predictions
        """
        
        metrics = {}
        
        # Standard bet sizing
        unit_size = 100  # $100 per unit
        total_wagered = len(df) * unit_size
        
        # Calculate returns (assuming -110 juice)
        win_return = 100 / 110  # Win $100 for every $110 risked
        
        df['bet_result'] = df['model_ats_correct'].map({
            True: unit_size * win_return,
            False: -unit_size
        })
        
        total_return = df['bet_result'].sum()
        
        metrics['total_units_wagered'] = len(df)
        metrics['total_units_won'] = total_return / unit_size
        metrics['roi_percentage'] = (total_return / total_wagered) * 100
        metrics['avg_unit_return'] = total_return / len(df)
        
        # Kelly Criterion sizing (for high confidence bets)
        high_conf = df[df['model_confidence'] > self.config.confidence_threshold].copy()
        
        if len(high_conf) > 0:
            # Kelly fraction = (p * b - q) / b
            # where p = win probability, q = 1-p, b = odds received on win
            p = high_conf['model_ats_correct'].mean()
            q = 1 - p
            b = win_return
            
            kelly_fraction = (p * b - q) / b
            conservative_kelly = kelly_fraction * self.config.kelly_fraction
            
            metrics['kelly_fraction'] = kelly_fraction
            metrics['conservative_kelly'] = conservative_kelly
            metrics['kelly_suggested_bankroll_pct'] = max(0, conservative_kelly * 100)
        
        # Cumulative performance
        df_sorted = df.sort_values(['season', 'week'])
        df_sorted['cumulative_return'] = df_sorted['bet_result'].cumsum()
        df_sorted['cumulative_roi'] = (
            df_sorted['cumulative_return'] / 
            (df_sorted.index + 1) / unit_size * 100
        )
        
        metrics['final_bankroll'] = 10000 + total_return  # Starting with $10k
        metrics['max_drawdown'] = self._calculate_max_drawdown(df_sorted['cumulative_return'])
        metrics['sharpe_ratio'] = self._calculate_sharpe_ratio(df['bet_result'])
        
        return metrics
    
    def _calculate_max_drawdown(self, cumulative_returns: pd.Series) -> float:
        """Calculate maximum drawdown from cumulative returns"""
        
        rolling_max = cumulative_returns.expanding().max()
        drawdown = cumulative_returns - rolling_max
        return abs(drawdown.min())
    
    def _calculate_sharpe_ratio(self, returns: pd.Series, risk_free_rate: float = 0.02) -> float:
        """Calculate Sharpe ratio for betting returns"""
        
        excess_returns = returns - (risk_free_rate / 365)  # Daily risk-free rate
        
        if returns.std() > 1e-8:  # Use small epsilon instead of 0
            return np.sqrt(252) * excess_returns.mean() / returns.std()  # Annualized
        return 0.0
    
    def _log_results_summary(self, metrics: Dict[str, Any]):
        """Log summary of evaluation results"""
        
        self.logger.info("=" * 60)
        self.logger.info("🎯 VEGAS SPREAD EVALUATION RESULTS")
        self.logger.info("=" * 60)
        
        # Core metrics
        self.logger.info(f"📊 Model Spread RMSE: {metrics['model_spread_rmse']:.2f}")
        self.logger.info(f"📊 Vegas Spread RMSE: {metrics['vegas_spread_rmse']:.2f}")
        
        if metrics['beats_vegas_rmse']:
            self.logger.info(f"🏆 MODEL BEATS VEGAS by {metrics['rmse_improvement_vs_vegas']:.1f}%!")
        else:
            self.logger.info(f"❌ Vegas wins (Model needs {metrics['model_spread_rmse'] - self.config.target_rmse:.2f} improvement)")
        
        # ATS Performance
        self.logger.info(f"\n💰 ATS Win Rate: {metrics['model_ats_win_rate']*100:.1f}%")
        self.logger.info(f"📈 Break-even Rate: {metrics['breakeven_win_rate']*100:.1f}%")
        
        if metrics['profitable']:
            self.logger.info(f"✅ PROFITABLE MODEL!")
        else:
            needed = metrics['breakeven_win_rate'] - metrics['model_ats_win_rate']
            self.logger.info(f"⚠️ Need {needed*100:.1f}% more wins for profitability")
        
        # ROI
        self.logger.info(f"\n💵 ROI: {metrics['roi_percentage']:.1f}%")
        self.logger.info(f"📊 Sharpe Ratio: {metrics['sharpe_ratio']:.2f}")
        self.logger.info(f"📉 Max Drawdown: ${metrics['max_drawdown']:.0f}")
        
        self.logger.info("=" * 60)
    
    def plot_evaluation_results(self, save_path: Optional[str] = None):
        """Generate comprehensive visualization of results"""
        
        if 'dataframe' not in self.evaluation_results:
            self.logger.warning("No evaluation results to plot")
            return
        
        df = self.evaluation_results['dataframe']
        metrics = self.evaluation_results['metrics']
        
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        fig.suptitle('Model vs Vegas Spread Evaluation', fontsize=16)
        
        # 1. Error Distribution
        ax = axes[0, 0]
        ax.hist(df['model_spread_error'], bins=30, alpha=0.5, label='Model', color='blue')
        ax.hist(df['vegas_spread_error'], bins=30, alpha=0.5, label='Vegas', color='red')
        ax.axvline(0, color='black', linestyle='--', alpha=0.5)
        ax.set_xlabel('Spread Error')
        ax.set_ylabel('Frequency')
        ax.set_title('Error Distribution')
        ax.legend()
        
        # 2. Actual vs Predicted
        ax = axes[0, 1]
        ax.scatter(df['actual_spread'], df['model_spread'], alpha=0.5, label='Model', s=10)
        ax.scatter(df['actual_spread'], df['vegas_spread'], alpha=0.5, label='Vegas', s=10)
        ax.plot([-50, 50], [-50, 50], 'k--', alpha=0.5)
        ax.set_xlabel('Actual Spread')
        ax.set_ylabel('Predicted Spread')
        ax.set_title('Predictions vs Actual')
        ax.legend()
        ax.set_xlim(-50, 50)
        ax.set_ylim(-50, 50)
        
        # 3. Cumulative ROI
        ax = axes[0, 2]
        df_sorted = df.sort_values(['season', 'week'])
        cumulative_roi = df_sorted['bet_result'].cumsum() / (np.arange(len(df_sorted)) + 1) / 100 * 100
        ax.plot(cumulative_roi.values)
        ax.axhline(0, color='red', linestyle='--', alpha=0.5)
        ax.set_xlabel('Games')
        ax.set_ylabel('ROI (%)')
        ax.set_title('Cumulative ROI')
        ax.grid(True, alpha=0.3)
        
        # 4. ATS Performance by Spread Size
        ax = axes[1, 0]
        if 'ats_by_spread_size' in metrics:
            spread_data = metrics['ats_by_spread_size']
            if 'mean' in spread_data:
                categories = list(spread_data['mean'].keys())
                values = list(spread_data['mean'].values())
                ax.bar(categories, values)
                ax.axhline(metrics['breakeven_win_rate'], color='red', linestyle='--', 
                          label='Break-even', alpha=0.7)
                ax.set_ylabel('ATS Win Rate')
                ax.set_title('ATS by Spread Size')
                ax.legend()
                ax.set_ylim(0, 1)
        
        # 5. Model vs Vegas Absolute Errors
        ax = axes[1, 1]
        data_to_plot = pd.DataFrame({
            'Model': df['model_abs_error'],
            'Vegas': df['vegas_abs_error']
        })
        data_to_plot.boxplot(ax=ax)
        ax.set_ylabel('Absolute Error')
        ax.set_title('Absolute Error Comparison')
        ax.axhline(self.config.target_rmse, color='green', linestyle='--', 
                  alpha=0.5, label='Target RMSE')
        ax.legend()
        
        # 6. Confidence vs Accuracy
        ax = axes[1, 2]
        if 'model_confidence' in df.columns:
            confidence_bins = pd.cut(df['model_confidence'], bins=5)
            conf_accuracy = df.groupby(confidence_bins)['model_ats_correct'].mean()
            conf_accuracy.plot(kind='bar', ax=ax)
            ax.axhline(metrics['breakeven_win_rate'], color='red', linestyle='--', alpha=0.7)
            ax.set_xlabel('Confidence Level')
            ax.set_ylabel('ATS Win Rate')
            ax.set_title('Accuracy by Confidence')
            ax.set_xticklabels(ax.get_xticklabels(), rotation=45)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            self.logger.info(f"📊 Plot saved to {save_path}")
        
        plt.show()
    
    def generate_betting_recommendations(self, 
                                        upcoming_games: pd.DataFrame,
                                        model_predictions: Dict[str, np.ndarray]) -> pd.DataFrame:
        """
        Generate betting recommendations for upcoming games
        
        Args:
            upcoming_games: DataFrame with game_id and vegas_spread
            model_predictions: Model predictions for these games
            
        Returns:
            DataFrame with betting recommendations
        """
        
        # Create predictions DataFrame
        pred_df = pd.DataFrame({
            'game_id': model_predictions['game_id'],
            'home_score_pred': model_predictions['home_score_pred'],
            'away_score_pred': model_predictions['away_score_pred']
        })
        
        # Merge with Vegas lines
        recs_df = upcoming_games.merge(pred_df, on='game_id', how='inner')
        
        # Calculate model spread
        recs_df['model_spread'] = recs_df['away_score_pred'] - recs_df['home_score_pred']
        
        # Calculate edge
        recs_df['spread_difference'] = recs_df['model_spread'] - recs_df['vegas_spread']
        recs_df['edge_size'] = np.abs(recs_df['spread_difference'])
        
        # Determine pick
        recs_df['pick'] = np.where(
            recs_df['spread_difference'] < 0,
            'HOME',  # Model likes home more than Vegas
            'AWAY'   # Model likes away more than Vegas
        )
        
        # Calculate confidence
        recs_df['confidence'] = 1 / (1 + np.exp(-recs_df['edge_size'] / 3))
        
        # Determine bet sizing
        recs_df['recommended_units'] = 0
        recs_df.loc[recs_df['confidence'] > 0.55, 'recommended_units'] = 1
        recs_df.loc[recs_df['confidence'] > 0.65, 'recommended_units'] = 2
        recs_df.loc[recs_df['confidence'] > 0.75, 'recommended_units'] = 3
        
        # Sort by confidence
        recs_df = recs_df.sort_values('confidence', ascending=False)
        
        # Select key columns for output
        output_columns = [
            'game_id', 'vegas_spread', 'model_spread', 'spread_difference',
            'pick', 'confidence', 'recommended_units'
        ]
        
        return recs_df[output_columns]

# Utility functions
def evaluate_model_vs_vegas(model_predictions: Dict[str, np.ndarray],
                           actual_results: Dict[str, np.ndarray],
                           years: List[int],
                           weeks: Optional[List[int]] = None,
                           config: Optional[VegasConfig] = None) -> Dict[str, Any]:
    """
    Convenience function to run complete Vegas evaluation
    
    Returns comprehensive metrics comparing model to Vegas spreads
    """
    evaluator = VegasSpreadEvaluator(config)
    
    # Load Vegas spreads
    vegas_spreads = evaluator.load_vegas_spreads(years, weeks)
    
    # Run evaluation
    metrics = evaluator.evaluate_spread_predictions(
        model_predictions, actual_results, vegas_spreads
    )
    
    return metrics

def generate_weekly_report(evaluator: VegasSpreadEvaluator,
                          week: int,
                          season: int) -> str:
    """Generate formatted weekly performance report"""
    
    if not evaluator.evaluation_results:
        return "No evaluation results available"
    
    metrics = evaluator.evaluation_results['metrics']
    
    report = f"""
    ╔══════════════════════════════════════════════════════════╗
    ║     WEEK {week} SEASON {season} - VEGAS COMPARISON REPORT    ║
    ╚══════════════════════════════════════════════════════════╝
    
    📊 SPREAD ACCURACY
    ├─ Model RMSE: {metrics['model_spread_rmse']:.2f}
    ├─ Vegas RMSE: {metrics['vegas_spread_rmse']:.2f}
    └─ Improvement: {metrics.get('rmse_improvement_vs_vegas', 0):.1f}%
    
    💰 BETTING PERFORMANCE
    ├─ ATS Win Rate: {metrics['model_ats_win_rate']*100:.1f}%
    ├─ ROI: {metrics['roi_percentage']:.1f}%
    ├─ Units Won: {metrics['total_units_won']:.1f}
    └─ Sharpe Ratio: {metrics['sharpe_ratio']:.2f}
    
    🎯 TARGET ACHIEVEMENT
    ├─ Beats Target RMSE (<{evaluator.config.target_rmse}): {'✅' if metrics['beats_target_rmse'] else '❌'}
    ├─ Beats Vegas: {'✅' if metrics['beats_vegas_rmse'] else '❌'}
    └─ Profitable: {'✅' if metrics['profitable'] else '❌'}
    
    📈 HIGH CONFIDENCE BETS
    ├─ Win Rate: {metrics.get('high_confidence_ats_rate', 0)*100:.1f}%
    └─ % of Games: {metrics.get('high_confidence_games_pct', 0)*100:.1f}%
    """
    
    return report