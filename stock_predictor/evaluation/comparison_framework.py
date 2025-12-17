"""
Comprehensive comparison framework for evaluating model-pattern combinations.
Implements statistical significance testing, visualization, and automated reporting.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Any, Tuple, Optional
from dataclasses import dataclass
import logging
from scipy import stats
from scipy.stats import ttest_ind, mannwhitneyu, friedmanchisquare
import warnings
warnings.filterwarnings('ignore')

from ..interfaces import BacktestResult
from .performance_evaluator import PerformanceEvaluator


@dataclass
class ComparisonResult:
    """Data model for comparison results."""
    model_type: str
    pattern_length: int
    performance_metrics: Dict[str, float]
    statistical_significance: Dict[str, Any]
    rank: int
    recommendation_score: float


@dataclass
class StatisticalTest:
    """Data model for statistical test results."""
    test_name: str
    statistic: float
    p_value: float
    is_significant: bool
    confidence_level: float
    interpretation: str


class ComparisonFramework:
    """
    Comprehensive comparison framework for evaluating all model-pattern combinations.
    
    Features:
    - Statistical significance testing for performance differences
    - Visualization components for performance comparison charts
    - Automated report generation with recommendations
    - Best configuration selection based on multiple criteria
    """
    
    def __init__(self, confidence_level: float = 0.95):
        """
        Initialize the comparison framework.
        
        Args:
            confidence_level: Statistical confidence level for significance testing
        """
        self.confidence_level = confidence_level
        self.alpha = 1 - confidence_level
        self.logger = logging.getLogger(__name__)
        self.performance_evaluator = PerformanceEvaluator()
    
    def compare_all_combinations(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Compare all model-pattern combinations with statistical significance testing.
        
        Args:
            results: List of result dictionaries containing model performance data
            
        Returns:
            Comprehensive comparison results with statistical analysis
        """
        if len(results) < 2:
            self.logger.warning("Need at least 2 results for meaningful comparison")
            return {'error': 'Insufficient data for comparison'}
        
        # Perform pairwise statistical tests
        statistical_tests = self._perform_statistical_tests(results)
        
        # Rank all combinations
        ranked_results = self.performance_evaluator.rank_model_combinations(results)
        
        # Generate comparison results
        comparison_results = []
        for i, result in enumerate(ranked_results):
            comparison_result = ComparisonResult(
                model_type=result.get('model_type', 'Unknown'),
                pattern_length=result.get('pattern_length', 0),
                performance_metrics=self._extract_performance_metrics(result),
                statistical_significance=statistical_tests.get(i, {}),
                rank=result.get('rank', i + 1),
                recommendation_score=self._calculate_recommendation_score(result)
            )
            comparison_results.append(comparison_result)
        
        # Generate comprehensive report
        report = self._generate_comparison_report(comparison_results, statistical_tests)
        
        self.logger.info(f"Completed comparison of {len(results)} model-pattern combinations")
        return report
    
    def _perform_statistical_tests(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Perform statistical significance tests between model combinations.
        
        Args:
            results: List of result dictionaries
            
        Returns:
            Dictionary containing statistical test results
        """
        statistical_tests = {}
        
        # Extract performance metrics for testing
        returns_data = []
        sharpe_data = []
        accuracy_data = []
        
        for result in results:
            financial_metrics = result.get('financial_metrics', {})
            prediction_metrics = result.get('prediction_metrics', {})
            
            returns_data.append(financial_metrics.get('total_return', 0.0))
            sharpe_data.append(financial_metrics.get('sharpe_ratio', 0.0))
            accuracy_data.append(prediction_metrics.get('accuracy', 0.0))
        
        # Perform Friedman test for multiple groups (non-parametric)
        if len(results) >= 3:
            try:
                # Prepare data for Friedman test (each metric as a separate group)
                friedman_stat, friedman_p = friedmanchisquare(returns_data, sharpe_data, accuracy_data)
                
                statistical_tests['friedman_test'] = StatisticalTest(
                    test_name='Friedman Chi-Square Test',
                    statistic=float(friedman_stat),
                    p_value=float(friedman_p),
                    is_significant=friedman_p < self.alpha,
                    confidence_level=self.confidence_level,
                    interpretation=self._interpret_friedman_test(friedman_p)
                )
            except Exception as e:
                self.logger.warning(f"Friedman test failed: {e}")
        
        # Perform pairwise t-tests for returns
        pairwise_tests = {}
        for i in range(len(results)):
            for j in range(i + 1, len(results)):
                try:
                    # Compare returns between two configurations
                    config1_returns = [returns_data[i]]
                    config2_returns = [returns_data[j]]
                    
                    # Use Mann-Whitney U test (non-parametric)
                    statistic, p_value = mannwhitneyu(config1_returns, config2_returns, alternative='two-sided')
                    
                    test_key = f"config_{i}_vs_{j}"
                    pairwise_tests[test_key] = StatisticalTest(
                        test_name='Mann-Whitney U Test',
                        statistic=float(statistic),
                        p_value=float(p_value),
                        is_significant=p_value < self.alpha,
                        confidence_level=self.confidence_level,
                        interpretation=self._interpret_mannwhitney_test(p_value, returns_data[i], returns_data[j])
                    )
                except Exception as e:
                    self.logger.warning(f"Pairwise test {i} vs {j} failed: {e}")
        
        statistical_tests['pairwise_tests'] = pairwise_tests
        return statistical_tests
    
    def _interpret_friedman_test(self, p_value: float) -> str:
        """Interpret Friedman test results."""
        if p_value < self.alpha:
            return f"Significant differences detected between model configurations (p={p_value:.4f})"
        else:
            return f"No significant differences between model configurations (p={p_value:.4f})"
    
    def _interpret_mannwhitney_test(self, p_value: float, return1: float, return2: float) -> str:
        """Interpret Mann-Whitney U test results."""
        if p_value < self.alpha:
            better_config = "first" if return1 > return2 else "second"
            return f"Significant difference detected, {better_config} configuration performs better (p={p_value:.4f})"
        else:
            return f"No significant difference between configurations (p={p_value:.4f})"
    
    def _extract_performance_metrics(self, result: Dict[str, Any]) -> Dict[str, float]:
        """Extract and combine performance metrics from result."""
        financial_metrics = result.get('financial_metrics', {})
        prediction_metrics = result.get('prediction_metrics', {})
        
        return {
            **financial_metrics,
            **prediction_metrics,
            'composite_score': result.get('composite_score', 0.0)
        }
    
    def _calculate_recommendation_score(self, result: Dict[str, Any]) -> float:
        """
        Calculate recommendation score based on multiple criteria.
        
        Args:
            result: Result dictionary
            
        Returns:
            Recommendation score (0-100)
        """
        financial_metrics = result.get('financial_metrics', {})
        prediction_metrics = result.get('prediction_metrics', {})
        
        # Normalize metrics to 0-1 scale
        sharpe_score = min(1.0, max(0.0, (financial_metrics.get('sharpe_ratio', 0) + 2) / 4))  # Assume range -2 to 2
        return_score = min(1.0, max(0.0, financial_metrics.get('total_return', 0) + 0.5))  # Assume range -0.5 to 0.5
        drawdown_score = min(1.0, max(0.0, 1 + financial_metrics.get('max_drawdown', 0)))  # Convert negative to positive
        accuracy_score = prediction_metrics.get('accuracy', 0.0)
        
        # Weighted recommendation score
        recommendation_score = (
            0.35 * sharpe_score +
            0.25 * return_score +
            0.25 * drawdown_score +
            0.15 * accuracy_score
        ) * 100
        
        return float(recommendation_score)
    
    def _generate_comparison_report(self, comparison_results: List[ComparisonResult], 
                                  statistical_tests: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate comprehensive comparison report.
        
        Args:
            comparison_results: List of comparison results
            statistical_tests: Statistical test results
            
        Returns:
            Comprehensive comparison report
        """
        # Sort by recommendation score
        sorted_results = sorted(comparison_results, key=lambda x: x.recommendation_score, reverse=True)
        
        # Best configuration
        best_config = sorted_results[0] if sorted_results else None
        
        # Pattern length analysis
        pattern_analysis = self._analyze_pattern_performance(comparison_results)
        
        # Model type analysis
        model_analysis = self._analyze_model_performance(comparison_results)
        
        # Generate recommendations
        recommendations = self._generate_recommendations(sorted_results, statistical_tests)
        
        report = {
            'executive_summary': {
                'total_configurations': len(comparison_results),
                'best_configuration': {
                    'model_type': best_config.model_type if best_config else 'Unknown',
                    'pattern_length': best_config.pattern_length if best_config else 0,
                    'recommendation_score': best_config.recommendation_score if best_config else 0.0,
                    'key_metrics': best_config.performance_metrics if best_config else {}
                },
                'statistical_significance': statistical_tests.get('friedman_test', {})
            },
            'detailed_results': [
                {
                    'model_type': result.model_type,
                    'pattern_length': result.pattern_length,
                    'rank': result.rank,
                    'recommendation_score': result.recommendation_score,
                    'performance_metrics': result.performance_metrics,
                    'statistical_significance': result.statistical_significance
                }
                for result in sorted_results
            ],
            'pattern_length_analysis': pattern_analysis,
            'model_type_analysis': model_analysis,
            'statistical_tests': statistical_tests,
            'recommendations': recommendations
        }
        
        return report
    
    def _analyze_pattern_performance(self, comparison_results: List[ComparisonResult]) -> Dict[str, Any]:
        """Analyze performance across different pattern lengths."""
        pattern_performance = {}
        
        # Group by pattern length
        pattern_groups = {}
        for result in comparison_results:
            pattern_length = result.pattern_length
            if pattern_length not in pattern_groups:
                pattern_groups[pattern_length] = []
            pattern_groups[pattern_length].append(result)
        
        # Calculate statistics for each pattern length
        for pattern_length, results in pattern_groups.items():
            scores = [r.recommendation_score for r in results]
            returns = [r.performance_metrics.get('total_return', 0.0) for r in results]
            sharpe_ratios = [r.performance_metrics.get('sharpe_ratio', 0.0) for r in results]
            
            pattern_performance[f'{pattern_length}_day'] = {
                'count': len(results),
                'avg_recommendation_score': float(np.mean(scores)),
                'avg_total_return': float(np.mean(returns)),
                'avg_sharpe_ratio': float(np.mean(sharpe_ratios)),
                'best_model': max(results, key=lambda x: x.recommendation_score).model_type,
                'rank_range': f"{min(r.rank for r in results)}-{max(r.rank for r in results)}"
            }
        
        return pattern_performance
    
    def _analyze_model_performance(self, comparison_results: List[ComparisonResult]) -> Dict[str, Any]:
        """Analyze performance across different model types."""
        model_performance = {}
        
        # Group by model type
        model_groups = {}
        for result in comparison_results:
            model_type = result.model_type
            if model_type not in model_groups:
                model_groups[model_type] = []
            model_groups[model_type].append(result)
        
        # Calculate statistics for each model type
        for model_type, results in model_groups.items():
            scores = [r.recommendation_score for r in results]
            returns = [r.performance_metrics.get('total_return', 0.0) for r in results]
            sharpe_ratios = [r.performance_metrics.get('sharpe_ratio', 0.0) for r in results]
            
            model_performance[model_type] = {
                'count': len(results),
                'avg_recommendation_score': float(np.mean(scores)),
                'avg_total_return': float(np.mean(returns)),
                'avg_sharpe_ratio': float(np.mean(sharpe_ratios)),
                'best_pattern_length': max(results, key=lambda x: x.recommendation_score).pattern_length,
                'rank_range': f"{min(r.rank for r in results)}-{max(r.rank for r in results)}"
            }
        
        return model_performance
    
    def _generate_recommendations(self, sorted_results: List[ComparisonResult], 
                                statistical_tests: Dict[str, Any]) -> List[str]:
        """Generate actionable recommendations based on analysis."""
        recommendations = []
        
        if not sorted_results:
            return ["No results available for recommendations"]
        
        best_config = sorted_results[0]
        
        # Primary recommendation
        recommendations.append(
            f"Recommended configuration: {best_config.model_type} with {best_config.pattern_length}-day "
            f"candlestick patterns (Recommendation Score: {best_config.recommendation_score:.1f}/100)"
        )
        
        # Performance insights
        if best_config.performance_metrics.get('total_return', 0) > 0:
            recommendations.append(
                f"This configuration achieved {best_config.performance_metrics.get('total_return', 0):.2%} "
                f"total return with a Sharpe ratio of {best_config.performance_metrics.get('sharpe_ratio', 0):.2f}"
            )
        
        # Statistical significance insights
        friedman_test = statistical_tests.get('friedman_test')
        if friedman_test and friedman_test.is_significant:
            recommendations.append(
                "Statistical analysis confirms significant differences between configurations, "
                "supporting the reliability of this recommendation"
            )
        
        # Pattern length insights
        pattern_lengths = list(set(r.pattern_length for r in sorted_results))
        if len(pattern_lengths) > 1:
            best_pattern = best_config.pattern_length
            recommendations.append(
                f"Among pattern lengths tested ({', '.join(map(str, sorted(pattern_lengths)))} days), "
                f"{best_pattern}-day patterns showed superior performance"
            )
        
        # Risk considerations
        max_drawdown = best_config.performance_metrics.get('max_drawdown', 0)
        if max_drawdown < -0.1:  # More than 10% drawdown
            recommendations.append(
                f"Consider risk management: Maximum drawdown was {max_drawdown:.2%}. "
                "Implement position sizing and stop-loss strategies"
            )
        
        # Alternative configurations
        if len(sorted_results) > 1:
            second_best = sorted_results[1]
            score_diff = best_config.recommendation_score - second_best.recommendation_score
            if score_diff < 5:  # Close competition
                recommendations.append(
                    f"Alternative consideration: {second_best.model_type} with {second_best.pattern_length}-day "
                    f"patterns showed competitive performance (Score: {second_best.recommendation_score:.1f}/100)"
                )
        
        return recommendations
    
    def generate_visualization_data(self, comparison_results: List[ComparisonResult]) -> Dict[str, Any]:
        """
        Generate data for visualization components.
        
        Args:
            comparison_results: List of comparison results
            
        Returns:
            Dictionary containing data for various charts and visualizations
        """
        if not comparison_results:
            return {}
        
        # Performance comparison data
        performance_data = []
        for result in comparison_results:
            performance_data.append({
                'model_type': result.model_type,
                'pattern_length': result.pattern_length,
                'total_return': result.performance_metrics.get('total_return', 0.0),
                'sharpe_ratio': result.performance_metrics.get('sharpe_ratio', 0.0),
                'max_drawdown': result.performance_metrics.get('max_drawdown', 0.0),
                'recommendation_score': result.recommendation_score,
                'rank': result.rank
            })
        
        # Heatmap data for pattern length vs model type
        heatmap_data = self._prepare_heatmap_data(comparison_results)
        
        # Scatter plot data for risk-return analysis
        scatter_data = []
        for result in comparison_results:
            scatter_data.append({
                'x': result.performance_metrics.get('volatility', 0.0),  # Risk (x-axis)
                'y': result.performance_metrics.get('total_return', 0.0),  # Return (y-axis)
                'size': result.recommendation_score,  # Bubble size
                'label': f"{result.model_type} ({result.pattern_length}d)",
                'color': result.model_type
            })
        
        visualization_data = {
            'performance_comparison': performance_data,
            'heatmap_data': heatmap_data,
            'risk_return_scatter': scatter_data,
            'ranking_data': sorted(performance_data, key=lambda x: x['rank'])
        }
        
        return visualization_data
    
    def _prepare_heatmap_data(self, comparison_results: List[ComparisonResult]) -> Dict[str, Any]:
        """Prepare data for heatmap visualization."""
        # Create matrix of recommendation scores
        model_types = sorted(list(set(r.model_type for r in comparison_results)))
        pattern_lengths = sorted(list(set(r.pattern_length for r in comparison_results)))
        
        heatmap_matrix = []
        for model_type in model_types:
            row = []
            for pattern_length in pattern_lengths:
                # Find matching result
                matching_result = next(
                    (r for r in comparison_results 
                     if r.model_type == model_type and r.pattern_length == pattern_length),
                    None
                )
                score = matching_result.recommendation_score if matching_result else 0.0
                row.append(score)
            heatmap_matrix.append(row)
        
        return {
            'matrix': heatmap_matrix,
            'x_labels': [f"{pl}d" for pl in pattern_lengths],
            'y_labels': model_types,
            'title': 'Recommendation Scores by Model Type and Pattern Length'
        }
    
    def create_performance_charts(self, comparison_results: List[ComparisonResult], 
                                save_path: str = None) -> Dict[str, str]:
        """
        Create performance comparison charts.
        
        Args:
            comparison_results: List of comparison results
            save_path: Optional path to save charts
            
        Returns:
            Dictionary with chart file paths or base64 encoded images
        """
        if not comparison_results:
            return {}
        
        chart_paths = {}
        
        # Set up the plotting style
        plt.style.use('seaborn-v0_8')
        
        # 1. Recommendation Score Comparison
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Prepare data for plotting
        models = [r.model_type for r in comparison_results]
        patterns = [f"{r.pattern_length}d" for r in comparison_results]
        scores = [r.recommendation_score for r in comparison_results]
        
        # Create labels combining model and pattern
        labels = [f"{m}\n({p})" for m, p in zip(models, patterns)]
        
        # Create bar chart
        bars = ax.bar(range(len(labels)), scores, color=plt.cm.viridis(np.linspace(0, 1, len(labels))))
        ax.set_xlabel('Model Configuration')
        ax.set_ylabel('Recommendation Score')
        ax.set_title('Model Configuration Recommendation Scores')
        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(labels, rotation=45, ha='right')
        
        # Add value labels on bars
        for bar, score in zip(bars, scores):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 1,
                   f'{score:.1f}', ha='center', va='bottom')
        
        plt.tight_layout()
        
        if save_path:
            chart_path = f"{save_path}/recommendation_scores.png"
            plt.savefig(chart_path, dpi=300, bbox_inches='tight')
            chart_paths['recommendation_scores'] = chart_path
        
        plt.close()
        
        # 2. Risk-Return Scatter Plot
        fig, ax = plt.subplots(figsize=(10, 8))
        
        returns = [r.performance_metrics.get('total_return', 0.0) for r in comparison_results]
        drawdowns = [abs(r.performance_metrics.get('max_drawdown', 0.0)) for r in comparison_results]
        colors = [r.recommendation_score for r in comparison_results]
        
        scatter = ax.scatter(drawdowns, returns, c=colors, s=100, alpha=0.7, cmap='viridis')
        
        # Add labels for each point
        for i, result in enumerate(comparison_results):
            ax.annotate(f"{result.model_type}\n({result.pattern_length}d)", 
                       (drawdowns[i], returns[i]), 
                       xytext=(5, 5), textcoords='offset points', fontsize=8)
        
        ax.set_xlabel('Maximum Drawdown (Risk)')
        ax.set_ylabel('Total Return')
        ax.set_title('Risk-Return Analysis of Model Configurations')
        
        # Add colorbar
        cbar = plt.colorbar(scatter)
        cbar.set_label('Recommendation Score')
        
        plt.tight_layout()
        
        if save_path:
            chart_path = f"{save_path}/risk_return_analysis.png"
            plt.savefig(chart_path, dpi=300, bbox_inches='tight')
            chart_paths['risk_return_analysis'] = chart_path
        
        plt.close()
        
        # 3. Performance Metrics Heatmap
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Prepare heatmap data
        heatmap_data = self._prepare_heatmap_data(comparison_results)
        
        im = ax.imshow(heatmap_data['matrix'], cmap='viridis', aspect='auto')
        
        # Set ticks and labels
        ax.set_xticks(range(len(heatmap_data['x_labels'])))
        ax.set_yticks(range(len(heatmap_data['y_labels'])))
        ax.set_xticklabels(heatmap_data['x_labels'])
        ax.set_yticklabels(heatmap_data['y_labels'])
        
        # Add text annotations
        for i in range(len(heatmap_data['y_labels'])):
            for j in range(len(heatmap_data['x_labels'])):
                text = ax.text(j, i, f'{heatmap_data["matrix"][i][j]:.1f}',
                             ha="center", va="center", color="white", fontweight='bold')
        
        ax.set_title(heatmap_data['title'])
        ax.set_xlabel('Pattern Length')
        ax.set_ylabel('Model Type')
        
        # Add colorbar
        cbar = plt.colorbar(im)
        cbar.set_label('Recommendation Score')
        
        plt.tight_layout()
        
        if save_path:
            chart_path = f"{save_path}/performance_heatmap.png"
            plt.savefig(chart_path, dpi=300, bbox_inches='tight')
            chart_paths['performance_heatmap'] = chart_path
        
        plt.close()
        
        self.logger.info(f"Generated {len(chart_paths)} performance charts")
        return chart_paths
    
    def select_best_configuration(self, comparison_results: List[ComparisonResult], 
                                criteria_weights: Dict[str, float] = None) -> ComparisonResult:
        """
        Select the best configuration based on multiple criteria.
        
        Args:
            comparison_results: List of comparison results
            criteria_weights: Optional custom weights for selection criteria
            
        Returns:
            Best configuration based on weighted criteria
        """
        if not comparison_results:
            raise ValueError("No comparison results provided")
        
        # Default criteria weights
        default_weights = {
            'total_return': 0.25,
            'sharpe_ratio': 0.30,
            'max_drawdown': 0.20,  # Lower is better
            'accuracy': 0.15,
            'win_rate': 0.10
        }
        
        weights = criteria_weights or default_weights
        
        # Calculate weighted scores
        scored_results = []
        for result in comparison_results:
            metrics = result.performance_metrics
            
            # Normalize metrics (0-1 scale)
            return_score = max(0, min(1, (metrics.get('total_return', 0) + 0.5) / 1.0))
            sharpe_score = max(0, min(1, (metrics.get('sharpe_ratio', 0) + 2) / 4))
            drawdown_score = max(0, min(1, 1 + metrics.get('max_drawdown', 0)))  # Invert negative
            accuracy_score = metrics.get('accuracy', 0)
            win_rate_score = metrics.get('win_rate', 0)
            
            # Calculate weighted score
            weighted_score = (
                weights.get('total_return', 0) * return_score +
                weights.get('sharpe_ratio', 0) * sharpe_score +
                weights.get('max_drawdown', 0) * drawdown_score +
                weights.get('accuracy', 0) * accuracy_score +
                weights.get('win_rate', 0) * win_rate_score
            )
            
            scored_results.append((result, weighted_score))
        
        # Select best configuration
        best_result, best_score = max(scored_results, key=lambda x: x[1])
        
        self.logger.info(f"Selected best configuration: {best_result.model_type} "
                        f"with {best_result.pattern_length}-day patterns (Score: {best_score:.3f})")
        
        return best_result