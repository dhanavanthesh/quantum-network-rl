"""
Performance Metrics Collection and Analysis
Implements comprehensive metrics for quantum network evaluation
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, field
import time


@dataclass
class NetworkMetrics:
    """Container for network performance metrics"""
    # Information metrics
    information_density: float = 0.0  # bits per photon
    throughput: float = 0.0  # successful transmissions per time unit
    spectral_efficiency: float = 0.0  # bits/Hz

    # Network efficiency
    network_efficiency: float = 0.0  # fraction of successful transmissions
    collision_rate: float = 0.0  # fraction of collisions
    channel_utilization: float = 0.0  # fraction of channels used

    # Quantum metrics
    average_fidelity: float = 0.0  # average quantum state fidelity
    qber: float = 0.0  # quantum bit error rate
    entanglement_fidelity: float = 0.0  # for entanglement-based protocols

    # Learning metrics
    convergence_rate: float = 0.0  # RL convergence speed
    learning_efficiency: float = 0.0  # reward per episode

    # Latency metrics
    average_latency: float = 0.0  # average communication latency
    latency_std: float = 0.0  # latency standard deviation

    # Fault tolerance
    node_failure_resilience: float = 0.0  # resilience to node failures
    link_failure_resilience: float = 0.0  # resilience to link failures

    # Timestamp
    timestamp: float = field(default_factory=time.time)


class MetricsCollector:
    """Collects and computes network performance metrics"""

    def __init__(self, n_nodes: int, n_temporal: int = 32, n_spectral: int = 16):
        """
        Initialize metrics collector

        Args:
            n_nodes: Number of nodes in network
            n_temporal: Number of temporal bins
            n_spectral: Number of spectral bins
        """
        self.n_nodes = n_nodes
        self.n_temporal = n_temporal
        self.n_spectral = n_spectral
        self.n_modes = n_temporal * n_spectral

        # Event counters
        self.total_transmissions = 0
        self.successful_transmissions = 0
        self.collisions = 0
        self.total_photons = 0

        # History
        self.metrics_history: List[NetworkMetrics] = []
        self.collision_events: List[Tuple[int, int, int]] = []  # (timestamp, node1, node2)
        self.fidelity_samples: List[float] = []
        self.latency_samples: List[float] = []

        # Episode tracking
        self.episode_rewards: List[float] = []
        self.episode_lengths: List[int] = []

    def record_transmission(self,
                           success: bool,
                           n_photons: int = 1,
                           fidelity: Optional[float] = None,
                           latency: Optional[float] = None):
        """
        Record a transmission event

        Args:
            success: Whether transmission was successful
            n_photons: Number of photons used
            fidelity: Quantum state fidelity
            latency: Communication latency
        """
        self.total_transmissions += 1
        self.total_photons += n_photons

        if success:
            self.successful_transmissions += 1
        else:
            self.collisions += 1

        if fidelity is not None:
            self.fidelity_samples.append(fidelity)

        if latency is not None:
            self.latency_samples.append(latency)

    def record_collision(self, timestamp: int, node1: int, node2: int):
        """Record collision event"""
        self.collision_events.append((timestamp, node1, node2))
        self.collisions += 1

    def record_episode(self, total_reward: float, episode_length: int):
        """Record episode statistics"""
        self.episode_rewards.append(total_reward)
        self.episode_lengths.append(episode_length)

    def calculate_information_density(self) -> float:
        """
        Calculate information density (bits per photon)

        Returns:
            Information density
        """
        if self.total_photons == 0:
            return 0.0

        bits_per_transmission = np.log2(self.n_modes)
        total_bits = self.successful_transmissions * bits_per_transmission

        return total_bits / self.total_photons

    def calculate_network_efficiency(self) -> float:
        """
        Calculate network efficiency

        Returns:
            Network efficiency (0 to 1)
        """
        if self.total_transmissions == 0:
            return 0.0

        return self.successful_transmissions / self.total_transmissions

    def calculate_collision_rate(self) -> float:
        """
        Calculate collision rate

        Returns:
            Collision rate (0 to 1)
        """
        if self.total_transmissions == 0:
            return 0.0

        return self.collisions / self.total_transmissions

    def calculate_throughput(self, time_elapsed: float) -> float:
        """
        Calculate throughput

        Args:
            time_elapsed: Elapsed time

        Returns:
            Throughput (transmissions per second)
        """
        if time_elapsed <= 0:
            return 0.0

        return self.successful_transmissions / time_elapsed

    def calculate_convergence_rate(self, window_size: int = 100) -> float:
        """
        Calculate RL convergence rate based on reward stability

        Args:
            window_size: Window for calculating variance

        Returns:
            Convergence rate (0 to 1, 1 = fully converged)
        """
        if len(self.episode_rewards) < window_size:
            return 0.0

        recent_rewards = self.episode_rewards[-window_size:]
        mean_reward = np.mean(recent_rewards)
        std_reward = np.std(recent_rewards)

        if mean_reward == 0:
            return 0.0

        # Convergence = 1 - (coefficient of variation)
        cv = std_reward / abs(mean_reward)
        convergence = max(0.0, 1.0 - cv)

        return convergence

    def calculate_average_fidelity(self) -> float:
        """
        Calculate average quantum state fidelity

        Returns:
            Average fidelity
        """
        if not self.fidelity_samples:
            return 0.0

        return np.mean(self.fidelity_samples)

    def calculate_qber(self, error_samples: List[bool]) -> float:
        """
        Calculate QBER from error samples

        Args:
            error_samples: List of boolean error indicators

        Returns:
            QBER
        """
        if not error_samples:
            return 0.0

        return sum(error_samples) / len(error_samples)

    def calculate_latency_statistics(self) -> Tuple[float, float]:
        """
        Calculate latency statistics

        Returns:
            (mean_latency, std_latency)
        """
        if not self.latency_samples:
            return 0.0, 0.0

        return np.mean(self.latency_samples), np.std(self.latency_samples)

    def compute_current_metrics(self, time_elapsed: float = 1.0) -> NetworkMetrics:
        """
        Compute current network metrics

        Args:
            time_elapsed: Time elapsed since start

        Returns:
            NetworkMetrics object
        """
        metrics = NetworkMetrics()

        # Information metrics
        metrics.information_density = self.calculate_information_density()
        metrics.throughput = self.calculate_throughput(time_elapsed)

        # Network efficiency
        metrics.network_efficiency = self.calculate_network_efficiency()
        metrics.collision_rate = self.calculate_collision_rate()
        metrics.channel_utilization = self.successful_transmissions / max(1, self.n_nodes)

        # Quantum metrics
        metrics.average_fidelity = self.calculate_average_fidelity()

        # Learning metrics
        metrics.convergence_rate = self.calculate_convergence_rate()
        if self.episode_rewards:
            metrics.learning_efficiency = np.mean(self.episode_rewards[-100:])

        # Latency metrics
        metrics.average_latency, metrics.latency_std = self.calculate_latency_statistics()

        # Store in history
        self.metrics_history.append(metrics)

        return metrics

    def get_metrics_summary(self) -> Dict:
        """
        Get summary of all metrics

        Returns:
            Dictionary with metric summaries
        """
        if not self.metrics_history:
            return {}

        latest = self.metrics_history[-1]

        return {
            'information_density': latest.information_density,
            'throughput': latest.throughput,
            'network_efficiency': latest.network_efficiency,
            'collision_rate': latest.collision_rate,
            'average_fidelity': latest.average_fidelity,
            'qber': latest.qber,
            'convergence_rate': latest.convergence_rate,
            'average_latency': latest.average_latency,
            'total_transmissions': self.total_transmissions,
            'successful_transmissions': self.successful_transmissions,
            'total_episodes': len(self.episode_rewards)
        }

    def get_time_series(self, metric_name: str) -> List[float]:
        """
        Get time series for a specific metric

        Args:
            metric_name: Name of metric

        Returns:
            List of metric values over time
        """
        return [getattr(m, metric_name, 0.0) for m in self.metrics_history]

    def reset(self):
        """Reset all metrics"""
        self.total_transmissions = 0
        self.successful_transmissions = 0
        self.collisions = 0
        self.total_photons = 0
        self.metrics_history = []
        self.collision_events = []
        self.fidelity_samples = []
        self.latency_samples = []
        self.episode_rewards = []
        self.episode_lengths = []


class ComparisonAnalyzer:
    """Analyzer for comparing different algorithms"""

    def __init__(self):
        """Initialize comparison analyzer"""
        self.algorithm_metrics: Dict[str, List[NetworkMetrics]] = {}

    def add_algorithm_metrics(self, algorithm_name: str, metrics: List[NetworkMetrics]):
        """
        Add metrics for an algorithm

        Args:
            algorithm_name: Name of algorithm
            metrics: List of metrics from multiple runs
        """
        self.algorithm_metrics[algorithm_name] = metrics

    def compare_algorithms(self, metric_name: str) -> Dict[str, Dict]:
        """
        Compare algorithms on a specific metric

        Args:
            metric_name: Metric to compare

        Returns:
            Dictionary with comparison statistics
        """
        comparison = {}

        for algo_name, metrics_list in self.algorithm_metrics.items():
            values = [getattr(m, metric_name, 0.0) for m in metrics_list]

            comparison[algo_name] = {
                'mean': np.mean(values),
                'std': np.std(values),
                'min': np.min(values),
                'max': np.max(values),
                'median': np.median(values)
            }

        return comparison

    def statistical_significance_test(self,
                                     algo1: str,
                                     algo2: str,
                                     metric_name: str,
                                     confidence: float = 0.95) -> Dict:
        """
        Perform statistical significance test between two algorithms

        Args:
            algo1: First algorithm name
            algo2: Second algorithm name
            metric_name: Metric to test
            confidence: Confidence level

        Returns:
            Test results
        """
        from scipy import stats

        if algo1 not in self.algorithm_metrics or algo2 not in self.algorithm_metrics:
            return {'error': 'Algorithm not found'}

        values1 = [getattr(m, metric_name, 0.0) for m in self.algorithm_metrics[algo1]]
        values2 = [getattr(m, metric_name, 0.0) for m in self.algorithm_metrics[algo2]]

        # Perform t-test
        t_stat, p_value = stats.ttest_ind(values1, values2)

        # Calculate confidence interval
        mean_diff = np.mean(values1) - np.mean(values2)
        std_diff = np.sqrt(np.var(values1)/len(values1) + np.var(values2)/len(values2))

        z_score = stats.norm.ppf((1 + confidence) / 2)
        ci_lower = mean_diff - z_score * std_diff
        ci_upper = mean_diff + z_score * std_diff

        return {
            't_statistic': t_stat,
            'p_value': p_value,
            'significant': p_value < (1 - confidence),
            'mean_difference': mean_diff,
            'confidence_interval': (ci_lower, ci_upper),
            'confidence_level': confidence
        }

    def generate_comparison_report(self) -> Dict:
        """
        Generate comprehensive comparison report

        Returns:
            Report dictionary
        """
        report = {
            'algorithms': list(self.algorithm_metrics.keys()),
            'metrics_comparison': {}
        }

        # Compare all tracked metrics
        metric_names = [
            'information_density',
            'network_efficiency',
            'collision_rate',
            'convergence_rate',
            'average_fidelity',
            'throughput'
        ]

        for metric in metric_names:
            report['metrics_comparison'][metric] = self.compare_algorithms(metric)

        # Add best algorithm for each metric
        report['best_algorithms'] = {}
        for metric in metric_names:
            comparison = report['metrics_comparison'][metric]
            best_algo = max(comparison.keys(), key=lambda k: comparison[k]['mean'])
            report['best_algorithms'][metric] = best_algo

        return report


def calculate_fault_tolerance(metrics_normal: NetworkMetrics,
                              metrics_with_faults: NetworkMetrics) -> float:
    """
    Calculate fault tolerance as relative performance degradation

    Args:
        metrics_normal: Metrics under normal operation
        metrics_with_faults: Metrics with faults

    Returns:
        Fault tolerance (0 to 1, higher is better)
    """
    if metrics_normal.network_efficiency == 0:
        return 0.0

    degradation = 1.0 - (metrics_with_faults.network_efficiency /
                        metrics_normal.network_efficiency)

    # Fault tolerance = 1 - degradation
    fault_tolerance = max(0.0, 1.0 - degradation)

    return fault_tolerance
