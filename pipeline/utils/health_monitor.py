"""
============================================================================
HEALTH MONITOR UTILITY
============================================================================
Purpose: Monitor pipeline health and resource usage
Features:
    - Budget monitoring with alerts
    - API rate limit tracking
    - Error rate monitoring
    - Performance metrics
    - Health checks for external services
    - Alert notifications

Usage:
    from utils.health_monitor import HealthMonitor
    
    monitor = HealthMonitor(budget_limit=420.0, cost_tracker=tracker)
    
    # Check budget
    if not monitor.check_budget(current_cost):
        logger.error("Budget exceeded!")
        break
    
    # Track errors
    monitor.track_error("stage_5", "API timeout")
    
    # Get health report
    report = monitor.get_health_report()

Author: GATE AE SOTA Pipeline
============================================================================
"""

import time
from typing import Dict, Any, Optional, List
from datetime import datetime, timedelta
from collections import defaultdict, deque

from utils.logging_utils import setup_logger

logger = setup_logger("health_monitor")


class HealthMonitor:
    """
    Monitor pipeline health and alert on issues
    
    Monitors:
    - Budget usage and alerts
    - API rate limits
    - Error rates by stage
    - Processing speed
    - Service connectivity
    """
    
    def __init__(
        self,
        budget_limit: float,
        cost_tracker=None,
        error_rate_threshold: float = 0.10,  # 10% error rate threshold
        rate_limit_window: int = 3600  # 1 hour window for rate limiting
    ):
        """
        Args:
            budget_limit: Total budget limit in USD
            cost_tracker: CostTracker instance (optional)
            error_rate_threshold: Alert if error rate exceeds this (0.0-1.0)
            rate_limit_window: Time window for rate limit tracking (seconds)
        """
        self.budget_limit = budget_limit
        self.cost_tracker = cost_tracker
        self.error_rate_threshold = error_rate_threshold
        self.rate_limit_window = rate_limit_window
        
        # Budget alerts
        self.budget_alerts = {
            0.75: False,  # 75% warning
            0.90: False,  # 90% critical warning
            1.00: False   # 100% exceeded
        }
        
        # Error tracking
        self.errors_by_stage = defaultdict(list)  # {stage: [timestamp, ...]}
        self.total_errors = 0
        self.total_operations = 0
        
        # Rate limit tracking
        self.api_calls_by_model = defaultdict(lambda: deque())  # {model: deque([timestamp, ...])}
        
        # Performance tracking
        self.processing_times = deque(maxlen=100)  # Last 100 question times
        
        # Service health
        self.service_status = {
            "redis": "unknown",
            "qdrant": "unknown",
            "s3": "unknown",
            "dynamodb": "unknown"
        }
        
        # Start time
        self.start_time = datetime.utcnow()
    
    def check_budget(self, current_cost: float) -> bool:
        """
        Check if budget is within limits and send alerts
        
        Args:
            current_cost: Current total cost in USD
        
        Returns:
            bool: True if within budget, False if exceeded
        """
        if self.budget_limit <= 0:
            return True  # No budget limit set
        
        usage_percentage = (current_cost / self.budget_limit)
        
        # Check alert thresholds
        for threshold, alerted in sorted(self.budget_alerts.items()):
            if usage_percentage >= threshold and not alerted:
                self._send_budget_alert(threshold, current_cost)
                self.budget_alerts[threshold] = True
        
        # Return whether within budget
        return current_cost <= self.budget_limit
    
    def _send_budget_alert(self, threshold: float, current_cost: float):
        """Send budget alert"""
        percentage = threshold * 100
        
        if threshold >= 1.00:
            logger.critical(
                f"🚨 BUDGET EXCEEDED! "
                f"Current: ${current_cost:.2f} / Limit: ${self.budget_limit:.2f}"
            )
        elif threshold >= 0.90:
            logger.error(
                f"⚠️  CRITICAL: {percentage:.0f}% of budget used! "
                f"Current: ${current_cost:.2f} / Limit: ${self.budget_limit:.2f}"
            )
        else:
            logger.warning(
                f"⚠️  WARNING: {percentage:.0f}% of budget used. "
                f"Current: ${current_cost:.2f} / Limit: ${self.budget_limit:.2f}"
            )
    
    def track_error(self, stage: str, error_message: str = ""):
        """
        Track an error occurrence
        
        Args:
            stage: Pipeline stage where error occurred
            error_message: Error message (optional)
        """
        timestamp = datetime.utcnow()
        self.errors_by_stage[stage].append(timestamp)
        self.total_errors += 1
        
        logger.debug(f"Error tracked in {stage}: {error_message}")
        
        # Check error rate
        self._check_error_rate()
    
    def track_operation(self, stage: str):
        """
        Track a successful operation
        
        Args:
            stage: Pipeline stage
        """
        self.total_operations += 1
    
    def _check_error_rate(self):
        """Check if error rate exceeds threshold"""
        if self.total_operations == 0:
            return
        
        error_rate = self.total_errors / self.total_operations
        
        if error_rate > self.error_rate_threshold:
            logger.warning(
                f"⚠️  High error rate detected: {error_rate:.1%} "
                f"({self.total_errors} errors / {self.total_operations} operations)"
            )
    
    def track_api_call(self, model_name: str):
        """
        Track API call for rate limiting
        
        Args:
            model_name: Model name (e.g., "gemini_2.5_pro")
        """
        timestamp = datetime.utcnow()
        
        # Add to deque
        self.api_calls_by_model[model_name].append(timestamp)
        
        # Remove old calls outside window
        cutoff = timestamp - timedelta(seconds=self.rate_limit_window)
        
        while (self.api_calls_by_model[model_name] and 
               self.api_calls_by_model[model_name][0] < cutoff):
            self.api_calls_by_model[model_name].popleft()
    
    def check_rate_limit(self, model_name: str, limit: int = 100) -> bool:
        """
        Check if within rate limit for model
        
        Args:
            model_name: Model name
            limit: Max calls per rate_limit_window
        
        Returns:
            bool: True if within limit, False if exceeded
        """
        calls_in_window = len(self.api_calls_by_model[model_name])
        
        if calls_in_window >= limit:
            logger.warning(
                f"⚠️  Rate limit warning for {model_name}: "
                f"{calls_in_window} calls in last {self.rate_limit_window}s"
            )
            return False
        
        return True
    
    def track_processing_time(self, seconds: float):
        """
        Track processing time for a question
        
        Args:
            seconds: Processing time in seconds
        """
        self.processing_times.append(seconds)
    
    def get_avg_processing_time(self) -> float:
        """Get average processing time for recent questions"""
        if not self.processing_times:
            return 0.0
        
        return sum(self.processing_times) / len(self.processing_times)
    
    def estimate_completion_time(self, questions_remaining: int) -> float:
        """
        Estimate time to completion
        
        Args:
            questions_remaining: Number of questions left to process
        
        Returns:
            float: Estimated seconds to completion
        """
        avg_time = self.get_avg_processing_time()
        
        if avg_time == 0:
            return 0.0
        
        return questions_remaining * avg_time
    
    def update_service_status(self, service: str, status: str):
        """
        Update service health status
        
        Args:
            service: Service name (redis, qdrant, s3, dynamodb)
            status: Status (healthy, degraded, down, unknown)
        """
        if service in self.service_status:
            old_status = self.service_status[service]
            self.service_status[service] = status
            
            if old_status != status:
                logger.info(f"Service {service} status: {old_status} → {status}")
    
    def check_service_health(self, clients: Dict) -> Dict[str, str]:
        """
        Check health of all external services
        
        Args:
            clients: Dict of initialized clients
        
        Returns:
            dict: {service_name: status}
        """
        # Check Redis
        if 'redis' in clients:
            try:
                clients['redis'].ping()
                self.update_service_status('redis', 'healthy')
            except Exception as e:
                logger.error(f"Redis health check failed: {e}")
                self.update_service_status('redis', 'down')
        
        # Check Qdrant
        if 'qdrant' in clients:
            try:
                clients['qdrant'].get_collections()
                self.update_service_status('qdrant', 'healthy')
            except Exception as e:
                logger.error(f"Qdrant health check failed: {e}")
                self.update_service_status('qdrant', 'down')
        
        # Check S3
        if 's3' in clients:
            try:
                # Try to list buckets or head bucket
                clients['s3'].list_buckets()
                self.update_service_status('s3', 'healthy')
            except Exception as e:
                logger.error(f"S3 health check failed: {e}")
                self.update_service_status('s3', 'down')
        
        # Check DynamoDB
        if 'dynamodb' in clients:
            try:
                clients['dynamodb'].list_tables()
                self.update_service_status('dynamodb', 'healthy')
            except Exception as e:
                logger.error(f"DynamoDB health check failed: {e}")
                self.update_service_status('dynamodb', 'down')
        
        return self.service_status
    
    def get_health_report(self) -> Dict[str, Any]:
        """
        Generate comprehensive health report
        
        Returns:
            dict: Health status and metrics
        """
        # Calculate uptime
        uptime_seconds = (datetime.utcnow() - self.start_time).total_seconds()
        
        # Calculate error rate
        error_rate = (
            self.total_errors / self.total_operations 
            if self.total_operations > 0 
            else 0.0
        )
        
        # Budget status
        current_cost = self.cost_tracker.get_total_cost() if self.cost_tracker else 0.0
        budget_usage = (current_cost / self.budget_limit) if self.budget_limit > 0 else 0.0
        
        # Determine overall health status
        if budget_usage >= 1.0:
            overall_status = "critical"
        elif error_rate > self.error_rate_threshold:
            overall_status = "warning"
        elif any(s == "down" for s in self.service_status.values()):
            overall_status = "degraded"
        else:
            overall_status = "healthy"
        
        report = {
            "overall_status": overall_status,
            "timestamp": datetime.utcnow().isoformat(),
            "uptime_seconds": round(uptime_seconds, 1),
            "budget": {
                "current_cost": round(current_cost, 4),
                "budget_limit": self.budget_limit,
                "usage_percentage": round(budget_usage * 100, 2),
                "remaining": round(self.budget_limit - current_cost, 4),
                "status": "exceeded" if budget_usage >= 1.0 else 
                         "critical" if budget_usage >= 0.90 else
                         "warning" if budget_usage >= 0.75 else "ok"
            },
            "errors": {
                "total_errors": self.total_errors,
                "total_operations": self.total_operations,
                "error_rate": round(error_rate, 4),
                "threshold": self.error_rate_threshold,
                "by_stage": {
                    stage: len(errors)
                    for stage, errors in self.errors_by_stage.items()
                }
            },
            "performance": {
                "avg_processing_time_seconds": round(self.get_avg_processing_time(), 2),
                "recent_samples": len(self.processing_times)
            },
            "services": self.service_status,
            "rate_limits": {
                model: len(calls)
                for model, calls in self.api_calls_by_model.items()
            }
        }
        
        return report
    
    def print_health_report(self):
        """Print formatted health report to console"""
        report = self.get_health_report()
        
        print("\n" + "="*80)
        print("HEALTH REPORT")
        print("="*80)
        
        # Overall status
        status_emoji = {
            "healthy": "✅",
            "warning": "⚠️",
            "degraded": "⚠️",
            "critical": "🚨"
        }
        
        print(f"\nOverall Status: {status_emoji.get(report['overall_status'], '❓')} {report['overall_status'].upper()}")
        print(f"Uptime: {report['uptime_seconds']/60:.1f} minutes")
        
        # Budget
        budget = report['budget']
        print(f"\nBudget:")
        print(f"  Current: ${budget['current_cost']:.4f} / ${budget['budget_limit']:.2f}")
        print(f"  Usage: {budget['usage_percentage']:.2f}%")
        print(f"  Remaining: ${budget['remaining']:.4f}")
        print(f"  Status: {budget['status'].upper()}")
        
        # Errors
        errors = report['errors']
        print(f"\nErrors:")
        print(f"  Total: {errors['total_errors']} / {errors['total_operations']} operations")
        print(f"  Error Rate: {errors['error_rate']:.2%}")
        
        if errors['by_stage']:
            print(f"  By Stage:")
            for stage, count in sorted(errors['by_stage'].items(), key=lambda x: x[1], reverse=True):
                print(f"    - {stage}: {count}")
        
        # Performance
        perf = report['performance']
        print(f"\nPerformance:")
        print(f"  Avg Processing Time: {perf['avg_processing_time_seconds']:.2f}s")
        
        # Services
        print(f"\nServices:")
        for service, status in report['services'].items():
            status_marker = "✅" if status == "healthy" else "❌" if status == "down" else "❓"
            print(f"  {status_marker} {service}: {status}")
        
        print("="*80 + "\n")
    
    def reset(self):
        """Reset all health monitoring data"""
        self.budget_alerts = {k: False for k in self.budget_alerts}
        self.errors_by_stage.clear()
        self.total_errors = 0
        self.total_operations = 0
        self.api_calls_by_model.clear()
        self.processing_times.clear()
        self.start_time = datetime.utcnow()
        
        logger.info("Health monitor reset")


# Example usage
if __name__ == "__main__":
    from utils.cost_tracker import CostTracker
    
    # Mock cost tracker
    class MockCostTracker:
        def __init__(self):
            self.cost = 0.0
        
        def get_total_cost(self):
            return self.cost
        
        def add_cost(self, amount):
            self.cost += amount
    
    tracker = MockCostTracker()
    monitor = HealthMonitor(budget_limit=100.0, cost_tracker=tracker)
    
    # Simulate pipeline execution
    print("Simulating pipeline execution...\n")
    
    for i in range(150):
        # Track operation
        monitor.track_operation("stage_5")
        
        # Simulate cost increase
        tracker.add_cost(0.75)
        
        # Check budget
        if not monitor.check_budget(tracker.get_total_cost()):
            print("Budget exceeded - would stop here")
            break
        
        # Simulate errors (10% error rate)
        if i % 10 == 0:
            monitor.track_error("stage_5", "Timeout")
        
        # Track API calls
        monitor.track_api_call("gemini_2.5_pro")
        
        # Track processing time
        monitor.track_processing_time(5.5)
    
    # Print health report
    monitor.print_health_report()