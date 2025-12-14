#!/usr/bin/env python3
"""
============================================================================
HEALTH MONITOR - SCRIPT WRAPPER
============================================================================
Purpose: CLI wrapper for health monitoring and system checks
Stage: Production utility
Usage: python pipeline/scripts/14_health_monitor.py [command] [options]
============================================================================
"""

import os
import sys
import json
import time
import argparse
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, Any

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.append(str(PROJECT_ROOT))

from pipeline.utils.health_monitor import HealthMonitor
from pipeline.utils.logging_utils import setup_logger
from pipeline.scripts.init_00_initialization import PipelineInitializer

# Setup logger
logger = setup_logger(__name__)


def check_command(args):
    """Run complete health check"""
    logger.info("Running health check...")
    
    # Initialize pipeline components
    try:
        initializer = PipelineInitializer()
        components = initializer.initialize_all()
    except Exception as e:
        logger.error(f"Failed to initialize components: {e}")
        print(f"❌ Failed to initialize: {e}")
        return 1
    
    # Create health monitor
    monitor = HealthMonitor(components=components)
    
    # Run health check
    health_status = monitor.check_health()
    
    # Display results
    print("\n" + "="*80)
    print("HEALTH CHECK RESULTS")
    print("="*80)
    
    overall_status = health_status.get('status', 'unknown')
    
    if overall_status == 'healthy':
        print("✓ Overall Status: HEALTHY")
    elif overall_status == 'degraded':
        print("⚠ Overall Status: DEGRADED")
    else:
        print("✗ Overall Status: UNHEALTHY")
    
    print(f"Timestamp: {health_status.get('timestamp', 'N/A')}")
    print("-" * 80)
    
    # Component checks
    checks = health_status.get('checks', {})
    
    for component, status in checks.items():
        status_symbol = "✓" if status.get('status') == 'healthy' else "✗"
        status_text = status.get('status', 'unknown').upper()
        message = status.get('message', '')
        
        print(f"{status_symbol} {component:<30} {status_text:<15} {message}")
    
    print("="*80 + "\n")
    
    # Save to file if requested
    if args.output:
        with open(args.output, 'w') as f:
            json.dump(health_status, f, indent=2)
        logger.info(f"Health report saved to: {args.output}")
        print(f"Report saved: {args.output}")
    
    # Return appropriate exit code
    return 0 if overall_status == 'healthy' else 1


def api_check_command(args):
    """Check API connectivity"""
    logger.info("Checking API connectivity...")
    
    try:
        initializer = PipelineInitializer()
        components = initializer.initialize_all()
    except Exception as e:
        logger.error(f"Failed to initialize: {e}")
        print(f"❌ Failed to initialize: {e}")
        return 1
    
    monitor = HealthMonitor(components=components)
    
    # Check APIs
    api_status = monitor.check_api_health()
    
    # Display results
    print("\n" + "="*80)
    print("API CONNECTIVITY CHECK")
    print("="*80)
    
    for api, status in api_status.items():
        is_healthy = status.get('status') == 'healthy'
        symbol = "✓" if is_healthy else "✗"
        response_time = status.get('response_time_ms', 'N/A')
        message = status.get('message', '')
        
        print(f"{symbol} {api:<25} {response_time:>10} ms   {message}")
    
    print("="*80 + "\n")
    
    return 0


def database_check_command(args):
    """Check database connectivity"""
    logger.info("Checking database connectivity...")
    
    try:
        initializer = PipelineInitializer()
        components = initializer.initialize_all()
    except Exception as e:
        logger.error(f"Failed to initialize: {e}")
        print(f"❌ Failed to initialize: {e}")
        return 1
    
    monitor = HealthMonitor(components=components)
    
    # Check databases
    db_status = monitor.check_database_health()
    
    # Display results
    print("\n" + "="*80)
    print("DATABASE CONNECTIVITY CHECK")
    print("="*80)
    
    for db, status in db_status.items():
        is_healthy = status.get('status') == 'healthy'
        symbol = "✓" if is_healthy else "✗"
        response_time = status.get('response_time_ms', 'N/A')
        message = status.get('message', '')
        
        print(f"{symbol} {db:<25} {response_time:>10} ms   {message}")
    
    print("="*80 + "\n")
    
    return 0


def model_check_command(args):
    """Check model availability"""
    logger.info("Checking model availability...")
    
    try:
        initializer = PipelineInitializer()
        components = initializer.initialize_all()
    except Exception as e:
        logger.error(f"Failed to initialize: {e}")
        print(f"❌ Failed to initialize: {e}")
        return 1
    
    monitor = HealthMonitor(components=components)
    
    # Check models
    model_status = monitor.check_model_health()
    
    # Display results
    print("\n" + "="*80)
    print("MODEL AVAILABILITY CHECK")
    print("="*80)
    
    for model, status in model_status.items():
        is_healthy = status.get('status') == 'healthy'
        symbol = "✓" if is_healthy else "✗"
        message = status.get('message', '')
        size = status.get('size_mb', 'N/A')
        
        print(f"{symbol} {model:<25} {size:>10} MB   {message}")
    
    print("="*80 + "\n")
    
    return 0


def watch_command(args):
    """Continuous health monitoring"""
    logger.info(f"Starting health monitoring (interval: {args.interval}s)...")
    
    try:
        initializer = PipelineInitializer()
        components = initializer.initialize_all()
    except Exception as e:
        logger.error(f"Failed to initialize: {e}")
        print(f"❌ Failed to initialize: {e}")
        return 1
    
    monitor = HealthMonitor(components=components)
    
    print(f"\nHealth monitoring started (checking every {args.interval}s)")
    print("Press Ctrl+C to stop\n")
    
    try:
        iteration = 0
        while True:
            iteration += 1
            
            # Run health check
            health_status = monitor.check_health()
            overall_status = health_status.get('status', 'unknown')
            
            # Display compact status
            timestamp = datetime.now().strftime('%H:%M:%S')
            
            if overall_status == 'healthy':
                print(f"[{timestamp}] ✓ HEALTHY (iteration {iteration})")
            elif overall_status == 'degraded':
                print(f"[{timestamp}] ⚠ DEGRADED (iteration {iteration})")
                # Show which components are degraded
                checks = health_status.get('checks', {})
                for component, status in checks.items():
                    if status.get('status') != 'healthy':
                        print(f"         └─ {component}: {status.get('message', '')}")
            else:
                print(f"[{timestamp}] ✗ UNHEALTHY (iteration {iteration})")
                # Show all failed components
                checks = health_status.get('checks', {})
                for component, status in checks.items():
                    if status.get('status') != 'healthy':
                        print(f"         └─ {component}: {status.get('message', '')}")
            
            # Wait for next iteration
            time.sleep(args.interval)
    
    except KeyboardInterrupt:
        print("\n\nHealth monitoring stopped")
        return 0


def cost_check_command(args):
    """Check current cost status"""
    logger.info("Checking cost status...")
    
    try:
        initializer = PipelineInitializer()
        components = initializer.initialize_all()
    except Exception as e:
        logger.error(f"Failed to initialize: {e}")
        print(f"❌ Failed to initialize: {e}")
        return 1
    
    monitor = HealthMonitor(components=components)
    
    # Get cost status
    cost_status = monitor.check_cost_status()
    
    # Display results
    print("\n" + "="*80)
    print("COST STATUS")
    print("="*80)
    
    total_cost = cost_status.get('total_cost', 0.0)
    budget_limit = cost_status.get('budget_limit', 420.0)
    percentage = cost_status.get('percentage_used', 0.0)
    alert_level = cost_status.get('alert_level', 'normal')
    
    print(f"Total Cost:      ${total_cost:.2f}")
    print(f"Budget Limit:    ${budget_limit:.2f}")
    print(f"Percentage Used: {percentage:.1f}%")
    print(f"Alert Level:     {alert_level.upper()}")
    
    # Show alert symbol
    if alert_level == 'normal':
        print("Status:          ✓ NORMAL")
    elif alert_level == 'warning':
        print("Status:          ⚠ WARNING (>75%)")
    elif alert_level == 'critical':
        print("Status:          ⚠⚠ CRITICAL (>90%)")
    else:
        print("Status:          ✗ EXCEEDED")
    
    # Cost breakdown
    if 'breakdown' in cost_status:
        print("\nCost Breakdown:")
        for model, cost in cost_status['breakdown'].items():
            print(f"  {model:<25} ${cost:.2f}")
    
    print("="*80 + "\n")
    
    return 0


def metrics_command(args):
    """Display current metrics"""
    logger.info("Fetching metrics...")
    
    try:
        initializer = PipelineInitializer()
        components = initializer.initialize_all()
    except Exception as e:
        logger.error(f"Failed to initialize: {e}")
        print(f"❌ Failed to initialize: {e}")
        return 1
    
    monitor = HealthMonitor(components=components)
    
    # Get metrics
    metrics = monitor.get_metrics()
    
    # Display results
    print("\n" + "="*80)
    print("PIPELINE METRICS")
    print("="*80)
    
    print(f"Questions Processed:  {metrics.get('questions_processed', 0)}")
    print(f"Total Cost:           ${metrics.get('total_cost', 0.0):.2f}")
    print(f"Average Cost/Q:       ${metrics.get('avg_cost_per_question', 0.0):.2f}")
    print(f"Cache Hit Rate:       {metrics.get('cache_hit_rate', 0.0):.1f}%")
    print(f"Average Quality:      {metrics.get('avg_quality_score', 0.0):.2f}")
    print(f"GOLD Rate:            {metrics.get('gold_rate', 0.0):.1f}%")
    print(f"Error Rate:           {metrics.get('error_rate', 0.0):.1f}%")
    
    print("="*80 + "\n")
    
    return 0


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description='Health Monitor - Monitor pipeline health and status',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run complete health check
  python pipeline/scripts/14_health_monitor.py check
  
  # Save health report to file
  python pipeline/scripts/14_health_monitor.py check --output health_report.json
  
  # Check API connectivity only
  python pipeline/scripts/14_health_monitor.py api
  
  # Check database connectivity only
  python pipeline/scripts/14_health_monitor.py database
  
  # Check model availability
  python pipeline/scripts/14_health_monitor.py model
  
  # Check cost status
  python pipeline/scripts/14_health_monitor.py cost
  
  # Display metrics
  python pipeline/scripts/14_health_monitor.py metrics
  
  # Continuous monitoring (every 30 seconds)
  python pipeline/scripts/14_health_monitor.py watch --interval 30
        """
    )
    
    # Subcommands
    subparsers = parser.add_subparsers(dest='command', help='Command to execute')
    
    # Check command
    check_parser = subparsers.add_parser('check', help='Run complete health check')
    check_parser.add_argument('--output', help='Save report to JSON file')
    
    # API check command
    api_parser = subparsers.add_parser('api', help='Check API connectivity')
    
    # Database check command
    db_parser = subparsers.add_parser('database', help='Check database connectivity')
    
    # Model check command
    model_parser = subparsers.add_parser('model', help='Check model availability')
    
    # Cost check command
    cost_parser = subparsers.add_parser('cost', help='Check cost status')
    
    # Metrics command
    metrics_parser = subparsers.add_parser('metrics', help='Display metrics')
    
    # Watch command
    watch_parser = subparsers.add_parser('watch', help='Continuous monitoring')
    watch_parser.add_argument('--interval', type=int, default=30, help='Check interval in seconds (default: 30)')
    
    # Parse arguments
    args = parser.parse_args()
    
    # Execute command
    if args.command == 'check':
        return check_command(args)
    elif args.command == 'api':
        return api_check_command(args)
    elif args.command == 'database':
        return database_check_command(args)
    elif args.command == 'model':
        return model_check_command(args)
    elif args.command == 'cost':
        return cost_check_command(args)
    elif args.command == 'metrics':
        return metrics_command(args)
    elif args.command == 'watch':
        return watch_command(args)
    else:
        parser.print_help()
        return 1


if __name__ == "__main__":
    sys.exit(main())