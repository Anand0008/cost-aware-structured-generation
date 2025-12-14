#!/usr/bin/env python3
"""
============================================================================
CHECKPOINT MANAGER - SCRIPT WRAPPER
============================================================================
Purpose: CLI wrapper for checkpoint management operations
Stage: Production utility
Usage: python pipeline/scripts/13_checkpoint_manager.py [command] [options]
============================================================================
"""

import os
import sys
import json
import argparse
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, Any

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.append(str(PROJECT_ROOT))

from pipeline.utils.checkpoint_manager import CheckpointManager
from pipeline.utils.logging_utils import setup_logger

# Setup logger
logger = setup_logger(__name__)


def save_checkpoint_command(args):
    """Save current pipeline state to checkpoint"""
    logger.info("Saving checkpoint...")
    
    # Initialize checkpoint manager
    checkpoint_dir = args.checkpoint_dir or os.getenv('CHECKPOINT_DIR', 'checkpoints')
    manager = CheckpointManager(checkpoint_dir=checkpoint_dir)
    
    # Load state from file if provided
    if args.state_file:
        logger.info(f"Loading state from: {args.state_file}")
        with open(args.state_file, 'r') as f:
            state = json.load(f)
    else:
        # Create minimal state
        state = {
            'current_index': args.current_index or 0,
            'processed_questions': args.processed_questions or [],
            'total_cost': args.total_cost or 0.0,
            'timestamp': datetime.utcnow().isoformat()
        }
    
    # Save checkpoint
    checkpoint_path = manager.save_checkpoint(state, name=args.name)
    
    logger.info(f"✓ Checkpoint saved: {checkpoint_path}")
    print(f"Checkpoint saved: {checkpoint_path}")
    
    return 0


def load_checkpoint_command(args):
    """Load checkpoint and display or export"""
    logger.info("Loading checkpoint...")
    
    # Initialize checkpoint manager
    checkpoint_dir = args.checkpoint_dir or os.getenv('CHECKPOINT_DIR', 'checkpoints')
    manager = CheckpointManager(checkpoint_dir=checkpoint_dir)
    
    # Load checkpoint
    state = manager.load_checkpoint(name=args.name)
    
    if state is None:
        logger.error("No checkpoint found")
        print("❌ No checkpoint found")
        return 1
    
    logger.info("✓ Checkpoint loaded")
    
    # Display or export
    if args.output:
        # Export to file
        with open(args.output, 'w') as f:
            json.dump(state, f, indent=2)
        logger.info(f"✓ Checkpoint exported to: {args.output}")
        print(f"Checkpoint exported: {args.output}")
    else:
        # Display to console
        print("\n" + "="*80)
        print("CHECKPOINT STATE")
        print("="*80)
        print(json.dumps(state, indent=2))
        print("="*80 + "\n")
    
    return 0


def list_checkpoints_command(args):
    """List all available checkpoints"""
    logger.info("Listing checkpoints...")
    
    # Initialize checkpoint manager
    checkpoint_dir = args.checkpoint_dir or os.getenv('CHECKPOINT_DIR', 'checkpoints')
    manager = CheckpointManager(checkpoint_dir=checkpoint_dir)
    
    # Get all checkpoints
    checkpoints = manager.list_checkpoints()
    
    if not checkpoints:
        print("No checkpoints found")
        return 0
    
    # Display table
    print("\n" + "="*100)
    print(f"{'NAME':<30} {'TIMESTAMP':<25} {'QUESTIONS':<15} {'COST':<10} {'SIZE':<10}")
    print("="*100)
    
    for cp in checkpoints:
        name = cp['name']
        timestamp = cp.get('timestamp', 'N/A')
        questions = len(cp.get('processed_questions', []))
        cost = f"${cp.get('total_cost', 0.0):.2f}"
        size = f"{cp.get('size_bytes', 0) / 1024:.1f} KB"
        
        print(f"{name:<30} {timestamp:<25} {questions:<15} {cost:<10} {size:<10}")
    
    print("="*100 + "\n")
    print(f"Total checkpoints: {len(checkpoints)}")
    
    return 0


def delete_checkpoint_command(args):
    """Delete a checkpoint"""
    logger.info(f"Deleting checkpoint: {args.name}")
    
    # Initialize checkpoint manager
    checkpoint_dir = args.checkpoint_dir or os.getenv('CHECKPOINT_DIR', 'checkpoints')
    manager = CheckpointManager(checkpoint_dir=checkpoint_dir)
    
    # Confirm deletion
    if not args.yes:
        response = input(f"Delete checkpoint '{args.name}'? (y/N): ")
        if response.lower() != 'y':
            print("Deletion cancelled")
            return 0
    
    # Delete checkpoint
    success = manager.delete_checkpoint(name=args.name)
    
    if success:
        logger.info(f"✓ Checkpoint deleted: {args.name}")
        print(f"✓ Checkpoint deleted: {args.name}")
        return 0
    else:
        logger.error(f"Failed to delete checkpoint: {args.name}")
        print(f"❌ Failed to delete checkpoint: {args.name}")
        return 1


def clean_old_checkpoints_command(args):
    """Clean old checkpoints (keep latest N)"""
    logger.info(f"Cleaning old checkpoints (keep latest {args.keep})...")
    
    # Initialize checkpoint manager
    checkpoint_dir = args.checkpoint_dir or os.getenv('CHECKPOINT_DIR', 'checkpoints')
    manager = CheckpointManager(checkpoint_dir=checkpoint_dir)
    
    # Get all checkpoints
    checkpoints = manager.list_checkpoints()
    
    if len(checkpoints) <= args.keep:
        print(f"Only {len(checkpoints)} checkpoints found. Nothing to clean.")
        return 0
    
    # Sort by timestamp (newest first)
    checkpoints.sort(key=lambda x: x.get('timestamp', ''), reverse=True)
    
    # Delete old ones
    to_delete = checkpoints[args.keep:]
    
    # Confirm deletion
    if not args.yes:
        print(f"\nWill delete {len(to_delete)} checkpoints:")
        for cp in to_delete:
            print(f"  - {cp['name']} ({cp.get('timestamp', 'N/A')})")
        response = input("\nProceed? (y/N): ")
        if response.lower() != 'y':
            print("Deletion cancelled")
            return 0
    
    # Delete
    deleted = 0
    for cp in to_delete:
        if manager.delete_checkpoint(name=cp['name']):
            deleted += 1
            logger.info(f"Deleted: {cp['name']}")
    
    logger.info(f"✓ Cleaned {deleted} old checkpoints")
    print(f"✓ Cleaned {deleted} old checkpoints (kept latest {args.keep})")
    
    return 0


def info_command(args):
    """Show checkpoint system information"""
    checkpoint_dir = args.checkpoint_dir or os.getenv('CHECKPOINT_DIR', 'checkpoints')
    manager = CheckpointManager(checkpoint_dir=checkpoint_dir)
    
    checkpoints = manager.list_checkpoints()
    
    # Calculate stats
    total_checkpoints = len(checkpoints)
    total_size = sum(cp.get('size_bytes', 0) for cp in checkpoints)
    total_questions = sum(len(cp.get('processed_questions', [])) for cp in checkpoints)
    
    # Display info
    print("\n" + "="*80)
    print("CHECKPOINT SYSTEM INFO")
    print("="*80)
    print(f"Checkpoint directory: {checkpoint_dir}")
    print(f"Total checkpoints:    {total_checkpoints}")
    print(f"Total size:           {total_size / 1024 / 1024:.2f} MB")
    print(f"Total questions:      {total_questions}")
    
    if checkpoints:
        latest = max(checkpoints, key=lambda x: x.get('timestamp', ''))
        print(f"Latest checkpoint:    {latest['name']}")
        print(f"Latest timestamp:     {latest.get('timestamp', 'N/A')}")
    
    print("="*80 + "\n")
    
    return 0


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description='Checkpoint Manager - Manage pipeline checkpoints',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Save checkpoint
  python pipeline/scripts/13_checkpoint_manager.py save --current-index 150 --total-cost 45.50
  
  # Save from state file
  python pipeline/scripts/13_checkpoint_manager.py save --state-file state.json --name checkpoint_150
  
  # Load checkpoint
  python pipeline/scripts/13_checkpoint_manager.py load
  
  # Load specific checkpoint
  python pipeline/scripts/13_checkpoint_manager.py load --name checkpoint_100
  
  # Export checkpoint to file
  python pipeline/scripts/13_checkpoint_manager.py load --output state.json
  
  # List all checkpoints
  python pipeline/scripts/13_checkpoint_manager.py list
  
  # Delete checkpoint
  python pipeline/scripts/13_checkpoint_manager.py delete --name checkpoint_50
  
  # Clean old checkpoints (keep latest 5)
  python pipeline/scripts/13_checkpoint_manager.py clean --keep 5
  
  # Show system info
  python pipeline/scripts/13_checkpoint_manager.py info
        """
    )
    
    # Global options
    parser.add_argument(
        '--checkpoint-dir',
        help='Checkpoint directory (default: checkpoints/)'
    )
    
    # Subcommands
    subparsers = parser.add_subparsers(dest='command', help='Command to execute')
    
    # Save command
    save_parser = subparsers.add_parser('save', help='Save checkpoint')
    save_parser.add_argument('--state-file', help='Load state from JSON file')
    save_parser.add_argument('--name', help='Checkpoint name')
    save_parser.add_argument('--current-index', type=int, help='Current question index')
    save_parser.add_argument('--total-cost', type=float, help='Total cost so far')
    save_parser.add_argument('--processed-questions', nargs='+', help='List of processed question IDs')
    
    # Load command
    load_parser = subparsers.add_parser('load', help='Load checkpoint')
    load_parser.add_argument('--name', help='Checkpoint name (default: latest)')
    load_parser.add_argument('--output', help='Export to JSON file')
    
    # List command
    list_parser = subparsers.add_parser('list', help='List checkpoints')
    
    # Delete command
    delete_parser = subparsers.add_parser('delete', help='Delete checkpoint')
    delete_parser.add_argument('--name', required=True, help='Checkpoint name')
    delete_parser.add_argument('--yes', '-y', action='store_true', help='Skip confirmation')
    
    # Clean command
    clean_parser = subparsers.add_parser('clean', help='Clean old checkpoints')
    clean_parser.add_argument('--keep', type=int, default=5, help='Number of checkpoints to keep (default: 5)')
    clean_parser.add_argument('--yes', '-y', action='store_true', help='Skip confirmation')
    
    # Info command
    info_parser = subparsers.add_parser('info', help='Show checkpoint system info')
    
    # Parse arguments
    args = parser.parse_args()
    
    # Execute command
    if args.command == 'save':
        return save_checkpoint_command(args)
    elif args.command == 'load':
        return load_checkpoint_command(args)
    elif args.command == 'list':
        return list_checkpoints_command(args)
    elif args.command == 'delete':
        return delete_checkpoint_command(args)
    elif args.command == 'clean':
        return clean_old_checkpoints_command(args)
    elif args.command == 'info':
        return info_command(args)
    else:
        parser.print_help()
        return 1


if __name__ == "__main__":
    sys.exit(main())