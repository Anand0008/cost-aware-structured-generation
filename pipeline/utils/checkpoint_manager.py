"""
============================================================================
CHECKPOINT MANAGER UTILITY
============================================================================
Purpose: Save and restore pipeline state for crash recovery
Features:
    - Save checkpoints every N questions
    - Resume from last successful question
    - Preserve statistics and costs
    - Atomic checkpoint saves (all or nothing)
    - Checkpoint versioning and cleanup

Usage:
    from utils.checkpoint_manager import CheckpointManager
    
    manager = CheckpointManager(checkpoint_dir="./checkpoints")
    
    # Save checkpoint
    manager.save_checkpoint({
        'last_processed_index': 100,
        'last_question_id': 'GATE_AE_2024_Q15',
        'stats': {...}
    })
    
    # Resume from checkpoint
    checkpoint = manager.load_checkpoint()
    if checkpoint:
        start_index = checkpoint['last_processed_index'] + 1

Author: GATE AE SOTA Pipeline
============================================================================
"""

import os
import json
from pathlib import Path
from typing import Dict, Any, Optional
from datetime import datetime
import shutil

from utils.logging_utils import setup_logger

logger = setup_logger("checkpoint_manager")


class CheckpointManager:
    """
    Manage pipeline checkpoints for crash recovery
    
    Checkpoint structure:
    {
        "checkpoint_version": "1.0",
        "timestamp": "2025-12-14T10:30:00",
        "last_processed_index": 100,
        "last_question_id": "GATE_AE_2024_Q15",
        "last_question_file": "/path/to/question.json",
        "stats": {
            "total_questions": 1300,
            "processed": 100,
            "cache_hits": 20,
            "cache_misses": 80,
            "failed": 0,
            "total_cost": 25.67,
            "total_time": 450.0
        },
        "config_snapshot": {...}  # Optional: config used
    }
    """
    
    def __init__(
        self,
        checkpoint_dir: str = "./checkpoints",
        max_checkpoints: int = 5
    ):
        """
        Args:
            checkpoint_dir: Directory to store checkpoints
            max_checkpoints: Maximum number of checkpoints to keep
        """
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        self.max_checkpoints = max_checkpoints
        self.checkpoint_version = "1.0"
        
        # Current checkpoint file
        self.current_checkpoint_file = self.checkpoint_dir / "current.json"
        
        # Archive directory for old checkpoints
        self.archive_dir = self.checkpoint_dir / "archive"
        self.archive_dir.mkdir(exist_ok=True)
    
    def save_checkpoint(
        self,
        data: Dict[str, Any],
        include_config: bool = False
    ):
        """
        Save checkpoint atomically
        
        Process:
        1. Write to temporary file
        2. Archive current checkpoint (if exists)
        3. Rename temp file to current
        
        This ensures checkpoint is never partially written
        
        Args:
            data: Checkpoint data dict
            include_config: Include config snapshot in checkpoint
        """
        # Prepare checkpoint data
        checkpoint = {
            "checkpoint_version": self.checkpoint_version,
            "timestamp": datetime.utcnow().isoformat(),
            **data
        }
        
        # Validate required fields
        required_fields = ['last_processed_index', 'last_question_file', 'stats']
        missing = [f for f in required_fields if f not in data]
        
        if missing:
            raise ValueError(f"Checkpoint missing required fields: {missing}")
        
        try:
            # Write to temporary file
            temp_file = self.checkpoint_dir / "checkpoint.tmp"
            
            with open(temp_file, 'w') as f:
                json.dump(checkpoint, f, indent=2)
            
            # Archive current checkpoint if exists
            if self.current_checkpoint_file.exists():
                self._archive_current_checkpoint()
            
            # Atomically rename temp to current
            temp_file.replace(self.current_checkpoint_file)
            
            logger.info(
                f"Checkpoint saved: index={data['last_processed_index']}, "
                f"question={data.get('last_question_id', 'N/A')}"
            )
            
            # Cleanup old checkpoints
            self._cleanup_old_checkpoints()
            
        except Exception as e:
            logger.error(f"Failed to save checkpoint: {e}")
            
            # Clean up temp file if exists
            if temp_file.exists():
                temp_file.unlink()
            
            raise
    
    def load_checkpoint(self) -> Optional[Dict[str, Any]]:
        """
        Load most recent checkpoint
        
        Returns:
            dict: Checkpoint data or None if no checkpoint exists
        """
        if not self.current_checkpoint_file.exists():
            logger.info("No checkpoint found")
            return None
        
        try:
            with open(self.current_checkpoint_file, 'r') as f:
                checkpoint = json.load(f)
            
            # Validate checkpoint version
            if checkpoint.get('checkpoint_version') != self.checkpoint_version:
                logger.warning(
                    f"Checkpoint version mismatch: "
                    f"{checkpoint.get('checkpoint_version')} != {self.checkpoint_version}"
                )
            
            logger.info(
                f"Checkpoint loaded: index={checkpoint['last_processed_index']}, "
                f"timestamp={checkpoint['timestamp']}"
            )
            
            return checkpoint
            
        except Exception as e:
            logger.error(f"Failed to load checkpoint: {e}")
            return None
    
    def _archive_current_checkpoint(self):
        """Archive current checkpoint before overwriting"""
        if not self.current_checkpoint_file.exists():
            return
        
        # Generate archive filename with timestamp
        timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        archive_file = self.archive_dir / f"checkpoint_{timestamp}.json"
        
        # Copy current to archive
        shutil.copy2(self.current_checkpoint_file, archive_file)
        
        logger.debug(f"Archived checkpoint to: {archive_file}")
    
    def _cleanup_old_checkpoints(self):
        """Remove old archived checkpoints, keeping only max_checkpoints"""
        # Get all archived checkpoints
        archived = sorted(
            self.archive_dir.glob("checkpoint_*.json"),
            key=lambda p: p.stat().st_mtime,
            reverse=True  # Newest first
        )
        
        # Remove old checkpoints
        for old_checkpoint in archived[self.max_checkpoints:]:
            old_checkpoint.unlink()
            logger.debug(f"Removed old checkpoint: {old_checkpoint}")
    
    def get_checkpoint_info(self) -> Optional[Dict[str, Any]]:
        """
        Get information about current checkpoint without fully loading it
        
        Returns:
            dict: {
                "exists": bool,
                "timestamp": str,
                "last_processed_index": int,
                "last_question_id": str,
                "total_processed": int,
                "total_cost": float
            }
        """
        if not self.current_checkpoint_file.exists():
            return {
                "exists": False
            }
        
        try:
            with open(self.current_checkpoint_file, 'r') as f:
                checkpoint = json.load(f)
            
            stats = checkpoint.get('stats', {})
            
            return {
                "exists": True,
                "timestamp": checkpoint.get('timestamp'),
                "last_processed_index": checkpoint.get('last_processed_index'),
                "last_question_id": checkpoint.get('last_question_id'),
                "total_processed": stats.get('processed', 0),
                "total_cost": stats.get('total_cost', 0.0),
                "cache_hit_rate": (
                    stats.get('cache_hits', 0) / max(stats.get('processed', 1), 1) * 100
                )
            }
            
        except Exception as e:
            logger.error(f"Failed to get checkpoint info: {e}")
            return {"exists": False, "error": str(e)}
    
    def list_checkpoints(self) -> list:
        """
        List all available checkpoints (current + archived)
        
        Returns:
            list: List of checkpoint info dicts sorted by timestamp (newest first)
        """
        checkpoints = []
        
        # Current checkpoint
        if self.current_checkpoint_file.exists():
            try:
                with open(self.current_checkpoint_file, 'r') as f:
                    checkpoint = json.load(f)
                
                checkpoints.append({
                    "file": str(self.current_checkpoint_file),
                    "type": "current",
                    "timestamp": checkpoint.get('timestamp'),
                    "last_processed_index": checkpoint.get('last_processed_index'),
                    "last_question_id": checkpoint.get('last_question_id')
                })
            except:
                pass
        
        # Archived checkpoints
        for archive_file in self.archive_dir.glob("checkpoint_*.json"):
            try:
                with open(archive_file, 'r') as f:
                    checkpoint = json.load(f)
                
                checkpoints.append({
                    "file": str(archive_file),
                    "type": "archived",
                    "timestamp": checkpoint.get('timestamp'),
                    "last_processed_index": checkpoint.get('last_processed_index'),
                    "last_question_id": checkpoint.get('last_question_id')
                })
            except:
                continue
        
        # Sort by timestamp (newest first)
        checkpoints.sort(key=lambda x: x['timestamp'], reverse=True)
        
        return checkpoints
    
    def restore_from_archive(self, archive_file: str):
        """
        Restore checkpoint from archive
        
        Args:
            archive_file: Path to archived checkpoint file
        """
        archive_path = Path(archive_file)
        
        if not archive_path.exists():
            raise FileNotFoundError(f"Archive checkpoint not found: {archive_file}")
        
        # Archive current checkpoint if exists
        if self.current_checkpoint_file.exists():
            self._archive_current_checkpoint()
        
        # Copy archive to current
        shutil.copy2(archive_path, self.current_checkpoint_file)
        
        logger.info(f"Restored checkpoint from: {archive_file}")
    
    def clear_checkpoints(self, confirm: bool = False):
        """
        Clear all checkpoints (current + archived)
        
        Args:
            confirm: Must be True to actually clear
        """
        if not confirm:
            logger.warning("Clear checkpoints called without confirmation - skipping")
            return
        
        # Remove current checkpoint
        if self.current_checkpoint_file.exists():
            self.current_checkpoint_file.unlink()
            logger.info("Removed current checkpoint")
        
        # Remove all archived checkpoints
        for archive_file in self.archive_dir.glob("checkpoint_*.json"):
            archive_file.unlink()
        
        logger.info("All checkpoints cleared")
    
    def get_resume_info(self) -> Dict[str, Any]:
        """
        Get information needed to resume pipeline
        
        Returns:
            dict: {
                "can_resume": bool,
                "start_index": int,
                "last_question_file": str,
                "stats_to_restore": dict,
                "message": str
            }
        """
        checkpoint = self.load_checkpoint()
        
        if not checkpoint:
            return {
                "can_resume": False,
                "start_index": 0,
                "message": "No checkpoint found - starting from beginning"
            }
        
        # Resume from next question after last processed
        start_index = checkpoint['last_processed_index'] + 1
        
        return {
            "can_resume": True,
            "start_index": start_index,
            "last_question_file": checkpoint['last_question_file'],
            "last_question_id": checkpoint.get('last_question_id'),
            "stats_to_restore": checkpoint.get('stats', {}),
            "message": (
                f"Resuming from question {start_index} "
                f"(last processed: {checkpoint.get('last_question_id')})"
            )
        }


# Example usage
if __name__ == "__main__":
    # Create checkpoint manager
    manager = CheckpointManager(checkpoint_dir="./test_checkpoints")
    
    # Save checkpoint
    manager.save_checkpoint({
        "last_processed_index": 100,
        "last_question_id": "GATE_AE_2024_Q15",
        "last_question_file": "/data/questions/2024/q15.json",
        "stats": {
            "total_questions": 1300,
            "processed": 100,
            "cache_hits": 20,
            "cache_misses": 80,
            "failed": 0,
            "total_cost": 25.67,
            "total_time": 450.0
        }
    })
    
    # Get checkpoint info
    info = manager.get_checkpoint_info()
    print(f"\nCheckpoint Info:")
    print(json.dumps(info, indent=2))
    
    # List all checkpoints
    checkpoints = manager.list_checkpoints()
    print(f"\nAvailable Checkpoints: {len(checkpoints)}")
    for cp in checkpoints:
        print(f"  - {cp['type']}: index={cp['last_processed_index']}, time={cp['timestamp']}")
    
    # Get resume info
    resume_info = manager.get_resume_info()
    print(f"\nResume Info:")
    print(json.dumps(resume_info, indent=2))
    
    # Simulate multiple checkpoints
    import time
    for i in range(110, 160, 10):
        time.sleep(0.1)
        manager.save_checkpoint({
            "last_processed_index": i,
            "last_question_id": f"GATE_AE_2024_Q{i}",
            "last_question_file": f"/data/questions/2024/q{i}.json",
            "stats": {
                "total_questions": 1300,
                "processed": i,
                "cache_hits": i // 5,
                "cache_misses": i - (i // 5),
                "failed": 0,
                "total_cost": i * 0.25,
                "total_time": i * 4.5
            }
        })
    
    # List checkpoints again
    checkpoints = manager.list_checkpoints()
    print(f"\nCheckpoints after multiple saves: {len(checkpoints)}")
    for cp in checkpoints:
        print(f"  - {cp['type']}: index={cp['last_processed_index']}")
    
    # Clean up
    manager.clear_checkpoints(confirm=True)
    print("\nCheckpoints cleared")