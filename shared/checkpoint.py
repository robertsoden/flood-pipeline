"""
Checkpoint utilities for resumable pipeline processing.

Provides save/load functionality for intermediate results to enable
resuming long-running pipeline stages after interruptions.
"""
import json
from pathlib import Path
from typing import List, Dict, Any, Optional, Set
from datetime import datetime
import logging

logger = logging.getLogger(__name__)


class CheckpointManager:
    """
    Manages checkpoints for resumable processing.

    Saves processed article IDs and results periodically to enable
    resuming from the last checkpoint after interruptions.

    Example:
        >>> checkpoint = CheckpointManager('stage2', project_root / 'checkpoints')
        >>> checkpoint.load()  # Load existing progress
        >>> for article in articles:
        ...     if checkpoint.is_processed(article['article_id']):
        ...         continue  # Skip already processed
        ...     result = process(article)
        ...     checkpoint.mark_processed(article['article_id'], result)
        ...     if checkpoint.should_save():
        ...         checkpoint.save()
        >>> checkpoint.save()  # Final save
    """

    def __init__(
        self,
        stage_name: str,
        checkpoint_dir: Path,
        save_interval: int = 100,
        max_checkpoints: int = 5
    ):
        """
        Initialize checkpoint manager.

        Args:
            stage_name: Name of the pipeline stage (e.g., 'stage2', 'stage3_llm')
            checkpoint_dir: Directory to store checkpoint files
            save_interval: Save checkpoint every N processed items
            max_checkpoints: Maximum number of checkpoint files to keep
        """
        self.stage_name = stage_name
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.save_interval = save_interval
        self.max_checkpoints = max_checkpoints

        # State
        self.processed_ids: Set[str] = set()
        self.results: List[Dict[str, Any]] = []
        self.items_since_save = 0
        self.start_time = datetime.now()

    def _get_checkpoint_path(self, timestamp: Optional[str] = None) -> Path:
        """Get path for checkpoint file."""
        if timestamp:
            return self.checkpoint_dir / f"{self.stage_name}_checkpoint_{timestamp}.json"
        return self.checkpoint_dir / f"{self.stage_name}_checkpoint_latest.json"

    def _get_latest_checkpoint(self) -> Optional[Path]:
        """Find the most recent checkpoint file."""
        pattern = f"{self.stage_name}_checkpoint_*.json"
        checkpoints = list(self.checkpoint_dir.glob(pattern))

        if not checkpoints:
            return None

        # Sort by modification time, most recent first
        checkpoints.sort(key=lambda p: p.stat().st_mtime, reverse=True)
        return checkpoints[0]

    def load(self) -> bool:
        """
        Load the most recent checkpoint.

        Returns:
            True if checkpoint was loaded, False if no checkpoint exists
        """
        checkpoint_path = self._get_latest_checkpoint()

        if not checkpoint_path or not checkpoint_path.exists():
            logger.info(f"No existing checkpoint found for {self.stage_name}")
            return False

        try:
            with open(checkpoint_path, 'r') as f:
                data = json.load(f)

            self.processed_ids = set(data.get('processed_ids', []))
            self.results = data.get('results', [])

            logger.info(
                f"Loaded checkpoint: {len(self.processed_ids)} processed items, "
                f"{len(self.results)} results"
            )
            return True

        except Exception as e:
            logger.error(f"Error loading checkpoint: {e}")
            return False

    def save(self, force: bool = False) -> Path:
        """
        Save current progress to checkpoint file.

        Args:
            force: Save even if save_interval not reached

        Returns:
            Path to saved checkpoint file
        """
        if not force and self.items_since_save < self.save_interval:
            return None

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        checkpoint_path = self._get_checkpoint_path(timestamp)

        data = {
            'stage_name': self.stage_name,
            'timestamp': timestamp,
            'processed_ids': list(self.processed_ids),
            'results': self.results,
            'stats': {
                'total_processed': len(self.processed_ids),
                'total_results': len(self.results),
                'start_time': self.start_time.isoformat(),
                'save_time': datetime.now().isoformat()
            }
        }

        with open(checkpoint_path, 'w') as f:
            json.dump(data, f, indent=2)

        # Also save as 'latest'
        latest_path = self._get_checkpoint_path()
        with open(latest_path, 'w') as f:
            json.dump(data, f, indent=2)

        self.items_since_save = 0

        # Cleanup old checkpoints
        self._cleanup_old_checkpoints()

        logger.info(f"Checkpoint saved: {checkpoint_path}")
        return checkpoint_path

    def _cleanup_old_checkpoints(self):
        """Remove old checkpoint files, keeping only max_checkpoints most recent."""
        pattern = f"{self.stage_name}_checkpoint_2*.json"  # Timestamped files
        checkpoints = list(self.checkpoint_dir.glob(pattern))

        if len(checkpoints) <= self.max_checkpoints:
            return

        # Sort by modification time, oldest first
        checkpoints.sort(key=lambda p: p.stat().st_mtime)

        # Remove oldest files
        for checkpoint in checkpoints[:-self.max_checkpoints]:
            try:
                checkpoint.unlink()
                logger.debug(f"Removed old checkpoint: {checkpoint}")
            except Exception as e:
                logger.warning(f"Failed to remove old checkpoint {checkpoint}: {e}")

    def is_processed(self, article_id: str) -> bool:
        """Check if an article has already been processed."""
        return article_id in self.processed_ids

    def mark_processed(self, article_id: str, result: Optional[Dict[str, Any]] = None):
        """
        Mark an article as processed and optionally store its result.

        Args:
            article_id: Unique article identifier
            result: Optional result dictionary to store
        """
        self.processed_ids.add(article_id)
        if result is not None:
            result['article_id'] = article_id  # Ensure ID is in result
            self.results.append(result)
        self.items_since_save += 1

    def should_save(self) -> bool:
        """Check if it's time to save a checkpoint."""
        return self.items_since_save >= self.save_interval

    def get_results(self) -> List[Dict[str, Any]]:
        """Get all stored results."""
        return self.results

    def get_stats(self) -> Dict[str, Any]:
        """Get processing statistics."""
        elapsed = (datetime.now() - self.start_time).total_seconds()
        rate = len(self.processed_ids) / elapsed * 60 if elapsed > 0 else 0

        return {
            'total_processed': len(self.processed_ids),
            'total_results': len(self.results),
            'elapsed_minutes': elapsed / 60,
            'rate_per_minute': rate
        }

    def clear(self):
        """Clear all checkpoint data (for fresh start)."""
        self.processed_ids.clear()
        self.results.clear()
        self.items_since_save = 0

        # Remove checkpoint files
        for path in self.checkpoint_dir.glob(f"{self.stage_name}_checkpoint_*.json"):
            try:
                path.unlink()
            except Exception:
                pass

        logger.info(f"Cleared all checkpoints for {self.stage_name}")


def filter_unprocessed(
    articles: List[Dict[str, Any]],
    checkpoint: CheckpointManager
) -> List[Dict[str, Any]]:
    """
    Filter articles to only those not yet processed.

    Args:
        articles: List of articles with article_id field
        checkpoint: CheckpointManager instance

    Returns:
        List of unprocessed articles
    """
    unprocessed = []
    for article in articles:
        article_id = article.get('article_id')
        if article_id and not checkpoint.is_processed(article_id):
            unprocessed.append(article)

    skipped = len(articles) - len(unprocessed)
    if skipped > 0:
        logger.info(f"Skipping {skipped} already-processed articles")

    return unprocessed
