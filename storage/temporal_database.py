import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
from datetime import datetime
from typing import List, Optional, Tuple, Dict, Any
from pathlib import Path
import logging

from core.data_structures import BaseSnapshot, Delta, ContentTimeline
from core.delta_computer import DeltaComputer
from core.reconstruction_service import ReconstructionResult, ReconstructionService
from storage.storage_engine import TemporalStorageEngine

logger = logging.getLogger(__name__)


class TemporalVectorDatabase:
    """
    Main interface for the sequential temporal vector database system.
    """

    def __init__(
        self,
        storage_path: str = "temporal_vector_db.h5",
        embedding_dim: int = 384,
        sparsity_threshold: float = 0.01,
        base_promotion_interval: int = 10,
        base_promotion_sparsity_threshold: float = 0.7,
    ):
        """
        Initialize the temporal vector database with sequential versioning.

        Args:
            storage_path: Path to storage file
            embedding_dim: Embedding dimension
            sparsity_threshold: Minimum change magnitude to store in deltas
            base_promotion_interval: Create base snapshot every N versions
        """
        self.storage_path = storage_path
        self.embedding_dim = embedding_dim
        self.base_promotion_interval = base_promotion_interval
        self.base_promotion_sparsity_threshold = base_promotion_sparsity_threshold

        self.storage_engine = TemporalStorageEngine(storage_path, embedding_dim)
        self.delta_computer = DeltaComputer(sparsity_threshold=sparsity_threshold)
        self.reconstruction_service = ReconstructionService(
            self.storage_engine, self.delta_computer
        )

        logger.info(
            "Initialized TemporalVectorDatabase with sequential versioning at %s",
            storage_path,
        )

    def _convert_numpy_types_in_metadata(
        self, metadata: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Clean numpy types from metadata before storage."""
        if isinstance(metadata, dict):
            return {
                key: self._convert_numpy_types_in_metadata(value)
                for key, value in metadata.items()
            }
        elif isinstance(metadata, list):
            return [self._convert_numpy_types_in_metadata(item) for item in metadata]
        elif isinstance(metadata, tuple):
            return tuple(
                self._convert_numpy_types_in_metadata(item) for item in metadata
            )
        elif isinstance(metadata, np.integer):
            return int(metadata)
        elif isinstance(metadata, np.floating):
            return float(metadata)
        elif isinstance(metadata, np.ndarray):
            return metadata.tolist()
        elif isinstance(metadata, np.bool_):
            return bool(metadata)
        elif hasattr(metadata, "item"):
            return metadata.item()
        else:
            return metadata

    def add_content_version(
        self,
        content_id: str,
        embedding: np.ndarray,
        timestamp: Optional[datetime] = None,
        metadata: Optional[Dict[str, Any]] = None,
        force_base_snapshot: bool = False,
    ) -> Tuple[bool, int]:
        """
        Add a new version of content with auto-incremented sequence number.

        Args:
            content_id: Unique content identifier
            embedding: Embedding vector
            timestamp: Version timestamp (defaults to now)
            metadata: Optional metadata
            force_base_snapshot: Force storage as base snapshot instead of delta

        Returns:
            Tuple of (success_status, assigned_sequence_number)
        """
        if timestamp is None:
            timestamp = datetime.now()

        if metadata is None:
            metadata = {}

        clean_metadata = self._convert_numpy_types_in_metadata(metadata)
        next_sequence = int(self.storage_engine.get_next_sequence_number(content_id))
        timeline = self.storage_engine.get_content_timeline(content_id)

        if force_base_snapshot or timeline is None or next_sequence == 1:
            snapshot = BaseSnapshot(
                content_id=content_id,
                sequence_number=next_sequence,
                timestamp=timestamp,
                embedding=embedding,
                metadata=clean_metadata,
            )
            success = self.storage_engine.store_base_snapshot(snapshot)
            if success:
                logger.info(
                    "Stored base snapshot for content_id=%s, sequence=%d",
                    content_id,
                    next_sequence,
                )
                return success, next_sequence

        prev_sequence = next_sequence - 1
        prev_embedding = self._get_embedding_at_sequence(timeline, prev_sequence)

        should_create_base = self._should_create_base_snapshot(
            timeline=timeline,
            sequence_number=next_sequence,
            force_base=False,
            previous_embedding=prev_embedding,
            current_embedding=embedding,
            sparsity_threshold=self.base_promotion_sparsity_threshold,
        )

        if prev_embedding is None or should_create_base:
            snapshot = BaseSnapshot(
                content_id=content_id,
                sequence_number=next_sequence,
                timestamp=timestamp,
                embedding=embedding,
                metadata=clean_metadata,
            )
            success = self.storage_engine.store_base_snapshot(snapshot)
            if success:
                logger.info(
                    "Stored base snapshot for content_id=%s, sequence=%d",
                    content_id,
                    next_sequence,
                )
            return success, next_sequence

        delta = self.delta_computer.compute_delta(
            from_embedding=prev_embedding,
            to_embedding=embedding,
            from_sequence_number=prev_sequence,
            to_sequence_number=next_sequence,
            timestamp=timestamp,
            content_id=content_id,
        )

        delta.metadata.update(clean_metadata)
        success = self.storage_engine.store_delta(delta)
        if success:
            logger.info(
                "Stored delta: %s (sequence %d)", delta.version_id, next_sequence
            )
        return success, next_sequence

    def get_version(
        self, content_id: str, sequence_number: int
    ) -> Optional[ReconstructionResult]:
        """
        Retrieve a specific version by sequence number.

        Args:
            content_id: Content identifier
            sequence_number: Sequential version number

        Returns:
            ReconstructionResult containing the embedding and metadata
        """
        return self.reconstruction_service.reconstruct_version(
            content_id, sequence_number
        )

    def get_version_by_id(self, version_id: str) -> Optional[ReconstructionResult]:
        """
        Retrieve a version by version_id string (e.g., "article_001_v5").

        Args:
            version_id: Version identifier string

        Returns:
            ReconstructionResult or None if not found
        """
        try:
            if "_v" not in version_id:
                logger.error("Invalid version_id format: %s", version_id)
                return None

            parts = version_id.rsplit("_v", 1)
            content_id = parts[0]
            sequence_number = int(parts[1])

            return self.get_version(content_id, sequence_number)

        except (ValueError, IndexError) as e:
            logger.error("Error parsing version_id %s: %s", version_id, str(e))
            return None

    def get_latest_version(self, content_id: str) -> Optional[ReconstructionResult]:
        """
        Get the most recent version of content.

        Args:
            content_id: Content identifier

        Returns:
            ReconstructionResult for the latest version
        """
        next_seq_num = self.storage_engine.get_next_sequence_number(content_id)
        if next_seq_num == 1:
            return None

        return self.get_version(content_id, next_seq_num - 1)

    def get_version_at_time(
        self, content_id: str, timestamp: datetime
    ) -> Optional[ReconstructionResult]:
        """
        Retrieve content state at specific timestamp.

        Args:
            content_id: Content identifier
            timestamp: Target timestamp

        Returns:
            ReconstructionResult for the latest version at or before timestamp
        """
        return self.reconstruction_service.reconstruct_at_timestamp(
            content_id, timestamp
        )

    def get_version_range(
        self, content_id: str, start_sequence: int, end_sequence: int
    ) -> List[ReconstructionResult]:
        """
        Efficiently retrieve multiple consecutive versions.

        Args:
            content_id: Content identifier
            start_sequence: Starting sequence number (inclusive)
            end_sequence: Ending sequence number (inclusive)

        Returns:
            List of ReconstructionResult objects
        """
        sequence_numbers = list(range(start_sequence, end_sequence + 1))
        return self.reconstruction_service.batch_reconstruct(
            content_id, sequence_numbers
        )

    def search_similar_content(
        self, query_embedding: np.ndarray, k: int = 5
    ) -> List[Tuple[str, int, float]]:
        """
        Search for similar content using base snapshot similarity.

        Args:
            query_embedding: Query embedding vector
            k: Number of results to return

        Returns:
            List of (content_id, sequence_number, similarity_score) tuples
        """
        return self.storage_engine.search_similar_snapshots(query_embedding, k)

    def get_content_timeline(self, content_id: str) -> Optional[ContentTimeline]:
        """
        Get the complete timeline for a piece of content.

        Args:
            content_id: Content identifier

        Returns:
            ContentTimeline object or None if not found
        """
        return self.storage_engine.get_content_timeline(content_id)

    def get_content_statistics(self, content_id: str) -> Dict[str, Any]:
        """
        Get comprehensive statistics for a specific content item.

        Args:
            content_id: Content identifier

        Returns:
            Dictionary with content statistics
        """
        timeline = self.storage_engine.get_content_timeline(content_id)
        if not timeline:
            return {"error": "Content not found"}

        timeline_stats = timeline.get_change_statistics()

        reconstruction_stats = (
            self.reconstruction_service.get_reconstruction_statistics(content_id)
        )

        integrity_validation = self.reconstruction_service.validate_timeline_integrity(
            content_id
        )

        return {
            "content_id": content_id,
            "timeline_stats": timeline_stats,
            "reconstruction_stats": reconstruction_stats,
            "integrity_validation": integrity_validation,
        }

    def list_content_ids(self) -> List[str]:
        """
        List all content IDs in the database.

        Returns:
            List of content identifiers
        """
        content_ids = set()

        try:
            import h5py

            with h5py.File(self.storage_path, "r") as f:
                if "base_snapshots" in f:
                    content_ids.update(f["base_snapshots"].keys())
                if "deltas" in f:
                    content_ids.update(f["deltas"].keys())
        except Exception as e:
            logger.error("Error listing content IDs: %s", e)

        return sorted(list(content_ids))

    def _should_create_base_snapshot(
        self,
        timeline: Optional[ContentTimeline],
        sequence_number: int,
        force_base: bool,
        previous_embedding: Optional[np.ndarray] = None,
        current_embedding: Optional[np.ndarray] = None,
        sparsity_threshold: float = 0.7,
    ) -> bool:
        """
        Determine whether to create a base snapshot using sequential versioning logic.
        Now includes sparsity-based promotion.

        Args:
            timeline: Existing content timeline (may be None for first version)
            sequence_number: Current sequence number
            force_base: Force base snapshot creation
            previous_embedding: Previous version embedding (for sparsity analysis)
            current_embedding: Current version embedding (for sparsity analysis)
            sparsity_threshold: Promote to base if sparsity ratio exceeds this

        Returns:
            True if should create base snapshot, False for delta
        """
        if force_base:
            return True

        if timeline is None or sequence_number == 1:
            return True

        if (sequence_number - 1) % self.base_promotion_interval == 0:
            logger.info("Base promotion due to interval: sequence %d", sequence_number)
            return True

        if previous_embedding is not None and current_embedding is not None:
            delta = current_embedding - previous_embedding
            changed_dims = np.sum(
                np.abs(delta) >= self.delta_computer.sparsity_threshold
            )
            sparsity_ratio = changed_dims / len(delta)

            if sparsity_ratio > sparsity_threshold:
                logger.info(
                    "Base promotion due to high sparsity: %.2f (threshold: %.2f) for sequence %d",
                    sparsity_ratio,
                    sparsity_threshold,
                    sequence_number,
                )
                return True

        if timeline.base_snapshots:
            latest_base_seq = max(timeline.base_snapshots.keys())
            gap = sequence_number - latest_base_seq
            max_gap = self.base_promotion_interval * 2

            if gap > max_gap:
                logger.info("Base promotion due to large gap: %d > %d", gap, max_gap)
                return True

        return False

    def _get_embedding_at_sequence(
        self, timeline: ContentTimeline, sequence_number: int
    ) -> Optional[np.ndarray]:
        """
        Get embedding at specific sequence number, with reconstruction if needed.

        Args:
            timeline: Content timeline
            sequence_number: Target sequence number

        Returns:
            Embedding array or None if not accessible
        """
        if sequence_number < 1:
            return None

        if sequence_number in timeline.base_snapshots:
            return timeline.base_snapshots[sequence_number].embedding

        try:
            embedding, _, _ = self.delta_computer.reconstruct_embedding_from_nearest(
                timeline, sequence_number
            )
            return embedding
        except Exception as e:
            logger.error("Failed to reconstruct sequence %d: %s", sequence_number, e)
            return None

    def optimize_content_bases(
        self, content_id: str, max_cost: int = 10
    ) -> Dict[str, Any]:
        """
        Analyze and suggest base snapshot optimizations for content.

        Args:
            content_id: Content identifier
            max_cost: Maximum acceptable reconstruction cost

        Returns:
            Dictionary with optimization recommendations
        """
        timeline = self.storage_engine.get_content_timeline(content_id)
        if not timeline:
            return {"error": "Content not found"}

        optimization_results = {
            "content_id": content_id,
            "current_bases": list(timeline.base_snapshots.keys()),
            "total_versions": timeline.max_sequence_number,
            "recommendations": [],
        }

        high_cost_sequences = []
        all_sequences = timeline.get_all_sequence_numbers()

        for seq_num in all_sequences:
            if seq_num not in timeline.base_snapshots:
                analysis = self.reconstruction_service.find_optimal_base_for_target(
                    content_id, seq_num
                )
                if analysis and analysis.get("reconstruction_cost", 0) > max_cost:
                    high_cost_sequences.append(
                        {
                            "sequence": seq_num,
                            "cost": analysis["reconstruction_cost"],
                            "recommended_base": analysis.get("base_sequence_used"),
                        }
                    )

        if high_cost_sequences:
            optimization_results["high_cost_versions"] = high_cost_sequences
            optimization_results["recommendations"].append(
                f"Consider promoting {len(high_cost_sequences)} versions to base snapshots"
            )
        else:
            optimization_results["recommendations"].append(
                "Current base snapshot placement is optimal"
            )

        return optimization_results

    def get_database_statistics(self) -> Dict[str, Any]:
        """
        Get comprehensive database statistics including sequential versioning metrics.

        Returns:
            Statistics dictionary
        """
        storage_stats = self.storage_engine.get_storage_statistics()

        content_ids = self.list_content_ids()
        reconstruction_summaries = []

        for content_id in content_ids[:5]:
            reconstruction_stats = (
                self.reconstruction_service.get_reconstruction_statistics(content_id)
            )
            if reconstruction_stats and "error" not in reconstruction_stats:
                reconstruction_summaries.append(reconstruction_stats)

        if reconstruction_summaries:
            avg_reconstruction_cost = np.mean(
                [s["avg_reconstruction_cost"] for s in reconstruction_summaries]
            )
            avg_quality_score = np.mean(
                [s["avg_quality_score"] for s in reconstruction_summaries]
            )
            min_quality_score = np.min(
                [s["min_quality_score"] for s in reconstruction_summaries]
            )
        else:
            avg_reconstruction_cost = 0.0
            avg_quality_score = 0.0
            min_quality_score = 0.0

        storage_stats.update(
            {
                "total_content_items": len(content_ids),
                "avg_reconstruction_cost": avg_reconstruction_cost,
                "avg_quality_score": avg_quality_score,
                "min_quality_score": min_quality_score,
                "base_promotion_interval": self.base_promotion_interval,
                "embedding_dimension": self.embedding_dim,
                "sequential_versioning": True,
            }
        )

        return storage_stats

    def close(self):
        """Close database and cleanup resources."""
        self.storage_engine.close()
        logger.info("Closed TemporalVectorDatabase")

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
