import numpy as np
import uuid
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
import pickle
import os
from pathlib import Path


@dataclass
class BaseSnapshot:
    """
    Represents a base snapshot containing a full embedding vector.

    Attributes:
        content_id: Identifier for the content (e.g., Wikipedia article ID)
        timestamp: When this snapshot was created
        embedding: The full embedding vector (numpy array)
        metadata: Additional metadata about this snapshot
    """

    content_id: str
    timestamp: datetime
    embedding: np.ndarray
    metadata: Dict[str, Any] = field(default_factory=dict)
    version_id: str = ""

    def __post_init__(self):
        if not isinstance(self.embedding, np.ndarray):
            self.embedding = np.array(self.embedding)

        if len(self.embedding.shape) != 1:
            raise ValueError("Embedding must be a 1D numpy array")

        if not self.version_id:
            self.version_id = f"{self.content_id}_{self.timestamp.isoformat()}"

    @property
    def dimension(self) -> int:
        """Return the dimensionality of the embedding."""
        return len(self.embedding)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "content_id": self.content_id,
            "timestamp": self.timestamp.isoformat(),
            "embedding": self.embedding.tolist(),
            "metadata": self.metadata,
            "version_id": self.version_id,
        }


@dataclass
class Delta:
    """
    Represents sparse changes between two embeddings.
    Stores only significant differences to minimize storage.
    """

    from_snapshot_id: str
    to_version_id: str
    timestamp: datetime
    sparse_delta: Dict[int, float]
    change_magnitude: float
    content_id: str
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        """Calculate change magnitude if not provided."""
        if self.change_magnitude == 0 and self.sparse_delta:
            self.change_magnitude = np.sqrt(
                sum(v**2 for v in self.sparse_delta.values())
            )

    def apply_to_embedding(self, base_embedding: np.ndarray) -> np.ndarray:
        """
        Apply this delta to a base embedding to reconstruct the target embedding.

        Args:
            base_embedding: The base embedding to apply delta to

        Returns:
            Reconstructed embedding after applying delta
        """
        result = base_embedding.copy()
        for idx, change in self.sparse_delta.items():
            if idx < len(result):
                result[idx] += change
        return result

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "from_snapshot_id": self.from_snapshot_id,
            "to_version_id": self.to_version_id,
            "timestamp": self.timestamp.isoformat(),
            "sparse_delta": self.sparse_delta,
            "change_magnitude": self.change_magnitude,
            "content_id": self.content_id,
            "metadata": self.metadata,
        }


@dataclass
class ContentTimeline:
    """
    Manages the temporal sequence of versions for a piece of content.
    Tracks base snapshots and their associated deltas.
    """

    content_id: str
    base_snapshots: List[BaseSnapshot] = field(default_factory=list)
    deltas: List[Delta] = field(default_factory=list)

    def add_base_snapshot(self, snapshot: BaseSnapshot):
        """Add a base snapshot to the timeline."""
        if snapshot.content_id != self.content_id:
            raise ValueError(
                f"Snapshot content_id {snapshot.content_id} doesn't match timeline {self.content_id}"
            )

        insert_idx = 0
        for i, existing in enumerate(self.base_snapshots):
            if existing.timestamp <= snapshot.timestamp:
                insert_idx = i + 1
            else:
                break
        self.base_snapshots.insert(insert_idx, snapshot)

    def add_delta(self, delta: Delta):
        """Add a delta to the timeline."""
        if delta.content_id != self.content_id:
            raise ValueError(
                f"Delta content_id {delta.content_id} doesn't match timeline {self.content_id}"
            )

        insert_idx = 0
        for i, existing in enumerate(self.deltas):
            if existing.timestamp <= delta.timestamp:
                insert_idx = i + 1
            else:
                break
        self.deltas.insert(insert_idx, delta)

    def get_versions_at_time(self, timestamp: datetime) -> List[str]:
        """Get all version IDs that existed at or before the given timestamp."""
        versions = []

        for snapshot in self.base_snapshots:
            if snapshot.timestamp <= timestamp:
                versions.append(snapshot.version_id)

        for delta in self.deltas:
            if delta.timestamp <= timestamp:
                versions.append(delta.to_version_id)

        return sorted(versions)

    def get_change_statistics(self) -> Dict[str, float]:
        """Get statistics about changes in this timeline."""
        if not self.deltas:
            return {"total_changes": 0, "avg_magnitude": 0, "max_magnitude": 0}

        magnitudes = [delta.change_magnitude for delta in self.deltas]
        return {
            "total_changes": len(magnitudes),
            "avg_magnitude": np.mean(magnitudes),
            "max_magnitude": np.max(magnitudes),
            "min_magnitude": np.min(magnitudes),
        }
