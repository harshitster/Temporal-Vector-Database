import numpy as np
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field


@dataclass
class BaseSnapshot:
    """
    Represents a base snapshot containing a full embedding vector.

    Attributes:
        content_id: Identifier for the content (e.g., Wikipedia article ID)
        sequence_number: Sequential version number within this content (1, 2, 3, ...)
        timestamp: When this snapshot was created
        embedding: The full embedding vector (numpy array)
        metadata: Additional metadata about this snapshot
        version_id: Human-readable version identifier (auto-generated)
    """

    content_id: str
    sequence_number: int
    timestamp: datetime
    embedding: np.ndarray
    metadata: Dict[str, Any] = field(default_factory=dict)
    version_id: str = ""

    def __post_init__(self):
        if not isinstance(self.embedding, np.ndarray):
            self.embedding = np.array(self.embedding)

        if len(self.embedding.shape) != 1:
            raise ValueError("Embedding must be a 1D numpy array")

        if self.sequence_number <= 0:
            raise ValueError("Sequence number must be positive")

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
            "sequence_number": self.sequence_number,
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

    Attributes:
        content_id: Content identifier
        sequence_number: Sequential version number this delta produces
        from_sequence_number: Previous sequence number this delta applies to
        timestamp: When this change occurred
        sparse_delta: Dictionary of index -> change value
        change_magnitude: L2 norm of the changes
        metadata: Additional metadata
        version_id: Human-readable version identifier (auto-generated)
    """

    content_id: str
    sequence_number: int
    from_sequence_number: int
    timestamp: datetime
    sparse_delta: Dict[int, float]
    change_magnitude: float
    metadata: Dict[str, Any] = field(default_factory=dict)
    version_id: str = ""

    def __post_init__(self):
        """Calculate change magnitude if not provided."""
        # Convert numpy types to Python types
        if hasattr(self.sequence_number, "item"):
            self.sequence_number = int(self.sequence_number)
        if hasattr(self.from_sequence_number, "item"):
            self.from_sequence_number = int(self.from_sequence_number)

        if self.change_magnitude == 0 and self.sparse_delta:
            self.change_magnitude = np.sqrt(
                sum(v**2 for v in self.sparse_delta.values())
            )

        if self.sequence_number <= 0 or self.from_sequence_number <= 0:
            raise ValueError("Sequence numbers must be positive")

        if self.sequence_number != self.from_sequence_number + 1:
            raise ValueError("Delta must increment sequence number by exactly 1")

        if not self.version_id:
            self.version_id = f"{self.content_id}_v{self.sequence_number}"

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
            "content_id": self.content_id,
            "sequence_number": self.sequence_number,
            "from_sequence_number": self.from_sequence_number,
            "timestamp": self.timestamp.isoformat(),
            "sparse_delta": self.sparse_delta,
            "change_magnitude": self.change_magnitude,
            "metadata": self.metadata,
            "version_id": self.version_id,
        }


@dataclass
class ContentTimeline:
    """
    Manages the temporal sequence of versions for a piece of content.
    Optimized for sequential access patterns with O(1) lookups.
    """

    content_id: str
    base_snapshots: Dict[int, BaseSnapshot] = field(default_factory=dict)
    deltas: Dict[int, Delta] = field(default_factory=dict)
    max_sequence_number: int = 0

    def add_base_snapshot(self, snapshot: BaseSnapshot):
        """Add a base snapshot to the timeline"""
        if snapshot.content_id != self.content_id:
            raise ValueError(
                f"Snapshot content_id {snapshot.content_id} doesn't match timeline {self.content_id}"
            )

        seq_num = snapshot.sequence_number
        if seq_num in self.base_snapshots:
            raise ValueError(
                f"Base snapshot with sequence number {seq_num} already exists"
            )

        if seq_num in self.deltas:
            raise ValueError(f"Delta with sequence number {seq_num} already exists")

        self.base_snapshots[seq_num] = snapshot
        self.max_sequence_number = max(self.max_sequence_number, seq_num)

    def add_delta(self, delta: Delta):
        """Add a delta to the timeline"""
        if delta.content_id != self.content_id:
            raise ValueError(
                f"Delta content_id {delta.content_id} doesn't match timeline {self.content_id}"
            )

        seq_num = delta.sequence_number

        if seq_num in self.deltas:
            raise ValueError(f"Delta with sequence number {seq_num} already exists")

        if seq_num in self.base_snapshots:
            raise ValueError(
                f"Base snapshot with sequence number {seq_num} already exists"
            )

        prev_seq = delta.from_sequence_number
        if prev_seq not in self.base_snapshots and prev_seq not in self.deltas:
            raise ValueError(f"Previous sequence number {prev_seq} does not exist")

        self.deltas[seq_num] = delta
        self.max_sequence_number = max(self.max_sequence_number, seq_num)

    def get_version_at_sequence(
        self, sequence_number: int
    ) -> Optional[Tuple[np.ndarray, str, bool]]:
        """
        Get version at specific sequence number

        Args:
            sequence_number: Target sequence number

        Returns:
            Tuple of (embedding, version_id, is_base_snapshot) or None if not found
        """

        if sequence_number in self.base_snapshots:
            snapshot = self.base_snapshots[sequence_number]
            return snapshot.embedding, snapshot.version_id, True
        elif sequence_number in self.deltas:
            delta = self.deltas[sequence_number]
            return None, delta.version_id, False
        else:
            return None

    def get_version_before_timestamp(self, timestamp: datetime) -> int:
        """Get the latest sequence number before the timestamp"""
        latest_seq_num = 0

        # Check base snapshots
        for seq_num, snapshot in self.base_snapshots.items():
            if snapshot.timestamp <= timestamp:
                latest_seq_num = max(latest_seq_num, snapshot.sequence_number)

        # Check deltas
        for seq_num, delta in self.deltas.items():
            if delta.timestamp <= timestamp:
                latest_seq_num = max(latest_seq_num, delta.sequence_number)

        return latest_seq_num

    def find_nearest_base_snapshot(
        self, target_sequence: int
    ) -> Optional[Tuple[BaseSnapshot, int]]:
        """
        Find the nearest base snapshot to a target sequence number.

        Args:
            target_sequence: Target sequence number

        Returns:
            Tuple of (BaseSnapshot, distance) or None if no snapshots exists
        """

        if not self.base_snapshots:
            return None

        seq_num = target_sequence
        while seq_num >= 0:
            if seq_num in self.base_snapshots:
                return self.base_snapshots[seq_num], target_sequence - seq_num

            seq_num -= 1

        return None

    def get_delta_chain(self, from_sequence: int, to_sequence: int) -> List[Delta]:
        """
        Get ordered list of deltas needed to reconstruct from one sequence to another.

        Args:
            from_sequence: Starting sequence number
            to_sequence: Target sequence number

        Returns:
            List of deltas in application order
        """
        if from_sequence >= to_sequence:
            return []

        chain = []
        for seq_num in range(from_sequence + 1, to_sequence + 1):
            if seq_num in self.deltas:
                chain.append(self.deltas[seq_num])
            else:
                raise ValueError(f"Missing delta for sequence number {seq_num}")

        return chain

    def get_change_statistics(self) -> Dict[str, float]:
        """Get statistics about changes in this timeline."""
        if not self.deltas:
            return {"total_changes": 0, "avg_magnitude": 0, "max_magnitude": 0}

        magnitudes = [delta.change_magnitude for delta in self.deltas.values()]
        return {
            "total_changes": len(magnitudes),
            "avg_magnitude": np.mean(magnitudes),
            "max_magnitude": np.max(magnitudes),
            "min_magnitude": np.min(magnitudes),
            "max_sequence_number": self.max_sequence_number,
            "base_snapshots_count": len(self.base_snapshots),
            "deltas_count": len(self.deltas),
        }

    def get_all_sequence_numbers(self) -> List[int]:
        """Get all sequence numbers in chronological order."""
        all_sequences = set(self.base_snapshots.keys()) | set(self.deltas.keys())
        return sorted(all_sequences)
