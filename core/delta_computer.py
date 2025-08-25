import numpy as np
from datetime import datetime
from typing import Tuple, List
import logging

from core.data_structures import BaseSnapshot, Delta

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DeltaComputer:
    """
    Computes chained deltas between consecutive embeddings with configurable sparsity thresholds.
    Supports reconstruction through sequential delta application for efficient temporal storage.
    """

    def __init__(self, sparsity_threshold: float = 0.01, max_delta_ratio: float = 0.8):
        """
        Initialize delta computer with sparsity settings.

        Args:
            sparsity_threshold: Minimum change magnitude to store (for sparsity)
            max_delta_ratio: Maximum ratio of dimensions that can change (safety check)
        """

        self.sparsity_threshold = sparsity_threshold
        self.max_delta_ratio = max_delta_ratio

    def compute_delta(
        self,
        from_embedding: np.ndarray,
        to_embedding: np.ndarray,
        from_version_id: str,
        to_version_id: str,
        timestamp: datetime,
        content_id: str,
    ) -> Delta:
        """
        Compute sparse delta between two consecutive embeddings.

        In the chained approach, this typically computes the difference between
        the immediately previous version and the current version.

        Args:
            from_embedding: Source embedding (previous version)
            to_embedding: Target embedding (current version)
            from_version_id: ID of the source version
            to_version_id: ID of the target version
            timestamp: When this change occurred
            content_id: Content identifier

        Returns:
            Delta object representing the sparse changes from source to target
        """
        if len(from_embedding) != len(to_embedding):
            raise ValueError("Embeddings must have the same dimension")

        raw_delta = to_embedding - from_embedding

        sparse_delta = {}
        for i, change in enumerate(raw_delta):
            if abs(change) >= self.sparsity_threshold:
                sparse_delta[i] = float(change)

        sparsity_ratio = len(sparse_delta) / len(raw_delta)
        if sparsity_ratio > self.max_delta_ratio:
            logger.warning(f"High sparsity ratio {sparsity_ratio:.2f} for {content_id}")

        change_magnitude = np.linalg.norm(raw_delta)

        return Delta(
            from_snapshot_id=from_version_id,
            to_version_id=to_version_id,
            timestamp=timestamp,
            sparse_delta=sparse_delta,
            change_magnitude=change_magnitude,
            content_id=content_id,
            metadata={
                "sparsity_ratio": sparsity_ratio,
                "dimensions_changed": len(sparse_delta),
                "total_dimensions": len(raw_delta),
            },
        )

    def reconstruct_embedding(
        self, base_snapshot: BaseSnapshot, deltas: List[Delta]
    ) -> np.ndarray:
        """
        Reconstruct an embedding by applying a sequence of chained deltas to a base snapshot.

        In the chained approach, each delta represents changes from the immediately
        previous version, so reconstruction applies them sequentially:
        result = base + delta1 + delta2 + delta3 + ...

        Args:
            base_snapshot: Base embedding to start from (checkpoint)
            deltas: Sequence of deltas to apply in chronological order

        Returns:
            Reconstructed embedding at the target time point
        """
        result = base_snapshot.embedding.copy()

        for delta in deltas:
            result = delta.apply_to_embedding(result)

        return result

    def validate_reconstruction(
        self, original: np.ndarray, reconstructed: np.ndarray, tolerance: float = 1e-6
    ) -> Tuple[bool, float]:
        """
        Validate that reconstruction is accurate within tolerance.

        Args:
            original: Original embedding
            reconstructed: Reconstructed embedding
            tolerance: Maximum allowed difference

        Returns:
            Tuple of (is_valid, cosine_similarity)
        """
        cos_sim = np.dot(original, reconstructed) / (
            np.linalg.norm(original) * np.linalg.norm(reconstructed)
        )

        l2_error = np.linalg.norm(original - reconstructed)

        is_valid = l2_error < tolerance

        return is_valid, float(cos_sim)
