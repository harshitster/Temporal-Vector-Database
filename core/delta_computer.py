import numpy as np
from datetime import datetime
from typing import Tuple, List, Dict, Any
import logging

from core.data_structures import BaseSnapshot, Delta, ContentTimeline

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DeltaComputer:
    """
    Computes chained deltas between consecutive embeddings with sequential versioning.
    Supports efficient reconstruction through sequential delta application and nearest
    base snapshot selection for optimal performance.
    """

    def __init__(self, sparsity_threshold: float = 0.01, max_delta_ratio: float = 0.8):
        """
        Initialize delta computer with sparsity settings

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
        from_sequence_number: int,
        to_sequence_number: int,
        timestamp: datetime,
        content_id: str,
    ) -> Delta:
        """
        Compute sparse delta between two consecutive embeddings using sequential versioning.

        Args:
            from_embedding: Source embedding (previous version)
            to_embedding: Target embedding (current version)
            from_sequence_number: Previous sequence number
            to_sequence_number: Target sequence number (must be from_sequence_number + 1)
            timestamp: When this change occurred
            content_id: Content identifier

        Returns:
            Delta object representing the sparse changes from source to target
        """

        if len(from_embedding) != len(to_embedding):
            raise ValueError("Embeddings must have the same dimension")

        if to_sequence_number != from_sequence_number + 1:
            raise ValueError("Delta must represent consecutive sequence numbers")

        raw_delta = to_embedding - from_embedding

        sparse_delta = {}
        for i, change in enumerate(raw_delta):
            if abs(change) >= self.sparsity_threshold:
                sparse_delta[i] = float(change)

        sparsity_ratio = len(sparse_delta) / len(raw_delta)
        if sparsity_ratio > self.max_delta_ratio:
            logger.warning(
                "High sparsity ratio %.2f for %s", sparsity_ratio, content_id
            )

        change_magnitude = np.linalg.norm(raw_delta)

        return Delta(
            content_id=content_id,
            sequence_number=to_sequence_number,
            from_sequence_number=from_sequence_number,
            timestamp=timestamp,
            sparse_delta=sparse_delta,
            change_magnitude=change_magnitude,
            metadata={
                "sparsity_ratio": float(sparsity_ratio),
                "dimensions_changed": int(len(sparse_delta)),
                "total_dimensions": int(len(raw_delta)),
            },
        )

    def reconstruct_embedding_from_nearest(
        self, timeline: ContentTimeline, target_sequence: int
    ) -> Tuple[np.ndarray, List[str], BaseSnapshot]:
        """
        Reconstruct embedding using the nearest base snapshot for optimal performance.

        Args:
            timeline: Content timeline containing snapshots and deltas
            target_sequence: Target sequence number to reconstruct

        Returns:
            Tuple of (reconstructed_embedding, deltas_applied_list, base_used)
        """

        nearest_result = timeline.find_nearest_base_snapshot(target_sequence)
        if not nearest_result:
            raise ValueError(
                f"No base snapshots available for content {timeline.content_id}"
            )

        base_snapshot, _ = nearest_result
        base_seq = base_snapshot.sequence_number

        if base_seq == target_sequence:
            return base_snapshot.embedding.copy(), [], base_snapshot

        if base_seq > target_sequence:
            raise ValueError(
                f"Cannot reconstruct sequence {target_sequence} from later base {base_seq}"
            )

        try:
            delta_chain = timeline.get_delta_chain(base_seq, target_sequence)
        except ValueError as e:
            raise ValueError(
                f"Cannot reconstruct sequence {target_sequence}: {e}"
            ) from e

        result = base_snapshot.embedding.copy()
        deltas_applied = []

        for delta in delta_chain:
            result = delta.apply_to_embedding(result)
            deltas_applied.append(delta.version_id)

        return result, deltas_applied, base_snapshot

    def reconstruct_embedding_legacy(
        self, base_snapshot: BaseSnapshot, deltas: List[Delta]
    ) -> np.ndarray:
        """
        Legacy reconstruction method for backward compatibility.

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

    def find_optimal_reconstruction_path(
        self, timeline: ContentTimeline, target_sequence: int
    ) -> Tuple[BaseSnapshot, List[Delta], int]:
        """
        Find the optimal reconstruction path using sequential versioning.

        Args:
            timeline: Content timeline
            target_sequence: Target sequence number to reconstruct

        Returns:
            Tuple of (base_snapshot, delta_list, reconstruction_cost)
        """
        if target_sequence in timeline.base_snapshots:
            base_snapshot = timeline.base_snapshots[target_sequence]
            return base_snapshot, [], 0

        nearest_result = timeline.find_nearest_base_snapshot(target_sequence)
        if not nearest_result:
            raise ValueError("No base snapshots available")

        base_snapshot, _ = nearest_result
        base_seq = base_snapshot.sequence_number

        if base_seq > target_sequence:
            raise ValueError(
                f"Target sequence {target_sequence} is before nearest base {base_seq}"
            )

        try:
            delta_chain = timeline.get_delta_chain(base_seq, target_sequence)
            reconstruction_cost = len(delta_chain)
            return base_snapshot, delta_chain, reconstruction_cost
        except ValueError as e:
            raise ValueError(f"Cannot build reconstruction path: {e}") from e

    def validate_reconstruction(
        self, original: np.ndarray, reconstructed: np.ndarray, tolerance: float = 0.01
    ) -> Tuple[bool, float]:
        """
        Validate that reconstruction is accurate within tolerance.
        Updated with more realistic tolerance for sequential reconstruction chains.

        Args:
            original: Original embedding
            reconstructed: Reconstructed embedding
            tolerance: Maximum allowed L2 error (default increased to 0.01)

        Returns:
            Tuple of (is_valid, cosine_similarity)
        """
        cos_sim = np.dot(original, reconstructed) / (
            np.linalg.norm(original) * np.linalg.norm(reconstructed)
        )

        l2_error = np.linalg.norm(original - reconstructed)

        is_valid = l2_error < tolerance

        return is_valid, float(cos_sim)

    def estimate_reconstruction_cost(
        self, timeline: ContentTimeline, target_sequence: int
    ) -> Dict[str, Any]:
        """
        Estimate the computational cost and accuracy of reconstructing a target sequence.

        Args:
            timeline: Content timeline
            target_sequence: Target sequence to analyze

        Returns:
            Dictionary with cost estimates and recommendations
        """
        try:
            base_snapshot, delta_chain, cost = self.find_optimal_reconstruction_path(
                timeline, target_sequence
            )

            base_error = cost * 0.001

            if delta_chain:
                avg_magnitude = np.mean([d.change_magnitude for d in delta_chain])
                magnitude_penalty = avg_magnitude * 0.1

                avg_sparsity = np.mean(
                    [
                        len(d.sparse_delta) / len(base_snapshot.embedding)
                        for d in delta_chain
                    ]
                )
                sparsity_penalty = avg_sparsity * 0.5
            else:
                magnitude_penalty = 0
                sparsity_penalty = 0

            estimated_error = base_error + magnitude_penalty + sparsity_penalty

            return {
                "reconstruction_cost": cost,
                "base_sequence_used": base_snapshot.sequence_number,
                "deltas_required": len(delta_chain),
                "estimated_error_bound": estimated_error,
                "recommended": cost < 10 and estimated_error < 0.05,
                "base_snapshot_distance": abs(
                    target_sequence - base_snapshot.sequence_number
                ),
            }

        except ValueError as e:
            return {
                "reconstruction_cost": float("inf"),
                "error": str(e),
                "recommended": False,
            }
