import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
from datetime import datetime
from typing import List, Optional, Dict, Any
from dataclasses import dataclass
import logging

from core.data_structures import ContentTimeline
from core.delta_computer import DeltaComputer
from storage.storage_engine import TemporalStorageEngine

logger = logging.getLogger(__name__)


@dataclass
class ReconstructionResult:
    """
    Result of embedding reconstruction with quality metrics and error bounds
    """

    embedding: np.ndarray
    sequence_number: int
    base_sequence_used: int
    deltas_applied: List[str]
    reconstruction_time_ms: float
    reconstruction_cost: int
    cosine_similarity: Optional[float]
    l2_error: Optional[float]
    error_bound_estimate: float
    quality_score: float
    base_distance: int


class ReconstructionService:
    """
    Service for reconstructing embeddings with error bounds and quality assessment.
    Implements base snapshot selection and reconstruction path optimization.
    """

    def __init__(
        self, storage_engine: TemporalStorageEngine, delta_computer: DeltaComputer
    ):
        """
        Initialize reconstruction service.

        Args:
            storage_engine: Storage engine for data access
            delta_computer: Delta computation engine
        """
        self.storage_engine = storage_engine
        self.delta_computer = delta_computer

        self.max_chain_length = 15
        self.error_accumulation_rate = 0.0005
        self.quality_threshold = 0.95

    def reconstruct_version(
        self,
        content_id: str,
        sequence_number: int,
        ground_truth: Optional[np.ndarray] = None,
    ) -> Optional[ReconstructionResult]:
        """
        Reconstruct a specific version with error bounds and quality assessment.

        Args:
            content_id: Content identifier
            sequence_number: Target sequence number
            ground_truth: Optional ground truth embedding for validation

        Returns:
            ReconstructionResult or None if reconstruction fails
        """
        start_time = datetime.now()

        timeline = self.storage_engine.get_content_timeline(content_id)
        if not timeline:
            logger.error("No timeline found for content %s", content_id)
            return None

        try:
            reconstructed_embedding, deltas_applied, base_used = (
                self.delta_computer.reconstruct_embedding_from_nearest(
                    timeline, sequence_number
                )
            )

            reconstruction_time = (datetime.now() - start_time).total_seconds() * 1000
            reconstruction_cost = len(deltas_applied)
            base_distance = abs(sequence_number - base_used.sequence_number)

            error_bound = self._estimate_error_bound_sequential(
                reconstruction_cost, timeline, sequence_number
            )
            quality_score = self._calculate_quality_score_sequential(
                reconstruction_cost, base_distance, error_bound
            )

            cosine_sim, l2_error = None, None
            if ground_truth is not None:
                _, cosine_sim = self.delta_computer.validate_reconstruction(
                    ground_truth, reconstructed_embedding
                )
                l2_error = float(np.linalg.norm(ground_truth - reconstructed_embedding))

            return ReconstructionResult(
                embedding=reconstructed_embedding,
                sequence_number=sequence_number,
                base_sequence_used=base_used.sequence_number,
                deltas_applied=deltas_applied,
                reconstruction_time_ms=reconstruction_time,
                reconstruction_cost=reconstruction_cost,
                cosine_similarity=cosine_sim,
                l2_error=l2_error,
                error_bound_estimate=error_bound,
                quality_score=quality_score,
                base_distance=base_distance,
            )
        except Exception as e:
            logger.error(
                "Reconstruction failed for %s_v%s: %s", content_id, sequence_number, e
            )
            return None

    def reconstruct_at_timestamp(
        self,
        content_id: str,
        timestamp: datetime,
        ground_truth: Optional[np.ndarray] = None,
    ) -> Optional[ReconstructionResult]:
        """
        Reconstruct embedding state at specific timestamp.

        Args:
            content_id: Content identifier
            timestamp: Target timestamp
            ground_truth: Optional ground truth for validation

        Returns:
            ReconstructionResult or None if reconstruction fails
        """
        timeline = self.storage_engine.get_content_timeline(content_id)
        if not timeline:
            return None

        target_sequence_number = timeline.get_version_before_timestamp(timestamp)
        if target_sequence_number == 0:
            logger.error(f"No versions found at timestamp {timestamp}")
            return None

        return self.reconstruct_version(
            content_id, target_sequence_number, ground_truth
        )

    def batch_reconstruct(
        self, content_id: str, sequence_numbers: List[int]
    ) -> List[ReconstructionResult]:
        """
        Efficiently reconstruct multiple versions by reusing intermediate reconstructions.

        Args:
            content_id: Content identifier
            sequence_numbers: List of sequence numbers to reconstruct

        Returns:
            List of ReconstructionResult objects
        """
        timeline = self.storage_engine.get_content_timeline(content_id)
        if not timeline:
            return []

        results = []
        sorted_sequences = sorted(sequence_numbers)

        for seq_num in sorted_sequences:
            result = self.reconstruct_version(content_id, seq_num)
            if result:
                results.append(result)

        return results

    def find_optimal_base_for_target(
        self, content_id: str, target_sequence: int
    ) -> Optional[Dict[str, Any]]:
        """
        Analyze which base snapshot would be optimal for reconstructing a target sequence.

        Args:
            content_id: Content identifier
            target_sequence: Target sequence number

        Returns:
            Dictionary with optimization analysis
        """
        timeline = self.storage_engine.get_content_timeline(content_id)
        if not timeline:
            return None

        try:
            cost_analysis = self.delta_computer.estimate_reconstruction_cost(
                timeline, target_sequence
            )

            cost_analysis.update(
                {
                    "content_id": content_id,
                    "target_sequence": target_sequence,
                    "total_versions": timeline.max_sequence_number,
                    "available_bases": list(timeline.base_snapshots.keys()),
                    "optimization_recommended": cost_analysis.get("recommended", False),
                }
            )

            return cost_analysis

        except Exception as e:
            logger.error(
                "Error analyzing optimal base for %s_v%s: %s",
                content_id,
                target_sequence,
                e,
            )
            return None

    def _estimate_error_bound_sequential(
        self, reconstruction_cost: int, timeline: ContentTimeline, target_sequence: int
    ) -> float:
        """
        Enhanced error bound estimation for sequential versioning.

        Args:
            reconstruction_cost: Number of deltas in reconstruction chain
            timeline: Content timeline
            target_sequence: Target sequence number

        Returns:
            Estimated error bound
        """
        if reconstruction_cost == 0:
            return 0.0

        base_error = reconstruction_cost * self.error_accumulation_rate

        try:
            base_result = timeline.find_nearest_base_snapshot(target_sequence)
            if base_result:
                base_snapshot, _ = base_result
                delta_chain = timeline.get_delta_chain(
                    base_snapshot.sequence_number, target_sequence
                )

                if delta_chain:
                    avg_magnitude = np.mean([d.change_magnitude for d in delta_chain])
                    magnitude_factor = 1.0 + avg_magnitude * 0.05

                    consistency_bonus = 0.9 if reconstruction_cost < 5 else 1.0

                    return base_error * magnitude_factor * consistency_bonus
        except:
            pass

        return base_error * 1.5

    def _calculate_quality_score_sequential(
        self, reconstruction_cost: int, base_distance: int, error_bound: float
    ) -> float:
        """
        Calculate quality score optimized for sequential versioning patterns.

        Args:
            reconstruction_cost: Number of deltas applied
            base_distance: Distance from base to target sequence
            error_bound: Estimated error bound

        Returns:
            Quality score between 0 and 1
        """
        if reconstruction_cost == 0:
            return 1.0

        chain_penalty = max(
            0, 1.0 - (reconstruction_cost / self.max_chain_length) * 0.3
        )

        distance_bonus = max(0.7, 1.0 - (base_distance / 20) * 0.3)

        error_penalty = max(0.5, 1.0 - error_bound * 20)

        sequential_bonus = 1.1 if reconstruction_cost < 8 else 1.0

        quality = chain_penalty * distance_bonus * error_penalty * sequential_bonus

        return min(1.0, max(0.0, quality))

    def validate_timeline_integrity(self, content_id: str) -> Dict[str, Any]:
        """
        Validate the integrity of a content timeline's sequential versioning.

        Args:
            content_id: Content identifier

        Returns:
            Dictionary with validation results
        """
        timeline = self.storage_engine.get_content_timeline(content_id)
        if not timeline:
            return {"valid": False, "error": "Timeline not found"}

        validation_results = {
            "valid": True,
            "content_id": content_id,
            "max_sequence": timeline.max_sequence_number,
            "base_snapshots": len(timeline.base_snapshots),
            "deltas": len(timeline.deltas),
            "issues": [],
        }

        all_sequences = timeline.get_all_sequence_numbers()
        if all_sequences:
            expected_sequences = set(range(1, max(all_sequences) + 1))
            actual_sequences = set(all_sequences)
            missing_sequences = expected_sequences - actual_sequences

            if missing_sequences:
                validation_results["issues"].append(
                    f"Missing sequences: {sorted(missing_sequences)}"
                )
                validation_results["valid"] = False

        for seq_num, delta in timeline.deltas.items():
            prev_seq = delta.from_sequence_number
            if (
                prev_seq not in timeline.base_snapshots
                and prev_seq not in timeline.deltas
            ):
                validation_results["issues"].append(
                    f"Delta v{seq_num} references missing v{prev_seq}"
                )
                validation_results["valid"] = False

        if timeline.base_snapshots and timeline.deltas:
            max_gap = 0
            base_sequences = sorted(timeline.base_snapshots.keys())
            for i in range(len(base_sequences) - 1):
                gap = base_sequences[i + 1] - base_sequences[i]
                max_gap = max(max_gap, gap)

            validation_results["max_base_gap"] = max_gap
            if max_gap > 20:
                validation_results["issues"].append(
                    f"Large gap between bases: {max_gap} versions"
                )

        return validation_results

    def get_reconstruction_statistics(self, content_id: str) -> Dict[str, Any]:
        """
        Get comprehensive reconstruction performance statistics for content.

        Args:
            content_id: Content identifier

        Returns:
            Dictionary with reconstruction statistics
        """
        timeline = self.storage_engine.get_content_timeline(content_id)
        if not timeline:
            return {"error": "No reconstruction statistics available"}

        all_sequences = timeline.get_all_sequence_numbers()
        sample_sequences = all_sequences[:: max(1, len(all_sequences) // 10)]

        costs = []
        distances = []
        quality_scores = []

        for seq_num in sample_sequences:
            try:
                cost_analysis = self.delta_computer.estimate_reconstruction_cost(
                    timeline, seq_num
                )
                if "reconstruction_cost" in cost_analysis:
                    costs.append(cost_analysis["reconstruction_cost"])
                    distances.append(cost_analysis.get("base_snapshot_distance", 0))

                    error_bound = self._estimate_error_bound_sequential(
                        cost_analysis["reconstruction_cost"], timeline, seq_num
                    )
                    quality = self._calculate_quality_score_sequential(
                        cost_analysis["reconstruction_cost"],
                        cost_analysis.get("base_snapshot_distance", 0),
                        error_bound,
                    )
                    quality_scores.append(quality)
            except:
                continue

        if not costs:
            return {"error": "No reconstruction statistics available"}

        return {
            "content_id": content_id,
            "versions_analyzed": len(costs),
            "avg_reconstruction_cost": np.mean(costs),
            "max_reconstruction_cost": np.max(costs),
            "avg_base_distance": np.mean(distances),
            "avg_quality_score": np.mean(quality_scores),
            "min_quality_score": np.min(quality_scores),
            "recommended_base_promotion": np.max(costs) > 15,
        }
