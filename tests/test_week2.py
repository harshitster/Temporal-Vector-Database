import os
import sys
import tempfile
import shutil
import numpy as np
from datetime import datetime, timedelta
from typing import List, Dict, Any
import logging

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from storage.temporal_database import TemporalVectorDatabase
from simulation.wikipedia import WikipediaSimulator

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SequentialVersioningTester:
    """
    Comprehensive testing framework for sequential versioning temporal database system.
    Tests auto-increment functionality, nearest base reconstruction, sparsity-based promotion,
    and performance optimizations.
    """

    def __init__(self, temp_dir: str = None):
        """
        Initialize tester with temporary storage directory.

        Args:
            temp_dir: Temporary directory for test databases
        """
        if temp_dir is None:
            self.temp_dir = tempfile.mkdtemp()
            self.cleanup_temp = True
        else:
            self.temp_dir = temp_dir
            self.cleanup_temp = False

        self.test_db_path = os.path.join(self.temp_dir, "test_sequential_db.h5")
        self.simulator = WikipediaSimulator(embedding_dim=100, num_articles=5)

        logger.info(
            f"Initialized sequential tester with storage at {self.test_db_path}"
        )

    def test_sparsity_based_promotion(self) -> Dict[str, Any]:
        """Test the new sparsity-based base promotion logic."""
        logger.info("Testing sparsity-based base promotion...")

        results = {}

        with TemporalVectorDatabase(
            self.test_db_path,
            embedding_dim=100,
            base_promotion_interval=20,  # Large interval so sparsity is main trigger
            base_promotion_sparsity_threshold=0.6,  # 60% threshold for testing
        ) as db:

            # Create base embedding
            base_embedding = np.random.normal(0, 1, 100)
            base_embedding = base_embedding / np.linalg.norm(base_embedding)

            # Test 1: Add first version (should be base)
            success1, seq1 = db.add_content_version("sparsity_test", base_embedding)
            results["first_version_stored"] = success1
            results["first_sequence"] = seq1

            # Test 2: Small change (low sparsity - should be delta)
            small_change = base_embedding.copy()
            small_change[:5] += np.random.normal(0, 0.02, 5)  # Only 5% of dimensions
            small_change = small_change / np.linalg.norm(small_change)

            success2, seq2 = db.add_content_version("sparsity_test", small_change)
            results["small_change_stored"] = success2
            results["small_change_sequence"] = seq2

            # Verify it's stored as delta (not base)
            timeline = db.get_content_timeline("sparsity_test")
            results["small_change_is_delta"] = (
                timeline
                and seq2 in timeline.deltas
                and seq2 not in timeline.base_snapshots
            )

            # Test 3: Large change (high sparsity - should trigger base promotion)
            large_change = base_embedding.copy()
            # Change 70% of dimensions (above 60% threshold)
            change_indices = np.random.choice(100, 70, replace=False)
            large_change[change_indices] += np.random.normal(0, 0.1, 70)
            large_change = large_change / np.linalg.norm(large_change)

            success3, seq3 = db.add_content_version("sparsity_test", large_change)
            results["large_change_stored"] = success3
            results["large_change_sequence"] = seq3

            # Verify it's stored as base (due to sparsity)
            timeline = db.get_content_timeline("sparsity_test")
            results["large_change_is_base"] = (
                timeline and seq3 in timeline.base_snapshots
            )

            # Test 4: Medium change (right at threshold - should be delta)
            medium_change = large_change.copy()
            # Change exactly 50% of dimensions (below 60% threshold)
            change_indices = np.random.choice(100, 50, replace=False)
            medium_change[change_indices] += np.random.normal(0, 0.05, 50)
            medium_change = medium_change / np.linalg.norm(medium_change)

            success4, seq4 = db.add_content_version("sparsity_test", medium_change)
            results["medium_change_stored"] = success4
            results["medium_change_sequence"] = seq4

            timeline = db.get_content_timeline("sparsity_test")
            results["medium_change_is_delta"] = (
                timeline
                and seq4 in timeline.deltas
                and seq4 not in timeline.base_snapshots
            )

            # Test 5: Verify sparsity calculations in metadata
            if timeline:
                results["total_bases"] = len(timeline.base_snapshots)
                results["total_deltas"] = len(timeline.deltas)
                results["base_sequences"] = list(timeline.base_snapshots.keys())
                results["delta_sequences"] = list(timeline.deltas.keys())

                # Check delta metadata for sparsity information
                if seq2 in timeline.deltas:
                    delta2 = timeline.deltas[seq2]
                    results["small_change_sparsity_ratio"] = delta2.metadata.get(
                        "sparsity_ratio", 0
                    )
                    results["small_change_dimensions_changed"] = delta2.metadata.get(
                        "dimensions_changed", 0
                    )

                if seq4 in timeline.deltas:
                    delta4 = timeline.deltas[seq4]
                    results["medium_change_sparsity_ratio"] = delta4.metadata.get(
                        "sparsity_ratio", 0
                    )
                    results["medium_change_dimensions_changed"] = delta4.metadata.get(
                        "dimensions_changed", 0
                    )

        logger.info(f"Sparsity-based promotion results: {results}")
        return results

    def test_sparsity_with_wikipedia_simulation(self) -> Dict[str, Any]:
        """Test sparsity-based promotion with realistic Wikipedia evolution patterns."""
        logger.info("Testing sparsity promotion with Wikipedia simulation...")

        results = {}

        with TemporalVectorDatabase(
            self.test_db_path,
            embedding_dim=100,
            base_promotion_interval=15,  # Moderate interval
            base_promotion_sparsity_threshold=0.7,  # 70% threshold
        ) as db:

            # Use Wikipedia simulator to generate realistic evolution patterns
            article_id = "article_001"  # Science category
            versions = self.simulator.simulate_article_evolution(
                article_id, num_versions=12, time_span_days=200
            )

            promotion_reasons = []
            sparsity_promotions = 0
            interval_promotions = 0
            total_bases = 0

            for i, (timestamp, embedding) in enumerate(versions):
                success, seq_num = db.add_content_version(
                    content_id=f"wiki_{article_id}",
                    embedding=embedding,
                    timestamp=timestamp,
                    metadata={"edit_index": i, "simulated": True},
                )

                if success:
                    # Check what type of storage was used
                    timeline = db.get_content_timeline(f"wiki_{article_id}")
                    if timeline and seq_num in timeline.base_snapshots:
                        total_bases += 1
                        # Try to determine promotion reason from logs or position
                        if (
                            seq_num % 15 == 1
                        ):  # Interval-based (every 15th, starting from 1)
                            interval_promotions += 1
                            promotion_reasons.append(f"v{seq_num}:interval")
                        else:
                            sparsity_promotions += 1
                            promotion_reasons.append(f"v{seq_num}:sparsity")

            results["total_versions_stored"] = len(versions)
            results["total_bases_created"] = total_bases
            results["sparsity_promotions"] = sparsity_promotions
            results["interval_promotions"] = interval_promotions
            results["promotion_reasons"] = promotion_reasons

            # Analyze the final timeline
            timeline = db.get_content_timeline(f"wiki_{article_id}")
            if timeline:
                results["final_base_sequences"] = sorted(timeline.base_snapshots.keys())
                results["final_delta_sequences"] = sorted(timeline.deltas.keys())
                results["max_sequence"] = timeline.max_sequence_number

                # Check reconstruction quality
                reconstruction_results = []
                test_sequences = (
                    [3, 6, 9, 12]
                    if timeline.max_sequence_number >= 12
                    else [timeline.max_sequence_number]
                )

                for seq in test_sequences:
                    if seq <= timeline.max_sequence_number:
                        result = db.get_version(f"wiki_{article_id}", seq)
                        if result:
                            reconstruction_results.append(
                                {
                                    "sequence": seq,
                                    "reconstruction_cost": result.reconstruction_cost,
                                    "quality_score": result.quality_score,
                                    "base_used": result.base_sequence_used,
                                    "base_distance": result.base_distance,
                                }
                            )

                results["reconstruction_analysis"] = reconstruction_results
                if reconstruction_results:
                    results["avg_reconstruction_cost"] = np.mean(
                        [r["reconstruction_cost"] for r in reconstruction_results]
                    )
                    results["avg_quality_score"] = np.mean(
                        [r["quality_score"] for r in reconstruction_results]
                    )

        logger.info(f"Wikipedia sparsity simulation results: {results}")
        return results

    def test_sparsity_threshold_configuration(self) -> Dict[str, Any]:
        """Test different sparsity threshold configurations."""
        logger.info("Testing sparsity threshold configuration...")

        results = {}

        # Test with different threshold values
        thresholds_to_test = [0.3, 0.5, 0.7, 0.9]

        for threshold in thresholds_to_test:
            threshold_results = {}

            # Create a fresh database for each threshold test
            test_path = os.path.join(self.temp_dir, f"threshold_{threshold}_test.h5")

            with TemporalVectorDatabase(
                test_path,
                embedding_dim=100,
                base_promotion_interval=50,  # Very large to isolate sparsity effect
                base_promotion_sparsity_threshold=threshold,
            ) as db:

                base_embedding = np.random.normal(0, 1, 100)
                base_embedding = base_embedding / np.linalg.norm(base_embedding)

                # Store initial version
                db.add_content_version(f"threshold_test_{threshold}", base_embedding)

                # Test changes with known sparsity levels
                test_sparsities = [0.2, 0.4, 0.6, 0.8]  # 20%, 40%, 60%, 80%
                bases_created = 0

                for sparsity in test_sparsities:
                    test_embedding = base_embedding.copy()
                    num_changes = int(sparsity * 100)
                    change_indices = np.random.choice(100, num_changes, replace=False)
                    test_embedding[change_indices] += np.random.normal(
                        0, 0.05, num_changes
                    )
                    test_embedding = test_embedding / np.linalg.norm(test_embedding)

                    success, seq_num = db.add_content_version(
                        f"threshold_test_{threshold}", test_embedding
                    )

                    # Check if it was stored as base
                    timeline = db.get_content_timeline(f"threshold_test_{threshold}")
                    if timeline and seq_num in timeline.base_snapshots:
                        bases_created += 1
                        threshold_results[f"sparsity_{sparsity}_promoted"] = True
                    else:
                        threshold_results[f"sparsity_{sparsity}_promoted"] = False

                threshold_results["total_bases_created"] = bases_created
                threshold_results["threshold_value"] = threshold

                # Expected promotions: sparsities above threshold should be promoted
                expected_promotions = sum(1 for s in test_sparsities if s > threshold)
                threshold_results["expected_promotions"] = expected_promotions
                threshold_results["threshold_working_correctly"] = (
                    bases_created == expected_promotions
                )

            results[f"threshold_{threshold}"] = threshold_results

        # Overall threshold testing assessment
        working_thresholds = sum(
            1
            for t in thresholds_to_test
            if results[f"threshold_{t}"]["threshold_working_correctly"]
        )
        results["overall_threshold_test_pass"] = working_thresholds == len(
            thresholds_to_test
        )
        results["working_thresholds_count"] = working_thresholds

        logger.info(f"Sparsity threshold configuration results: {results}")
        return results

    def test_auto_increment_functionality(self) -> Dict[str, Any]:
        """Test automatic sequence number assignment and basic operations."""
        logger.info("Testing auto-increment functionality...")

        results = {}

        with TemporalVectorDatabase(
            self.test_db_path,
            embedding_dim=100,
            base_promotion_interval=3,  # Create base every 3 versions for testing
        ) as db:

            # Test 1: Add first version (should be base snapshot, sequence 1)
            embedding1 = np.random.normal(0, 1, 100)
            embedding1 = embedding1 / np.linalg.norm(embedding1)

            success1, seq1 = db.add_content_version("test_content", embedding1)
            results["first_version_stored"] = success1
            results["first_sequence_number"] = seq1

            # Test 2: Add second version (should be delta, sequence 2)
            embedding2 = embedding1.copy()
            embedding2[:10] += np.random.normal(0, 0.05, 10)
            embedding2 = embedding2 / np.linalg.norm(embedding2)

            success2, seq2 = db.add_content_version("test_content", embedding2)
            results["second_version_stored"] = success2
            results["second_sequence_number"] = seq2
            results["sequence_increment_correct"] = seq2 == seq1 + 1

            # Test 3: Add third version (should be delta, sequence 3)
            embedding3 = embedding2.copy()
            embedding3[10:20] += np.random.normal(0, 0.05, 10)
            embedding3 = embedding3 / np.linalg.norm(embedding3)

            success3, seq3 = db.add_content_version("test_content", embedding3)
            results["third_version_stored"] = success3
            results["third_sequence_number"] = seq3

            # Test 4: Add fourth version (should be base due to promotion interval)
            embedding4 = embedding3.copy()
            embedding4[20:30] += np.random.normal(0, 0.05, 10)
            embedding4 = embedding4 / np.linalg.norm(embedding4)

            success4, seq4 = db.add_content_version("test_content", embedding4)
            results["fourth_version_stored"] = success4
            results["fourth_sequence_number"] = seq4

            # Test 5: Verify timeline structure
            timeline = db.get_content_timeline("test_content")
            if timeline:
                results["timeline_loaded"] = True
                results["base_snapshots_count"] = len(timeline.base_snapshots)
                results["deltas_count"] = len(timeline.deltas)
                results["max_sequence"] = timeline.max_sequence_number
                results["base_promotion_working"] = (
                    1 in timeline.base_snapshots and 4 in timeline.base_snapshots
                )
            else:
                results["timeline_loaded"] = False

            # Test 6: Retrieve versions by different methods
            latest = db.get_latest_version("test_content")
            results["latest_version_retrieved"] = latest is not None
            if latest:
                results["latest_sequence"] = latest.sequence_number

            specific = db.get_version("test_content", 2)
            results["specific_version_retrieved"] = specific is not None
            if specific:
                results["specific_reconstruction_cost"] = specific.reconstruction_cost
                results["specific_quality_score"] = specific.quality_score

        logger.info(f"Auto-increment test results: {results}")
        return results

    def test_nearest_base_reconstruction(self) -> Dict[str, Any]:
        """Test nearest base snapshot selection and reconstruction optimization."""
        logger.info("Testing nearest base reconstruction...")

        results = {}

        with TemporalVectorDatabase(
            self.test_db_path, embedding_dim=100, base_promotion_interval=5
        ) as db:

            # Create a sequence with multiple base snapshots
            embeddings = []
            base_embedding = np.random.normal(0, 1, 100)
            base_embedding = base_embedding / np.linalg.norm(base_embedding)
            current_embedding = base_embedding.copy()

            # Store 12 versions (bases at 1, 6, 11 due to interval=5)
            for i in range(12):
                # Evolve the embedding slightly
                if i > 0:
                    current_embedding[:5] += np.random.normal(0, 0.02, 5)
                    current_embedding = current_embedding / np.linalg.norm(
                        current_embedding
                    )

                embeddings.append(current_embedding.copy())
                success, seq_num = db.add_content_version(
                    "reconstruction_test", current_embedding
                )
                if not success:
                    results[f"version_{i+1}_failed"] = True

            # Test reconstruction of various sequences
            timeline = db.get_content_timeline("reconstruction_test")
            if timeline:
                results["bases_created"] = list(timeline.base_snapshots.keys())

                # Test reconstruction of version 3 (should use base 1)
                result_v3 = db.get_version("reconstruction_test", 3)
                if result_v3:
                    results["v3_base_used"] = result_v3.base_sequence_used
                    results["v3_reconstruction_cost"] = result_v3.reconstruction_cost
                    results["v3_base_distance"] = result_v3.base_distance

                # Test reconstruction of version 7 (should use base 6)
                result_v7 = db.get_version("reconstruction_test", 7)
                if result_v7:
                    results["v7_base_used"] = result_v7.base_sequence_used
                    results["v7_reconstruction_cost"] = result_v7.reconstruction_cost
                    results["v7_base_distance"] = result_v7.base_distance

                # Test reconstruction of version 10 (should use base 6, not base 1)
                result_v10 = db.get_version("reconstruction_test", 10)
                if result_v10:
                    results["v10_base_used"] = result_v10.base_sequence_used
                    results["v10_reconstruction_cost"] = result_v10.reconstruction_cost
                    results["v10_base_distance"] = result_v10.base_distance

                # Verify nearest base selection is working
                results["nearest_base_optimization"] = (
                    result_v7
                    and result_v7.base_sequence_used == 6
                    and result_v10
                    and result_v10.base_sequence_used == 6
                )

        logger.info(f"Nearest base reconstruction results: {results}")
        return results

    def test_wikipedia_simulation_sequential(self) -> Dict[str, Any]:
        """Test sequential versioning with Wikipedia article simulation."""
        logger.info("Testing Wikipedia simulation with sequential versioning...")

        results = {}

        with TemporalVectorDatabase(
            self.test_db_path, embedding_dim=100, base_promotion_interval=8
        ) as db:

            stored_articles = 0
            total_versions = 0
            reconstruction_accuracies = []
            sequence_assignments = {}

            for i in range(3):  # Test 3 articles
                article_id = f"article_{i:03d}"
                versions = self.simulator.simulate_article_evolution(
                    article_id, num_versions=10
                )

                article_sequences = []

                # Store each version with auto-increment
                for j, (timestamp, embedding) in enumerate(versions):
                    success, seq_num = db.add_content_version(
                        content_id=article_id,
                        embedding=embedding,
                        timestamp=timestamp,
                        metadata={"edit_index": j, "simulated": True},
                    )

                    if success:
                        total_versions += 1
                        article_sequences.append(seq_num)

                sequence_assignments[article_id] = article_sequences
                stored_articles += 1

                # Test reconstruction quality for this article
                timeline = db.get_content_timeline(article_id)
                if timeline:
                    # Test reconstruction of various versions
                    test_sequences = (
                        [3, 6, 10]
                        if timeline.max_sequence_number >= 10
                        else [timeline.max_sequence_number]
                    )

                    for seq in test_sequences:
                        if seq <= timeline.max_sequence_number:
                            result = db.get_version(article_id, seq)
                            if result and result.quality_score:
                                reconstruction_accuracies.append(result.quality_score)

            results["articles_stored"] = stored_articles
            results["total_versions_stored"] = total_versions
            results["sequence_assignments"] = sequence_assignments

            if reconstruction_accuracies:
                results["avg_reconstruction_quality"] = float(
                    np.mean(reconstruction_accuracies)
                )
                results["min_reconstruction_quality"] = float(
                    np.min(reconstruction_accuracies)
                )
                results["max_reconstruction_quality"] = float(
                    np.max(reconstruction_accuracies)
                )
            else:
                results["avg_reconstruction_quality"] = 0.0

            # Test database statistics
            stats = db.get_database_statistics()
            results["database_stats"] = {
                "total_content_items": stats["total_content_items"],
                "avg_reconstruction_cost": stats["avg_reconstruction_cost"],
                "sequential_versioning": stats["sequential_versioning"],
            }

        logger.info(f"Wikipedia simulation results: {results}")
        return results

    def test_version_access_patterns(self) -> Dict[str, Any]:
        """Test different ways to access versions with sequential system."""
        logger.info("Testing version access patterns...")

        results = {}

        with TemporalVectorDatabase(self.test_db_path, embedding_dim=100) as db:

            # Create test content with known properties
            base_time = datetime.now() - timedelta(days=5)
            test_embeddings = []

            for i in range(6):
                embedding = np.random.normal(
                    i * 0.1, 1, 100
                )  # Slight shift each version
                embedding = embedding / np.linalg.norm(embedding)
                test_embeddings.append(embedding)

                timestamp = base_time + timedelta(days=i)
                success, seq_num = db.add_content_version(
                    "access_test",
                    embedding,
                    timestamp=timestamp,
                    metadata={"version_index": i},
                )
                results[f"version_{i+1}_stored"] = success
                results[f"version_{i+1}_sequence"] = seq_num

            # Test 1: Direct sequence access
            v3_result = db.get_version("access_test", 3)
            results["direct_sequence_access"] = v3_result is not None
            if v3_result:
                results["v3_sequence_number"] = v3_result.sequence_number
                results["v3_reconstruction_time"] = v3_result.reconstruction_time_ms

            # Test 2: Version ID string access
            v4_result = db.get_version_by_id("access_test_v4")
            results["version_id_string_access"] = v4_result is not None
            if v4_result:
                results["v4_from_string_sequence"] = v4_result.sequence_number

            # Test 3: Latest version access
            latest_result = db.get_latest_version("access_test")
            results["latest_version_access"] = latest_result is not None
            if latest_result:
                results["latest_sequence"] = latest_result.sequence_number

            # Test 4: Temporal access
            query_time = base_time + timedelta(days=2, hours=12)  # Between v3 and v4
            temporal_result = db.get_version_at_time("access_test", query_time)
            results["temporal_access"] = temporal_result is not None
            if temporal_result:
                results["temporal_sequence_found"] = temporal_result.sequence_number

            # Test 5: Range access
            range_results = db.get_version_range("access_test", 2, 4)
            results["range_access_count"] = len(range_results)
            results["range_access_sequences"] = [
                r.sequence_number for r in range_results
            ]

            # Test 6: Content statistics
            content_stats = db.get_content_statistics("access_test")
            results["content_statistics_available"] = "error" not in content_stats
            if "timeline_stats" in content_stats:
                timeline_stats = content_stats["timeline_stats"]
                results["max_sequence_in_stats"] = timeline_stats.get(
                    "max_sequence_number", timeline_stats.get("max_sequence", 0)
                )

        logger.info(f"Version access patterns results: {results}")
        return results

    def test_optimization_features(self) -> Dict[str, Any]:
        """Test optimization and analysis features."""
        logger.info("Testing optimization features...")

        results = {}

        with TemporalVectorDatabase(
            self.test_db_path,
            embedding_dim=100,
            base_promotion_interval=20,  # Large interval to test optimization
        ) as db:

            # Create content with suboptimal base placement
            base_embedding = np.random.normal(0, 1, 100)
            base_embedding = base_embedding / np.linalg.norm(base_embedding)
            current_embedding = base_embedding.copy()

            # Store 15 versions (only base at v1 due to large interval)
            for i in range(15):
                if i > 0:
                    # Make increasingly larger changes
                    change_magnitude = 0.01 + (i * 0.01)
                    current_embedding[:10] += np.random.normal(0, change_magnitude, 10)
                    current_embedding = current_embedding / np.linalg.norm(
                        current_embedding
                    )

                success, seq_num = db.add_content_version(
                    "optimization_test", current_embedding
                )
                if not success:
                    results[f"version_{i+1}_failed"] = True

            # Test optimization analysis
            optimization_analysis = db.optimize_content_bases(
                "optimization_test", max_cost=5
            )
            results["optimization_analysis_available"] = (
                "error" not in optimization_analysis
            )

            if "high_cost_versions" in optimization_analysis:
                results["high_cost_versions_found"] = len(
                    optimization_analysis["high_cost_versions"]
                )
                results["optimization_recommended"] = (
                    len(optimization_analysis["recommendations"]) > 0
                )

            # Test reconstruction cost analysis for later versions
            timeline = db.get_content_timeline("optimization_test")
            if timeline:
                v10_result = db.get_version("optimization_test", 10)
                v15_result = db.get_version("optimization_test", 15)

                if v10_result and v15_result:
                    results["v10_reconstruction_cost"] = v10_result.reconstruction_cost
                    results["v15_reconstruction_cost"] = v15_result.reconstruction_cost
                    results["reconstruction_cost_increases"] = (
                        v15_result.reconstruction_cost > v10_result.reconstruction_cost
                    )
                    results["v10_quality_score"] = v10_result.quality_score
                    results["v15_quality_score"] = v15_result.quality_score

            # Test integrity validation
            timeline = db.get_content_timeline("optimization_test")
            if timeline:
                from core.reconstruction_service import (
                    ReconstructionService,
                )

                reconstruction_service = ReconstructionService(
                    db.storage_engine, db.delta_computer
                )

                integrity_results = reconstruction_service.validate_timeline_integrity(
                    "optimization_test"
                )
                results["integrity_validation_passed"] = integrity_results["valid"]
                results["integrity_issues_count"] = len(
                    integrity_results.get("issues", [])
                )

        logger.info(f"Optimization features results: {results}")
        return results

    def test_persistence_and_recovery(self) -> Dict[str, Any]:
        """Test data persistence across database sessions with sequential versioning."""
        logger.info("Testing persistence and recovery...")

        results = {}

        # Session 1: Create and store data
        with TemporalVectorDatabase(self.test_db_path, embedding_dim=100) as db:
            embeddings = []
            for i in range(7):
                embedding = np.random.normal(i * 0.05, 1, 100)
                embedding = embedding / np.linalg.norm(embedding)
                embeddings.append(embedding)

                success, seq_num = db.add_content_version(
                    "persistence_test",
                    embedding,
                    metadata={"creation_session": 1, "version_index": i},
                )
                results[f"session1_v{i+1}_stored"] = success
                results[f"session1_v{i+1}_sequence"] = seq_num

            # Get timeline stats before closing
            timeline = db.get_content_timeline("persistence_test")
            if timeline:
                results["session1_max_sequence"] = timeline.max_sequence_number
                results["session1_base_count"] = len(timeline.base_snapshots)
                results["session1_delta_count"] = len(timeline.deltas)

        # Session 2: Reload and verify data persistence
        with TemporalVectorDatabase(self.test_db_path, embedding_dim=100) as db:
            # Test timeline loading
            timeline = db.get_content_timeline("persistence_test")
            results["session2_timeline_loaded"] = timeline is not None

            if timeline:
                results["session2_max_sequence"] = timeline.max_sequence_number
                results["session2_base_count"] = len(timeline.base_snapshots)
                results["session2_delta_count"] = len(timeline.deltas)
                results["sequence_numbers_preserved"] = (
                    timeline.max_sequence_number
                    == results.get("session1_max_sequence", 0)
                )

            # Test version reconstruction after reload
            v3_result = db.get_version("persistence_test", 3)
            v7_result = db.get_version("persistence_test", 7)

            results["session2_v3_reconstructed"] = v3_result is not None
            results["session2_v7_reconstructed"] = v7_result is not None

            if v3_result and v7_result:
                results["session2_reconstruction_quality_v3"] = v3_result.quality_score
                results["session2_reconstruction_quality_v7"] = v7_result.quality_score

                # Verify reconstruction accuracy by comparing to stored embeddings
                if len(embeddings) >= 3:
                    similarity_v3 = np.dot(embeddings[2], v3_result.embedding) / (
                        np.linalg.norm(embeddings[2])
                        * np.linalg.norm(v3_result.embedding)
                    )
                    results["session2_v3_accuracy"] = float(similarity_v3)

            # Add new version in session 2
            new_embedding = np.random.normal(0.5, 1, 100)
            new_embedding = new_embedding / np.linalg.norm(new_embedding)
            success, seq_num = db.add_content_version(
                "persistence_test", new_embedding, metadata={"creation_session": 2}
            )
            results["session2_new_version_added"] = success
            results["session2_new_sequence"] = seq_num
            results["session2_sequence_continues_correctly"] = (
                seq_num == results.get("session1_max_sequence", 0) + 1
            )

        logger.info(f"Persistence and recovery results: {results}")
        return results

    def run_comprehensive_test_suite(self) -> Dict[str, Any]:
        """Run all sequential versioning tests and compile results."""
        logger.info("Running comprehensive sequential versioning test suite...")

        comprehensive_results = {
            "sparsity_based_promotion": self.test_sparsity_based_promotion(),
            "sparsity_wikipedia_simulation": self.test_sparsity_with_wikipedia_simulation(),
            "sparsity_threshold_configuration": self.test_sparsity_threshold_configuration(),
            "auto_increment": self.test_auto_increment_functionality(),
            "nearest_base_reconstruction": self.test_nearest_base_reconstruction(),
            "wikipedia_simulation": self.test_wikipedia_simulation_sequential(),
            "version_access_patterns": self.test_version_access_patterns(),
            "optimization_features": self.test_optimization_features(),
            "persistence_recovery": self.test_persistence_and_recovery(),
            "timestamp": datetime.now().isoformat(),
        }

        # Overall assessment with safe key access
        sparsity_promotion_pass = (
            comprehensive_results["sparsity_based_promotion"].get(
                "large_change_is_base", False
            )
            and comprehensive_results["sparsity_based_promotion"].get(
                "small_change_is_delta", False
            )
            and comprehensive_results["sparsity_threshold_configuration"].get(
                "working_thresholds_count", 0
            )
            >= 1
        )

        auto_increment_pass = (
            comprehensive_results["auto_increment"]["first_version_stored"]
            and comprehensive_results["auto_increment"]["sequence_increment_correct"]
            and comprehensive_results["auto_increment"].get(
                "base_promotion_working", False
            )
        )

        reconstruction_pass = (
            comprehensive_results["nearest_base_reconstruction"].get(
                "nearest_base_optimization", False
            )
            and comprehensive_results["nearest_base_reconstruction"].get(
                "v7_reconstruction_cost", float("inf")
            )
            < 5
        )

        persistence_pass = comprehensive_results["persistence_recovery"].get(
            "sequence_numbers_preserved", False
        ) and comprehensive_results["persistence_recovery"].get(
            "session2_sequence_continues_correctly", False
        )

        access_patterns_pass = (
            comprehensive_results["version_access_patterns"]["direct_sequence_access"]
            and comprehensive_results["version_access_patterns"][
                "latest_version_access"
            ]
            and comprehensive_results["version_access_patterns"]["range_access_count"]
            == 3
        )

        wiki_sparsity_results = comprehensive_results["sparsity_wikipedia_simulation"]
        wikipedia_sparsity_pass = (
            wiki_sparsity_results["total_versions_stored"] > 0
            and wiki_sparsity_results["total_bases_created"]
            > 0  # At least interval promotions work
            and "reconstruction_analysis" in wiki_sparsity_results
            and len(wiki_sparsity_results["reconstruction_analysis"]) > 0
        )

        comprehensive_results["overall_assessment"] = {
            "sparsity_based_promotion": sparsity_promotion_pass,
            "auto_increment_functionality": auto_increment_pass,
            "nearest_base_reconstruction": reconstruction_pass,
            "data_persistence": persistence_pass,
            "version_access_patterns": access_patterns_pass,
            "wikipedia_sparsity_integration": wikipedia_sparsity_pass,
            "sequential_versioning_complete": (
                sparsity_promotion_pass
                and auto_increment_pass
                and reconstruction_pass
                and persistence_pass
                and access_patterns_pass
                and wikipedia_sparsity_pass
            ),
        }

        return comprehensive_results

    def cleanup(self):
        """Clean up temporary files."""
        if self.cleanup_temp and os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)
            logger.info(f"Cleaned up temporary directory: {self.temp_dir}")


def main():
    """Main function for sequential versioning testing."""
    print("TemporalVec: Sequential Versioning System Testing")
    print("=" * 60)

    tester = SequentialVersioningTester()

    try:
        results = tester.run_comprehensive_test_suite()

        # Display results
        print("\nSequential Versioning Test Results:")

        print(f"\n🔹 Sparsity-Based Promotion:")
        sparsity = results["sparsity_based_promotion"]
        print(
            f"  - Small Change (Low Sparsity): {'Delta ✓' if sparsity.get('small_change_is_delta', False) else 'Base ✗'}"
        )
        print(
            f"  - Large Change (High Sparsity): {'Base ✓' if sparsity.get('large_change_is_base', False) else 'Delta ✗'}"
        )
        print(f"  - Total Bases Created: {sparsity.get('total_bases', 0)}")
        print(f"  - Base Sequences: {sparsity.get('base_sequences', [])}")

        print(f"\n🔹 Sparsity Threshold Configuration:")
        threshold_test = results["sparsity_threshold_configuration"]
        print(
            f"  - Threshold Tests Passed: {'✓' if threshold_test.get('overall_threshold_test_pass', False) else '✗'}"
        )
        print(
            f"  - Working Thresholds: {threshold_test.get('working_thresholds_count', 0)}/4"
        )

        print(f"\n🔹 Wikipedia Sparsity Integration:")
        wiki_sparsity = results["sparsity_wikipedia_simulation"]
        print(f"  - Sparsity Promotions: {wiki_sparsity.get('sparsity_promotions', 0)}")
        print(f"  - Interval Promotions: {wiki_sparsity.get('interval_promotions', 0)}")
        print(f"  - Total Versions: {wiki_sparsity.get('total_versions_stored', 0)}")
        if "avg_reconstruction_cost" in wiki_sparsity:
            print(
                f"  - Avg Reconstruction Cost: {wiki_sparsity['avg_reconstruction_cost']:.2f}"
            )

        print(f"\n🔹 Auto-Increment Functionality:")
        auto = results["auto_increment"]
        print(
            f"  - First Version Stored: {'✓' if auto['first_version_stored'] else '✗'}"
        )
        print(
            f"  - Sequence Assignment: {auto['first_sequence_number']} → {auto['second_sequence_number']}"
        )
        print(
            f"  - Base Promotion Working: {'✓' if auto.get('base_promotion_working', False) else '✗'}"
        )
        print(
            f"  - Timeline Structure: {auto.get('base_snapshots_count', 0)} bases, {auto.get('deltas_count', 0)} deltas"
        )

        print(f"\n🔹 Nearest Base Reconstruction:")
        nearest = results["nearest_base_reconstruction"]
        print(f"  - Bases Created: {nearest.get('bases_created', [])}")
        print(
            f"  - V7 Uses Base: {nearest.get('v7_base_used')} (cost: {nearest.get('v7_reconstruction_cost', 'N/A')})"
        )
        print(
            f"  - V10 Uses Base: {nearest.get('v10_base_used')} (cost: {nearest.get('v10_reconstruction_cost', 'N/A')})"
        )
        print(
            f"  - Optimization Working: {'✓' if nearest.get('nearest_base_optimization', False) else '✗'}"
        )

        print(f"\n🔹 Wikipedia Simulation:")
        wiki = results["wikipedia_simulation"]
        print(f"  - Articles Stored: {wiki['articles_stored']}")
        print(f"  - Total Versions: {wiki['total_versions_stored']}")
        print(f"  - Avg Quality: {wiki['avg_reconstruction_quality']:.4f}")
        print(
            f"  - Sequential Versioning: {'✓' if wiki['database_stats']['sequential_versioning'] else '✗'}"
        )

        print(f"\n🔹 Version Access Patterns:")
        access = results["version_access_patterns"]
        print(
            f"  - Direct Sequence Access: {'✓' if access['direct_sequence_access'] else '✗'}"
        )
        print(
            f"  - Version ID String Access: {'✓' if access['version_id_string_access'] else '✗'}"
        )
        print(
            f"  - Latest Version Access: {'✓' if access['latest_version_access'] else '✗'}"
        )
        print(f"  - Range Access (2-4): {access['range_access_count']} versions")
        print(f"  - Temporal Access: {'✓' if access['temporal_access'] else '✗'}")

        print(f"\n🔹 Optimization Features:")
        opt = results["optimization_features"]
        print(f"  - High Cost Versions: {opt.get('high_cost_versions_found', 0)}")
        print(
            f"  - V15 Reconstruction Cost: {opt.get('v15_reconstruction_cost', 'N/A')}"
        )
        print(
            f"  - Quality Degradation: v10={opt.get('v10_quality_score', 0):.3f} → v15={opt.get('v15_quality_score', 0):.3f}"
        )
        print(
            f"  - Integrity Validation: {'✓' if opt.get('integrity_validation_passed', False) else '✗'}"
        )

        print(f"\n🔹 Persistence & Recovery:")
        persist = results["persistence_recovery"]
        print(
            f"  - Data Persisted: {'✓' if persist.get('sequence_numbers_preserved', False) else '✗'}"
        )
        print(
            f"  - Session Continuity: {'✓' if persist.get('session2_sequence_continues_correctly', False) else '✗'}"
        )
        print(
            f"  - V3 Accuracy After Reload: {persist.get('session2_v3_accuracy', 0):.4f}"
        )

        print(f"\n🔹 Overall Assessment:")
        assessment = results["overall_assessment"]
        print(
            f"  - Sparsity-Based Promotion: {'PASS' if assessment['sparsity_based_promotion'] else 'FAIL'}"
        )
        print(
            f"  - Auto-Increment Functionality: {'PASS' if assessment['auto_increment_functionality'] else 'FAIL'}"
        )
        print(
            f"  - Nearest Base Reconstruction: {'PASS' if assessment['nearest_base_reconstruction'] else 'FAIL'}"
        )
        print(
            f"  - Data Persistence: {'PASS' if assessment['data_persistence'] else 'FAIL'}"
        )
        print(
            f"  - Version Access Patterns: {'PASS' if assessment['version_access_patterns'] else 'FAIL'}"
        )
        print(
            f"  - Wikipedia Sparsity Integration: {'PASS' if assessment['wikipedia_sparsity_integration'] else 'FAIL'}"
        )
        print(
            f"  - Sequential Versioning: {'COMPLETE' if assessment['sequential_versioning_complete'] else 'INCOMPLETE'}"
        )

        if assessment["sequential_versioning_complete"]:
            print(
                f"\n🎉 Sequential versioning system with sparsity-based promotion is fully functional!"
            )
        else:
            print(
                f"\n⚠️  Sequential versioning system needs attention - check failing components above"
            )

    finally:
        tester.cleanup()


if __name__ == "__main__":
    main()
