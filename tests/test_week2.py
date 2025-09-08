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
    Tests auto-increment functionality, nearest base reconstruction, and performance optimizations.
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
        self.simulator = WikipediaSimulator(embedding_dim=100, num_articles=3)

        logger.info(
            f"Initialized sequential tester with storage at {self.test_db_path}"
        )

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

            for i in range(2):  # Test 2 articles
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
                results["max_sequence_in_stats"] = content_stats["timeline_stats"][
                    "max_sequence_number"
                ]

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
            "auto_increment": self.test_auto_increment_functionality(),
            "nearest_base_reconstruction": self.test_nearest_base_reconstruction(),
            "wikipedia_simulation": self.test_wikipedia_simulation_sequential(),
            "version_access_patterns": self.test_version_access_patterns(),
            "optimization_features": self.test_optimization_features(),
            "persistence_recovery": self.test_persistence_and_recovery(),
            "timestamp": datetime.now().isoformat(),
        }

        # Overall assessment with safe key access
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

        comprehensive_results["overall_assessment"] = {
            "auto_increment_functionality": auto_increment_pass,
            "nearest_base_reconstruction": reconstruction_pass,
            "data_persistence": persistence_pass,
            "version_access_patterns": access_patterns_pass,
            "sequential_versioning_complete": (
                auto_increment_pass
                and reconstruction_pass
                and persistence_pass
                and access_patterns_pass
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

        print(f"\nAuto-Increment Functionality:")
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

        print(f"\nNearest Base Reconstruction:")
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

        print(f"\nWikipedia Simulation:")
        wiki = results["wikipedia_simulation"]
        print(f"  - Articles Stored: {wiki['articles_stored']}")
        print(f"  - Total Versions: {wiki['total_versions_stored']}")
        print(f"  - Avg Quality: {wiki['avg_reconstruction_quality']:.4f}")
        print(
            f"  - Sequential Versioning: {'✓' if wiki['database_stats']['sequential_versioning'] else '✗'}"
        )

        print(f"\nVersion Access Patterns:")
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

        print(f"\nOptimization Features:")
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

        print(f"\nPersistence & Recovery:")
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

        print(f"\nOverall Assessment:")
        assessment = results["overall_assessment"]
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
            f"  - Sequential Versioning: {'COMPLETE' if assessment['sequential_versioning_complete'] else 'INCOMPLETE'}"
        )

        if assessment["sequential_versioning_complete"]:
            print(
                f"\nSequential versioning system is fully functional with optimized performance!"
            )
        else:
            print(
                f"\nSequential versioning system needs attention - check failing components above"
            )

    finally:
        tester.cleanup()


if __name__ == "__main__":
    main()
