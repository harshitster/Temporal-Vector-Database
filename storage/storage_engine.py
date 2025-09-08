import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import h5py
import numpy as np
import faiss
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Any
from pathlib import Path
import logging
import json

from core.data_structures import BaseSnapshot, Delta, ContentTimeline

logger = logging.getLogger(__name__)


# Enhance the convert_numpy_types function
def convert_numpy_types(obj):
    """Convert numpy types to Python native types for JSON serialization."""
    if isinstance(obj, dict):
        return {key: convert_numpy_types(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_types(item) for item in obj]
    elif isinstance(obj, tuple):
        return tuple(convert_numpy_types(item) for item in obj)
    elif isinstance(obj, (np.integer, np.int64, np.int32)):
        return int(obj)
    elif isinstance(obj, (np.floating, np.float64, np.float32)):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, np.bool_):
        return bool(obj)
    elif hasattr(obj, "item"):  # Handle scalar numpy types
        return obj.item()
    else:
        return obj


class TemporalStorageEngine:
    """
    HDF5-based storage engine with sequential versioning support.
    Optimized for O(1) version lookups and nearest base snapshot selection.
    """

    def __init__(
        self, storage_path: str = "temporal_vector_db.h5", embedding_dim: int = 384
    ):
        """
        Initialize storage engine with HDF5 backend and FAISS index.

        Args:
            storage_path: Path to HDF5 storage file
            embedding_dim: Dimension of embedding vectors
        """
        self.storage_path = Path(storage_path)
        self.embedding_dim = embedding_dim
        self.h5_file = None
        self.faiss_index: faiss.IndexFlatIP
        self.base_snapshot_mapping: Dict[int, Tuple[str, int]] = {}

        self._initialize_storage()
        self._initialize_faiss_index()

    def _initialize_storage(self):
        """Initialize HDF5 file structure for sequential versioning."""
        with h5py.File(self.storage_path, "a") as f:
            if "base_snapshots" not in f:
                f.create_group("base_snapshots")
            if "deltas" not in f:
                f.create_group("deltas")
            if "content_metadata" not in f:
                f.create_group("content_metadata")
            if "metadata" not in f:
                metadata_group = f.create_group("metadata")
                metadata_group.attrs["embedding_dim"] = self.embedding_dim
                metadata_group.attrs["created_at"] = datetime.now().isoformat()
                metadata_group.attrs["version"] = "2.0"

    def _initialize_faiss_index(self):
        """Initialize FAISS index for base snapshot similarity search."""
        self.faiss_index = faiss.IndexFlatIP(self.embedding_dim)
        self.base_snapshot_mapping = {}
        self._rebuild_faiss_index()

    def _rebuild_faiss_index(self):
        """Rebuild FAISS index from stored base snapshots."""
        embeddings = []
        mapping = {}

        with h5py.File(self.storage_path, "r") as f:
            if "base_snapshots" in f:
                base_group = f["base_snapshots"]
                for content_id, content_group in base_group.items():
                    for seq_str, version_group in content_group.items():
                        seq_num = int(seq_str.replace("v", ""))
                        embedding = version_group["embedding"][:]
                        embedding = embedding / np.linalg.norm(embedding)
                        embeddings.append(embedding)
                        mapping[len(embeddings) - 1] = (content_id, seq_num)

        if embeddings:
            embeddings_array = np.array(embeddings).astype("float32")
            self.faiss_index.add(embeddings_array)
            self.base_snapshot_mapping = mapping

        logger.info(f"FAISS index rebuilt with {len(embeddings)} base snapshots")

    def store_base_snapshot(self, snapshot: BaseSnapshot) -> bool:
        """
        Store a base snapshot with sequential versioning

        Args:
            snapshot: BaseSnapshot object to store

        Returns:
            bool: Success status
        """
        try:
            with h5py.File(self.storage_path, "a") as f:
                base_group = f["base_snapshots"]

                if snapshot.content_id not in base_group:
                    base_group.create_group(snapshot.content_id)

                content_group = base_group[snapshot.content_id]
                seq_key = f"v{snapshot.sequence_number}"

                delta_group = f.get("deltas", {})
                if (
                    snapshot.content_id in delta_group
                    and seq_key in delta_group[snapshot.content_id]
                ):
                    logger.error(
                        f"Sequence number {snapshot.sequence_number} already exists as delta"
                    )
                    return False

                if seq_key not in content_group:
                    version_group = content_group.create_group(seq_key)
                    version_group.create_dataset("embedding", data=snapshot.embedding)
                    version_group.attrs["timestamp"] = snapshot.timestamp.isoformat()
                    version_group.attrs["content_id"] = snapshot.content_id
                    version_group.attrs["sequence_number"] = snapshot.sequence_number
                    version_group.attrs["version_id"] = snapshot.version_id

                    clean_metadata = convert_numpy_types(snapshot.metadata)
                    version_group.attrs["metadata"] = json.dumps(clean_metadata)

                    normalized_embedding = snapshot.embedding / np.linalg.norm(
                        snapshot.embedding
                    )
                    self.faiss_index.add(
                        normalized_embedding.reshape(1, -1).astype("float32")
                    )

                    new_index = self.faiss_index.ntotal - 1
                    self.base_snapshot_mapping[new_index] = (
                        snapshot.content_id,
                        snapshot.sequence_number,
                    )

                    self._update_content_metadata_internal(
                        f, snapshot.content_id, snapshot.sequence_number, True
                    )

                    logger.info(f"Stored base snapshot: {snapshot.version_id}")
                    return True
                else:
                    logger.warning(
                        f"Base snapshot {snapshot.version_id} already exists"
                    )
                    return False
        except Exception as e:
            logger.error(f"Error storing base snapshot {snapshot.version_id}: {e}")
            return False

    def store_delta(self, delta: Delta) -> bool:
        """
        Store a delta with sequential versioning validation

        Args:
            delta: Delta object to store

        Returns:
            bool: Success status
        """
        try:
            with h5py.File(self.storage_path, "a") as f:
                delta_group = f["deltas"]

                if delta.content_id not in delta_group:
                    delta_group.create_group(delta.content_id)

                content_group = delta_group[delta.content_id]
                seq_key = f"v{delta.sequence_number}"

                if seq_key not in content_group:
                    version_group = content_group.create_group(seq_key)

                    if delta.sparse_delta:
                        indices = list(delta.sparse_delta.keys())
                        values = list(delta.sparse_delta.values())
                        version_group.create_dataset("sparse_indices", data=indices)
                        version_group.create_dataset("sparse_values", data=values)
                    else:
                        version_group.create_dataset("sparse_indices", data=[])
                        version_group.create_dataset("sparse_values", data=[])

                    version_group.attrs["content_id"] = delta.content_id
                    version_group.attrs["sequence_number"] = delta.sequence_number
                    version_group.attrs["from_sequence_number"] = (
                        delta.from_sequence_number
                    )
                    version_group.attrs["timestamp"] = delta.timestamp.isoformat()
                    version_group.attrs["change_magnitude"] = delta.change_magnitude
                    version_group.attrs["version_id"] = delta.version_id

                    clean_metadata = convert_numpy_types(delta.metadata)
                    version_group.attrs["metadata"] = json.dumps(clean_metadata)

                    self._update_content_metadata_internal(
                        f, delta.content_id, delta.sequence_number, False
                    )

                    logger.debug(f"Stored delta: {delta.version_id}")
                    return True
                else:
                    logger.warning(f"Delta {delta.version_id} already exists")
                    return False
        except Exception as e:
            logger.error(f"Error storing delta {delta.version_id}: {e}")
            return False

    def _update_content_metadata(
        self, content_id: str, sequence_number: int, is_base: bool
    ):
        """Update metadata tracking for content - external interface."""
        with h5py.File(self.storage_path, "a") as f:
            self._update_content_metadata_internal(
                f, content_id, sequence_number, is_base
            )

    def _update_content_metadata_internal(
        self, h5_file, content_id: str, sequence_number: int, is_base: bool
    ):
        """Update metadata tracking for content - internal method with file handle."""
        metadata_group = h5_file["content_metadata"]

        if content_id not in metadata_group:
            content_meta = metadata_group.create_group(content_id)
            content_meta.attrs["max_sequence"] = sequence_number
            content_meta.attrs["base_snapshots"] = json.dumps(
                [sequence_number] if is_base else []
            )
        else:
            content_meta = metadata_group[content_id]
            current_max = content_meta.attrs["max_sequence"]
            content_meta.attrs["max_sequence"] = max(current_max, sequence_number)

            if is_base:
                current_bases = json.loads(
                    content_meta.attrs.get("base_snapshots", "[]")
                )
                current_bases.append(sequence_number)
                current_bases.sort()
                content_meta.attrs["base_snapshots"] = json.dumps(current_bases)

    def load_base_snapshot(
        self, content_id: str, sequence_number: int
    ) -> Optional[BaseSnapshot]:
        """
        Load a base snapshot by content ID and sequence number

        Args:
            content_id: Content identifier
            sequence_number: Sequence number

        Returns:
            BaseSnapshot object or None if not found
        """
        try:
            with h5py.File(self.storage_path, "r") as f:
                if "base_snapshots" not in f:
                    return None

                base_group = f["base_snapshots"]
                if content_id not in base_group:
                    return None

                content_group = base_group[content_id]
                seq_key = f"v{sequence_number}"

                if seq_key not in content_group:
                    return None

                version_group = content_group[seq_key]

                embedding = version_group["embedding"][:]
                timestamp = datetime.fromisoformat(version_group.attrs["timestamp"])
                metadata = json.loads(version_group.attrs.get("metadata", "{}"))
                version_id = version_group.attrs.get(
                    "version_id", f"{content_id}_v{sequence_number}"
                )

                return BaseSnapshot(
                    content_id=content_id,
                    sequence_number=sequence_number,
                    timestamp=timestamp,
                    embedding=embedding,
                    metadata=metadata,
                    version_id=version_id,
                )
        except Exception as e:
            logger.error(
                f"Error loading base snapshot {content_id}_v{sequence_number}: {e}"
            )
            return None

    def load_delta(self, content_id: str, sequence_number: int) -> Optional[Delta]:
        """
        Load a delta by content ID and sequence number.

        Args:
            content_id: Content identifier
            sequence_number: Target sequence number

        Returns:
            Delta object or None if not found
        """
        try:
            with h5py.File(self.storage_path, "r") as f:
                if "deltas" not in f:
                    return None

                delta_group = f["deltas"]
                if content_id not in delta_group:
                    return None

                content_group = delta_group[content_id]
                seq_key = f"v{sequence_number}"

                if seq_key not in content_group:
                    return None

                version_group = content_group[seq_key]

                indices = version_group["sparse_indices"][:]
                values = version_group["sparse_values"][:]
                sparse_delta = dict(zip(indices, values))

                from_sequence_number = version_group.attrs["from_sequence_number"]
                timestamp = datetime.fromisoformat(version_group.attrs["timestamp"])
                change_magnitude = version_group.attrs["change_magnitude"]
                metadata = json.loads(version_group.attrs.get("metadata", "{}"))
                version_id = version_group.attrs.get(
                    "version_id", f"{content_id}_v{sequence_number}"
                )

                return Delta(
                    content_id=content_id,
                    sequence_number=sequence_number,
                    from_sequence_number=from_sequence_number,
                    timestamp=timestamp,
                    sparse_delta=sparse_delta,
                    change_magnitude=change_magnitude,
                    metadata=metadata,
                    version_id=version_id,
                )
        except Exception as e:
            logger.error(f"Error loading delta {content_id}_v{sequence_number}: {e}")
            return None

    def get_content_timeline(self, content_id: str) -> Optional[ContentTimeline]:
        timeline = ContentTimeline(content_id=content_id)
        found_data = False

        try:
            with h5py.File(self.storage_path, "r") as f:
                if "base_snapshots" in f:
                    base_group = f["base_snapshots"]
                    if content_id in base_group:
                        content_group = base_group[content_id]
                        seq_keys = list(content_group.keys())
                        seq_nums = [int(key.replace("v", "")) for key in seq_keys]
                        sorted_indices = np.argsort(seq_nums)

                        for i in sorted_indices:
                            snapshot = self.load_base_snapshot(content_id, seq_nums[i])
                            if snapshot:
                                timeline.add_base_snapshot(snapshot)
                                found_data = True

                if "deltas" in f:
                    delta_group = f["deltas"]
                    if content_id in delta_group:
                        content_group = delta_group[content_id]
                        seq_keys = list(content_group.keys())
                        seq_nums = [int(key.replace("v", "")) for key in seq_keys]
                        sorted_indices = np.argsort(seq_nums)

                        for i in sorted_indices:
                            delta = self.load_delta(content_id, seq_nums[i])
                            if delta:
                                timeline.add_delta(delta)
                                found_data = True

        except Exception as e:
            logger.error("Error loading timeline for %s: %s", content_id, e)
            return None

        return timeline if found_data else None

    def get_next_sequence_number(self, content_id: str) -> int:
        """
        Get the next sequence number for new content versions.

        Args:
            content_id: Content identifier

        Returns:
            Next sequence number (starting from 1)
        """
        try:
            with h5py.File(self.storage_path, "r") as f:
                if "content_metadata" in f:
                    metadata_group = f["content_metadata"]
                    if content_id in metadata_group:
                        content_meta = metadata_group[content_id]
                        return content_meta.attrs["max_sequence"] + 1
                return 1
        except Exception as e:
            logger.error("Error getting next sequence number for %s: %s", content_id, e)
            return 1

    def search_similar_snapshots(
        self, query_embedding: np.ndarray, k: int = 5
    ) -> List[Tuple[str, int, float]]:
        """
        Find similar base snapshots using FAISS similarity search.
        Now returns sequence numbers instead of version_id strings.

        Args:
            query_embedding: Query embedding vector
            k: Number of similar snapshots to return

        Returns:
            List of (content_id, sequence_number, similarity_score) tuples
        """
        if self.faiss_index.ntotal == 0:
            return []

        query_normalized = query_embedding / np.linalg.norm(query_embedding)
        query_normalized = query_normalized.reshape(1, -1).astype("float32")

        similarities, indices = self.faiss_index.search(
            query_normalized, min(k, self.faiss_index.ntotal)
        )

        results = []
        for sim, idx in zip(similarities[0], indices[0]):
            if idx in self.base_snapshot_mapping and sim > 0:
                content_id, sequence_number = self.base_snapshot_mapping[idx]
                results.append((content_id, sequence_number, float(sim)))

        return results

    def get_storage_statistics(self) -> Dict[str, Any]:
        """
        Get enhanced storage statistics with sequential versioning metrics.

        Returns:
            Dictionary with storage statistics
        """
        stats = {
            "storage_path": str(self.storage_path),
            "file_size_mb": 0,
            "total_base_snapshots": 0,
            "total_deltas": 0,
            "content_timelines": 0,
            "faiss_index_size": self.faiss_index.ntotal,
            "version_schema": "sequential_v2.0",
        }

        if self.storage_path.exists():
            stats["file_size_mb"] = self.storage_path.stat().st_size / (1024 * 1024)

        try:
            with h5py.File(self.storage_path, "r") as f:
                content_stats = {}
                if "base_snapshots" in f:
                    base_group = f["base_snapshots"]
                    for content_id, content_group in base_group.items():
                        snapshot_count = len(content_group.keys())
                        stats["total_base_snapshots"] += snapshot_count
                        content_stats[content_id] = {
                            "base_snapshots": snapshot_count,
                            "deltas": 0,
                        }
                    stats["content_timelines"] = len(base_group.keys())

                if "deltas" in f:
                    delta_group = f["deltas"]
                    for content_id, content_group in delta_group.items():
                        delta_count = len(content_group.keys())
                        stats["total_deltas"] += delta_count
                        if content_id in content_stats:
                            content_stats[content_id]["deltas"] = delta_count

                if content_stats:
                    stats["per_content_stats"] = content_stats
                    stats["avg_versions_per_content"] = np.mean(
                        [
                            info["base_snapshots"] + info["deltas"]
                            for info in content_stats.values()
                        ]
                    )
        except Exception as e:
            logger.error(f"Error getting storage statistics: {e}")

        return stats

    def close(self):
        """Close storage engine and cleanup resources."""
        logger.info("Closing sequential storage engine")

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
