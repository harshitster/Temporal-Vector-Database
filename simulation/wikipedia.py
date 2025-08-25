import numpy as np
from datetime import datetime, timedelta
from typing import Tuple, List, Dict


class WikipediaSimulator:
    """
    Simulates Wikipedia article evolution with realistic embedding changes.
    Generates sample data for testing the delta system.
    """

    def __init__(self, embedding_dim: int = 384, num_articles: int = 100):
        """
        Initialize the Wikipedia simulator.

        Args:
            embedding_dim: Dimensionality of embeddings
            num_articles: Number of articles to simulate
        """

        self.embedding_dim = embedding_dim
        self.num_articles = num_articles
        self.articles = {}

        self._initialize_articles()

    def _initialize_articles(self):
        """Initialize articles with base embeddings representing different topics."""
        np.random.seed(42)

        topic_categories = [
            "science",
            "history",
            "politics",
            "technology",
            "arts",
            "sports",
            "geography",
            "biology",
            "mathematics",
            "literature",
        ]

        for i in range(self.num_articles):
            article_id = f"article_{i:03d}"
            category = topic_categories[i % len(topic_categories)]

            base_embedding = self._generate_category_embedding(category)

            self.articles[article_id] = {
                "category": category,
                "base_embedding": base_embedding,
                "versions": [],
                "edit_dates": [],
            }

    def _generate_category_embedding(self, category: str) -> np.ndarray:
        category_seeds = {
            "science": 1,
            "technology": 2,
            "mathematics": 3,
            "biology": 4,
            "history": 10,
            "politics": 11,
            "geography": 12,
            "arts": 20,
            "literature": 21,
            "sports": 30,
        }

        seed = category_seeds.get(category, 0)
        np.random.seed(seed)

        embedding = np.random.normal(0, 1, self.embedding_dim)

        if category in ["science", "technology", "mathematics", "biology"]:
            embedding[: self.embedding_dim // 4] += 0.5
        elif category in ["arts", "literature"]:
            embedding[self.embedding_dim // 4 : self.embedding_dim // 2] += 0.5
        elif category in ["history", "politics", "geography"]:
            embedding[self.embedding_dim // 2 : 3 * self.embedding_dim // 4] += 0.5

        embedding = embedding / np.linalg.norm(embedding)

        return embedding

    def simulate_article_evolution(
        self, article_id: str, num_versions: int = 10, time_span_days: int = 365
    ) -> List[Tuple[datetime, np.ndarray]]:
        """
        Simulate the evolution of an article over time.

        Args:
            article_id: ID of the article to evolve
            num_versions: Number of versions to generate
            time_span_days: Time span over which evolution occurs

        Returns:
            List of (timestamp, embedding) tuples representing article versions
        """
        if article_id not in self.articles:
            raise ValueError(f"Article {article_id} not found")

        article = self.articles[article_id]
        base_embedding = article["base_embedding"].copy()

        timestamps = self._generate_realistic_timestamps(num_versions, time_span_days)

        versions = []
        current_embedding = base_embedding.copy()

        for i, timestamp in enumerate(timestamps):
            if i == 0:
                versions.append((timestamp, current_embedding.copy()))
            else:
                time_gap = (timestamp - timestamps[i - 1]).days
                current_embedding = self._evolve_embedding(
                    current_embedding, i / num_versions, time_gap
                )
                versions.append((timestamp, current_embedding.copy()))

        return versions

    def _generate_realistic_timestamps(
        self, num_versions: int, time_span_days: int
    ) -> List[datetime]:
        """Generate realistic timestamp patterns for article edits."""

        start_date = datetime.now() - timedelta(days=time_span_days)
        timestamps = [start_date]

        edit_pattern = np.random.choice(
            ["steady", "burst", "declining"], p=[0.4, 0.4, 0.2]
        )

        if edit_pattern == "steady":
            for i in range(1, num_versions):
                days_ahead = (i * time_span_days / num_versions) + np.random.normal(
                    0, time_span_days * 0.05
                )
                days_ahead = max(1, min(time_span_days, days_ahead))
                timestamps.append(start_date + timedelta(days=days_ahead))

        elif edit_pattern == "burst":
            first_third_versions = int(num_versions * 0.6)
            first_third_days = int(time_span_days * 0.3)

            for i in range(1, first_third_versions):
                days_ahead = (
                    i * first_third_days / first_third_versions
                ) + np.random.normal(0, 5)
                days_ahead = max(1, days_ahead)
                timestamps.append(start_date + timedelta(days=days_ahead))

            remaining_versions = num_versions - first_third_versions
            remaining_days = time_span_days - first_third_days
            for i in range(remaining_versions):
                days_ahead = (
                    first_third_days
                    + (i * remaining_days / remaining_versions)
                    + np.random.normal(0, 20)
                )
                days_ahead = max(first_third_days + 1, min(time_span_days, days_ahead))
                timestamps.append(start_date + timedelta(days=days_ahead))

        else:
            for i in range(1, num_versions):
                base_position = i / num_versions
                decay_factor = 1 - np.exp(-3 * base_position)
                days_ahead = decay_factor * time_span_days + np.random.normal(
                    0, time_span_days * 0.05
                )
                days_ahead = (
                    max(timestamps[-1].timestamp() - start_date.timestamp()) / 86400 + 1
                )
                days_ahead = min(time_span_days, days_ahead)
                timestamps.append(start_date + timedelta(days=days_ahead))

        return sorted(timestamps)

    def _evolve_embedding(
        self, current_embedding: np.ndarray, evolution_factor: float, time_gap_days: int
    ) -> np.ndarray:
        """
        Evolve an embedding to simulate content changes.

        Args:
            current_embedding: Current embedding state
            evolution_factor: How far through the evolution we are (0-1)
            time_gap_days: Days since last edit (affects change magnitude)

        Returns:
            Evolved embedding
        """
        # Combine evolution stage with time gaps for realistic change patterns

        # Early stage (0-0.3): New articles get major additions and structure
        # Middle stage (0.3-0.7): Steady content improvements and refinements
        # Late stage (0.7-1.0): Mostly maintenance, minor fixes, occasional updates

        if evolution_factor < 0.3:
            # Early article development - more structural changes
            if time_gap_days < 7:
                change_types = np.random.choice(
                    ["content_addition", "minor_edit", "small_addition"],
                    p=[0.4, 0.3, 0.3],
                )
            elif time_gap_days < 30:
                change_types = np.random.choice(
                    ["major_revision", "content_addition", "reorganization"],
                    p=[0.4, 0.4, 0.2],
                )
            else:
                change_types = np.random.choice(
                    ["major_revision", "content_addition", "reorganization"],
                    p=[0.5, 0.3, 0.2],
                )

        elif evolution_factor < 0.7:
            # Middle stage - balanced improvements
            if time_gap_days < 7:
                change_types = np.random.choice(
                    ["minor_edit", "typo_fix", "small_addition"], p=[0.5, 0.3, 0.2]
                )
            elif time_gap_days < 30:
                change_types = np.random.choice(
                    ["content_addition", "minor_edit", "reorganization"],
                    p=[0.4, 0.4, 0.2],
                )
            else:
                change_types = np.random.choice(
                    ["content_addition", "major_revision", "reorganization"],
                    p=[0.4, 0.3, 0.3],
                )

        else:
            # Late stage - mostly maintenance
            if time_gap_days < 7:
                change_types = np.random.choice(
                    ["typo_fix", "minor_edit", "small_addition"], p=[0.6, 0.3, 0.1]
                )
            elif time_gap_days < 30:
                change_types = np.random.choice(
                    ["minor_edit", "small_addition", "content_addition"],
                    p=[0.5, 0.3, 0.2],
                )
            else:
                # Even long gaps in mature articles tend toward moderate changes
                change_types = np.random.choice(
                    ["content_addition", "minor_edit", "major_revision"],
                    p=[0.5, 0.3, 0.2],
                )

        evolved = current_embedding.copy()

        # Apply changes based on selected type
        if change_types == "typo_fix":
            # Very small changes to very few dimensions
            num_changes = np.random.randint(1, max(2, self.embedding_dim // 50))
            indices = np.random.choice(self.embedding_dim, num_changes, replace=False)
            for idx in indices:
                evolved[idx] += np.random.normal(0, 0.005)

        elif change_types == "minor_edit":
            # Small random changes to few dimensions
            num_changes = np.random.randint(1, self.embedding_dim // 20)
            indices = np.random.choice(self.embedding_dim, num_changes, replace=False)
            for idx in indices:
                evolved[idx] += np.random.normal(0, 0.02)

        elif change_types == "small_addition":
            # Small addition - affects related dimensions
            start_idx = np.random.randint(0, self.embedding_dim - 20)
            for i in range(start_idx, min(start_idx + 20, self.embedding_dim)):
                evolved[i] += np.random.normal(0, 0.01)

        elif change_types == "content_addition":
            # Medium changes to moderate number of dimensions
            num_changes = np.random.randint(
                self.embedding_dim // 20, self.embedding_dim // 5
            )
            indices = np.random.choice(self.embedding_dim, num_changes, replace=False)
            for idx in indices:
                evolved[idx] += np.random.normal(0, 0.05)

        elif change_types == "reorganization":
            # Affects structure - changes to multiple sections
            section_size = self.embedding_dim // 4
            for section_start in [0, section_size, 2 * section_size, 3 * section_size]:
                if np.random.random() < 0.6:  # 60% chance each section changes
                    section_changes = np.random.randint(5, section_size // 3)
                    section_indices = np.random.choice(
                        range(
                            section_start,
                            min(section_start + section_size, self.embedding_dim),
                        ),
                        section_changes,
                        replace=False,
                    )
                    for idx in section_indices:
                        evolved[idx] += np.random.normal(0, 0.03)

        elif change_types == "major_revision":
            # Larger changes to many dimensions
            num_changes = np.random.randint(
                self.embedding_dim // 5, self.embedding_dim // 2
            )
            indices = np.random.choice(self.embedding_dim, num_changes, replace=False)
            for idx in indices:
                evolved[idx] += np.random.normal(0, 0.1)

        # Normalize to maintain embedding properties
        evolved = evolved / np.linalg.norm(evolved)

        return evolved

    def get_article_info(self, article_id: str) -> Dict:
        """Get information about a specific article."""
        if article_id not in self.articles:
            raise ValueError(f"Article {article_id} not found")

        return {
            "article_id": article_id,
            "category": self.articles[article_id]["category"],
            "base_embedding_norm": np.linalg.norm(
                self.articles[article_id]["base_embedding"]
            ),
            "embedding_dimension": self.embedding_dim,
        }

    def get_all_articles(self) -> List[str]:
        """Get list of all article IDs."""
        return list(self.articles.keys())

    def get_category_articles(self, category: str) -> List[str]:
        """Get all articles in a specific category."""
        return [
            article_id
            for article_id, info in self.articles.items()
            if info["category"] == category
        ]
