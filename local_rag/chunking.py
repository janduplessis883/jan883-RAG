from __future__ import annotations

from dataclasses import dataclass
import math
import re


def cosine_similarity(left: list[float], right: list[float]) -> float:
    numerator = sum(a * b for a, b in zip(left, right))
    left_norm = math.sqrt(sum(a * a for a in left))
    right_norm = math.sqrt(sum(b * b for b in right))
    if left_norm == 0 or right_norm == 0:
        return 0.0
    return numerator / (left_norm * right_norm)


def split_sentences(text: str) -> list[str]:
    normalized = re.sub(r"\s+", " ", text).strip()
    if not normalized:
        return []
    return [part.strip() for part in re.split(r"(?<=[.!?])\s+", normalized) if part.strip()]


def average_embedding(vectors: list[list[float]]) -> list[float]:
    if not vectors:
        return []
    length = len(vectors[0])
    totals = [0.0] * length
    for vector in vectors:
        for index, value in enumerate(vector):
            totals[index] += value
    return [value / len(vectors) for value in totals]


@dataclass(slots=True)
class SemanticChunk:
    text: str
    char_count: int


class SemanticChunker:
    def __init__(self, config: dict) -> None:
        self.target_chars = int(config["target_chars"])
        self.min_chars = int(config["min_chars"])
        self.max_chars = int(config["max_chars"])
        self.hard_split_chars = int(config["hard_split_chars"])
        self.threshold = float(config["semantic_merge_threshold"])

    def chunk(self, text: str, embedder) -> list[SemanticChunk]:
        sentences = split_sentences(text)
        if not sentences:
            return []
        if len(text) <= self.max_chars:
            return [SemanticChunk(text=text.strip(), char_count=len(text.strip()))]

        embeddings = embedder(sentences)
        chunks: list[SemanticChunk] = []
        current_sentences: list[str] = []
        current_vectors: list[list[float]] = []

        for sentence, vector in zip(sentences, embeddings):
            if not current_sentences:
                current_sentences.append(sentence)
                current_vectors.append(vector)
                continue

            current_text = " ".join(current_sentences)
            current_chars = len(current_text)
            centroid = average_embedding(current_vectors)
            similarity = cosine_similarity(centroid, vector)

            should_merge = (
                current_chars < self.min_chars
                or (current_chars < self.target_chars and similarity >= self.threshold)
                or (current_chars < self.max_chars and similarity >= self.threshold + 0.05)
            )

            if should_merge:
                current_sentences.append(sentence)
                current_vectors.append(vector)
                continue

            chunks.extend(self._finalize_chunk(" ".join(current_sentences)))
            current_sentences = [sentence]
            current_vectors = [vector]

        if current_sentences:
            chunks.extend(self._finalize_chunk(" ".join(current_sentences)))

        return chunks

    def _finalize_chunk(self, text: str) -> list[SemanticChunk]:
        text = text.strip()
        if len(text) <= self.hard_split_chars:
            return [SemanticChunk(text=text, char_count=len(text))]

        pieces: list[SemanticChunk] = []
        start = 0
        while start < len(text):
            end = min(start + self.target_chars, len(text))
            pieces.append(SemanticChunk(text=text[start:end].strip(), char_count=len(text[start:end].strip())))
            start = end
        return [piece for piece in pieces if piece.text]


class FixedChunker:
    def __init__(self, chunk_size: int, overlap: int = 0) -> None:
        self.chunk_size = int(chunk_size)
        self.overlap = int(overlap)
        if self.chunk_size <= 0:
            raise ValueError("Fixed chunk size must be greater than zero.")
        if self.overlap < 0 or self.overlap >= self.chunk_size:
            raise ValueError("Fixed chunk overlap must be at least zero and smaller than the chunk size.")

    def chunk(self, text: str, embedder=None) -> list[SemanticChunk]:
        normalized = text.strip()
        if not normalized:
            return []

        step = self.chunk_size - self.overlap
        chunks: list[SemanticChunk] = []
        for start in range(0, len(normalized), step):
            chunk_text = normalized[start : start + self.chunk_size].strip()
            if chunk_text:
                chunks.append(SemanticChunk(text=chunk_text, char_count=len(chunk_text)))
            if start + self.chunk_size >= len(normalized):
                break
        return chunks
