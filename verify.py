# verify.py - heuristic verifier for checking if LLM answers are grounded in retrieved context
import re
from typing import Dict, List, Tuple

_CIT_ANY_RE = re.compile(r"\[chunk_id=([0-9,\s]+)\]")

_STOP = {
    "the", "and", "that", "with", "have", "has", "from", "this", "they", "them",
    "into", "only", "using", "your", "their", "been", "also", "were", "are",
    "for", "you", "not", "can", "will",
}

_META_PATTERNS = [
    r"\b i do not have enough information\b",
    r"\b i cannot determine\b",
    r"\b i do not see explicit evidence\b",
    r"\b does not clearly indicate\b",
    r"\b not clearly established\b",
    r"\b there is no mention\b",
    r"\b however, i can provide some relevant information\b",
]


def split_units(answer: str) -> List[str]:
    parts = []
    for block in answer.splitlines():
        block = block.strip()
        if not block:
            continue
        subparts = re.split(r"(?<=[.!?])\s+", block)
        parts.extend(s.strip() for s in subparts if s.strip())
    return parts


def is_meta_sentence(sentence: str) -> bool:
    s = sentence.lower()
    return any(re.search(pattern, s) for pattern in _META_PATTERNS)


def parse_cited_ids(sentence: str) -> List[int]:
    """
    Extract all cited chunk ids from all [chunk_id=...] occurrences in a sentence.
    Deduplicates while preserving order.
    """
    matches = _CIT_ANY_RE.findall(sentence)
    cited_ids: List[int] = []

    for match in matches:
        for x in match.replace(" ", "").split(","):
            if x:
                cited_ids.append(int(x))

    # Deduplicate while preserving order
    return list(dict.fromkeys(cited_ids))


def verify_grounding(answer: str, chunk_text_by_id: Dict[int, str]) -> Tuple[float, List[str]]:
    sentences = [s.strip() for s in split_units(answer) if s.strip()]
    if not sentences:
        return 0.0, ["<empty answer>"]

    unsupported: List[str] = []
    verifiable_count = 0

    for s in sentences:
        # Remove citations first, then inspect the actual sentence content
        sentence_wo_citation = _CIT_ANY_RE.sub("", s).strip()
        sentence_wo_citation_lower = sentence_wo_citation.lower()

        # Skip epistemic/meta sentences from strict grounding checks
        if is_meta_sentence(sentence_wo_citation):
            continue

        # Ignore very short structural fragments
        plain = re.sub(r"\s+", " ", sentence_wo_citation).strip(" -•*:\n\t")
        if len(plain) < 8:
            continue

        verifiable_count += 1

        # Find all cited chunk ids in the sentence
        cited_ids = parse_cited_ids(s)
        if not cited_ids:
            unsupported.append(s)
            continue

        # All cited ids must exist
        if any(cid not in chunk_text_by_id for cid in cited_ids):
            unsupported.append(s)
            continue

        # Join all cited chunk text and tokenize it
        cited_text = " ".join(chunk_text_by_id[cid].lower() for cid in cited_ids)
        cited_tokens = set(
            t for t in re.findall(r"[a-zA-Z0-9]{4,}", cited_text)
            if t not in _STOP
        )

        # Tokenize the sentence (without citations)
        tokens = [
            t for t in re.findall(r"[a-zA-Z0-9]{4,}", sentence_wo_citation_lower)
            if t not in _STOP
        ]

        # If there are no meaningful content tokens, skip strict scoring
        if not tokens:
            continue

        unique_tokens = set(tokens)
        overlap = sum(1 for t in unique_tokens if t in cited_tokens)
        overlap_ratio = overlap / max(len(unique_tokens), 1)

        # Slightly lenient threshold for short claims
        min_threshold = 0.20 if len(unique_tokens) <= 4 else 0.25
        if overlap_ratio < min_threshold:
            unsupported.append(s)

    if verifiable_count == 0:
        return 0.5, []

    groundedness = 1.0 - (len(unsupported) / verifiable_count)
    return groundedness, unsupported