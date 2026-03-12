# verify.py - heuristic verifier for checking if LLM answers are grounded in retrieved context
import re
from typing import Dict, List, Tuple

_SENT_SPLIT = re.compile(r"(?<=[.!?])\s+")
# citation must be at end of sentence (allow trailing whitespace)
_CIT_RE = re.compile(r"\[chunk_id=([0-9,\s]+)\][.\s]*$")

_STOP = {
    "the","and","that","with","have","has","from","this","they","them","into","only",
    "using","your","their","been","also","were","are","for","you","not","can","will",
}

def verify_grounding(answer: str, chunk_text_by_id: Dict[int, str]) -> Tuple[float, List[str]]:
    """
    Returns:
      groundedness in [0,1], list of unsupported sentences
    Heuristic verifier:
      - each sentence must have [chunk_id=...]
      - cited chunk_ids must exist
      - minimal keyword overlap between sentence and cited chunk text
    """
    sentences = [s.strip() for s in _SENT_SPLIT.split(answer) if s.strip()]
    if not sentences:
        return 0.0, ["<empty answer>"]

    unsupported: List[str] = []

    for s in sentences:
        m = _CIT_RE.search(s)
        if not m:
            unsupported.append(s)
            continue

        cited_ids = [int(x) for x in m.group(1).replace(" ", "").split(",") if x]
        if not cited_ids or any(cid not in chunk_text_by_id for cid in cited_ids):
            unsupported.append(s)
            continue

        cited_text = " ".join(chunk_text_by_id[cid].lower() for cid in cited_ids)

        tokens = [
            t for t in re.findall(r"[a-zA-Z0-9]{4,}", s.lower())
            if t not in _STOP
        ]
        overlap = sum(1 for t in set(tokens) if t in cited_text)

        if len(tokens) == 0:
            unsupported.append(s)
            continue
        overlap_ratio = overlap / max(len(set(tokens)), 1)
        # if overlap_ratio < 0.25:  # at least 25% of content words must appear in cited chunk
        min_threshold = 0.20 if len(set(tokens)) <= 4 else 0.25
        if overlap_ratio < min_threshold:
            unsupported.append(s)

    groundedness = 1.0 - (len(unsupported) / len(sentences))
    return groundedness, unsupported