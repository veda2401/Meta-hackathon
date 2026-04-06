"""tasks package — Episode grading and evaluation for the Power Grid Environment."""

from __future__ import annotations

from .graders import grade_episode, PassCriteria, CRITERIA, GRADE_THRESHOLDS

__all__ = [
    "grade_episode",       # one-shot grading function
    "PassCriteria",        # per-difficulty pass/fail criteria dataclass
    "CRITERIA",            # dict[Difficulty, PassCriteria]
    "GRADE_THRESHOLDS",    # [(score, letter), ...]
]
