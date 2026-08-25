"""Subject-aware cross-validation protocols.

Two protocols:

1. :class:`LeaveOneSubjectOutCV` — LOSO. Standard for biosignal evaluation
   (prevents subject-level leakage). Each fold holds out one subject.
2. :class:`LeaveOneDatasetOutCV` — LODO. For measuring cross-dataset
   generalization. Each fold holds out an entire dataset.

Both yield ``(train_indices, test_indices)`` pairs so that downstream
metric computation never leaks across folds.
"""

from __future__ import annotations

from collections.abc import Iterator

import numpy as np

__all__ = ["LeaveOneSubjectOutCV", "LeaveOneDatasetOutCV"]


class LeaveOneSubjectOutCV:
    """Leave-One-Subject-Out cross-validation.

    Examples
    --------
    >>> import numpy as np
    >>> subjects = np.array([0, 0, 0, 1, 1, 1, 2, 2, 2])
    >>> cv = LeaveOneSubjectOutCV()
    >>> folds = list(cv.split(subjects))
    >>> len(folds)
    3
    >>> # For fold 0: train on subjects 1,2; test on subject 0
    >>> train_idx, test_idx = folds[0]
    >>> sorted(int(s) for s in set(subjects[train_idx]))
    [1, 2]
    >>> sorted(int(s) for s in set(subjects[test_idx]))
    [0]
    """

    def split(
        self,
        subjects: np.ndarray | list[int],
    ) -> Iterator[tuple[np.ndarray, np.ndarray]]:
        """Generate LOSO splits.

        Parameters
        ----------
        subjects : np.ndarray or list of int
            Subject ID for each sample.

        Yields
        ------
        (train_idx, test_idx) : tuple of np.ndarray
            Indices into the input array. Train = all subjects except one,
            test = the held-out subject.
        """
        subjects = np.asarray(subjects)
        unique_subjects = np.unique(subjects)
        for subj in unique_subjects:
            test_mask = subjects == subj
            train_mask = ~test_mask
            train_idx = np.where(train_mask)[0]
            test_idx = np.where(test_mask)[0]
            yield train_idx, test_idx

    def get_n_splits(self, subjects: np.ndarray | list[int]) -> int:
        """Number of splits = number of unique subjects."""
        return len(np.unique(np.asarray(subjects)))


class LeaveOneDatasetOutCV:
    """Leave-One-Dataset-Out cross-validation.

    Each fold holds out an entire dataset (group of subjects).
    """

    def split(
        self,
        datasets: np.ndarray | list[int] | list[str],
    ) -> Iterator[tuple[np.ndarray, np.ndarray]]:
        """Generate LODO splits.

        Parameters
        ----------
        datasets : array-like
            Dataset ID (int or str) for each sample.

        Yields
        ------
        (train_idx, test_idx) : tuple of np.ndarray
        """
        datasets = np.asarray(datasets)
        unique = np.unique(datasets)
        for ds in unique:
            test_mask = datasets == ds
            train_mask = ~test_mask
            yield np.where(train_mask)[0], np.where(test_mask)[0]

    def get_n_splits(self, datasets: np.ndarray | list[int] | list[str]) -> int:
        """Number of splits = number of unique datasets."""
        return len(np.unique(np.asarray(datasets)))
