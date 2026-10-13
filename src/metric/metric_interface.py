from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List, Optional, Sequence, Dict, Any, Tuple

from data.data_sample_interface import DataSample


# ============================================================
# Result Object
# ============================================================

@dataclass
class MetricResult:
    """
    Standardized return object for all metrics.

    Must contain:
        - score: main scalar domain gap
        - details: optional sub-scores / diagnostics
    """
    score: float
    details: Dict[str, Any] = field(default_factory=dict)

    # bookkeeping
    n_A: int = 0
    n_B: int = 0
    paired: bool = False


# ============================================================
# Metric Base Interface
# ============================================================

class Metric(ABC):
    """
    Base class for all domain gap metrics.

    Design rules (from spec):

    - Metrics operate only on cached representations.
    - No file I/O.
    - No embedding computation.
    - Deterministic given inputs.
    - Must declare required_fields.
    - Must declare whether cross-domain pairing is required.
    """

    # Human-readable name (e.g., "MMD-RGB")
    name: str = "BaseMetric"

    # List of DataSample attributes required (e.g. ["rgb_emb"])
    required_fields: Sequence[str] = ()

    # Whether cross-domain pairing (A_i <-> B_i) is required
    requires_paired: bool = False

    # --------------------------------------------------------

    def validate_inputs(
        self,
        A: Sequence[DataSample],
        B: Sequence[DataSample],
        pairing: Optional[Sequence[Tuple[int, int]]] = None,
    ) -> None:
        """
        Validates required fields and pairing constraints.
        Called automatically before compute().
        """

        if len(A) == 0 or len(B) == 0:
            raise ValueError(f"{self.name}: empty input samples.")

        # Validate required fields
        for field in self.required_fields:
            for sample in A:
                if getattr(sample, field, None) is None:
                    raise ValueError(
                        f"{self.name}: missing required field '{field}' in A sample."
                    )
            for sample in B:
                if getattr(sample, field, None) is None:
                    raise ValueError(
                        f"{self.name}: missing required field '{field}' in B sample."
                    )

        # Validate pairing
        if self.requires_paired:
            if pairing is None:
                raise ValueError(
                    f"{self.name}: requires paired samples but no pairing provided."
                )
            if len(pairing) == 0:
                raise ValueError(
                    f"{self.name}: pairing provided but empty."
                )

    # --------------------------------------------------------

    def __call__(
        self,
        A: Sequence[DataSample],
        B: Sequence[DataSample],
        pairing: Optional[Sequence[Tuple[int, int]]] = None,
    ) -> MetricResult:
        """
        Wrapper that performs validation then calls compute().
        """
        self.validate_inputs(A, B, pairing)
        result = self.compute(A, B, pairing)

        result.n_A = len(A)
        result.n_B = len(B)
        result.paired = pairing is not None

        return result

    # --------------------------------------------------------

    @abstractmethod
    def compute(
        self,
        A: Sequence[DataSample],
        B: Sequence[DataSample],
        pairing: Optional[Sequence[Tuple[int, int]]] = None,
    ) -> MetricResult:
        """
        Implement the pure mathematical computation here.

        Must:
            - Not perform I/O
            - Not compute embeddings
            - Not modify DataSample
        """
        pass