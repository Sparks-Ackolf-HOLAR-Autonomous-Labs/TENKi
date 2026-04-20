"""
The octahedral symmetry group O_h acting on the RGB unit cube [0,1]^3.

The RGB cube has 48 symmetries:
  - 6 permutations of (R,G,B) channels  → symmetric group S_3
  - 8 negations of each channel subset  → group Z_2^3, via x ↦ 1−x
  Combined: 48 = 6 × 8 elements.

Each element is represented as a callable transform  f: (r,g,b) → (r',g',b')
and also as a (3×3 permutation matrix P, 3-vector offset d) such that:
    [r',g',b']^T = P [r,g,b]^T + d
where all values are in [0,1].
"""

from __future__ import annotations
from itertools import permutations, product
from typing import Callable, NamedTuple
import numpy as np


class SymmetryTransform(NamedTuple):
    """One element of O_h acting on the unit RGB cube."""
    name: str
    perm: tuple[int, int, int]    # which input channel maps to output (r,g,b)
    flip: tuple[int, int, int]    # 0 = keep, 1 = negate (x ↦ 1−x) per channel

    def __call__(self, rgb: np.ndarray) -> np.ndarray:
        """
        Apply transform to an array of shape (..., 3) with values in [0,1].

        Returns array of same shape.
        """
        rgb = np.asarray(rgb, dtype=float)
        out = rgb[..., list(self.perm)].copy()
        for i, f in enumerate(self.flip):
            if f:
                out[..., i] = 1.0 - out[..., i]
        return out

    def inverse(self) -> "SymmetryTransform":
        """Return the inverse transform.

        __call__ does: out[i] = rgb[perm[i]], then flip channel i if flip[i].
        Inverse: inv_perm[perm[i]] = i, and inv_flip[j] = flip[inv_perm[j]].
        """
        inv_perm = [0, 0, 0]
        for i, p in enumerate(self.perm):
            inv_perm[p] = i
        inv_perm = tuple(inv_perm)
        # inv_flip[j] = flip at the source channel for output j = flip[inv_perm[j]]
        inv_flip = tuple(self.flip[inv_perm[j]] for j in range(3))
        name = f"inv({self.name})"
        return SymmetryTransform(name=name, perm=inv_perm, flip=inv_flip)

    def compose(self, other: "SymmetryTransform") -> "SymmetryTransform":
        """Return self ∘ other (apply other first, then self).

        (self ∘ other)(x)[i]
          = self(other(x))[i]
          = other(x)[self.perm[i]]          (self permutes)
          = x[other.perm[self.perm[i]]]     (other permutes)
        with flip: other.flip[self.perm[i]] XOR self.flip[i].
        """
        new_perm = tuple(other.perm[self.perm[i]] for i in range(3))
        new_flip = tuple(
            (other.flip[self.perm[i]] + self.flip[i]) % 2
            for i in range(3)
        )
        return SymmetryTransform(
            name=f"{self.name}∘{other.name}",
            perm=new_perm,
            flip=new_flip,
        )


def build_Oh_group() -> list[SymmetryTransform]:
    """
    Build all 48 elements of O_h acting on [0,1]^3.

    Returns list of 48 SymmetryTransform objects.
    """
    elements = []
    channel_names = ["R", "G", "B"]
    for perm in permutations([0, 1, 2]):
        for flip in product([0, 1], repeat=3):
            perm_str = "".join(
                ("-" if flip[i] else "") + channel_names[perm[i]]
                for i in range(3)
            )
            name = f"[{perm_str}]"
            elements.append(SymmetryTransform(name=name, perm=perm, flip=flip))
    assert len(elements) == 48
    return elements


# Subgroups
def build_S3_group() -> list[SymmetryTransform]:
    """6 pure permutations (no flips)."""
    return [t for t in build_Oh_group() if sum(t.flip) == 0]


def build_Z2cubed_group() -> list[SymmetryTransform]:
    """8 pure negations (no permutation = identity perm)."""
    return [t for t in build_Oh_group() if t.perm == (0, 1, 2)]


def build_cyclic_group() -> list[SymmetryTransform]:
    """3 cyclic permutations (R→G→B→R) without flips."""
    return [t for t in build_S3_group() if t.perm in {(0,1,2),(1,2,0),(2,0,1)}]


# Singleton group instances
OH_GROUP: list[SymmetryTransform] = build_Oh_group()
S3_GROUP: list[SymmetryTransform] = build_S3_group()
Z2_GROUP: list[SymmetryTransform] = build_Z2cubed_group()
CYCLIC_GROUP: list[SymmetryTransform] = build_cyclic_group()

SUBGROUPS: dict[str, list[SymmetryTransform]] = {
    "Oh": OH_GROUP,
    "S3": S3_GROUP,
    "Z2^3": Z2_GROUP,
    "cyclic": CYCLIC_GROUP,
    "identity": [OH_GROUP[0]],  # [(RGB)] — no transform
}


if __name__ == "__main__":
    print(f"O_h has {len(OH_GROUP)} elements")
    print(f"S3  has {len(S3_GROUP)} elements")
    print(f"Z2^3 has {len(Z2_GROUP)} elements")
    print(f"Cyclic has {len(CYCLIC_GROUP)} elements")
    print()

    # Verify closure: compose two random elements, check result is in the group
    import random
    random.seed(0)
    a, b = random.sample(OH_GROUP, 2)
    c = a.compose(b)
    # Apply to a test point
    pt = np.array([0.2, 0.5, 0.8])
    via_compose = c(pt)
    via_sequential = a(b(pt))
    print(f"Closure check: {np.allclose(via_compose, via_sequential)}")

    # Verify inverse
    a = OH_GROUP[17]
    a_inv = a.inverse()
    pt = np.array([0.3, 0.6, 0.1])
    roundtrip = a_inv(a(pt))
    print(f"Inverse check: {np.allclose(roundtrip, pt)}")
