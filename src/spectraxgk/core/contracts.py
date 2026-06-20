"""Typed contracts for differentiable refactors and validation gates.

These containers are intentionally independent of solver implementation files.
They make refactor ownership, JAX transformability, validation gates, and public
extension points explicit before large modules are split.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal


GateCategory = Literal["unit", "numerics", "physics", "parity", "autodiff", "performance", "artifact"]


def _tuple_of_nonempty_strings(values: tuple[str, ...], *, field_name: str) -> tuple[str, ...]:
    """Normalize and validate a tuple of labels used by public contracts."""

    normalized = tuple(str(value).strip() for value in values)
    if not normalized or any(not value for value in normalized):
        raise ValueError(f"{field_name} must contain at least one non-empty string")
    if len(set(normalized)) != len(normalized):
        raise ValueError(f"{field_name} must not contain duplicates")
    return normalized


@dataclass(frozen=True)
class ShapeContract:
    """Array shape metadata for a public kernel, diagnostic, or artifact field.

    Parameters
    ----------
    axes:
        Ordered axis names, for example ``("species", "ky", "kx", "z")``.
    description:
        Human-readable role of the array in the equation or diagnostic.
    dtype:
        Optional dtype policy, such as ``"real"``, ``"complex"``, or
        ``"inherits_state"``. This is descriptive; kernels still enforce actual
        dtypes locally.
    """

    axes: tuple[str, ...]
    description: str
    dtype: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "axes", _tuple_of_nonempty_strings(self.axes, field_name="axes"))
        if not self.description.strip():
            raise ValueError("description must be non-empty")
        if self.dtype is not None and not self.dtype.strip():
            raise ValueError("dtype must be non-empty when provided")


@dataclass(frozen=True)
class DifferentiabilityContract:
    """Declare how a callable is expected to behave under JAX transforms.

    This contract separates traced array inputs from static model choices and
    records the gradient evidence required before a differentiable claim is
    promoted. It does not itself call JAX.
    """

    differentiable: bool
    jit_safe: bool
    vmap_safe: bool
    static_arg_names: tuple[str, ...] = ()
    dynamic_arg_names: tuple[str, ...] = ()
    gradient_checks: tuple[str, ...] = ()
    custom_derivative: Literal["none", "jvp", "vjp", "jvp_and_vjp"] = "none"
    notes: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "static_arg_names",
            tuple(str(value).strip() for value in self.static_arg_names),
        )
        object.__setattr__(
            self,
            "dynamic_arg_names",
            tuple(str(value).strip() for value in self.dynamic_arg_names),
        )
        object.__setattr__(
            self,
            "gradient_checks",
            tuple(str(value).strip() for value in self.gradient_checks),
        )
        if any(not value for value in self.static_arg_names + self.dynamic_arg_names):
            raise ValueError("static and dynamic argument names must be non-empty")
        overlap = set(self.static_arg_names) & set(self.dynamic_arg_names)
        if overlap:
            raise ValueError(f"arguments cannot be both static and dynamic: {sorted(overlap)}")
        if self.differentiable and not self.gradient_checks:
            raise ValueError("differentiable contracts must declare gradient_checks")
        if self.custom_derivative != "none" and not self.gradient_checks:
            raise ValueError("custom derivatives require gradient_checks")


@dataclass(frozen=True)
class ValidationGateContract:
    """Physics, numerical, autodiff, parity, or performance gate declaration."""

    name: str
    category: GateCategory
    observable: str
    evidence: tuple[str, ...]
    tolerance: str | None = None
    literature_anchors: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not self.name.strip():
            raise ValueError("name must be non-empty")
        if not self.observable.strip():
            raise ValueError("observable must be non-empty")
        object.__setattr__(
            self,
            "evidence",
            _tuple_of_nonempty_strings(self.evidence, field_name="evidence"),
        )
        if self.literature_anchors:
            object.__setattr__(
                self,
                "literature_anchors",
                _tuple_of_nonempty_strings(self.literature_anchors, field_name="literature_anchors"),
            )
        if self.tolerance is not None and not self.tolerance.strip():
            raise ValueError("tolerance must be non-empty when provided")


@dataclass(frozen=True)
class ExtensionPointContract:
    """Declare a supported community extension surface."""

    name: str
    protocol: str
    required_methods: tuple[str, ...]
    validation_tests: tuple[str, ...]
    documentation: str

    def __post_init__(self) -> None:
        if not self.name.strip():
            raise ValueError("name must be non-empty")
        if not self.protocol.strip():
            raise ValueError("protocol must be non-empty")
        object.__setattr__(
            self,
            "required_methods",
            _tuple_of_nonempty_strings(self.required_methods, field_name="required_methods"),
        )
        object.__setattr__(
            self,
            "validation_tests",
            _tuple_of_nonempty_strings(self.validation_tests, field_name="validation_tests"),
        )
        if not self.documentation.strip():
            raise ValueError("documentation must be non-empty")


@dataclass(frozen=True)
class ModuleRefactorContract:
    """Trace a large source module to its target split modules and gates."""

    source_module: str
    facade_module: str
    target_modules: tuple[str, ...]
    public_api_compatible: bool
    max_lines_target: int
    validation_gates: tuple[ValidationGateContract, ...]
    extension_points: tuple[ExtensionPointContract, ...] = field(default_factory=tuple)
    differentiability: DifferentiabilityContract | None = None

    def __post_init__(self) -> None:
        if not self.source_module.startswith("spectraxgk."):
            raise ValueError("source_module must be a spectraxgk module")
        if not self.facade_module.startswith("spectraxgk."):
            raise ValueError("facade_module must be a spectraxgk module")
        object.__setattr__(
            self,
            "target_modules",
            _tuple_of_nonempty_strings(self.target_modules, field_name="target_modules"),
        )
        if not self.validation_gates:
            raise ValueError("validation_gates must contain at least one gate")
        if self.max_lines_target <= 0:
            raise ValueError("max_lines_target must be positive")

    @property
    def gate_names(self) -> tuple[str, ...]:
        """Return the declared validation-gate names in execution order."""

        return tuple(gate.name for gate in self.validation_gates)

    @property
    def target_package_names(self) -> tuple[str, ...]:
        """Return unique parent packages for the target modules."""

        packages: list[str] = []
        for module in self.target_modules:
            package = module.rsplit(".", 1)[0]
            if package not in packages:
                packages.append(package)
        return tuple(packages)
