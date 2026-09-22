"""Pin ``GridConfig.ky_layout`` for one harness process, from the environment.

Q9's ``rhs_identity.py`` and ``gate_traj.py`` are reused verbatim by this row,
as three earlier entries reuse them, so that the identity numbers stay
comparable across the campaign.  They build their grids from a deck that names
no ``ky_layout``, which means they take :class:`gkx.config.GridConfig`'s
default -- and the default is exactly what this row changes.

Rather than fork the harness, the driver puts this directory on ``PYTHONPATH``
and sets ``Q10_KY_LAYOUT``.  Python imports ``sitecustomize`` at interpreter
start, so the default is pinned before the harness runs and every grid it
builds, including the one inside ``run_runtime_nonlinear``, is on the named
axis.  With ``Q10_KY_LAYOUT`` unset nothing happens at all.

This is a harness control, not a product feature: nothing in ``src/`` reads
the environment for a layout, and a deck says which axis it wants with the
``[grid] ky_layout`` key.
"""

import os

_LAYOUT = os.environ.get("Q10_KY_LAYOUT", "").strip().lower()
if _LAYOUT:
    if _LAYOUT not in ("full", "half"):
        raise SystemExit(f"Q10_KY_LAYOUT must be 'full' or 'half', got {_LAYOUT!r}")
    try:
        from gkx.config import GridConfig
    except Exception as exc:  # the tree under test may not import yet
        raise SystemExit(f"Q10_KY_LAYOUT set but gkx.config did not import: {exc}")
    if "ky_layout" not in GridConfig.__dataclass_fields__:
        # `origin/main` before this row has no such field; the two-sided axis
        # is its only layout, so asking for "full" is satisfied and asking for
        # "half" cannot be.
        if _LAYOUT != "full":
            raise SystemExit(
                "this tree has no GridConfig.ky_layout; only Q10_KY_LAYOUT=full applies"
            )
    else:
        # A dataclass bakes its defaults into the generated ``__init__``
        # signature, so assigning the class attribute or the field's
        # ``default`` changes neither what ``GridConfig()`` builds nor what
        # ``replace()`` leaves alone. The first version of this file did
        # exactly that and the "full" arm quietly produced half-spectrum
        # states; the assertion below is there so a pin that does not take
        # cannot be mistaken for a measurement.
        #
        # Patching ``GridConfig`` is also not enough on its own, and the second
        # version of this file missed that. ``RuntimeConfig.grid``,
        # ``Case.grid`` and the two base cases default to a ``GridConfig``
        # *instance*, built once when ``gkx.config`` was imported -- which is
        # before this file could patch anything. ``load_runtime_from_toml``
        # merges a deck onto ``RuntimeConfig().grid``, so every deck that
        # named no ``ky_layout`` kept the half axis while ``GridConfig()``
        # reported "full", and the ``new_full`` arm compared half against
        # ``main``'s full. Those instances are rebuilt below, and a walk over
        # every dataclass default in every loaded ``gkx`` module refuses to
        # start if a stale one is left.
        import dataclasses
        import inspect
        import sys

        def _set_init_default(cls, name, value):
            init = cls.__init__
            kw = dict(init.__kwdefaults__ or {})
            if name in kw:
                kw[name] = value
                init.__kwdefaults__ = kw
                return
            positional = [
                p.name
                for p in inspect.signature(init).parameters.values()
                if p.default is not inspect.Parameter.empty
                and p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD)
            ]
            defaults = list(init.__defaults__ or ())
            defaults[positional.index(name)] = value
            init.__defaults__ = tuple(defaults)

        def _pin_field(cls, name, value):
            _set_init_default(cls, name, value)
            setattr(cls, name, value)
            cls.__dataclass_fields__[name].default = value

        def _dataclasses_in_gkx():
            for mod_name, mod in list(sys.modules.items()):
                if mod_name == "gkx" or mod_name.startswith("gkx."):
                    for obj in list(vars(mod).values()):
                        if isinstance(obj, type) and dataclasses.is_dataclass(obj):
                            yield obj

        def _stale(value, path, out):
            if isinstance(value, GridConfig):
                if value.ky_layout != _LAYOUT:
                    out.append(path)
                return
            if dataclasses.is_dataclass(value) and not isinstance(value, type):
                for f in dataclasses.fields(value):
                    _stale(getattr(value, f.name), f"{path}.{f.name}", out)

        _pin_field(GridConfig, "ky_layout", _LAYOUT)
        built = GridConfig().ky_layout
        if built != _LAYOUT:
            raise SystemExit(
                f"Q10_KY_LAYOUT={_LAYOUT} did not take: GridConfig() reports {built}"
            )
        for cls in set(_dataclasses_in_gkx()):
            for f in dataclasses.fields(cls):
                if isinstance(f.default, GridConfig) and f.default.ky_layout != _LAYOUT:
                    _pin_field(
                        cls, f.name, dataclasses.replace(f.default, ky_layout=_LAYOUT)
                    )
        stale: list[str] = []
        for cls in set(_dataclasses_in_gkx()):
            for f in dataclasses.fields(cls):
                if f.default is not dataclasses.MISSING:
                    _stale(
                        f.default, f"{cls.__module__}.{cls.__name__}.{f.name}", stale
                    )
        if stale:
            raise SystemExit(
                f"Q10_KY_LAYOUT={_LAYOUT} left stale GridConfig defaults: {sorted(stale)}"
            )
        from gkx.config import RuntimeConfig

        if RuntimeConfig().grid.ky_layout != _LAYOUT:
            raise SystemExit(
                f"Q10_KY_LAYOUT={_LAYOUT} did not reach RuntimeConfig().grid"
            )
