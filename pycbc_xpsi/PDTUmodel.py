#!/usr/bin/env python
"""
PDT-U PyCBC/X-PSI wrapper
"""

import math
import shlex

import numpy
import xpsi
from scipy.interpolate import Akima1DInterpolator
from xpsi.cellmesh.mesh_tools import eval_cedeCentreCoords

from .STUmodel import (
    XPSI_STUModel,
    interstellar_from_config,
    nicer_from_config,
    spacetime_from_config,
    photosphere_from_config,
)


# Constants copied from the PDT_U/CustomPrior_H.py.
_A_F = 0.001
_B_F = 2.0
_A_XI = 0.001
_B_XI = math.pi / 2.0 - _A_XI
_TWO_PI = 2.0 * math.pi
_MISSING = object()


class XPSI_PDTUModel(XPSI_STUModel):
    """PyCBC model wrapper around the X-PSI likelihood for PDT-U geometry."""

    name = "pycbc_xpsi_pdtu"

    def __init__(self, variable_params, star, signals, num_energies, **kwargs):
        super().__init__(variable_params, star, signals, num_energies, **kwargs)

        vals = numpy.linspace(0.0, _B_XI, 1000)
        masses = _vector_super_radius_mass(vals)
        self._pdtu_super_radius_interpolator = Akima1DInterpolator(masses, vals)
        self._pdtu_super_radius_interpolator.extrapolate = True

        self._last_transformed_params = None

    @classmethod
    def from_config(cls, cp, **kwargs):
        """Build the X-PSI star/signals from a PyCBC config file."""
        args = cls._init_args_from_config(cp)

        section = "model"
        instruments = shlex.split(cp.get(section, "instruments"))
        num_energies = int(cp.get(section, "num-energies"))

        interstellar = interstellar_from_config(cp)
        signals = [[],]

        if "nicer" in instruments:
            nicer = nicer_from_config(cp, interstellar)
            signals[0].append(nicer.signal)

        if "xmm" in instruments:
            raise NotImplementedError("XMM is not supported by this PDT-U wrapper.")

        spacetime = spacetime_from_config(cp)
        hotregions = hotregions_from_config_pdtu(cp)
        photosphere = photosphere_from_config(cp, hotregions, spacetime.f)

        star = xpsi.Star(spacetime=spacetime, photospheres=photosphere)

        args["star"] = star
        args["signals"] = signals
        args["num_energies"] = num_energies
        args.update(kwargs)
        return cls(**args)

    def _loglikelihood(self):
        """Transform auxiliary PyCBC parameters, then evaluate X-PSI."""
        params = self.current_params
        xpsi_params = self._transform_pdtu_params(params)
        self._last_transformed_params = xpsi_params

        for name in self._xpsi_likelihood.names:
            try:
                self._xpsi_likelihood[name] = xpsi_params[name]
            except KeyError as exc:
                missing = (
                    f"The transformed PDT-U parameter dictionary does not contain "
                    f"the X-PSI parameter {name!r}. Check the [variable_params] "
                    f"section of the config and the auxiliary *_u parameters."
                )
                raise KeyError(missing) from exc

        if not self.apply_additional_constraints():
            return -numpy.inf

        try:
            logl = self._xpsi_likelihood()
        except (ArithmeticError, ValueError, FloatingPointError):
            return -numpy.inf

        if isinstance(logl, numpy.ndarray):
            logl = logl.item()

        return float(logl)

    def _transform_pdtu_params(self, params):
        """
        Convert PyCBC auxiliary PDT-U variables into physical X-PSI parameters
        following the PDT_U/CustomPrior_H.py inverse-sampling logic
        """
        out = {}

        # First copy all X-PSI parameters that are sampled directly by PyCBC
        for name in self._xpsi_likelihood.names:
            value = _try_get_param(params, name)
            if value is not _MISSING:
                out[name] = float(value)

        for tag in ("p", "s"):
            u_super = _unit_interval(_require_param(params, f"{tag}__super_radius_u"),
                                     f"{tag}__super_radius_u")
            u_cede = _unit_interval(_require_param(params, f"{tag}__cede_radius_u"),
                                    f"{tag}__cede_radius_u")
            u_sep = _unit_interval(_require_param(params, f"{tag}__cede_colatitude_u"),
                                   f"{tag}__cede_colatitude_u")
            u_az = _unit_interval(_require_param(params, f"{tag}__cede_azimuth_u"),
                                  f"{tag}__cede_azimuth_u")

            super_colatitude_name = f"{tag}__super_colatitude"
            if super_colatitude_name not in out:
                raise KeyError(
                    f"Missing directly sampled parameter {super_colatitude_name!r}; "
                    f"it is needed to transform the {tag}-spot ceding center."
                )

            # This is CustomPrior_H._interpolator(hypercube[idx]).
            super_radius = float(self._pdtu_super_radius_interpolator(u_super))

            # This is CustomPrior_H._inverse_sample_cede_radius().
            cede_radius = float(_inverse_sample_cede_radius(u_cede, super_radius))

            # This reproduces the two branches used for the local angular
            # separation between the superseding and ceding centers.
            if super_radius <= cede_radius:
                local_sep = u_sep * (cede_radius + super_radius)
            else:
                local_sep = super_radius - cede_radius + 2.0 * u_sep * cede_radius

            # The original X-PSI prior obtains an azimuthal variable from the
            # base prior before calling eval_cedeCentreCoords().  Here PyCBC
            # samples a unit variable and we map it to a full 2pi angle.
            raw_azimuth = _TWO_PI * u_az

            cede_colatitude, cede_azimuth = eval_cedeCentreCoords(
                out[super_colatitude_name],
                local_sep,
                raw_azimuth,
            )

            out[f"{tag}__super_radius"] = super_radius
            out[f"{tag}__cede_radius"] = cede_radius
            out[f"{tag}__cede_colatitude"] = float(cede_colatitude)
            out[f"{tag}__cede_azimuth"] = float(cede_azimuth)

        return out

    def apply_additional_constraints(self):
        """Apply hard physical constraints after the PDT-U transform."""
        if not super().apply_additional_constraints():
            return False

        ref = self._xpsi_likelihood

        # Enforce the -U label convention: the primary spot has lower
        # superseding colatitude than the secondary spot.
        if ref["p__super_colatitude"] > ref["s__super_colatitude"]:
            return False

        # The four primary-secondary cap-pair non-overlap constraints below
        # are a direct translation of the PDT_U/CustomPrior_H.py.
        phi1 = ref["p__phase_shift"] * _TWO_PI
        phi1 -= (0.5 + ref["s__phase_shift"]) * _TWO_PI
        ang_sep1 = xpsi.HotRegion.psi(ref["s__super_colatitude"],
                                      phi1,
                                      ref["p__super_colatitude"])

        phi2 = ref["p__phase_shift"] * _TWO_PI
        phi2 -= (0.5 + ref["s__phase_shift"]) * _TWO_PI + ref["s__cede_azimuth"]
        ang_sep2 = xpsi.HotRegion.psi(ref["s__cede_colatitude"],
                                      phi2,
                                      ref["p__super_colatitude"])

        phi3 = ref["p__phase_shift"] * _TWO_PI + ref["p__cede_azimuth"]
        phi3 -= (0.5 + ref["s__phase_shift"]) * _TWO_PI
        ang_sep3 = xpsi.HotRegion.psi(ref["s__super_colatitude"],
                                      phi3,
                                      ref["p__cede_colatitude"])

        phi4 = ref["p__phase_shift"] * _TWO_PI + ref["p__cede_azimuth"]
        phi4 -= (0.5 + ref["s__phase_shift"]) * _TWO_PI + ref["s__cede_azimuth"]
        ang_sep4 = xpsi.HotRegion.psi(ref["s__cede_colatitude"],
                                      phi4,
                                      ref["p__cede_colatitude"])

        if ((ang_sep1 < ref["p__super_radius"] + ref["s__super_radius"])
                or (ang_sep2 < ref["p__super_radius"] + ref["s__cede_radius"])
                or (ang_sep3 < ref["p__cede_radius"] + ref["s__super_radius"])
                or (ang_sep4 < ref["p__cede_radius"] + ref["s__cede_radius"])):
            return False

        return True


def hotregions_from_config_pdtu(cp):
    """Build the primary and secondary X-PSI PDT hot regions."""
    section = "hotspots"
    spottags = list(cp.get_subsections(section))

    ordered = [tag for tag in ("p", "s") if tag in spottags]
    ordered += [tag for tag in spottags if tag not in ordered]

    common_opts = _read_hotspot_args_fixed(cp, section)

    spots = []
    for tag in ordered:
        spot_section = "-".join([section, tag])
        spotopts = _read_hotspot_args_fixed(cp, spot_section, common=common_opts)

        # Force the original PDT-U geometry used by the released main.py.
        spotopts["symmetry"] = True
        spotopts["cede"] = True
        spotopts["omit"] = False
        spotopts["concentric"] = False
        spotopts["is_antiphased"] = (tag == "s")

        spots.append(xpsi.HotRegion(bounds=_pdtu_bounds_for_one_hotspot(),
                                    values={},
                                    prefix=tag,
                                    atm_ext="Num4D",
                                    **spotopts))

    return xpsi.HotRegions(tuple(spots))


def _pdtu_bounds_for_one_hotspot():
    """X-PSI bounds copied from PDT_U/main.py."""
    return dict(
        super_colatitude=(0.001, math.pi - 0.001),
        super_radius=(0.001, math.pi / 2.0 - 0.001),
        phase_shift=(-0.5, 0.5),
        super_temperature=(5.1, 6.8),
        cede_colatitude=(0.0, math.pi - 0.001),
        cede_radius=(None, None),
        cede_azimuth=(None, None),
        cede_temperature=(5.1, 6.8),
    )


def _read_hotspot_args_fixed(cp, section, common=None):
    """
    Read hotspot options while preserving boolean options inherited from
    [hotspots].
    """
    if common is None:
        out = {}
    else:
        out = common.copy()

    boolopts = ["symmetry", "omit", "cede", "concentric", "is_antiphased"]
    intopts = ["sqrt_num_cells",
               "min_sqrt_num_cells",
               "max_sqrt_num_cells",
               "num_leaves",
               "num_rays",
               "image_order_limit"]

    for opt in boolopts:
        optname = opt.replace("_", "-")
        if common is None:
            out[opt] = cp.has_option(section, optname)
        elif cp.has_option(section, optname):
            out[opt] = True

    for opt in cp.options(section):
        storeopt = opt.replace("-", "_")
        if storeopt in boolopts:
            continue
        val = cp.get(section, opt)
        if storeopt in intopts:
            val = int(val)
        out[storeopt] = val

    return out


def _I(x):
    return x * numpy.log(_B_XI / _A_XI)


def _II(x):
    return 2.0 * (x - _A_XI) - x * numpy.log(x / _B_XI)


def _scalar_super_radius_mass(x):
    if x >= _A_XI:
        mass = _II(x)
    else:
        mass = _I(x)
    return mass


def _vector_super_radius_mass(x):
    masses = numpy.zeros(len(x))
    for i, val in enumerate(x):
        masses[i] = _scalar_super_radius_mass(val)
    masses /= (_B_F - _A_F)
    masses /= (_B_XI - _A_XI)
    return masses


def _inverse_sample_cede_radius(u, psi):
    """PDT_U/CustomPrior_H.py ceding-radius inverse sampler."""
    if psi < _A_XI:
        return _A_XI * numpy.exp(u * numpy.log(_B_XI / _A_XI))
    if psi >= _A_XI and u <= 1.0 / (1.0 + numpy.log(_B_XI / psi)):
        return u * psi * (1.0 + numpy.log(_B_XI / psi))
    return psi * numpy.exp(u * (1.0 + numpy.log(_B_XI / psi)) - 1.0)


def _try_get_param(params, name):
    try:
        return params[name]
    except (KeyError, IndexError, ValueError, TypeError):
        return _MISSING


def _require_param(params, name):
    value = _try_get_param(params, name)
    if value is _MISSING:
        raise KeyError(
            f"Missing required PyCBC auxiliary parameter {name!r}. "
            f"Add it to [variable_params] and give it a uniform [0, 1] prior."
        )
    return float(value)


def _unit_interval(value, name):
    value = float(value)
    if not numpy.isfinite(value):
        raise ValueError(f"Auxiliary parameter {name!r} is not finite: {value}")
    if value < 0.0 or value > 1.0:
        raise ValueError(
            f"Auxiliary parameter {name!r} must be in [0, 1], got {value}."
        )
    return value
