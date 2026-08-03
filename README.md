# GNPy: Optical Route Planning and DWDM Network Optimization

> ## ⚠️ Academic fork — this is not the official GNPy
>
> This repository is an independent fork of
> [Telecominfraproject/oopt-gnpy](https://github.com/Telecominfraproject/oopt-gnpy),
> branched from upstream tag **`v2.13`** and released here as **`v3.0.0+thesis`**.
> It is maintained for the doctoral research of Matheus Lôbo dos Santos
> (Federal University of Pernambuco, Brazil) and is **not endorsed by the Telecom
> Infra Project**. For production use, go upstream.
>
> ### What changed
>
> Three CPU-bound routines of the nonlinear-interference (NLI) computation in
> `gnpy.core.science_utils` were reimplemented with [Numba](https://numba.pydata.org/)
> just-in-time compilation. The mathematical expressions and the physical models are
> untouched — the changes are to memory traffic and loop structure only. The compiled
> kernels live in the new module `gnpy.core.numba_optimizations`:
>
> | Kernel | Change |
> |---|---|
> | `_approx_psi_leff_computation_numba` | effective length reorganized algebraically from `O(N²)` to `O(N)`, which also removes the intermediate matrices that only held partial results |
> | `scalar_raised_cosine`, `raised_cosine_numba` | raised-cosine pulse shaping evaluated as a compiled scalar inside the integration loop, instead of rebuilding NumPy temporaries at every step |
> | `_generalized_psi_inner_loop_numba` | generalized Ψ kernel restructured to reuse preallocated buffers, with an explicit trapezoidal rule for non-uniform frequency spacing replacing the general-purpose quadrature |
>
> Two upstream failure modes are also removed: a division by zero for ideal filters
> (roll-off equal to zero) and silent NaN propagation in the limiting cases of vanishing
> attenuation or dispersion. In both limits this fork converges to the correct analytical
> limit of the GGN approximation.
>
> ### Numerical equivalence
>
> Verified against the upstream implementation over **30,606 independent scenarios**
> spanning WDM and EON transmission in the L, C and S bands, under a pre-emphasis profile
> with central launch powers of −1, 0 and +1 dBm:
>
> - maximum relative error **2.23 × 10⁻¹⁵**
> - maximum absolute GSNR difference **1.07 × 10⁻¹⁴ dB**
>
> Both are at the level of floating-point rounding. Measured speedups: **2.58× to 2.82×**
> for the multiband EON scenarios and **5.32×** for the reference WDM case.
>
> ### Install
>
> ```
> pip install .[performance]
> ```
>
> enables the acceleration. Without Numba the code falls back automatically to the pure
> NumPy path, so results are unchanged and only the runtime differs.
>
> `examples/numba_performance_demo.py`, `examples/test_numba_integration.py` and
> `examples/test_generalized_psi_optimization.py` are demonstration scripts, not part of
> the official test suite — run them manually.
>
> ### Citing
>
> If you use this fork, cite it as described in [`CITATION.cff`](CITATION.cff). Please also
> cite upstream GNPy — [doi:10.5281/zenodo.3458319](https://doi.org/10.5281/zenodo.3458319)
> — since everything here is a derivative work of it.

[![Install via pip](https://img.shields.io/pypi/v/gnpy)](https://pypi.org/project/gnpy/)
[![Python versions](https://img.shields.io/pypi/pyversions/gnpy)](https://pypi.org/project/gnpy/)
[![Documentation status](https://readthedocs.org/projects/gnpy/badge/?version=master)](http://gnpy.readthedocs.io/en/master/?badge=master)
[![GitHub Workflow Status](https://img.shields.io/github/actions/workflow/status/Telecominfraproject/oopt-gnpy/main.yml)](https://github.com/Telecominfraproject/oopt-gnpy/actions/workflows/main.yml)
[![Gerrit](https://img.shields.io/badge/patches-via%20Gerrit-blue)](https://review.gerrithub.io/q/project:Telecominfraproject/oopt-gnpy+is:open)
[![Contributors](https://img.shields.io/github/contributors-anon/Telecominfraproject/oopt-gnpy)](https://github.com/Telecominfraproject/oopt-gnpy/graphs/contributors)
[![Code Coverage via codecov](https://img.shields.io/codecov/c/github/Telecominfraproject/oopt-gnpy)](https://codecov.io/gh/Telecominfraproject/oopt-gnpy)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.3458319.svg)](https://doi.org/10.5281/zenodo.3458319)
[![Matrix chat](https://img.shields.io/matrix/oopt-gnpy:matrix.org)](https://matrix.to/#/%23oopt-gnpy%3Amatrix.org?via=matrix.org)

GNPy is an open-source, community-developed library for building route planning and optimization tools in real-world mesh optical networks.
We are a consortium of operators, vendors, and academic researchers sponsored via the [Telecom Infra Project](http://telecominfraproject.com)'s [OOPT/PSE](https://telecominfraproject.com/open-optical-packet-transport) working group.
Together, we are building this tool for rapid development of production-grade route planning tools which is easily extensible to include custom network elements and performant to the scale of real-world mesh optical networks.

![GNPy with an OLS system](docs/images/GNPy-banner.png)

## Quick Start

Install either via [Docker](https://gnpy.readthedocs.io/en/master/install.html#using-prebuilt-docker-images), or as a [Python package](https://gnpy.readthedocs.io/en/master/install.html#using-python-on-your-computer).
Read our [documentation](https://gnpy.readthedocs.io/), learn from the demos, and [get in touch with us](https://github.com/Telecominfraproject/oopt-gnpy/discussions).

This example demonstrates how GNPy can be used to check the expected SNR at the end of the line by varying the channel input power:

![Running a simple simulation example](docs/images/gnpy-transmission-example.svg)

GNPy can do much more, including acting as a Path Computation Engine, tracking bandwidth requests, or advising the SDN controller about a best possible path through a large DWDM network.
Learn more about this [in the documentation](https://gnpy.readthedocs.io/), or give it a [try online at `gnpy.app`](https://gnpy.app/):

[![Path propagation at gnpy.app](docs/images/2022-04-12-gnpy-app.png)](https://gnpy.app/)

## Project Calendar

See upcoming meetings on the [Project Calendar](https://telecominfraproject.github.io/oopt-gnpy/calendar.html). The calendar is embedded from Google Calendar and updates automatically.
