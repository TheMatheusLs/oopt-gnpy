# GNPy: Optical Route Planning and DWDM Network Optimization

> **⚠️ AVISO DE FORK (Versão de Tese):** 
> Esta versão do GNPy é um fork independente do repositório upstream (oficial) criado exclusivamente para fins acadêmicos e para a tese de Matheus Lôbo dos Santos. 
> 
> **Principais Modificações:**
> Foi introduzida aceleração via JIT (Just-In-Time) compiler através da biblioteca **Numba** para acelerar substancialmente o cálculo pesado de NLI e "generalized psi" no módulo `gnpy.core.science_utils` (e funções auxiliares em `gnpy.core.numba_optimizations`).
> 
> **Instalação e Uso:**
> Para usar com a aceleração, instale com suporte a performance: `pip install .[performance]`. Caso o Numba não esteja presente, o sistema fará um *fallback* automático para o código puro em NumPy.
> *Nota:* Scripts de demonstração específicos, como `examples/test_numba_integration.py` e `numba_performance_demo.py`, não fazem parte da suíte oficial de testes e devem ser executados manualmente.

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
