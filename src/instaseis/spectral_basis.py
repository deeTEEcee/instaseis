#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Numba translations for some functions from the spectral_basis module from
AxiSEM's kernel module.

:copyright:
    Martin van Driel (Martin@vanDriel.de), 2020-2025
    Lion Krischer (lion.krischer@gmail.com), 2020-2025
:license:
    GNU Lesser General Public License, Version 3 [non-commercial/academic use]
    (http://www.gnu.org/copyleft/lgpl.html)
"""

import instaseis

import numpy as np
from numba import njit


@njit(cache=instaseis._use_numba_cache)
def lagrange_interpol_2D_td(points1, points2, coefficients, x1, x2):
    """Computes the 2D Lagrange interpolation for time-dependent coefficients.

    This function performs a 2D Lagrange interpolation on a grid defined by
    `points1` and `points2`. The `coefficients` are given for each point
    on this grid and for each time sample. The interpolation is evaluated
    at the point (x1, x2).

    Parameters
    ----------
    points1 : np.ndarray
        1D array of coordinates for the first dimension.
    points2 : np.ndarray
        1D array of coordinates for the second dimension.
    coefficients : np.ndarray
        3D array of coefficients. The shape is expected to be
        (nsamp, len(points1), len(points2)), where `nsamp` is the
        number of time samples.
    x1 : float
        The coordinate in the first dimension at which to interpolate.
    x2 : float
        The coordinate in the second dimension at which to interpolate.

    Returns
    -------
    np.ndarray
        1D array of interpolated values, one for each time sample.
        The length of the array is `nsamp`.

    """
    n1 = len(points1) - 1
    n2 = len(points2) - 1

    # coefficients shape is assumed to be (nsamp, n1 + 1, n2 + 1)
    nsamp = coefficients.shape[0]

    interpolant = np.zeros(nsamp, dtype=np.float64)

    # Precompute l_i_vals for x1 and points1
    # l_i_vals[i] = product_{m!=i} (x1 - points1[m]) / (points1[i] - points1[m])
    l_i_vals = np.empty(n1 + 1, dtype=np.float64)
    for i_idx in range(n1 + 1):
        val = 1.0
        for m1 in range(n1 + 1):
            if m1 == i_idx:
                continue
            denominator = points1[i_idx] - points1[m1]
            # Avoid division by zero if points are not distinct, though Lagrange
            # interpolation assumes distinct points.
            if denominator == 0.0:
                # This case implies duplicate points in points1, which is problematic
                # for Lagrange interpolation. For simplicity matching Fortran,
                # we don't add explicit error handling here, assuming valid inputs.
                # If x1 is one of points1[m1] where m1 != i_idx, val becomes 0, which is correct.
                # If points1[i_idx] == points1[m1] for i_idx != m1, this is an issue with input points.
                val = 0.0  # Or handle error appropriately
                break
            val *= (x1 - points1[m1]) / denominator
        l_i_vals[i_idx] = val

    # Precompute l_j_vals for x2 and points2
    # l_j_vals[j] = product_{m!=j} (x2 - points2[m]) / (points2[j] - points2[m])
    l_j_vals = np.empty(n2 + 1, dtype=np.float64)
    for j_idx in range(n2 + 1):
        val = 1.0
        for m2 in range(n2 + 1):
            if m2 == j_idx:
                continue
            denominator = points2[j_idx] - points2[m2]
            if denominator == 0.0:
                val = 0.0
                break
            val *= (x2 - points2[m2]) / denominator
        l_j_vals[j_idx] = val

    # Compute interpolant
    # interpolant[s] = sum_{i=0..n1} sum_{j=0..n2} coefficients[s, i, j] * l_i_vals[i] * l_j_vals[j]
    for s_idx in range(nsamp):  # loop over samples
        current_sum = 0.0
        for i_idx in range(n1 + 1):  # loop over points1 dimension
            if (
                l_i_vals[i_idx] == 0.0
            ):  # Optimization: if l_i is zero, the inner sum term is zero
                continue
            for j_idx in range(n2 + 1):  # loop over points2 dimension
                if (
                    l_j_vals[j_idx] == 0.0
                ):  # Optimization: if l_j is zero, this term is zero
                    continue
                current_sum += (
                    coefficients[s_idx, i_idx, j_idx]
                    * l_i_vals[i_idx]
                    * l_j_vals[j_idx]
                )
        interpolant[s_idx] = current_sum

    return interpolant
