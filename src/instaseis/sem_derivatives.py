#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Wrappers using Numba around functions for computing strain tensors.

:copyright:
    Martin van Driel (Martin@vanDriel.de), 2020-2025
    Lion Krischer (lion.krischer@gmail.com), 2020-2025
:license:
    GNU Lesser General Public License, Version 3 [non-commercial/academic use]
    (http://www.gnu.org/copyleft/lgpl.html)
"""

import numpy as np
from numba import njit

import instaseis
from instaseis.finite_elem_mapping import (
    mapping_subpar,
    mapping_spheroid,
    mapping_semino,
    mapping_semiso,
    inv_jacobian_subpar,
    inv_jacobian_spheroid,
    inv_jacobian_semino,
    inv_jacobian_semiso,
)


@njit(cache=instaseis._use_numba_cache)
def _mapping_dispatch(xi_val, eta_val, nodes, element_type):
    if element_type == 0:  # spheroid
        return mapping_spheroid(xi_val, eta_val, nodes)
    elif element_type == 1:  # subpar
        return mapping_subpar(xi_val, eta_val, nodes)
    elif element_type == 2:  # semino
        return mapping_semino(xi_val, eta_val, nodes)
    elif element_type == 3:  # semiso
        return mapping_semiso(xi_val, eta_val, nodes)
    else:
        return np.array([0.0, 0.0])


@njit(cache=instaseis._use_numba_cache)
def _inv_jacobian_dispatch(xi_val, eta_val, nodes, element_type):
    if element_type == 0:  # spheroid
        return inv_jacobian_spheroid(xi_val, eta_val, nodes)
    elif element_type == 1:  # subpar
        return inv_jacobian_subpar(xi_val, eta_val, nodes)
    elif element_type == 2:  # semino
        return inv_jacobian_semino(xi_val, eta_val, nodes)
    elif element_type == 3:  # semiso
        return inv_jacobian_semiso(xi_val, eta_val, nodes)
    else:
        return np.eye(2)


@njit(cache=instaseis._use_numba_cache)
def mxm_atd(a, b):
    nsamp = a.shape[0]
    N = a.shape[1]
    M = a.shape[2]
    P = b.shape[1]
    result = np.zeros((nsamp, N, P), dtype=np.float64)
    for s_idx in range(nsamp):
        for i_idx in range(N):
            for j_idx in range(P):
                sum_val = 0.0
                for k_idx in range(M):
                    sum_val += a[s_idx, i_idx, k_idx] * b[k_idx, j_idx]
                result[s_idx, i_idx, j_idx] = sum_val
    return result


@njit(cache=instaseis._use_numba_cache)
def mxm_btd(a, b):
    N = a.shape[0]
    M = a.shape[1]
    nsamp = b.shape[0]
    P = b.shape[2]
    result = np.zeros((nsamp, N, P), dtype=np.float64)
    for s_idx in range(nsamp):
        for i_idx in range(N):
            for j_idx in range(P):
                sum_val = 0.0
                for k_idx in range(M):
                    sum_val += a[i_idx, k_idx] * b[s_idx, k_idx, j_idx]
                result[s_idx, i_idx, j_idx] = sum_val
    return result


@njit(cache=instaseis._use_numba_cache)
def mxm_ipol0_atd(a, b):
    nsamp = a.shape[0]
    M = a.shape[2]
    P = b.shape[1]
    result = np.zeros((nsamp, P), dtype=np.float64)
    i_idx = 0
    for s_idx in range(nsamp):
        for j_idx in range(P):
            sum_val = 0.0
            for k_idx in range(M):
                sum_val += a[s_idx, i_idx, k_idx] * b[k_idx, j_idx]
            result[s_idx, j_idx] = sum_val
    return result


@njit(cache=instaseis._use_numba_cache)
def mxm_ipol0_btd(a, b):
    M = a.shape[1]
    nsamp = b.shape[0]
    P = b.shape[2]
    result = np.zeros((nsamp, P), dtype=np.float64)
    i_idx = 0
    for s_idx in range(nsamp):
        for j_idx in range(P):
            sum_val = 0.0
            for k_idx in range(M):
                sum_val += a[i_idx, k_idx] * b[s_idx, k_idx, j_idx]
            result[s_idx, j_idx] = sum_val
    return result


@njit(cache=instaseis._use_numba_cache)
def dsdf_axis_td(
    f, G_mat, GT_mat, xi_coords, eta_coords, npol, nsamp, nodes, element_type
):
    dsdf_result = np.zeros((nsamp, npol + 1), dtype=np.float64)
    inv_j_npol_temp = np.zeros((npol + 1, 2, 2), dtype=np.float64)

    ipol = 0
    for jpol_idx in range(npol + 1):
        inv_j_npol_temp[jpol_idx, :, :] = _inv_jacobian_dispatch(
            xi_coords[ipol], eta_coords[jpol_idx], nodes, element_type
        )

    mxm1 = mxm_ipol0_btd(GT_mat, f)
    mxm2 = mxm_ipol0_atd(f, G_mat)

    for jpol_idx in range(npol + 1):
        for s_idx in range(nsamp):
            dsdf_result[s_idx, jpol_idx] = (
                inv_j_npol_temp[jpol_idx, 0, 0] * mxm1[s_idx, jpol_idx]
                + inv_j_npol_temp[jpol_idx, 1, 0] * mxm2[s_idx, jpol_idx]
            )
    return dsdf_result


@njit(cache=instaseis._use_numba_cache)
def axisym_gradient_td(
    u, G_mat, GT_mat, xi_coords, eta_coords, npol, nsamp, nodes, element_type
):
    grad = np.zeros((nsamp, npol + 1, npol + 1, 2), dtype=np.float64)
    inv_j_npol = np.zeros((npol + 1, npol + 1, 2, 2), dtype=np.float64)
    for ipol_idx in range(npol + 1):
        for jpol_idx in range(npol + 1):
            inv_j_npol[ipol_idx, jpol_idx, :, :] = _inv_jacobian_dispatch(
                xi_coords[ipol_idx], eta_coords[jpol_idx], nodes, element_type
            )

    mxm1 = mxm_btd(GT_mat, u)
    mxm2 = mxm_atd(u, G_mat)

    for ipol_idx in range(npol + 1):
        for jpol_idx in range(npol + 1):
            for s_idx in range(nsamp):
                grad[s_idx, ipol_idx, jpol_idx, 0] = (
                    inv_j_npol[ipol_idx, jpol_idx, 0, 0]
                    * mxm1[s_idx, ipol_idx, jpol_idx]
                    + inv_j_npol[ipol_idx, jpol_idx, 1, 0]
                    * mxm2[s_idx, ipol_idx, jpol_idx]
                )
                grad[s_idx, ipol_idx, jpol_idx, 1] = (
                    inv_j_npol[ipol_idx, jpol_idx, 0, 1]
                    * mxm1[s_idx, ipol_idx, jpol_idx]
                    + inv_j_npol[ipol_idx, jpol_idx, 1, 1]
                    * mxm2[s_idx, ipol_idx, jpol_idx]
                )
    return grad


@njit(cache=instaseis._use_numba_cache)
def f_over_s_td(
    f_val,
    G_mat,
    GT_mat,
    xi_coords,
    eta_coords,
    npol,
    nsamp,
    nodes,
    element_type,
    axial,
):
    f_over_s = np.zeros((nsamp, npol + 1, npol + 1), dtype=np.float64)
    sz = np.zeros((npol + 1, npol + 1, 2), dtype=np.float64)
    for ipol_idx in range(npol + 1):
        for jpol_idx in range(npol + 1):
            sz[ipol_idx, jpol_idx, :] = _mapping_dispatch(
                xi_coords[ipol_idx], eta_coords[jpol_idx], nodes, element_type
            )

    if not axial:
        for ipol_idx in range(npol + 1):
            for jpol_idx in range(npol + 1):
                s_coord = sz[ipol_idx, jpol_idx, 0]
                for s_idx in range(nsamp):
                    f_over_s[s_idx, ipol_idx, jpol_idx] = (
                        f_val[s_idx, ipol_idx, jpol_idx] / s_coord
                    )
    else:
        for jpol_idx in range(npol + 1):
            for ipol_idx in range(1, npol + 1):
                s_coord = sz[ipol_idx, jpol_idx, 0]
                for s_idx in range(nsamp):
                    f_over_s[s_idx, ipol_idx, jpol_idx] = (
                        f_val[s_idx, ipol_idx, jpol_idx] / s_coord
                    )

        dsdf_val = dsdf_axis_td(
            f_val,
            G_mat,
            GT_mat,
            xi_coords,
            eta_coords,
            npol,
            nsamp,
            nodes,
            element_type,
        )
        for jpol_idx in range(npol + 1):
            for s_idx in range(nsamp):
                f_over_s[s_idx, 0, jpol_idx] = dsdf_val[s_idx, jpol_idx]
    return f_over_s


@njit(cache=instaseis._use_numba_cache)
def strain_monopole_td(
    u, G, GT, xi, eta, npol, nsamp, nodes, element_type, axial
):
    strain_tensor = np.zeros((nsamp, npol + 1, npol + 1, 6), dtype=np.float64)
    grad_buff1 = axisym_gradient_td(
        u[:, :, :, 0], G, GT, xi, eta, npol, nsamp, nodes, element_type
    )
    grad_buff2 = axisym_gradient_td(
        u[:, :, :, 2], G, GT, xi, eta, npol, nsamp, nodes, element_type
    )

    strain_tensor[:, :, :, 0] = grad_buff1[:, :, :, 0]
    strain_tensor[:, :, :, 1] = f_over_s_td(
        u[:, :, :, 0], G, GT, xi, eta, npol, nsamp, nodes, element_type, axial
    )
    strain_tensor[:, :, :, 2] = grad_buff2[:, :, :, 1]
    strain_tensor[:, :, :, 3] = 0
    strain_tensor[:, :, :, 4] = (
        grad_buff1[:, :, :, 1] + grad_buff2[:, :, :, 0]
    ) / 2.0
    strain_tensor[:, :, :, 5] = 0

    return strain_tensor


@njit(cache=instaseis._use_numba_cache)
def strain_dipole_td(
    u, G, GT, xi, eta, npol, nsamp, nodes, element_type, axial
):
    strain_tensor = np.zeros((nsamp, npol + 1, npol + 1, 6), dtype=np.float64)
    grad_buff1 = axisym_gradient_td(
        u[:, :, :, 0], G, GT, xi, eta, npol, nsamp, nodes, element_type
    )
    grad_buff2 = axisym_gradient_td(
        u[:, :, :, 1], G, GT, xi, eta, npol, nsamp, nodes, element_type
    )
    grad_buff3 = axisym_gradient_td(
        u[:, :, :, 2], G, GT, xi, eta, npol, nsamp, nodes, element_type
    )

    strain_tensor[:, :, :, 0] = grad_buff1[:, :, :, 0]
    strain_tensor[:, :, :, 1] = f_over_s_td(
        u[:, :, :, 0] - u[:, :, :, 1],
        G,
        GT,
        xi,
        eta,
        npol,
        nsamp,
        nodes,
        element_type,
        axial,
    )
    strain_tensor[:, :, :, 2] = grad_buff3[:, :, :, 1]
    strain_tensor[:, :, :, 3] = -0.5 * (
        f_over_s_td(
            u[:, :, :, 2],
            G,
            GT,
            xi,
            eta,
            npol,
            nsamp,
            nodes,
            element_type,
            axial,
        )
        + grad_buff2[:, :, :, 1]
    )
    strain_tensor[:, :, :, 4] = (
        grad_buff1[:, :, :, 1] + grad_buff3[:, :, :, 0]
    ) / 2.0
    strain_tensor[:, :, :, 5] = (
        -f_over_s_td(
            (u[:, :, :, 0] - u[:, :, :, 1]) / 2.0,
            G,
            GT,
            xi,
            eta,
            npol,
            nsamp,
            nodes,
            element_type,
            axial,
        )
        - grad_buff2[:, :, :, 0] / 2.0
    )

    return strain_tensor


@njit(cache=instaseis._use_numba_cache)
def strain_quadpole_td(
    u, G, GT, xi, eta, npol, nsamp, nodes, element_type, axial
):
    strain_tensor = np.zeros((nsamp, npol + 1, npol + 1, 6), dtype=np.float64)
    grad_buff1 = axisym_gradient_td(
        u[:, :, :, 0], G, GT, xi, eta, npol, nsamp, nodes, element_type
    )
    grad_buff2 = axisym_gradient_td(
        u[:, :, :, 1], G, GT, xi, eta, npol, nsamp, nodes, element_type
    )
    grad_buff3 = axisym_gradient_td(
        u[:, :, :, 2], G, GT, xi, eta, npol, nsamp, nodes, element_type
    )

    strain_tensor[:, :, :, 0] = grad_buff1[:, :, :, 0]
    strain_tensor[:, :, :, 1] = f_over_s_td(
        u[:, :, :, 0] - 2 * u[:, :, :, 1],
        G,
        GT,
        xi,
        eta,
        npol,
        nsamp,
        nodes,
        element_type,
        axial,
    )
    strain_tensor[:, :, :, 2] = grad_buff3[:, :, :, 1]
    strain_tensor[:, :, :, 3] = (
        -f_over_s_td(
            u[:, :, :, 2],
            G,
            GT,
            xi,
            eta,
            npol,
            nsamp,
            nodes,
            element_type,
            axial,
        )
        - grad_buff2[:, :, :, 1] / 2.0
    )
    strain_tensor[:, :, :, 4] = (
        grad_buff1[:, :, :, 1] + grad_buff3[:, :, :, 0]
    ) / 2.0
    strain_tensor[:, :, :, 5] = (
        f_over_s_td(
            0.5 * u[:, :, :, 1] - u[:, :, :, 0],
            G,
            GT,
            xi,
            eta,
            npol,
            nsamp,
            nodes,
            element_type,
            axial,
        )
        - grad_buff2[:, :, :, 0] / 2.0
    )

    return strain_tensor
