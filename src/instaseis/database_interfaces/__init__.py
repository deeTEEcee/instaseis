#!/usr/bin/env python
# -*- coding: utf-8 -*-
""":copyright:
    Lion Krischer (lion.krischer@gmail.com), 2020
:license:
    GNU Lesser General Public License, Version 3 [non-commercial/academic use]
    (http://www.gnu.org/copyleft/lgpl.html).
"""

import collections

from cloudpathlib import AnyPath

from .. import InstaseisError, InstaseisNotFoundError
from .forward_instaseis_db import ForwardInstaseisDB
from .forward_merged_instaseis_db import ForwardMergedInstaseisDB
from .mesh import _open_h5py
from .reciprocal_instaseis_db import ReciprocalInstaseisDB
from .reciprocal_merged_instaseis_db import ReciprocalMergedInstaseisDB


_TARGET_FILENAMES = frozenset([
    "ordered_output.nc4",
    "axisem_output.nc4",
    "merged_output.nc4",
])


def find_and_open_files(path, *args, **kwargs):
    """Find and open Instaseis databases with the corresponding database
    interface.

    Will recursively search the path and return an instaseis database class
    if possible. Supports both local paths and S3 URIs (s3://...).
    """
    root_path = AnyPath(path)
    found_files = []

    def _walk(current, depth):
        try:
            children = list(current.iterdir())
        except Exception:
            return
        dirs = [c for c in children if c.is_dir()]
        by_name = {c.name: c for c in children if c.is_file()}
        for name in sorted(by_name, reverse=True):
            if name in _TARGET_FILENAMES:
                found_files.append(by_name[name])
                break
        if depth < 4:
            for d in dirs:
                _walk(d, depth + 1)

    _walk(root_path, 0)

    if len(found_files) == 0:
        raise InstaseisNotFoundError(
            "No suitable netCDF files found under '%s'" % path
        )
    elif len(found_files) not in [1, 2, 4]:
        raise InstaseisError(
            "1, 2 or 4 netCDF must be present in the folder structure. "
            "Found %i: \t%s" % (len(found_files), "\n\t".join(str(f) for f in found_files))
        )

    # Catch the merged file first because its easy.
    if len(found_files) == 1 and found_files[0].name == "merged_output.nc4":
        # Now we have to open the file and find the number of dimensions.
        try:
            f, s3_fobj = _open_h5py(found_files[0])
            ds = f["/MergedSnapshots"]
            dims = ds.shape[1]
        finally:
            # File closing seems to act up in the tests for maybe locking
            # related reasons? If this proves an issue in production we'll
            # have to look into alternative solutions.
            try:
                f.close()
            except Exception:  # pragma: no cover
                pass
            if s3_fobj is not None:
                try:
                    s3_fobj.close()
                except Exception:  # pragma: no cover
                    pass

        if dims in (2, 3, 5):
            return ReciprocalMergedInstaseisDB(
                db_path=path, netcdf_file=str(found_files[0]), *args, **kwargs
            )
        elif dims == 10:
            return ForwardMergedInstaseisDB(
                db_path=path, netcdf_file=str(found_files[0]), *args, **kwargs
            )
        else:  # pragma: no cover
            raise NotImplementedError

    # Parse to find the correct components.
    netcdf_files = collections.defaultdict(list)
    patterns = ["PX", "PZ", "MZZ", "MXX_P_MYY", "MXZ_MYZ", "MXY_MXX_M_MYY"]
    for f in found_files:
        parts = f.relative_to(root_path).parts
        for p in patterns:
            if p in parts:
                netcdf_files[p].append(str(f))

    # Assert at most one file per type.
    for key, files in netcdf_files.items():
        if len(files) != 1:
            raise InstaseisError(
                "Found %i files for component %s:\n\t%s"
                % (len(files), key, "\n\t".join(files))
            )
        netcdf_files[key] = files[0]

    # Two valid cases.
    if "PX" in netcdf_files or "PZ" in netcdf_files:
        return ReciprocalInstaseisDB(
            db_path=path, netcdf_files=netcdf_files, *args, **kwargs
        )
    elif (
        "MZZ" in netcdf_files
        or "MXX_P_MYY" in netcdf_files
        or "MXZ_MYZ" in netcdf_files
        or "MXY_MXX_M_MYY" in netcdf_files
    ):
        if sorted(netcdf_files.keys()) != sorted(
            ["MZZ", "MXX_P_MYY", "MXZ_MYZ", "MXY_MXX_M_MYY"]
        ):
            raise InstaseisError(
                "Expecting all four elemental moment tensor subfolders "
                "to be present."
            )
        return ForwardInstaseisDB(
            db_path=path, netcdf_files=netcdf_files, *args, **kwargs
        )
    else:
        raise InstaseisError(
            "Could not find any suitable netCDF files. Did you pass the "
            "correct directory? E.g. if the 'ordered_output.nc4' files "
            "are located in '/path/to/PZ/Data', please pass '/path/to/' "
            "to Instaseis."
        )
