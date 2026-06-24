# (C) Copyright 2025 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

import logging

import numpy as np
from anemoi.utils.dates import frequency_to_timedelta

from . import source_registry
from .legacy import LegacySource

LOG = logging.getLogger(__name__)


@source_registry.register("accumulate_zarr")
class AccumulateZarrSource(LegacySource):
    """Compute rolling accumulations over an existing anemoi-datasets zarr.

    For each requested output date D, fetches the preceding N fields
    (N = period / dataset_frequency) from the zarr and sums the target variables.
    The output valid_datetime is D (end-of-period convention).

    Incomplete windows (not enough preceding data in the zarr) are silently skipped.

    Parameters
    ----------
    dataset : str
        Path to the source zarr dataset (passed via **kwargs to open_dataset).
    period : str
        Accumulation window, e.g. ``"6h"``.
    variables : list[str], optional
        Variables to accumulate. Defaults to all variables in the dataset.
    rename : dict, optional
        Rename output variables, e.g. ``{tp: tp_6h}``.

    Example
    -------
    .. code-block:: yaml

        input:
          join:
            - anemoi_dataset:
                dataset: &src dataset.zarr
            - accumulate_zarr:
                dataset: *src
                period: 6h
                variables: [tp, ssrd]
                rename: {tp: tp_6h, ssrd: ssrd_6h}
    """

    @staticmethod
    def _execute(context, dates, period, variables=None, rename={}, **kwargs):
        import earthkit.data as ekd

        from anemoi.datasets import open_dataset

        ds = open_dataset(**kwargs)

        # How many source timesteps make up one accumulation window.
        # E.g. period=6h, ds.frequency=1h -> n_steps=6.
        period_td = frequency_to_timedelta(period)
        step = ds.frequency
        n_steps = int(period_td.total_seconds() / step.total_seconds())

        if n_steps < 1:
            raise ValueError(f"period={period} is shorter than dataset frequency={step}")

        # Variables to accumulate; defaults to all if not specified.
        all_vars = ds.variables
        target_vars = set(variables) if variables is not None else set(all_vars)

        date_to_idx = {np.datetime64(d, "s"): i for i, d in enumerate(ds.dates)}
        ensemble = ds.shape[2] > 1
        latitudes = ds.latitudes
        longitudes = ds.longitudes

        def _accumulate_for_date(date):
            # Build the list of source timestamps that are needed in the current window.
            # For date=T06, n_steps=6, step=1h: [T01, T02, T03, T04, T05, T06].
            # The step is read from dataset frequency. In 1h frequency dataset it is 
            # assumed that the accumulations are from the last 1h.
            needed = [
                date - np.timedelta64(int(i * step.total_seconds()), "s")
                for i in range(n_steps - 1, -1, -1)
            ]

            # Resolve each timestamp to a zarr index. Missing timestamps (before
            # the start of the zarr) map to None and are dropped.
            # This allows partial windows at the beginning of the dataset.
            idxs = [date_to_idx.get(d) for d in needed]
            idxs = [idx for idx in idxs if idx is not None]
            if not idxs:
                # No source data available at all for this date — skip it.
                return []

            out = []
            metadata = dict(valid_datetime=str(date), latitudes=latitudes, longitudes=longitudes)
            for j, var in enumerate(all_vars):
                if var not in target_vars:
                    continue
                # Apply optional rename (e.g. tp → tp_6h).
                metadata["param"] = rename.get(var, var)

                # Loop ensemble
                for k in range(ds.shape[2]):
                    # Should this be "member" instead of "number"?
                    # Ref.: https://earthkit-data.readthedocs.io/en/latest/concepts/xarray/dim.html
                    # I copied this logic from anemoi_dataset.py source type where it is "number" so 
                    # maybe earthkit-data handles this internally.
                    if ensemble:
                        metadata["number"] = k + 1

                    # idxs: time window, j: variable, k: ensemble member 
                    metadata["values"] = ds[idxs, j, k].sum(axis=0)

                    out.append(metadata.copy())
            return out

        results = []
        for date in dates:
            results.extend(_accumulate_for_date(np.datetime64(date, "s")))

        if not results:
            # The pipeline's init probe calls this with only the first output date,
            # which may predate any complete window. Scan forward in the zarr to
            # find the earliest date that produces output and use it for schema
            # inference (variable names, grid shape, etc.).
            LOG.debug("No results for requested dates — scanning zarr for first available window.")
            for d in ds.dates:
                rows = _accumulate_for_date(d)
                if rows:
                    results = rows
                    break

        if not results:
            raise ValueError(
                f"No complete {period} window found in the zarr dataset. "
                f"Need at least {n_steps} consecutive steps at {step} frequency."
            )

        return ekd.from_source("list-of-dicts", results)
