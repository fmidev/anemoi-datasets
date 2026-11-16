# (C) Copyright 2025-2026 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

import numpy as np

from . import source_registry
from .legacy import LegacySource


@source_registry.register("anemoi_dataset")
class AnemoiDatasetSource(LegacySource):
    def execute_forecast_dates(self, dates):
        """Read by valid time and attach the caller's base date and forecast step.

        The input must already contain the intended forecast for each valid
        time: a flat dataset cannot distinguish overlapping model runs.
        """
        from anemoi.transform.fields import new_field_with_metadata
        from anemoi.transform.fields import new_fieldlist_from_list

        requests = []
        for valid_time, basetime in dates:
            hours = (valid_time - basetime).total_seconds() / 3600
            if hours < 0 or not hours.is_integer():
                raise ValueError("anemoi_dataset trajectories require non-negative, whole-hour forecast steps")
            requests.append((valid_time, basetime, int(hours)))

        if not requests:
            return new_fieldlist_from_list([])

        # Read each snapshot once, even when requested for multiple base dates.
        valid_times = list(dict.fromkeys(vt for vt, _bt, _step in requests))
        fields = self._execute(self.context, valid_times, *self.args, **self.kwargs)
        by_valid_time = {}
        for field in fields:
            key = np.datetime64(field.metadata("valid_datetime"), "us")
            by_valid_time.setdefault(key, []).append(field)

        result = []
        for valid_time, basetime, step in requests:
            key = np.datetime64(valid_time, "us")
            if key not in by_valid_time:
                raise ValueError(f"No fields in anemoi_dataset for requested valid time {valid_time.isoformat()}")
            for field in by_valid_time[key]:
                result.append(
                    new_field_with_metadata(
                        field,
                        date=int(basetime.strftime("%Y%m%d")),
                        time=int(basetime.strftime("%H%M")),
                        step=step,
                    )
                )
        return new_fieldlist_from_list(result)

    @staticmethod
    def _execute(context, dates, params=None, **kwargs):
        import earthkit.data as ekd

        from anemoi.datasets import open_dataset

        ds = open_dataset(**kwargs)
        # dates_to_index = {date: i for i, date in enumerate(ds.dates)}

        indices = []
        for date in dates:
            idx = np.where(ds.dates == date)[0]
            if len(idx) == 0:
                continue
            indices.append((int(idx[0]), date))

        vars = ds.variables
        if params is None:
            params = vars

        if not isinstance(params, (list, tuple, set)):
            params = [params]

        params = set(params)
        results = []

        ensemble = ds.shape[2] > 1
        latitudes = ds.latitudes
        longitudes = ds.longitudes
        resolution = ds.resolution

        for idx, date in indices:
            metadata = dict(valid_datetime=date, latitudes=latitudes, longitudes=longitudes, resolution=resolution)

            for j, y in enumerate(ds[idx]):
                param = vars[j]
                if param not in params:
                    continue

                # metadata['name'] = param
                # metadata['param_level'] = param
                metadata["param"] = param

                for k, e in enumerate(y):
                    if ensemble:
                        metadata["number"] = k + 1

                    metadata["values"] = e

                    results.append(metadata.copy())

        # "list-of-dicts" does support resolution
        results = ekd.from_source("list-of-dicts", results)

        # return new_fieldlist_from_list([new_field_from_latitudes_longitudes(x, latitudes, longitudes) for x in results])
        return results
