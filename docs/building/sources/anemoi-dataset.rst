.. _anemoi-dataset_source:

################
 anemoi-dataset
################

.. admonition:: Experimental
   :class: important

      This source is experimental and may change in the future.

An anemoi-dataset can be a source for a dataset:

.. literalinclude:: yaml/anemoi-dataset.yaml
   :language: yaml

The parameters are the same as those used in the ``open_dataset``
function, which allows you to subset and combine datasets. See
:ref:`opening-datasets` for more information.

In particular, this is how local zarr datasets created with anemoi in a
can be used as a source, contrary to :ref:`xarray-zarr` :

.. literalinclude:: yaml/anemoi-zarr-dataset.yaml
   :language: yaml

Creating trajectories from a continuous dataset
================================================

``anemoi_dataset`` also supports :ref:`trajectory recipes
<layouts-trajectories>`. Each requested ``(valid_time, base_date)`` pair is
read from the source at ``valid_time = base_date + step`` and annotated
with the requested base date and forecast step. Steps must be non-negative
whole hours.

For example, the following recipe interprets a continuous hourly dataset
as consecutive 18-sample blocks containing forecast leads 18 through 35:

.. literalinclude:: yaml/anemoi-trajectories-dataset.yaml
   :language: yaml

The source must already contain the intended forecast for each valid
time. This operation supplies the model-run metadata; it cannot recover
it from a continuous date axis, or recover overlapping forecasts that
were discarded when the original dataset was built. Verify both the run
cadence (``base_dates.frequency``) and the lead-time range against the
original archive.

If any source hour is missing, mark the corresponding **base date** in
``base_dates.missing``. This excludes the entire incomplete trajectory;
missing source data are not silently filled. Missingness is currently
represented along the base-date axis, not separately for each step.

Creation writes a new dataset and computes its statistics. It is not a
metadata-only or zero-copy conversion. Test a small range of base dates
before converting a large source.
