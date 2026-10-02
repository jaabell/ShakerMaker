"""SW4 sources: tabular rows, ``source`` lines and the discrete STF files.

The exporter calls :func:`source_rows` once to flatten every PointSource
into a dict with both ShakerMaker and SW4 coordinates, plus angles in
degrees and the discrete STF data. The same row collection is fed to
:func:`sw4_source_lines` to build the ``source ...`` text lines, to
:func:`source_file_text` to build one ``.txt`` per source with the
discretised slip-rate, and to the HDF5 packager to be archived inside the
transport package.
"""

from pathlib import Path
import numpy as np


def source_rows(model, transform):
    """Flatten the model sources into a list of dicts.

    Inputs
    ------
    model : ShakerMaker
        Provides ``_source`` (a :class:`FaultSource`) whose iteration
        yields :class:`PointSource` instances.
    transform : CoordinateTransform
        Used to express each source position in SW4 local metres.

    Returns
    -------
    list of dict
        One entry per point source. Keys include the position in three
        coordinate systems (``x_km``/``y_km``/``z_km``, ``x_m``/``y_m``/``z_m``,
        ``x_sw4_m``/``y_sw4_m``/``z_sw4_m``), the focal mechanism in degrees
        (``strike_deg``, ``dip_deg``, ``rake_deg``), the trigger time, STF
        metadata (``dt``, ``stf_type``, ``stf_local_t0_s``), the relative
        path of the source ``.txt`` file, and a reference to the STF object
        itself in ``stf``.

    Raises
    ------
    ValueError
        If any source-time function still has ``dt <= 0`` (i.e. it has not
        been discretised). The exporter cannot guess a sampling step.
    """
    rows = []
    for i_source, psource in enumerate(model._source):
        x_km = np.asarray(psource.x, dtype=float)
        x_m = x_km * 1000.0
        x_sw4_m = transform.from_shakermaker_km_to_sw4_m(x_km)
        angles_deg = np.degrees(np.asarray(psource.angles, dtype=float))
        stf = psource.stf
        dt = float(stf.dt)
        if dt <= 0:
            raise ValueError(
                f"Source {i_source} source-time function has no valid dt. "
                "Set psource.stf.dt before exporting to SW4.")
        rows.append({
            "id": i_source,
            "x_km": x_km[0], "y_km": x_km[1], "z_km": x_km[2],
            "x_m": x_m[0], "y_m": x_m[1], "z_m": x_m[2],
            "x_sw4_m": x_sw4_m[0], "y_sw4_m": x_sw4_m[1], "z_sw4_m": x_sw4_m[2],
            "strike_deg": angles_deg[0], "dip_deg": angles_deg[1], "rake_deg": angles_deg[2],
            "trigger_time_s": float(psource.tt),
            "stf_local_t0_s": float(stf.t[0]) if len(stf.t) else 0.0,
            "dt": dt,
            "stf_type": type(stf).__name__,
            "dfile": f"sw4/sources/source_{i_source:06d}.txt",
            "stf": stf,
        })
    return rows


def source_file_text(row):
    """Format the discrete slip-rate for a single source as SW4 expects.

    The first line is ``t0 dt npts``; the remaining lines are the slip-rate
    samples, one per line.

    Inputs
    ------
    row : dict
        One element of :func:`source_rows`. Must carry ``trigger_time_s``,
        ``dt`` and ``stf`` (whose ``.data`` provides the samples).

    Returns
    -------
    str
        File body terminated with a newline.
    """
    data = np.asarray(row["stf"].data, dtype=float).reshape(-1)
    lines = [f"{row['trigger_time_s']:.16g} {row['dt']:.16g} {len(data)}"]
    lines.extend(f"{float(value):.16g}" for value in data)
    return "\n".join(lines) + "\n"


def sw4_source_lines(rows, m0):
    """SW4 ``source`` lines, one per row.

    The ``dfile=`` field points at the per-source ``.txt`` relative to the
    SW4 working directory (i.e. ``sources/<file>``).

    Inputs
    ------
    rows : list of dict
        Output of :func:`source_rows`.
    m0 : float
        Seismic moment scaling factor. Same value for every source.

    Returns
    -------
    list of str
        SW4 ``source`` lines in writing order.
    """
    lines = []
    for row in rows:
        dfile = f"sources/{Path(row['dfile']).name}"
        lines.append(
            f"source x={float(row['x_sw4_m']):.16g} "
            f"y={float(row['y_sw4_m']):.16g} "
            f"z={float(row['z_sw4_m']):.16g} "
            f"m0={float(m0):.16g} "
            f"strike={float(row['strike_deg']):.16g} "
            f"dip={float(row['dip_deg']):.16g} "
            f"rake={float(row['rake_deg']):.16g} "
            f"t0={float(row['trigger_time_s']):.16g} "
            f"dfile={dfile}"
        )
    return lines
