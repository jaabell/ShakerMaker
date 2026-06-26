"""Assemble the SW4 ``.in`` text.

The exporter formats every piece of the input (grid, materials, sources,
receivers, optional topography) somewhere else and hands the parts to
:func:`sw4_input_text`, which stitches them into the final file body.
"""


def sw4_input_text(grid_line, tmax, fileio_path, supergrid_gp,
                   material_lines, source_lines, receiver_lines,
                   topography_line=None):
    """Build the SW4 ``.in`` file body from already-formatted pieces.

    Inputs
    ------
    grid_line : str
        Single ``grid h=... x=... y=... z=...`` line.
    tmax : float
        Simulation duration in seconds (the ``time t=`` value).
    fileio_path : str
        Directory where SW4 will drop its ``rec`` output files.
    supergrid_gp : int
        Number of grid points in the SW4 supergrid layer.
    material_lines : list of str
        One ``block`` line per crust layer.
    source_lines : list of str
        One ``source`` line per point source.
    receiver_lines : list of str or list of (title, list of str)
        Flat list of ``rec`` lines, or a list of ``(section_title, lines)``
        pairs to emit grouped, commented receiver blocks.
    topography_line : str, optional
        ``topography input=cartesian file=...`` line. When ``None``, no
        topography keyword is emitted.

    Returns
    -------
    str
        Full ``.in`` file body, terminated with a newline.
    """
    lines = [
        "# SW4 input generated from a ShakerMaker model",
        "# Source time functions are read from sources/",
        "",
        grid_line,
        f"time t={float(tmax):.16g}",
        f"fileio path={fileio_path}",
        "",
        f"supergrid gp={int(supergrid_gp)}",
    ]
    if topography_line:
        lines.append(topography_line)
    lines += [
        "",
        "# Material model",
        *material_lines,
        "",
        "# Sources",
        *source_lines,
        "",
        "# Receivers",
    ]
    if _is_receiver_blocks(receiver_lines):
        for title, block_lines in receiver_lines:
            lines += ["", f"# {title}", *block_lines]
    else:
        lines += receiver_lines
    return "\n".join(lines) + "\n"


def _is_receiver_blocks(receiver_lines):
    """True if ``receiver_lines`` is a list of ``(title, lines)`` blocks.

    The exporter passes a flat ``list[str]`` today, but the blocked form is
    kept supported so callers can group receivers under commented section
    headers in the ``.in`` file.
    """
    if not receiver_lines:
        return False
    first = receiver_lines[0]
    return isinstance(first, tuple) and len(first) == 2
