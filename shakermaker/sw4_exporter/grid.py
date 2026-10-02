"""SW4 grid-line generation.

A single helper that formats the SW4 ``grid`` keyword for the input file.
Domain size is decided upstream by :class:`SW4Exporter`.
"""


def grid_line(h, x_domain, y_domain, z_domain):
    """Format the SW4 ``grid`` line.

    Inputs
    ------
    h : float
        Grid spacing in metres.
    x_domain, y_domain, z_domain : float
        Domain extents in metres along the SW4 local axes.

    Returns
    -------
    str
        Single ``grid h=... x=... y=... z=...`` line, no trailing newline.
    """
    return f"grid h={h:.16g} x={x_domain:.16g} y={y_domain:.16g} z={z_domain:.16g}"
