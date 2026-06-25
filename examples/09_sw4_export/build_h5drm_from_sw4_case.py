"""
Build an .h5drm file from an SW4 case using only the compact package HDF5.

All coordinates stored in the package are ShakerMaker/georeferenced. SW4 local
coordinates are derived only when requested for the output h5drm.
"""

import datetime
from pathlib import Path

import h5py
import numpy as np


def _read_strings(dataset):
    return [
        value.decode("utf-8") if isinstance(value, bytes) else str(value)
        for value in dataset[:]
    ]


def _read_scalar_string(dataset):
    value = dataset[()]
    return value.decode("utf-8") if isinstance(value, bytes) else str(value)


def _find_package(exports_dir):
    matches = []
    for candidate in sorted(Path(exports_dir).glob("*.h5")):
        try:
            with h5py.File(candidate, "r") as f:
                if f.attrs.get("purpose", "") == "transport_unpack_to_sw4_files":
                    matches.append(candidate)
        except OSError:
            continue
    if not matches:
        raise FileNotFoundError(f"No compact SW4 package .h5 found in {exports_dir}")
    if len(matches) > 1:
        names = ", ".join(path.name for path in matches)
        raise RuntimeError(f"More than one compact SW4 package found: {names}. Pass package_h5 explicitly.")
    return matches[0]


def _load_package(package_h5):
    with h5py.File(package_h5, "r") as f:
        config = f["config"]
        receivers = f["receivers"]
        sw4_origin_km = f["coordinates/sw4_origin_in_shakermaker_km"][:].astype(np.float64)

        files = _read_strings(receivers["file"])
        kinds = _read_strings(receivers["kind"])
        xyz_km = receivers["xyz_km"][:].astype(np.float64)
        internal = receivers["internal"][:].astype(bool)
        is_qa = receivers["is_qa"][:].astype(bool)
        model_index = receivers["model_index"][:].astype(int)

        qa_xyz = f["drm_template/qa_xyz_km"][:].astype(np.float64)
        qa_file = _read_scalar_string(f["drm_template/qa_file"]) if "qa_file" in f["drm_template"] else ""

        crust = {}
        if "crust" in f:
            crust = {
                key: f[f"crust/{key}"][:]
                for key in ("depth_top_km", "thickness_km", "vp_km_s", "vs_km_s", "rho_g_cm3", "qp", "qs")
                if key in f["crust"]
            }

        sw4 = {
            "fileio_path": _read_scalar_string(config["fileio_path"]),
            "h": float(config["h"][()]),
            "x_domain": float(config["x_domain"][()]),
            "y_domain": float(config["y_domain"][()]),
            "z_domain": float(config["z_domain"][()]),
            "tmax": float(config["tmax"][()]),
            "m0": float(config["m0"][()]),
        }

    records = []
    for file, kind, xyz, is_internal, qa, midx in zip(files, kinds, xyz_km, internal, is_qa, model_index):
        records.append({
            "file": file,
            "kind": kind,
            "xyz_km": np.asarray(xyz, dtype=np.float64),
            "internal": bool(is_internal),
            "is_qa": bool(qa),
            "model_index": int(midx),
        })

    return {
        "records": records,
        "qa_xyz_km": qa_xyz,
        "qa_file": qa_file,
        "sw4_origin_km": sw4_origin_km,
        "crust": crust,
        "sw4": sw4,
    }


def _fast_loadtxt(path):
    with open(path, "rb") as fh:
        raw = fh.read()
    lines = [line for line in raw.split(b"\n") if line and line[0:1] != b"#"]
    return np.fromstring(b" ".join(lines), dtype=np.float64, sep=" ").reshape(-1, 4)


def _signals(data, t, dt, use_filter, freqmin, freqmax, corners, zerophase):
    n_v = data[:, 1]
    e_v = data[:, 2]
    z_v = data[:, 3]

    if use_filter:
        from obspy import Trace, Stream
        traces = [Trace(data=a.astype(np.float32)) for a in (e_v, n_v, z_v)]
        for tr in traces:
            tr.stats.delta = dt
        st = Stream(traces)
        st.filter("bandpass", freqmin=freqmin, freqmax=freqmax,
                  corners=corners, zerophase=zerophase)
        e_v = st[0].data.astype(np.float64)
        n_v = st[1].data.astype(np.float64)
        z_v = st[2].data.astype(np.float64)

    velocity = np.vstack((e_v, n_v, z_v)).astype(np.float64)
    displacement = np.zeros_like(velocity)
    displacement[:, 1:] = np.cumsum(
        0.5 * (velocity[:, 1:] + velocity[:, :-1]) * dt, axis=1)
    acceleration = np.empty_like(velocity)
    acceleration[:, 1:-1] = (velocity[:, 2:] - velocity[:, :-2]) / (2.0 * dt)
    acceleration[:, 0] = (velocity[:, 1] - velocity[:, 0]) / dt
    acceleration[:, -1] = (velocity[:, -1] - velocity[:, -2]) / dt
    return velocity, displacement, acceleration


def _first_existing_station(stations):
    for station in stations:
        if station["txt"].exists():
            return station
    raise FileNotFoundError("No SW4 station .txt files were found for the package receivers.")


def _kind_for_h5drm(kind):
    if kind == "topography_surface":
        return "topo_surface"
    if kind == "topography_to_z0":
        return "topo_z0"
    if kind == "sw4_domain":
        return "domain_grid"
    return kind


def build_h5drm_from_sw4_case(
    case_path,
    package_h5=None,
    output_name="motions.h5drm",
    use_filter=False,
    freqmin=0.25,
    freqmax=10.0,
    corners=4,
    zerophase=True,
    move_2_shakermaker_coor=False,
):
    case_path = Path(case_path)
    sw4_dir = case_path / "sw4"
    exports_dir = case_path / "shakermakerexports"
    exports_dir.mkdir(parents=True, exist_ok=True)

    if package_h5 is None:
        package_h5 = _find_package(exports_dir)
    else:
        package_h5 = Path(package_h5)
        if not package_h5.is_absolute():
            package_h5 = exports_dir / package_h5

    output_h5drm = Path(output_name)
    if not output_h5drm.is_absolute():
        output_h5drm = exports_dir / output_h5drm

    package = _load_package(package_h5)
    station_dir = (sw4_dir / package["sw4"]["fileio_path"]).resolve()
    sw4_origin_km = package["sw4_origin_km"]

    drm_stations = []
    qa_station = None
    for record in package["records"]:
        fname = Path(record["file"]).name
        txt = station_dir / f"{fname}.txt"
        if record["is_qa"]:
            qa_station = {"fname": fname, "txt": txt}
            continue
        xyz_km = record["xyz_km"] if move_2_shakermaker_coor else record["xyz_km"] - sw4_origin_km
        drm_stations.append({
            "fname": fname,
            "txt": txt,
            "xyz_km": xyz_km.tolist(),
            "internal": bool(record["internal"]),
            "kind": _kind_for_h5drm(record["kind"]),
            "model_index": int(record["model_index"]),
        })

    first_station = _first_existing_station(drm_stations)
    first_data = _fast_loadtxt(first_station["txt"])
    t = first_data[:, 0]
    nt = len(t)
    dt = float((t[-1] - t[0]) / (nt - 1))

    n_total = len(drm_stations)
    n_sm = sum(1 for station in drm_stations if station["model_index"] >= 0)
    n_extra = n_total - n_sm
    qa_xyz = package["qa_xyz_km"] if move_2_shakermaker_coor else package["qa_xyz_km"] - sw4_origin_km.reshape(1, 3)

    print(f"\n{'=' * 55}")
    print(f"case    : {case_path}")
    print(f"package : {package_h5}")
    print(f"output  : {output_h5drm}")
    print(f"{'=' * 55}")
    print(f"  SM stations    : {n_sm}")
    print(f"  Extra stations : {n_extra}")
    print(f"  Total          : {n_total}")
    print(f"  nt={nt}  dt={dt:.6g}s  duration={t[-1]:.3f}s")

    string_dtype = h5py.string_dtype(encoding="utf-8")
    xyz_all = np.asarray([station["xyz_km"] for station in drm_stations], dtype=np.float64)
    internal_all = np.asarray([station["internal"] for station in drm_stations], dtype=bool)
    kind_all = np.asarray([station["kind"] for station in drm_stations], dtype=object)
    name_all = np.asarray([station["fname"] for station in drm_stations], dtype=object)
    data_loc_all = np.arange(n_total, dtype=np.int32) * 3

    with h5py.File(output_h5drm, "w") as f:
        grp = f.create_group("DRM_Data")
        grp_qa = f.create_group("DRM_QA_Data")
        meta = f.create_group("DRM_Metadata")
        sw4_rec = f.create_group("SW4_Receivers")

        grp.create_dataset("xyz", data=xyz_all)
        grp.create_dataset("internal", data=internal_all, dtype=bool)
        grp.create_dataset("data_location", data=data_loc_all, dtype=np.int32)
        grp.create_dataset("kind", data=kind_all, dtype=string_dtype)
        grp.create_dataset("name", data=name_all, dtype=string_dtype)
        grp.create_dataset("velocity", shape=(3 * n_total, nt), dtype=np.float64)
        grp.create_dataset("displacement", shape=(3 * n_total, nt), dtype=np.float64)
        grp.create_dataset("acceleration", shape=(3 * n_total, nt), dtype=np.float64)

        grp_qa.create_dataset("xyz", data=qa_xyz)
        grp_qa.create_dataset("velocity", shape=(3, nt), dtype=np.float64)
        grp_qa.create_dataset("displacement", shape=(3, nt), dtype=np.float64)
        grp_qa.create_dataset("acceleration", shape=(3, nt), dtype=np.float64)

        meta.create_dataset("dt", data=dt)
        meta.create_dataset("tstart", data=float(t[0]))
        meta.create_dataset("tend", data=float(t[-1]))
        meta.create_dataset("nt", data=int(nt))
        meta.create_dataset("receiver_count", data=int(n_total))
        meta.create_dataset("created_on", data=datetime.datetime.now().isoformat(), dtype=string_dtype)
        meta.create_dataset("program_used", data="ShakerMaker", dtype=string_dtype)
        meta.create_dataset("writer_mode", data="sw4_package_txt_filled", dtype=string_dtype)
        meta.create_dataset("component_order", data="E,N,Z", dtype=string_dtype)
        meta.create_dataset("component_map", data="E=SW4_Y, N=SW4_X, Z=SW4_Z(down+)", dtype=string_dtype)
        meta.create_dataset("coordinate_units", data="km", dtype=string_dtype)
        meta.create_dataset("filter_enabled", data=bool(use_filter))
        meta.create_dataset("filter_freqmin", data=float(freqmin))
        meta.create_dataset("filter_freqmax", data=float(freqmax))
        meta.create_dataset("filter_corners", data=int(corners))
        meta.create_dataset("filter_zerophase", data=bool(zerophase))
        meta.create_dataset("sw4_origin_in_shakermaker_km", data=sw4_origin_km)
        meta.create_dataset(
            "coordinate_system",
            data="shakermaker_utm_km" if move_2_shakermaker_coor else "sw4_local_km",
            dtype=string_dtype,
        )
        meta.create_dataset("source_package_h5", data=str(package_h5), dtype=string_dtype)

        if package["crust"]:
            crust_grp = f.create_group("Crust")
            for key, value in package["crust"].items():
                crust_grp.create_dataset(key, data=value)

        rec_xyz = np.asarray([
            record["xyz_km"] if move_2_shakermaker_coor else record["xyz_km"] - sw4_origin_km
            for record in package["records"]
        ], dtype=np.float64)
        sw4_rec.create_dataset("file", data=np.asarray([r["file"] for r in package["records"]], dtype=object), dtype=string_dtype)
        sw4_rec.create_dataset("kind", data=np.asarray([r["kind"] for r in package["records"]], dtype=object), dtype=string_dtype)
        sw4_rec.create_dataset("xyz_km", data=rec_xyz)
        sw4_rec.create_dataset("internal", data=np.asarray([r["internal"] for r in package["records"]], dtype=bool))
        sw4_rec.create_dataset("is_qa", data=np.asarray([r["is_qa"] for r in package["records"]], dtype=bool))
        sw4_rec.create_dataset("model_index", data=np.asarray([r["model_index"] for r in package["records"]], dtype=np.int32))

        for i, station in enumerate(drm_stations):
            if not station["txt"].exists():
                print(f"  WARNING: {station['fname']}.txt not found - zeroed row")
                continue
            data = _fast_loadtxt(station["txt"])
            if data.shape[0] != nt:
                raise ValueError(f"nt mismatch in {station['fname']}.txt")
            vel, disp, acc = _signals(data, t, dt, use_filter, freqmin, freqmax, corners, zerophase)
            row = i * 3
            grp["velocity"][row:row + 3] = vel
            grp["displacement"][row:row + 3] = disp
            grp["acceleration"][row:row + 3] = acc
            if (i + 1) % 5000 == 0:
                print(f"  {i + 1}/{n_total} ({100 * (i + 1) / n_total:.1f}%)", flush=True)

        if qa_station and qa_station["txt"].exists():
            print(f"  QA station: {qa_station['fname']}")
            qa_data = _fast_loadtxt(qa_station["txt"])
        else:
            nearest = int(np.argmin(np.sum((xyz_all - qa_xyz[0]) ** 2, axis=1)))
            print(f"  QA station (nearest fallback): {drm_stations[nearest]['fname']}")
            qa_data = _fast_loadtxt(drm_stations[nearest]["txt"])

        vel, disp, acc = _signals(qa_data, t, dt, use_filter, freqmin, freqmax, corners, zerophase)
        grp_qa["velocity"][:] = vel
        grp_qa["displacement"][:] = disp
        grp_qa["acceleration"][:] = acc
        f.flush()

    coord_label = "ShakerMaker UTM km" if move_2_shakermaker_coor else "SW4 local km"
    print(f"\n  -> {output_h5drm}  ({coord_label})")
    print(f"  DRM stations: {n_total}  (SM={n_sm}, extra={n_extra})")
    print(f"{'=' * 55}\n")


if __name__ == "__main__":
    import argparse
    import sys

    if len(sys.argv) == 1:
        print("SKIP: helper module; use build_h5drm_from_sw4.py or pass a case_path")
        raise SystemExit(0)

    parser = argparse.ArgumentParser(description="Build .h5drm from an SW4 case and compact package H5")
    parser.add_argument("case_path", help="Path to the SW4 case directory")
    parser.add_argument("--package-h5", help="Compact package H5 name/path. Default: auto-detect in shakermakerexports")
    parser.add_argument("--output-name", default="motions.h5drm", help="Output .h5drm name")
    parser.add_argument("--use-filter", action="store_true", help="Apply bandpass filter")
    parser.add_argument("--freqmin", type=float, default=0.25, help="Filter low corner (Hz)")
    parser.add_argument("--freqmax", type=float, default=10.0, help="Filter high corner (Hz)")
    parser.add_argument("--corners", type=int, default=4, help="Filter corners")
    parser.add_argument("--no-zerophase", dest="zerophase", action="store_false",
                        help="Disable zero-phase filtering")
    parser.add_argument("--move-2-shakermaker-coor", action="store_true",
                        help="Write h5drm coordinates in ShakerMaker/UTM coordinates")
    args = parser.parse_args()

    build_h5drm_from_sw4_case(
        case_path=args.case_path,
        package_h5=args.package_h5,
        output_name=args.output_name,
        use_filter=args.use_filter,
        freqmin=args.freqmin,
        freqmax=args.freqmax,
        corners=args.corners,
        zerophase=args.zerophase,
        move_2_shakermaker_coor=args.move_2_shakermaker_coor,
    )
