import unittest

from shakermaker.crustmodel import CrustModel
from shakermaker.sw4_exporter.input_writer import sw4_input_text
from shakermaker.sw4_exporter.refinement import compute_layer_refinement


def make_stg_crust():
    """4-layer crust matching STG_FFSP_Complete.ipynb's production model."""
    crust = CrustModel(4)
    crust.add_layer(0.200, 1.32, 0.75, 2.40, 1000.0, 1000.0)
    crust.add_layer(0.800, 2.75, 1.57, 2.50, 1000.0, 1000.0)
    crust.add_layer(14.500, 5.50, 3.14, 2.50, 1000.0, 1000.0)
    crust.add_layer(0.000, 7.00, 4.00, 2.67, 1000.0, 1000.0)
    return crust


class ComputeLayerRefinementTests(unittest.TestCase):
    def test_matches_hand_built_reference_case(self):
        # Reproduces armonic_aleatory_space_refinement/STG_complete/sw4/
        # shakermaker2sw4.in, which was hand-edited to demonstrate the
        # desired pattern: "refinement zmax=1000" / "refinement zmax=200"
        # with h_base=20.
        crust = make_stg_crust()
        refinement = compute_layer_refinement(
            crust, h_base=20.0, fmax=15.0, n_per_wavelength=10.0)

        self.assertEqual(
            refinement["refinement_lines"],
            ["refinement zmax=1000", "refinement zmax=200"],
        )
        self.assertEqual(refinement["zmax_values"], [1000.0, 200.0])
        self.assertEqual(refinement["levels"], [2, 1, 0, 0])

    def test_no_refinement_when_h_base_already_resolves_every_layer(self):
        crust = make_stg_crust()
        refinement = compute_layer_refinement(
            crust, h_base=5.0, fmax=15.0, n_per_wavelength=10.0)

        self.assertEqual(refinement["refinement_lines"], [])
        self.assertEqual(refinement["levels"], [0, 0, 0, 0])

    def test_coarser_base_h_needs_more_levels(self):
        # Same fmax/n_per_wavelength as the reference case, but a coarser
        # base grid (h=25, matching export_sw4_topo's default in the
        # notebook) needs one extra halving to resolve the top layer.
        crust = make_stg_crust()
        refinement = compute_layer_refinement(
            crust, h_base=25.0, fmax=15.0, n_per_wavelength=10.0)

        self.assertEqual(
            refinement["refinement_lines"],
            ["refinement zmax=15500", "refinement zmax=1000", "refinement zmax=200"],
        )

    def test_half_space_below_h_base_warns_and_keeps_base_grid(self):
        crust = make_stg_crust()
        with self.assertWarns(UserWarning):
            refinement = compute_layer_refinement(
                crust, h_base=20.0, fmax=200.0, n_per_wavelength=10.0)
        # The half-space layer stays at level 0 (the base grid) even though
        # it would technically need refinement too -- there is nothing
        # coarser than the base grid to fall back to.
        self.assertEqual(refinement["levels"][-1], 0)

    def test_rejects_non_positive_inputs(self):
        crust = make_stg_crust()
        with self.assertRaises(ValueError):
            compute_layer_refinement(crust, h_base=20.0, fmax=0.0)
        with self.assertRaises(ValueError):
            compute_layer_refinement(crust, h_base=20.0, fmax=15.0, n_per_wavelength=0.0)
        with self.assertRaises(ValueError):
            compute_layer_refinement(crust, h_base=20.0, fmax=15.0, round_zmax="bogus")


class Sw4InputTextRefinementTests(unittest.TestCase):
    def test_refinement_lines_land_right_after_grid_line(self):
        text = sw4_input_text(
            "grid h=20 x=100 y=100 z=100",
            tmax=60,
            fileio_path="fileio",
            supergrid_gp=30,
            material_lines=["block vp=1 vs=1 rho=1"],
            source_lines=["source x=0"],
            receiver_lines=["rec x=0"],
            refinement_lines=["refinement zmax=1000", "refinement zmax=200"],
        )
        lines = text.splitlines()
        grid_idx = lines.index("grid h=20 x=100 y=100 z=100")
        self.assertEqual(lines[grid_idx + 1], "refinement zmax=1000")
        self.assertEqual(lines[grid_idx + 2], "refinement zmax=200")
        self.assertEqual(lines[grid_idx + 3], "time t=60")

    def test_no_refinement_lines_when_omitted(self):
        text = sw4_input_text(
            "grid h=20 x=100 y=100 z=100",
            tmax=60,
            fileio_path="fileio",
            supergrid_gp=30,
            material_lines=["block vp=1 vs=1 rho=1"],
            source_lines=["source x=0"],
            receiver_lines=["rec x=0"],
        )
        self.assertNotIn("refinement", text)


if __name__ == "__main__":
    unittest.main()
