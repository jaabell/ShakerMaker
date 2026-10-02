import os
from pathlib import Path
import tempfile
import unittest

import h5py
import numpy as np

from shakermaker.crustmodel import CrustModel
from shakermaker.ffspsource import FFSPSource, _validate_ffsp_kernel_contract


KINEMATIC_FIELDS = (
    "x", "y", "z", "slip", "rupture_time", "rise_time",
    "peak_time", "strike", "dip", "rake",
)


def make_source(first=1, last=3):
    crust = CrustModel(4)
    crust.add_layer(0.200, 1.32, 0.75, 2.40, 1000.0, 1000.0)
    crust.add_layer(0.800, 2.75, 1.57, 2.50, 1000.0, 1000.0)
    crust.add_layer(14.500, 5.50, 3.14, 2.50, 1000.0, 1000.0)
    crust.add_layer(0.000, 7.00, 4.00, 2.67, 1000.0, 1000.0)
    return FFSPSource(
        id_sf_type=8, freq_min=0.01, freq_max=24.0,
        fault_length=30.0, fault_width=16.0,
        x_hypc=15.0, y_hypc=8.0, depth_hypc=8.0,
        xref_hypc=0.0, yref_hypc=0.0,
        magnitude=6.0, fc_main_1=0.09, fc_main_2=3.0,
        rv_avg=3.0, ratio_rise=0.3,
        strike=358.0, dip=40.0, rake=113.0,
        pdip_max=15.0, prake_max=30.0,
        nsubx=16, nsuby=8, nb_taper_trbl=[5, 5, 5, 5],
        seeds=[52, 448, 4446], id_ran1=first, id_ran2=last,
        angle_north_to_x=0.0, is_moment=3,
        crust_model=crust, output_name="FFSP_TEST", verbose=False,
    )


class TestAllRealizationProducts(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp_dir = tempfile.TemporaryDirectory()
        cls.original_cwd = os.getcwd()
        os.chdir(cls.temp_dir.name)
        cls.batch = make_source(1, 3)
        cls.batch.run()
        cls.singleton = make_source(2, 2)
        cls.singleton.run()

    @classmethod
    def tearDownClass(cls):
        os.chdir(cls.original_cwd)
        cls.temp_dir.cleanup()

    def test_every_realization_has_complete_products(self):
        results = self.batch.all_realizations
        self.assertEqual(results["stf_time"]["stf"].shape, (131072, 3))
        self.assertEqual(results["spectrum"]["moment_rate_synth"].shape, (65536, 3))
        self.assertEqual(results["spectrum_octave"]["logmean_synth"].shape, (16, 3))
        np.testing.assert_array_equal(results["realization_id"], [1, 2, 3])

    def test_stf_normalization_and_spectral_misfit_are_consistent(self):
        results = self.batch.all_realizations
        time = results["stf_time"]["time"]
        stf = results["stf_time"]["stf"]
        integral = np.trapz(stf, x=time, axis=0)
        np.testing.assert_allclose(integral, np.ones(3), rtol=1.0e-5, atol=1.0e-5)

        synthetic = results["spectrum_octave"]["logmean_synth"][:12, :]
        target = results["spectrum_octave"]["logmean_dcf"][:12, None]
        reconstructed = np.sqrt(np.mean((synthetic - target) ** 2, axis=0))
        np.testing.assert_allclose(
            reconstructed, results["metrics"]["err_spectra"],
            rtol=2.0e-5, atol=2.0e-5,
        )

    def test_singleton_matches_its_batch_column_bitwise(self):
        batch = self.batch.all_realizations
        single = self.singleton.all_realizations
        for field in KINEMATIC_FIELDS:
            np.testing.assert_array_equal(batch[field][:, 1], single[field][:, 0])
        for field in batch["metrics"]:
            np.testing.assert_array_equal(batch["metrics"][field][1], single["metrics"][field][0])
        np.testing.assert_array_equal(batch["stf_time"]["time"], single["stf_time"]["time"])
        np.testing.assert_array_equal(batch["stf_time"]["stf"][:, 1], single["stf_time"]["stf"][:, 0])
        np.testing.assert_array_equal(batch["spectrum"]["freq"], single["spectrum"]["freq"])
        np.testing.assert_array_equal(batch["spectrum"]["moment_rate_dcf"], single["spectrum"]["moment_rate_dcf"])
        np.testing.assert_array_equal(batch["spectrum"]["moment_rate_synth"][:, 1], single["spectrum"]["moment_rate_synth"][:, 0])
        np.testing.assert_array_equal(batch["spectrum_octave"]["logmean_synth"][:, 1], single["spectrum_octave"]["logmean_synth"][:, 0])

    def test_best_and_active_realizations_carry_their_own_spectra(self):
        best_index = int(np.argmin(self.batch.source_stats["source_score"]["pdf"]))
        expected = self.batch.get_realization(best_index)
        self.assertEqual(self.batch.best_realization["realization_id"], expected["realization_id"])
        np.testing.assert_array_equal(
            self.batch.best_realization["spectrum"]["moment_rate_synth"],
            expected["spectrum"]["moment_rate_synth"],
        )
        self.batch.set_active_realization(0)
        np.testing.assert_array_equal(
            self.batch.subfaults["stf_time"]["stf"],
            self.batch.all_realizations["stf_time"]["stf"][:, 0],
        )

    def test_hdf5_round_trip_preserves_every_product(self):
        path = Path(self.temp_dir.name) / "round_trip.h5"
        self.batch.write_hdf5(str(path))
        loaded = FFSPSource.from_hdf5(str(path))
        with h5py.File(path, "r") as h5:
            self.assertEqual(h5.attrs["schema_version"], "2.0")
            self.assertIn("shakermaker_version", h5.attrs)
            self.assertIn("spectrum", h5["realizations"])
            self.assertEqual(h5["realizations/spectrum/moment_rate_synth"].shape, (65536, 3))
            for dataset_name in (
                    "realizations/stf_time/stf",
                    "realizations/spectrum/moment_rate_synth"):
                dataset = h5[dataset_name]
                self.assertEqual(dataset.compression, "gzip")
                self.assertTrue(dataset.shuffle)
                self.assertEqual(dataset.chunks[1], 1)
        for index in range(3):
            original = self.batch.get_realization(index)
            restored = loaded.get_realization(index)
            np.testing.assert_array_equal(original["slip"], restored["slip"])
            np.testing.assert_array_equal(
                original["spectrum"]["moment_rate_synth"],
                restored["spectrum"]["moment_rate_synth"],
            )
            np.testing.assert_array_equal(original["stf_time"]["stf"], restored["stf_time"]["stf"])

    def test_kernel_does_not_write_shared_spectral_files(self):
        generated = {path.name for path in Path(self.temp_dir.name).iterdir()}
        self.assertTrue({"calsvf.dat", "calsvf_tim.dat", "logsvf.dat"}.isdisjoint(generated))

    def test_effective_corner_frequencies_are_traceable(self):
        self.assertEqual(self.batch.params["fc_main_1_requested"], 0.09)
        self.assertEqual(self.batch.params["fc_main_2_requested"], 3.0)
        self.assertAlmostEqual(self.batch.params["fc_main_1"], 0.073282577, places=7)
        self.assertAlmostEqual(self.batch.params["fc_main_2"], 1.778281, places=6)

    def test_invalid_spectral_band_is_rejected_before_fortran(self):
        source = make_source(1, 1)
        source.params.update({
            "fault_length": 1.0,
            "fault_width": 1.0,
            "x_hypc": 0.5,
            "y_hypc": 0.5,
            "nsubx": 8,
            "nsuby": 8,
            "nb_taper_trbl": [1, 1, 1, 1],
        })
        with self.assertRaisesRegex(ValueError, "outside the FFSP octave range"):
            source.run()


class TestFFSPInputValidation(unittest.TestCase):
    def test_realization_ids_are_one_based(self):
        source = make_source(0, 1)
        with self.assertRaisesRegex(ValueError, "id_ran1 must be greater"):
            _validate_ffsp_kernel_contract(source.params)

    def test_seed_vector_has_exactly_three_entries(self):
        source = make_source(1, 1)
        source.params["seeds"] = [52, 448]
        with self.assertRaisesRegex(ValueError, "exactly three integers"):
            _validate_ffsp_kernel_contract(source.params)

    def test_derived_seed_overflow_is_rejected_before_fortran(self):
        source = make_source(71584, 71584)
        with self.assertRaisesRegex(ValueError, "derived seed3.*exceeds int32"):
            _validate_ffsp_kernel_contract(source.params)

    def test_last_safe_realization_seed_is_accepted(self):
        source = make_source(71583, 71583)
        _validate_ffsp_kernel_contract(source.params)


if __name__ == "__main__":
    unittest.main()
