import numpy as np

from demt_li.metrics import exact_signflip, holm, image_metrics, paired_statistics


def test_six_concordant_patients_give_minimum_exact_p():
    assert exact_signflip(np.arange(1, 7)) == 2 / 64


def test_holm_step_down():
    assert np.allclose(holm([.01, .04, .03]), [.03, .06, .06])


def test_identical_volumes():
    volume = np.random.default_rng(0).normal(size=(16, 16, 16))
    result = image_metrics(volume, volume)
    assert result["ssim"] == 1 and result["mae_hu"] == 0 and result["psnr_db"] == float("inf")


def test_paired_statistics_family():
    people = [dict(method=m, patient=p, ssim=v + p * .01, psnr_db=v, mae_hu=-v)
              for m, v in [("image_2d", 1.), ("rpm_1d", 0.), ("no_surrogate", .5)] for p in range(1, 7)]
    rows = paired_statistics(people)
    assert len(rows) == 6
    assert all(r["p_exact"] == 2 / 64 and abs(r["p_holm"] - 6 * 2 / 64) < 1e-12 for r in rows)
