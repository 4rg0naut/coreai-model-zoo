"""Run s6_lfm2_check.py with the builder's norm scales forced to 1 (= the original, unscaled bundle's math)."""
import runpy, sys
import coreai_models.models.ios.lfm2 as b
b.NORM_K_STREAM = 1.0; b.NORM_K_QK = 1.0
sys.argv = ["s6_lfm2_check.py"] + sys.argv[1:]
runpy.run_path("s6_lfm2_check.py", run_name="__main__")
