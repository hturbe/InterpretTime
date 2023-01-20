import os
from os.path import join as pj
from os.path import abspath as ap

FILEPATH = os.path.dirname(os.path.realpath(__file__))

src_path = ap(pj(FILEPATH, ".."))
utils_path = ap(pj(FILEPATH, "..", "shared_utils"))
assets_path = ap(pj(FILEPATH, "..", "assets"))
data_path = ap(pj(FILEPATH, "..","..", "data","datasets"))
trained_model_path = ap(pj(FILEPATH, "..","..", "data","trained_models"))
results_interp_path = ap(pj(FILEPATH, "..","..", "results"))
config_path = ap(pj(FILEPATH, "..","..", "environment"))
