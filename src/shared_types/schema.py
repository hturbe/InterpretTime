#!/usr/bin/env python3
# -*- coding: utf-8 -*-

'''
	Written by H.Turbé, December 2020.
    Schema to store the ECGs with Petastorm
'''
import numpy as np
from pyspark.sql.types import StringType, IntegerType
from petastorm.codecs import ScalarCodec, CompressedImageCodec, NdarrayCodec
from petastorm.unischema import Unischema, UnischemaField

ShapeSchema = Unischema(
    "ShapeSchema",
    [
        UnischemaField("noun_id", np.string_, (), ScalarCodec(StringType()), False),
        UnischemaField("signal_names", np.string_, (None,), NdarrayCodec(), False),
        UnischemaField("signal", np.float32, (None, None), NdarrayCodec(), False),
        UnischemaField("target_sum", np.string_, (), ScalarCodec(StringType()), False),
        UnischemaField(
            "target_ratio", np.string_, (), ScalarCodec(StringType()), False
        ),
        UnischemaField("signal_length", np.int_, (), ScalarCodec(IntegerType()), False),
        UnischemaField("pos_sin_wave", np.int_, (None,), NdarrayCodec(), False),
        UnischemaField("f_sine_sum", np.int_, (), ScalarCodec(IntegerType()), False),
        UnischemaField("feature_idx_sin_wave", np.int_, (None,), NdarrayCodec(), False),
        UnischemaField(
            "f_sine_ratio", np.float32, (), ScalarCodec(IntegerType()), False
        ),
    ],
)
