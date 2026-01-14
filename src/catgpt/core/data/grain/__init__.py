# Copyright 2025 DeepMind Technologies Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""PyGrain and Bagz data loading utilities."""

from catgpt.core.data.grain.bagz import (
    BagDataSource,
    BagFileReader,
    BagGlobShardReader,
    BagReader,
    BagShardReader,
    BagWriter,
)
from catgpt.core.data.grain.bagz_to_bag import (
    MetaFeatures,
    TrainingPositionData,
    VerificationError,
    compute_meta_features,
    convert_bagz_to_bag,
    decode_training_position,
    encode_training_position,
)
from catgpt.core.data.grain.coders import (
    LeelaGameCoder,
    LeelaPositionCoder,
    LeelaPositionData,
    decode_game,
    decode_position,
    encode_game,
    encode_position,
)
from catgpt.core.data.grain.tar_to_bag import convert_tar_to_bag

__all__ = [
    "BagDataSource",
    "BagFileReader",
    "BagGlobShardReader",
    "BagReader",
    "BagShardReader",
    "BagWriter",
    "LeelaGameCoder",
    "LeelaPositionCoder",
    "LeelaPositionData",
    "MetaFeatures",
    "TrainingPositionData",
    "VerificationError",
    "compute_meta_features",
    "convert_bagz_to_bag",
    "convert_tar_to_bag",
    "decode_game",
    "decode_position",
    "decode_training_position",
    "encode_game",
    "encode_position",
    "encode_training_position",
]
