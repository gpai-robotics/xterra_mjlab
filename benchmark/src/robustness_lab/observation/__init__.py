"""Declarative compilation of canonical policy observations."""

from .action import ActionDecoder, compile_action
from .compiler import (
    EncoderTensorPlan,
    MultiTensorPlan,
    ObservationPlan,
    compile_encoder_tensor,
    compile_source_list,
    compile_tensors,
    compile_terms,
)
from .encoders import CompiledEncoder, compile_encoder
from .sensor_geometry import (
    height_scan_points,
    sample_height_scan,
    validate_camera_geometry,
    validate_lidar_geometry,
)

__all__ = [
    "ActionDecoder",
    "CompiledEncoder",
    "EncoderTensorPlan",
    "MultiTensorPlan",
    "ObservationPlan",
    "compile_action",
    "compile_encoder",
    "compile_encoder_tensor",
    "compile_source_list",
    "compile_tensors",
    "compile_terms",
    "height_scan_points",
    "sample_height_scan",
    "validate_camera_geometry",
    "validate_lidar_geometry",
]
