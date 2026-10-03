# Device calibration: times the route families a device profile can steer on the current GPU.
add_library(ninfer_calibration STATIC device_calibration.cu)
ninfer_internal_includes(ninfer_calibration)
ninfer_cuda_archive(ninfer_calibration)
target_link_libraries(ninfer_calibration PUBLIC ninfer_ops ninfer_core)
