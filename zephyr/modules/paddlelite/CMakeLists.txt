# Copyright (c) 2021 Intel Corporation
# Copyright 2022 Arm Limited and/or its affiliates <open-source-office@arm.com>
# SPDX-License-Identifier: Apache-2.0

if(CONFIG_PADDLE_LITE)

  set(PADDLE_LITE_DIR ${ZEPHYR_CURRENT_MODULE_DIR})

  zephyr_library()

  zephyr_include_directories(
    ${PADDLE_LITE_DIR}/.
  )


  zephyr_library_sources(
    ${PADDLE_LITE_DIR}/lite/api/paddle_api.cc
    ${PADDLE_LITE_DIR}/lite/api/paddle_place.cc
    ${PADDLE_LITE_DIR}/lite/backends/arm/math/activation.cc
    ${PADDLE_LITE_DIR}/lite/backends/arm/math/affine_channel.cc
    ${PADDLE_LITE_DIR}/lite/backends/arm/math/anchor_generator.cc
    ${PADDLE_LITE_DIR}/lite/backends/arm/math/argmax.cc
    ${PADDLE_LITE_DIR}/lite/backends/arm/math/axpy.cc
    ${PADDLE_LITE_DIR}/lite/backends/arm/math/beam_search.cc
    ${PADDLE_LITE_DIR}/lite/backends/arm/math/box_coder.cc
    ${PADDLE_LITE_DIR}/lite/backends/arm/math/col_im_transform.cc
    ${PADDLE_LITE_DIR}/lite/backends/arm/math/concat.cc
    ${PADDLE_LITE_DIR}/lite/backends/arm/math/conv3x3s1_depthwise_int8.cc
    ${PADDLE_LITE_DIR}/lite/backends/arm/math/conv3x3s1_direct_fp32.cc
    ${PADDLE_LITE_DIR}/lite/backends/arm/math/conv3x3s1_direct_int8.cc
    ${PADDLE_LITE_DIR}/lite/backends/arm/math/conv3x3s1p01_depthwise_fp32.cc
    ${PADDLE_LITE_DIR}/lite/backends/arm/math/conv3x3s1p01_depthwise_fp32_relu.cc
    ${PADDLE_LITE_DIR}/lite/backends/arm/math/conv3x3s1px_depthwise_fp32.cc
    ${PADDLE_LITE_DIR}/lite/backends/arm/math/conv3x3s2_depthwise_int8.cc
    ${PADDLE_LITE_DIR}/lite/backends/arm/math/conv3x3s2_direct_fp32.cc
    ${PADDLE_LITE_DIR}/lite/backends/arm/math/conv3x3s2_direct_int8.cc
    ${PADDLE_LITE_DIR}/lite/backends/arm/math/conv3x3s2p01_depthwise_fp32.cc
    ${PADDLE_LITE_DIR}/lite/backends/arm/math/conv3x3s2p01_depthwise_fp32_relu.cc
    ${PADDLE_LITE_DIR}/lite/backends/arm/math/conv3x3s2px_depthwise_fp32.cc
    ${PADDLE_LITE_DIR}/lite/backends/arm/math/conv3x3_winograd_fp32_c4.cc
    ${PADDLE_LITE_DIR}/lite/backends/arm/math/conv3x3_winograd_int8.cc
    ${PADDLE_LITE_DIR}/lite/backends/arm/math/conv5x5s1_depthwise_fp32.cc
    ${PADDLE_LITE_DIR}/lite/backends/arm/math/conv5x5s1_depthwise_int8.cc
    ${PADDLE_LITE_DIR}/lite/backends/arm/math/conv5x5s2_depthwise_fp32.cc
    ${PADDLE_LITE_DIR}/lite/backends/arm/math/conv5x5s2_depthwise_int8.cc
    ${PADDLE_LITE_DIR}/lite/backends/arm/math/conv_impl.cc
    ${PADDLE_LITE_DIR}/lite/backends/arm/math/conv_winograd_3x3.cc
    ${PADDLE_LITE_DIR}/lite/backends/arm/math/decode_bboxes.cc
    ${PADDLE_LITE_DIR}/lite/backends/arm/math/dropout.cc
    ${PADDLE_LITE_DIR}/lite/backends/arm/math/elementwise.cc
    ${PADDLE_LITE_DIR}/lite/backends/arm/math/fill_bias_relu.cc
    ${PADDLE_LITE_DIR}/lite/backends/arm/math/funcs.cc
    ${PADDLE_LITE_DIR}/lite/backends/arm/math/gemm_prepacked_int8.cc
    ${PADDLE_LITE_DIR}/lite/backends/arm/math/gemm_s8.cc
    ${PADDLE_LITE_DIR}/lite/backends/arm/math/gemv_arm_int8.cc
    ${PADDLE_LITE_DIR}/lite/backends/arm/math/im2sequence.cc
    ${PADDLE_LITE_DIR}/lite/backends/arm/math/increment.cc
    ${PADDLE_LITE_DIR}/lite/backends/arm/math/interpolate.cc
    ${PADDLE_LITE_DIR}/lite/backends/arm/math/layout.cc
    ${PADDLE_LITE_DIR}/lite/backends/arm/math/lrn.cc
    ${PADDLE_LITE_DIR}/lite/backends/arm/math/lstm.cc
    ${PADDLE_LITE_DIR}/lite/backends/arm/math/negative.cc
    ${PADDLE_LITE_DIR}/lite/backends/arm/math/norm.cc
    ${PADDLE_LITE_DIR}/lite/backends/arm/math/packed_sgemm.cc
    ${PADDLE_LITE_DIR}/lite/backends/arm/math/packed_sgemm_c4.cc
    ${PADDLE_LITE_DIR}/lite/backends/arm/math/pad2d.cc
    ${PADDLE_LITE_DIR}/lite/backends/arm/math/pooling.cc
    ${PADDLE_LITE_DIR}/lite/backends/arm/math/power.cc
    ${PADDLE_LITE_DIR}/lite/backends/arm/math/prior_box.cc
    ${PADDLE_LITE_DIR}/lite/backends/arm/math/reduce_max.cc
    ${PADDLE_LITE_DIR}/lite/backends/arm/math/reduce_mean.cc
    ${PADDLE_LITE_DIR}/lite/backends/arm/math/reduce_prod.cc
    ${PADDLE_LITE_DIR}/lite/backends/arm/math/scale.cc
    ${PADDLE_LITE_DIR}/lite/backends/arm/math/sequence_expand.cc
    ${PADDLE_LITE_DIR}/lite/backends/arm/math/sequence_pool.cc
    ${PADDLE_LITE_DIR}/lite/backends/arm/math/sequence_softmax.cc
    ${PADDLE_LITE_DIR}/lite/backends/arm/math/sgemm.cc
    ${PADDLE_LITE_DIR}/lite/backends/arm/math/sgemv.cc
    ${PADDLE_LITE_DIR}/lite/backends/arm/math/shuffle_channel.cc
    ${PADDLE_LITE_DIR}/lite/backends/arm/math/slice.cc
    ${PADDLE_LITE_DIR}/lite/backends/arm/math/softmax.cc
    ${PADDLE_LITE_DIR}/lite/backends/arm/math/split.cc
    ${PADDLE_LITE_DIR}/lite/backends/arm/math/split_merge_lod_tenosr.cc
    ${PADDLE_LITE_DIR}/lite/backends/arm/math/stack.cc
    ${PADDLE_LITE_DIR}/lite/backends/arm/math/topk.cc
    ${PADDLE_LITE_DIR}/lite/backends/arm/math/type_trans.cc
    ${PADDLE_LITE_DIR}/lite/backends/arm/math/yolo_box.cc
    ${PADDLE_LITE_DIR}/lite/backends/host/target_wrapper.cc
    ${PADDLE_LITE_DIR}/lite/core/context.cc
    ${PADDLE_LITE_DIR}/lite/core/device_info.cc
    ${PADDLE_LITE_DIR}/lite/core/kernel.cc
    ${PADDLE_LITE_DIR}/lite/api/light_api.cc
    ${PADDLE_LITE_DIR}/lite/core/memory.cc
    ${PADDLE_LITE_DIR}/lite/api/light_api_impl.cc
    ${PADDLE_LITE_DIR}/lite/core/op_lite.cc
    ${PADDLE_LITE_DIR}/lite/core/op_registry.cc
    ${PADDLE_LITE_DIR}/lite/core/program.cc
    ${PADDLE_LITE_DIR}/lite/core/scope.cc
    ${PADDLE_LITE_DIR}/lite/core/target_wrapper.cc
    ${PADDLE_LITE_DIR}/lite/core/tensor.cc
    ${PADDLE_LITE_DIR}/lite/core/types.cc
    ${PADDLE_LITE_DIR}/lite/core/type_system.cc
    ${PADDLE_LITE_DIR}/lite/core/variable.cc
    ${PADDLE_LITE_DIR}/lite/kernels/arm/activation_compute.cc
    ${PADDLE_LITE_DIR}/lite/kernels/arm/affine_channel_compute.cc
    ${PADDLE_LITE_DIR}/lite/kernels/arm/argmax_compute.cc
    ${PADDLE_LITE_DIR}/lite/kernels/arm/assign_value_compute.cc
    ${PADDLE_LITE_DIR}/lite/kernels/arm/batch_norm_compute.cc
    ${PADDLE_LITE_DIR}/lite/kernels/arm/box_coder_compute.cc
    ${PADDLE_LITE_DIR}/lite/kernels/arm/calib_compute.cc
    ${PADDLE_LITE_DIR}/lite/kernels/arm/cast_compute.cc
    ${PADDLE_LITE_DIR}/lite/kernels/arm/concat_compute.cc
    ${PADDLE_LITE_DIR}/lite/kernels/arm/conv_compute.cc
    ${PADDLE_LITE_DIR}/lite/kernels/arm/conv_direct.cc
    ${PADDLE_LITE_DIR}/lite/kernels/arm/conv_depthwise.cc
    ${PADDLE_LITE_DIR}/lite/kernels/arm/conv_gemmlike.cc
    ${PADDLE_LITE_DIR}/lite/kernels/arm/conv_transpose_compute.cc
    ${PADDLE_LITE_DIR}/lite/kernels/arm/conv_winograd.cc
    ${PADDLE_LITE_DIR}/lite/kernels/arm/density_prior_box_compute.cc
    ${PADDLE_LITE_DIR}/lite/kernels/arm/dropout_compute.cc
    ${PADDLE_LITE_DIR}/lite/kernels/arm/elementwise_compute.cc
    ${PADDLE_LITE_DIR}/lite/kernels/arm/expand_compute.cc
    ${PADDLE_LITE_DIR}/lite/kernels/arm/fc_compute.cc
    ${PADDLE_LITE_DIR}/lite/kernels/arm/fill_constant_batch_size_like_compute.cc
    ${PADDLE_LITE_DIR}/lite/kernels/arm/fill_constant_compute.cc
    ${PADDLE_LITE_DIR}/lite/kernels/arm/grid_sampler_compute.cc
    ${PADDLE_LITE_DIR}/lite/kernels/arm/instance_norm_compute.cc
    ${PADDLE_LITE_DIR}/lite/kernels/arm/interpolate_compute.cc
    ${PADDLE_LITE_DIR}/lite/kernels/arm/layout_compute.cc
    ${PADDLE_LITE_DIR}/lite/kernels/arm/matmul_compute.cc
    ${PADDLE_LITE_DIR}/lite/kernels/arm/mul_compute.cc
    ${PADDLE_LITE_DIR}/lite/kernels/arm/pad2d_compute.cc
    ${PADDLE_LITE_DIR}/lite/kernels/arm/pool_compute.cc
    ${PADDLE_LITE_DIR}/lite/kernels/arm/prior_box_compute.cc
    ${PADDLE_LITE_DIR}/lite/kernels/arm/range_compute.cc
    ${PADDLE_LITE_DIR}/lite/kernels/arm/reduce_mean_compute.cc
    ${PADDLE_LITE_DIR}/lite/kernels/arm/scale_compute.cc
    ${PADDLE_LITE_DIR}/lite/kernels/arm/shuffle_channel_compute.cc
    ${PADDLE_LITE_DIR}/lite/kernels/arm/slice_compute.cc
    ${PADDLE_LITE_DIR}/lite/kernels/arm/softmax_compute.cc
    ${PADDLE_LITE_DIR}/lite/kernels/arm/split_compute.cc
    ${PADDLE_LITE_DIR}/lite/kernels/arm/squeeze_compute.cc
    ${PADDLE_LITE_DIR}/lite/kernels/arm/stack_compute.cc
    ${PADDLE_LITE_DIR}/lite/kernels/arm/transpose_compute.cc
    ${PADDLE_LITE_DIR}/lite/kernels/arm/unsqueeze_compute.cc
    ${PADDLE_LITE_DIR}/lite/kernels/arm/yolo_box_compute.cc
    ${PADDLE_LITE_DIR}/lite/kernels/host/feed_compute.cc
    ${PADDLE_LITE_DIR}/lite/kernels/host/fetch_compute.cc
    ${PADDLE_LITE_DIR}/lite/kernels/host/multiclass_nms_compute.cc
    ${PADDLE_LITE_DIR}/lite/kernels/host/reshape_compute.cc
    ${PADDLE_LITE_DIR}/lite/model_parser/compatible_pb.cc
    ${PADDLE_LITE_DIR}/lite/model_parser/model_parser.cc
    ${PADDLE_LITE_DIR}/lite/model_parser/cpp/block_desc.cc
    ${PADDLE_LITE_DIR}/lite/model_parser/cpp/op_desc.cc
    ${PADDLE_LITE_DIR}/lite/model_parser/cpp/program_desc.cc
    ${PADDLE_LITE_DIR}/lite/model_parser/cpp/var_desc.cc
    ${PADDLE_LITE_DIR}/lite/model_parser/naive_buffer/block_desc.cc
    ${PADDLE_LITE_DIR}/lite/model_parser/naive_buffer/combined_params_desc.cc
    ${PADDLE_LITE_DIR}/lite/model_parser/naive_buffer/naive_buffer.cc
    ${PADDLE_LITE_DIR}/lite/model_parser/naive_buffer/op_desc.cc
    ${PADDLE_LITE_DIR}/lite/model_parser/naive_buffer/param_desc.cc
    ${PADDLE_LITE_DIR}/lite/model_parser/naive_buffer/program_desc.cc
    ${PADDLE_LITE_DIR}/lite/model_parser/naive_buffer/var_desc.cc
    ${PADDLE_LITE_DIR}/lite/model_parser/naive_buffer/proto/framework.nb.cc
    ${PADDLE_LITE_DIR}/lite/operators/activation_ops.cc
    ${PADDLE_LITE_DIR}/lite/operators/affine_channel_op.cc
    ${PADDLE_LITE_DIR}/lite/operators/argmax_op.cc
    ${PADDLE_LITE_DIR}/lite/operators/batch_norm_op.cc
    ${PADDLE_LITE_DIR}/lite/operators/box_coder_op.cc
    ${PADDLE_LITE_DIR}/lite/operators/calib_op.cc
    ${PADDLE_LITE_DIR}/lite/operators/cast_op.cc
    ${PADDLE_LITE_DIR}/lite/operators/concat_op.cc
    ${PADDLE_LITE_DIR}/lite/operators/conv_op.cc
    ${PADDLE_LITE_DIR}/lite/operators/density_prior_box_op.cc
    ${PADDLE_LITE_DIR}/lite/operators/conv_transpose_op.cc
    ${PADDLE_LITE_DIR}/lite/operators/dropout_op.cc
    ${PADDLE_LITE_DIR}/lite/operators/elementwise_ops.cc
    ${PADDLE_LITE_DIR}/lite/operators/expand_op.cc
    ${PADDLE_LITE_DIR}/lite/operators/fake_channel_wise_dequantize_max_abs.cc
    ${PADDLE_LITE_DIR}/lite/operators/fake_dequantize_max_abs.cc
    ${PADDLE_LITE_DIR}/lite/operators/fake_quantize_dequantize_moving_avg_max_abs.cc
    ${PADDLE_LITE_DIR}/lite/operators/fake_quantize_moving_avg_max_abs.cc
    ${PADDLE_LITE_DIR}/lite/operators/fake_quantize_range_abs_max.cc
    ${PADDLE_LITE_DIR}/lite/operators/fc_op.cc
    ${PADDLE_LITE_DIR}/lite/operators/feed_op.cc
    ${PADDLE_LITE_DIR}/lite/operators/fetch_op.cc
    ${PADDLE_LITE_DIR}/lite/operators/fill_constant_batch_size_like_op.cc
    ${PADDLE_LITE_DIR}/lite/operators/fill_constant_op.cc
    ${PADDLE_LITE_DIR}/lite/operators/flatten_op.cc
    ${PADDLE_LITE_DIR}/lite/operators/fusion_elementwise_activation_ops.cc
    ${PADDLE_LITE_DIR}/lite/operators/grid_sampler_op.cc
    ${PADDLE_LITE_DIR}/lite/operators/instance_norm_op.cc
    ${PADDLE_LITE_DIR}/lite/operators/interpolate_op.cc
    ${PADDLE_LITE_DIR}/lite/operators/io_copy_op.cc
    ${PADDLE_LITE_DIR}/lite/operators/layout_op.cc
    ${PADDLE_LITE_DIR}/lite/operators/matmul_op.cc
    ${PADDLE_LITE_DIR}/lite/operators/mul_op.cc
    ${PADDLE_LITE_DIR}/lite/operators/multiclass_nms_op.cc
    ${PADDLE_LITE_DIR}/lite/operators/op_params.cc
    ${PADDLE_LITE_DIR}/lite/operators/pad2d_op.cc
    ${PADDLE_LITE_DIR}/lite/operators/pool_op.cc
    ${PADDLE_LITE_DIR}/lite/operators/prior_box_op.cc
    ${PADDLE_LITE_DIR}/lite/operators/range_op.cc
    ${PADDLE_LITE_DIR}/lite/operators/reduce_mean_op.cc
    ${PADDLE_LITE_DIR}/lite/operators/relu_op.cc
    ${PADDLE_LITE_DIR}/lite/operators/reshape_op.cc
    ${PADDLE_LITE_DIR}/lite/operators/scale_op.cc
    ${PADDLE_LITE_DIR}/lite/operators/search_fc_op.cc
    ${PADDLE_LITE_DIR}/lite/operators/sequence_topk_avg_pooling_op.cc
    ${PADDLE_LITE_DIR}/lite/operators/shuffle_channel_op.cc
    ${PADDLE_LITE_DIR}/lite/operators/slice_op.cc
    ${PADDLE_LITE_DIR}/lite/operators/softmax_op.cc
    ${PADDLE_LITE_DIR}/lite/operators/split_op.cc
    ${PADDLE_LITE_DIR}/lite/operators/squeeze_op.cc
    ${PADDLE_LITE_DIR}/lite/operators/stack_op.cc
    ${PADDLE_LITE_DIR}/lite/operators/subgraph_op.cc
    ${PADDLE_LITE_DIR}/lite/operators/transpose_op.cc
    ${PADDLE_LITE_DIR}/lite/operators/unsqueeze_op.cc
    ${PADDLE_LITE_DIR}/lite/operators/yolo_box_op.cc
    ${PADDLE_LITE_DIR}/lite/utils/any.cc
    ${PADDLE_LITE_DIR}/lite/utils/logging.cc
    ${PADDLE_LITE_DIR}/lite/utils/replace_stl/stream.cc
    #${PADDLE_LITE_DIR}/tensorflow/lite/kernels/a.cc
  )

endif()
