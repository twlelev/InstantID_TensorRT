from collections import OrderedDict
from cuda import cudart
from diffusers.utils.torch_utils import randn_tensor
from enum import Enum, auto
import gc
from io import BytesIO
import numpy as np
import onnx
from onnx import numpy_helper
import onnx_graphsurgeon as gs
import os
from PIL import Image
from polygraphy.backend.common import bytes_from_path
from polygraphy.backend.trt import (
    CreateConfig,
    ModifyNetworkOutputs,
    Profile,
    engine_from_bytes,
    engine_from_network,
    network_from_onnx_path,
    save_engine
)
import random
import requests
from scipy import integrate
import tensorrt as trt
import torch

TRT_LOGGER = trt.Logger(trt.Logger.ERROR)

# Map of numpy dtype -> torch dtype
numpy_to_torch_dtype_dict = {
    np.uint8      : torch.uint8,
    np.int8       : torch.int8,
    np.int16      : torch.int16,
    np.int32      : torch.int32,
    np.int64      : torch.int64,
    np.float16    : torch.float16,
    np.float32    : torch.float32,
    np.float64    : torch.float64,
    np.complex64  : torch.complex64,
    np.complex128 : torch.complex128
}
if np.version.full_version >= "1.24.0":
    numpy_to_torch_dtype_dict[np.bool_] = torch.bool
else:
    numpy_to_torch_dtype_dict[np.bool] = torch.bool

# Map of torch dtype -> numpy dtype
torch_to_numpy_dtype_dict = {value : key for (key, value) in numpy_to_torch_dtype_dict.items()}

def unload_model(model):
    if model:
        del model
        torch.cuda.empty_cache()
        gc.collect()

def CUASSERT(cuda_ret):
    err = cuda_ret[0]
    if err != cudart.cudaError_t.cudaSuccess:
         raise RuntimeError(f"CUDA ERROR: {err}, error code reference: https://nvidia.github.io/cuda-python/module/cudart.html#cuda.cudart.cudaError_t")
    if len(cuda_ret) > 1:
        return cuda_ret[1]
    return None



class Engine():
    def __init__(
        self,
        engine_path,
    ):
        self.engine_path = engine_path
        self.engine = None
        self.context = None
        self.buffers = OrderedDict()
        self.tensors = OrderedDict()
        self.cuda_graph_instance = None # cuda graph

    def __del__(self):
        del self.engine
        del self.context
        del self.buffers
        del self.tensors

    def refit(self, refit_weights, is_fp16):
        # Initialize refitter
        refitter = trt.Refitter(self.engine, TRT_LOGGER)

        refitted_weights = set()
        # iterate through all tensorrt refittable weights
        for trt_weight_name in refitter.get_all_weights():
            if trt_weight_name not in refit_weights:
                continue

            # get weight from state dict
            trt_datatype = trt.DataType.FLOAT
            if is_fp16:
                refit_weights[trt_weight_name] = refit_weights[trt_weight_name].half()
                trt_datatype = trt.DataType.HALF

            # trt.Weight and trt.TensorLocation
            trt_wt_tensor = trt.Weights(trt_datatype, refit_weights[trt_weight_name].data_ptr(), torch.numel(refit_weights[trt_weight_name]))
            trt_wt_location = trt.TensorLocation.DEVICE if refit_weights[trt_weight_name].is_cuda else trt.TensorLocation.HOST

            # apply refit
            refitter.set_named_weights(trt_weight_name, trt_wt_tensor, trt_wt_location)
            refitted_weights.add(trt_weight_name)

        assert set(refitted_weights) == set(refit_weights.keys())
        if not refitter.refit_cuda_engine():
            print("Error: failed to refit new weights.")
            exit(0)

        print(f"[I] Total refitted weights {len(refitted_weights)}.")

    def build(self,
        onnx_path,
        fp16=True,
        tf32=False,
        input_profile=None,
        enable_refit=False,
        enable_all_tactics=False,
        timing_cache=None,
        update_output_names=None,
        weight_streaming=False,
    ):
        print(f"Building TensorRT engine for {onnx_path}: {self.engine_path}")
        p = Profile()
        if input_profile:
            for name, dims in input_profile.items():
                assert len(dims) == 3
                p.add(name, min=dims[0], opt=dims[1], max=dims[2])

        config_kwargs = {}
        if not enable_all_tactics:
            config_kwargs['tactic_sources'] = []
        if weight_streaming:
            config_kwargs['weight_streaming'] = True
            
        network = network_from_onnx_path(onnx_path, flags=[trt.OnnxParserFlag.NATIVE_INSTANCENORM])
        if update_output_names:
            print(f"Updating network outputs to {update_output_names}")
            network = ModifyNetworkOutputs(network, update_output_names)
        
        config=CreateConfig(fp16=fp16,
                tf32=tf32,
                refittable=enable_refit,
                profiles=[p],
                load_timing_cache=timing_cache,
                **config_kwargs
            )
        
        engine = engine_from_network(
            network,
            config=config,
            save_timing_cache=timing_cache
        )
        save_engine(engine, path=self.engine_path)

    def load(self):
        print(f"Loading TensorRT engine: {self.engine_path}")
        self.engine = engine_from_bytes(bytes_from_path(self.engine_path))

    def activate(self, reuse_device_memory=None):
        if reuse_device_memory:
            self.context = self.engine.create_execution_context_without_device_memory()
            self.context.device_memory = reuse_device_memory
        else:
            self.context = self.engine.create_execution_context()

    def allocate_buffers(self, shape_dict=None, device='cuda'):
        for binding in range(self.engine.num_io_tensors):
            name = self.engine.get_tensor_name(binding)
            if shape_dict and name in shape_dict:
                shape = shape_dict[name]
            else:
                shape = self.engine.get_tensor_shape(name)
            dtype = trt.nptype(self.engine.get_tensor_dtype(name))
            if self.engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT:
                self.context.set_input_shape(name, shape)
            tensor = torch.empty(tuple(shape), dtype=numpy_to_torch_dtype_dict[dtype]).to(device=device)
            self.tensors[name] = tensor

    
    # feed_dict, self.stream, self.device, use_cuda_graph=False
    def infer(self, feed_dict, stream, device, use_cuda_graph=False):
        for name, buf in feed_dict.items():
            self.tensors[name].copy_(buf)
            # print(f"{name}: {buf.shape}, dtype: {buf.dtype}")
            #         # Make sure the tensor is on the correct device
            if not buf.is_cuda: 
                feed_dict[name] = buf.cuda(device)
            self.tensors[name].copy_(feed_dict[name])
            
        for name, tensor in self.tensors.items():
            self.context.set_tensor_address(name, tensor.data_ptr())

        if use_cuda_graph:
            if self.cuda_graph_instance is not None:
                CUASSERT(cudart.cudaGraphLaunch(self.cuda_graph_instance, stream))
                CUASSERT(cudart.cudaStreamSynchronize(stream))
            else:
                # do inference before CUDA graph capture
                noerror = self.context.execute_async_v3(stream)
                if not noerror:
                    raise ValueError(f"ERROR: inference failed.")
                # capture cuda graph
                CUASSERT(cudart.cudaStreamBeginCapture(stream, cudart.cudaStreamCaptureMode.cudaStreamCaptureModeGlobal))
                self.context.execute_async_v3(stream)
                self.graph = CUASSERT(cudart.cudaStreamEndCapture(stream))
                self.cuda_graph_instance = CUASSERT(cudart.cudaGraphInstantiate(self.graph, 0))
        else:
            noerror = self.context.execute_async_v3(stream)
            if not noerror:
                raise ValueError(f"ERROR: inference failed.")

        return self.tensors




# class Engine():
#     def __init__(
#         self,
#         engine_path,
#     ):
#         self.engine_path = engine_path
#         self.engine = None
#         self.context = None
#         self.buffers = OrderedDict()
#         self.tensors = OrderedDict()
#         self.cuda_graph_instance = None # cuda graph

#     def __del__(self):
#         del self.engine
#         del self.context
#         del self.buffers
#         del self.tensors

#     def refit(self, refit_weights, is_fp16):
#         # Initialize refitter
#         refitter = trt.Refitter(self.engine, TRT_LOGGER)

#         refitted_weights = set()
#         # iterate through all tensorrt refittable weights
#         for trt_weight_name in refitter.get_all_weights():
#             if trt_weight_name not in refit_weights:
#                 continue

#             # get weight from state dict
#             trt_datatype = trt.DataType.FLOAT
#             if is_fp16:
#                 refit_weights[trt_weight_name] = refit_weights[trt_weight_name].half()
#                 trt_datatype = trt.DataType.HALF

#             # trt.Weight and trt.TensorLocation
#             trt_wt_tensor = trt.Weights(trt_datatype, refit_weights[trt_weight_name].data_ptr(), torch.numel(refit_weights[trt_weight_name]))
#             trt_wt_location = trt.TensorLocation.DEVICE if refit_weights[trt_weight_name].is_cuda else trt.TensorLocation.HOST

#             # apply refit
#             refitter.set_named_weights(trt_weight_name, trt_wt_tensor, trt_wt_location)
#             refitted_weights.add(trt_weight_name)

#         assert set(refitted_weights) == set(refit_weights.keys())
#         if not refitter.refit_cuda_engine():
#             print("Error: failed to refit new weights.")
#             exit(0)

#         print(f"[I] Total refitted weights {len(refitted_weights)}.")

#     def build(self,
#         onnx_path,
#         fp16=True,
#         tf32=False,
#         input_profile=None,
#         enable_refit=False,
#         enable_all_tactics=False,
#         timing_cache=None,
#         update_output_names=None
#     ):
#         print(f"Building TensorRT engine for {onnx_path}: {self.engine_path}")
#         p = Profile()
#         if input_profile:
#             for name, dims in input_profile.items():
#                 assert len(dims) == 3
#                 p.add(name, min=dims[0], opt=dims[1], max=dims[2])

#         config_kwargs = {}
#         if not enable_all_tactics:
#             config_kwargs['tactic_sources'] = []

#         network = network_from_onnx_path(onnx_path, flags=[trt.OnnxParserFlag.NATIVE_INSTANCENORM])
#         if update_output_names:
#             print(f"Updating network outputs to {update_output_names}")
#             network = ModifyNetworkOutputs(network, update_output_names)
#         engine = engine_from_network(
#             network,
#             config=CreateConfig(fp16=fp16,
#                 tf32=tf32,
#                 refittable=enable_refit,
#                 profiles=[p],
#                 load_timing_cache=timing_cache,
#                 **config_kwargs
#             ),
#             save_timing_cache=timing_cache
#         )
#         save_engine(engine, path=self.engine_path)

#     def load(self):
#         print(f"Loading TensorRT engine: {self.engine_path}")
#         self.engine = engine_from_bytes(bytes_from_path(self.engine_path))

#     def activate(self, reuse_device_memory=None):
#         if reuse_device_memory:
#             self.context = self.engine.create_execution_context_without_device_memory()
#             self.context.device_memory = reuse_device_memory
#         else:
#             self.context = self.engine.create_execution_context()

#     def allocate_buffers(self, shape_dict=None, device='cuda'):
#         for binding in range(self.engine.num_io_tensors):
#             name = self.engine.get_tensor_name(binding)
#             if shape_dict and name in shape_dict:
#                 shape = shape_dict[name]
#             else:
#                 shape = self.engine.get_tensor_shape(name)
#             dtype = trt.nptype(self.engine.get_tensor_dtype(name))
#             if self.engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT:
#                 self.context.set_input_shape(name, shape)
#             tensor = torch.empty(tuple(shape), dtype=numpy_to_torch_dtype_dict[dtype]).to(device=device)
#             self.tensors[name] = tensor

#     def infer(self, feed_dict, stream, use_cuda_graph=False):

#         for name, buf in feed_dict.items():
#             self.tensors[name].copy_(buf)

#         for name, tensor in self.tensors.items():
#             self.context.set_tensor_address(name, tensor.data_ptr())

#         if use_cuda_graph:
#             if self.cuda_graph_instance is not None:
#                 CUASSERT(cudart.cudaGraphLaunch(self.cuda_graph_instance, stream))
#                 CUASSERT(cudart.cudaStreamSynchronize(stream))
#             else:
#                 # do inference before CUDA graph capture
#                 noerror = self.context.execute_async_v3(stream)
#                 if not noerror:
#                     raise ValueError(f"ERROR: inference failed.")
#                 # capture cuda graph
#                 CUASSERT(cudart.cudaStreamBeginCapture(stream, cudart.cudaStreamCaptureMode.cudaStreamCaptureModeGlobal))
#                 self.context.execute_async_v3(stream)
#                 self.graph = CUASSERT(cudart.cudaStreamEndCapture(stream))
#                 self.cuda_graph_instance = CUASSERT(cudart.cudaGraphInstantiate(self.graph, 0))
#         else:
#             noerror = self.context.execute_async_v3(stream)
#             if not noerror:
#                 raise ValueError(f"ERROR: inference failed.")

#         return self.tensors


