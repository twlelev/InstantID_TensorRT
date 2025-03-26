#
# SPDX-FileCopyrightText: Copyright (c) 1993-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from diffusers.loaders import LoraLoaderMixin
from diffusers.models import (
    AutoencoderKL,
    ControlNetModel,
    UNet2DConditionModel
)
from diffusers.utils import convert_state_dict_to_diffusers
import json
import numpy as np
import onnx
from onnx import numpy_helper, shape_inference
import onnx_graphsurgeon as gs
import os
from polygraphy.backend.onnx.loader import fold_constants
import tempfile
import torch
import torch.nn.functional as F
from transformers import (
    CLIPTextModel,
    CLIPTextModelWithProjection,
    CLIPTokenizer
)
import math
import onnxruntime as ort

class Optimizer():
    def __init__(
        self,
        onnx_graph,
        verbose=False
    ):
        self.graph = gs.import_onnx(onnx_graph)
        self.verbose = verbose

    def info(self, prefix):
        if self.verbose:
            print(f"{prefix} .. {len(self.graph.nodes)} nodes, {len(self.graph.tensors().keys())} tensors, {len(self.graph.inputs)} inputs, {len(self.graph.outputs)} outputs")

    def cleanup(self, return_onnx=False):
        self.graph.cleanup().toposort()
        if return_onnx:
            return gs.export_onnx(self.graph)

    def select_outputs(self, keep, names=None):
        self.graph.outputs = [self.graph.outputs[o] for o in keep]
        if names:
            for i, name in enumerate(names):
                self.graph.outputs[i].name = name

    def fold_constants(self, return_onnx=False):
        onnx_graph = fold_constants(gs.export_onnx(self.graph), allow_onnxruntime_shape_inference=True)
        self.graph = gs.import_onnx(onnx_graph)
        if return_onnx:
            return onnx_graph

    def infer_shapes(self, return_onnx=False):
        onnx_graph = gs.export_onnx(self.graph)
        if onnx_graph.ByteSize() > 2147483648:
            temp_dir = tempfile.TemporaryDirectory().name
            os.makedirs(temp_dir, exist_ok=True)
            onnx_orig_path = os.path.join(temp_dir, 'model.onnx')
            onnx_inferred_path = os.path.join(temp_dir, 'inferred.onnx')
            onnx.save_model(onnx_graph,
                onnx_orig_path,
                save_as_external_data=True,
                all_tensors_to_one_file=True,
                convert_attribute=False)
            onnx.shape_inference.infer_shapes_path(onnx_orig_path, onnx_inferred_path)
            onnx_graph = onnx.load(onnx_inferred_path)
        else:
            onnx_graph = shape_inference.infer_shapes(onnx_graph)

        self.graph = gs.import_onnx(onnx_graph)
        if return_onnx:
            return onnx_graph

    def clip_add_hidden_states(self, return_onnx=False):
        hidden_layers = -1
        onnx_graph = gs.export_onnx(self.graph)
        for i in range(len(onnx_graph.graph.node)):
            for j in range(len(onnx_graph.graph.node[i].output)):
                name = onnx_graph.graph.node[i].output[j]
                if "layers" in name:
                    hidden_layers = max(int(name.split(".")[1].split("/")[0]), hidden_layers)
        for i in range(len(onnx_graph.graph.node)):
            for j in range(len(onnx_graph.graph.node[i].output)):
                if onnx_graph.graph.node[i].output[j] == "/text_model/encoder/layers.{}/Add_1_output_0".format(hidden_layers-1):
                    onnx_graph.graph.node[i].output[j] = "hidden_states"
            for j in range(len(onnx_graph.graph.node[i].input)):
                if onnx_graph.graph.node[i].input[j] == "/text_model/encoder/layers.{}/Add_1_output_0".format(hidden_layers-1):
                    onnx_graph.graph.node[i].input[j] = "hidden_states"
        if return_onnx:
            return onnx_graph

def get_path(version, pipeline, controlnets=None):
    if controlnets is not None:
        return ["lllyasviel/sd-controlnet-" + modality for modality in controlnets]
    
    if version == "1.4":
        if pipeline.is_inpaint():
            return "runwayml/stable-diffusion-inpainting"
        else:
            return "CompVis/stable-diffusion-v1-4"
    elif version == "1.5":
        if pipeline.is_inpaint():
            return "runwayml/stable-diffusion-inpainting"
        else:
            return "runwayml/stable-diffusion-v1-5"
    elif version == 'dreamshaper-7':
        return 'Lykon/dreamshaper-7'
    elif version == "2.0-base":
        if pipeline.is_inpaint():
            return "stabilityai/stable-diffusion-2-inpainting"
        else:
            return "stabilityai/stable-diffusion-2-base"
    elif version == "2.0":
        if pipeline.is_inpaint():
            return "stabilityai/stable-diffusion-2-inpainting"
        else:
            return "stabilityai/stable-diffusion-2"
    elif version == "2.1":
        return "stabilityai/stable-diffusion-2-1"
    elif version == "2.1-base":
        return "stabilityai/stable-diffusion-2-1-base"
    elif version == 'xl-1.0':
        if pipeline.is_sd_xl_base():
            return "stabilityai/stable-diffusion-xl-base-1.0"
        elif pipeline.is_sd_xl_refiner():
            return "stabilityai/stable-diffusion-xl-refiner-1.0"
        else:
            raise ValueError(f"Unsupported SDXL 1.0 pipeline {pipeline.name}")
    elif version == 'xl-turbo':
        if pipeline.is_sd_xl_base():
            return "stabilityai/sdxl-turbo"
        else:
            raise ValueError(f"Unsupported SDXL Turbo pipeline {pipeline.name}")
    else:
        raise ValueError(f"Incorrect version {version}")

def get_clip_embedding_dim(version, pipeline):
    if version in ("1.4", "1.5", "dreamshaper-7"):
        return 768
    elif version in ("2.0", "2.0-base", "2.1", "2.1-base"):
        return 1024
    elif version in ("xl-1.0", "xl-turbo") and pipeline.is_sd_xl_base():
        return 768
    else:
        raise ValueError(f"Invalid version {version} + pipeline {pipeline}")

def get_clipwithproj_embedding_dim(version, pipeline):
    if version in ("xl-1.0", "xl-turbo"):
        return 1280
    else:
        raise ValueError(f"Invalid version {version} + pipeline {pipeline}")

def get_unet_embedding_dim(version, pipeline):
    if version in ("1.4", "1.5", "dreamshaper-7"):
        return 768
    elif version in ("2.0", "2.0-base", "2.1", "2.1-base"):
        return 1024
    elif version in ("xl-1.0", "xl-turbo") and pipeline.is_sd_xl_base():
        return 2048
    elif version in ("xl-1.0", "xl-turbo") and pipeline.is_sd_xl_refiner():
        return 1280
    else:
        raise ValueError(f"Invalid version {version} + pipeline {pipeline}")

# FIXME after serialization support for torch.compile is added
def get_checkpoint_dir(framework_model_dir, version, pipeline, subfolder, torch_inference):
    return os.path.join(framework_model_dir, version, pipeline, subfolder)

torch_inference_modes = ['default', 'reduce-overhead', 'max-autotune']
# FIXME update callsites after serialization support for torch.compile is added
def optimize_checkpoint(model, torch_inference):
    if not torch_inference or torch_inference == 'eager':
        return model
    assert torch_inference in torch_inference_modes
    return torch.compile(model, mode=torch_inference, dynamic=False, fullgraph=False)

class LoraLoader(LoraLoaderMixin):
    def __init__(self,
        paths,
    ):
        self.paths = paths
        self.state_dict = dict()
        self.network_alphas = dict()

        for path in paths:
            state_dict, network_alphas = self.lora_state_dict(path)
            is_correct_format = all("lora" in key for key in state_dict.keys())
            if not is_correct_format:
                raise ValueError("Invalid LoRA checkpoint.")

            self.state_dict[path] = state_dict
            self.network_alphas[path] = network_alphas

    def get_dicts(self,
        prefix='unet',
        convert_to_diffusers=False,
    ):
        state_dict = dict()
        network_alphas = dict()

        for path in self.paths:
            keys = list(self.state_dict[path].keys())
            if all(key.startswith(('unet', 'text_encoder')) for key in keys):
                keys = [k for k in keys if k.startswith(prefix)]
                if keys:
                    print(f"Processing {prefix} LoRA: {path}")
                state_dict[path] = {k.replace(f"{prefix}.", ""): v for k, v in self.state_dict[path].items() if k in keys}

                if path in self.network_alphas:
                    alpha_keys = [k for k in self.network_alphas[path].keys() if k.startswith(prefix)]
                    network_alphas[path] = {
                        k.replace(f"{prefix}.", ""): v for k, v in self.network_alphas[path].items() if k in alpha_keys
                    }

            else:
                # Otherwise, we're dealing with the old format.
                warn_message = "You have saved the LoRA weights using the old format. To convert LoRA weights to the new format, first load them in a dictionary and then create a new dictionary as follows: `new_state_dict = {f'unet.{module_name}': params for module_name, params in old_state_dict.items()}`."
                print(warn_message)

        return state_dict, network_alphas


class BaseModel():
    def __init__(self,
        device='cuda',
        verbose=True,
        framework_model_dir='pytorch_model',
        fp16=False,
        max_batch_size=16,
        text_maxlen=77,
        embedding_dim=768,
    ):
        self.name = self.__class__.__name__
        self.device = device
        self.verbose = verbose
        self.framework_model_dir = framework_model_dir

        self.fp16 = fp16

        self.min_batch = 1
        self.max_batch = max_batch_size
        self.min_image_shape = 256   # min image resolution: 256x256
        self.max_image_shape = 1024  # max image resolution: 1024x1024
        self.min_latent_shape = self.min_image_shape // 8
        self.max_latent_shape = self.max_image_shape // 8

        self.text_maxlen = text_maxlen
        self.embedding_dim = embedding_dim
        self.extra_output_names = []

        self.lora_dict = None

    def get_model(self, torch_inference=''):
        pass

    def get_input_names(self):
        pass

    def get_output_names(self):
        pass

    def get_dynamic_axes(self):
        return None

    def get_sample_input(self, batch_size, image_height, image_width):
        pass

    def get_input_profile(self, batch_size, image_height, image_width, static_batch, static_shape):
        return None

    def get_shape_dict(self, batch_size, image_height, image_width):
        return None

    # Helper utility for ONNX export
    def export_onnx(self, model, onnx_path, onnx_opt_path, onnx_opset, opt_image_height, opt_image_width, batch):
        onnx_opt_graph = None
        
        sess_options = ort.SessionOptions()
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        sess_options.enable_mem_pattern = True
        sess_options.enable_cpu_mem_arena = True
        # Export optimized ONNX model (if missing)
        if not os.path.exists(onnx_opt_path):
            if not os.path.exists(onnx_path):
                print(f"Exporting ONNX model: {onnx_path}")
                with torch.inference_mode(), torch.autocast("cuda"):
                    inputs = self.get_sample_input(batch, opt_image_height, opt_image_width)
                    torch.onnx.export(model,
                            inputs,
                            onnx_path,
                            export_params=True,
                            opset_version=onnx_opset,
                            do_constant_folding=True,
                            input_names=self.get_input_names(),
                            output_names=self.get_output_names(),
                            dynamic_axes=self.get_dynamic_axes(),
                    )
            else:
                print(f"[I] Found cached ONNX model: {onnx_path}")

            print(f"Optimizing ONNX model: {onnx_opt_path}")
            onnx_opt_graph = self.optimize(onnx.load(onnx_path))
            if onnx_opt_graph.ByteSize() > 2147483648:
                onnx.save_model(
                    onnx_opt_graph,
                    onnx_opt_path,
                    save_as_external_data=True,
                    all_tensors_to_one_file=True,
                    convert_attribute=False)
            else:
                onnx.save(onnx_opt_graph, onnx_opt_path)
            
        else:
            print(f"[I] Found cached optimized ONNX model: {onnx_opt_path} ")

    # Helper utility for weights map
    def export_weights_map(self, onnx_opt_path, weights_map_path):
        if not os.path.exists(weights_map_path):
            onnx_opt_dir = os.path.dirname(onnx_opt_path)
            onnx_opt_model = onnx.load(onnx_opt_path)
            state_dict = self.get_model().state_dict()
            # Create initializer data hashes
            initializer_hash_mapping = {}
            for initializer in onnx_opt_model.graph.initializer:
                initializer_data = numpy_helper.to_array(initializer, base_dir=onnx_opt_dir).astype(np.float16)
                initializer_hash = hash(initializer_data.data.tobytes())
                initializer_hash_mapping[initializer.name] = (initializer_hash, initializer_data.shape)

            weights_name_mapping = {}
            weights_shape_mapping = {}
            # set to keep track of initializers already added to the name_mapping dict
            initializers_mapped = set()
            for wt_name, wt in state_dict.items():
                # get weight hash
                wt = wt.cpu().detach().numpy().astype(np.float16)
                wt_hash = hash(wt.data.tobytes())
                wt_t_hash = hash(np.transpose(wt).data.tobytes())

                for initializer_name, (initializer_hash, initializer_shape) in initializer_hash_mapping.items():
                    # Due to constant folding, some weights are transposed during export
                    # To account for the transpose op, we compare the initializer hash to the
                    # hash for the weight and its transpose
                    if wt_hash == initializer_hash or wt_t_hash == initializer_hash:
                        # The assert below ensures there is a 1:1 mapping between
                        # PyTorch and ONNX weight names. It can be removed in cases where 1:many
                        # mapping is found and name_mapping[wt_name] = list()
                        assert initializer_name not in initializers_mapped
                        weights_name_mapping[wt_name] = initializer_name
                        initializers_mapped.add(initializer_name)
                        is_transpose = False if wt_hash == initializer_hash else True
                        weights_shape_mapping[wt_name] = (initializer_shape, is_transpose)

                # Sanity check: Were any weights not matched
                if wt_name not in weights_name_mapping:
                    print(f'[I] PyTorch weight {wt_name} not matched with any ONNX initializer')
            print(f'[I] {len(weights_name_mapping.keys())} PyTorch weights were matched with ONNX initializers')
            assert weights_name_mapping.keys() == weights_shape_mapping.keys()
            with open(weights_map_path, 'w') as fp:
                json.dump([weights_name_mapping, weights_shape_mapping], fp)
        else:
            print(f"[I] Found cached weights map: {weights_map_path} ")

    def optimize(self, onnx_graph):
        opt = Optimizer(onnx_graph, verbose=self.verbose)
        opt.info(self.name + ': original')
        opt.cleanup()
        opt.info(self.name + ': cleanup')
        opt.fold_constants()
        opt.info(self.name + ': fold constants')
        opt.infer_shapes()
        opt.info(self.name + ': shape inference')
        onnx_opt_graph = opt.cleanup(return_onnx=True)
        opt.info(self.name + ': finished')
        return onnx_opt_graph

    def check_dims(self, batch_size, image_height, image_width):
        assert batch_size >= self.min_batch and batch_size <= self.max_batch
        assert image_height % 8 == 0 or image_width % 8 == 0
        latent_height = image_height // 8
        latent_width = image_width // 8
        assert latent_height >= self.min_latent_shape and latent_height <= self.max_latent_shape
        assert latent_width >= self.min_latent_shape and latent_width <= self.max_latent_shape
        return (latent_height, latent_width)

    def get_minmax_dims(self, batch_size, image_height, image_width, static_batch, static_shape):
        min_batch = batch_size if static_batch else self.min_batch
        max_batch = batch_size if static_batch else self.max_batch
        latent_height = image_height // 8
        latent_width = image_width // 8
        min_image_height = image_height if static_shape else self.min_image_shape
        max_image_height = image_height if static_shape else self.max_image_shape
        min_image_width = image_width if static_shape else self.min_image_shape
        max_image_width = image_width if static_shape else self.max_image_shape
        min_latent_height = latent_height if static_shape else self.min_latent_shape
        max_latent_height = latent_height if static_shape else self.max_latent_shape
        min_latent_width = latent_width if static_shape else self.min_latent_shape
        max_latent_width = latent_width if static_shape else self.max_latent_shape
        return (min_batch, max_batch, min_image_height, max_image_height, min_image_width, max_image_width, min_latent_height, max_latent_height, min_latent_width, max_latent_width)


class CLIPModel(BaseModel):
    def __init__(self,
        device,
        framework_model_dir,
        embedding_dim,
        output_hidden_states=True,
    ):
        super(CLIPModel, self).__init__(device=device, framework_model_dir=framework_model_dir, embedding_dim=embedding_dim)
        # Output the final hidden state
        if output_hidden_states:
            self.extra_output_names = ['hidden_states']

    def get_model(self, model, torch_inference=''):
        model = optimize_checkpoint(model, torch_inference)
        return model

    def get_input_names(self):
        return ['input_ids']

    def get_output_names(self):
       return ['text_embeddings']

    def get_dynamic_axes(self):
        return {
            'input_ids': {0: 'B'},
            'text_embeddings': {0: 'B'}
        }

    def get_input_profile(self, batch_size, image_height, image_width, static_batch, static_shape):
        self.check_dims(batch_size, image_height, image_width)
        min_batch, max_batch, _, _, _, _, _, _, _, _ = self.get_minmax_dims(batch_size, image_height, image_width, static_batch, static_shape)
        return {
            'input_ids': [(min_batch, self.text_maxlen), (batch_size, self.text_maxlen), (max_batch, self.text_maxlen)]
        }

    def get_shape_dict(self, batch_size, image_height, image_width):
        self.check_dims(batch_size, image_height, image_width)
        output = {
            'input_ids': (batch_size, self.text_maxlen),
            'text_embeddings': (batch_size, self.text_maxlen, self.embedding_dim)
        }
        if 'hidden_states' in self.extra_output_names:
            output["hidden_states"] = (batch_size, self.text_maxlen, self.embedding_dim)
        return output

    def get_sample_input(self, batch_size, image_height, image_width):
        self.check_dims(batch_size, image_height, image_width)
        return torch.zeros(batch_size, self.text_maxlen, dtype=torch.int32, device=self.device)

    def optimize(self, onnx_graph):
        opt = Optimizer(onnx_graph, verbose=self.verbose)
        opt.info(self.name + ': original')
        opt.select_outputs([0]) # delete graph output#1
        opt.cleanup()
        opt.info(self.name + ': remove output[1]')
        opt.fold_constants()
        opt.info(self.name + ': fold constants')
        opt.infer_shapes()
        opt.info(self.name + ': shape inference')
        opt.select_outputs([0], names=['text_embeddings']) # rename network output
        opt.info(self.name + ': remove output[0]')
        opt_onnx_graph = opt.cleanup(return_onnx=True)
        if 'hidden_states' in self.extra_output_names:
            opt_onnx_graph = opt.clip_add_hidden_states(return_onnx=True)
            opt.info(self.name + ': added hidden_states')
        opt.info(self.name + ': finished')
        return opt_onnx_graph


class CLIPWithProjModel(CLIPModel):
    def __init__(self,
        device,
        framework_model_dir,
        embedding_dim,
        output_hidden_states=True,
    ):

        super(CLIPWithProjModel, self).__init__( device=device, framework_model_dir=framework_model_dir,  embedding_dim=embedding_dim, output_hidden_states=output_hidden_states)

    def get_model(self, model, torch_inference=''):
        model = optimize_checkpoint(model, torch_inference)
        return model

    def get_shape_dict(self, batch_size, image_height, image_width):
        self.check_dims(batch_size, image_height, image_width)
        output = {
            'input_ids': (batch_size, self.text_maxlen),
            'text_embeddings': (batch_size, self.embedding_dim)
        }
        if 'hidden_states' in self.extra_output_names:
            output["hidden_states"] = (batch_size, self.text_maxlen, self.embedding_dim)

        return output


class UNet2DConditionControlNetModel(torch.nn.Module):
    def __init__(self, unet, controlnets) -> None:
        super().__init__()
        self.unet = unet
        self.controlnets = controlnets
        
    def forward(self, sample, timestep, encoder_hidden_states, images, controlnet_scales):
        for i, (image, conditioning_scale, controlnet) in enumerate(zip(images, controlnet_scales, self.controlnets)):
            down_samples, mid_sample = controlnet(
                sample,
                timestep,
                encoder_hidden_states=encoder_hidden_states,
                controlnet_cond=image,
                return_dict=False,
            )

            down_samples = [
                    down_sample * conditioning_scale
                    for down_sample in down_samples
                ]
            mid_sample *= conditioning_scale
            
            # merge samples
            if i == 0:
                down_block_res_samples, mid_block_res_sample = down_samples, mid_sample
            else:
                down_block_res_samples = [
                    samples_prev + samples_curr
                    for samples_prev, samples_curr in zip(down_block_res_samples, down_samples)
                ]
                mid_block_res_sample += mid_sample
        
        noise_pred = self.unet(
            sample,
            timestep,
            encoder_hidden_states=encoder_hidden_states,
            down_block_additional_residuals=down_block_res_samples,
            mid_block_additional_residual=mid_block_res_sample
        )
        return noise_pred


class ControlNetModelTrt(BaseModel):
    def __init__(self,
        device,
        framework_model_dir,
        fp16 = False,
        max_batch_size = 16,
        do_classifier_free_guidance = False,
        embedding_dim = 16,
    ):
        super(ControlNetModelTrt, self).__init__(device=device, framework_model_dir=framework_model_dir, fp16=fp16, max_batch_size=max_batch_size)
        self.xB = 2 if do_classifier_free_guidance else 1
        self.embedding_dim = embedding_dim

    def get_model(self, model):
        model  = ControlNetWrapper(model)
        return model 

    def get_input_names(self): 
            return  ["control_model_input", "timestep", "encoder_hidden_states", 
                        "controlnet_cond","conditioning_scale",
                        "text_embeds", "time_ids"
                    ]

    def get_output_names(self):
       return ["down_block_additional_residuals_0",
                "down_block_additional_residuals_1",
                "down_block_additional_residuals_2",
                "down_block_additional_residuals_3",
                "down_block_additional_residuals_4",
                "down_block_additional_residuals_5",
                "down_block_additional_residuals_6",
                "down_block_additional_residuals_7",
                "down_block_additional_residuals_8",
                "mid_block_additional_residual"
               ]

    def get_dynamic_axes(self):
        return {
                "control_model_input": {0: "batch", 2: "latent_height", 3: "latent_width"},
                "encoder_hidden_states": {0: "batch", 1: "sequence"},
                "controlnet_cond": {0: "batch", 2: "image_height", 3: "image_width"},
                "text_embeds": {0: "batch"},
                "time_ids": {0: "batch"},
                "down_block_additional_residuals_0": {0: "batch", 2: "res0_height", 3: "res0_width"},
                "down_block_additional_residuals_1": {0: "batch", 2: "res1_height", 3: "res1_width"},
                "down_block_additional_residuals_2": {0: "batch", 2: "res2_height", 3: "res2_width"},
                "down_block_additional_residuals_3": {0: "batch", 2: "res3_height", 3: "res3_width"},
                "down_block_additional_residuals_4": {0: "batch", 2: "res4_height", 3: "res4_width"},
                "down_block_additional_residuals_5": {0: "batch", 2: "res5_height", 3: "res5_width"},
                "down_block_additional_residuals_6": {0: "batch", 2: "res6_height", 3: "res6_width"},
                "down_block_additional_residuals_7": {0: "batch", 2: "res7_height", 3: "res7_width"},
                "down_block_additional_residuals_8": {0: "batch", 2: "res8_height", 3: "res8_width"},
                "mid_block_additional_residual": {0: "batch", 2: "mid_height", 3: "mid_width"},
                }

    def get_input_profile(self, batch_size, image_height, image_width, static_batch, static_shape):
        latent_height, latent_width = self.check_dims(batch_size, image_height, image_width)
        min_batch, max_batch, min_image_height, max_image_height, min_image_width, max_image_width, min_latent_height, max_latent_height, min_latent_width, max_latent_width = \
        self.get_minmax_dims(batch_size, image_height, image_width, static_batch, static_shape)

        return {
            'control_model_input': [(self.xB*min_batch, 4, latent_height, latent_width), (self.xB*batch_size, 4, latent_height, latent_width), (self.xB*max_batch, 4, latent_height, latent_width)],
            'timestep': [(1,), (1,), (1,)],
            'controlnet_cond': [(self.xB*min_batch, 3, image_height, image_width), (self.xB*batch_size, 3, image_height, image_width), (self.xB*max_batch, 3, image_height, image_width)],
            'encoder_hidden_states':[(self.xB*min_batch, self.embedding_dim, 2048), (self.xB*batch_size, self.embedding_dim, 2048), (self.xB*max_batch, self.embedding_dim, 2048)],
            'conditioning_scale': [(1,), (1,), (1,)],
            'text_embeds': [(self.xB*min_batch, 1280), (self.xB*batch_size, 1280), (self.xB*max_batch, 1280)],
            'time_ids': [(self.xB*min_batch, 6), (self.xB*batch_size, 6), (self.xB*max_batch, 6)],
        }

    def get_shape_dict(self, batch_size, image_height, image_width):
        latent_height, latent_width = self.check_dims(batch_size, image_height, image_width)

        return {
            'control_model_input': (self.xB*batch_size, 4, latent_height, latent_width),
            'timestep': (1,), 
            'controlnet_cond': (self.xB*batch_size, 3, image_height, image_width), 
            'encoder_hidden_states':(self.xB*batch_size, self.embedding_dim, 2048),
            'conditioning_scale': (1,), 
            'text_embeds':  (self.xB*batch_size, 1280),
            'time_ids':  (self.xB*batch_size, 6),
            "down_block_additional_residuals_0": (self.xB*batch_size, 320, math.ceil(image_height/8), math.ceil(image_width/8)),
            "down_block_additional_residuals_1": (self.xB*batch_size, 320, math.ceil(image_height/8), math.ceil(image_width/8)), 
            "down_block_additional_residuals_2": (self.xB*batch_size, 320, math.ceil(image_height/8), math.ceil(image_width/8)), 
            "down_block_additional_residuals_3": (self.xB*batch_size, 320, math.ceil(image_height/16), math.ceil(image_width/16)), 
            "down_block_additional_residuals_4": (self.xB*batch_size, 640, math.ceil(image_height/16), math.ceil(image_width/16)),
            "down_block_additional_residuals_5": (self.xB*batch_size, 640, math.ceil(image_height/16), math.ceil(image_width/16)),
            "down_block_additional_residuals_6": (self.xB*batch_size, 640, math.ceil(image_height/32), math.ceil(image_width/32)), 
            "down_block_additional_residuals_7": (self.xB*batch_size,1280, math.ceil(image_height/32), math.ceil(image_width/32)),
            "down_block_additional_residuals_8": (self.xB*batch_size,1280, math.ceil(image_height/32), math.ceil(image_width/32)), 
            "mid_block_additional_residual": (self.xB*batch_size,1280, math.ceil(image_height/32), math.ceil(image_width/32))
        }

    def get_sample_input(self, batch_size, image_height, image_width):
        latent_height, latent_width = self.check_dims(batch_size, image_height, image_width)
        dtype = torch.float16 if self.fp16 else torch.float32
        return(
                torch.randn(self.xB*batch_size, 4, image_height//8, image_width//8, dtype=dtype, device=self.device),  
                torch.tensor([1.], dtype=dtype, device=self.device),
                torch.randn(self.xB*batch_size, self.embedding_dim, 2048, dtype=dtype, device=self.device),
                torch.randn(self.xB*batch_size, 3, image_height, image_width, dtype=dtype, device=self.device), 
                torch.tensor([1.], dtype=dtype, device=self.device),  
                torch.randn(self.xB*batch_size, 1280, dtype=dtype, device=self.device),
                torch.randn(self.xB*batch_size, 6, dtype=dtype, device=self.device)
        )


class ControlNetWrapper(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
        
    def forward(self, control_model_input, timestep, encoder_hidden_states, 
                controlnet_cond, conditioning_scale, text_embeds, time_ids):
        added_cond_kwargs = {
            "text_embeds": text_embeds,
            "time_ids": time_ids
        }
        dummy = encoder_hidden_states.sum() * 0.0
        control_model_input = control_model_input + dummy
        return self.model(
            control_model_input,
            timestep,
            encoder_hidden_states=encoder_hidden_states,
            controlnet_cond=controlnet_cond,
            conditioning_scale=conditioning_scale,
            added_cond_kwargs=added_cond_kwargs,
            guess_mode=False,
            return_dict=False
        )

class UNetXLModel(BaseModel):
    def __init__(self,
        device,
        framework_model_dir,
        fp16 = False,
        max_batch_size = 16,
        text_maxlen = 77,
        do_classifier_free_guidance = False,
    ):
        super(UNetXLModel, self).__init__(device=device, framework_model_dir=framework_model_dir, fp16=fp16, max_batch_size=max_batch_size, text_maxlen=text_maxlen, embedding_dim=2048)
        self.subfolder = 'unet'
        self.unet_dim = 4
        self.time_dim = 6
        self.xB = 2 if do_classifier_free_guidance else 1

    def get_model(self, model, ip_adapter_scale):
        model  = UNetXLWrapper(model, ip_adapter_scale)
        return model 
    
    def get_input_names(self):
        return ["sample", "timestep", "encoder_hidden_states", "text_embeds", "time_ids",
                             "down_block_additional_residuals_0",
                             "down_block_additional_residuals_1",
                             "down_block_additional_residuals_2",
                             "down_block_additional_residuals_3",
                             "down_block_additional_residuals_4",
                             "down_block_additional_residuals_5",
                             "down_block_additional_residuals_6",
                             "down_block_additional_residuals_7",
                             "down_block_additional_residuals_8",
                             "mid_block_additional_residual",
                             ]

    def get_output_names(self):
        # return ['latent']
        return ["out_sample"]

    def get_dynamic_axes(self):
        return {
                    "sample": {0: "batch", 2: "sample_height", 3: "sample_width"},
                    "encoder_hidden_states": {0: "batch"},
                    "text_embeds": {0: "batch"},
                    "time_ids": {0: "batch"},
                    "down_block_additional_residuals_0": {0: "batch", 2: "res0_height", 3: "res0_width"},
                    "down_block_additional_residuals_1": {0: "batch", 2: "res1_height", 3: "res1_width"},
                    "down_block_additional_residuals_2": {0: "batch", 2: "res2_height", 3: "res2_width"},
                    "down_block_additional_residuals_3": {0: "batch", 2: "res3_height", 3: "res3_width"},
                    "down_block_additional_residuals_4": {0: "batch", 2: "res4_height", 3: "res4_width"},
                    "down_block_additional_residuals_5": {0: "batch", 2: "res5_height", 3: "res5_width"},
                    "down_block_additional_residuals_6": {0: "batch", 2: "res6_height", 3: "res6_width"},
                    "down_block_additional_residuals_7": {0: "batch", 2: "res7_height", 3: "res7_width"},
                    "down_block_additional_residuals_8": {0: "batch", 2: "res8_height", 3: "res8_width"},
                    "mid_block_additional_residual": {0: "batch", 2: "mid_height", 3: "mid_width"},
                    "out_sample": {0: "batch", 2: "out_height", 3: "out_width"},
                }

    def get_input_profile(self, batch_size, image_height, image_width, static_batch, static_shape):
        latent_height, latent_width = self.check_dims(batch_size, image_height, image_width)
        min_batch, max_batch, _, _, _, _, min_latent_height, max_latent_height, min_latent_width, max_latent_width = \
            self.get_minmax_dims(batch_size, image_height, image_width, static_batch, static_shape)
        return {
            'sample': [(self.xB*min_batch, self.unet_dim, min_latent_height, min_latent_width), (self.xB*batch_size, self.unet_dim, latent_height, latent_width), (self.xB*max_batch, self.unet_dim, max_latent_height, max_latent_width)],
            'encoder_hidden_states': [(self.xB*min_batch, 93, self.embedding_dim), (self.xB*batch_size, 93, self.embedding_dim), (self.xB*max_batch, 93, self.embedding_dim)],
            'text_embeds': [(self.xB*min_batch, 1280), (self.xB*batch_size, 1280), (self.xB*max_batch, 1280)],
            'time_ids': [(self.xB*min_batch, self.time_dim), (self.xB*batch_size, self.time_dim), (self.xB*max_batch, self.time_dim)],
            "down_block_additional_residuals_0": [(self.xB*min_batch, 320, math.ceil(image_height/8), math.ceil(image_width/8)), (self.xB*batch_size, 320, math.ceil(image_height/8), math.ceil(image_width/8)), (self.xB*max_batch, 320, math.ceil(image_height/8), math.ceil(image_width/8))],
            "down_block_additional_residuals_1": [(self.xB*min_batch, 320, math.ceil(image_height/8), math.ceil(image_width/8)), (self.xB*batch_size, 320, math.ceil(image_height/8), math.ceil(image_width/8)), (self.xB*max_batch, 320, math.ceil(image_height/8), math.ceil(image_width/8))],
            "down_block_additional_residuals_2": [(self.xB*min_batch, 320, math.ceil(image_height/8), math.ceil(image_width/8)), (self.xB*batch_size, 320, math.ceil(image_height/8), math.ceil(image_width/8)), (self.xB*max_batch, 320, math.ceil(image_height/8), math.ceil(image_width/8))],
            "down_block_additional_residuals_3": [(self.xB*min_batch, 320, math.ceil(image_height/16), math.ceil(image_width/16)), (self.xB*batch_size, 320, math.ceil(image_height/16), math.ceil(image_width/16)), (self.xB*max_batch, 320, math.ceil(image_height/16), math.ceil(image_width/16))],
            "down_block_additional_residuals_4": [(self.xB*min_batch, 640, math.ceil(image_height/16), math.ceil(image_width/16)), (self.xB*batch_size, 640, math.ceil(image_height/16), math.ceil(image_width/16)), (self.xB*max_batch, 640, math.ceil(image_height/16), math.ceil(image_width/16))],
            "down_block_additional_residuals_5": [(self.xB*min_batch, 640, math.ceil(image_height/16), math.ceil(image_width/16)), (self.xB*batch_size, 640, math.ceil(image_height/16), math.ceil(image_width/16)), (self.xB*max_batch, 640, math.ceil(image_height/16), math.ceil(image_width/16))],
            "down_block_additional_residuals_6": [(self.xB*min_batch, 640, math.ceil(image_height/32), math.ceil(image_width/32)), (self.xB*batch_size, 640, math.ceil(image_height/32), math.ceil(image_width/32)), (self.xB*max_batch, 640, math.ceil(image_height/32), math.ceil(image_width/32))],
            "down_block_additional_residuals_7": [(self.xB*min_batch, 1280, math.ceil(image_height/32), math.ceil(image_width/32)), (self.xB*batch_size,1280, math.ceil(image_height/32), math.ceil(image_width/32)), (self.xB*max_batch, 1280, math.ceil(image_height/32), math.ceil(image_width/32))],
            "down_block_additional_residuals_8": [(self.xB*min_batch, 1280, math.ceil(image_height/32), math.ceil(image_width/32)), (self.xB*batch_size,1280, math.ceil(image_height/32), math.ceil(image_width/32)), (self.xB*max_batch, 1280, math.ceil(image_height/32), math.ceil(image_width/32))],
            "mid_block_additional_residual": [(self.xB*min_batch, 1280, math.ceil(image_height/32), math.ceil(image_width/32)), (self.xB*batch_size,1280, math.ceil(image_height/32), math.ceil(image_width/32)), (self.xB*max_batch, 1280, math.ceil(image_height/32), math.ceil(image_width/32))],
        }


    def get_shape_dict(self, batch_size, image_height, image_width):
        latent_height, latent_width = self.check_dims(batch_size, image_height, image_width)
        return {
            'sample': (self.xB*batch_size, self.unet_dim, latent_height, latent_width),
            'encoder_hidden_states': (self.xB*batch_size, 93, self.embedding_dim),
            'out_sample': (self.xB*batch_size, 4, latent_height, latent_width),
            'text_embeds': (self.xB*batch_size, 1280),
            'time_ids': (self.xB*batch_size, self.time_dim),
            'down_block_additional_residuals_0': (self.xB*batch_size, 320, math.ceil(image_height/8), math.ceil(image_width/8)),
            'down_block_additional_residuals_1': (self.xB*batch_size, 320, math.ceil(image_height/8), math.ceil(image_width/8)),
            'down_block_additional_residuals_2': (self.xB*batch_size, 320, math.ceil(image_height/8), math.ceil(image_width/8)),
            'down_block_additional_residuals_3': (self.xB*batch_size, 320, math.ceil(image_height/16), math.ceil(image_width/16)),
            'down_block_additional_residuals_4': (self.xB*batch_size, 640, math.ceil(image_height/16), math.ceil(image_width/16)),
            'down_block_additional_residuals_5': (self.xB*batch_size, 640, math.ceil(image_height/16), math.ceil(image_width/16)),
            'down_block_additional_residuals_6': (self.xB*batch_size, 640, math.ceil(image_height/32), math.ceil(image_width/32)),
            'down_block_additional_residuals_7': (self.xB*batch_size, 1280, math.ceil(image_height/32), math.ceil(image_width/32)),
            'down_block_additional_residuals_8': (self.xB*batch_size, 1280, math.ceil(image_height/32), math.ceil(image_width/32)),
            'mid_block_additional_residual': (self.xB*batch_size, 1280, math.ceil(image_height/32), math.ceil(image_width/32)),
        }
        
    def get_sample_input(self, batch_size, image_height, image_width):
        latent_height, latent_width = self.check_dims(batch_size, image_height, image_width)
        dtype = torch.float16 if self.fp16 else torch.float32
        return (
            torch.randn(self.xB*batch_size, self.unet_dim, latent_height, latent_width, dtype=torch.float32, device=self.device),
            torch.tensor([1.], dtype=torch.float32, device=self.device),
            torch.randn(self.xB*batch_size, 93, self.embedding_dim, dtype=dtype, device=self.device),
            torch.randn(self.xB*batch_size, 1280, dtype=dtype, device=self.device),
            torch.randn(self.xB*batch_size, 6, dtype=dtype, device=self.device),
            torch.randn(self.xB*batch_size, 320, math.ceil(image_height/8), math.ceil(image_width/8), dtype=dtype, device=self.device),
            torch.randn(self.xB*batch_size, 320, math.ceil(image_height/8), math.ceil(image_width/8), dtype=dtype, device=self.device),
            torch.randn(self.xB*batch_size, 320, math.ceil(image_height/8), math.ceil(image_width/8), dtype=dtype, device=self.device),
            torch.randn(self.xB*batch_size, 320, math.ceil(image_height/16), math.ceil(image_width/16), dtype=dtype, device=self.device),
            torch.randn(self.xB*batch_size, 640, math.ceil(image_height/16), math.ceil(image_width/16), dtype=dtype, device=self.device),
            torch.randn(self.xB*batch_size, 640, math.ceil(image_height/16), math.ceil(image_width/16), dtype=dtype, device=self.device),
            torch.randn(self.xB*batch_size, 640, math.ceil(image_height/32), math.ceil(image_width/32), dtype=dtype, device=self.device),
            torch.randn(self.xB*batch_size, 1280, math.ceil(image_height/32), math.ceil(image_width/32), dtype=dtype, device=self.device),
            torch.randn(self.xB*batch_size, 1280, math.ceil(image_height/32), math.ceil(image_width/32), dtype=dtype, device=self.device),
            torch.randn(self.xB*batch_size, 1280, math.ceil(image_height/32), math.ceil(image_width/32), dtype=dtype, device=self.device),
        )



from ip_adapter.attention_processor import IPAttnProcessor
class UNetXLWrapper(torch.nn.Module):
    def __init__(self, unet, ip_adapter_scale=0.8):
        super().__init__()
        self.unet = unet
        self.set_ip_adapter_scale(ip_adapter_scale)
    
    def set_ip_adapter_scale(self, scale):
        for attn_processor in self.unet.attn_processors.values():
            if isinstance(attn_processor, IPAttnProcessor):
                attn_processor.scale = scale
    
    def forward(self, sample, timestep, encoder_hidden_states, text_embeds, time_ids,
                down_block_res_0, down_block_res_1, down_block_res_2, down_block_res_3,
                down_block_res_4, down_block_res_5, down_block_res_6, down_block_res_7,
                down_block_res_8, mid_block_res):
        # Reconstruct down_block_res list
        down_block_res = [down_block_res_0, down_block_res_1, down_block_res_2, down_block_res_3,
                            down_block_res_4, down_block_res_5, down_block_res_6, down_block_res_7, 
                            down_block_res_8]
        
        # Reconstruct added_cond_kwargs dictionary
        added_cond_kwargs = {"text_embeds": text_embeds, "time_ids": time_ids}
        
        return self.unet(
            sample,
            timestep,
            encoder_hidden_states,
            added_cond_kwargs=added_cond_kwargs,
            down_block_additional_residuals=down_block_res,
            mid_block_additional_residual=mid_block_res,
            return_dict=False
        )[0]


class VAEModel(BaseModel):
    def __init__(self,
        version,
        pipeline,
        device,
        hf_token,
        verbose,
        framework_model_dir,
        max_batch_size=16,
    ):
        super(VAEModel, self).__init__(version, pipeline, device=device, hf_token=hf_token, verbose=verbose, framework_model_dir=framework_model_dir, max_batch_size=max_batch_size)
        self.subfolder = 'vae'

    def get_model(self, torch_inference=''):
        vae_decoder_model_path = get_checkpoint_dir(self.framework_model_dir, self.version, self.pipeline, self.subfolder, torch_inference)
        if not os.path.exists(vae_decoder_model_path):
            model = AutoencoderKL.from_pretrained(self.path,
                subfolder=self.subfolder,
                use_safetensors=self.hf_safetensor,
                use_auth_token=self.hf_token).to(self.device)
            model.save_pretrained(vae_decoder_model_path)
        else:
            print(f"[I] Load VAE decoder pytorch model from: {vae_decoder_model_path}")
            model = AutoencoderKL.from_pretrained(vae_decoder_model_path).to(self.device)
        model.forward = model.decode
        model = optimize_checkpoint(model, torch_inference)
        return model

    def get_input_names(self):
        return ['latent']

    def get_output_names(self):
       return ['images']

    def get_dynamic_axes(self):
        return {
            'latent': {0: 'B', 2: 'H', 3: 'W'},
            'images': {0: 'B', 2: '8H', 3: '8W'}
        }

    def get_input_profile(self, batch_size, image_height, image_width, static_batch, static_shape):
        latent_height, latent_width = self.check_dims(batch_size, image_height, image_width)
        min_batch, max_batch, _, _, _, _, min_latent_height, max_latent_height, min_latent_width, max_latent_width = \
            self.get_minmax_dims(batch_size, image_height, image_width, static_batch, static_shape)
        return {
            'latent': [(min_batch, 4, min_latent_height, min_latent_width), (batch_size, 4, latent_height, latent_width), (max_batch, 4, max_latent_height, max_latent_width)]
        }

    def get_shape_dict(self, batch_size, image_height, image_width):
        latent_height, latent_width = self.check_dims(batch_size, image_height, image_width)
        return {
            'latent': (batch_size, 4, latent_height, latent_width),
            'images': (batch_size, 3, image_height, image_width)
        }

    def get_sample_input(self, batch_size, image_height, image_width):
        latent_height, latent_width = self.check_dims(batch_size, image_height, image_width)
        return torch.randn(batch_size, 4, latent_height, latent_width, dtype=torch.float32, device=self.device)


class TorchVAEEncoder(torch.nn.Module):
    def __init__(self, version, pipeline, hf_token, device, path, framework_model_dir, hf_safetensor=False):
        super().__init__()
        vae_encoder_model_dir = get_checkpoint_dir(framework_model_dir, version, pipeline, 'vae_encoder', '')
        if not os.path.exists(vae_encoder_model_dir):
            self.vae_encoder = AutoencoderKL.from_pretrained(path,
                subfolder='vae',
                use_safetensors=hf_safetensor,
                use_auth_token=hf_token).to(device)
            self.vae_encoder.save_pretrained(vae_encoder_model_dir)
        else:
            print(f"[I] Load VAE encoder pytorch model from: {vae_encoder_model_dir}")
            self.vae_encoder = AutoencoderKL.from_pretrained(vae_encoder_model_dir).to(device)

    def forward(self, x):
        return self.vae_encoder.encode(x).latent_dist.sample()


class VAEEncoderModel(BaseModel):
    def __init__(self,
        version,
        pipeline,
        device,
        hf_token,
        verbose,
        framework_model_dir,
        max_batch_size=16,
    ):
        super(VAEEncoderModel, self).__init__(version, pipeline, device=device, hf_token=hf_token, verbose=verbose, framework_model_dir=framework_model_dir, max_batch_size=max_batch_size)

    def get_model(self, torch_inference=''):
        vae_encoder = TorchVAEEncoder(self.version, self.pipeline, self.hf_token, self.device, self.path, self.framework_model_dir, hf_safetensor=self.hf_safetensor)
        return vae_encoder

    def get_input_names(self):
        return ['images']

    def get_output_names(self):
       return ['latent']

    def get_dynamic_axes(self):
        return {
            'images': {0: 'B', 2: '8H', 3: '8W'},
            'latent': {0: 'B', 2: 'H', 3: 'W'}
        }

    def get_input_profile(self, batch_size, image_height, image_width, static_batch, static_shape):
        assert batch_size >= self.min_batch and batch_size <= self.max_batch
        min_batch = batch_size if static_batch else self.min_batch
        max_batch = batch_size if static_batch else self.max_batch
        self.check_dims(batch_size, image_height, image_width)
        min_batch, max_batch, min_image_height, max_image_height, min_image_width, max_image_width, _, _, _, _ = \
            self.get_minmax_dims(batch_size, image_height, image_width, static_batch, static_shape)

        return {
            'images': [(min_batch, 3, min_image_height, min_image_width), (batch_size, 3, image_height, image_width), (max_batch, 3, max_image_height, max_image_width)],
        }

    def get_shape_dict(self, batch_size, image_height, image_width):
        latent_height, latent_width = self.check_dims(batch_size, image_height, image_width)
        return {
            'images': (batch_size, 3, image_height, image_width),
            'latent': (batch_size, 4, latent_height, latent_width)
        }

    def get_sample_input(self, batch_size, image_height, image_width):
        self.check_dims(batch_size, image_height, image_width)
        return torch.randn(batch_size, 3, image_height, image_width, dtype=torch.float32, device=self.device)


def make_tokenizer(version, pipeline, hf_token, framework_model_dir, subfolder="tokenizer", **kwargs):
    tokenizer_model_dir = get_checkpoint_dir(framework_model_dir, version, pipeline.name, subfolder, '')
    if not os.path.exists(tokenizer_model_dir):
        model = CLIPTokenizer.from_pretrained(get_path(version, pipeline),
                subfolder=subfolder,
                use_safetensors=pipeline.is_sd_xl(),
                use_auth_token=hf_token)
        model.save_pretrained(tokenizer_model_dir)
    else:
        print(f"[I] Load tokenizer pytorch model from: {tokenizer_model_dir}")
        model = CLIPTokenizer.from_pretrained(tokenizer_model_dir)
    return model
