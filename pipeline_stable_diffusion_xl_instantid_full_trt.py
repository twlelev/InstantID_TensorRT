# Copyright 2024 The InstantX Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import cv2
import math

import numpy as np
import PIL.Image
import torch
import torch.nn.functional as F

from diffusers.image_processor import PipelineImageInput

from diffusers.models import ControlNetModel

from diffusers.utils import (
    deprecate,
    logging,
    replace_example_docstring,
)
from diffusers.utils.torch_utils import is_compiled_module, is_torch_version
from diffusers.pipelines.stable_diffusion_xl import StableDiffusionXLPipelineOutput

from diffusers import StableDiffusionXLControlNetPipeline
from diffusers.pipelines.controlnet.multicontrolnet import MultiControlNetModel
from diffusers.utils.import_utils import is_xformers_available

from ip_adapter.resampler import Resampler
from ip_adapter.utils import is_torch2_available

if is_torch2_available():
    from ip_adapter.attention_processor import IPAttnProcessor2_0 as IPAttnProcessor, AttnProcessor2_0 as AttnProcessor
else:
    from ip_adapter.attention_processor import IPAttnProcessor, AttnProcessor
from ip_adapter.attention_processor import region_control

from Tensorrt_Build.utilities import Engine
import os
from Tensorrt_Build.models import *
from cuda import cudart
import pathlib
import tensorrt as trt
import time


logger = logging.get_logger(__name__)  # pylint: disable=invalid-name


EXAMPLE_DOC_STRING = """
    Examples:
        ```py
        >>> # !pip install opencv-python transformers accelerate insightface
        >>> import diffusers
        >>> from diffusers.utils import load_image
        >>> from diffusers.models import ControlNetModel

        >>> import cv2
        >>> import torch
        >>> import numpy as np
        >>> from PIL import Image
        
        >>> from insightface.app import FaceAnalysis
        >>> from pipeline_stable_diffusion_xl_instantid import StableDiffusionXLInstantIDPipeline, draw_kps

        >>> # download 'antelopev2' under ./models
        >>> app = FaceAnalysis(name='antelopev2', root='./', providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
        >>> app.prepare(ctx_id=0, det_size=(640, 640))
        
        >>> # download models under ./checkpoints
        >>> face_adapter = f'./checkpoints/ip-adapter.bin'
        >>> controlnet_path = f'./checkpoints/ControlNetModel'
        
        >>> # load IdentityNet
        >>> controlnet = ControlNetModel.from_pretrained(controlnet_path, torch_dtype=torch.float16)
        
        >>> pipe = StableDiffusionXLInstantIDPipeline.from_pretrained(
        ...     "stabilityai/stable-diffusion-xl-base-1.0", controlnet=controlnet, torch_dtype=torch.float16
        ... )
        >>> pipe.cuda()
        
        >>> # load adapter
        >>> pipe.load_ip_adapter_instantid(face_adapter)

        >>> prompt = "analog film photo of a man. faded film, desaturated, 35mm photo, grainy, vignette, vintage, Kodachrome, Lomography, stained, highly detailed, found footage, masterpiece, best quality"
        >>> negative_prompt = "(lowres, low quality, worst quality:1.2), (text:1.2), watermark, painting, drawing, illustration, glitch, deformed, mutated, cross-eyed, ugly, disfigured (lowres, low quality, worst quality:1.2), (text:1.2), watermark, painting, drawing, illustration, glitch,deformed, mutated, cross-eyed, ugly, disfigured"

        >>> # load an image
        >>> image = load_image("your-example.jpg")
        
        >>> face_info = app.get(cv2.cvtColor(np.array(face_image), cv2.COLOR_RGB2BGR))[-1]
        >>> face_emb = face_info['embedding']
        >>> face_kps = draw_kps(face_image, face_info['kps'])
        
        >>> pipe.set_ip_adapter_scale(0.8)

        >>> # generate image
        >>> image = pipe(
        ...     prompt, image_embeds=face_emb, image=face_kps, controlnet_conditioning_scale=0.8
        ... ).images[0]
        ```
""" 

from transformers import CLIPTokenizer
from diffusers.pipelines.stable_diffusion_xl import StableDiffusionXLPipeline
def draw_kps(image_pil, kps, color_list=[(255,0,0), (0,255,0), (0,0,255), (255,255,0), (255,0,255)]):
    
    stickwidth = 4
    limbSeq = np.array([[0, 2], [1, 2], [3, 2], [4, 2]])
    kps = np.array(kps)

    w, h = image_pil.size
    out_img = np.zeros([h, w, 3])

    for i in range(len(limbSeq)):
        index = limbSeq[i]
        color = color_list[index[0]]

        x = kps[index][:, 0]
        y = kps[index][:, 1]
        length = ((x[0] - x[1]) ** 2 + (y[0] - y[1]) ** 2) ** 0.5
        angle = math.degrees(math.atan2(y[0] - y[1], x[0] - x[1]))
        polygon = cv2.ellipse2Poly((int(np.mean(x)), int(np.mean(y))), (int(length / 2), stickwidth), int(angle), 0, 360, 1)
        out_img = cv2.fillConvexPoly(out_img.copy(), polygon, color)
    out_img = (out_img * 0.6).astype(np.uint8)

    for idx_kp, kp in enumerate(kps):
        color = color_list[idx_kp]
        x, y = kp
        out_img = cv2.circle(out_img.copy(), (int(x), int(y)), 10, color, -1)

    out_img_pil = PIL.Image.fromarray(out_img.astype(np.uint8))
    return out_img_pil
    
class StableDiffusionXLInstantIDPipeline(StableDiffusionXLControlNetPipeline):

    def tensorrtInit(self, args):
        self.args = args

        try:
            self.loadEngines(args)
            # Load resources
            _, shared_device_memory = cudart.cudaMalloc(self.get_max_device_memory())
            self.activateEngines(shared_device_memory)
            self.loadResources(args.height, args.width, args.batch, None)
            cudart.cudaDeviceSynchronize()
        except Exception as e:
            raise e   

    def loadEngines(self, args):
        self.models = {}
        self.engine = {}

        for directory in [args.engine_dir, args.onnx_dir]:
            if not os.path.exists(directory):
                print(f"[I] Create directory: {directory}")
                pathlib.Path(directory).mkdir(parents=True)
        
        self._guidance_scale = args.guidance_scale
        if 'unet' in args.stages:
            self.models['unet'] = UNetXLModel(device=self.device_infer, fp16=True,  framework_model_dir=args.onnx_dir, do_classifier_free_guidance=self.do_classifier_free_guidance)
        
        if 'controlnet' in args.stages:
            self.models['controlnet'] = ControlNetModelTrt(device=self.device_infer, fp16=True, framework_model_dir=args.onnx_dir, do_classifier_free_guidance=self.do_classifier_free_guidance)

        if 'clip' in args.stages:
            self.models['clip'] = CLIPModel(device=self.device_infer, framework_model_dir=args.onnx_dir, embedding_dim=768)

        if 'clip2' in args.stages:
            self.models['clip2'] = CLIPWithProjModel(device=self.device_infer, framework_model_dir=args.onnx_dir, embedding_dim=1280)
        
        if 'multicontrolnet' in args.stages:
            if not hasattr(self, 'controlnet_num'):
                self.controlnet_num = len(self.controlnet.nets)
                for i in range(self.controlnet_num):
                    name = 'controlnet'+ str(i)
                    if i==0:
                        self.models[name] = ControlNetModelTrt(device=self.device_infer, fp16=True, framework_model_dir=args.onnx_dir, do_classifier_free_guidance=self.do_classifier_free_guidance, embedding_dim=16)
                    else:
                        self.models[name] = ControlNetModelTrt(device=self.device_infer, fp16=True, framework_model_dir=args.onnx_dir, do_classifier_free_guidance=self.do_classifier_free_guidance, embedding_dim=77)
        
        # Configure pipeline models to load
        model_names = self.models.keys()
        onnx_path = dict(zip(model_names, [self.getOnnxPath(model_name, args.onnx_dir, opt=False) for model_name in model_names]))
        onnx_opt_path = dict(zip(model_names, [self.getOnnxPath(model_name, args.onnx_dir,) for model_name in model_names]))
        engine_path = dict(zip(model_names, [self.getEnginePath(model_name, args.engine_dir) for model_name in model_names]))

        # Export models to ONNX and save weights name mapping
        for model_name, obj in self.models.items():
            do_export_onnx = not os.path.exists(engine_path[model_name]) and not os.path.exists(onnx_opt_path[model_name])
            # FIXME do_export_weights_map needs ONNX graph
            if do_export_onnx :
    
                if model_name=='unet':
                    model = obj.get_model(self.unet, args.ip_adapter_scale)
                elif model_name=='controlnet':
                    model = obj.get_model(self.controlnet)
                elif model_name=='clip':
                    model = obj.get_model(self.text_encoder)      
                elif model_name=='clip2':
                    model = obj.get_model(self.text_encoder_2) 
                elif model_name.startswith('controlnet'):
                    controlnet_idx = int(model_name[10:])
                    model = obj.get_model(self.controlnet.nets[controlnet_idx]) 
                obj.export_onnx(model, onnx_path[model_name], onnx_opt_path[model_name], args.onnx_opset, args.height, args.width, args.batch)
                self.release_model(model_name)
        # Build TensorRT engines
        for model_name, obj in self.models.items():
            engine = Engine(engine_path[model_name])
            if not os.path.exists(engine_path[model_name]):
                update_output_names = obj.get_output_names() + obj.extra_output_names if obj.extra_output_names else None
                use_tf32 = model_name == 'vae' 
                engine.build(onnx_opt_path[model_name],
                    fp16=not use_tf32,
                    tf32=use_tf32,
                    input_profile=obj.get_input_profile(
                        args.batch, args.height, args.width,
                        static_batch=False, static_shape=True
                    ), 
                    enable_refit=False,
                    enable_all_tactics=False,
                    timing_cache=None,
                    update_output_names=update_output_names)
                
            self.engine[model_name] = engine   
                      
        for model_name, obj in self.models.items():
            self.release_model(model_name)
            self.engine[model_name].load()


    def release_model(self, model_name):
        if model_name=='unet':
            config = self.unet.config
            add_embedding = self.unet.add_embedding
            dtype = self.unet.dtype
            if hasattr(self.unet, 'to'):
                    self.unet = self.unet.to('cpu')
            del self.unet
            class MinimalUNet:
                def __init__(self, config, add_embedding):
                    self.config = config
                    self.add_embedding = add_embedding
            self.unet = MinimalUNet(config, add_embedding)
            self.unet.dtype = dtype
            
        if model_name=='controlnet':
            if hasattr(self, 'controlnet'):
                if self.controlnet is not None:
                    try:
                        self.controlnet = self.controlnet.to('cpu')
                        del self.controlnet 
                    except AttributeError:
                        pass  
                    
        elif model_name=='clip':
            if hasattr(self, 'text_encoder'):
                if self.text_encoder is not None:
                    try:
                        self.text_encoder = self.text_encoder.to('cpu')
                        del self.text_encoder 
                    except AttributeError:
                        pass  
                    
        elif model_name=='clip2':
            if hasattr(self, 'text_encoder_2'):
                if self.text_encoder_2 is not None:
                    projection_dim = self.text_encoder_2.config.projection_dim
                    
                    class MinimalConfig:
                        def __init__(self, projection_dim):
                            self.projection_dim = projection_dim
                    class MinimalTextEncoder2:
                        def __init__(self, config):
                            self.config = config
                    try:
                        self.text_encoder_2 = self.text_encoder_2.to('cpu')
                        del self.text_encoder_2 
                    except AttributeError:
                        pass  
                    self.text_encoder_2 = MinimalTextEncoder2(MinimalConfig(projection_dim))
                    assert hasattr(self.text_encoder_2, 'config') and hasattr(self.text_encoder_2.config, 'projection_dim')
                    assert self.text_encoder_2.config.projection_dim == projection_dim   
        
        elif model_name.startswith('controlnet'):
            controlnet_idx = int(model_name[10:])
            if self.controlnet is not None:
                try:
                    self.controlnet.nets[controlnet_idx] = self.controlnet.nets[controlnet_idx].to('cpu')
                    self.controlnet.nets[controlnet_idx] = None
                except AttributeError:
                    pass  
        
        torch.cuda.empty_cache()

    def runEngine(self, model_name, feed_dict):
        # Ensure device is set correctly
        cudart.cudaSetDevice(self.device_infer.index)
        
        # Check if engine exists
        if model_name not in self.engine:
            raise RuntimeError(f"Engine {model_name} not found")
        
        engine = self.engine[model_name]
        
        # Ensure the stream is valid
        if not self.stream:
            _, self.stream = cudart.cudaStreamCreate()

        # Run inference with proper error handling
        try:
            result = engine.infer(feed_dict, self.stream, self.device_infer, use_cuda_graph=True)
            # Ensure operations are complete
            status = cudart.cudaStreamSynchronize(self.stream)
            if status != 0:
                raise RuntimeError(f"CUDA stream synchronize failed with error: {status}")
            return result
        except Exception as e:
            print(f"Error running engine {model_name}: {e}")
            # Try to recover from error
            cudart.cudaDeviceSynchronize()
            torch.cuda.empty_cache()
            raise

    def teardown(self):
        for e in self.events.values():
            cudart.cudaEventDestroy(e[0])
            cudart.cudaEventDestroy(e[1])

        for engine in self.engine.values():
            del engine

        if self.shared_device_memory:
            cudart.cudaFree(self.shared_device_memory)

        cudart.cudaStreamDestroy(self.stream)
        del self.stream
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        import gc 
        gc.collect()

    def get_max_device_memory(self):
        max_device_memory = self.calculateMaxDeviceMemory()
        return max_device_memory    

    def calculateMaxDeviceMemory(self):
        max_device_memory = 0
        for model_name, engine in self.engine.items():
            max_device_memory = max(max_device_memory, engine.engine.device_memory_size)
        return max_device_memory

    def activateEngines(self, shared_device_memory=None):
        if shared_device_memory is None:
            max_device_memory = self.calculateMaxDeviceMemory()
            _, shared_device_memory = cudart.cudaMalloc(max_device_memory)
        self.shared_device_memory = shared_device_memory
        # Load and activate TensorRT engines
        for engine in self.engine.values():
            engine.activate(reuse_device_memory=self.shared_device_memory)

    def loadResources(self, image_height, image_width, batch_size, seed=None):
        # Initialize noise generator
        if seed:
            self.seed = seed
            self.generator = torch.Generator(device=self.device_infer).manual_seed(seed)


        # Create CUDA events and stream
        if not hasattr(self, 'events'):
            self.events = {}
        for stage in ['clip', 'denoise', 'vae', 'vae_encoder']:
            self.events[stage] = [cudart.cudaEventCreate()[1], cudart.cudaEventCreate()[1]]

        # Create CUDA stream
        self.stream = cudart.cudaStreamCreate()[1]

        for model_name, obj in self.models.items():
            if not (model_name == 'vae' and self.config.get('vae_torch_fallback', False)):
                if model_name == 'clip':
                    self.engine[model_name].allocate_buffers(shape_dict=obj.get_shape_dict(1, image_height, image_width), device=self.device_infer)
                else:
                    self.engine[model_name].allocate_buffers(shape_dict=obj.get_shape_dict(batch_size, image_height, image_width), device=self.device_infer)

    def runEngine(self, model_name, feed_dict):
        engine = self.engine[model_name]
        return engine.infer(feed_dict, self.stream, self.device_infer, use_cuda_graph=True)

    def cuda(self, device, dtype=torch.float16, use_xformers=False):
        self.device_infer = torch.device(device)
        self.to(device, dtype)
        if hasattr(self, "image_proj_model"):
            self.image_proj_model.to(device).to(self.unet.dtype)
    
    def load_ip_adapter_instantid(self, model_ckpt, image_emb_dim=512, num_tokens=16, scale=0.5):     
        self.set_image_proj_model(model_ckpt, image_emb_dim, num_tokens)
        self.set_ip_adapter(model_ckpt, num_tokens, scale)
        
    def set_image_proj_model(self, model_ckpt, image_emb_dim=512, num_tokens=16):
        image_proj_model = Resampler(
            dim=1280,
            depth=4,
            dim_head=64,
            heads=20,
            num_queries=num_tokens,
            embedding_dim=image_emb_dim,
            output_dim=self.unet.config.cross_attention_dim,
            ff_mult=4,
        )

        image_proj_model.eval()
        
        self.image_proj_model = image_proj_model.to(self.device_infer, dtype=self.dtype)
        state_dict = torch.load(model_ckpt, map_location="cpu")
        if 'image_proj' in state_dict:
            state_dict = state_dict["image_proj"]
        self.image_proj_model.load_state_dict(state_dict)
        
        self.image_proj_model_in_features = image_emb_dim
    
    def set_ip_adapter(self, model_ckpt, num_tokens, scale):
        
        unet = self.unet
        attn_procs = {}
        for name in unet.attn_processors.keys():
            cross_attention_dim = None if name.endswith("attn1.processor") else unet.config.cross_attention_dim
            if name.startswith("mid_block"):
                hidden_size = unet.config.block_out_channels[-1]
            elif name.startswith("up_blocks"):
                block_id = int(name[len("up_blocks.")])
                hidden_size = list(reversed(unet.config.block_out_channels))[block_id]
            elif name.startswith("down_blocks"):
                block_id = int(name[len("down_blocks.")])
                hidden_size = unet.config.block_out_channels[block_id]
            if cross_attention_dim is None:
                attn_procs[name] = AttnProcessor().to(unet.device, dtype=unet.dtype)
            else:
                attn_procs[name] = IPAttnProcessor(hidden_size=hidden_size, 
                                                   cross_attention_dim=cross_attention_dim, 
                                                   scale=scale,
                                                   num_tokens=num_tokens).to(unet.device, dtype=unet.dtype)
        unet.set_attn_processor(attn_procs)
        
        state_dict = torch.load(model_ckpt, map_location="cpu")
        ip_layers = torch.nn.ModuleList(self.unet.attn_processors.values())
        if 'ip_adapter' in state_dict:
            state_dict = state_dict['ip_adapter']
        ip_layers.load_state_dict(state_dict)
    
    def set_ip_adapter_scale(self, scale):
        unet = getattr(self, self.unet_name) if not hasattr(self, "unet") else self.unet
        for attn_processor in unet.attn_processors.values():
            if isinstance(attn_processor, IPAttnProcessor):
                attn_processor.scale = scale

    def _encode_prompt_image_emb(self, prompt_image_emb, device, num_images_per_prompt, dtype, do_classifier_free_guidance):
        
        if isinstance(prompt_image_emb, torch.Tensor):
            prompt_image_emb = prompt_image_emb.clone().detach()
        else:
            prompt_image_emb = torch.tensor(prompt_image_emb)
            
        prompt_image_emb = prompt_image_emb.reshape([1, -1, self.image_proj_model_in_features])
        
        if do_classifier_free_guidance:
            prompt_image_emb = torch.cat([torch.zeros_like(prompt_image_emb), prompt_image_emb], dim=0)
        else:
            prompt_image_emb = torch.cat([prompt_image_emb], dim=0)
        
        prompt_image_emb = prompt_image_emb.to(device=self.image_proj_model.latents.device, 
                                               dtype=self.image_proj_model.latents.dtype)
        prompt_image_emb = self.image_proj_model(prompt_image_emb)

        bs_embed, seq_len, _ = prompt_image_emb.shape
        prompt_image_emb = prompt_image_emb.repeat(1, num_images_per_prompt, 1)
        prompt_image_emb = prompt_image_emb.view(bs_embed * num_images_per_prompt, seq_len, -1)
        
        return prompt_image_emb.to(device=device, dtype=dtype)


    def cachedModelName(self, model_name):
            return model_name

    def getOnnxPath(self, model_name, onnx_dir, opt=True, suffix=''):
        onnx_model_dir = os.path.join(onnx_dir, self.cachedModelName(model_name)+('.opt' if opt else ''))
        os.makedirs(onnx_model_dir, exist_ok=True)
        return os.path.join(onnx_model_dir, 'model.onnx')

    def getEnginePath(self, model_name, engine_dir, enable_refit=False, suffix=''):
        return os.path.join(engine_dir, self.cachedModelName(model_name)+('.refit' if enable_refit else '')+'.trt'+trt.__version__+'.plan')

    def getWeightsMapPath(self, model_name, onnx_dir):
        onnx_model_dir = os.path.join(onnx_dir, self.cachedModelName(model_name)+'.opt')
        os.makedirs(onnx_model_dir, exist_ok=True)
        return os.path.join(onnx_model_dir, 'weights_map.json')

    def getRefitNodesPath(self, model_name, onnx_dir, suffix=''):
        onnx_model_dir = os.path.join(onnx_dir, self.cachedModelName(model_name)+'.opt')
        os.makedirs(onnx_model_dir, exist_ok=True)
        return os.path.join(onnx_model_dir, 'refit'+suffix+'.json')


    def profile_start(self, name, color='blue'):
        if name in self.events:
            cudart.cudaEventRecord(self.events[name][0], 0)

    def profile_stop(self, name):
        if name in self.events:
            cudart.cudaEventRecord(self.events[name][1], 0)

    def encode_prompt(self, prompt, negative_prompt, encoder='clip', pooled_outputs=False, output_hidden_states=False):
        self.profile_start('clip', color='green')

        tokenizer = self.tokenizer_2 if encoder == 'clip2' else self.tokenizer

        def tokenize(prompt, output_hidden_states):
            text_input_ids = tokenizer(
                prompt,
                padding="max_length",
                max_length=tokenizer.model_max_length,
                truncation=True,
                return_tensors="pt",
            ).input_ids.type(torch.int32).to(self.device_infer)

            text_hidden_states = None
            # NOTE: output tensor for CLIP must be cloned because it will be overwritten when called again for negative prompt
            if encoder not in self.args.stages:
                if encoder == 'clip':
                    outputs = self.text_encoder(text_input_ids)

                elif encoder == 'clip2':
                    outputs = self.text_encoder_2(text_input_ids)

                text_embeddings = outputs[0].clone()
                if output_hidden_states:
                    text_hidden_states = outputs['last_hidden_state'].clone()            
        
            else:
                outputs = self.runEngine(encoder, {'input_ids': text_input_ids})
                text_embeddings = outputs['text_embeddings'].clone()
                if output_hidden_states:
                    text_hidden_states = outputs['hidden_states'].clone()
            return text_embeddings, text_hidden_states

        # Tokenize prompt
        text_embeddings, text_hidden_states = tokenize(prompt, output_hidden_states)

        if self.do_classifier_free_guidance:
            # Tokenize negative prompt
            uncond_embeddings, uncond_hidden_states = tokenize(negative_prompt, output_hidden_states)

            # Concatenate the unconditional and text embeddings into a single batch to avoid doing two forward passes for classifier free guidance
            text_embeddings = torch.cat([uncond_embeddings, text_embeddings]).to(dtype=torch.float16)

        if pooled_outputs:
            pooled_output = text_embeddings

        if output_hidden_states:
            text_embeddings = torch.cat([uncond_hidden_states, text_hidden_states]).to(dtype=torch.float16) if self.do_classifier_free_guidance else text_hidden_states

        self.profile_stop('clip')
        if pooled_outputs:
            return text_embeddings, pooled_output
        return text_embeddings

    @torch.no_grad()
    @replace_example_docstring(EXAMPLE_DOC_STRING)
    def __call__(
        self,
        prompt: Union[str, List[str]] = None,
        prompt_2: Optional[Union[str, List[str]]] = None,
        image: PipelineImageInput = None,
        height: Optional[int] = None,
        width: Optional[int] = None,
        num_inference_steps: int = 50,
        guidance_scale: float = 5.0,
        negative_prompt: Optional[Union[str, List[str]]] = None,
        negative_prompt_2: Optional[Union[str, List[str]]] = None,
        num_images_per_prompt: Optional[int] = 1,
        eta: float = 0.0,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.FloatTensor] = None,
        prompt_embeds: Optional[torch.FloatTensor] = None,
        negative_prompt_embeds: Optional[torch.FloatTensor] = None,
        pooled_prompt_embeds: Optional[torch.FloatTensor] = None,
        negative_pooled_prompt_embeds: Optional[torch.FloatTensor] = None,
        image_embeds: Optional[torch.FloatTensor] = None,
        output_type: Optional[str] = "pil",
        return_dict: bool = True,
        cross_attention_kwargs: Optional[Dict[str, Any]] = None,
        controlnet_conditioning_scale: Union[float, List[float]] = 1.0,
        guess_mode: bool = False,
        control_guidance_start: Union[float, List[float]] = 0.0,
        control_guidance_end: Union[float, List[float]] = 1.0,
        original_size: Tuple[int, int] = None,
        crops_coords_top_left: Tuple[int, int] = (0, 0),
        target_size: Tuple[int, int] = None,
        negative_original_size: Optional[Tuple[int, int]] = None,
        negative_crops_coords_top_left: Tuple[int, int] = (0, 0),
        negative_target_size: Optional[Tuple[int, int]] = None,
        clip_skip: Optional[int] = None,
        callback_on_step_end: Optional[Callable[[int, int, Dict], None]] = None,
        callback_on_step_end_tensor_inputs: List[str] = ["latents"],

        # IP adapter
        ip_adapter_scale=None,

        # Enhance Face Region
        control_mask = None,

        **kwargs,
    ):
        r"""
        The call function to the pipeline for generation.

        Args:
            prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts to guide image generation. If not defined, you need to pass `prompt_embeds`.
            prompt_2 (`str` or `List[str]`, *optional*):
                The prompt or prompts to be sent to `tokenizer_2` and `text_encoder_2`. If not defined, `prompt` is
                used in both text-encoders.
            image (`torch.FloatTensor`, `PIL.Image.Image`, `np.ndarray`, `List[torch.FloatTensor]`, `List[PIL.Image.Image]`, `List[np.ndarray]`,:
                    `List[List[torch.FloatTensor]]`, `List[List[np.ndarray]]` or `List[List[PIL.Image.Image]]`):
                The ControlNet input condition to provide guidance to the `unet` for generation. If the type is
                specified as `torch.FloatTensor`, it is passed to ControlNet as is. `PIL.Image.Image` can also be
                accepted as an image. The dimensions of the output image defaults to `image`'s dimensions. If height
                and/or width are passed, `image` is resized accordingly. If multiple ControlNets are specified in
                `init`, images must be passed as a list such that each element of the list can be correctly batched for
                input to a single ControlNet.
            height (`int`, *optional*, defaults to `self.unet.config.sample_size * self.vae_scale_factor`):
                The height in pixels of the generated image. Anything below 512 pixels won't work well for
                [stabilityai/stable-diffusion-xl-base-1.0](https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0)
                and checkpoints that are not specifically fine-tuned on low resolutions.
            width (`int`, *optional*, defaults to `self.unet.config.sample_size * self.vae_scale_factor`):
                The width in pixels of the generated image. Anything below 512 pixels won't work well for
                [stabilityai/stable-diffusion-xl-base-1.0](https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0)
                and checkpoints that are not specifically fine-tuned on low resolutions.
            num_inference_steps (`int`, *optional*, defaults to 50):
                The number of denoising steps. More denoising steps usually lead to a higher quality image at the
                expense of slower inference.
            guidance_scale (`float`, *optional*, defaults to 5.0):
                A higher guidance scale value encourages the model to generate images closely linked to the text
                `prompt` at the expense of lower image quality. Guidance scale is enabled when `guidance_scale > 1`.
            negative_prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts to guide what to not include in image generation. If not defined, you need to
                pass `negative_prompt_embeds` instead. Ignored when not using guidance (`guidance_scale < 1`).
            negative_prompt_2 (`str` or `List[str]`, *optional*):
                The prompt or prompts to guide what to not include in image generation. This is sent to `tokenizer_2`
                and `text_encoder_2`. If not defined, `negative_prompt` is used in both text-encoders.
            num_images_per_prompt (`int`, *optional*, defaults to 1):
                The number of images to generate per prompt.
            eta (`float`, *optional*, defaults to 0.0):
                Corresponds to parameter eta (η) from the [DDIM](https://arxiv.org/abs/2010.02502) paper. Only applies
                to the [`~schedulers.DDIMScheduler`], and is ignored in other schedulers.
            generator (`torch.Generator` or `List[torch.Generator]`, *optional*):
                A [`torch.Generator`](https://pytorch.org/docs/stable/generated/torch.Generator.html) to make
                generation deterministic.
            latents (`torch.FloatTensor`, *optional*):
                Pre-generated noisy latents sampled from a Gaussian distribution, to be used as inputs for image
                generation. Can be used to tweak the same generation with different prompts. If not provided, a latents
                tensor is generated by sampling using the supplied random `generator`.
            prompt_embeds (`torch.FloatTensor`, *optional*):
                Pre-generated text embeddings. Can be used to easily tweak text inputs (prompt weighting). If not
                provided, text embeddings are generated from the `prompt` input argument.
            negative_prompt_embeds (`torch.FloatTensor`, *optional*):
                Pre-generated negative text embeddings. Can be used to easily tweak text inputs (prompt weighting). If
                not provided, `negative_prompt_embeds` are generated from the `negative_prompt` input argument.
            pooled_prompt_embeds (`torch.FloatTensor`, *optional*):
                Pre-generated pooled text embeddings. Can be used to easily tweak text inputs (prompt weighting). If
                not provided, pooled text embeddings are generated from `prompt` input argument.
            negative_pooled_prompt_embeds (`torch.FloatTensor`, *optional*):
                Pre-generated negative pooled text embeddings. Can be used to easily tweak text inputs (prompt
                weighting). If not provided, pooled `negative_prompt_embeds` are generated from `negative_prompt` input
                argument.
            image_embeds (`torch.FloatTensor`, *optional*):
                Pre-generated image embeddings.
            output_type (`str`, *optional*, defaults to `"pil"`):
                The output format of the generated image. Choose between `PIL.Image` or `np.array`.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`~pipelines.stable_diffusion.StableDiffusionPipelineOutput`] instead of a
                plain tuple.
            cross_attention_kwargs (`dict`, *optional*):
                A kwargs dictionary that if specified is passed along to the [`AttentionProcessor`] as defined in
                [`self.processor`](https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/attention_processor.py).
            controlnet_conditioning_scale (`float` or `List[float]`, *optional*, defaults to 1.0):
                The outputs of the ControlNet are multiplied by `controlnet_conditioning_scale` before they are added
                to the residual in the original `unet`. If multiple ControlNets are specified in `init`, you can set
                the corresponding scale as a list.
            guess_mode (`bool`, *optional*, defaults to `False`):
                The ControlNet encoder tries to recognize the content of the input image even if you remove all
                prompts. A `guidance_scale` value between 3.0 and 5.0 is recommended.
            control_guidance_start (`float` or `List[float]`, *optional*, defaults to 0.0):
                The percentage of total steps at which the ControlNet starts applying.
            control_guidance_end (`float` or `List[float]`, *optional*, defaults to 1.0):
                The percentage of total steps at which the ControlNet stops applying.
            original_size (`Tuple[int]`, *optional*, defaults to (1024, 1024)):
                If `original_size` is not the same as `target_size` the image will appear to be down- or upsampled.
                `original_size` defaults to `(height, width)` if not specified. Part of SDXL's micro-conditioning as
                explained in section 2.2 of
                [https://huggingface.co/papers/2307.01952](https://huggingface.co/papers/2307.01952).
            crops_coords_top_left (`Tuple[int]`, *optional*, defaults to (0, 0)):
                `crops_coords_top_left` can be used to generate an image that appears to be "cropped" from the position
                `crops_coords_top_left` downwards. Favorable, well-centered images are usually achieved by setting
                `crops_coords_top_left` to (0, 0). Part of SDXL's micro-conditioning as explained in section 2.2 of
                [https://huggingface.co/papers/2307.01952](https://huggingface.co/papers/2307.01952).
            target_size (`Tuple[int]`, *optional*, defaults to (1024, 1024)):
                For most cases, `target_size` should be set to the desired height and width of the generated image. If
                not specified it will default to `(height, width)`. Part of SDXL's micro-conditioning as explained in
                section 2.2 of [https://huggingface.co/papers/2307.01952](https://huggingface.co/papers/2307.01952).
            negative_original_size (`Tuple[int]`, *optional*, defaults to (1024, 1024)):
                To negatively condition the generation process based on a specific image resolution. Part of SDXL's
                micro-conditioning as explained in section 2.2 of
                [https://huggingface.co/papers/2307.01952](https://huggingface.co/papers/2307.01952). For more
                information, refer to this issue thread: https://github.com/huggingface/diffusers/issues/4208.
            negative_crops_coords_top_left (`Tuple[int]`, *optional*, defaults to (0, 0)):
                To negatively condition the generation process based on a specific crop coordinates. Part of SDXL's
                micro-conditioning as explained in section 2.2 of
                [https://huggingface.co/papers/2307.01952](https://huggingface.co/papers/2307.01952). For more
                information, refer to this issue thread: https://github.com/huggingface/diffusers/issues/4208.
            negative_target_size (`Tuple[int]`, *optional*, defaults to (1024, 1024)):
                To negatively condition the generation process based on a target image resolution. It should be as same
                as the `target_size` for most cases. Part of SDXL's micro-conditioning as explained in section 2.2 of
                [https://huggingface.co/papers/2307.01952](https://huggingface.co/papers/2307.01952). For more
                information, refer to this issue thread: https://github.com/huggingface/diffusers/issues/4208.
            clip_skip (`int`, *optional*):
                Number of layers to be skipped from CLIP while computing the prompt embeddings. A value of 1 means that
                the output of the pre-final layer will be used for computing the prompt embeddings.
            callback_on_step_end (`Callable`, *optional*):
                A function that calls at the end of each denoising steps during the inference. The function is called
                with the following arguments: `callback_on_step_end(self: DiffusionPipeline, step: int, timestep: int,
                callback_kwargs: Dict)`. `callback_kwargs` will include a list of all tensors as specified by
                `callback_on_step_end_tensor_inputs`.
            callback_on_step_end_tensor_inputs (`List`, *optional*):
                The list of tensor inputs for the `callback_on_step_end` function. The tensors specified in the list
                will be passed as `callback_kwargs` argument. You will only be able to include variables listed in the
                `._callback_tensor_inputs` attribute of your pipeine class.

        Examples:

        Returns:
            [`~pipelines.stable_diffusion.StableDiffusionPipelineOutput`] or `tuple`:
                If `return_dict` is `True`, [`~pipelines.stable_diffusion.StableDiffusionPipelineOutput`] is returned,
                otherwise a `tuple` is returned containing the output images.
        """

        callback = kwargs.pop("callback", None)
        callback_steps = kwargs.pop("callback_steps", None)

        if callback is not None:
            deprecate(
                "callback",
                "1.0.0",
                "Passing `callback` as an input argument to `__call__` is deprecated, consider using `callback_on_step_end`",
            )
        if callback_steps is not None:
            deprecate(
                "callback_steps",
                "1.0.0",
                "Passing `callback_steps` as an input argument to `__call__` is deprecated, consider using `callback_on_step_end`",
            )

        controlnet = self.controlnet._orig_mod if is_compiled_module(self.controlnet) else self.controlnet

        # align format for control guidance
        if not isinstance(control_guidance_start, list) and isinstance(control_guidance_end, list):
            control_guidance_start = len(control_guidance_end) * [control_guidance_start]
        elif not isinstance(control_guidance_end, list) and isinstance(control_guidance_start, list):
            control_guidance_end = len(control_guidance_start) * [control_guidance_end]
        elif not isinstance(control_guidance_start, list) and not isinstance(control_guidance_end, list):
            mult = len(controlnet.nets) if isinstance(controlnet, MultiControlNetModel) else 1
            control_guidance_start, control_guidance_end = (
                mult * [control_guidance_start],
                mult * [control_guidance_end],
            )
   
        self._guidance_scale = guidance_scale
        self._clip_skip = clip_skip
        self._cross_attention_kwargs = cross_attention_kwargs

        # 2. Define call parameters
        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        device = self.device_infer

        if isinstance(controlnet, MultiControlNetModel) and isinstance(controlnet_conditioning_scale, float):
            controlnet_conditioning_scale = [controlnet_conditioning_scale] * len(controlnet.nets)



        # 3.1 Encode input prompt

        text_embeddings2, pooled_prompt_embeds = self.encode_prompt(prompt, negative_prompt,
                encoder='clip2', pooled_outputs=True, output_hidden_states=True)

        text_embeddings = self.encode_prompt(prompt, negative_prompt, output_hidden_states=True)

        prompt_embeds = torch.cat([text_embeddings, text_embeddings2], dim=-1)
        prompt_embeds = prompt_embeds.to(dtype=torch.float16)

        # 3.2 Encode image prompt
        prompt_image_emb = self._encode_prompt_image_emb(image_embeds, 
                                                         device,
                                                         num_images_per_prompt,
                                                         self.unet.dtype,
                                                         self.do_classifier_free_guidance)
        
        # 4. Prepare image
        images = []
        for image_ in image:
            image_ = self.prepare_image(
                image=image_,
                width=width,
                height=height,
                batch_size=batch_size * num_images_per_prompt,
                num_images_per_prompt=num_images_per_prompt,
                device=device,
                dtype=controlnet.dtype,
                do_classifier_free_guidance=self.do_classifier_free_guidance,
                guess_mode=guess_mode,
            )

            images.append(image_)

        image = images
        height, width = image[0].shape[-2:]


        # 4.1 Region control
        if control_mask is not None:
            mask_weight_image = control_mask
            mask_weight_image = np.array(mask_weight_image)
            mask_weight_image_tensor = torch.from_numpy(mask_weight_image).to(device=device, dtype=prompt_embeds.dtype)
            mask_weight_image_tensor = mask_weight_image_tensor[:, :, 0] / 255.
            mask_weight_image_tensor = mask_weight_image_tensor[None, None]
            h, w = mask_weight_image_tensor.shape[-2:]
            control_mask_wight_image_list = []
            for scale in [8, 8, 8, 16, 16, 16, 32, 32, 32]:
                scale_mask_weight_image_tensor = F.interpolate(
                    mask_weight_image_tensor,(h // scale, w // scale), mode='bilinear')
                control_mask_wight_image_list.append(scale_mask_weight_image_tensor)
            region_mask = torch.from_numpy(np.array(control_mask)[:, :, 0]).to(self.device_infer, dtype=self.unet.dtype) / 255.
            region_control.prompt_image_conditioning = [dict(region_mask=region_mask)]
        else:
            control_mask_wight_image_list = None
            region_control.prompt_image_conditioning = [dict(region_mask=None)]

        # 5. Prepare timesteps
        self.scheduler.set_timesteps(num_inference_steps, device=device)
        timesteps = self.scheduler.timesteps
        self._num_timesteps = len(timesteps)

        # 6. Prepare latent variables
        num_channels_latents = self.unet.config.in_channels
        latents = self.prepare_latents(
            batch_size * num_images_per_prompt,
            num_channels_latents,
            height,
            width,
            prompt_embeds.dtype,
            device,
            generator,
            latents,
        )

        # 6.5 Optionally get Guidance Scale Embedding
        timestep_cond = None
        if self.unet.config.time_cond_proj_dim is not None:
            guidance_scale_tensor = torch.tensor(self.guidance_scale - 1).repeat(batch_size * num_images_per_prompt)
            timestep_cond = self.get_guidance_scale_embedding(
                guidance_scale_tensor, embedding_dim=self.unet.config.time_cond_proj_dim
            ).to(device=device, dtype=latents.dtype)

        # 7. Prepare extra step kwargs. TODO: Logic should ideally just be moved out of the pipeline
        extra_step_kwargs = self.prepare_extra_step_kwargs(generator, eta)

        # 7.1 Create tensor stating which controlnets to keep
        controlnet_keep = []
        for i in range(len(timesteps)):
            keeps = [
                1.0 - float(i / len(timesteps) < s or (i + 1) / len(timesteps) > e)
                for s, e in zip(control_guidance_start, control_guidance_end)
            ]
            controlnet_keep.append(keeps[0] if isinstance(controlnet, ControlNetModel) else keeps)

        # 7.2 Prepare added time ids & embeddings
        if isinstance(image, list):
            original_size = original_size or image[0].shape[-2:]
        else:
            original_size = original_size or image.shape[-2:]
        target_size = target_size or (height, width)

        add_text_embeds = pooled_prompt_embeds
        if self.text_encoder_2 is None:
            text_encoder_projection_dim = int(pooled_prompt_embeds.shape[-1])
        else:
            text_encoder_projection_dim = self.text_encoder_2.config.projection_dim

        add_time_ids = self._get_add_time_ids(
            original_size,
            crops_coords_top_left,
            target_size,
            dtype=prompt_embeds.dtype,
            text_encoder_projection_dim=text_encoder_projection_dim,
        )

        if negative_original_size is not None and negative_target_size is not None:
            negative_add_time_ids = self._get_add_time_ids(
                negative_original_size,
                negative_crops_coords_top_left,
                negative_target_size,
                dtype=prompt_embeds.dtype,
                text_encoder_projection_dim=text_encoder_projection_dim,
            )
        else:
            negative_add_time_ids = add_time_ids

        if self.do_classifier_free_guidance:
            add_time_ids = torch.cat([negative_add_time_ids, add_time_ids], dim=0)

        prompt_embeds = prompt_embeds.to(device)
        add_text_embeds = add_text_embeds.to(device)
        add_time_ids = add_time_ids.to(device).repeat(batch_size * num_images_per_prompt, 1)
        encoder_hidden_states = torch.cat([prompt_embeds, prompt_image_emb], dim=1)

        # 8. Denoising loop
        num_warmup_steps = len(timesteps) - num_inference_steps * self.scheduler.order
     
        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):

                # expand the latents if we are doing classifier free guidance
                latent_model_input = torch.cat([latents] * 2) if self.do_classifier_free_guidance else latents
                latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)

                added_cond_kwargs = {"text_embeds": add_text_embeds, "time_ids": add_time_ids}

                # controlnet(s) inference
                if guess_mode and self.do_classifier_free_guidance:
                    # Infer ControlNet only for the conditional batch.
                    control_model_input = latents
                    control_model_input = self.scheduler.scale_model_input(control_model_input, t)
                    controlnet_prompt_embeds = prompt_embeds.chunk(2)[1]
                    controlnet_added_cond_kwargs = {
                        "text_embeds": add_text_embeds.chunk(2)[1],
                        "time_ids": add_time_ids.chunk(2)[1],
                    }
                else:
                    control_model_input = latent_model_input
                    controlnet_prompt_embeds = prompt_embeds
                    controlnet_added_cond_kwargs = added_cond_kwargs
                
                if isinstance(controlnet_keep[i], list):
                    cond_scale = [c * s for c, s in zip(controlnet_conditioning_scale, controlnet_keep[i])]
                else:
                    controlnet_cond_scale = controlnet_conditioning_scale
                    if isinstance(controlnet_cond_scale, list):
                        controlnet_cond_scale = controlnet_cond_scale[0]
                    cond_scale = controlnet_cond_scale * controlnet_keep[i]

                if 'multicontrolnet' in self.args.stages:
                        down_block_res_samples_list, mid_block_res_sample_list = [], []
                        for control_index in range(self.controlnet_num):
                            if control_index == 0:
                                # assume fhe first controlnet is IdentityNet
                                controlnet_prompt_embeds = prompt_image_emb
                            else:
                                controlnet_prompt_embeds = prompt_embeds

                            cond_scale_t = torch.tensor([cond_scale[control_index]], dtype=torch.float16, device=self.device_infer)
                            params_controlnet = {"control_model_input": control_model_input, "timestep": t, "encoder_hidden_states": controlnet_prompt_embeds, 'conditioning_scale':cond_scale_t}
                            params_controlnet.update({'time_ids': controlnet_added_cond_kwargs['time_ids']})
                            params_controlnet.update({'text_embeds': controlnet_added_cond_kwargs['text_embeds']})
                            params_controlnet.update({'controlnet_cond': image[control_index]})
                            model_name = 'controlnet' + str(control_index)
                            controlnet_pred = self.runEngine(model_name, params_controlnet)
                            down_block_res_samples0 = controlnet_pred['down_block_additional_residuals_0']
                            down_block_res_samples1 = controlnet_pred['down_block_additional_residuals_1']
                            down_block_res_samples2 = controlnet_pred['down_block_additional_residuals_2']
                            down_block_res_samples3 = controlnet_pred['down_block_additional_residuals_3']
                            down_block_res_samples4 = controlnet_pred['down_block_additional_residuals_4']
                            down_block_res_samples5 = controlnet_pred['down_block_additional_residuals_5']
                            down_block_res_samples6 = controlnet_pred['down_block_additional_residuals_6']
                            down_block_res_samples7 = controlnet_pred['down_block_additional_residuals_7']
                            down_block_res_samples8 = controlnet_pred['down_block_additional_residuals_8']
                            mid_block_res_sample = controlnet_pred['mid_block_additional_residual']
                            down_block_res_samples = [down_block_res_samples0, down_block_res_samples1, down_block_res_samples2,down_block_res_samples3, \
                                down_block_res_samples4, down_block_res_samples5, down_block_res_samples6, down_block_res_samples7, down_block_res_samples8]
                                    
                            down_block_res_samples_list.append(down_block_res_samples)
                            mid_block_res_sample_list.append(mid_block_res_sample)  
                        
                        
                        mid_block_res_sample = torch.stack(mid_block_res_sample_list).sum(dim=0)
                        down_block_res_samples = [torch.stack(down_block_res_samples).sum(dim=0) for down_block_res_samples in
                                                zip(*down_block_res_samples_list)]
                else:
                    if isinstance(self.controlnet, MultiControlNetModel):
                        down_block_res_samples_list, mid_block_res_sample_list = [], []
                        for control_index in range(len(self.controlnet.nets)):
                            controlnet = self.controlnet.nets[control_index]
                            if control_index == 0:
                                # assume fhe first controlnet is IdentityNet
                                controlnet_prompt_embeds = prompt_image_emb
                            else:
                                controlnet_prompt_embeds = prompt_embeds
                                
                            down_block_res_samples, mid_block_res_sample = controlnet(control_model_input,
                                                                                    t,
                                                                                    encoder_hidden_states=controlnet_prompt_embeds,
                                                                                    controlnet_cond=image[control_index],
                                                                                    conditioning_scale=cond_scale[control_index],
                                                                                    guess_mode=guess_mode,
                                                                                    added_cond_kwargs=controlnet_added_cond_kwargs,
                                                                                    return_dict=False)

                            
                            # controlnet mask
                            if control_index == 0 and control_mask_wight_image_list is not None:
                                down_block_res_samples = [
                                    down_block_res_sample * mask_weight
                                    for down_block_res_sample, mask_weight in zip(down_block_res_samples, control_mask_wight_image_list)
                                ]
                                mid_block_res_sample *= control_mask_wight_image_list[-1]

                            down_block_res_samples_list.append(down_block_res_samples)
                            mid_block_res_sample_list.append(mid_block_res_sample)

                        mid_block_res_sample = torch.stack(mid_block_res_sample_list).sum(dim=0)
                        down_block_res_samples = [torch.stack(down_block_res_samples).sum(dim=0) for down_block_res_samples in
                                                zip(*down_block_res_samples_list)]
                    else:
                        down_block_res_samples, mid_block_res_sample = self.controlnet(
                            control_model_input,
                            t,
                            encoder_hidden_states=prompt_image_emb,
                            controlnet_cond=image,
                            conditioning_scale=cond_scale,
                            guess_mode=guess_mode,
                            added_cond_kwargs=controlnet_added_cond_kwargs,
                            return_dict=False,
                        )

                        # controlnet mask
                        if control_mask_wight_image_list is not None:
                            down_block_res_samples = [
                                down_block_res_sample * mask_weight
                                for down_block_res_sample, mask_weight in zip(down_block_res_samples, control_mask_wight_image_list)
                            ]
                            mid_block_res_sample *= control_mask_wight_image_list[-1]

                if guess_mode and self.do_classifier_free_guidance:
                    # Infered ControlNet only for the conditional batch.
                    # To apply the output of ControlNet to both the unconditional and conditional batches,
                    # add 0 to the unconditional batch to keep it unchanged.
                    down_block_res_samples = [torch.cat([torch.zeros_like(d), d]) for d in down_block_res_samples]
                    mid_block_res_sample = torch.cat([torch.zeros_like(mid_block_res_sample), mid_block_res_sample])

                # predict the noise residual
                if 'unet' in self.args.stages:
                    params = {"sample": latent_model_input, "timestep": t, "encoder_hidden_states": encoder_hidden_states}
                    params.update({'time_ids': added_cond_kwargs['time_ids']})
                    params.update({'text_embeds': added_cond_kwargs['text_embeds']})
                    params.update({'down_block_additional_residuals_0': down_block_res_samples[0]})
                    params.update({'down_block_additional_residuals_1': down_block_res_samples[1]})
                    params.update({'down_block_additional_residuals_2': down_block_res_samples[2]})
                    params.update({'down_block_additional_residuals_3': down_block_res_samples[3]})
                    params.update({'down_block_additional_residuals_4': down_block_res_samples[4]})
                    params.update({'down_block_additional_residuals_5': down_block_res_samples[5]})
                    params.update({'down_block_additional_residuals_6': down_block_res_samples[6]})
                    params.update({'down_block_additional_residuals_7': down_block_res_samples[7]})
                    params.update({'down_block_additional_residuals_8': down_block_res_samples[8]})
                    params.update({'mid_block_additional_residual': mid_block_res_sample}),
    
                    noise_pred = self.runEngine('unet', params)['out_sample']  #['latent']
                    
                else:
                    noise_pred = self.unet(
                            latent_model_input,
                            t,
                            encoder_hidden_states=encoder_hidden_states,
                            timestep_cond=timestep_cond,
                            cross_attention_kwargs=self.cross_attention_kwargs,
                            down_block_additional_residuals=down_block_res_samples,
                            mid_block_additional_residual=mid_block_res_sample,
                            added_cond_kwargs=added_cond_kwargs,
                            return_dict=False,
                        )[0]       
                          
                # perform guidance
                if self.do_classifier_free_guidance:
                    noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                    noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

                # compute the previous noisy sample x_t -> x_t-1
                latents = self.scheduler.step(noise_pred, t, latents, **extra_step_kwargs, return_dict=False)[0]

                if callback_on_step_end is not None:
                    callback_kwargs = {}
                    for k in callback_on_step_end_tensor_inputs:
                        callback_kwargs[k] = locals()[k]
                    callback_outputs = callback_on_step_end(self, i, t, callback_kwargs)

                    latents = callback_outputs.pop("latents", latents)
                    prompt_embeds = callback_outputs.pop("prompt_embeds", prompt_embeds)
                    negative_prompt_embeds = callback_outputs.pop("negative_prompt_embeds", negative_prompt_embeds)

                # call the callback, if provided
                if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0):
                    progress_bar.update()
                    if callback is not None and i % callback_steps == 0:
                        step_idx = i // getattr(self.scheduler, "order", 1)
                        callback(step_idx, t, latents)
        
        torch.cuda.empty_cache()
        if not output_type == "latent":
            # make sure the VAE is in float32 mode, as it overflows in float16
            needs_upcasting = self.vae.dtype == torch.float16 and self.vae.config.force_upcast

            if needs_upcasting:
                self.upcast_vae()
                latents = latents.to(next(iter(self.vae.post_quant_conv.parameters())).dtype)

            # unscale/denormalize the latents
            # denormalize with the mean and std if available and not None
            has_latents_mean = hasattr(self.vae.config, "latents_mean") and self.vae.config.latents_mean is not None
            has_latents_std = hasattr(self.vae.config, "latents_std") and self.vae.config.latents_std is not None
            if has_latents_mean and has_latents_std:
                latents_mean = (
                    torch.tensor(self.vae.config.latents_mean).view(1, 4, 1, 1).to(latents.device, latents.dtype)
                )
                latents_std = (
                    torch.tensor(self.vae.config.latents_std).view(1, 4, 1, 1).to(latents.device, latents.dtype)
                )
                latents = latents * latents_std / self.vae.config.scaling_factor + latents_mean
            else:
                latents = latents / self.vae.config.scaling_factor

            image = self.vae.decode(latents, return_dict=False)[0]

            # cast back to fp16 if needed
            if needs_upcasting:
                self.vae.to(dtype=torch.float16)
        else:
            image = latents

        if not output_type == "latent":
            # apply watermark if available
            if self.watermark is not None:
                image = self.watermark.apply_watermark(image)

            image = self.image_processor.postprocess(image, output_type=output_type)

        # Offload all models
        self.maybe_free_model_hooks()

        if not return_dict:
            return (image,)

        return StableDiffusionXLPipelineOutput(images=image)
