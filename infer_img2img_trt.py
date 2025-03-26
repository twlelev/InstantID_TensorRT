import cv2
import torch
import numpy as np
from PIL import Image
import argparse
from diffusers.utils import load_image
from diffusers.models import ControlNetModel
from insightface.app import FaceAnalysis
from pipeline_stable_diffusion_xl_instantid_img2img_trt import StableDiffusionXLInstantIDImg2ImgPipeline, draw_kps
import time 


def resize_img(input_image, max_side=1280, min_side=1024, size=None, 
               pad_to_max_side=False, mode=Image.BILINEAR, base_pixel_number=64):

    w, h = input_image.size
    if size is not None:
        w_resize_new, h_resize_new = size
    else:
        ratio = min_side / min(h, w)
        w, h = round(ratio*w), round(ratio*h)
        ratio = max_side / max(h, w)
        input_image = input_image.resize([round(ratio*w), round(ratio*h)], mode)
        w_resize_new = (round(ratio * w) // base_pixel_number) * base_pixel_number
        h_resize_new = (round(ratio * h) // base_pixel_number) * base_pixel_number
    input_image = input_image.resize([w_resize_new, h_resize_new], mode)

    if pad_to_max_side:
        res = np.ones([max_side, max_side, 3], dtype=np.uint8) * 255
        offset_x = (max_side - w_resize_new) // 2
        offset_y = (max_side - h_resize_new) // 2
        res[offset_y:offset_y+h_resize_new, offset_x:offset_x+w_resize_new] = np.array(input_image)
        input_image = Image.fromarray(res)
    return input_image

def parseArgs():
    parser = argparse.ArgumentParser(description="Options for Stable Diffusion XL InstantIDTrt Demo", conflict_handler='resolve')
    parser.add_argument('--batch', type=int, default=1, help="Batch of image to generate")
    parser.add_argument('--height', type=int, default=1024, help="Height of image to generate (must be multiple of 8)")
    parser.add_argument('--width', type=int, default=1024, help="Height of image to generate (must be multiple of 8)")
    parser.add_argument('--num-inference-steps', type=int, default=30, help="Number of denoising steps")
    parser.add_argument('--num-warmup-runs', type=int, default=5, help="Number of warmup runs before benchmarking performance")
    parser.add_argument('--guidance-scale', type=float, default=5, help="Value of classifier-free guidance scale (must be greater than 1)")
    parser.add_argument('--image-strength', type=float, default=0.85, help="Strength of transformation applied to input_image (must be between 0 and 1)")
    parser.add_argument('--onnx-dir', default='./Tensorrt_Build/onnx_xl_img2img_test', help="Directory for SDXL ONNX models")
    parser.add_argument('--engine-dir', default='./Tensorrt_Build/engine_xl_img2img_test', help="Directory for SDXL TensorRT engines")
    parser.add_argument('--onnx-opset', type=int, default=19, choices=range(7,19), help="Select ONNX opset version to target for exported models")
    parser.add_argument('--stages', nargs='+', type=str, default=['unet', 'controlnet', 'clip', 'clip2'], help=" Model inference use Tensorrt")
    parser.add_argument('--device', type=str, default='cuda:0')
    #IPAdapter
    parser.add_argument('--ip-adapter-scale', type=float, default=0.8)

    return parser.parse_args()

if __name__ == "__main__":
    args = parseArgs()
    # Load face encoder
    app = FaceAnalysis(name='antelopev2', root='/', providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
    app.prepare(ctx_id=0, det_size=(640, 640))

    # Path to InstantID models
    face_adapter = f'./checkpoints/ip-adapter.bin'
    controlnet_path = f'./checkpoints/ControlNetModel'

    # Load pipeline
    controlnet = ControlNetModel.from_pretrained(controlnet_path, torch_dtype=torch.float16)

    base_model_path = './checkpoints/StableDiffusion/sd_xl_base_1.0.safetensors'

    pipe = StableDiffusionXLInstantIDImg2ImgPipeline.from_single_file(
        base_model_path,
        controlnet=controlnet,
        torch_dtype=torch.float16,
    )
    pipe.cuda(args.device)
    pipe.load_ip_adapter_instantid(face_adapter)
    
    pipe.tensorrtInit(args)
    # Infer setting
    prompt = "analog film photo of a man. faded film, desaturated, 35mm photo, grainy, vignette, vintage, Kodachrome, Lomography, stained, highly detailed, found footage, masterpiece, best quality"
    n_prompt = "(lowres, low quality, worst quality:1.2), (text:1.2), watermark, painting, drawing, illustration, glitch, deformed, mutated, cross-eyed, ugly, disfigured (lowres, low quality, worst quality:1.2), (text:1.2), watermark, painting, drawing, illustration, glitch,deformed, mutated, cross-eyed, ugly, disfigured"

    face_image = load_image("./examples/yann-lecun_resize.jpg")
    face_image = resize_img(face_image, size=(args.width, args.height))

    face_info = app.get(cv2.cvtColor(np.array(face_image), cv2.COLOR_RGB2BGR))
    face_info = sorted(face_info, key=lambda x:(x['bbox'][2]-x['bbox'][0])*(x['bbox'][3]-x['bbox'][1]))[-1] # only use the maximum face
    face_emb = face_info['embedding']
    face_kps = draw_kps(face_image, face_info['kps'])

    # num_warmup_runs
    _ = pipe(
        prompt=prompt,
        negative_prompt=n_prompt,
        image=face_image,
        image_embeds=face_emb,
        control_image=face_kps,
        controlnet_conditioning_scale=0.8,
        ip_adapter_scale=args.ip_adapter_scale,
        num_inference_steps=args.num_warmup_runs,
        guidance_scale=args.guidance_scale,
        strength=args.image_strength
    ).images[0]
    
    start_time = time.time()
    image = pipe(
        prompt=prompt,
        negative_prompt=n_prompt,
        image=face_image,
        image_embeds=face_emb,
        control_image=face_kps,
        controlnet_conditioning_scale=0.8,
        ip_adapter_scale=args.ip_adapter_scale,
        num_inference_steps=args.num_inference_steps,
        guidance_scale=args.guidance_scale,
        strength=args.image_strength
    ).images[0]
    execution_time = time.time() - start_time
    print(f"execution_time: {execution_time:.2f}s")
        
    image.save('results/iresult_infer_img2img_trt.jpg')    
    pipe.teardown()