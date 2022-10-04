import argparse
import onnxruntime
import sys
import time
from diffusers import StableDiffusionOnnxPipeline

def _parse_arguments():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "-p",
        "--pipeline",
        required=False,
        type=str,
        default=None,
        help="Directory of saved onnx pipeline for CompVis/stable-diffusion-v1-4.",
    )

    parser.add_argument(
        '-t',
        '--prompt',
        required=False,
        type=str,
        default='A gray cat with blue eyes, in a bowtie, cubist acrylic'
        help='The prompt to use'
    )
        
    args = parser.parse_args()
    return args

def _main():
    session_options = {}
    for name in ['vae_decoder', 'text_encoder', 'unet', 'safety_checker']:
        opts = onnxruntime.SessionOptions()
        opts.enable_profiling = True
        opts.profile_file_prefix = f'onnxruntime_{name}_profile_'
        session_options[name] = opts
        
    load_start = time.time()
    pipe = StableDiffusionOnnxPipeline.from_pretrained(
        "/s/bench_models/sd_onnx",
        # "/home/abudup/work/git/benchmark_suite/payload/inference/bench_models/sd_onnx",
        provider='ROCMExecutionProvider'
        session_options=session_options,
    )
    load_end = time.time()

    inference_start = time.time()
    image = pipe(args.prompt).images[0]
    inference_end = time.time()

    prof_file = pipe.end_profiling()

    if prof_file is not None:
        print(f'Profile output saved into: {prof_file}')
        
    print(f'Inference took {inference_end - inference_start} seconds')
    image.save("onnx_sd_output.jpg")

if __name__ == '__main__':
    _main()
