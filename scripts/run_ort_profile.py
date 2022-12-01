import argparse
import os
import onnxruntime
import sys
import time
from diffusers import StableDiffusionOnnxPipeline

def _parse_arguments():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "-p",
        "--pipeline",
        required=True,
        type=str,
        default=None,
        help="Directory of saved onnx pipeline for CompVis/stable-diffusion-v1-4.",
    )

    parser.add_argument(
        '-t',
        '--prompt',
        required=False,
        type=str,
        default='A gray cat with blue eyes, in a bowtie, cubist acrylic',
        help='The prompt to use'
    )

    parser.add_argument(
        '-e',
        '--enable-profiling',
        required=False,
        type=bool,
        action='store_true',
        help='Enable ORT profiling during the run'
    )

    parser.add_argument(
        '-w',
        '--warmup',
        required=False,
        type=int,
        default=0,
        help='Number of warmup iterations to run before the timed run'
    )
        
    args = parser.parse_args()
    return args

def _main():
    args = _parse_arguments()
    session_options = {}
    for name in ['vae_decoder', 'text_encoder', 'unet', 'safety_checker']:
        opts = onnxruntime.SessionOptions()
        if args.enable_profiling:
            opts.enable_profiling = True
            opts.profile_file_prefix = f'onnxruntime_{name}_profile_'
        session_options[name] = opts
        
    load_start = time.time()
    pipe = StableDiffusionOnnxPipeline.from_pretrained(
        args.pipeline,
        provider='ROCMExecutionProvider',
        session_options=session_options,
    )
    load_end = time.time()
    print(f'Model loaded, took {load_end - load_start} seconds to load model.')

    if args.warmup != 0:
        print(f'Running {args.warmup} warmup iterations...')
        pipe(args.prompt, num_inference_steps=args.warmup)
        print(f'Done running {args.warmup} warmup iterations!')

    print('Starting timed inference run...')
    inference_start = time.time()
    image = pipe(args.prompt).images[0]
    inference_end = time.time()
    print(f'Inference took {inference_end - inference_start} seconds')
    image.save("onnx_sd_output.jpg")

    if args.enable_profiling:
        prof_file = pipe.end_profiling()

        if prof_file is not None:
            print(f'Profile output saved into: {prof_file}')
        else:
            print('Failed to generate profile files')        

if __name__ == '__main__':
    _main()
