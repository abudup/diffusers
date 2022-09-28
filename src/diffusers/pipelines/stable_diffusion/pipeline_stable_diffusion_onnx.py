import datetime
import inspect
import json
from typing import List, Optional, Union

import numpy as np

from transformers import CLIPFeatureExtractor, CLIPTokenizer

from ...onnx_utils import OnnxRuntimeModel
from ...pipeline_utils import DiffusionPipeline
from ...schedulers import DDIMScheduler, LMSDiscreteScheduler, PNDMScheduler
from . import StableDiffusionPipelineOutput


class StableDiffusionOnnxPipeline(DiffusionPipeline):
    vae_decoder: OnnxRuntimeModel
    text_encoder: OnnxRuntimeModel
    tokenizer: CLIPTokenizer
    unet: OnnxRuntimeModel
    scheduler: Union[DDIMScheduler, PNDMScheduler, LMSDiscreteScheduler]
    safety_checker: OnnxRuntimeModel
    feature_extractor: CLIPFeatureExtractor

    def __init__(
        self,
        vae_decoder: OnnxRuntimeModel,
        text_encoder: OnnxRuntimeModel,
        tokenizer: CLIPTokenizer,
        unet: OnnxRuntimeModel,
        scheduler: Union[DDIMScheduler, PNDMScheduler, LMSDiscreteScheduler],
        safety_checker: OnnxRuntimeModel,
        feature_extractor: CLIPFeatureExtractor,
    ):
        super().__init__()
        scheduler = scheduler.set_format("np")
        self.register_modules(
            vae_decoder=vae_decoder,
            text_encoder=text_encoder,
            tokenizer=tokenizer,
            unet=unet,
            scheduler=scheduler,
            safety_checker=safety_checker,
            feature_extractor=feature_extractor,
        )

    def __call__(
        self,
        prompt: Union[str, List[str]],
        height: Optional[int] = 512,
        width: Optional[int] = 512,
        num_inference_steps: Optional[int] = 50,
        guidance_scale: Optional[float] = 7.5,
        eta: Optional[float] = 0.0,
        latents: Optional[np.ndarray] = None,
        output_type: Optional[str] = "pil",
        return_dict: bool = True,
        **kwargs,
    ):
        if isinstance(prompt, str):
            batch_size = 1
        elif isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            raise ValueError(f"`prompt` has to be of type `str` or `list` but is {type(prompt)}")

        if height % 8 != 0 or width % 8 != 0:
            raise ValueError(f"`height` and `width` have to be divisible by 8 but are {height} and {width}.")

        # get prompt text embeddings
        text_input = self.tokenizer(
            prompt,
            padding="max_length",
            max_length=self.tokenizer.model_max_length,
            truncation=True,
            return_tensors="np",
        )
        text_embeddings = self.text_encoder(input_ids=text_input.input_ids.astype(np.int32))[0]

        # here `guidance_scale` is defined analog to the guidance weight `w` of equation (2)
        # of the Imagen paper: https://arxiv.org/pdf/2205.11487.pdf . `guidance_scale = 1`
        # corresponds to doing no classifier free guidance.
        do_classifier_free_guidance = guidance_scale > 1.0
        # get unconditional embeddings for classifier free guidance
        if do_classifier_free_guidance:
            max_length = text_input.input_ids.shape[-1]
            uncond_input = self.tokenizer(
                [""] * batch_size, padding="max_length", max_length=max_length, return_tensors="np"
            )
            uncond_embeddings = self.text_encoder(input_ids=uncond_input.input_ids.astype(np.int32))[0]

            # For classifier free guidance, we need to do two forward passes.
            # Here we concatenate the unconditional and text embeddings into a single batch
            # to avoid doing two forward passes
            text_embeddings = np.concatenate([uncond_embeddings, text_embeddings])

        # get the initial random noise unless the user supplied it
        latents_shape = (batch_size, 4, height // 8, width // 8)
        if latents is None:
            latents = np.random.randn(*latents_shape).astype(np.float32)
        elif latents.shape != latents_shape:
            raise ValueError(f"Unexpected latents shape, got {latents.shape}, expected {latents_shape}")

        # set timesteps
        accepts_offset = "offset" in set(inspect.signature(self.scheduler.set_timesteps).parameters.keys())
        extra_set_kwargs = {}
        if accepts_offset:
            extra_set_kwargs["offset"] = 1

        self.scheduler.set_timesteps(num_inference_steps, **extra_set_kwargs)

        # if we use LMSDiscreteScheduler, let's make sure latents are mulitplied by sigmas
        if isinstance(self.scheduler, LMSDiscreteScheduler):
            latents = latents * self.scheduler.sigmas[0]

        # prepare extra kwargs for the scheduler step, since not all schedulers have the same signature
        # eta (η) is only used with the DDIMScheduler, it will be ignored for other schedulers.
        # eta corresponds to η in DDIM paper: https://arxiv.org/abs/2010.02502
        # and should be between [0, 1]
        accepts_eta = "eta" in set(inspect.signature(self.scheduler.step).parameters.keys())
        extra_step_kwargs = {}
        if accepts_eta:
            extra_step_kwargs["eta"] = eta

        for i, t in enumerate(self.progress_bar(self.scheduler.timesteps)):
            # expand the latents if we are doing classifier free guidance
            latent_model_input = np.concatenate([latents] * 2) if do_classifier_free_guidance else latents
            if isinstance(self.scheduler, LMSDiscreteScheduler):
                sigma = self.scheduler.sigmas[i]
                # the model input needs to be scaled to match the continuous ODE formulation in K-LMS
                latent_model_input = latent_model_input / ((sigma**2 + 1) ** 0.5)

            # predict the noise residual
            noise_pred = self.unet(
                sample=latent_model_input, timestep=np.array([t]), encoder_hidden_states=text_embeddings
            )
            noise_pred = noise_pred[0]

            # perform guidance
            if do_classifier_free_guidance:
                noise_pred_uncond, noise_pred_text = np.split(noise_pred, 2)
                noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

            # compute the previous noisy sample x_t -> x_t-1
            if isinstance(self.scheduler, LMSDiscreteScheduler):
                latents = self.scheduler.step(noise_pred, i, latents, **extra_step_kwargs).prev_sample
            else:
                latents = self.scheduler.step(noise_pred, t, latents, **extra_step_kwargs).prev_sample

        # scale and decode the image latents with vae
        latents = 1 / 0.18215 * latents
        image = self.vae_decoder(latent_sample=latents)[0]

        image = np.clip(image / 2 + 0.5, 0, 1)
        image = image.transpose((0, 2, 3, 1))

        # run safety checker
        safety_checker_input = self.feature_extractor(self.numpy_to_pil(image), return_tensors="np")
        image, has_nsfw_concept = self.safety_checker(clip_input=safety_checker_input.pixel_values, images=image)

        if output_type == "pil":
            image = self.numpy_to_pil(image)

        if not return_dict:
            return (image, has_nsfw_concept)

        return StableDiffusionPipelineOutput(images=image, nsfw_content_detected=has_nsfw_concept)

    def end_profiling(self):
        vae_decoder_profiling_start_ts = self.vae_decoder.get_profiling_start_time_ns() / 1000
        text_encoder_profiling_start_ts = self.text_encoder.get_profiling_start_time_ns() / 1000
        unet_profiling_start_ts = self.unet.get_profiling_start_time_ns() / 1000
        safety_checker_profiling_start_ts = self.safety_checker.get_profiling_start_time_ns() / 1000

        min_start_time_ns = min(vae_decoder_profiling_start_ts, 
                                text_encoder_profiling_start_ts,
                                unet_profiling_start_ts,
                                safety_checker_profiling_start_ts)

        vae_decoder_profile_file = self.vae_decoder.end_profiling()
        text_encoder_profile_file = self.text_encoder.end_profiling()
        unet_profile_file = self.unet.end_profiling()
        safety_checker_profile_file = self.safety_checker.end_profiling()

        def load_and_sort_events(file_name, time_offset):
            with open(file_name) as f:
                events = sorted(json.load(f), key=lambda d: d['ts'])
            for event in events:
                event['ts'] -= time_offset
            return events
            
        vae_decoder_events = load_and_sort_events(vae_decoder_profile_file)
        text_encoder_events = load_and_sort_events(text_encoder_profile_file)
        unet_events = load_and_sort_events(unet_profile_file)
        safety_checker_events = load_and_sort_events(safety_checker_profile_file)

        event_lists = [vae_decoder_events, text_encoder_events, unet_events, safety_checker_events]
        merged_event_list = []

        def pop_next_event(event_lists: List[List[dict]]):
            min_value = None
            min_index = -1

            for idx, event_list in enumerate(event_lists):
                if len(event_list) == 0:
                    continue
                if min_value is None or min_value > event_list[0]['ts']:
                    min_value = event_list[0]['ts']
                    min_index = idx
            return event_lists[min_index].pop()

        while any(len(l) > 0 for l in event_lists):
            merged_event_list.append(pop_next_event(event_lists))

        merged_event_list = [json.dumps(x) for x in merged_event_list]

        # write out the merged event list
        suffix = datetime.datetime.now().strftime("%Y%m%d_%H%M%S.%f")
        with open(f'sd_onnx_profile_{suffix}.json', 'w') as f:
            json.dump(merged_event_list, f)
