import gc
import importlib
from typing import List
from PIL import Image
import logging
import threading
from src.genai.ImageGenerationParameters import Image2ImageParameters
from src.utils.singleton import singleton
import src.utils.config as config
import torch
from diffusers.pipelines import StableDiffusionInpaintPipeline, StableDiffusionXLInpaintPipeline
import transformers

# Set up module logger
logger = logging.getLogger(__name__)


class ImageGenerationException(Exception):
    """Exception which can be raised if the image generation process failed."""

    def __init__(self, message, cause=None):
        super().__init__(message)
        self.message = message
        self.cause = cause


@singleton
class SDInpaint():
    def __init__(self, default_model: str = config.get_model(), max_size: int = 1024):
        logger.info("Initialize ConvertImage2ImageByStyle")
        self.transformers_pipeline = transformers.pipeline

        if "xl" in default_model.lower():
            self.inpaint_pipeline = StableDiffusionXLInpaintPipeline
        else:
            self.inpaint_pipeline = StableDiffusionInpaintPipeline

        self.max_image_size = max_size
        self.default_model = default_model
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self._cached_img2img_pipeline = None
        self.generation_lock = threading.Lock()

    def __del__(self):
        logger.info("free memory used for pipeline")
        self.unload_img2img_pipeline()

    def _load_model(self, model=config.get_model(), use_cached_model=True):
        """Load and return the Stable Diffusion model to generate images"""
        if self._cached_img2img_pipeline and use_cached_model:
            return self._cached_img2img_pipeline

        try:
            logger.debug(f"Loading model {model}")

            pipeline = None
            # self.inpaint_pipeline = StableDiffusionInpaintPipeline.from_pretrained("stable-diffusion-v1-5/stable-diffusion-inpainting")

            if model.endswith("safetensors"):
                logger.info(f"Using 'from_single_file' to load model {model} from local folder")
                pipeline = self.inpaint_pipeline.from_single_file(
                    model,
                    torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                    cache_dir=config.get_model_folder(),
                    safety_checker=None,
                    requires_safety_checker=False,
                    use_safetensors=True
                )
            else:
                logger.info(f"Using 'from_pretrained' option to load model {model} from hugging face")
                pipeline = self.inpaint_pipeline.from_pretrained(
                    model,
                    cache_dir=config.get_model_folder(),
                    torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                    safety_checker=None,
                    requires_safety_checker=False
                )

            logger.debug("diffuser initiated")
            pipeline = pipeline.to(self.device)
            if self.device == "cuda":
                pipeline.enable_xformers_memory_efficient_attention()
            logger.debug("Pipeline created")
            self._cached_img2img_pipeline = pipeline
            return pipeline
        except Exception as e:
            logger.error("Pipeline could not be created. Error in load_model: %s", str(e))
            logger.debug("Exception details:", e)
            raise Exception("Error while loading the pipeline for image conversion.\nSee logfile for details.")

    def unload_img2img_pipeline(self):
        try:
            logger.info("Unload image captioner")
            if self._cached_img2img_pipeline:
                del self._cached_img2img_pipeline
                self._cached_img2img_pipeline = None
            gc.collect()
            torch.cuda.empty_cache()
        except Exception:
            logger.error("Error while unloading IMAGE_TO_IMAGE_PIPELINE")

    def change_img2img_model(self, model):
        logger.info("Reloading model %s", model)
        try:
            self.default_model = model
            self.unload_img2img_pipeline()
            self._load_model(model=model, use_cached_model=False)
        except Exception as e:
            logger.error("Error while changing text2img model: %s", str(e))
            logger.debug("Exception details: {e}")
            raise (f"Loading new model '{model}' failed", e)

    def calculate_crop(self, width, height):
        new_width = width - (width % 8)
        new_height = height - (height % 8)
        crop_x = (width - new_width) // 2
        crop_y = (height - new_height) // 2
        return new_width, new_height, crop_x, crop_y

    def generate_images(self, params: Image2ImageParameters, count: int) -> List[Image.Image]:
        """Generate a list of images."""
        """Convert the input image using AI style transfer"""
        with self.generation_lock:
            try:
                # Validate parameters
                params.validate()

                logger.debug("starting image generation")

                image = params.input_image
                logger.debug(f"Input image Size {image.width}x{image.height}")
                image.thumbnail((self.max_image_size, self.max_image_size))
                # make sure that image heigth and width can be divided by 8
                new_width = image.width - (image.width % 8)
                new_height = image.height - (image.height % 8)
                logger.debug(f"Crop Result {new_width}x{new_height}")
                image = image.crop((0, 0, new_width, new_height))

                logger.debug(f"used image Size {image.width}x{image.height}")
                # Create default mask if none provided
                mask = params.mask_image or Image.new("L", image.size, 255)
                logger.debug(f"MaskSize {mask.width}x{mask.height}")
                logger.debug("Strength: %f, Steps: %d",
                             params.strength, params.steps)

                model = self._load_model()
                if not model:
                    logger.error("No model loaded")
                    raise Exception("No model loaded. Generation not available")

                result_image = model(
                    prompt=params.prompt,
                    negative_prompt=params.negative_prompt,
                    num_inference_steps=params.steps,
                    image=[image] * count,
                    mask_image=mask,
                    width=image.width,
                    height=image.height,
                    strength=params.strength,
                    num_images_per_prompt=count,
                ).images

                return result_image

            except RuntimeError as e:
                logger.error("Error while generating Image: %s", str(e))
                self.unload_img2img_pipeline()
                raise ImageGenerationException(message=f"Error while creating the image. {e}")

    def ui_elements(self) -> dict:
        """Return a dictionary of UI elements and their required status."""
        pass
