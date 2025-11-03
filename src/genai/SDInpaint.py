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
        logger.info("Initialize")

        # if "xl" in default_model.lower():
        #     self.inpaint_pipeline = StableDiffusionXLInpaintPipeline
        # else:
        #     self.inpaint_pipeline = StableDiffusionInpaintPipeline

        self.inpaint_pipeline = StableDiffusionInpaintPipeline
        self.max_image_size = max_size
        self.default_model = default_model
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self._cached_pipeline = None
        self.generation_lock = threading.Lock()

    def __del__(self):
        logger.info("free memory used for pipeline")
        self.unload_pipeline()

    def _get_pipeline(self, model=config.get_model(), use_cached_model=True):
        """Load and return the Stable Diffusion Pipeline to generate images"""
        if self._cached_pipeline and use_cached_model:
            return self._cached_pipeline

        try:
            logger.debug(f"Loading model {model}")

            pipeline = None

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
            self._cached_pipeline = pipeline
            return pipeline
        except Exception as e:
            logger.error("Pipeline could not be created. Error in load_model: %s", str(e))
            logger.debug("Exception details:", e)
            raise Exception("Error while loading the pipeline for image conversion.\nSee logfile for details.")

    def unload_pipeline(self):
        try:
            logger.info("Unload pipeline")
            if self._cached_pipeline:
                del self._cached_pipeline
                self._cached_pipeline = None
            gc.collect()
            torch.cuda.empty_cache()
        except Exception:
            logger.error("Error while unloading pipeline")

    def change_pipeline_model(self, model):
        logger.info("Reloading model %s", model)
        try:
            self.default_model = model
            self.unload_pipeline()
            self._get_pipeline(model=model, use_cached_model=False)
        except Exception as e:
            logger.error("Error while changing model: %s", str(e))
            logger.debug("Exception details: {e}")
            raise Exception(f"Loading new model '{model}' failed", e)

    def generate_images(self, params: Image2ImageParameters, count: int) -> List[Image.Image]:
        """Generate a list of images."""
        """Convert the input image using AI style transfer"""
        """mask mus be black for parts to keep, white for parts to replace"""
        with self.generation_lock:
            try:
                # Validate parameters
                params.validate()

                logger.debug("starting image generation")

                image = params.input_image
                image = self.resize_image(image)

                # Create default mask if none provided
                mask = params.mask_image or Image.new("L", image.size, 255)
                mask = mask.convert('L')
                mask = self.resize_image(mask)

                logger.debug(f"used image Size {image.width}x{image.height}")
                logger.debug(f"used MaskSize {mask.width}x{mask.height}")
                logger.debug("Strength: %f, Steps: %d", params.strength, params.steps)
                pipeline = self._get_pipeline()
                if not pipeline:
                    logger.error("No model loaded")
                    raise Exception("No model loaded. Generation not available")
                blurred_mask = pipeline.mask_processor.blur(mask, blur_factor=33)
                result_image = pipeline(
                    prompt=params.prompt,
                    # negative_prompt=params.negative_prompt,
                    num_inference_steps=params.steps,
                    image=image,
                    mask_image=blurred_mask,
                    width=image.width,
                    height=image.height,
                    strength=params.strength,
                    num_images_per_prompt=count
                ).images
                logger.debug("images created")
                return result_image, blurred_mask

            except RuntimeError as e:
                logger.error("Error while generating Image: %s", str(e))
                self.unload_pipeline()
                raise ImageGenerationException(message=f"Error while creating the image. {e}")

    def resize_image(self, image):
        """make sure that image heigth and width can be divided by 8 as required for inpainting"""
        logger.debug(f"Input image Size {image.width}x{image.height}")
        # make sure that image heigth and width can be divided by 8
        image.thumbnail((self.max_image_size, self.max_image_size))
        logger.debug(f"Reduzed to max image Size {image.width}x{image.height}")
        new_width = image.width - (image.width % 8)
        new_height = image.height - (image.height % 8)
        logger.debug(f"Expected Crop Result {new_width}x{new_height}")
        image = image.crop((0, 0, new_width, new_height))
        logger.debug(f"Cropped fo /8 image Size {image.width}x{image.height}")
        return image
