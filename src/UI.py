import json
import os
from PIL import ImageOps, Image
import gradio as gr
from hashlib import sha1
from datetime import datetime, timedelta
import logging

import src.utils.config as config
from src.genai.ImageGenerationParameters import Image2ImageParameters
import src.utils.fileIO as utils
from src.genai import SDInpaint

# Set up module logger
logger = logging.getLogger(__name__)

if config.SKIP_ONNX or not config.is_feature_generation_with_token_enabled():
    def analyze_faces(pil_image: Image.Image):
        """ without onnx we cant detect and analyze faces"""
        return []
else:
    logger.info("Activating ONNX functions")
    from src.detectors.FaceAnalyzer import FaceAnalyzer
    _face_analyzer = None

    def analyze_faces(pil_image):
        global _face_analyzer
        if _face_analyzer == None:
            _face_analyzer = FaceAnalyzer()
        return _face_analyzer.get_gender_and_age_from_image(pil_image)

_saved_hashes = {}
_saved_hashes_path = os.path.join(config.get_output_folder(), "uploaded_images.json")
if os.path.exists(_saved_hashes_path):
    with open(_saved_hashes_path, "r") as f:
        _saved_hashes.update(json.load(f))

_AIHandler = SDInpaint(config.get_model(), max_size=config.get_max_size())
_ImageCaptioner = None#ImageCaptioner()


def action_update_all_local_models():
    """updates the list of available models in the ui"""
    return gr.update(choices=utils.get_all_local_models(config.get_model_folder()))


def action_handle_input_file(request: gr.Request, image: dict, prompt):
    """Analyze the Image, Save the input image in a cache if enabled, count token."""
    # deactivate the start button on error
    if image is None:
        return wrap_handle_input_response(False, "")

    # API Users don't have a request (by documentation)
    if not request:
        return wrap_handle_input_response(False, "")

    if isinstance(image, dict):
        # TODO: check what we get
        if len(image) > 1:
            return wrap_handle_input_response(True, prompt)

    input_file_path = ""
    image_sha1 = sha1(image.tobytes()).hexdigest()
    if config.is_input_cache_enabled():
        dir = config.get_output_folder()
        dir = os.path.join(dir, datetime.now().strftime("%Y%m%d"))
        input_file_path = utils.save_image_as_file(image, dir)

    logger.info(f"UPLOAD with ID: {image_sha1}")

    image_description = ""
    try:
        image_description = action_describe_image(image)
    except Exception as e:
        logger.error("Error creating image description: %s", str(e))
        # logger.debug("Exception details:", exc_info=True)
        gr.Warning("Could not create a proper image description. Please describe your image shortly for better results.")

    # variables used for analytics if enabled
    face_detected = False
    min_age = 0
    max_age = 0
    gender = 0

    # analyzation of images will be done always
    detected_faces = []
    try:
        detected_faces = analyze_faces(image)
    except Exception as e:
        logger.error("Error while analyzing face: %s", str(e))
        logger.debug("Exception details:", exc_info=True)

    if len(detected_faces) > 0:
        face_detected = True
        # we have minimum one face
        logger.debug("face detected")

        for face in detected_faces:
            # just save the jungest and oldest if we have multiple faces
            min_age = face["minAge"] if face["minAge"] < min_age or min_age == 0 else min_age
            max_age = face["maxAge"] if face["maxAge"] > max_age or max_age == 0 else max_age
            if face["isFemale"]:
                gender |= 2
                # until we can recognize smiles we give bonus for other properties
                # prevent misuse here ...  FIXME

    # return wrap_handle_input_response(face_detected, image_description)
    return wrap_handle_input_response(True, image_description)


def action_describe_image(image):
    """describe an image for better inpaint results."""
    if config.SKIP_AI:
        return "ai deactivated"
    # Fallback
    value = "please describe your image here"
    try:
        value = _ImageCaptioner.describe_image(image)
        logger.debug("Image description: %s", value)
    except Exception:
        pass
    return value


def action_reload_model(model):
    if config.SKIP_AI:
        return
    logger.warning("Reloading model %s", model)
    try:
        _AIHandler.change_pipeline_model(model=model)
        gr.Info(message=f"Model {model} loaded.", title="Model changed")
    except Exception as e:
        gr.Error(message=e.message)


def save_hashes():
    try:
        with open(_saved_hashes_path, "w") as f:
            json.dump(_saved_hashes, f, indent=4)
    except Exception as e:
        logger.error(f"Error while saving {_saved_hashes_path}: {e}")


def action_generate_image(request: gr.Request, image, strength, steps, image_description):
    """Convert the entire input image."""
    global _saved_hashes
    try:
        if image is None:
            gr.Error("Start of Generation without image!")
            return wrap_generate_image_response(None)
        # API Users don't have a request object (by documentation)
        if request is None:
            logger.warning("No request object. API usage?")
            return wrap_generate_image_response(None)

        strength = strength / 100  # we use values 1 - 100 in UI instead of 0.1--1
        logger.debug("Starting image generation")
        if image_description == None or image_description == "":
            image_description = _ImageCaptioner.describe_image(image)

        logger.info(f"GENERATE now")

        input_img = None or image
        mask_img = None
        # if len(image) > 1:
        #     input_img = image["background"]
        #     mask_img = image["layers"][0]
        if not input_img:
            raise Exception("no input image")
        mask_sha1 = "empty"
        if isinstance(image, dict):
            # Gradio ImageEditor-Format
            input_img = image["background"]
            if not input_img:
                raise Exception("No Input Image")
            # ImageEditor sends RGBA, which triggers The size of tensor a (680) must match the size of tensor b (85) at non-singleton dimension 3
            input_img = input_img.convert('RGB')
            # Die "layers" werden typischerweise als RGBA/PNG-Daten für die Maske verwendet
            masks = image.get("layers", None)
            if masks:
                logger.debug(f"Masks: {masks}")
                if len(masks) > 0:
                    mask_img = masks[0]
                    #mask_img = mask_img.convert('RGB')
                    mask_with_alpha = mask_img.convert('RGBA')
                    mask_img = mask_with_alpha.split()[-1].convert('RGB')
                    threshold = 128
                    mask_img = mask_img.point(lambda p: 0 if p > threshold else 255)
                    mask_img = ImageOps.invert(mask_img)

                    mask_sha1 = sha1(mask_img.tobytes()).hexdigest()
        else:
            input_img = image

        image_sha1 = sha1(input_img.tobytes()).hexdigest()

        if config.is_save_output_enabled():
            folder_path = config.get_output_folder()
            folder_path = os.path.join(folder_path, datetime.now().strftime("%Y%m%d"))

            if not _saved_hashes.get(image_sha1, False):
                _saved_hashes[image_sha1] = True
                save_hashes()
                utils.save_image_with_timestamp(
                    image=input_img,
                    folder_path=folder_path,
                    reference=f"{image_sha1}-input",
                    ignore_errors=True)

            if not _saved_hashes.get(mask_sha1, False):
                _saved_hashes[mask_sha1] = True
                save_hashes()
                utils.save_image_with_timestamp(
                    image=mask_img,
                    folder_path=folder_path,
                    reference=f"{image_sha1}-mask",
                    ignore_errors=True)

        # use always the sliders for strength and steps if they are enabled
        if not config.UI_show_strength_slider():
            strength = 0.8
        if not config.UI_show_steps_slider():
            steps = 40

        params = Image2ImageParameters(
            input_image=input_img,
            prompt=image_description,
            negative_prompt="",  # sd["negative_prompt"],
            strength=strength,
            steps=steps,
            mask_image=mask_img
        )
        result_images = _AIHandler.generate_images(params, count=1)

        # save generated file if enabled
        if config.is_save_output_enabled():
            folder_path = config.get_output_folder()
            folder_path = os.path.join(folder_path, datetime.now().strftime("%Y%m%d"))
            for img in result_images:
                utils.save_image_with_timestamp(
                    image=img,
                    folder_path=folder_path,
                    reference=f"{image_sha1}",
                    ignore_errors=True)

        return wrap_generate_image_response(result_images)
    except Exception as e:
        logger.error("RuntimeError: %s", str(e))
        logger.debug("Exception details:", exc_info=True)
        gr.Error(e)
        return wrap_generate_image_response(None)

# --------------------------------------------------------------
# Gradio - Render UI
# --------------------------------------------------------------


def wrap_handle_input_response(start_enabled: bool, image_description: str) -> list:
    """Create a consistent response format for handle_input_file action.

    Args:
        start_enabled: Whether the start button should be enabled
        image_description: The generated image description

    Returns:
        List of values in the order: [start_button, text_description, local_storage, area_description, token_counter]
    """
    return [
        gr.update(interactive=start_enabled),
        image_description,
        gr.update(visible=True)
    ]


def wrap_generate_image_response(result_image: any) -> list:
    """Create a consistent response format for generate_image action.

    Args:
        result_image: The generated image result

    Returns:
        List of values in the order: [output_image, local_storage, start_button, token_counter]
    """
    return [
        result_image,
        gr.update(interactive=True)
    ]


def create_gradio_interface():
    with gr.Blocks(
        title=config.get_app_title(),
        theme=config.UI_get_gradio_theme(),
        css="footer {visibility: hidden}",
        analytics_enabled=False
    ) as app:
        with gr.Row():
            gr.Markdown("### " + config.get_app_title() + "\n\n" + config.get_user_message())

        if config.DEBUG:
            gr.Markdown("*DEBUG enabled*")
            with gr.Row():
                with gr.Column():
                    model_dropdown = gr.Dropdown(choices=utils.get_all_local_models(
                        config.get_model_folder()), value=config.get_model(), label="Models", allow_custom_value=True)
                with gr.Column():
                    refresh_model_list_button = gr.Button("refresh model list")
                    reload_button = gr.Button("load model")
                    reload_button.click(
                        fn=action_reload_model,
                        inputs=[model_dropdown],
                        outputs=[]
                    )
                    refresh_model_list_button.click(
                        fn=action_update_all_local_models,
                        inputs=[],
                        outputs=[model_dropdown]
                    )
        with gr.Row():
            with gr.Column():
                image_input = gr.ImageEditor(label="Input", type="pil", height=512)
                # describe_button = gr.Button("Describe your Image", interactive=False)
                with gr.Column(visible=False) as area_description:
                    text_description = gr.Textbox(label="Prompt", info="change the image description for better results",
                                                  show_label=True, max_length=150, max_lines=3, submit_btn="↻")
                    strength_slider = gr.Slider(label="Prompt importance",
                                                info="Low value = close to original picture, high value = more randomized",
                                                minimum=1,
                                                maximum=100,
                                                value=config.get_default_strength() * 100,
                                                step=5,
                                                visible=config.UI_show_strength_slider())
                    steps_slider = gr.Slider(label="Steps",
                                             info="Higher value for better result, but longer wait time",
                                             minimum=10, maximum=100, value=config.get_default_steps(),
                                             step=5, visible=config.UI_show_steps_slider())

            with gr.Column():
                output_image = gr.Gallery(
                    label="Result",
                    type="pil",
                    height=512,
                    show_download_button=True
                )
                start_button = gr.Button("Start Creation", interactive=True, variant="primary")

        # Save input image immediately on change
        # adapt wrap_handle_input_response if you change output params
        image_input.change(
            fn=action_handle_input_file,
            inputs=[image_input, text_description],
            outputs=[start_button, text_description, area_description],
            concurrency_limit=None,
            concurrency_id="new_image"
        )

        text_description.submit(
            fn=action_describe_image,
            inputs=[image_input],
            outputs=[text_description],
            concurrency_limit=10,
            concurrency_id="describe"
        )

        # adapt wrap_generate_image_response if you change output parameters
        start_button.click(
            fn=lambda: gr.Button(interactive=False),
            outputs=[start_button],
        ).then(
            fn=action_generate_image,
            inputs=[image_input, strength_slider, steps_slider, text_description],
            outputs=[output_image, start_button],
            concurrency_limit=config.GenAI_get_execution_batch_size(),
            concurrency_id="gpu_queue",
            show_progress="minimal"
            # batch=False,
            # max_batch_size=config.GenAI_get_execution_batch_size(),
        ).then(
            fn=lambda: gr.Button(interactive=True),
            outputs=[start_button],
        )

        def action__activate_button():
            return gr.update(interactive=True)

        disclaimer = config.get_app_disclaimer()
        if disclaimer:
            js = f"""
            () => {{
                // Overlay erstellen
                const overlay = document.createElement('div');
                overlay.style.position = 'fixed';
                overlay.style.top = 0;
                overlay.style.left = 0;
                overlay.style.width = '100%';
                overlay.style.height = '100%';
                overlay.style.color = 'white';
                overlay.style.backgroundColor = 'rgba(10, 0, 0, 0.9)';
                overlay.style.display = 'flex';
                overlay.style.justifyContent = 'center';
                overlay.style.alignItems = 'center';
                overlay.style.zIndex = 1000;

                // Nachricht und Button hinzufügen
                overlay.innerHTML = `
                <div style="padding: 20px; border-radius: 20px; text-align: center; box-shadow: 0px 4px 10px rgba(0, 0, 255, 0.3);">
                    <p style="margin-bottom: 20px;">{disclaimer}</p>
                    <button id="accept-btn" style="padding: 10px 20px; border: none; background: #4caf50; color: white; border-radius: 5px; cursor: pointer;">Accept</button>
                </div>
                `;

                // Overlay zur Seite hinzufügen
                document.body.appendChild(overlay);

                // Button-Click-Event, um Overlay zu schließen
                document.getElementById('accept-btn').onclick = () => {{
                    document.body.removeChild(overlay);
                }};
            }}
            """

            app.load(
                fn=None,
                inputs=None,
                outputs=None,
                js=js
            )
        # end if disclaimer activated
        return app
