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
from src.masks import MediaPipeMaskGenerator

# Set up module logger
logger = logging.getLogger(__name__)

if True or config.SKIP_ONNX or not config.is_feature_generation_with_token_enabled():
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
_ImageCaptioner = None  # ImageCaptioner()
_mp_mask_generator = MediaPipeMaskGenerator()
_mp_mask_generator.download_models(models_dir=os.path.join(config.get_model_folder(), "mediapipe"))


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
        # it's an image editor and we get for each paint in mask an update-->ignore
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
    else:
        gr.Warning("No face detected. Please use another image!")
    # return wrap_handle_input_response(face_detected, image_description)
    return wrap_handle_input_response(True, image_description)


def action_reload_model(model):
    if config.SKIP_AI:
        return
    logger.warning("Reloading model %s", model)
    try:
        _AIHandler.change_pipeline_model(model=model)
        gr.Info(message=f"Model {model} loaded.", title="Model changed")
    except Exception as e:
        gr.Error(message=e.message)


def save_hashes_to_file():
    try:
        with open(_saved_hashes_path, "w") as f:
            json.dump(_saved_hashes, f, indent=4)
    except Exception as e:
        logger.error(f"Error while saving {_saved_hashes_path}: {e}")


def process_inpaint_request(
    image,
    text_description,
    area,
    clothing_option,
    clothing_details1,
    clothing_details2,
    cloth_text,
    tattoos_option,
    hair_text,
    background_text
):
    """Convert the entire input image."""
    global _saved_hashes
    try:
        if image is None:
            gr.Error("You must provide an Image to continue!")
            return wrap_generate_image_response(None)

        strength = config.get_default_strength()
        steps = config.get_default_steps()

        flag_mask_skin = False
        flag_mask_cloth = False
        flag_mask_hair = False
        flag_mask_background = False
        prompt = text_description + "," if len(text_description) > 0 else ""
        if area == "Clothing":
            flag_mask_cloth = True
            if clothing_option == "Remove":
                clothing_details1 = clothing_details1.lower()
                p2 = f"fully naked, {clothing_details1} chest, {clothing_details2.lower()}"
                if "flat" in clothing_details1 or "small" in clothing_details1:
                    p2 = f"little {p2}"
                if "large" in clothing_details1:
                    p2 = p2.replace("chest", "tits")
                prompt = f"{prompt}{p2} "
            else:
                prompt = f"{prompt}{cloth_text}"

        elif area == "Tattoos":
            flag_mask_skin = True
            if tattoos_option == "Remove":
                prompt += " no tattoos"
            elif "few" in tattoos_option:
                prompt += " one small nice tattoo"
                strength = 0.6
            else:
                prompt += " intense tattoos"
        elif area == "Hair":
            flag_mask_hair = True
            prompt = hair_text
        elif area == "Background":
            flag_mask_background = True
            prompt = background_text
        else:
            # FIXME: custom mask via ImageEditor in teh Future
            flag_mask_background = True
            prompt = "flowers"

        org_input_img = None or image
        mask_img = None

        if not org_input_img:
            raise Exception("no input image")
        mask_sha1 = "empty"
        if _AIHandler.size_ok(org_input_img):
            input_img = org_input_img
        else:
            input_img = _AIHandler.resize_image(org_input_img)
        
        mask_img = _mp_mask_generator.generate_mask(input_img,
                                                    face=False,
                                                    clothes=flag_mask_cloth,
                                                    accessories=False,
                                                    body=flag_mask_skin,
                                                    hair=flag_mask_hair,
                                                    background=flag_mask_background)
        mask_img = _mp_mask_generator.add_blur_on_borders(mask_img)

        mask_sha1 = sha1(mask_img.tobytes()).hexdigest()
        image_sha1 = sha1(org_input_img.tobytes()).hexdigest()

        logger.info(f"Generate Image. Prompt: {prompt}")
        if config.is_save_output_enabled():
            folder_path = config.get_output_folder()
            folder_path = os.path.join(folder_path, datetime.now().strftime("%Y%m%d"))

            if not _saved_hashes.get(image_sha1, False):
                _saved_hashes[image_sha1] = True
                save_hashes_to_file()
                utils.save_image_with_timestamp(
                    image=org_input_img,
                    folder_path=folder_path,
                    reference=f"{image_sha1}-0-source",
                    ignore_errors=True)

            if not _saved_hashes.get(mask_sha1, False):
                _saved_hashes[mask_sha1] = True
                save_hashes_to_file()
                utils.save_image_with_timestamp(
                    image=mask_img,
                    folder_path=folder_path,
                    reference=f"{image_sha1}-{mask_sha1}",
                    ignore_errors=True)

        # use always the sliders for strength and steps if they are enabled
        if not config.UI_show_strength_slider():
            strength = 0.8
        if not config.UI_show_steps_slider():
            steps = 40

        params = Image2ImageParameters(
            input_image=input_img,
            prompt=prompt,
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
                    reference=f"{image_sha1}-{mask_sha1}",
                    ignore_errors=True)

        return result_images[0], gr.update(interactive=True)
    except Exception as e:
        logger.error("RuntimeError: %s", str(e))
        logger.debug("Exception details:", exc_info=True)
        gr.Error("There was an internal issue while modify your image")
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
        gr.update(visible=True)
    ]


def wrap_generate_image_response(result_image: Image.Image) -> list:
    """Create a consistent response format for generate_image action.

    Args:
        result_image: The generated image result

    Returns:
        List of values in the order: [output_image, start_button]
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
            gr.Markdown("---")
            gr.Markdown("**Note: black cloth will currently not work!**")
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
                image_source = gr.Image(
                    label="Input",
                    type="pil",
                    height=512)
                # image_custom_mask = gr.ImageEditor(label="Input", type="pil", height=512)
                # describe_button = gr.Button("Describe your Image", interactive=False)
                with gr.Column(visible=True):
                    text_description = gr.Textbox(
                        label="Details",
                        info="Add additional Information like gender (woman, man) or 'age 55' (optional)",
                        show_label=False,
                        max_length=150,
                        max_lines=1)
            with gr.Column():
                image_inpainted = gr.Image(
                    label="Result",
                    type="pil",
                    height=512,
                    interactive=False,
                    show_download_button=True
                )
                # start_button = gr.Button("Start Creation", interactive=True, variant="primary")
                copy_button = gr.Button("use as Input", variant="secondary")
            # --- RIGHT COLUMN: INPAINT CONTROLS ---
            with gr.Column(scale=2):

                # gr.Markdown("## ⚙️ Inpaint Settings")

                # --- SELECTION: INPAINT AREA (Ensures only ONE area is selected) ---
                area_selection = gr.Radio(
                    ["Clothing", "Tattoos", "Hair", "Background"],
                    label="1. Select Inpaint Area",
                    value="Clothing",  # Default selection
                    interactive=True
                )

                # Use gr.Group to visually and logically group the options
                with gr.Group():

                    # --- CLOTHING SECTION (with nested details) ---
                    with gr.Accordion("Clothing Options", open=True) as clothing_acc:

                        # Option: Remove, Replace (only ONE option allowed)
                        clothing_option = gr.Radio(
                            ["Remove", "Replace"],
                            label="2. Choose Option (Clothing)",
                            value="Remove"
                        )
                        # Detail 1: Breast Size (only ONE detail allowed)
                        clothing_details1 = gr.Radio(
                            ["Flat", "Small", "Medium", "Large", "very Large"],
                            label="3a. Detail: Breast Size",
                            value="Medium"
                        )

                        # Detail 2: Hair/Shaved (only ONE detail allowed)
                        clothing_details2 = gr.Radio(
                            ["", "Shaved", "Hairy"],
                            label="3b. Detail: Pubic Hair",
                            visible=False,
                            value=""
                        )
                        cloth_text = gr.Textbox(
                            label="Describe new Cloth or other details",
                            placeholder="e.g., 'Long blonde hair, slightly wavy'",
                            lines=2
                        )
                        clothing_option.change(
                            fn=lambda x: [
                                gr.update(visible=(x == "Remove")),
                                gr.update(visible=(x == "Remove")),
                                gr.update(visible=(x == "Replace"))],
                            inputs=[clothing_option],
                            outputs=[clothing_details1, clothing_details2, cloth_text]
                        )

                    # --- TATTOOS SECTION ---
                    with gr.Accordion("Tattoos Options", visible=False) as tattoos_acc:
                        tattoos_option = gr.Radio(
                            ["Add few", "Add many", "Remove"],
                            label="2. Choose Option (Tattoos)",
                            value="Remove"
                        )

                    # --- HAIR SECTION (Text Input) ---
                    with gr.Accordion("Hair Style & Color", visible=False) as hair_acc:
                        hair_text = gr.Textbox(
                            label="Describe Hair Color and Style",
                            placeholder="e.g., 'Long blonde hair, slightly wavy'",
                            lines=2
                        )

                    # --- BACKGROUND SECTION (Text Input) ---
                    with gr.Accordion("Background Adjustment", visible=False) as background_acc:
                        background_text = gr.Textbox(
                            label="Describe New Background",
                            placeholder="e.g., 'A sunny beach at sunset'",
                            lines=2
                        )

                    # # --- CUSTOM SECTION (Text Input) ---
                    # with gr.Accordion("Custom Inpaint Prompt", visible=False) as custom_acc:
                    #     custom_text = gr.Textbox(
                    #         # label="Enter Custom Prompt (e.g., 'Change color to red')",
                    #         placeholder="Type your custom inpainting instruction here...",
                    #         lines=3
                    #     )

                gr.Markdown("---")

                # --- ACTION BUTTON ---
                apply_btn = gr.Button("🚀 Apply Changes", variant="primary")

                # def update_visibility(selected_area):
                #     # Returns a list of gr.update() commands for each accordion/group
                #     return [
                #         gr.update(visible=(selected_area == "Clothing")),    # clothing_acc
                #         gr.update(visible=(selected_area == "Tattoos")),    # tattoos_acc
                #         gr.update(visible=(selected_area == "Hair")),       # hair_acc
                #         gr.update(visible=(selected_area == "Background")),  # background_acc
                #         gr.update(visible=(selected_area == "Custom"))      # custom_acc
                #     ]
                # --- EVENT BINDING: DYNAMIC VISIBILITY CONTROL ---
                area_selection.change(
                    fn=lambda selected_area:
                    [
                        gr.update(visible=(selected_area == "Clothing")),    # clothing_acc
                        gr.update(visible=(selected_area == "Tattoos")),    # tattoos_acc
                        gr.update(visible=(selected_area == "Hair")),       # hair_acc
                        gr.update(visible=(selected_area == "Background")),  # background_acc
                        # gr.update(visible=(selected_area == "Custom"))      # custom_acc
                    ],
                    inputs=[area_selection],
                    outputs=[clothing_acc, tattoos_acc, hair_acc, background_acc]
                )
                # --- EVENT BINDING: Connect button click to the function ---
                apply_btn.click(
                    fn=lambda: gr.Button(interactive=False),
                    outputs=[apply_btn],
                ).then(
                    fn=process_inpaint_request,
                    inputs=[
                        image_source,
                        text_description,
                        area_selection,
                        # Pass all potential options/details, they will be handled by the function
                        clothing_option,
                        clothing_details1,
                        clothing_details2,
                        cloth_text,
                        tattoos_option,
                        hair_text,
                        background_text
                    ],
                    outputs=[image_inpainted, apply_btn],  # Update log and the resulting image
                    concurrency_id="gpu_queue",
                    show_progress="minimal"
                ).then(
                    fn=lambda: gr.Button(interactive=True),
                    outputs=[apply_btn],
                )

        copy_button.click(
            fn=lambda x: x,
            inputs=[image_inpainted],
            outputs=[image_source]
        )

        # adapt wrap_generate_image_response if you change output parameters
        # start_button.click(
        #     fn=lambda: gr.Button(interactive=False),
        #     outputs=[start_button],
        # ).then(
        #     # fn=action_generate_image,
        #     # inputs=[image_source, strength_slider, steps_slider, text_description],
        #     outputs=[image_inpainted, start_button],
        #     concurrency_limit=config.GenAI_get_execution_batch_size(),
        #     concurrency_id="gpu_queue",
        #     show_progress="minimal"
        #     # batch=False,
        #     # max_batch_size=config.GenAI_get_execution_batch_size(),
        # ).then(
        #     fn=lambda: gr.Button(interactive=True),
        #     outputs=[start_button],
        # )

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
