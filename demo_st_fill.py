from mmgp import offload as offloadobj
import os
import re
import tempfile
import time
from glob import iglob
from io import BytesIO

import numpy as np
import streamlit as st
import torch
from einops import rearrange
from PIL import ExifTags, Image
from st_keyup import st_keyup
from streamlit_drawable_canvas import st_canvas
from transformers import pipeline

from flux.sampling import denoise, get_noise, get_schedule, prepare, prepare_fill, unpack
from flux.util import embed_watermark, load_ae, load_clip, load_flow_model, load_t5

NSFW_THRESHOLD = 0.85
MAXMP = 4 #Max limit in MegaPixels for output image

def add_border_and_mask(image, zoom_all=1.0, zoom_left=0, zoom_right=0, zoom_up=0, zoom_down=0, overlap=0):
    """Adds a black border around the image with individual side control and mask overlap"""
    orig_width, orig_height = image.size

    # Calculate padding for each side (in pixels)
    left_pad = int(orig_width * zoom_left)
    right_pad = int(orig_width * zoom_right)
    top_pad = int(orig_height * zoom_up)
    bottom_pad = int(orig_height * zoom_down)

    # Calculate overlap in pixels
    overlap_left = int(orig_width * overlap)
    overlap_right = int(orig_width * overlap)
    overlap_top = int(orig_height * overlap)
    overlap_bottom = int(orig_height * overlap)

    # If using the all-sides zoom, add it to each side
    if zoom_all > 1.0:
        extra_each_side = (zoom_all - 1.0) / 2
        left_pad += int(orig_width * extra_each_side)
        right_pad += int(orig_width * extra_each_side)
        top_pad += int(orig_height * extra_each_side)
        bottom_pad += int(orig_height * extra_each_side)

    # Calculate new dimensions (ensure they're multiples of 32)
    new_width = 32 * round((orig_width + left_pad + right_pad) / 32)
    new_height = 32 * round((orig_height + top_pad + bottom_pad) / 32)

    # Create new image with black border
    bordered_image = Image.new("RGB", (new_width, new_height), (0, 0, 0))
    # Paste original image in position
    paste_x = left_pad
    paste_y = top_pad
    bordered_image.paste(image, (paste_x, paste_y))

    # Create mask (white where the border is, black where the original image was)
    mask = Image.new("L", (new_width, new_height), 255)  # White background
    # Paste black rectangle with overlap adjustment
    mask.paste(
        0,
        (
            paste_x + overlap_left,  # Left edge moves right
            paste_y + overlap_top,  # Top edge moves down
            paste_x + orig_width - overlap_right,  # Right edge moves left
            paste_y + orig_height - overlap_bottom,  # Bottom edge moves up
        ),
    )

    return bordered_image, mask


@st.cache_resource()
def get_models(name: str, device: torch.device, offload: bool):
    offloadobj.default_verboseLevel = 2

    t5 = load_t5(device, max_length=128)
    #if offload: t5.to("cpu")
    clip = load_clip(device)
    model = load_flow_model(name, "cpu")
    ae = load_ae(name, device="cpu" if offload else device)


    pipe = { "text_encoder": clip, "text_encoder_2": t5, "transformer": model, "vae":ae }
     
    # offloadobj.profile(pipe, quantizeTransformer = False,  profile_no = 1 ) # uncomment this line and comment the previous one if you have 24 GB of VRAM and wants faster generation  
    offloadobj.profile(pipe, quantizeTransformer = False,  extraModelsToQuantize = [], profile_no = 4 ) 

    nsfw_classifier = None # pipeline("image-classification", model="Falconsai/nsfw_image_detection", device=device)
    return model, ae, t5, clip, nsfw_classifier

def resize2(img: Image.Image, min: int = 256, max_mp: float = 6.0) -> Image.Image:
    width, height = img.size
    mp = (width * height) / 1_000_000  # Current megapixels

    if width >=min and height>=min and mp <= max_mp:
        # Even if MP is in range, ensure dimensions are multiples of 32
        new_width = int(32 * round(width / 32))
        new_height = int(32 * round(height / 32))
        if new_width == width and  new_height == height:
            return img

        if  new_width > width or new_height > height  :
            return img.resize((new_width, new_height), Image.Resampling.LANCZOS)
        else:
            return img.resize((new_width, new_height), Image.Resampling.BILINEAR)
            
        return img

    # Calculate scaling factor
    if mp >= max_mp:
        scale = (max_mp / mp) ** 0.5
    elif width < height:
        scale = min / width
    else:
        scale = min / height


    new_width = int(32 * round(width * scale / 32))
    new_height = int(32 * round(height * scale / 32))

    return img.resize((new_width, new_height), Image.Resampling.LANCZOS)


def resize(img: Image.Image, min_mp: float = 0.5, max_mp: float = 2.0) -> Image.Image:
    width, height = img.size
    mp = (width * height) / 1_000_000  # Current megapixels

    if min_mp <= mp <= max_mp:
        # Even if MP is in range, ensure dimensions are multiples of 32
        new_width = int(32 * round(width / 32))
        new_height = int(32 * round(height / 32))
        if new_width != width or new_height != height:
            return img.resize((new_width, new_height), Image.Resampling.LANCZOS)
        return img

    # Calculate scaling factor
    if mp < min_mp:
        scale = (min_mp / mp) ** 0.5
    else:  # mp > max_mp
        scale = (max_mp / mp) ** 0.5

    new_width = int(32 * round(width * scale / 32))
    new_height = int(32 * round(height * scale / 32))

    return img.resize((new_width, new_height), Image.Resampling.LANCZOS)


def clear_canvas_state():
    """Clear all canvas-related state"""
    # keys_to_clear = ["canvas", "reference_image_dims"]
    # for key in keys_to_clear:
    #     if key in st.session_state:
    #         del st.session_state[key]

    # for key in st.session_state:
    #     if "canvas" in key:
    #         del st.session_state[key]

    if "version" in st.session_state: 
        version = st.session_state["version"]
        canvas_key = f"canvas_{version}"
        if canvas_key in st.session_state:
            del st.session_state[canvas_key]
    else:
        version = 0

    st.session_state["version"] = version + 1


def set_new_image(img: Image.Image):
    """Safely set a new image and clear relevant state"""
    st.session_state["current_image"] = img
    if "image_scale_factor" in st.session_state:
        del st.session_state["image_scale_factor"] 

    clear_canvas_state()
    # if rerun:
    #     st.rerun()


def downscale_image(img: Image.Image, scale_factor: float) -> Image.Image:
    """Downscale image by a given factor while maintaining 32-pixel multiple dimensions"""
    if scale_factor >= 1.0:
        return img

    width, height = img.size
    new_width = int(32 * round(width * scale_factor / 32))
    new_height = int(32 * round(height * scale_factor / 32))

    # Ensure minimum dimensions
    new_width = max(64, new_width)  # minimum 64 pixels
    new_height = max(64, new_height)  # minimum 64 pixels

    return img.resize((new_width, new_height), Image.Resampling.LANCZOS)


@torch.inference_mode()
def main(
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    offload: bool = False,
    output_dir: str = "output",
):
    torch_device = torch.device(device)
    load_device = torch.device("cpu") if offload else torch.device(device)
    st.title("Flux Fill GP: Inpainting & Outpainting for the GPU Poor")
    st.markdown("Original tool and models by Black forest labs.")
    st.markdown("*Bug fixing, improvements and support for consumer GPUs (24GB) by Deepbeepmeep.*")
    # Model selection and loading
    name = "flux-dev-fill"
    # if not st.checkbox("Load model", False):
    #     return

    try:
        model, ae, t5, clip, nsfw_classifier = get_models(
            name,
            device=load_device,
            offload=offload,
        )
    except Exception as e:
        st.error(f"Error loading models: {e}")
        return

    # Mode selection
    mode = st.radio("Select Mode", ["Inpainting", "Outpainting"])

    if "version" in st.session_state:
        version = st.session_state["version"]
    else:   
        version = 0


    st.session_state["version"] = version  
    def resetupload():
        if "current_image_name" in st.session_state:
            del st.session_state["current_image_name"]

        
    uploaded_image = st.file_uploader("Upload image", on_change = resetupload,  type=["jpg", "jpeg", "png","webp","jfif"])

    # Image handling - either from previous generation or new upload
    if "input_image" in st.session_state:
        image = st.session_state["input_image"]
        del st.session_state["input_image"]
        st.session_state["reference_image"] = image
        st.session_state["scale_factor"]= 1.0
        set_new_image(image)
        st.session_state["image_status"]= "**Continuing from previous result**"

    else:
        if uploaded_image is None:
            st.warning("Please upload an image")
            return

        if (
            "current_image_name" not in st.session_state
            or st.session_state["current_image_name"] != uploaded_image.name
        ):
            try:
                new_image = Image.open(uploaded_image).convert("RGB")
                st.session_state["current_image_name"] = uploaded_image.name
                #do no rerun to keep zoom and overlap parameters
                st.session_state["Reference_Is_Generated"]= False
                image = resize2(new_image, max_mp = MAXMP)
                if image.width != new_image.width:
                    st.session_state["image_status"]= f"Image has been automatically downscaled from {new_image.width}x{new_image.height} to {image.width}x{image.height} to stay within bounds (256x256 - {MAXMP}MP)"
                    new_image = image
                else:
                    st.session_state["image_status"]= ""

                st.session_state["reference_image"] = new_image
                    
                st.session_state["scale_factor"]= 1.0
                set_new_image(image)
            except Exception as e:
                st.error(f"Error loading image: {e}")
                return
        else:
            image = st.session_state.get("current_image")
            if image is None:
                st.error("Error: Image state is invalid. Please reupload the image.")
                clear_canvas_state()
                return

               
    image_status = "" if "image_status" not in st.session_state else st.session_state["image_status"] 
    if len(image_status)>0:
        st.write(image_status)

    # Add downscale control
    # with st.expander("Downscale image"):
    original_image = st.session_state["reference_image"]
    if "scale_factor" not in st.session_state:
        st.session_state["scale_factor"] = 1
    current_mp = (original_image.size[0] * original_image.size[1]) / 1_000_000
    st.write(f"**Source image dimensions: {original_image.size[0]}x{original_image.size[1]} ({current_mp:.1f}MP)**")

    scale_factor = st.slider(
        "Downscale Factor",
        min_value=0.1,
        max_value=1.0,
        step=0.01,
        key="scale_factor",
        help="1.0 = original size, 0.5 = half size, etc.",
    )


    # if st.button("Apply Downscaling"): #scale_factor < 1.0 and
    image_scale_factor = 0 if "image_scale_factor" not in st.session_state else st.session_state["image_scale_factor"]
    
    if image_scale_factor != scale_factor:
        image = original_image
        image = downscale_image(image, scale_factor)
        set_new_image(image)
        st.session_state["image_scale_factor"] = scale_factor
       # st.session_state["image_status"]= ""

    #       st.rerun()


    width, height = image.size
    st.write(f"**Current image dimensions: {width}x{height} pixels**")
 
    if mode == "Outpainting":
        # Outpainting controls
        if st.button("Reset Zoom and Overlap Parameters") or "zoom_all" not in st.session_state:
            st.session_state.zoom_all = 1
            st.session_state.zoom_left = 0
            st.session_state.zoom_right = 0
            st.session_state.zoom_up = 0
            st.session_state.zoom_down = 0
            st.session_state.overlap = 0.01

        zoom_all = st.slider("Zoom Out Amount (All Sides)", min_value=1.0, max_value=3.0, step=0.01, key="zoom_all")

        with st.expander("Advanced Zoom Controls"):
            st.info("These controls add additional zoom to specific sides")
            col1, col2 = st.columns(2)
            with col1:
                zoom_left = st.slider("Left", min_value=0.0, max_value=1.0, step=0.1,key="zoom_left")
                zoom_right = st.slider("Right", min_value=0.0, max_value=1.0,  step=0.1,key="zoom_right")
            with col2:
                zoom_up = st.slider("Up", min_value=0.0, max_value=1.0,  step=0.1,key="zoom_up")
                zoom_down = st.slider("Down", min_value=0.0, max_value=1.0,  step=0.1, key="zoom_down")

        overlap = st.slider("Overlap", min_value=0.01, max_value=0.25,  step=0.01,key="overlap")

        # Generate bordered image and mask
        image_for_generation, mask = add_border_and_mask(
            image,
            zoom_all=zoom_all,
            zoom_left=zoom_left,    
            zoom_right=zoom_right,
            zoom_up=zoom_up,
            zoom_down=zoom_down,
            overlap=overlap,
        )
        width, height = image_for_generation.size

        # Show preview
        col1, col2 = st.columns(2)
        with col1:
            st.image(image_for_generation, caption="Image with Border")
        with col2:
            st.image(mask, caption="Mask (white areas will be generated)")

    else:  # Inpainting mode
        # Canvas setup with dimension tracking
        version = st.session_state["version"]
        #canvas_key = f"canvas_{width}_{height}_{version}"
        canvas_key = f"canvas_{version}"
        # if "reference_image_dims" not in st.session_state:
        #     st.session_state.reference_image_dims = (width, height)
        # elif st.session_state.reference_image_dims != (width, height):
        #     clear_canvas_state()
        #     st.session_state.reference_image_dims = (width, height)
        #     st.rerun()

        try:
            canvas_result = st_canvas(
                fill_color="rgba(255, 255, 255, 0.0)",
                stroke_width=st.slider("Brush size", 1, 500, 50),
                stroke_color="#fff",
                background_image=image,
                height=height,
                width=width,
                drawing_mode="freedraw",
                key=canvas_key,
                display_toolbar=True,
            )
        except Exception as e:
            st.error(f"Error creating canvas: {e}")
            clear_canvas_state()
            st.rerun()
            return
        image_for_generation = image
       
    try:
        original_width= image_for_generation.width
        original_height= image_for_generation.height
        original_mp = (image_for_generation.size[0] * image_for_generation.size[1]) / 1_000_000
        image_for_generation = resize2(image_for_generation, max_mp= MAXMP)

        width, height = image_for_generation.size
        current_mp = (width * height) / 1_000_000

        if width % 32 != 0 or height % 32 != 0:
            st.error("Error: Image dimensions must be multiples of 32")
            return

        if original_mp != current_mp:
            st.write(
                f"Image will be resized from {original_width}x{original_height} to {width}x{height} to stay within bounds (256x256 - {MAXMP}MP)"
            )

        st.write(f"**Target image dimensions: {width}x{height} pixels**")

    except Exception as e:
        st.error(f"Error processing image: {e}")
        return        

    # Sampling parameters
    num_steps = int(st.number_input("Number of steps", min_value=1, value=50))
    guidance = float(st.number_input("Guidance", min_value=1.0, value=30.0))
    seed_str = st.text_input("Seed")
    if seed_str.isdecimal():
        seed = int(seed_str)
    else:
        st.info("No seed set, using random seed")
        seed = None

    save_samples = st.checkbox("Save samples?", True)
    add_sampling_metadata = st.checkbox("Add sampling parameters to metadata?", True)

    # Prompt input
    prompt = st_keyup("Enter a prompt", value="", debounce=300, key="interactive_text")

    # Setup output path
    output_name = os.path.join(output_dir, "img_{idx}.jpg")
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        idx = 0
    else:
        fns = [fn for fn in iglob(output_name.format(idx="*")) if re.search(r"img_[0-9]+\.jpg$", fn)]
        idx = len(fns)



    if st.button("Generate"):
        valid_input = False

        if mode == "Inpainting" and canvas_result.image_data is not None:
            valid_input = True
            # Create mask from canvas
            try:
                mask = Image.fromarray(canvas_result.image_data)
                mask = mask.getchannel("A")  # Get alpha channel
                mask_array = np.array(mask)
                mask_array = (mask_array > 0).astype(np.uint8) * 255
                mask = Image.fromarray(mask_array)
            except Exception as e:
                st.error(f"Error creating mask: {e}")
                return

        elif mode == "Outpainting":
            valid_input = True
            # image_for_generation and mask are already set above

        if not valid_input:
            st.error("Please draw a mask or configure outpainting settings")
            return

        # Resize image
        image_for_generation = resize2(image_for_generation, max_mp= MAXMP)
        mask = resize2(mask,max_mp = MAXMP)

        # Create temporary files
        with (
            tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp_img,
            tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp_mask,
        ):
            try:
                image_for_generation.save(tmp_img.name)
                mask.save(tmp_mask.name)
            except Exception as e:
                st.error(f"Error saving temporary files: {e}")
                return

            try:
                # Generate inpainting/outpainting
                rng = torch.Generator(device="cpu")
                if seed is None:
                    seed = rng.seed()

                print(f"Generating with seed {seed}:\n{prompt}")
                t0 = time.perf_counter()

                placeholder = st.empty()
                placeholder.write("Initializing...")
                x = get_noise(
                    1,
                    height,
                    width,
                    device=torch_device,
                    dtype=torch.bfloat16,
                    seed=seed,
                )

                placeholder.write("Encoding prompt...")
                inp = prepare(t5, clip, x, prompt)

                placeholder.write("Encoding image...")
                inp = prepare_fill(
                    x,
                    prompt=prompt,
                    ae=ae,
                    img_cond_path=tmp_img.name,
                    mask_path=tmp_mask.name,
                    return_dict = inp
                )

                timesteps = get_schedule(num_steps, inp["img"].shape[1], shift=True)
 
                progress_text = "Denoising..."
                my_bar = placeholder.progress(0, text=progress_text)

                def progress_noise(percent_complete):
                    my_bar.progress(percent_complete, text=progress_text)
                    
                x = denoise(model, **inp, timesteps=timesteps, guidance=guidance, progress_callback =progress_noise)

                placeholder.write("Decoding latents...")
                x = unpack(x.float(), height, width)
                with torch.autocast(device_type=torch_device.type, dtype=torch.bfloat16):
                    x = ae.decode(x)

                t1 = time.perf_counter()
                print(f"Done in {t1 - t0:.1f}s")
                placeholder.write(f"Done in {t1 - t0:.1f}s")

                # Process and display result
                x = x.clamp(-1, 1)
                x = embed_watermark(x.float())
                x = rearrange(x[0], "c h w -> h w c")
                img = Image.fromarray((127.5 * (x + 1.0)).cpu().byte().numpy())
                x = None
                import gc
                gc.collect()

                #nsfw_score = [x["score"] for x in nsfw_classifier(img) if x["label"] == "nsfw"][0]

                if True: #nsfw_score < NSFW_THRESHOLD:
                    buffer = BytesIO()
                    exif_data = Image.Exif()
                    exif_data[ExifTags.Base.Software] = "AI generated;inpainting;flux"
                    exif_data[ExifTags.Base.Make] = "Black Forest Labs"
                    exif_data[ExifTags.Base.Model] = name
                    if add_sampling_metadata:
                        exif_data[ExifTags.Base.ImageDescription] = prompt
                    img.save(buffer, format="jpeg", exif=exif_data, quality=98, subsampling=0)

                    img_bytes = buffer.getvalue()
                    if save_samples:
                        fn = output_name.format(idx=idx)
                        print(f"Saving {fn}")
                        with open(fn, "wb") as file:
                            file.write(img_bytes)

                    st.session_state["samples"] = {
                        "prompt": prompt,
                        "img": img,
                        "seed": seed,
                        "bytes": img_bytes,
                    }
                else:
                    st.warning("Your generated image may contain NSFW content.")
                    st.session_state["samples"] = None

            except Exception as e:
                st.error(f"Error during generation: {e}")
                return
            finally:
                # Clean up temporary files
                try:
                    os.unlink(tmp_img.name)
                    os.unlink(tmp_mask.name)
                except Exception as e:
                    print(f"Error cleaning up temporary files: {e}")

    # Display results
    samples = st.session_state.get("samples", None)
    if samples is not None:
        st.image(samples["img"], caption=samples["prompt"])
        col1, col2 = st.columns(2)
        with col1:
            st.download_button(
                "Download full-resolution",
                samples["bytes"],
                file_name="generated.jpg",
                mime="image/jpg",
            )
        with col2:
            if st.button("Continue from this image"):
                # Clear ALL canvas state
                clear_canvas_state()
                # Store the generated image
                new_image = samples["img"]
                if "samples" in st.session_state:
                    del st.session_state["samples"]
                # Set as current image
                st.session_state["input_image"] = new_image
#                st.session_state["current_image"] = new_image
                st.session_state["reference_image"] = new_image

                st.rerun()

        st.write(f"Seed: {samples['seed']}")


if __name__ == "__main__":
    st.set_page_config(layout="wide")
    main(offload = True)
