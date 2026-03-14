import os
import json
import time
import shutil
import subprocess
import tempfile
import psutil
import ffmpeg
import imageio
import argparse
from pathlib import Path
from PIL import Image

import cv2
import torch
import numpy as np
import gradio as gr

from tools.painter import mask_painter
from tools.interact_tools import SamControler
from tools.misc import get_device
from tools.download_util import load_file_from_url

from matanyone2_wrapper import matanyone2
from matanyone2.utils.get_default_model import get_matanyone2_model
from matanyone2.utils.device import clean_vram
from matanyone2.inference.inference_core import InferenceCore
from hydra.core.global_hydra import GlobalHydra

import warnings
warnings.filterwarnings("ignore")

# Suppress annoyingly persistent Windows asyncio proactor errors
if os.name == 'nt':  # Windows only
    import asyncio
    from functools import wraps
    import socket # Required for the ConnectionResetError
    
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    
    def silence_connection_errors(func):
        @wraps(func)
        def wrapper(self, *args, **kwargs):
            try:
                return func(self, *args, **kwargs)
            except (ConnectionResetError, BrokenPipeError):
                pass
            except RuntimeError as e:
                if str(e) != 'Event loop is closed':
                    raise
        return wrapper
    
    from asyncio import proactor_events
    if hasattr(proactor_events, '_ProactorBasePipeTransport'):
        proactor_events._ProactorBasePipeTransport._call_connection_lost = silence_connection_errors(
            proactor_events._ProactorBasePipeTransport._call_connection_lost
        )
        
# -------------------------------------------------------------------
# UI Settings
# -------------------------------------------------------------------
UI_SETTINGS_PATH = Path("./ui_settings.json")
GRADIO_TEMP_DIR = Path(os.environ.get("GRADIO_TEMP_DIR", tempfile.gettempdir()))

def load_ui_settings():
    if UI_SETTINGS_PATH.exists():
        try:
            with open(UI_SETTINGS_PATH, "r") as f:
                return json.load(f)
        except Exception:
            return {}
    return {}

def save_ui_settings(settings: dict):
    try:
        with open(UI_SETTINGS_PATH, "w") as f:
            json.dump(settings, f, indent=2)
    except Exception:
        pass

def get_theme_from_settings():
    settings = load_ui_settings()
    theme_name = settings.get("theme", "Default")
    theme_map = {
        "Default": gr.themes.Default(),
        "Soft": gr.themes.Soft(),
        "Monochrome": gr.themes.Monochrome(),
        "Glass": gr.themes.Glass(),
        "Base": gr.themes.Base(),
        "Ocean": gr.themes.Ocean(),
        "Origin": gr.themes.Origin(),
        "Citrus": gr.themes.Citrus(),
    }
    community_themes = {
        "Miku": "NoCrypt/miku",
        "Interstellar": "Nymbo/Interstellar",
        "xkcd": "gstaff/xkcd",
        "kotaemon": "lone17/kotaemon",
    }
    if theme_name in theme_map:
        return theme_map[theme_name]
    elif theme_name in community_themes:
        return community_themes[theme_name]
    return gr.themes.Monochrome()

# -------------------------------------------------------------------
# Startup Banner
# -------------------------------------------------------------------
def _print_banner():
    """Print a stylish startup banner with current settings."""
    W = 60
    CYAN    = "\033[96m"
    BOLD    = "\033[1m"
    DIM     = "\033[2m"
    RESET   = "\033[0m"

    border = f"{CYAN}{'━' * W}{RESET}"

    def center(text, raw_len=None):
        visible = raw_len if raw_len is not None else len(text)
        pad = max(0, (W - visible) // 2)
        return " " * pad + text

    logo_lines = [
        r" __  __       _      _                                 ____  ",
        r"|  \/  | __ _| |_   / \   _ __  _   _  ___  _ __   __|___ \ ",
        r"| |\/| |/ _` | __| / _ \ | '_ \| | | |/ _ \| '_ \ / _ \__) |",
        r"| |  | | (_| | |_ / ___ \| | | | |_| | (_) | | | |  __/ __/ ",
        r"|_|  |_|\__,_|\__/_/   \_\_| |_|\__, |\___/|_| |_|\___|_____|",
        r"                                 |___/                        ",
    ]

    print()
    print(border)
    for line in logo_lines:
        print(center(f"{BOLD}{CYAN}{line}{RESET}", len(line)))
    print(center(f"{DIM}Video & Image Background Removal{RESET}", 32))
    print(border)
    print()

def _print_settings(settings):
    """Print the active settings being applied."""
    W = 60
    CYAN    = "\033[96m"
    GREEN   = "\033[92m"
    DIM     = "\033[2m"
    BOLD    = "\033[1m"
    RESET   = "\033[0m"

    thin = f"{DIM}{'─' * W}{RESET}"

    theme      = settings.get("theme", "Default")
    output_dir = settings.get("output_dir", "./outputs")
    clear_temp = settings.get("clear_temp_on_start", False)

    print(f"  {BOLD}⚙  Settings{RESET}")
    print(thin)
    print(f"  {DIM}Theme{RESET}            {CYAN}{theme}{RESET}")
    print(f"  {DIM}Output dir{RESET}       {CYAN}{output_dir}{RESET}")
    clear_icon = f"{GREEN}✓ enabled{RESET}" if clear_temp else f"{DIM}✗ disabled{RESET}"
    print(f"  {DIM}Clear temp{RESET}       {clear_icon}")
    print()

def _print_action(icon, message):
    """Print a single startup action line."""
    DIM   = "\033[2m"
    RESET = "\033[0m"
    print(f"  {icon}  {DIM}{message}{RESET}")

def _print_ready(port):
    """Print the final ready banner."""
    W = 60
    CYAN    = "\033[96m"
    GREEN   = "\033[92m"
    BOLD    = "\033[1m"
    RESET   = "\033[0m"

    border = f"{CYAN}{'━' * W}{RESET}"
    print()
    print(border)
    print(f"  {GREEN}{BOLD}✓  Ready!{RESET}  Launching Gradio on port {CYAN}{port}{RESET}")
    print(border)
    print()

# -------------------------------------------------------------------
# Apply startup settings
# -------------------------------------------------------------------
_settings = load_ui_settings()
_print_banner()
_print_settings(_settings)

OUTPUT_DIR = Path(_settings.get("output_dir", "./outputs"))
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
_print_action("📂", f"Output directory ensured: {OUTPUT_DIR}")

# Clear temp on start if enabled
if _settings.get("clear_temp_on_start", False):
    _temp_cleared = 0
    for temp_dir in [GRADIO_TEMP_DIR]:
        if temp_dir.exists():
            for item in temp_dir.iterdir():
                try:
                    if item.is_file():
                        item.unlink()
                        _temp_cleared += 1
                    elif item.is_dir():
                        shutil.rmtree(item)
                        _temp_cleared += 1
                except Exception:
                    pass
    _print_action("🧹", f"Cleared {_temp_cleared} temp items from {GRADIO_TEMP_DIR}")
else:
    _print_action("🧹", "Temp cleanup on start: skipped (disabled)")

# Clean any stale temp results from previous sessions
_results_dir = Path("./_temp_results")
if _results_dir.exists():
    _stale = 0
    for item in _results_dir.iterdir():
        try:
            item.unlink()
            _stale += 1
        except Exception:
            pass
    if _stale:
        _print_action("🗑️ ", f"Cleaned {_stale} stale result(s) from _temp_results")

print()

def parse_augment():
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', type=str, default=None)
    parser.add_argument('--sam_model_type', type=str, default="vit_h")
    parser.add_argument('--port', type=int, default=8000, help="only useful when running gradio applications")
    parser.add_argument('--mask_save', default=False)
    args = parser.parse_args()

    if not args.device:
        args.device = str(get_device())

    return args

# SAM generator
class MaskGenerator():
    def __init__(self, sam_checkpoint, args):
        self.args = args
        self.samcontroler = SamControler(sam_checkpoint, args.sam_model_type, args.device)

    def first_frame_click(self, image: np.ndarray, points:np.ndarray, labels: np.ndarray, multimask=True):
        mask, logit, painted_image = self.samcontroler.first_frame_click(image, points, labels, multimask)
        return mask, logit, painted_image

# convert points input to prompt state
def get_prompt(click_state, click_input):
    inputs = json.loads(click_input)
    points = click_state[0]
    labels = click_state[1]
    for input in inputs:
        points.append(input[:2])
        labels.append(input[2])
    click_state[0] = points
    click_state[1] = labels
    prompt = {
        "prompt_type":["click"],
        "input_point":click_state[0],
        "input_label":click_state[1],
        "multimask_output":"True",
    }
    return prompt

# -------------------------------------------------------------------
# Resolution management
# -------------------------------------------------------------------
RESOLUTION_PRESETS = {
    "Original": None,
    "2160p (4K)": 2160,
    "1440p (2K)": 1440,
    "1080p (Full HD)": 1080,
    "720p (HD)": 720,
    "540p": 540,
}

def calc_resize_dims(h, w, target_short_side):
    """Calculate new dimensions constraining the short side, preserving aspect ratio.
    Returns (new_h, new_w) or None if no resize needed.
    Dimensions are rounded up to nearest multiple of 16 for model compatibility."""
    if target_short_side is None:
        return None
    short_side = min(h, w)
    if short_side <= target_short_side:
        return None  # already smaller than target
    scale = target_short_side / short_side
    new_h = int(h * scale)
    new_w = int(w * scale)
    # ensure dimensions are divisible by 16 (round up)
    new_h = ((new_h + 15) // 16) * 16
    new_w = ((new_w + 15) // 16) * 16
    return (new_h, new_w)

def resize_frames(frames, target_short_side):
    """Resize a list of frames. Returns (resized_frames, was_resized)."""
    if not frames or target_short_side is None:
        return frames, False
    h, w = frames[0].shape[:2]
    dims = calc_resize_dims(h, w, target_short_side)
    if dims is None:
        return frames, False
    new_h, new_w = dims
    resized = [cv2.resize(f, (new_w, new_h), interpolation=cv2.INTER_AREA) for f in frames]
    return resized, True

def upscale_matte_to_original(alpha_frames, orig_h, orig_w):
    """Upscale alpha matte frames back to original resolution using Lanczos."""
    return [cv2.resize(f, (orig_w, orig_h), interpolation=cv2.INTER_LANCZOS4) for f in alpha_frames]

def composite_at_original_res(orig_frames, alpha_frames_upscaled):
    """Composite foreground at original resolution using upscaled alpha."""
    bgr = (np.array([120, 255, 155], dtype=np.float32) / 255).reshape((1, 1, 3))
    composited = []
    for frame, alpha in zip(orig_frames, alpha_frames_upscaled):
        pha = alpha.astype(np.float32) / 255.0
        if pha.ndim == 2:
            pha = pha[:, :, np.newaxis]
        com = frame.astype(np.float32) / 255.0 * pha + bgr * (1 - pha)
        composited.append((com * 255).astype(np.uint8))
    return composited

def format_resolution_info(h, w, num_frames=None, fps=None, name=None, work_h=None, work_w=None):
    """Format an HTML badge matching the UI button style, single line."""
    parts = [f"{w}×{h}"]
    if fps is not None:
        parts.append(f"{round(fps, 1)} fps")
    if num_frames is not None:
        parts.append(f"{num_frames} frames")
        if fps and fps > 0:
            duration = num_frames / fps
            parts.append(f"{duration:.1f}s")
    if work_h is not None and work_w is not None and (work_h != h or work_w != w):
        reduction = (1 - (work_h * work_w) / (h * w)) * 100
        parts.append(f"→ {work_w}×{work_h} ({reduction:.0f}% ↓)")

    inner = " &nbsp;·&nbsp; ".join(parts)
    return (
        f'<div style="display:inline-block; padding:8px 16px; background:#171717; '
        f'color:#fff; border-radius:8px; font-size:0.85em; font-family:Arial,sans-serif; '
        f'text-align:center; width:100%; box-sizing:border-box;">'
        f'{inner}</div>'
    )

def get_frames_from_image(image_input, image_state, resize_preset):
    user_name = time.time()
    orig_h, orig_w = image_input.shape[:2]
    target = RESOLUTION_PRESETS.get(resize_preset)

    yield gr.update(value="⏳ Processing image..."), image_state, gr.update(), gr.update(), gr.update(), gr.update(), gr.update(), gr.update(), gr.update(), gr.update(), gr.update(), gr.update(), gr.update(), gr.update(), gr.update(), gr.update(), gr.update()

    frames_orig = [image_input] * 2
    frames_work, was_resized = resize_frames(frames_orig, target)
    work_h, work_w = frames_work[0].shape[:2]

    image_state = {
        "user_name": user_name,
        "image_name": "output.png",
        "origin_images": frames_work,
        "painted_images": frames_work.copy(),
        "masks": [np.zeros((work_h, work_w), np.uint8)] * len(frames_work),
        "logits": [None] * len(frames_work),
        "select_frame_number": 0,
        "fps": None,
        "original_size": (orig_h, orig_w),
        "original_images": frames_orig if was_resized else None,
        }
    resize_note = ""
    image_info = format_resolution_info(orig_h, orig_w, work_h=work_h if was_resized else None, work_w=work_w if was_resized else None)
    model.samcontroler.sam_controler.reset_image()
    model.samcontroler.sam_controler.set_image(image_state["origin_images"][0])
    yield gr.update(value="✅ Image loaded. Click the image to place mask points on your target."), image_state, gr.update(visible=True, value=image_info), image_state["origin_images"][0], \
                        gr.update(visible=True, maximum=10, value=10), gr.update(visible=False, maximum=len(frames_work), value=len(frames_work)), \
                        gr.update(visible=True), gr.update(visible=True), \
                        gr.update(visible=True), gr.update(visible=True),\
                        gr.update(visible=True), gr.update(visible=True), \
                        gr.update(visible=True), gr.update(visible=False), \
                        gr.update(visible=False), gr.update(visible=True), \
                        gr.update(visible=True)

# extract frames from upload video
def get_frames_from_video(video_input, video_state, resize_preset):
    video_path = video_input
    frames = []
    user_name = time.time()

    yield gr.update(value="⏳ Extracting audio..."), video_state, gr.update(), gr.update(), gr.update(), gr.update(), gr.update(), gr.update(), gr.update(), gr.update(), gr.update(), gr.update(), gr.update(), gr.update(), gr.update(), gr.update(), gr.update()

    # extract Audio (only if the video has an audio stream)
    audio_path = ""
    try:
        probe = ffmpeg.probe(video_path, select_streams='a')
        if probe.get("streams"):
            audio_path = video_input.replace(".mp4", "_audio.wav")
            ffmpeg.input(video_path).output(audio_path, format='wav', acodec='pcm_s16le', ac=2, ar='44100').run(overwrite_output=True, quiet=True)
    except Exception as e:
        print(f"Audio extraction error: {str(e)}")
        audio_path = ""

    yield gr.update(value="⏳ Extracting frames from video..."), video_state, gr.update(), gr.update(), gr.update(), gr.update(), gr.update(), gr.update(), gr.update(), gr.update(), gr.update(), gr.update(), gr.update(), gr.update(), gr.update(), gr.update(), gr.update()

    # extract frames
    try:
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        while cap.isOpened():
            ret, frame = cap.read()
            if ret == True:
                current_memory_usage = psutil.virtual_memory().percent
                frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                if current_memory_usage > 90:
                    break
            else:
                break
    except (OSError, TypeError, ValueError, KeyError, SyntaxError) as e:
        print("read_frame_source:{} error. {}\n".format(video_path, str(e)))

    orig_h, orig_w = frames[0].shape[:2]
    target = RESOLUTION_PRESETS.get(resize_preset)

    if target is not None:
        yield gr.update(value=f"⏳ Resizing {len(frames)} frames..."), video_state, gr.update(), gr.update(), gr.update(), gr.update(), gr.update(), gr.update(), gr.update(), gr.update(), gr.update(), gr.update(), gr.update(), gr.update(), gr.update(), gr.update(), gr.update()

    frames_orig = [f.copy() for f in frames] if target is not None else None
    frames_work, was_resized = resize_frames(frames, target)
    work_h, work_w = frames_work[0].shape[:2]

    yield gr.update(value="⏳ Preparing SAM model for masking..."), video_state, gr.update(), gr.update(), gr.update(), gr.update(), gr.update(), gr.update(), gr.update(), gr.update(), gr.update(), gr.update(), gr.update(), gr.update(), gr.update(), gr.update(), gr.update()

    video_state = {
        "user_name": user_name,
        "video_name": os.path.split(video_path)[-1],
        "origin_images": frames_work,
        "painted_images": frames_work.copy(),
        "masks": [np.zeros((work_h, work_w), np.uint8)] * len(frames_work),
        "logits": [None] * len(frames_work),
        "select_frame_number": 0,
        "fps": fps,
        "audio": audio_path,
        "original_size": (orig_h, orig_w),
        "original_images": frames_orig if was_resized else None,
        }
    video_info = format_resolution_info(orig_h, orig_w, len(frames_work), fps, video_state["video_name"], work_h=work_h if was_resized else None, work_w=work_w if was_resized else None)
    model.samcontroler.sam_controler.reset_image()
    model.samcontroler.sam_controler.set_image(video_state["origin_images"][0])
    yield gr.update(value="✅ Video loaded. Click the frame to place mask points on your target."), video_state, gr.update(visible=True, value=video_info), video_state["origin_images"][0], gr.update(visible=True, maximum=len(frames), value=1), gr.update(visible=False, maximum=len(frames), value=len(frames)), \
                        gr.update(visible=True), gr.update(visible=True), \
                        gr.update(visible=True), gr.update(visible=True),\
                        gr.update(visible=True), gr.update(visible=True), \
                        gr.update(visible=True), gr.update(visible=False), \
                        gr.update(visible=False), gr.update(visible=True), \
                        gr.update(visible=True)

# get the select frame from gradio slider
def select_video_template(image_selection_slider, video_state, interactive_state):
    image_selection_slider -= 1
    video_state["select_frame_number"] = image_selection_slider
    model.samcontroler.sam_controler.reset_image()
    model.samcontroler.sam_controler.set_image(video_state["origin_images"][image_selection_slider])
    return video_state["painted_images"][image_selection_slider], video_state, interactive_state

def select_image_template(image_selection_slider, video_state, interactive_state):
    image_selection_slider = 0 # fixed for image
    video_state["select_frame_number"] = image_selection_slider
    model.samcontroler.sam_controler.reset_image()
    model.samcontroler.sam_controler.set_image(video_state["origin_images"][image_selection_slider])
    return video_state["painted_images"][image_selection_slider], video_state, interactive_state

# set the tracking end frame
def get_end_number(track_pause_number_slider, video_state, interactive_state):
    interactive_state["track_end_number"] = track_pause_number_slider
    return video_state["painted_images"][track_pause_number_slider],interactive_state

# use sam to get the mask
def sam_refine(video_state, point_prompt, click_state, interactive_state, evt:gr.SelectData):
    # Scale click coordinates to match actual working frame dimensions.
    # Gradio may report coordinates in display/original space rather than
    # the resized image data space, so we scale proportionally.
    current_frame = video_state["origin_images"][video_state["select_frame_number"]]
    img_h, img_w = current_frame.shape[:2]
    raw_x, raw_y = evt.index[0], evt.index[1]

    # If coordinates exceed image bounds, scale them down proportionally
    if raw_x >= img_w or raw_y >= img_h:
        orig_size = video_state.get("original_size")
        if orig_size is not None:
            orig_h, orig_w = orig_size
            scale_x = img_w / orig_w
            scale_y = img_h / orig_h
            cx = int(raw_x * scale_x)
            cy = int(raw_y * scale_y)
        else:
            cx, cy = raw_x, raw_y
        cx = min(max(cx, 0), img_w - 1)
        cy = min(max(cy, 0), img_h - 1)
    else:
        cx, cy = int(raw_x), int(raw_y)

    if point_prompt == "Positive":
        coordinate = "[[{},{},1]]".format(cx, cy)
        interactive_state["positive_click_times"] += 1
    else:
        coordinate = "[[{},{},0]]".format(cx, cy)
        interactive_state["negative_click_times"] += 1

    model.samcontroler.sam_controler.reset_image()
    model.samcontroler.sam_controler.set_image(video_state["origin_images"][video_state["select_frame_number"]])
    prompt = get_prompt(click_state=click_state, click_input=coordinate)

    mask, logit, painted_image = model.first_frame_click(
                                                      image=video_state["origin_images"][video_state["select_frame_number"]],
                                                      points=np.array(prompt["input_point"]),
                                                      labels=np.array(prompt["input_label"]),
                                                      multimask=prompt["multimask_output"],
                                                      )
    video_state["masks"][video_state["select_frame_number"]] = mask
    video_state["logits"][video_state["select_frame_number"]] = logit
    video_state["painted_images"][video_state["select_frame_number"]] = painted_image

    return painted_image, video_state, interactive_state

def add_multi_mask(video_state, interactive_state, mask_dropdown):
    mask = video_state["masks"][video_state["select_frame_number"]]
    interactive_state["multi_mask"]["masks"].append(mask)
    interactive_state["multi_mask"]["mask_names"].append("mask_{:03d}".format(len(interactive_state["multi_mask"]["masks"])))
    mask_dropdown.append("mask_{:03d}".format(len(interactive_state["multi_mask"]["masks"])))
    select_frame = show_mask(video_state, interactive_state, mask_dropdown)
    return interactive_state, gr.update(choices=interactive_state["multi_mask"]["mask_names"], value=mask_dropdown), select_frame, [[],[]]

def clear_click(video_state, click_state):
    click_state = [[],[]]
    template_frame = video_state["origin_images"][video_state["select_frame_number"]]
    return template_frame, click_state

def remove_multi_mask(interactive_state, mask_dropdown):
    interactive_state["multi_mask"]["mask_names"]= []
    interactive_state["multi_mask"]["masks"] = []
    return interactive_state, gr.update(choices=[],value=[])

def show_mask(video_state, interactive_state, mask_dropdown):
    mask_dropdown.sort()
    if video_state["origin_images"]:
        select_frame = video_state["origin_images"][video_state["select_frame_number"]]
        for i in range(len(mask_dropdown)):
            mask_number = int(mask_dropdown[i].split("_")[1]) - 1
            mask = interactive_state["multi_mask"]["masks"][mask_number]
            select_frame = mask_painter(select_frame, mask.astype('uint8'), mask_color=mask_number+2)
        return select_frame

# image matting
def image_matting(video_state, interactive_state, mask_dropdown, erode_kernel_size, dilate_kernel_size, refine_iter, model_selection):
    progress_state = {"text": "⏳ Loading model..."}

    yield gr.update(value=progress_state["text"]), gr.update(), gr.update()

    try:
        selected_model = load_model(model_selection)
    except (FileNotFoundError, ValueError) as e:
        if available_models:
            print(f"Warning: {str(e)}. Using {available_models[0]} instead.")
            selected_model = load_model(available_models[0])
        else:
            raise ValueError("No models are available! Please check if the model files exist.")
    matanyone_processor = InferenceCore(selected_model, cfg=selected_model.cfg)
    if interactive_state["track_end_number"]:
        following_frames = video_state["origin_images"][video_state["select_frame_number"]:interactive_state["track_end_number"]]
    else:
        following_frames = video_state["origin_images"][video_state["select_frame_number"]:]

    if interactive_state["multi_mask"]["masks"]:
        if len(mask_dropdown) == 0:
            mask_dropdown = ["mask_001"]
        mask_dropdown.sort()
        template_mask = interactive_state["multi_mask"]["masks"][int(mask_dropdown[0].split("_")[1]) - 1] * (int(mask_dropdown[0].split("_")[1]))
        for i in range(1,len(mask_dropdown)):
            mask_number = int(mask_dropdown[i].split("_")[1]) - 1
            template_mask = np.clip(template_mask+interactive_state["multi_mask"]["masks"][mask_number]*(mask_number+1), 0, mask_number+1)
        video_state["masks"][video_state["select_frame_number"]]= template_mask
    else:
        template_mask = video_state["masks"][video_state["select_frame_number"]]

    if len(np.unique(template_mask))==1:
        template_mask[0][0]=1

    yield gr.update(value=f"⏳ Running inference ({int(refine_iter)} refinement iterations)..."), gr.update(), gr.update()

    def on_progress(current, total, phase):
        if phase == "warmup":
            progress_state["text"] = f"⏳ Warmup: {current}/{total}"
        else:
            progress_state["text"] = f"⏳ Matting: iteration {current}/{total}"

    foreground, alpha = matanyone2(matanyone_processor, following_frames, template_mask*255, r_erode=erode_kernel_size, r_dilate=dilate_kernel_size, n_warmup=refine_iter, progress_callback=on_progress)

    del matanyone_processor
    clean_vram()

    yield gr.update(value="⏳ Preparing outputs..."), gr.update(), gr.update()

    # Upscale to original resolution if we processed at reduced res
    orig_size = video_state.get("original_size")
    orig_images = video_state.get("original_images")
    if orig_images is not None and orig_size is not None:
        orig_h, orig_w = orig_size
        if interactive_state["track_end_number"]:
            orig_following = orig_images[video_state["select_frame_number"]:interactive_state["track_end_number"]]
        else:
            orig_following = orig_images[video_state["select_frame_number"]:]

        yield gr.update(value=f"⏳ Upscaling to original resolution ({orig_w}×{orig_h})..."), gr.update(), gr.update()
        alpha_upscaled = upscale_matte_to_original(alpha, orig_h, orig_w)
        foreground = composite_at_original_res(orig_following, alpha_upscaled)
        alpha = alpha_upscaled
        fg_frame = foreground[-1]
        alpha_frame = alpha[-1]
    else:
        fg_frame = foreground[-1]
        alpha_frame = alpha[-1][:,:,0] if alpha[-1].ndim == 3 else alpha[-1]

    foreground_output = Image.fromarray(fg_frame)
    alpha_output = Image.fromarray(alpha_frame[:,:,0] if alpha_frame.ndim == 3 else alpha_frame)
    # Save to results dir (clean previous results first)
    _cleanup_results()
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    ts = time.strftime("%Y%m%d_%H%M%S")
    fg_path = str(RESULTS_DIR / f"image_fg_{ts}.png")
    alpha_path = str(RESULTS_DIR / f"image_alpha_{ts}.png")
    foreground_output.save(fg_path)
    alpha_output.save(alpha_path)
    _last_foreground_path["path"] = fg_path
    _last_alpha_path["path"] = alpha_path
    _last_source_name["name"] = video_state.get("image_name", "image") if video_state.get("image_name", "output.png") != "output.png" else "image"

    yield gr.update(value="✅ Done! Scroll down to view results."), foreground_output, alpha_output

# video matting
def video_matting(video_state, interactive_state, mask_dropdown, erode_kernel_size, dilate_kernel_size, model_selection):
    # Mutable container for progress state accessible from callback
    progress_state = {"text": "⏳ Loading model..."}

    yield gr.update(value=progress_state["text"]), gr.update(), gr.update()

    try:
        selected_model = load_model(model_selection)
    except (FileNotFoundError, ValueError) as e:
        if available_models:
            print(f"Warning: {str(e)}. Using {available_models[0]} instead.")
            selected_model = load_model(available_models[0])
        else:
            raise ValueError("No models are available! Please check if the model files exist.")
    matanyone_processor = InferenceCore(selected_model, cfg=selected_model.cfg)
    if interactive_state["track_end_number"]:
        following_frames = video_state["origin_images"][video_state["select_frame_number"]:interactive_state["track_end_number"]]
    else:
        following_frames = video_state["origin_images"][video_state["select_frame_number"]:]

    if interactive_state["multi_mask"]["masks"]:
        if len(mask_dropdown) == 0:
            mask_dropdown = ["mask_001"]
        mask_dropdown.sort()
        template_mask = interactive_state["multi_mask"]["masks"][int(mask_dropdown[0].split("_")[1]) - 1] * (int(mask_dropdown[0].split("_")[1]))
        for i in range(1,len(mask_dropdown)):
            mask_number = int(mask_dropdown[i].split("_")[1]) - 1
            template_mask = np.clip(template_mask+interactive_state["multi_mask"]["masks"][mask_number]*(mask_number+1), 0, mask_number+1)
        video_state["masks"][video_state["select_frame_number"]]= template_mask
    else:
        template_mask = video_state["masks"][video_state["select_frame_number"]]
    fps = video_state["fps"]
    audio_path = video_state["audio"]

    if len(np.unique(template_mask))==1:
        template_mask[0][0]=1

    num_frames = len(following_frames)
    yield gr.update(value=f"⏳ Running inference on {num_frames} frames..."), gr.update(), gr.update()

    def on_progress(current, total, phase):
        if phase == "warmup":
            progress_state["text"] = f"⏳ Warmup: {current}/{total}"
        else:
            progress_state["text"] = f"⏳ Matting: frame {current}/{total}"

    foreground, alpha = matanyone2(matanyone_processor, following_frames, template_mask*255, r_erode=erode_kernel_size, r_dilate=dilate_kernel_size, progress_callback=on_progress)

    # Release the inference processor — the wrapper already cleared its GPU state,
    # but dropping the reference lets Python GC the object itself
    del matanyone_processor
    clean_vram()

    # Show final inference status
    yield gr.update(value=f"⏳ Inference complete. Preparing outputs..."), gr.update(), gr.update()

    # Upscale to original resolution if we processed at reduced res
    orig_size = video_state.get("original_size")
    orig_images = video_state.get("original_images")
    if orig_images is not None and orig_size is not None:
        orig_h, orig_w = orig_size
        # Get the matching original frames for the range we processed
        if interactive_state["track_end_number"]:
            orig_following = orig_images[video_state["select_frame_number"]:interactive_state["track_end_number"]]
        else:
            orig_following = orig_images[video_state["select_frame_number"]:]

        yield gr.update(value=f"⏳ Upscaling {len(alpha)} frames to original resolution ({orig_w}×{orig_h})..."), gr.update(), gr.update()
        alpha_upscaled = upscale_matte_to_original(alpha, orig_h, orig_w)

        yield gr.update(value=f"⏳ Compositing foreground at original resolution..."), gr.update(), gr.update()
        foreground = composite_at_original_res(orig_following, alpha_upscaled)
        alpha = alpha_upscaled

    yield gr.update(value=f"⏳ Encoding foreground video..."), gr.update(), gr.update()
    # Clean previous results before writing new ones
    _cleanup_results()
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    foreground_output = generate_video_from_frames(foreground, output_path=str(RESULTS_DIR / "{}_fg.mp4".format(video_state["video_name"])), fps=fps, audio_path=audio_path)

    yield gr.update(value=f"⏳ Encoding alpha video..."), gr.update(), gr.update()
    alpha_output = generate_video_from_frames(alpha, output_path=str(RESULTS_DIR / "{}_alpha.mp4".format(video_state["video_name"])), fps=fps, gray2rgb=True, audio_path=audio_path)
    _last_foreground_path["path"] = foreground_output
    _last_alpha_path["path"] = alpha_output
    _last_source_name["name"] = video_state.get("video_name", "video")

    yield gr.update(value="✅ Done! Scroll down to view results."), foreground_output, alpha_output


def add_audio_to_video(video_path, audio_path, output_path):
    try:
        video_input = ffmpeg.input(video_path)
        audio_input = ffmpeg.input(audio_path)
        _ = (
            ffmpeg
            .output(video_input, audio_input, output_path, vcodec="copy", acodec="aac")
            .run(overwrite_output=True, capture_stdout=True, capture_stderr=True)
        )
        return output_path
    except ffmpeg.Error as e:
        print(f"FFmpeg error:\n{e.stderr.decode()}")
        return None


def generate_video_from_frames(frames, output_path, fps=30, gray2rgb=False, audio_path=""):
    frames = np.asarray(frames)
    if gray2rgb and frames.ndim == 3:
        # (N, H, W) grayscale → (N, H, W, 3)
        frames = np.stack([frames] * 3, axis=-1)
    elif gray2rgb and frames.ndim == 4 and frames.shape[-1] == 1:
        frames = np.repeat(frames, 3, axis=3)

    n, h, w = frames.shape[:3]
    # ensure even dimensions for codec compatibility
    h_even = h - (h % 2)
    w_even = w - (w % 2)
    if h_even != h or w_even != w:
        frames = frames[:, :h_even, :w_even]
        h, w = h_even, w_even

    if not os.path.exists(os.path.dirname(output_path)):
        os.makedirs(os.path.dirname(output_path))
    video_temp_path = output_path.replace(".mp4", "_temp.mp4")

    imageio.mimwrite(video_temp_path, frames, fps=fps, quality=7,
                     codec='libx264', macro_block_size=1)

    if audio_path != "" and os.path.exists(audio_path):
        output_path = add_audio_to_video(video_temp_path, audio_path, output_path)
        os.remove(video_temp_path)
        return output_path
    else:
        return video_temp_path

# -------------------------------------------------------------------
# Output file management
# -------------------------------------------------------------------
RESULTS_DIR = Path("./_temp_results")
_last_foreground_path = {"path": None}
_last_alpha_path = {"path": None}
_last_source_name = {"name": None}  # original input media name stem

def _sanitize_stem(name, max_len=40):
    """Extract a filesystem-safe portion of the original media name."""
    stem = Path(name).stem
    # keep only alphanumeric, hyphens, underscores, spaces
    safe = "".join(c for c in stem if c.isalnum() or c in "-_ ")
    safe = safe.strip()[:max_len].rstrip()
    return safe or "untitled"

def _cleanup_results():
    """Remove previous intermediate files from the results dir."""
    if RESULTS_DIR.exists():
        for item in RESULTS_DIR.iterdir():
            try:
                item.unlink()
            except Exception:
                pass

def save_foreground_to_outputs():
    src = _last_foreground_path.get("path")
    if not src or not Path(src).exists():
        return gr.update(value="No foreground output to save.")
    src_path = Path(src)
    ts = time.strftime("%Y%m%d_%H%M%S")
    source_stem = _sanitize_stem(_last_source_name.get("name") or "")
    dst = OUTPUT_DIR / f"{source_stem}_fg_{ts}{src_path.suffix}"
    shutil.copy2(str(src_path), str(dst))
    # Clean up temp file after successful save
    try:
        src_path.unlink()
    except Exception:
        pass
    _last_foreground_path["path"] = None
    return gr.update(value=f"Saved: {dst.name}")

def save_alpha_to_outputs():
    src = _last_alpha_path.get("path")
    if not src or not Path(src).exists():
        return gr.update(value="No alpha output to save.")
    src_path = Path(src)
    ts = time.strftime("%Y%m%d_%H%M%S")
    source_stem = _sanitize_stem(_last_source_name.get("name") or "")
    dst = OUTPUT_DIR / f"{source_stem}_alpha_{ts}{src_path.suffix}"
    shutil.copy2(str(src_path), str(dst))
    # Clean up temp file after successful save
    try:
        src_path.unlink()
    except Exception:
        pass
    _last_alpha_path["path"] = None
    return gr.update(value=f"Saved: {dst.name}")

def open_outputs_folder():
    folder = str(OUTPUT_DIR.resolve())
    try:
        if os.name == "nt":
            os.startfile(folder)
        elif os.name == "posix":
            subprocess.Popen(["xdg-open", folder])
        else:
            subprocess.Popen(["open", folder])
    except Exception:
        pass
    return gr.update(value=f"Opened: {folder}")

def on_clear_temp_now():
    cleared_count = 0
    for temp_dir in [GRADIO_TEMP_DIR, RESULTS_DIR]:
        if temp_dir.exists():
            for item in temp_dir.iterdir():
                try:
                    if item.is_file():
                        item.unlink()
                        cleared_count += 1
                    elif item.is_dir():
                        shutil.rmtree(item)
                        cleared_count += 1
                except Exception:
                    pass
    return f"Cleared {cleared_count} temp items"

# reset all states for a new input
def restart():
    clean_vram()
    _cleanup_results()
    _last_foreground_path["path"] = None
    _last_alpha_path["path"] = None
    _last_source_name["name"] = None
    return {
            "user_name": "",
            "video_name": "",
            "origin_images": None,
            "painted_images": None,
            "masks": None,
            "inpaint_masks": None,
            "logits": None,
            "select_frame_number": 0,
            "fps": 30,
            "original_size": None,
            "original_images": None,
        }, {
            "inference_times": 0,
            "negative_click_times" : 0,
            "positive_click_times": 0,
            "mask_save": args.mask_save,
            "multi_mask": {
                "mask_names": [],
                "masks": []
            },
            "track_end_number": None,
        }, [[],[]], None, None, \
        gr.update(visible=False), gr.update(visible=False), gr.update(visible=False), gr.update(visible=False),\
        gr.update(visible=False), gr.update(visible=False), gr.update(visible=False), gr.update(visible=False), \
        gr.update(visible=False), gr.update(visible=False), gr.update(visible=False), gr.update(visible=False), \
        gr.update(visible=False), gr.update(visible=False, choices=[], value=[]), gr.update(visible=False, value=None), gr.update(visible=False), "Original", \
        gr.update(value="")

# args, defined in track_anything.py
args = parse_augment()

_gpu_name = "N/A"
if torch.cuda.is_available():
    _gpu_name = torch.cuda.get_device_name(0)
elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
    _gpu_name = "Apple MPS"
_print_action("🖥️ ", f"Device: {args.device}  |  GPU: {_gpu_name}")
_print_action("🐍", f"PyTorch {torch.__version__}  |  CUDA {torch.version.cuda or 'N/A'}")
print()
sam_checkpoint_url_dict = {
    'vit_h': "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth",
    'vit_l': "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_l_0b3195.pth",
    'vit_b': "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth"
}

# All paths are now relative to app/ (the CWD when launched)
checkpoint_folder = os.path.join('pretrained_models')

_print_action("🔍", f"Loading SAM model ({args.sam_model_type}) on {args.device}...")
sam_checkpoint = load_file_from_url(sam_checkpoint_url_dict[args.sam_model_type], checkpoint_folder)
# initialize sams
model = MaskGenerator(sam_checkpoint, args)
_print_action("✅", "SAM model ready")

# initialize matanyone - lazy loading
model_display_to_file = {
    "MatAnyone": "matanyone.pth",
    "MatAnyone 2": "matanyone2.pth"
}

model_urls = {
    "matanyone.pth": "https://github.com/pq-yang/MatAnyone/releases/download/v1.0.0/matanyone.pth",
    "matanyone2.pth": "https://github.com/pq-yang/MatAnyone2/releases/download/v1.0.0/matanyone2.pth"
}

model_paths = {
    "matanyone.pth": load_file_from_url(model_urls["matanyone.pth"], checkpoint_folder),
    "matanyone2.pth": load_file_from_url(model_urls["matanyone2.pth"], checkpoint_folder)
}

loaded_models = {}

def load_model(display_name):
    """Load a model if not already loaded"""
    if display_name in model_display_to_file:
        model_file = model_display_to_file[display_name]
    elif display_name in model_paths:
        model_file = display_name
    else:
        raise ValueError(f"Unknown model: {display_name}")

    if model_file in loaded_models:
        return loaded_models[model_file]

    if model_file not in model_paths:
        raise ValueError(f"Unknown model file: {model_file}")

    ckpt_path = model_paths[model_file]
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Model file not found: {ckpt_path}")

    try:
        GlobalHydra.instance().clear()
    except:
        pass

    print(f"Loading model: {display_name} ({model_file})...")
    model = get_matanyone2_model(ckpt_path, args.device)
    model = model.to(args.device).eval()
    loaded_models[model_file] = model
    print(f"Model {display_name} loaded successfully.")
    return model

# Get available model choices for the UI
available_models = []
if "MatAnyone 2" in model_display_to_file:
    file_name = model_display_to_file["MatAnyone 2"]
    if file_name in model_paths and os.path.exists(model_paths[file_name]):
        available_models.append("MatAnyone 2")
if "MatAnyone" in model_display_to_file:
    file_name = model_display_to_file["MatAnyone"]
    if file_name in model_paths and os.path.exists(model_paths[file_name]):
        available_models.append("MatAnyone")

if not available_models:
    raise RuntimeError("No models are available! Please ensure at least one model file exists in pretrained_models/")
default_model = "MatAnyone 2" if "MatAnyone 2" in available_models else available_models[0]
_print_action("🧠", f"Available models: {', '.join(available_models)} (default: {default_model})")

# download test samples
_print_action("📥", "Checking test samples & assets...")
test_sample_path = os.path.join('.', "test_sample/")
load_file_from_url('https://github.com/pq-yang/MatAnyone2/releases/download/media/test-sample-0-1080p.mp4', test_sample_path)
load_file_from_url('https://github.com/pq-yang/MatAnyone2/releases/download/media/test-sample-1-1080p.mp4', test_sample_path)
load_file_from_url('https://github.com/pq-yang/MatAnyone2/releases/download/media/test-sample-2-720p.mp4', test_sample_path)
load_file_from_url('https://github.com/pq-yang/MatAnyone2/releases/download/media/test-sample-3-720p.mp4', test_sample_path)
load_file_from_url('https://github.com/pq-yang/MatAnyone2/releases/download/media/test-sample-4-720p.mp4', test_sample_path)
load_file_from_url('https://github.com/pq-yang/MatAnyone2/releases/download/media/test-sample-5-720p.mp4', test_sample_path)
load_file_from_url('https://github.com/pq-yang/MatAnyone2/releases/download/media/test-sample-0.jpg', test_sample_path)
load_file_from_url('https://github.com/pq-yang/MatAnyone2/releases/download/media/test-sample-1.jpg', test_sample_path)
load_file_from_url('https://github.com/pq-yang/MatAnyone2/releases/download/media/test-sample-2.jpg', test_sample_path)
load_file_from_url('https://github.com/pq-yang/MatAnyone2/releases/download/media/test-sample-3.jpg', test_sample_path)

# download assets
assets_path = os.path.join('.', "assets/")
load_file_from_url('https://github.com/pq-yang/MatAnyone/releases/download/media/tutorial_single_target.mp4', assets_path)
load_file_from_url('https://github.com/pq-yang/MatAnyone/releases/download/media/tutorial_multi_targets.mp4', assets_path)

# documents
title = r"""<div class="multi-layer" align="center"><span>MatAnyone Series</span></div>
"""
description = r"""
<b>Official Gradio demo</b> for <a href='https://github.com/pq-yang/MatAnyone2' target='_blank'><b>MatAnyone 2</b></a> and <a href='https://github.com/pq-yang/MatAnyone' target='_blank'><b>MatAnyone</b></a>.<br>
🔥 MatAnyone series provide practical human video matting framework supporting target assignment.<br>
🧐 <b>We use <u>MatAnyone 2</u> as the default model. You can also choose <u>MatAnyone</u> in "Model Selection".</b><br>
🎪 Try to drop your video/image, assign the target masks with a few clicks, and get the the matting results!<br>
"""
article = r"""<h3>
<b>If our projects are helpful, please help to 🌟 the Github Repo for <a href='https://github.com/pq-yang/MatAnyone2' target='_blank'>MatAnyone 2</a> and <a href='https://github.com/pq-yang/MatAnyone' target='_blank'>MatAnyone</a>. Thanks!</b></h3>

---

📑 **Citation**
<br>
If our work is useful for your research, please consider citing:
```bibtex
@InProceedings{yang2026matanyone2,
      title     = {{MatAnyone 2}: Scaling Video Matting via a Learned Quality Evaluator},
      author    = {Yang, Peiqing and Zhou, Shangchen and Hao, Kai and Tao, Qingyi},
      booktitle = {CVPR},
      year      = {2026}
}

@InProceedings{yang2025matanyone,
     title     = {{MatAnyone}: Stable Video Matting with Consistent Memory Propagation},
     author    = {Yang, Peiqing and Zhou, Shangchen and Zhao, Jixin and Tao, Qingyi and Loy, Chen Change},
     booktitle = {arXiv preprint arXiv:2501.14677},
     year      = {2025}
}
```
📝 **License**
<br>
This project is licensed under <a rel="license" href="https://github.com/pq-yang/MatAnyone/blob/main/LICENSE">S-Lab License 1.0</a>.
Redistribution and use for non-commercial purposes should follow this license.
<br>
📧 **Contact**
<br>
If you have any questions, please feel free to reach me out at <b>peiqingyang99@outlook.com</b>.
<br>
👏 **Acknowledgement**
<br>
This project is built upon [Cutie](https://github.com/hkchengrex/Cutie), with the interactive demo adapted from [ProPainter](https://github.com/sczhou/ProPainter), leveraging segmentation capabilities from [Segment Anything](https://github.com/facebookresearch/segment-anything). Thanks for their awesome works!
"""

my_custom_css = """
.gradio-container {width: 85% !important; margin: 0 auto;}
.gr-monochrome-group {border-radius: 5px !important; border: revert-layer !important; border-width: 2px !important; color: black !important}
button {border-radius: 8px !important;}
.new_button {
  background: linear-gradient(135deg, rgba(30,30,30,0.9), rgba(50,50,50,0.7)) !important;
  color: #e0e0e0 !important;
  border: 1px solid rgba(255,255,255,0.08) !important;
  backdrop-filter: blur(12px) !important;
  box-shadow: 0 2px 8px rgba(0,0,0,0.3), inset 0 1px 0 rgba(255,255,255,0.06) !important;
  transition: all 0.25s ease !important;
}
.green_button {
  background: linear-gradient(135deg, rgba(56,142,60,0.9), rgba(76,175,80,0.75)) !important;
  color: #fff !important;
  border: 1px solid rgba(255,255,255,0.12) !important;
  backdrop-filter: blur(12px) !important;
  box-shadow: 0 2px 12px rgba(76,175,80,0.3), inset 0 1px 0 rgba(255,255,255,0.15) !important;
  text-shadow: 0 1px 2px rgba(0,0,0,0.2) !important;
  font-weight: 600 !important;
  transition: all 0.25s ease !important;
}
.amber_button {
  background: linear-gradient(135deg, rgba(180,95,6,0.95), rgba(210,120,20,0.8)) !important;
  color: #fff !important;
  border: 1px solid rgba(255,255,255,0.1) !important;
  backdrop-filter: blur(12px) !important;
  box-shadow: 0 2px 12px rgba(180,95,6,0.35), inset 0 1px 0 rgba(255,255,255,0.12) !important;
  text-shadow: 0 1px 2px rgba(0,0,0,0.25) !important;
  font-weight: 600 !important;
  transition: all 0.25s ease !important;
}
.new_button:hover {
  background: linear-gradient(135deg, rgba(55,55,55,0.95), rgba(75,75,75,0.8)) !important;
  box-shadow: 0 4px 16px rgba(0,0,0,0.4), inset 0 1px 0 rgba(255,255,255,0.1) !important;
  transform: translateY(-1px) !important;
}
.green_button:hover {
  background: linear-gradient(135deg, rgba(76,175,80,0.95), rgba(102,200,106,0.85)) !important;
  box-shadow: 0 4px 20px rgba(76,175,80,0.45), inset 0 1px 0 rgba(255,255,255,0.2) !important;
  transform: translateY(-1px) !important;
}
.amber_button:hover {
  background: linear-gradient(135deg, rgba(200,110,10,1), rgba(230,140,30,0.9)) !important;
  box-shadow: 0 4px 20px rgba(200,110,10,0.5), inset 0 1px 0 rgba(255,255,255,0.15) !important;
  transform: translateY(-1px) !important;
}
.mask_button_group {gap: 10px !important;}
.video .wrap.svelte-lcpz3o {display: flex !important; align-items: center !important; justify-content: center !important; height: auto !important; max-height: 300px !important;}
.video .wrap.svelte-lcpz3o > :first-child {height: auto !important; width: 100% !important; object-fit: contain !important;}
.video .container.svelte-sxyn79 {display: none !important;}
.margin_center {width: 50% !important; margin: auto !important;}
.jc_center {justify-content: center !important;}
.video-title {margin-bottom: 5px !important;}
.custom-bg {background-color: #f0f0f0; padding: 10px; border-radius: 10px;}
.video-window video, .image-window img {
    max-height: 60vh !important;
    object-fit: contain;
    width: 100%;
}
/* Tighten vertical gaps: reduce space between video/image and buttons below */
.video-window, .image-window, .image {
    margin-bottom: 0 !important;
}
/* Reduce gap between info badge / status textbox and the HR divider */
.info-badge, .status-textbox {
    margin-top: 2px !important;
    margin-bottom: 2px !important;
}
.info-badge > div {
    padding: 4px 16px !important;
}
/* Align the status textbox to match the info badge styling */
.status-textbox textarea {
    padding: 8px 16px !important;
    font-size: 0.85em !important;
    min-height: unset !important;
}
<style>
@import url('https://fonts.googleapis.com/css2?family=Sarpanch:wght@400;500;600;700;800;900&family=Sen:wght@400..800&family=Sixtyfour+Convergence&family=Stardos+Stencil:wght@400;700&display=swap');
body {display: flex; justify-content: center; align-items: center; height: 100vh; margin: 0; background-color: #0d1117; font-family: Arial, sans-serif; font-size: 18px;}
.title-container {text-align: center; padding: 0; margin: 0; height: 5vh; width: 80vw; font-family: "Sarpanch", sans-serif; font-weight: 60;}
#custom-markdown {font-family: "Roboto", sans-serif; font-size: 18px; color: #333333; font-weight: bold;}
small {font-size: 60%;}
</style>

"""

with gr.Blocks(theme=get_theme_from_settings(), css=my_custom_css) as demo:

    with gr.Tabs():
        # ===== VIDEO TAB =====
        with gr.TabItem("Video"):
            click_state = gr.State([[],[]])
            interactive_state = gr.State({
                "inference_times": 0, "negative_click_times": 0, "positive_click_times": 0,
                "mask_save": args.mask_save,
                "multi_mask": {"mask_names": [], "masks": []},
                "track_end_number": None,
            })
            video_state = gr.State({
                "user_name": "", "video_name": "", "origin_images": None, "painted_images": None,
                "masks": None, "inpaint_masks": None, "logits": None,
                "select_frame_number": 0, "fps": 30, "audio": "",
                "original_size": None, "original_images": None,
            })

            with gr.Group(elem_classes="gr-monochrome-group", visible=True):
                with gr.Row():
                    model_selection = gr.Radio(choices=available_models, value=default_model, label="Model Selection", info="Choose the model to use for matting", interactive=True)
                    video_resize_preset = gr.Dropdown(
                        choices=list(RESOLUTION_PRESETS.keys()), value="Original",
                        label="Processing Resolution",
                        info="Lower res = less VRAM. Alpha is upscaled back to original for output.",
                        interactive=True, scale=1)
                with gr.Row():
                    with gr.Accordion('Model Settings (click to expand)', open=False):
                        with gr.Row():
                            erode_kernel_size = gr.Slider(label='Erode Kernel Size', minimum=0, maximum=30, step=1, value=10, info="Erosion on the added mask", interactive=True)
                            dilate_kernel_size = gr.Slider(label='Dilate Kernel Size', minimum=0, maximum=30, step=1, value=10, info="Dilation on the added mask", interactive=True)
                        with gr.Row():
                            image_selection_slider = gr.Slider(minimum=1, maximum=100, step=1, value=1, label="Start Frame", info="Choose the start frame for target assignment and video matting", visible=False)
                            track_pause_number_slider = gr.Slider(minimum=1, maximum=100, step=1, value=1, label="Track end frame", visible=False)
                        with gr.Row():
                            point_prompt = gr.Radio(choices=["Positive", "Negative"], value="Positive", label="Point Prompt", info="Click to add positive or negative point for target mask", interactive=True, visible=False, min_width=100, scale=1)
                            mask_dropdown = gr.Dropdown(multiselect=True, value=[], label="Mask Selection", info="Choose 1~all mask(s) added in Step 2", visible=False)

            gr.HTML('<hr style="border: none; height: 1.5px; background: linear-gradient(to right, #a566b4, #74a781);margin: 5px 0;">')

            with gr.Column():
                with gr.Row():
                    with gr.Column(scale=2):
                        gr.Markdown("## Step1: Upload video")
                    with gr.Column(scale=2):
                        step2_title = gr.Markdown("## Step2: Add mask(s) <small>(For multiple masked objects use **`Add Mask`** for each)</small>", visible=False)
                with gr.Row():
                    with gr.Column(scale=2):
                        video_input = gr.Video(label="Input Video", elem_classes="video-window")
                        extract_frames_button = gr.Button(value="Load Video", interactive=True, elem_classes="amber_button")
                        video_info = gr.HTML(visible=False, elem_classes="info-badge")
                    with gr.Column(scale=2):
                        template_frame = gr.Image(label="Start Frame", type="pil", interactive=True, elem_id="template_frame", visible=False, elem_classes="image-window")
                        with gr.Row(elem_classes="mask_button_group"):
                            clear_button_click = gr.Button(value="Clear Clicks", interactive=True, visible=False, elem_classes="new_button", min_width=100)
                            add_mask_button = gr.Button(value="Add Mask", interactive=True, visible=False, elem_classes="new_button", min_width=100)
                            remove_mask_button = gr.Button(value="Remove Mask", interactive=True, visible=False, elem_classes="new_button", min_width=100)
                            matting_button = gr.Button(value="Video Matting", interactive=True, visible=False, elem_classes="green_button", min_width=100)
                        with gr.Row():                            
                            video_processing_status = gr.Textbox(label="", show_label=False, container=False, interactive=False, max_lines=2, visible=True, elem_classes="status-textbox",
                                placeholder="Upload a video → click Load Video → click frame to place mask points → Add Mask → Video Matting. Use Processing Resolution to reduce VRAM.")
                            
                gr.HTML('<hr style="border: none; height: 1.5px; background: linear-gradient(to right, #a566b4, #74a781);margin: 5px 0;">')

                with gr.Row():
                    with gr.Column(scale=2):
                        foreground_video_output = gr.Video(label="Foreground Output", visible=False, elem_classes="video-window")
                        foreground_output_button = gr.Button(value="Foreground Output", visible=False, elem_classes="new_button")
                    with gr.Column(scale=2):
                        alpha_video_output = gr.Video(label="Alpha Output", visible=False, elem_classes="video-window")
                        alpha_output_button = gr.Button(value="Alpha Mask Output", visible=False, elem_classes="new_button")

                with gr.Row():
                    save_fg_video_btn = gr.Button("💾 Save Foreground", elem_classes="new_button", min_width=120)
                    save_alpha_video_btn = gr.Button("💾 Save Alpha", elem_classes="new_button", min_width=120)
                    open_folder_video_btn = gr.Button("📂 Open Output Folder", elem_classes="new_button", min_width=120)
                save_video_status = gr.Textbox(label="", show_label=False, interactive=False, visible=True)

            # Video tab event bindings
            extract_frames_button.click(fn=get_frames_from_video, inputs=[video_input, video_state, video_resize_preset],
                outputs=[video_processing_status, video_state, video_info, template_frame, image_selection_slider, track_pause_number_slider, point_prompt, clear_button_click, add_mask_button, matting_button, template_frame, foreground_video_output, alpha_video_output, foreground_output_button, alpha_output_button, mask_dropdown, step2_title])
            image_selection_slider.release(fn=select_video_template, inputs=[image_selection_slider, video_state, interactive_state], outputs=[template_frame, video_state, interactive_state], api_name="select_image")
            track_pause_number_slider.release(fn=get_end_number, inputs=[track_pause_number_slider, video_state, interactive_state], outputs=[template_frame, interactive_state], api_name="end_image")
            template_frame.select(fn=sam_refine, inputs=[video_state, point_prompt, click_state, interactive_state], outputs=[template_frame, video_state, interactive_state])
            add_mask_button.click(fn=add_multi_mask, inputs=[video_state, interactive_state, mask_dropdown], outputs=[interactive_state, mask_dropdown, template_frame, click_state])
            remove_mask_button.click(fn=remove_multi_mask, inputs=[interactive_state, mask_dropdown], outputs=[interactive_state, mask_dropdown])
            matting_button.click(fn=video_matting, inputs=[video_state, interactive_state, mask_dropdown, erode_kernel_size, dilate_kernel_size, model_selection], outputs=[video_processing_status, foreground_video_output, alpha_video_output])
            mask_dropdown.change(fn=show_mask, inputs=[video_state, interactive_state, mask_dropdown], outputs=[template_frame])

            video_input.change(fn=restart, inputs=[], outputs=[video_state, interactive_state, click_state, foreground_video_output, alpha_video_output, template_frame, image_selection_slider, track_pause_number_slider, point_prompt, clear_button_click, add_mask_button, matting_button, template_frame, foreground_video_output, alpha_video_output, remove_mask_button, foreground_output_button, alpha_output_button, mask_dropdown, video_info, step2_title, video_resize_preset, video_processing_status], queue=False, show_progress=False)
            video_input.clear(fn=restart, inputs=[], outputs=[video_state, interactive_state, click_state, foreground_video_output, alpha_video_output, template_frame, image_selection_slider, track_pause_number_slider, point_prompt, clear_button_click, add_mask_button, matting_button, template_frame, foreground_video_output, alpha_video_output, remove_mask_button, foreground_output_button, alpha_output_button, mask_dropdown, video_info, step2_title, video_resize_preset, video_processing_status], queue=False, show_progress=False)
            clear_button_click.click(fn=clear_click, inputs=[video_state, click_state], outputs=[template_frame, click_state])

            # Save/open buttons
            save_fg_video_btn.click(fn=save_foreground_to_outputs, outputs=[save_video_status])
            save_alpha_video_btn.click(fn=save_alpha_to_outputs, outputs=[save_video_status])
            open_folder_video_btn.click(fn=open_outputs_folder, outputs=[save_video_status])

            gr.HTML('<hr style="border: none; height: 1.5px; background: linear-gradient(to right, #a566b4, #74a781);margin: 5px 0;">')
            with gr.Accordion("📂 Examples & Tutorials", open=False):
                gr.Examples(
                    examples=[os.path.join(os.path.dirname(__file__), "test_sample", test_sample) for test_sample in ["test-sample-0-1080p.mp4", "test-sample-1-1080p.mp4", "test-sample-2-720p.mp4", "test-sample-3-720p.mp4", "test-sample-4-720p.mp4", "test-sample-5-720p.mp4"]],
                    inputs=[video_input],
                )
                gr.Markdown("---")
                gr.Markdown("### Video Tutorials")
                with gr.Row():
                    with gr.Column():
                        gr.Markdown("#### Single Target")
                        gr.Video(value="./assets/tutorial_single_target.mp4", elem_classes="video")
                    with gr.Column():
                        gr.Markdown("#### Multiple Targets")
                        gr.Video(value="./assets/tutorial_multi_targets.mp4", elem_classes="video")

        # ===== IMAGE TAB =====
        with gr.TabItem("Image"):
            click_state = gr.State([[],[]])
            interactive_state = gr.State({
                "inference_times": 0, "negative_click_times": 0, "positive_click_times": 0,
                "mask_save": args.mask_save,
                "multi_mask": {"mask_names": [], "masks": []},
                "track_end_number": None,
            })
            image_state = gr.State({
                "user_name": "", "image_name": "", "origin_images": None, "painted_images": None,
                "masks": None, "inpaint_masks": None, "logits": None,
                "select_frame_number": 0, "fps": 30,
                "original_size": None, "original_images": None,
            })

            with gr.Group(elem_classes="gr-monochrome-group", visible=True):
                with gr.Row():
                    model_selection = gr.Radio(choices=available_models, value=default_model, label="Model Selection", info="Choose the model to use for matting", interactive=True)
                    image_resize_preset = gr.Dropdown(
                        choices=list(RESOLUTION_PRESETS.keys()), value="Original",
                        label="Processing Resolution",
                        info="Lower res = less VRAM. Alpha is upscaled back to original for output.",
                        interactive=True, scale=1)
                with gr.Row():
                    with gr.Accordion('Model Settings (click to expand)', open=False):
                        with gr.Row():
                            erode_kernel_size = gr.Slider(label='Erode Kernel Size', minimum=0, maximum=30, step=1, value=10, info="Erosion on the added mask", interactive=True)
                            dilate_kernel_size = gr.Slider(label='Dilate Kernel Size', minimum=0, maximum=30, step=1, value=10, info="Dilation on the added mask", interactive=True)
                        with gr.Row():
                            image_selection_slider = gr.Slider(minimum=1, maximum=100, step=1, value=1, label="Num of Refinement Iterations", info="More iterations → More details & More time", visible=False)
                            track_pause_number_slider = gr.Slider(minimum=1, maximum=100, step=1, value=1, label="Track end frame", visible=False)
                        with gr.Row():
                            point_prompt = gr.Radio(choices=["Positive", "Negative"], value="Positive", label="Point Prompt", info="Click to add positive or negative point for target mask", interactive=True, visible=False, min_width=100, scale=1)
                            mask_dropdown = gr.Dropdown(multiselect=True, value=[], label="Mask Selection", info="Choose 1~all mask(s) added in Step 2", visible=False)

            gr.HTML('<hr style="border: none; height: 1.5px; background: linear-gradient(to right, #a566b4, #74a781);margin: 5px 0;">')

            with gr.Column():
                with gr.Row():
                    with gr.Column(scale=2):
                        gr.Markdown("## Step1: Upload image")
                    with gr.Column(scale=2):
                        step2_title = gr.Markdown("## Step2: Add mask(s) <small>(For multiple masked objects use **`Add Mask`** for each)</small>", visible=False)
                with gr.Row():
                    with gr.Column(scale=2):
                        image_input = gr.Image(label="Input Image", elem_classes="image")
                        extract_frames_button = gr.Button(value="Load Image", interactive=True, elem_classes="amber_button")
                        image_info = gr.HTML(visible=False, elem_classes="info-badge")
                    with gr.Column(scale=2):
                        template_frame = gr.Image(type="pil", label="Start Frame", interactive=True, elem_id="template_frame", visible=False, elem_classes="image")
                        with gr.Row(elem_classes="mask_button_group"):
                            clear_button_click = gr.Button(value="Clear Clicks", interactive=True, visible=False, elem_classes="new_button", min_width=100)
                            add_mask_button = gr.Button(value="Add Mask", interactive=True, visible=False, elem_classes="new_button", min_width=100)
                            remove_mask_button = gr.Button(value="Remove Mask", interactive=True, visible=False, elem_classes="new_button", min_width=100)
                            matting_button = gr.Button(value="Image Matting", interactive=True, visible=False, elem_classes="green_button", min_width=100)
                        with gr.Row():                            
                            image_processing_status = gr.Textbox(label="", show_label=False, container=False, interactive=False, max_lines=2, visible=True, elem_classes="status-textbox",
                                placeholder="Upload an image → click Load Image → click image to place mask points → Add Mask → Image Matting. Use Processing Resolution to reduce VRAM.")

                gr.HTML('<hr style="border: none; height: 1.5px; background: linear-gradient(to right, #a566b4, #74a781);margin: 5px 0;">')

                with gr.Row():
                    with gr.Column(scale=2):
                        foreground_image_output = gr.Image(type="pil", label="Foreground Output", visible=False, elem_classes="image")
                        foreground_output_button = gr.Button(value="Foreground Output", visible=False, elem_classes="new_button")
                    with gr.Column(scale=2):
                        alpha_image_output = gr.Image(type="pil", label="Alpha Output", visible=False, elem_classes="image")
                        alpha_output_button = gr.Button(value="Alpha Mask Output", visible=False, elem_classes="new_button")

                with gr.Row():
                    save_fg_image_btn = gr.Button("💾 Save Foreground", elem_classes="new_button", min_width=120)
                    save_alpha_image_btn = gr.Button("💾 Save Alpha", elem_classes="new_button", min_width=120)
                    open_folder_image_btn = gr.Button("📂 Open Output Folder", elem_classes="new_button", min_width=120)

                save_image_status = gr.Textbox(label="", show_label=False, container=False, interactive=False, lines=1, visible=True)

            # Image tab event bindings
            extract_frames_button.click(fn=get_frames_from_image, inputs=[image_input, image_state, image_resize_preset],
                outputs=[image_processing_status, image_state, image_info, template_frame, image_selection_slider, track_pause_number_slider, point_prompt, clear_button_click, add_mask_button, matting_button, template_frame, foreground_image_output, alpha_image_output, foreground_output_button, alpha_output_button, mask_dropdown, step2_title])
            image_selection_slider.release(fn=select_image_template, inputs=[image_selection_slider, image_state, interactive_state], outputs=[template_frame, image_state, interactive_state], api_name="select_image")
            track_pause_number_slider.release(fn=get_end_number, inputs=[track_pause_number_slider, image_state, interactive_state], outputs=[template_frame, interactive_state], api_name="end_image")
            template_frame.select(fn=sam_refine, inputs=[image_state, point_prompt, click_state, interactive_state], outputs=[template_frame, image_state, interactive_state])
            add_mask_button.click(fn=add_multi_mask, inputs=[image_state, interactive_state, mask_dropdown], outputs=[interactive_state, mask_dropdown, template_frame, click_state])
            remove_mask_button.click(fn=remove_multi_mask, inputs=[interactive_state, mask_dropdown], outputs=[interactive_state, mask_dropdown])
            matting_button.click(fn=image_matting, inputs=[image_state, interactive_state, mask_dropdown, erode_kernel_size, dilate_kernel_size, image_selection_slider, model_selection], outputs=[image_processing_status, foreground_image_output, alpha_image_output])
            mask_dropdown.change(fn=show_mask, inputs=[image_state, interactive_state, mask_dropdown], outputs=[template_frame])

            image_input.change(fn=restart, inputs=[], outputs=[image_state, interactive_state, click_state, foreground_image_output, alpha_image_output, template_frame, image_selection_slider, track_pause_number_slider, point_prompt, clear_button_click, add_mask_button, matting_button, template_frame, foreground_image_output, alpha_image_output, remove_mask_button, foreground_output_button, alpha_output_button, mask_dropdown, image_info, step2_title, image_resize_preset, image_processing_status], queue=False, show_progress=False)
            image_input.clear(fn=restart, inputs=[], outputs=[image_state, interactive_state, click_state, foreground_image_output, alpha_image_output, template_frame, image_selection_slider, track_pause_number_slider, point_prompt, clear_button_click, add_mask_button, matting_button, template_frame, foreground_image_output, alpha_image_output, remove_mask_button, foreground_output_button, alpha_output_button, mask_dropdown, image_info, step2_title, image_resize_preset, image_processing_status], queue=False, show_progress=False)
            clear_button_click.click(fn=clear_click, inputs=[image_state, click_state], outputs=[template_frame, click_state])

            # Save/open buttons
            save_fg_image_btn.click(fn=save_foreground_to_outputs, outputs=[save_image_status])
            save_alpha_image_btn.click(fn=save_alpha_to_outputs, outputs=[save_image_status])
            open_folder_image_btn.click(fn=open_outputs_folder, outputs=[save_image_status])

            gr.HTML('<hr style="border: none; height: 1.5px; background: linear-gradient(to right, #a566b4, #74a781);margin: 5px 0;">')
            with gr.Accordion("📂 Example Images", open=False):
                gr.Examples(
                    examples=[os.path.join(os.path.dirname(__file__), "test_sample", test_sample) for test_sample in ["test-sample-0.jpg", "test-sample-1.jpg", "test-sample-2.jpg", "test-sample-3.jpg"]],
                    inputs=[image_input],
                )

        # ===== SETTINGS TAB =====
        with gr.TabItem("⚙ Settings"):
            gr.Markdown("### Settings")

            current_settings = load_ui_settings()

            with gr.Group():
                current_theme = current_settings.get("theme", "Monochrome")
                theme_choices = [
                    "Default", "Soft", "Monochrome", "Glass", "Base", "Ocean", "Origin", "Citrus",
                    "Miku", "Interstellar", "xkcd", "kotaemon"
                ]
                theme_dropdown = gr.Dropdown(
                    choices=theme_choices, value=current_theme, label="Theme",
                    info="Select a theme for the interface. Changes apply on next app startup.",
                )

            gr.Markdown("---")

            with gr.Accordion("File Management", open=True):
                with gr.Group():
                    current_output_dir = current_settings.get("output_dir", "./outputs")
                    output_dir_textbox = gr.Textbox(
                        label="Output Directory", value=current_output_dir,
                        info="Custom location for saved files. Requires app restart.",
                        placeholder="./outputs"
                    )

                gr.Markdown("---")

                with gr.Group():
                    current_clear_temp = current_settings.get("clear_temp_on_start", False)
                    clear_temp_on_start_checkbox = gr.Checkbox(
                        label="Clear temp files on app start", value=current_clear_temp,
                        info="Automatically clears Gradio temporary files when the app starts.",
                    )
                    clear_temp_now_btn = gr.Button("Clear Temp Files Now", variant="secondary")

            with gr.Row():
                clear_temp_status = gr.Textbox(label="", show_label=False, interactive=False)

            # Settings event wiring
            def on_theme_change(theme_name):
                settings = load_ui_settings()
                settings["theme"] = theme_name
                save_ui_settings(settings)

            def on_output_dir_change(dir_value):
                settings = load_ui_settings()
                settings["output_dir"] = dir_value
                save_ui_settings(settings)

            def on_clear_temp_on_start_toggle(value):
                settings = load_ui_settings()
                settings["clear_temp_on_start"] = value
                save_ui_settings(settings)

            theme_dropdown.change(on_theme_change, inputs=[theme_dropdown], outputs=[])
            output_dir_textbox.change(on_output_dir_change, inputs=[output_dir_textbox], outputs=[])
            clear_temp_on_start_checkbox.change(on_clear_temp_on_start_toggle, inputs=[clear_temp_on_start_checkbox], outputs=[])
            clear_temp_now_btn.click(on_clear_temp_now, outputs=[clear_temp_status])

    with gr.Accordion("📝 Original Project's Citation & Credits", open=False):
        gr.Markdown(article)

demo.queue()
_print_ready(args.port)
demo.launch(debug=True, share=False, server_name="127.0.0.1", server_port=args.port,
            allowed_paths=[str(OUTPUT_DIR), str(GRADIO_TEMP_DIR), "./_temp_results", "./test_sample", "./assets"])
