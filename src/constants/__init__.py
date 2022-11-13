import enum
import re

from typing import Any

from discord import app_commands


DOCARRAY_STORAGE_FOLDER = '../image_docarrays'
IMAGE_STORAGE_FOLDER = '../images'
TEMP_JSON_STORAGE_FOLDER = '../temp_json'

MANUAL_LINK = 'https://github.com/AmericanPresidentJimmyCarter/yasd-discord-bot/tree/master/manual#readme'

IMAGETOOL_MODULE_NAME = 'imagetool'

REGEX_FOR_ID = re.compile('([0-9a-zA-Z]){12}$')
ID_LENGTH = 12

BUTTON_STORE_FOUR_IMAGES_BUTTONS_KEY = 'four_img_views'
BUTTON_STORE_ONE_IMAGE_BUTTONS_KEY = 'one_img_views'

JSON_IMAGE_TOOL_INPUT_FILE_FN = lambda uid, nonce: f'../temp_json/request-{uid}_{nonce}.json'
JSON_IMAGE_TOOL_OUTPUT_FILE_FN = lambda uid, nonce: f'../temp_json/output-{uid}_{nonce}.json'

DISCORD_EMBED_MAX_LENGTH = 1024
IMAGE_LOCATION_FN = lambda sid: f'../images/{sid}.png'

DEFAULT_IMAGE_HEIGHT_WIDTH = 512
MIN_ITERATIONS = 1
MAX_ITERATIONS = 16
MIN_SCALE = 1.0
MAX_SCALE = 50.0
MAX_SEED = 2 ** 32 - 1
MIN_STEPS = 10
MAX_STEPS = 250
MIN_STRENGTH = 0.01
MIN_STRENGTH_INTERPOLATE = 0.50
MAX_STRENGTH = 0.99
MAX_UPSCALE_SIZE = 768
NUM_IMAGES_MAX = 9


CLIP_TOKENIZER_MERGES_FN = '../clip_vit_large_patch14/merges.txt'
CLIP_TOKENIZER_VOCAB_FN = '../clip_vit_large_patch14/vocab.json'
DOCARRAY_LOCATION_FN = lambda docarray_id: f'../image_docarrays/{docarray_id}.bin'
ID_LENGTH = 12
IMAGE_LOCATION_FN = lambda sid: f'../images/{sid}.png'
IMAGE_LOCATION_FN_JPG = lambda sid: f'../images/{sid}.jpg'
MAX_MODEL_CLIP_TOKENS_PER_PROMPT = 77
MIN_IMAGE_HEIGHT_WIDTH = 384
REGEX_FOR_TAGS = re.compile('<.*?>')
SD_CONCEPTS_URL_FN = lambda concept: f'https://huggingface.co/sd-concepts-library/{concept}/resolve/main/'
VALID_TAG_CONCEPTS: dict[str, Any] = {}


UPSCALER_SWINIR = 'swinir'
UPSCALER_REALESRGAN_4X = 'resrgan_4x'
UPSCALER_REALESRGAN_4X_FACE = 'resrgan_4x_face'
UPSCALER_REALESRGAN_4X_ANIME = 'resrgan_4x_anime'
UPSCALER_STABLE_1 = 'sd_based_upscaler_1'
UPSCALER_STABLE_2 = 'sd_based_upscaler_2'
UPSCALER_STABLE_3 = 'sd_based_upscaler_3'
UPSCALER_STABLE_4 = 'sd_based_upscaler_4'
UPSCALER_STABLE_5 = 'sd_based_upscaler_5'
UPSCALER_NONE = 'no_upscale'

DEFAULT_SD_UPSCALE_SAMPLER = 'dpmpp_2m'
DEFAULT_SD_UPSCALE_SCALE = 7.5
DEFAULT_SD_UPSCALE_STEPS = 35
DEFAULT_SD_UPSCALE_STRENGTH = 0.2


class RealESRGANModels(str, enum.Enum):
    RealESRGAN_x4plus = 'RealESRGAN_x4plus'
    RealESRNet_x4plus = 'RealESRNet_x4plus'
    RealESRGAN_x4plus_anime_6B = 'RealESRGAN_x4plus_anime_6B'
    RealESRGAN_x2plus = 'RealESRGAN_x2plus'
    RealESR_animevideov3 = 'realesr-animevideov3'
    RealESR_general_x4v3 = 'realesr-general-x4v3'


class OutpaintingModes(str, enum.Enum):
    OUTPAINT_25_ALL = 'outpaint_25'
    OUTPAINT_25_LEFT = 'outpaint_25_l'
    OUTPAINT_25_RIGHT = 'outpaint_25_r'
    OUTPAINT_25_UP = 'outpaint_25_u'
    OUTPAINT_25_DOWN = 'outpaint_25_d'


OUTPAINT_CHOICES = [
    app_commands.Choice(name="25% all sides",
        value=OutpaintingModes.OUTPAINT_25_ALL),
    app_commands.Choice(name="Left 25%",
        value=OutpaintingModes.OUTPAINT_25_LEFT),
    app_commands.Choice(name="Right 25%",
        value=OutpaintingModes.OUTPAINT_25_RIGHT),
    app_commands.Choice(name="Up 25%",
        value=OutpaintingModes.OUTPAINT_25_UP),
    app_commands.Choice(name="Down 25%",
        value=OutpaintingModes.OUTPAINT_25_DOWN),
]


SAMPLER_CHOICES = [
    app_commands.Choice(name="k_lms", value="k_lms"),
    app_commands.Choice(name="dpm2", value="dpm2"),
    app_commands.Choice(name="dpm2_ancestral", value="dpm2_ancestral"),
    app_commands.Choice(name="heun", value="heun"),
    app_commands.Choice(name="euler", value="euler"),
    app_commands.Choice(name="euler_ancestral", value="euler_ancestral"),
    app_commands.Choice(name="dpm_fast", value="dpm_fast"),
    app_commands.Choice(name="dpm_adaptive", value="dpm_adaptive"),
    app_commands.Choice(name="dpmpp_2s_ancestral", value="dpmpp_2s_ancestral"),
    app_commands.Choice(name="dpmpp_2m", value="dpmpp_2m"),
]
VALID_SAMPLERS = ['k_lms', 'dpm2', 'dpm2_ancestral', 'heun', 'euler',
    'euler_ancestral', 'dpm_fast', 'dpm_adaptive', 'dpmpp_2s_ancestral',
    'dpmpp_2m']


UPSCALER_CHOICES = [
    app_commands.Choice(name="SwinIR (default, photos and art)",
        value=UPSCALER_SWINIR),
    app_commands.Choice(name="RealESRGAN (photos and art)",
        value=UPSCALER_REALESRGAN_4X),
    app_commands.Choice(name="RealESRGAN Face-Fixing (photos)",
        value=UPSCALER_REALESRGAN_4X_FACE),
    app_commands.Choice(name="RealESRGAN Anime (line art and anime)",
        value=UPSCALER_REALESRGAN_4X_ANIME),
    app_commands.Choice(name="Diffusion Upscale (0.1 strength)",
        value=UPSCALER_STABLE_1),
    app_commands.Choice(name="Diffusion Upscale (0.2 strength)",
        value=UPSCALER_STABLE_2),
    app_commands.Choice(name="Diffusion Upscale (0.3 strength)",
        value=UPSCALER_STABLE_3),
    app_commands.Choice(name="Diffusion Upscale (0.4 strength)",
        value=UPSCALER_STABLE_4),
    app_commands.Choice(name="Diffusion Upscale (0.5 strength)",
        value=UPSCALER_STABLE_5),
    app_commands.Choice(name="No Upscale (return original image)",
        value=UPSCALER_NONE),
]
