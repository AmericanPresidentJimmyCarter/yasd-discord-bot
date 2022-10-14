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

JSON_IMAGE_TOOL_INPUT_FILE_FN = lambda uid, nonce: f'../temp_json/request-{uid}_{nonce}.json'
JSON_IMAGE_TOOL_OUTPUT_FILE_FN = lambda uid, nonce: f'../temp_json/output-{uid}_{nonce}.json'

DISCORD_EMBED_MAX_LENGTH = 1024
IMAGE_LOCATION_FN = lambda sid: f'../images/{sid}.png'

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
NUM_IMAGES_MAX = 9

VALID_IMAGE_HEIGHT_WIDTH = { 384, 416, 448, 480, 512, 544, 576, 608, 640, 672,
    704, 736, 768 }

UPSCALER_SWINIR = 'swinir'
UPSCALER_REALESRGAN_4X = 'resrgan_4x'
UPSCALER_REALESRGAN_4X_FACE = 'resrgan_4x_face'
UPSCALER_REALESRGAN_4X_ANIME = 'resrgan_4x_anime'
UPSCALER_NONE = 'no_upscale'


CLIP_TOKENIZER_MERGES_FN = '../clip_vit_large_patch14/merges.txt'
CLIP_TOKENIZER_VOCAB_FN = '../clip_vit_large_patch14/vocab.json'
DOCARRAY_LOCATION_FN = lambda docarray_id: f'../image_docarrays/{docarray_id}.bin'
ID_LENGTH = 12
IMAGE_LOCATION_FN = lambda sid: f'../images/{sid}.png'
MAX_IMAGE_HEIGHT_WIDTH = 768
MAX_MODEL_CLIP_TOKENS_PER_PROMPT = 77
MIN_IMAGE_HEIGHT_WIDTH = 384
REGEX_FOR_TAGS = re.compile('<.*?>')
SD_CONCEPTS_URL_FN = lambda concept: f'https://huggingface.co/sd-concepts-library/{concept}/resolve/main/'
VALID_TAG_CONCEPTS: dict[str, Any] = {}


class RESRGAN_MODELS(str, enum.Enum):
    RealESRGAN_x4plus = 'RealESRGAN_x4plus'
    RealESRNet_x4plus = 'RealESRNet_x4plus'
    RealESRGAN_x4plus_anime_6B = 'RealESRGAN_x4plus_anime_6B'
    RealESRGAN_x2plus = 'RealESRGAN_x2plus'
    RealESR_animevideov3 = 'realesr-animevideov3'
    RealESR_general_x4v3 = 'realesr-general-x4v3'


UPSCALER_SWINIR = 'swinir'
UPSCALER_REALESRGAN_4X = 'resrgan_4x'
UPSCALER_REALESRGAN_4X_FACE = 'resrgan_4x_face'
UPSCALER_REALESRGAN_4X_ANIME = 'resrgan_4x_anime'
UPSCALER_NONE = 'no_upscale'


HEIGHT_AND_WIDTH_CHOICES = [
    app_commands.Choice(name="512 (Default)", value=512),
    app_commands.Choice(name="384", value=384),
    app_commands.Choice(name="416", value=416),
    app_commands.Choice(name="448", value=448),
    app_commands.Choice(name="480", value=480),
    app_commands.Choice(name="544", value=544),
    app_commands.Choice(name="576", value=576),
    app_commands.Choice(name="608", value=608),
    app_commands.Choice(name="640", value=640),
    app_commands.Choice(name="672", value=672),
    app_commands.Choice(name="704", value=704),
    app_commands.Choice(name="736", value=736),
    app_commands.Choice(name="768", value=768),
]


SAMPLER_CHOICES = [
    app_commands.Choice(name="k_lms", value="k_lms"),
    app_commands.Choice(name="dpm2", value="dpm2"),
    app_commands.Choice(name="dpm2_ancestral", value="dpm2_ancestral"),
    app_commands.Choice(name="heun", value="heun"),
    app_commands.Choice(name="euler", value="euler"),
    app_commands.Choice(name="euler_ancestral", value="euler_ancestral"),
]


UPSCALER_CHOICES = [
    app_commands.Choice(name="SwinIR (default, photos and art)",
        value=UPSCALER_SWINIR),
    app_commands.Choice(name="RealESRGAN (photos and art)",
        value=UPSCALER_REALESRGAN_4X),
    app_commands.Choice(name="RealESRGAN Face-Fixing (photos)",
        value=UPSCALER_REALESRGAN_4X_FACE),
    app_commands.Choice(name="RealESRGAN Anime (line art and anime)",
        value=UPSCALER_REALESRGAN_4X_ANIME),
    app_commands.Choice(name="No Upscale (return original image)",
        value=UPSCALER_NONE),
]
