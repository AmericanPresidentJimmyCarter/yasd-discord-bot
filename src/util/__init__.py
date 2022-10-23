import argparse
import asyncio
import datetime
import json
import math
import os
import pathlib
import random
import re
import string
import sys
import time

from io import BytesIO
from typing import TYPE_CHECKING, Any, Callable, Optional, Union
from urllib.error import URLError
from urllib.request import urlopen

import discord
import numpy as np

from PIL import Image, ImageDraw, ImageFilter, ImageOps
from discord import app_commands
from docarray import Document, DocumentArray
from tqdm import tqdm
from transformers import CLIPTokenizer

if TYPE_CHECKING:
    from torch import Tensor

    from client import YASDClient

from constants import (
    CLIP_TOKENIZER_MERGES_FN,
    CLIP_TOKENIZER_VOCAB_FN,
    DOCARRAY_LOCATION_FN,
    ID_LENGTH,
    IMAGETOOL_MODULE_NAME,
    JSON_IMAGE_TOOL_INPUT_FILE_FN,
    JSON_IMAGE_TOOL_OUTPUT_FILE_FN,
    MAX_IMAGE_HEIGHT_WIDTH,
    MAX_MODEL_CLIP_TOKENS_PER_PROMPT,
    MIN_IMAGE_HEIGHT_WIDTH,
    OutpaintingModes,
    REGEX_FOR_TAGS,
    SD_CONCEPTS_URL_FN,
    VALID_TAG_CONCEPTS,
)


random.seed()

prompt_parser = re.compile("""
    (?P<prompt>     # capture group for 'prompt'
    (?:\\\:|[^:])+  # match one or more non ':' characters or escaped colons '\:'
    )               # end 'prompt'
    (?:             # non-capture group
    :+              # match one or more ':' characters
    (?P<weight>     # capture group for 'weight'
    -?\d+(?:\.\d+)? # match positive or negative integer or decimal number
    )?              # end weight capture group, make optional
    \s*             # strip spaces after weight
    |               # OR
    $               # else, if no ':' then match end of line
    )               # end non-capture group
    """, re.VERBOSE)


def bump_nonce_and_return(user_image_generation_nonces: dict, user_id: str):
    if user_image_generation_nonces.get(user_id, None) is None:
        user_image_generation_nonces[user_id] = 0
    else:
        user_image_generation_nonces[user_id] += 1
    return user_image_generation_nonces[user_id]


def check_safety(
    img_loc: str,
    safety_feature_extractor: Callable,
    safety_checker: Callable,
) -> bool:
    try:
        img = Image.open(img_loc)
        safety_checker_input = safety_feature_extractor(img,
            return_tensors="pt")
        _, has_nsfw_concept = safety_checker(
            images=[img_to_tensor(img)],
            clip_input=safety_checker_input.pixel_values)
    except Exception:
        import traceback
        traceback.print_exc()
        return False
    return has_nsfw_concept[0]


async def check_queue_and_maybe_write_to(
    context: 'YASDClient',
    channel: discord.abc.GuildChannel,
    author_id: str,
    prompt: str,
) -> bool:
    if not context.cli_args.allow_queue and context.currently_fetching_ai_image.get(author_id, False) is not False: # type: ignore
        await channel.send(f'Sorry, I am currently working on the image prompt "{context.currently_fetching_ai_image[author_id]}". Please be patient until I finish that.', # type: ignore
            delete_after=5)
        return False
    context.currently_fetching_ai_image[author_id] = prompt # type: ignore
    return True


async def check_restricted_to_channel(
    context: 'YASDClient',
    channel: discord.abc.GuildChannel,
) -> bool:
    if context.cli_args.restrict_all_to_channel: # type: ignore
        if channel.id != context.cli_args.restrict_all_to_channel: # type: ignore
            await channel.send('You are not allowed to use this in this channel!')
            return False
    return True


async def check_subprompt_token_length(
    channel: discord.abc.GuildChannel,
    user_id: str,
    prompt: str,
):
    if prompt is None or prompt == '':
        return True

    parsed_prompts = [match.group('prompt').replace('\\:', ':')
        for match in re.finditer(prompt_parser, prompt)]

    tokenizer = CLIPTokenizer(CLIP_TOKENIZER_VOCAB_FN, CLIP_TOKENIZER_MERGES_FN)
    for subprompt in parsed_prompts:
        if subprompt is None or subprompt == '':
            continue
        as_tokens = tokenizer(subprompt)
        if as_tokens.get('input_ids', None) is None:
            await channel.send(f'Unable to subprompt parse prompt "{subprompt}"')
            return False
        n_tokens = len(as_tokens['input_ids'])
        if n_tokens > MAX_MODEL_CLIP_TOKENS_PER_PROMPT:
            await channel.send(f'Subprompt "{subprompt}" for user <@{user_id}> ' +
                f'is too long (got {n_tokens}, max ' +
                f'{MAX_MODEL_CLIP_TOKENS_PER_PROMPT}). Shorten this subprompt ' +
                'or break into multiple weighted subprompts.')
            return False
    return True


async def check_user_joined_at(
    hours_needed: int|None,
    channel: discord.abc.GuildChannel,
    user: discord.abc.User,
) -> bool:
    if not hours_needed:
        return True
    duration = datetime.datetime.utcnow() - user.joined_at.replace(tzinfo=None)
    hours_int = int(duration.total_seconds()) // 60 ** 2
    if duration < datetime.timedelta(hours=hours_needed):
        await channel.send('Sorry, you have not been on this server long enough ' +
            f'to use the bot (needed {hours_needed} hours, have ' +
            f'{hours_int} hours).')
        return False
    return True


def document_to_pil(doc):
    uri_data = urlopen(doc.uri)
    return Image.open(BytesIO(uri_data.read()))


def img_to_tensor(img: Image) -> 'Tensor':
    import torch
    w, h = img.size
    w, h = map(lambda x: x - x % 32, (w, h))  # resize to integer multiple of 32
    img = img.convert('RGB')
    img = img.resize((w, h), resample=Image.LANCZOS)
    img = np.array(img).astype(np.float32) / 255.0
    img = img[None].transpose(0, 3, 1, 2)
    img = torch.from_numpy(img)
    return 2.*img - 1.


def maybe_split_long_prompt_based_on_tokens(prompt: str) -> str:
    '''
    Attempt to break up a prompt that is too long by converting to tokens and
    then splitting the token lists into chunks and attempting to make those into
    subprompts.
    '''
    def chunk(lst, chunk_size):
        chunk_size = max(1, chunk_size)
        return (lst[i:i+chunk_size] for i in range(0, len(lst), chunk_size))

    if prompt is None or prompt == '':
        return prompt

    parsed_prompts = [(
            match.group('prompt').replace('\\:', ':'),
            float(match.group('weight') or 1),
        ) for match in re.finditer(prompt_parser, prompt)]

    tokenizer = CLIPTokenizer(CLIP_TOKENIZER_VOCAB_FN, CLIP_TOKENIZER_MERGES_FN)

    prompt_maybe_cut_up = ''
    for subprompt_tup in parsed_prompts:
        subprompt = subprompt_tup[0]
        weight = subprompt_tup[1]
        if subprompt is None or subprompt == '':
            continue
        as_tokens = tokenizer(subprompt)
        if as_tokens.get('input_ids', None) is None:
            return prompt
        n_tokens = len(as_tokens['input_ids'])

        if n_tokens > MAX_MODEL_CLIP_TOKENS_PER_PROMPT:
            for tkns in chunk(as_tokens['input_ids'],
                MAX_MODEL_CLIP_TOKENS_PER_PROMPT-2):
                decoded = tokenizer.decode(tkns, skip_special_tokens=True)
                if decoded == '' or decoded is None:
                    continue
                prompt_maybe_cut_up += f'{decoded}:{weight} '
        else:
            prompt_maybe_cut_up += f'{subprompt}:{weight} '

    return prompt_maybe_cut_up


def mono_gradient(
    draw: ImageDraw,
    offset: int,
    sz: tuple[int, int],
    fr: int,
    to: int,
    horizontal: bool=True,
):
    def interpolate(f_co, t_co, interval):
        det_co = (t_co - f_co) / interval
        for i in range(interval):
            yield round(f_co + det_co * i)

    for i, color in enumerate(interpolate(fr, to, sz[0])):
        if horizontal:
            draw.line([(i + offset, 0), (i + offset, sz[1])], color, width=1)
        else: 
            draw.line([(0, i + offset), (sz[0], i + offset)], color, width=1)


def preserve_transparency_resize(img: Image, size: tuple[int, int]) -> Image:
    '''
    By default PIL resize destroys alpha channel information, so when doing
    a resize for an image mask through an alpha channel we have to do things
    a little differently.
    '''
    bands = img.split()
    if len(bands) == 4:
        bands = [
            b.resize(size, Image.LANCZOS)
                if i < 3
                else b.resize(size, Image.NEAREST)
            for i, b in enumerate(bands)
        ]
        return Image.merge('RGBA', bands)

    return img.resize(size, resample=Image.LANCZOS)


def prompt_contains_nsfw(
    prompt: str,
    nsfw_toxic_detection_fn: Callable|None,
    nsfw_wordlist: list[str],
) -> bool:
    if prompt is None:
        return False
    if nsfw_toxic_detection_fn is not None:
        results = nsfw_toxic_detection_fn(prompt)
        # TODO Allow setting these cutoffs?
        if results['sexual_explicit'] > 0.1 or \
            results['obscene'] > 0.5 or \
            results['toxicity'] > 0.8 or \
            results['severe_toxicity'] > 0.5 or \
            results['identity_attack'] > 0.5:
            return True

    if len(nsfw_wordlist) == 0:
        return False
    return any(word in prompt.lower() for word in nsfw_wordlist)


def prompt_has_valid_sd_custom_embeddings(prompt: str|None):
    '''
    Ensure all custom SD embeddings from huggingface are valid by checking
    their website. Cache the requests too.
    '''
    if type(prompt) != str:
        return

    for tag in re.findall(REGEX_FOR_TAGS, prompt):
        concept = tag[1:-1]
        if VALID_TAG_CONCEPTS.get(concept, False):
            continue
        urlopen(SD_CONCEPTS_URL_FN(concept) + 'token_identifier.txt')
        VALID_TAG_CONCEPTS[concept] = True


def resize_image(img: Image) -> Image:
    w, h = img.size
    ratio = float(w) / float(h)
    
    w = MAX_IMAGE_HEIGHT_WIDTH
    h = MAX_IMAGE_HEIGHT_WIDTH
    if ratio > 1:
        w = MAX_IMAGE_HEIGHT_WIDTH
        h = int(float(MAX_IMAGE_HEIGHT_WIDTH) / ratio)
        if h < MIN_IMAGE_HEIGHT_WIDTH:
            h = MIN_IMAGE_HEIGHT_WIDTH
    if ratio < 1:
        w = int(float(MAX_IMAGE_HEIGHT_WIDTH) * ratio)
        if w < MIN_IMAGE_HEIGHT_WIDTH:
            w = MIN_IMAGE_HEIGHT_WIDTH
        h = MAX_IMAGE_HEIGHT_WIDTH
    w, h = map(lambda x: x - x % 32, (w, h))  # resize to integer multiple of 32

    # If the image is RGBA and the mask layer is empty, remove it.
    if img.mode == 'RGBA' and img.split()[-1].getextrema() == (255, 255):
        img.load()
        img_temp = Image.new('RGB', img.size, (255, 255, 255))
        img_temp.paste(img, mask=img.split()[3])
        img = img_temp

    return preserve_transparency_resize(img, (w, h))


def resize_for_outpainting_modes(img: Image, mode: OutpaintingModes) -> Image:
    '''
    Resize an image an add a mask for outpainting.

    OUTPAINT_25_ALL = 'outpaint_25'
    OUTPAINT_25_LEFT = 'outpaint_25_l'
    OUTPAINT_25_RIGHT = 'outpaint_25_r'
    OUTPAINT_25_UP = 'outpaint_25_u'
    OUTPAINT_25_DOWN = 'outpaint_25_d'
    '''
    w, h = img.size

    image_expanded_and_masked = None
    if mode == OutpaintingModes.OUTPAINT_25_ALL:
        canvas_width = int(math.floor(w * 1.5))
        canvas_height = int(math.floor(h * 1.5))
        x1 = int(math.floor((canvas_width - w) / 2))
        y1 = int(math.floor((canvas_height - h) / 2))
        image_expanded = Image.new(img.mode, (canvas_width, canvas_height))
        coords_paste = (
            x1,
            y1,
            x1 + w,
            y1 + h,
        )
        coords_mask = (
            x1,
            y1,
            x1 + w - 1,
            y1 + h - 1,
        )
        image_expanded.paste(img, coords_paste)
        mask = Image.new('L', image_expanded.size, 0)
        draw = ImageDraw.Draw(mask)
        draw.rectangle(coords_mask, 255)

        noised = Image.new(img.mode, image_expanded.size)
        pixels = noised.load()
        for x in range(noised.size[0]):
            for y in range(noised.size[1]):
                pixels[x, y] = (
                    random.randint(0, 255),
                    random.randint(0, 255),
                    random.randint(0, 255),
                )

        together = Image.composite(image_expanded, noised, mask)
        together.putalpha(mask)
        image_expanded_and_masked = together

    if mode in [
        OutpaintingModes.OUTPAINT_25_LEFT,
        OutpaintingModes.OUTPAINT_25_RIGHT,
        OutpaintingModes.OUTPAINT_25_UP,
        OutpaintingModes.OUTPAINT_25_DOWN,
    ]:
        canvas_width = w
        canvas_height = h
        if mode in [OutpaintingModes.OUTPAINT_25_LEFT,
            OutpaintingModes.OUTPAINT_25_RIGHT]:
            canvas_width = int(math.floor(w * 1.25))
        if mode in [OutpaintingModes.OUTPAINT_25_UP,
            OutpaintingModes.OUTPAINT_25_DOWN]:
            canvas_height = int(math.floor(h * 1.25))

        coords_paste = None
        coords_mask = None
        if mode == OutpaintingModes.OUTPAINT_25_LEFT:
            coords_paste = (
                canvas_width - w,
                0,
                canvas_width,
                canvas_height,
            )
            coords_mask = (
                canvas_width - w,
                0,
                canvas_width - 1,
                canvas_height - 1,
            )
        if mode == OutpaintingModes.OUTPAINT_25_RIGHT:
            coords_paste = (
                0,
                0,
                w,
                canvas_height,
            )
            coords_mask = (
                0,
                0,
                w - 1,
                canvas_height - 1,
            )
        if mode == OutpaintingModes.OUTPAINT_25_UP:
            coords_paste = (
                0,
                canvas_height - h,
                canvas_width,
                canvas_height,
            )
            coords_mask = (
                0,
                canvas_height - h,
                canvas_width - 1,
                canvas_height - 1,
            )
        if mode == OutpaintingModes.OUTPAINT_25_DOWN:
            coords_paste = (
                0,
                0,
                canvas_width,
                h,
            )
            coords_mask = (
                0,
                0,
                canvas_width - 1,
                h - 1,
            )

        image_expanded = Image.new(img.mode, (canvas_width, canvas_height))
        image_expanded.paste(img, coords_paste)
        mask = Image.new('L', image_expanded.size, 0)
        draw = ImageDraw.Draw(mask)
        draw.rectangle(coords_mask, 255)

        noised = Image.new(img.mode, image_expanded.size)
        pixels = noised.load()
        for x in range(noised.size[0]):
            for y in range(noised.size[1]):
                pixels[x, y] = (
                    random.randint(0, 255),
                    random.randint(0, 255),
                    random.randint(0, 255),
                )

        together = Image.composite(image_expanded, noised, mask)
        together.putalpha(mask)
        image_expanded_and_masked = together

    return resize_image(image_expanded_and_masked)


def resize_with_mask(img: Image, expected_size: tuple[int, int]) -> Image:
    '''
    Resize an image and add a mask to any empty portions that have been filled
    with noise.

    TODO: Remove as outriffing has been deprecated.
    '''
    img_thumb = img.copy()
    img_thumb.thumbnail(expected_size)

    delta_width = expected_size[0] - img_thumb.size[0]
    delta_height = expected_size[1] - img_thumb.size[1]
    pad_width = delta_width // 2
    pad_height = delta_height // 2
    padding = (
        pad_width,
        pad_height,
        delta_width - pad_width,
        delta_height - pad_height,
    )
    expanded = ImageOps.expand(img_thumb, padding)
    noised = Image.new(img.mode, expanded.size)
    pixels = noised.load()
    for x in range(noised.size[0]):
        for y in range(noised.size[1]):
            pixels[x, y] = (
                random.randint(0, 255),
                random.randint(0, 255),
                random.randint(0, 255),
            )

    mask = Image.new('L', expanded.size, 0)
    draw = ImageDraw.Draw(mask)
    inpaint_region_coords = (
        pad_width,
        pad_height,
        expected_size[0] - pad_width - 1,
        expected_size[1] - pad_height - 1,
    )
    draw.rectangle(inpaint_region_coords, 255)

    together = Image.composite(expanded, noised, mask)
    together.putalpha(mask)
    return together


def resize_with_padding(img: Image, expected_size: tuple[int, int]) -> Image:
    '''
    "Outriffing" resize, where we add some noise while expanding the image
    and blend it in with a gradient.
    '''
    img_thumb = img.copy()
    img_thumb.thumbnail(expected_size)
    
    delta_width = expected_size[0] - img_thumb.size[0]
    delta_height = expected_size[1] - img_thumb.size[1]
    pad_width = delta_width // 2
    pad_height = delta_height // 2
    padding = (
        pad_width,
        pad_height,
        delta_width - pad_width,
        delta_height - pad_height,
    )
    expanded = ImageOps.expand(img_thumb, padding)
    noised = Image.new(img.mode, expanded.size)
    pixels = noised.load()
    for x in range(noised.size[0]):
        for y in range(noised.size[1]):
            pixels[x, y] = (
                random.randint(0, 255),
                random.randint(0, 255),
                random.randint(0, 255),
            )

    mask = Image.new('L', expanded.size, 0)
    draw = ImageDraw.Draw(mask)

    # Apply gradient overlays.
    need_horiz = expanded.size[0] > img.size[0]
    need_vert = expanded.size[1] > img.size[1]

    blend = Image.new('L', expanded.size, 48)
    if need_horiz:
        blurred = shuffle_by_lines_and_blur(img, expanded.size, False)
        noised = Image.composite(noised, blurred, blend)
        mono_gradient(draw, expanded.size[0] // 2 - img_thumb.size[0] // 2, (
            img_thumb.size[0] // 2,
            expanded.size[1],
        ), 0, 255)
        mono_gradient(draw, expanded.size[0] // 2, (
            img_thumb.size[0] // 2,
            expanded.size[1],
        ), 255, 0)
    if need_vert:
        blurred = shuffle_by_lines_and_blur(img, expanded.size, False)
        noised = Image.composite(noised, blurred, blend)
        mono_gradient(draw, expanded.size[1] // 2 - img_thumb.size[1] // 2, (
            expanded.size[0],
            img_thumb.size[1] // 2,
        ), 0, 255, False)
        mono_gradient(draw, expanded.size[1] // 2, (
            expanded.size[0],
            img_thumb.size[1] // 2,
        ), 255, 0, False)

    return Image.composite(expanded, noised, mask)


def seed_from_docarray_id(docarray_id: str) -> int|None:
    '''
    Retrieve a seed from a docarray ID and return it. If it is unset return
    None.
    '''
    docarray_loc = DOCARRAY_LOCATION_FN(docarray_id)
    da = DocumentArray.load_binary(
        docarray_loc, protocol='protobuf', compress='lz4'
    )
    original_request = da[0].tags.get('request', None)
    if original_request is not None:
        return int(original_request['seed'])
    return None


def short_id_generator() -> str:
    return ''.join(random.choices(string.ascii_lowercase +
        string.ascii_uppercase + string.digits, k=ID_LENGTH))


def shuffle_by_lines_and_blur(
    img: Image,
    sz: tuple[int],
    horizontal: bool=True,
) -> Image:
    img = img.convert('RGB')
    img = img.resize(sz)
    if not horizontal:
        img = img.rotate(90, expand=True)
    channel_count = len(img.getbands())
    img_arr = np.reshape(img, (img.height, img.width, channel_count))
    channels = [img_arr[:,:,x] for x in range(channel_count)]
    random_perm = np.random.permutation(img.height)
    shuffled_img_arr = np.dstack([x[random_perm, :] for x in channels]).astype(np.uint8)
    shuffled_img = Image.fromarray(shuffled_img_arr)
    shuffled_img = shuffled_img.filter(ImageFilter.GaussianBlur(radius = 24))
    if not horizontal:
        shuffled_img = shuffled_img.rotate(-90, expand=True)
    return shuffled_img


async def spawn_image_tool_instance(
    author_id: str,
    nonce: int,
    req: dict[str, Any],
) -> dict[str, Any]:
    '''
    The main handler for image generation. Spawns a new process which
    communicates to the flow instance through the gRPC and generates images,
    or else returns a dict indicating what the failure was.
    '''
    with open(JSON_IMAGE_TOOL_INPUT_FILE_FN(author_id, nonce), 'w') as inp:
        inp.write(json.dumps(req))
    proc = await asyncio.create_subprocess_exec(
        'python', '-m', IMAGETOOL_MODULE_NAME, f'{author_id}_{nonce}',
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE)

    # When debugging, you can print the result of this, but it's often full
    # of warnings etc.
    await proc.communicate()
    output = None
    with open(JSON_IMAGE_TOOL_OUTPUT_FILE_FN(author_id, nonce), 'r') as out:
        output = json.load(out)

    return output


def strip_square(s: str) -> str:
    '''
    Remove square brackets from a string.
    '''
    ret = ''
    skip1c = 0
    skip2c = 0
    for i in s:
        if i == '[':
            skip1c += 1
        elif i == ']' and skip1c > 0:
            skip1c -= 1
        elif skip1c == 0 and skip2c == 0:
            ret += i
    return ret



def tweak_docarray_tags(doc_arr: DocumentArray, key: str, val: Any):
    for doc in doc_arr:
        if 'request' in doc.tags and isinstance(doc.tags['request'], dict):
            doc.tags['request'][key] = val


def tweak_docarray_tags_request(
    docarray_id: str,
    kvs: dict[str, Any],
):
    '''
    Load a DocArray, tweaks the request tag, then save it.
    '''
    docarray_loc = DOCARRAY_LOCATION_FN(docarray_id)
    doc_arr = DocumentArray.load_binary(
        docarray_loc, protocol='protobuf', compress='lz4'
    )
    for doc in doc_arr:
        if 'request' in doc.tags and isinstance(doc.tags['request'], dict):
            doc.tags['request'] = { **doc.tags['request'], **kvs }

    doc_arr.save_binary(docarray_loc, protocol='protobuf',
        compress='lz4')


def to_discord_file_and_maybe_check_safety(
    auto_spoiler: bool|None,
    img_loc: str,
    safety_feature_extractor: Callable|None,
    safety_checker: Callable|None,
) -> discord.File:
    nsfw = False
    if auto_spoiler:
        assert safety_feature_extractor is not None
        assert safety_checker is not None
        nsfw = check_safety(img_loc, safety_feature_extractor, safety_checker)
    return discord.File(img_loc, spoiler=nsfw)


def write_button_store(bs_filename: str, button_store_dict: dict):
    with open(bs_filename, 'w') as bs:
        json.dump(button_store_dict, bs)
