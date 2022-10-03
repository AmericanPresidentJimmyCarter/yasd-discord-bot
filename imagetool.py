import argparse
import enum
import json
import numpy as np
import os
import random
import string
import traceback

from copy import deepcopy
from io import BytesIO
from typing import Any
from urllib.request import urlopen

from PIL import Image, ImageDraw, ImageFilter, ImageOps
from docarray import Document, DocumentArray

# You may need to set this environmental variable.
ENV_SERVER_URL = 'DALLE_FLOW_SERVER'
JINA_SERVER_URL = os.environ.get(ENV_SERVER_URL, False) or \
    'grpc://127.0.0.1:51005'
ID_LENGTH = 12

parser = argparse.ArgumentParser()
parser.add_argument('suffix', help='Input/output file suffix')
args = parser.parse_args()

DOCARRAY_LOCATION_FN = lambda docarray_id: f'image_docarrays/{docarray_id}.bin'
IMAGE_LOCATION_FN = lambda sid: f'images/{sid}.png'
FILE_NAME_IN = f'temp_json/request-{args.suffix}.json'
FILE_NAME_OUT = f'temp_json/output-{args.suffix}.json'
UPSCALER_SWINIR = 'swinir'
UPSCALER_REALESRGAN_4X = 'resrgan_4x'
UPSCALER_REALESRGAN_4X_FACE = 'resrgan_4x_face'
UPSCALER_REALESRGAN_4X_ANIME = 'resrgan_4x_anime'
UPSCALER_NONE = 'no_upscale'


class RESRGAN_MODELS(str, enum.Enum):
    RealESRGAN_x4plus = 'RealESRGAN_x4plus'
    RealESRNet_x4plus = 'RealESRNet_x4plus'
    RealESRGAN_x4plus_anime_6B = 'RealESRGAN_x4plus_anime_6B'
    RealESRGAN_x2plus = 'RealESRGAN_x2plus'
    RealESR_animevideov3 = 'realesr-animevideov3'
    RealESR_general_x4v3 = 'realesr-general-x4v3'


def document_to_pil(doc):
    uri_data = urlopen(doc.uri)
    return Image.open(BytesIO(uri_data.read()))


def short_id_generator():
    return ''.join(random.choices(string.ascii_lowercase +
        string.ascii_uppercase + string.digits, k=ID_LENGTH))


def strip_square(s):
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


def shuffle_by_lines_and_blur(img: Image, sz: tuple[int], horizontal=True):
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


def mono_gradient(draw: ImageDraw, offset: int, sz: tuple[int], fr: int, to:int,
    horizontal: bool=True):
    def interpolate(f_co, t_co, interval):
        det_co = (t_co - f_co) / interval
        for i in range(interval):
            yield round(f_co + det_co * i)

    for i, color in enumerate(interpolate(fr, to, sz[0])):
        if horizontal:
            draw.line([(i + offset, 0), (i + offset, sz[1])], color, width=1)
        else: 
            draw.line([(0, i + offset), (sz[0], i + offset)], color, width=1)


def resize_with_padding(img, expected_size):
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


def tweak_docarray_tags_request(doc_arr: DocumentArray, key: str, val: Any):
    for doc in doc_arr:
        if 'request' in doc.tags and isinstance(doc.tags['request'], dict):
            doc.tags['request'][key] = val


output = {}
with open(FILE_NAME_IN, 'r') as request_json:
    request = json.load(request_json)
    try:
        # Prompt
        if request['type'] == 'prompt':
            prompt = request['prompt']
            params = {'num_images': 4}
            if request.get('height', None) is not None:
                params['height'] = request['height']
            if request.get('sampler', None) is not None:
                params['sampler'] = request['sampler']
            if request.get('scale', None) is not None:
                params['scale'] = request['scale']
            if request.get('seed', None) is not None:
                params['seed'] = request['seed']
            if request.get('steps', None) is not None:
                params['steps'] = request['steps']
            if request.get('width', None) is not None:
                params['width'] = request['width']
            da = Document(text=prompt.strip()).post(JINA_SERVER_URL,
                parameters=params).matches
            short_id = short_id_generator()
            image_loc = IMAGE_LOCATION_FN(short_id)
            docarray_loc = DOCARRAY_LOCATION_FN(short_id)

            image = document_to_pil(da[0])
            orig_width, _ = image.size

            da.plot_image_sprites(output=image_loc, canvas_size=orig_width*2,
                keep_aspect_ratio=True, show_index=True)
            da.save_binary(docarray_loc, protocol='protobuf', compress='lz4')

            output['image_loc'] = image_loc
            output['docarray_loc'] = docarray_loc
            output['id'] = short_id

        # Prompt array
        if request['type'] == 'promptarray':
            prompt = request['prompt']
            arr_idx = prompt.find('[')
            if arr_idx == -1:
                raise Exception('Can not find array to iterate on')
            arr_idx_end = prompt.find(']')
            if arr_idx_end == -1:
                raise Exception('Can not find array to iterate on')
            prompt_variations = [ val.strip()
                for val in prompt[arr_idx+1:arr_idx_end].split(',') ]

            prompt_stripped = strip_square(prompt.strip())
            prompts = [
                prompt_stripped[0:arr_idx] + val + prompt_stripped[arr_idx:]
                for val in prompt_variations
            ]
            if len(prompts) > 16:
                prompts = prompts[0:16]

            params = {'num_images': 1}

            if request.get('height', None) is not None:
                params['height'] = request['height']
            if request.get('sampler', None) is not None:
                params['sampler'] = request['sampler']
            if request.get('scale', None) is not None:
                params['scale'] = request['scale']
            if request.get('seed', None) is not None:
                params['seed'] = request['seed']
            else:
                params['seed'] = 12345
            if request.get('steps', None) is not None:
                params['steps'] = request['steps']
            if request.get('width', None) is not None:
                params['width'] = request['width']

            docs = []
            for pr in prompts:
                docs.append(Document(text=pr).post(JINA_SERVER_URL,
                    parameters=params).matches[0])
            da = DocumentArray(docs)

            image = document_to_pil(da[0])
            orig_width, _ = image.size

            short_id = short_id_generator()
            image_loc = IMAGE_LOCATION_FN(short_id)
            docarray_loc = DOCARRAY_LOCATION_FN(short_id)
            da.plot_image_sprites(output=image_loc, canvas_size=orig_width*2,
                keep_aspect_ratio=True, show_index=True)
            da.save_binary(docarray_loc, protocol='protobuf', compress='lz4')

            output['image_loc'] = image_loc
            output['docarray_loc'] = docarray_loc
            output['id'] = short_id

        # Prompt search
        if request['type'] == 'promptsearch':
            prompt = request['prompt']

            params = {'num_images': 1}
            if request.get('height', None) is not None:
                params['height'] = request['height']
            if request.get('sampler', None) is not None:
                params['sampler'] = request['sampler']
            if request.get('scale', None) is not None:
                params['scale'] = request['scale']
            if request.get('seed', None) is not None:
                params['seed'] = request['seed']
            else:
                params['seed'] = 1
            if request.get('steps', None) is not None:
                params['steps'] = request['steps']
            if request.get('width', None) is not None:
                params['width'] = request['width']

            seeds = []
            docs = []
            for _ in range(9):
                seeds.append(params['seed'])
                docs.append(Document(text=prompt).post(JINA_SERVER_URL,
                    parameters=params).matches[0])
                params['seed'] += 1
            da = DocumentArray(docs)
            image = document_to_pil(da[0])
            orig_width, _ = image.size

            short_id = short_id_generator()
            image_loc = IMAGE_LOCATION_FN(short_id)
            docarray_loc = DOCARRAY_LOCATION_FN(short_id)
            da.plot_image_sprites(output=image_loc, canvas_size=orig_width*2,
                keep_aspect_ratio=True, show_index=True)
            da.save_binary(docarray_loc, protocol='protobuf', compress='lz4')

            output['image_loc'] = image_loc
            output['docarray_loc'] = docarray_loc
            output['seeds'] = seeds
            output['id'] = short_id

        # Riff
        if request['type'] == 'riff':
            iterations = request.get('iterations', None)
            if iterations is None:
                iterations = 1

            da = None
            orig_width = None
            orig_height = None
            orig_image = None
            orig_prompt = None

            if not request.get('from_discord', False):
                docarray_id = request['docarray_id']
                idx = request['index']
                old_docarray_loc = DOCARRAY_LOCATION_FN(docarray_id)
                da = DocumentArray.load_binary(
                    old_docarray_loc, protocol='protobuf', compress='lz4'
                )
                da = DocumentArray([da[idx]])
                image = document_to_pil(da[0])
                orig_width, orig_height = image.size
                orig_image = image
                orig_prompt = da[0].text
            else:
                prompt = request.get('prompt', None)
                orig_prompt = prompt
                img = Image.open(request['filename'])
                orig_width, orig_height = image.size
                orig_image = image
                buffered = BytesIO()
                img.save(buffered, format='PNG')
                _d = Document(
                    blob=buffered.getvalue(),
                    mime_type='image/png',
                ).convert_blob_to_datauri()
                _d.text = prompt
                da = DocumentArray([_d])

            if request.get('resize', False) is True and \
                request.get('height', None) is not None and \
                request.get('width', None) is not None:
                resized_image = orig_image.resize((
                    request['width'],
                    request['height'],
                ), Image.LANCZOS)
                buffered = BytesIO()
                resized_image.save(buffered, format='PNG')
                _d = Document(
                    blob=buffered.getvalue(),
                    mime_type='image/png',
                ).convert_blob_to_datauri()
                _d.text = orig_prompt
                da = DocumentArray([_d])

            if request.get('prompt_mask', None) is not None:
                da[0].text = request['prompt_mask']
                mask_params = { 'invert': False }
                if request['prompt_mask'][0:4] == 'not ':
                    mask_params['invert'] = True
                masked = DocumentArray([da[0]]).post(
                    f'{JINA_SERVER_URL}/segment', parameters=mask_params)[0].matches
                masked[0].text = orig_prompt
                da = masked

            params = {'num_images': 4}
            if request.get('height', None) is not None:
                params['height'] = request['height']
            else:
                params['height'] = orig_height
            if request.get('latentless', False) is not False:
                params['latentless'] = request['latentless']
            if request.get('prompt', None) is not None:
                params['prompt'] = request['prompt']
            if request.get('sampler', None) is not None:
                params['sampler'] = request['sampler']
            if request.get('scale', None) is not None:
                params['scale'] = request['scale']
            if request.get('seed', None) is not None:
                params['seed'] = request['seed']
            if request.get('steps', None) is not None:
                params['steps'] = request['steps']
            if request.get('strength', None) is not None:
                params['strength'] = request['strength']
            if request.get('width', None) is not None:
                params['width'] = request['width']
            else:
                params['width'] = orig_width

            diffused_da = None
            if not request.get('resize', False) and (
                    params['height'] != orig_height or
                    params['width'] != orig_width
                ):
                img_new = resize_with_padding(orig_image,
                    (params['width'], params['height']))
                buffered = BytesIO()
                img_new.save(buffered, format='PNG')
                _d = Document(
                    blob=buffered.getvalue(),
                    mime_type='image/png',
                ).convert_blob_to_datauri()
                _d.text = orig_prompt

                # "Not outpainting"
                FINAL_STAGE = 0.65
                for _strength in np.linspace(0.15, FINAL_STAGE, 8):
                    da = DocumentArray([_d])
                    params_copy = deepcopy(params)
                    params_copy['num_images'] = 1
                    params_copy['seed'] = random.randint(0, 2 ** 32 - 1)
                    params_copy['strength'] = _strength
                    if _strength != FINAL_STAGE:
                        da = da.post(f'{JINA_SERVER_URL}/stablediffuse',
                            parameters=params_copy)[0].matches
                    else:
                        params_copy['num_images'] = 4
                        da = da.post(f'{JINA_SERVER_URL}/stablediffuse',
                            parameters=params_copy)[0].matches
                        
                diffused_da = da
            else:
                if iterations > 1:
                    for _ in range(iterations - 1):
                        params_copy = deepcopy(params)
                        params_copy['num_images'] = 1
                        params_copy['seed'] = random.randint(0, 2 ** 32 - 1)
                        da = da.post(f'{JINA_SERVER_URL}/stablediffuse',
                            parameters=params_copy)[0].matches

                diffused_da = da.post(f'{JINA_SERVER_URL}/stablediffuse',
                    parameters=params)[0].matches

            short_id = short_id_generator()
            image_loc = IMAGE_LOCATION_FN(short_id)
            docarray_loc = DOCARRAY_LOCATION_FN(short_id)

            tweak_docarray_tags_request(diffused_da, 'prompt_mask',
                request.get('prompt_mask', None))

            tweak_docarray_tags_request(diffused_da, 'resize',
                request.get('resize', False))

            diffused_da.plot_image_sprites(output=image_loc, show_index=True,
                keep_aspect_ratio=True, canvas_size=params['width']*2)
            diffused_da.save_binary(docarray_loc, protocol='protobuf',
                compress='lz4')

            output['image_loc'] = image_loc
            output['docarray_loc'] = docarray_loc
            output['id'] = short_id

        # Interpolate
        if request['type'] == 'interpolate':
            params = {'num_images': 9}
            prompt = request['prompt']

            if request.get('height', None) is not None:
                params['height'] = request['height']
            if request.get('resample_prior', True) is False:
                params['resample_prior'] = request['resample_prior']
            if request.get('sampler', None) is not None:
                params['sampler'] = request['sampler']
            if request.get('scale', None) is not None:
                params['scale'] = request['scale']
            if request.get('seed', None) is not None:
                params['seed'] = request['seed']
            if request.get('steps', None) is not None:
                params['steps'] = request['steps']
            if request.get('strength', None) is not None:
                params['strength'] = request['strength']
            if request.get('width', None) is not None:
                params['width'] = request['width']

            interpolated_da = Document(text=prompt.strip()).post(
                f'{JINA_SERVER_URL}/stableinterpolate',
                parameters=params).matches
            image = document_to_pil(interpolated_da[0])
            orig_width, _ = image.size

            short_id = short_id_generator()
            image_loc = IMAGE_LOCATION_FN(short_id)
            docarray_loc = DOCARRAY_LOCATION_FN(short_id)

            interpolated_da.plot_image_sprites(output=image_loc, show_index=True,
                keep_aspect_ratio=True, canvas_size=orig_width*3)
            interpolated_da.save_binary(docarray_loc, protocol='protobuf',
                compress='lz4')

            output['image_loc'] = image_loc
            output['docarray_loc'] = docarray_loc
            output['id'] = short_id


        # Upscale
        if request['type'] == 'upscale':
            docarray_id = request['docarray_id']
            idx = request['index']
            old_docarray_loc = DOCARRAY_LOCATION_FN(docarray_id)
            da  = DocumentArray.load_binary(
                old_docarray_loc, protocol='protobuf', compress='lz4'
            )
            image = document_to_pil(da[idx])
            orig_width, _ = image.size

            canvas_scale = 4
            upscale = None
            upscaler = request.get('upscaler', None)
            if upscaler is None or upscaler == UPSCALER_SWINIR:
                upscale = da[idx].post(f'{JINA_SERVER_URL}/upscale')
            if upscaler in [UPSCALER_REALESRGAN_4X,
                UPSCALER_REALESRGAN_4X_ANIME, UPSCALER_REALESRGAN_4X_FACE]:
                realesrgan_params = {
                    'model_name': RESRGAN_MODELS.RealESRGAN_x4plus.value,
                    'face_enhance': False,
                }
                if upscaler == UPSCALER_REALESRGAN_4X_FACE:
                    realesrgan_params['face_enhance'] = True
                if upscaler == UPSCALER_REALESRGAN_4X_ANIME:
                    realesrgan_params['face_enhance'] = False
                    realesrgan_params['model_name'] = \
                        RESRGAN_MODELS.RealESRGAN_x4plus_anime_6B
                upscale = DocumentArray([da[idx]]).post(f'{JINA_SERVER_URL}/realesrgan',
                    parameters=realesrgan_params)[0].matches[0]
            if upscaler == UPSCALER_NONE:
                canvas_scale = 1
                upscale = da[idx]

            short_id = short_id_generator()
            image_loc = IMAGE_LOCATION_FN(short_id)
            image_loc_jpeg = f'images/{short_id}.jpg'

            da_upscale = DocumentArray([upscale])
            da_upscale.plot_image_sprites(image_loc,
                keep_aspect_ratio=True, canvas_size=orig_width * canvas_scale)
            image_png = Image.open(image_loc)
            image_jpg = image_png.save(image_loc_jpeg,
                quality=95, optimize=True, progressive=True)
            output['image_loc'] = image_loc_jpeg
    except Exception as e:
        traceback.print_exc()
        output['error'] = str(e)

with open(FILE_NAME_OUT, 'w') as output_file:
    output_file.write(json.dumps(output))
