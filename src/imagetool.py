import argparse
import json
import os
import random
import traceback

from copy import deepcopy
from io import BytesIO
from typing import Any

from PIL import Image
from docarray import Document, DocumentArray


from constants import (
    DEFAULT_SD_UPSCALE_SAMPLER,
    DEFAULT_SD_UPSCALE_SCALE,
    DEFAULT_SD_UPSCALE_STEPS,
    DEFAULT_SD_UPSCALE_STRENGTH,
    DOCARRAY_LOCATION_FN,
    IMAGE_LOCATION_FN,
    IMAGE_LOCATION_FN_JPG,
    IMAGE_STORAGE_FOLDER,
    OutpaintingModes,
    RealESRGANModels,
    TEMP_JSON_STORAGE_FOLDER,
    UPSCALER_NONE,
    UPSCALER_REALESRGAN_4X,
    UPSCALER_REALESRGAN_4X_ANIME,
    UPSCALER_REALESRGAN_4X_FACE,
    UPSCALER_STABLE_1,
    UPSCALER_STABLE_2,
    UPSCALER_STABLE_3,
    UPSCALER_STABLE_4,
    UPSCALER_STABLE_5,
    UPSCALER_SWINIR,
)

from util import (
    check_subprompt_token_length,
    document_to_pil,
    maybe_split_long_prompt_based_on_tokens,
    resize_for_outpainting_modes,
    resize_image,
    resize_with_mask,
    short_id_generator,
    strip_square,
    tile_with_upscaler_fn_and_reassemble,
    tweak_docarray_tags,
)


# You may need to set this environmental variable.
ENV_SERVER_URL = 'DALLE_FLOW_SERVER'
JINA_SERVER_URL = os.environ.get(ENV_SERVER_URL, False) or \
    'grpc://127.0.0.1:51005'

parser = argparse.ArgumentParser()
parser.add_argument('suffix', help='Input/output file suffix')
args = parser.parse_args()

FILE_NAME_IN = f'{TEMP_JSON_STORAGE_FOLDER}/request-{args.suffix}.json'
FILE_NAME_OUT = f'{TEMP_JSON_STORAGE_FOLDER}/output-{args.suffix}.json'

output: dict[str, Any] = {}

with open(FILE_NAME_IN, 'r') as request_json:
    request = json.load(request_json)
    try:
        # Prompt
        if request['type'] == 'prompt':
            prompt = request['prompt']
            prompt = maybe_split_long_prompt_based_on_tokens(prompt)
            check_subprompt_token_length(prompt)

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
            image_loc = IMAGE_LOCATION_FN_JPG(short_id)
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
            prompts = [
                maybe_split_long_prompt_based_on_tokens(val)
                for val in prompts
            ]
            for val in prompts:
                check_subprompt_token_length(val)
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

            docs_pa: list[Document] = []
            for pr in prompts:
                docs_pa.append(Document(text=pr).post(JINA_SERVER_URL,
                    parameters=params).matches[0])
            da = DocumentArray(docs_pa)

            image = document_to_pil(da[0])
            orig_width, _ = image.size

            short_id = short_id_generator()
            image_loc = IMAGE_LOCATION_FN_JPG(short_id)
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
            prompt = maybe_split_long_prompt_based_on_tokens(prompt)
            check_subprompt_token_length(prompt)

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

            seeds: list[int] = []
            docs_ps: list[Document] = []
            for _ in range(9):
                seeds.append(int(params['seed']))
                docs_ps.append(Document(text=prompt).post(JINA_SERVER_URL,
                    parameters=params).matches[0])
                params['seed'] += 1
            da = DocumentArray(docs_ps)
            image = document_to_pil(da[0])
            orig_width, _ = image.size

            short_id = short_id_generator()
            image_loc = IMAGE_LOCATION_FN_JPG(short_id)
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
            max_image_size = request.get('max_image_size', 768)
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
                img.save(buffered, format='PNG', compress_level=1)
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
                resized_image.save(buffered, format='PNG', compress_level=1)
                _d = Document(
                    blob=buffered.getvalue(),
                    mime_type='image/png',
                ).convert_blob_to_datauri()
                _d.text = orig_prompt
                da = DocumentArray([_d])

            masked = None
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
                prompt = maybe_split_long_prompt_based_on_tokens(request['prompt'])
                check_subprompt_token_length(prompt)
                params['prompt'] = prompt
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
            params_copy = None

            # Outpainting either through arbitrary resizing or by outpainting
            # modes.
            if not request.get('resize', False) and (
                    params['height'] != orig_height or
                    params['width'] != orig_width
                ):
                params_copy = deepcopy(params)
                img_new = resize_with_mask(orig_image,
                    (params['width'], params['height']))
                buffered = BytesIO()
                img_new.save(buffered, format='PNG', compress_level=1)
                _d = Document(
                    blob=buffered.getvalue(),
                    mime_type='image/png',
                ).convert_blob_to_datauri()
                _d.text = orig_prompt

                da = DocumentArray([_d])
                diffused_da = da.post(f'{JINA_SERVER_URL}/stablediffuse',
                    parameters=params_copy)[0].matches
            elif not request.get('resize', False) and \
                request.get('outpaint_mode', None) is not None:
                # Outpainting modes.
                img_new = resize_for_outpainting_modes(orig_image,
                    OutpaintingModes(request['outpaint_mode']),
                    max_image_size)
                (width_new, height_new) = img_new.size
                params_copy = deepcopy(params)
                params_copy['width'] = width_new
                params_copy['height'] = height_new
                buffered = BytesIO()
                img_new.save(buffered, format='PNG', compress_level=1)
                _d = Document(
                    blob=buffered.getvalue(),
                    mime_type='image/png',
                ).convert_blob_to_datauri()
                _d.text = orig_prompt

                da = DocumentArray([_d])
                diffused_da = da.post(f'{JINA_SERVER_URL}/stablediffuse',
                    parameters=params_copy)[0].matches
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
            image_loc = IMAGE_LOCATION_FN_JPG(short_id)
            docarray_loc = DOCARRAY_LOCATION_FN(short_id)

            tweak_docarray_tags(diffused_da, 'outpaint_mode',
                request.get('outpaint_mode', None))

            tweak_docarray_tags(diffused_da, 'prompt_mask',
                request.get('prompt_mask', None))

            tweak_docarray_tags(diffused_da, 'resize',
                request.get('resize', False))

            final_width = params_copy['width'] if params_copy is not None else \
                params['width']
            diffused_da.plot_image_sprites(output=image_loc, show_index=True,
                keep_aspect_ratio=True, canvas_size=final_width * 2)
            diffused_da.save_binary(docarray_loc, protocol='protobuf',
                compress='lz4')

            output['image_loc'] = image_loc
            output['docarray_loc'] = docarray_loc
            output['id'] = short_id

        # Interpolate
        if request['type'] == 'interpolate':
            params = {'num_images': 9}
            prompt = request['prompt']

            prompts = prompt.split('|')
            prompts[0] = maybe_split_long_prompt_based_on_tokens(prompts[0].strip())
            prompts[1] = maybe_split_long_prompt_based_on_tokens(prompts[1].strip())
            check_subprompt_token_length(prompts[0])
            check_subprompt_token_length(prompts[1])
            prompt = '|'.join(prompts)

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
            image_loc = IMAGE_LOCATION_FN_JPG(short_id)
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
            max_image_size = request.get('max_image_size', 768)
            if max(image.size[0] * canvas_scale, image.size[1] * canvas_scale) > \
                max_image_size * canvas_scale:
                raise ValueError('This image is already the maximum size of ' +
                    f'{max_image_size * canvas_scale} for either height or ' +
                    'width')

            upscale = None
            upscaler = request.get('upscaler', None)
            doc = da[idx]

            # TODO Investigate why upscaling requires some of the metadata
            # from the original document in order to work. Simply making the
            # document/docarray as done above seems to have issues.
            def img_to_doc_fn(_img: Image) -> Document:
                copy = deepcopy(doc)
                _buffered = BytesIO()
                _img.save(_buffered, format='PNG', compress_level=1)
                copy.blob = _buffered.getvalue()
                copy.uri = None
                copy = copy.convert_blob_to_datauri()
                return copy

            if upscaler is None or upscaler == UPSCALER_SWINIR:
                def upscaler_fn(_d: Document) -> Document:
                    return _d.post(f'{JINA_SERVER_URL}/upscale')
                upscale = tile_with_upscaler_fn_and_reassemble(
                    image,
                    doc,
                    upscaler_fn,
                    img_to_doc_fn,
                )
            if upscaler in [UPSCALER_REALESRGAN_4X,
                UPSCALER_REALESRGAN_4X_ANIME, UPSCALER_REALESRGAN_4X_FACE]:
                realesrgan_params = {
                    'model_name': RealESRGANModels.RealESRGAN_x4plus.value,
                    'face_enhance': False,
                }
                if upscaler == UPSCALER_REALESRGAN_4X_FACE:
                    realesrgan_params['face_enhance'] = True
                if upscaler == UPSCALER_REALESRGAN_4X_ANIME:
                    realesrgan_params['face_enhance'] = False
                    realesrgan_params['model_name'] = \
                        RealESRGANModels.RealESRGAN_x4plus_anime_6B
                def upscaler_fn_real(_d: Document) -> Document:
                    return DocumentArray([_d]).post(
                        f'{JINA_SERVER_URL}/realesrgan',
                        parameters=realesrgan_params,
                    )[0].matches[0]
                upscale = tile_with_upscaler_fn_and_reassemble(
                    image,
                    doc,
                    upscaler_fn_real,
                    img_to_doc_fn,
                )

            # "SD Upscale", resize the image to the maximum size and then
            # riff at some low strength level. SD was trained at 512x512 so
            # for large sizes anything >0.3 strength can get pretty weird.
            if upscaler in [UPSCALER_STABLE_1, UPSCALER_STABLE_2,
                UPSCALER_STABLE_3, UPSCALER_STABLE_4, UPSCALER_STABLE_5]:
                # Use SwinIR first.
                def upscaler_fn(_d: Document) -> Document:
                    return _d.post(f'{JINA_SERVER_URL}/upscale')
                upscale = tile_with_upscaler_fn_and_reassemble(
                    image,
                    doc,
                    upscaler_fn,
                    img_to_doc_fn,
                )
                image = document_to_pil(upscale)

                strength = request.get('strength', DEFAULT_SD_UPSCALE_STRENGTH)
                if upscaler == UPSCALER_STABLE_1:
                    strength = 0.1
                if upscaler == UPSCALER_STABLE_3:
                    strength = 0.3
                if upscaler == UPSCALER_STABLE_4:
                    strength = 0.4
                if upscaler == UPSCALER_STABLE_5:
                    strength = 0.5

                resized_image = resize_image(image, max_image_size)
                buffered = BytesIO()
                resized_image.save(buffered, format='PNG', compress_level=1)
                _d = Document(
                    blob=buffered.getvalue(),
                    mime_type='image/png',
                ).convert_blob_to_datauri()
                _d.text = doc.text
                da = DocumentArray([_d])

                steps = doc.tags.get('steps', DEFAULT_SD_UPSCALE_STEPS)
                if steps < DEFAULT_SD_UPSCALE_STEPS:
                    steps = DEFAULT_SD_UPSCALE_STEPS

                params = {
                    'width': resized_image.size[0],
                    'height': resized_image.size[1],
                    'prompt': request.get('prompt', None),
                    'seed': random.randint(0, 2 ** 32 - 1),
                    'steps': steps,
                    'sampler': doc.tags.get('sampler',
                        DEFAULT_SD_UPSCALE_SAMPLER),
                    'scale': doc.tags.get('steps', DEFAULT_SD_UPSCALE_SCALE),
                    'strength': strength,
                }
                upscale = da.post(f'{JINA_SERVER_URL}/stablediffuse',
                    parameters=params)[0].matches[0]
                canvas_scale = resized_image.size[0] / orig_width

            if upscaler == UPSCALER_NONE:
                canvas_scale = 1
                upscale = da[idx]

            short_id = short_id_generator()
            image_loc = IMAGE_LOCATION_FN(short_id)
            image_loc_jpeg = IMAGE_LOCATION_FN_JPG(short_id)

            da_upscale = DocumentArray([upscale])
            da_upscale.plot_image_sprites(
                image_loc,
                keep_aspect_ratio=True,
                canvas_size=int(orig_width * canvas_scale),
            )
            image_png = Image.open(image_loc)
            image_jpg = image_png.save(image_loc_jpeg,
                quality=95, optimize=True, progressive=True)

            docarray_loc = DOCARRAY_LOCATION_FN(short_id)
            da_upscale.save_binary(docarray_loc, protocol='protobuf',
                compress='lz4')

            output['id'] = short_id
            output['docarray_loc'] = docarray_loc
            if upscaler != UPSCALER_NONE:
                output['image_loc'] = image_loc_jpeg
            else:
                output['image_loc'] = image_loc
    except Exception as e:
        traceback.print_exc()
        output['error'] = str(e)

with open(FILE_NAME_OUT, 'w') as output_file:
    output_file.write(json.dumps(output))
