import json
import traceback

from copy import deepcopy
from io import BytesIO

from docarray import Document, DocumentArray
from shortid import ShortId
from PIL import Image

# You may need to change this.
JINA_SERVER_URL = 'grpc://127.0.0.1:51005'

short_id_generator = ShortId()


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


output = {}
with open('temp_json/request.json', 'r') as request_json:
    request = json.load(request_json)
    try:
        # Prompt
        if request['type'] == 'prompt':
            prompt = request['prompt']
            params = {'num_images': 4}
            if request.get('scale', None) is not None:
                params['scale'] = request['scale']
            if request.get('seed', None) is not None:
                params['seed'] = request['seed']
            if request.get('steps', None) is not None:
                params['steps'] = request['steps']
            da = Document(text=prompt.strip()).post(JINA_SERVER_URL,
                parameters=params).matches
            short_id = short_id_generator.generate()
            image_loc = f'images/{short_id}.png'
            docarray_loc = f'image_docarrays/{short_id}.bin'
            da.plot_image_sprites(output=image_loc, canvas_size=1024,
                show_index=True)
            da.save_binary(docarray_loc, protocol='protobuf', compress='lz4')

            output['image_loc'] = image_loc
            output['docarray_loc'] = docarray_loc
            output['id'] = short_id

        # Prompt
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

            params = {'num_images': 1}
            if request.get('scale', None) is not None:
                params['scale'] = request['scale']
            if request.get('seed', None) is not None:
                params['seed'] = request['seed']
            else:
                params['seed'] = 12345
            if request.get('steps', None) is not None:
                params['steps'] = request['steps']

            docs = []
            for pr in prompts:
                docs.append(Document(text=pr).post(JINA_SERVER_URL,
                    parameters=params).matches[0])
            da = DocumentArray(docs)

            short_id = short_id_generator.generate()
            image_loc = f'images/{short_id}.png'
            docarray_loc = f'image_docarrays/{short_id}.bin'
            da.plot_image_sprites(output=image_loc, canvas_size=1024,
                show_index=True)
            da.save_binary(docarray_loc, protocol='protobuf', compress='lz4')

            output['image_loc'] = image_loc
            output['docarray_loc'] = docarray_loc
            output['id'] = short_id

        # Prompt
        if request['type'] == 'promptsearch':
            prompt = request['prompt']

            params = {'num_images': 1}
            if request.get('scale', None) is not None:
                params['scale'] = request['scale']
            if request.get('seed', None) is not None:
                params['seed'] = request['seed']
            else:
                params['seed'] = 1
            if request.get('steps', None) is not None:
                params['steps'] = request['steps']

            seeds = []
            docs = []
            for _ in range(9):
                seeds.append(params['seed'])
                docs.append(Document(text=prompt).post(JINA_SERVER_URL,
                    parameters=params).matches[0])
                params['seed'] += 1
            da = DocumentArray(docs)

            short_id = short_id_generator.generate()
            image_loc = f'images/{short_id}.png'
            docarray_loc = f'image_docarrays/{short_id}.bin'
            da.plot_image_sprites(output=image_loc, canvas_size=1024,
                show_index=True)
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
            if not request.get('from_discord', False):
                docarray_id = request['docarray_id']
                idx = request['index']
                old_docarray_loc = f'image_docarrays/{docarray_id}.bin'
                da = DocumentArray.load_binary(
                    old_docarray_loc, protocol='protobuf', compress='lz4'
                )
                da = DocumentArray([da[idx]])
            else:
                prompt = request.get('prompt', '')
                img = Image.open(request['filename'])
                buffered = BytesIO()
                img.save(buffered, format='PNG')
                _d = Document(
                    blob=buffered.getvalue(),
                    mime_type='image/png',
                ).convert_blob_to_datauri()
                _d.text = prompt
                da = DocumentArray([_d])

            params = {'num_images': 4}
            if request.get('prompt', None) is not None:
                params['prompt'] = request['prompt']
            if request.get('scale', None) is not None:
                params['scale'] = request['scale']
            if request.get('seed', None) is not None:
                params['seed'] = request['seed']
            if request.get('strength', None) is not None:
                params['strength'] = request['strength']

            if iterations > 1:
                for _ in range(iterations - 1):
                    params_copy = deepcopy(params)
                    params_copy['num_images'] = 1
                    da = da.post(f'{JINA_SERVER_URL}/stablediffuse',
                        parameters=params_copy)[0].matches

            diffused_da = da.post(f'{JINA_SERVER_URL}/stablediffuse',
                parameters=params)[0].matches

            short_id = short_id_generator.generate()
            image_loc = f'images/{short_id}.png'
            docarray_loc = f'image_docarrays/{short_id}.bin'

            diffused_da.plot_image_sprites(output=image_loc, show_index=True,
                canvas_size=1024)
            diffused_da.save_binary(docarray_loc, protocol='protobuf',
                compress='lz4')

            output['image_loc'] = image_loc
            output['docarray_loc'] = docarray_loc
            output['id'] = short_id

        # Upscale
        if request['type'] == 'upscale':
            docarray_id = request['docarray_id']
            idx = request['index']
            old_docarray_loc = f'image_docarrays/{docarray_id}.bin'
            da  = DocumentArray.load_binary(
                old_docarray_loc, protocol='protobuf', compress='lz4'
            )

            upscale = da[idx].post(f'{JINA_SERVER_URL}/upscale')

            short_id = short_id_generator.generate()
            image_loc = f'images/{short_id}.png'

            da_upscale = DocumentArray([upscale])
            da_upscale.plot_image_sprites(image_loc,
                canvas_size=1024)
            output['image_loc'] = image_loc
    except Exception as e:
        traceback.print_exc()
        output['error'] = str(e)

with open('temp_json/output.json', 'w') as output_file:
    output_file.write(json.dumps(output))
