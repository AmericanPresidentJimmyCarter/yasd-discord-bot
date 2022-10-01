from typing import Optional


def remove_quotes_from_cmd_kwargs(cmd_kwargs):
    split = cmd_kwargs.split(',')
    keywordarg_list = []
    for keywordarg in split:
        as_pair = keywordarg.split('=')
        if as_pair[1].startswith('\'') and as_pair[1].endswith('\''):
            as_pair[1] = as_pair[1][1:-1]
        if as_pair[1] == 'False':
            continue
        keywordarg_list.append(f'{as_pair[0]}={as_pair[1]}')
    return ', '.join(keywordarg_list)


def prompt_un_parenthesis_un_comma(prompt, uncomma=False):
    '''
    Handle parenthesis in slash command prompts.
    '''
    if '(' in prompt:
        prompt = prompt.replace('(', '「')
    if ')' in prompt:
        prompt = prompt.replace(')', '」')
    if uncomma and ',' in prompt:
        prompt = prompt.replace(',', '，')
    return prompt


def serialize_image_request(
    prompt: str,

    height: Optional[int]=None,
    sampler: Optional[str]=None,
    scale: Optional[float]=None,
    seed: Optional[int]=None,
    seed_search: bool=None,
    steps: Optional[int]=None,
    width: Optional[int]=None,
):
    '''
    Serialize an image request to >image format.
    '''
    options = ''
    if height is not None:
        options += f'{height=},'
    if sampler is not None:
        options += f'{sampler=},'
    if scale is not None:
        options += f'{scale=},'
    if seed is not None:
        options += f'{seed=},'
    if seed_search is not None:
        options += f'{seed_search=},'
    if steps is not None:
        options += f'{steps=},'
    if width is not None:
        options += f'{width=},'
    if len(options) > 0 and options[-1] == ',':
        options = f'{options[:-1]}'
        options = remove_quotes_from_cmd_kwargs(options)

    prompt = prompt_un_parenthesis_un_comma(prompt)

    as_string = f'>image {prompt}'
    if options == '':
        return as_string
    return f'{as_string} ({options})'


def serialize_riff_request(
    docarray_id: str,
    idx: int,

    height: Optional[int]=None,
    iterations: Optional[int]=None,
    latentless: bool=False,
    prompt: Optional[str]=None,
    resize: bool=False,
    sampler: Optional[str]=None,
    scale: Optional[float]=None,
    seed: Optional[int]=None,
    steps: Optional[int]=None,
    strength: Optional[float]=None,
    width: Optional[int]=None,
):
    '''
    Serialize a riff request to >riff format.
    '''
    options = ''
    if height is not None:
        options += f'{height=},'
    if iterations is not None:
        options += f'{iterations=},'
    if latentless is True:
        options += f'{latentless=},'
    if prompt is not None:
        prompt = prompt_un_parenthesis_un_comma(prompt, uncomma=True)
        options += f'{prompt=},'
    if resize is True:
        options += f'{resize=},'
    if sampler is not None:
        options += f'{sampler=},'
    if scale is not None:
        options += f'{scale=},'
    if seed is not None:
        options += f'{seed=},'
    if steps is not None:
        options += f'{steps=},'
    if strength is not None:
        options += f'{strength=},'
    if width is not None:
        options += f'{width=},'
    if len(options) > 0 and options[-1] == ',':
        options = f'{options[:-1]}'
        options = remove_quotes_from_cmd_kwargs(options)

    as_string = f'>riff {docarray_id} {idx}'
    if options == '':
        return as_string
    return f'{as_string} ({options})'


def serialize_interpolate_request(
    prompt1: str,
    prompt2: str,

    height: Optional[int]=None,
    resample_prior: bool=True,
    sampler: Optional[str]=None,
    scale: Optional[float]=None,
    seed: Optional[int]=None,
    steps: Optional[int]=None,
    strength: Optional[float]=None,
    width: Optional[int]=None,
):
    '''
    Serialize an interpolate request to >interpolate format.
    '''
    options = ''
    if height is not None:
        options += f'{height=},'
    if resample_prior is False:
        options += 'resample_prior=False,'
    if sampler is not None:
        options += f'{sampler=},'
    if scale is not None:
        options += f'{scale=},'
    if seed is not None:
        options += f'{seed=},'
    if steps is not None:
        options += f'{steps=},'
    if strength is not None:
        options += f'{strength=},'
    if width is not None:
        options += f'{width=},'
    if len(options) > 0 and options[-1] == ',':
        options = f'{options[:-1]}'
        options = remove_quotes_from_cmd_kwargs(options)

    prompt1 = prompt_un_parenthesis_un_comma(prompt1)
    if '|' in prompt1:
        prompt1 = prompt1.replace('|', '')
    prompt2 = prompt_un_parenthesis_un_comma(prompt2)
    if '|' in prompt2:
        prompt2 = prompt2.replace('|', '')

    as_string = f'>interpolate {prompt1} | {prompt2}'
    if options == '':
        return as_string
    return f'{as_string} ({options})'


def serialize_upscale_request(
    docarray_id: str,
    idx: int,
):
    '''
    Serialize an upscale request to '>upscale'.
    '''
    return f'>upscale {docarray_id} {idx}'
