# Enable Blank Space Recognition

It is noteworthy that the `LineStrParser` should **NOT** be used to parse the annotation files containing multiple blank spaces (in file name or recognition transcriptions). The users have to convert the `plain txt` annotations to `json lines` to enable space recognition. For example:

```txt
% A plain txt annotation file that contains blank spaces
test/img 1.jpg Hello World!
test/img 2.jpg Hello Open MMLab!
test/img 3.jpg Hello MMOCR!
```

The `LineStrParser` will split the above annotation line to pieces (e.g. \['test/img', '1.jpg', 'Hello', 'World!'\]) that cannot be matched to the `keys` (e.g. \['filename', 'text'\]). Therefore, we need to convert it to a json line format by `json.dumps` (check [here](https://github.com/open-mmlab/mmocr/blob/main/tools/data/textrecog/funsd_converter.py#L175-L180) to see how to dump `jsonl`), and then the annotation file will look like as follows:

```txt
% A json line annotation file that contains blank spaces
{"filename": "test/img 1.jpg", "text": "Hello World!"}
{"filename": "test/img 2.jpg", "text": "Hello Open MMLab!"}
{"filename": "test/img 3.jpg", "text": "Hello MMOCR!"}
```

After converting the annotation format, you just need to set the parser arguments as:

```python
parser=dict(
    type='LineJsonParser',
    keys=['filename', 'text']))
```

Besides, you need to specify a dict that contains blank space to enable blank recognition. Particularly, MMOCR provides two [built-in dicts](https://github.com/open-mmlab/mmocr/blob/main/mmocr/models/textrecog/convertors/base.py) `DICT37` and `DICT91` that contain blank space. For example, change the default `dict_type` in `configs/_base_/recog_models/crnn.py` to `DICT37`.

```python
label_convertor = dict(
    type='CTCConvertor', dict_type='DICT37', with_unknown=False, lower=True) # ['DICT36', 'DICT37', 'DICT90', 'DICT91']
```
