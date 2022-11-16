**Text Detection/Recognition/Spotting**

```json
{
  "imgs": {
    "OpenImages_ImageID_1": {
      "id": "OpenImages_ImageID_1",
      "width": "INT, Width of the image",
      "height": "INT, Height of the image",
      "set": "Split train|val|test",
      "filename": "train|test/OpenImages_ImageID_1.jpg"
    },
    "OpenImages_ImageID_2": {
      "...": "..."
    }
  },
  "anns": {
    "OpenImages_ImageID_1_1": {
      "id": "STR, OpenImages_ImageID_1_1, Specifies the nth annotation for an image",
      "image_id": "OpenImages_ImageID_1",
      "bbox": [
        "FLOAT x1",
        "FLOAT y1",
        "FLOAT x2",
        "FLOAT y2"
      ],
      "points": [
        "FLOAT x1",
        "FLOAT y1",
        "FLOAT x2",
        "FLOAT y2",
        "...",
        "FLOAT xN",
        "FLOAT yN"
      ],
      "utf8_string": "text for this annotation",
      "area": "FLOAT, area of this box"
    },
    "OpenImages_ImageID_1_2": {
      "...": "..."
    },
    "OpenImages_ImageID_2_1": {
      "...": "..."
    }
  },
  "img2Anns": {
    "OpenImages_ImageID_1": [
      "OpenImages_ImageID_1_1",
      "OpenImages_ImageID_1_2",
      "OpenImages_ImageID_1_2"
    ],
    "OpenImages_ImageID_N": [
      "..."
    ]
  }
}
```
