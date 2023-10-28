**Semantic Entity Recognition / Relation Extraction**

```json
{
    "lang": "zh",
    "version": "0.1",
    "split": "val",
    "documents": [
        {
            "id": "zh_val_0",
            "uid": "0ac15750a098682aa02b51555f7c49ff43adc0436c325548ba8dba560cde4e7e",
            "document": [
                {
                    "box": [
                        410,
                        541,
                        535,
                        590
                    ],
                    "text": "夏艳辰",
                    "label": "answer",
                    "words": [
                        {
                            "box": [
                                413,
                                541,
                                447,
                                587
                            ],
                            "text": "夏"
                        },
                        {
                            "box": [
                                458,
                                542,
                                489,
                                588
                            ],
                            "text": "艳"
                        },
                        {
                            "box": [
                                497,
                                544,
                                531,
                                590
                            ],
                            "text": "辰"
                        }
                    ],
                    "linking": [
                        [
                            30,
                            26
                        ]
                    ],
                    "id": 26
                },
                // ...
            ],
            "img": {
                "fname": "zh_val_0.jpg",
                "width": 2480,
                "height": 3508
            }
        },
        // ...
    ]
}
```
