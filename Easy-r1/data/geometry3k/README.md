---
dataset_info:
  features:
  - name: images
    sequence: image
  - name: problem
    dtype: string
  - name: answer
    dtype: string
  splits:
  - name: train
    num_bytes: 43071692.441
    num_examples: 2101
  - name: validation
    num_bytes: 5995999.0
    num_examples: 300
  - name: test
    num_bytes: 12206692.0
    num_examples: 601
  download_size: 59259794
  dataset_size: 61274383.441
configs:
- config_name: default
  data_files:
  - split: train
    path: data/train-*
  - split: validation
    path: data/validation-*
  - split: test
    path: data/test-*
license: mit
task_categories:
- visual-question-answering
language:
- en
size_categories:
- 1K<n<10K
---

This dataset was converted from [https://github.com/lupantech/InterGPS](https://github.com/lupantech/InterGPS) using the following script.

```python
import json
import os
from datasets import Dataset, DatasetDict, Sequence
from datasets import Image as ImageData
from PIL import Image


MAPPING = {"A": 0, "B": 1, "C": 2, "D": 3}


def generate_data(data_path: str):
    for folder in os.listdir(data_path):
        folder_path = os.path.join(data_path, folder)
        image = Image.open(os.path.join(folder_path, "img_diagram.png"), "r")
        with open(os.path.join(folder_path, "data.json"), "r", encoding="utf-8") as f:
            data = json.load(f)
            yield {
                "images": [image],
                "problem": "<image>" + data["annotat_text"],
                "answer": data["choices"][MAPPING[data["answer"]]],
            }


def main():
    trainset = Dataset.from_generator(generate_data, gen_kwargs={"data_path": os.path.join("data", "geometry3k", "train")})
    valset = Dataset.from_generator(generate_data, gen_kwargs={"data_path": os.path.join("data", "geometry3k", "val")})
    testset = Dataset.from_generator(generate_data, gen_kwargs={"data_path": os.path.join("data", "geometry3k", "test")})
    dataset = DatasetDict({"train": trainset, "validation": valset, "test": testset}).cast_column("images", Sequence(ImageData()))
    dataset.push_to_hub("hiyouga/geometry3k")


if __name__ == "__main__":
    main()
```
