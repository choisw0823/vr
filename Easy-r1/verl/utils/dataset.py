# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import math
import os
from collections import defaultdict
from io import BytesIO
from typing import Any, Optional, Union

import numpy as np
import torch
from datasets import load_dataset
from jinja2 import Template
from PIL import Image
from PIL.Image import Image as ImageObject
from qwen_vl_utils.vision_process import fetch_video
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizer, ProcessorMixin

from ..models.transformers.qwen2_vl import get_rope_index
from . import torch_functional as VF


def collate_fn(features: list[dict[str, Any]]) -> dict[str, Any]:
    tensors = defaultdict(list)
    non_tensors = defaultdict(list)
    for feature in features:
        for key, value in feature.items():
            if isinstance(value, torch.Tensor):
                tensors[key].append(value)
            else:
                non_tensors[key].append(value)

    for key, value in tensors.items():
        tensors[key] = torch.stack(value, dim=0)

    for key, value in non_tensors.items():
        non_tensors[key] = np.array(value, dtype=object)

    return {**tensors, **non_tensors}


def process_image(
    image: Union[dict[str, Any], ImageObject, str], min_pixels: Optional[int], max_pixels: Optional[int]
) -> ImageObject:
    if isinstance(image, str):
        image = Image.open(image)
    elif isinstance(image, dict):
        image = Image.open(BytesIO(image["bytes"]))
    elif isinstance(image, bytes):
        image = Image.open(BytesIO(image))

    image.load()  # avoid "Too many open files" errors
    if max_pixels is not None and (image.width * image.height) > max_pixels:
        resize_factor = math.sqrt(max_pixels / (image.width * image.height))
        width, height = int(image.width * resize_factor), int(image.height * resize_factor)
        image = image.resize((width, height))

    if min_pixels is not None and (image.width * image.height) < min_pixels:
        resize_factor = math.sqrt(min_pixels / (image.width * image.height))
        width, height = int(image.width * resize_factor), int(image.height * resize_factor)
        image = image.resize((width, height))

    if image.mode != "RGB":
        image = image.convert("RGB")

    return image


def process_video(
    video: str, min_pixels: Optional[int], max_pixels: Optional[int], video_fps: float, return_fps: bool = False
) -> Union[list[ImageObject], tuple[list[ImageObject], list[float]]]:
    vision_info = {"video": video, "min_pixels": min_pixels, "max_pixels": max_pixels, "fps": video_fps}
    return fetch_video(vision_info, return_video_sample_fps=return_fps)


def _decode_video_uniform(
    video_path: str, min_pixels: Optional[int], max_pixels: Optional[int], video_fps: float, num_frames: int
) -> list[ImageObject]:
    """
    비디오에서 균등 간격으로 num_frames개의 프레임을 추출합니다.
    
    Args:
        video_path: 비디오 파일 경로
        min_pixels: 최소 픽셀 수
        max_pixels: 최대 픽셀 수  
        video_fps: 비디오 FPS
        num_frames: 추출할 프레임 수
        
    Returns:
        PIL Image 객체 리스트
    """
    vision_info = {"video": video_path, "min_pixels": min_pixels, "max_pixels": max_pixels, "fps": video_fps}
    frames, _ = fetch_video(vision_info, return_video_sample_fps=True)
    
    if len(frames) == 0:
        return []
    
    # 균등 간격으로 num_frames개 선택
    if len(frames) >= num_frames:
        indices = np.linspace(0, len(frames) - 1, num_frames, dtype=int)
        selected_frames = [frames[i] for i in indices]
    else:
        # 프레임이 부족하면 마지막 프레임을 반복
        selected_frames = frames + [frames[-1]] * (num_frames - len(frames))
    
    return selected_frames


def _process_image(
    image: Union[dict[str, Any], ImageObject, str, bytes], min_pixels: Optional[int], max_pixels: Optional[int]
) -> ImageObject:
    """
    이미지를 처리하고 리사이즈합니다.
    
    Args:
        image: 이미지 데이터 (경로, PIL Image, bytes, dict)
        min_pixels: 최소 픽셀 수
        max_pixels: 최대 픽셀 수
        
    Returns:
        처리된 PIL Image 객체
    """
    return process_image(image, min_pixels, max_pixels)


class RLHFDataset(Dataset):
    """
    We assume the dataset contains a column that contains prompts and other information
    """

    def __init__(
        self,
        data_path: str,
        tokenizer: PreTrainedTokenizer,
        processor: Optional[ProcessorMixin],
        prompt_key: str = "prompt",
        answer_key: str = "answer",
        image_key: str = "images",
        video_key: str = "videos",
        image_dir: Optional[str] = None,
        video_fps: float = 2.0,
        max_prompt_length: int = 1024,
        truncation: str = "error",
        format_prompt: Optional[str] = None,
        min_pixels: Optional[int] = None,
        max_pixels: Optional[int] = None,
        filter_overlong_prompts: bool = True,
        filter_overlong_prompts_workers: int = 16,
    ):
        self.tokenizer = tokenizer
        self.processor = processor
        self.prompt_key = prompt_key
        self.answer_key = answer_key
        self.image_key = image_key
        self.video_key = video_key
        self.image_dir = image_dir
        self.video_fps = video_fps
        self.max_prompt_length = max_prompt_length
        self.truncation = truncation
        self.min_pixels = min_pixels
        self.max_pixels = max_pixels

        if "@" in data_path:
            data_path, data_split = data_path.split("@")
        else:
            data_split = "train"

        if os.path.isdir(data_path):
            # when we use dataset builder, we should always refer to the train split
            file_type = os.path.splitext(os.listdir(data_path)[0])[-1][1:].replace("jsonl", "json")
            self.dataset = load_dataset(file_type, data_dir=data_path, split=data_split)
        elif os.path.isfile(data_path):
            file_type = os.path.splitext(data_path)[-1][1:].replace("jsonl", "json")
            self.dataset = load_dataset(file_type, data_files=data_path, split=data_split)
        else:
            # load remote dataset from huggingface hub
            self.dataset = load_dataset(data_path, split=data_split)

        self.format_prompt = None
        if format_prompt:
            with open(format_prompt, encoding="utf-8") as f:
                self.format_prompt = f.read()

        if filter_overlong_prompts:
            self.dataset = self.dataset.filter(
                self._filter_overlong_prompts,
                desc="Filtering overlong prompts",
                num_proc=filter_overlong_prompts_workers,
            )

    def _build_messages(self, example: dict[str, Any]) -> list[dict[str, Any]]:
        prompt_str: str = example[self.prompt_key]
        if self.format_prompt:
            format_prompt = Template(self.format_prompt.strip())
            prompt_str = format_prompt.render(content=prompt_str)

        if self.image_key in example:
            # https://huggingface.co/docs/transformers/en/tasks/image_text_to_text
            content_list = []
            for i, content in enumerate(prompt_str.split("<image>")):
                if i != 0:
                    content_list.append({"type": "image"})

                if content:
                    content_list.append({"type": "text", "text": content})

            return [{"role": "user", "content": content_list}]
        elif self.video_key in example:
            content_list = []
            for i, content in enumerate(prompt_str.split("<video>")):
                if i != 0:
                    content_list.append({"type": "video"})

                if content:
                    content_list.append({"type": "text", "text": content})

            return [{"role": "user", "content": content_list}]
        else:
            return [{"role": "user", "content": prompt_str}]

    def _filter_overlong_prompts(self, example: dict[str, Any]) -> bool:
        messages = self._build_messages(example)
        if self.image_key in example:
            prompt = self.processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
            images = example[self.image_key]
            if self.image_dir is not None and len(images) != 0 and isinstance(images[0], str):  # image paths
                images = [os.path.join(self.image_dir, image) for image in images]

            processed_images = [] if len(images) != 0 else None  # text-only data
            for image in images:
                processed_images.append(process_image(image, self.min_pixels, self.max_pixels))

            model_inputs = self.processor(processed_images, [prompt], add_special_tokens=False, return_tensors="pt")
            return model_inputs["input_ids"].size(-1) <= self.max_prompt_length
        elif self.video_key in example:
            prompt = self.processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
            videos = example[self.video_key]
            if self.image_dir is not None and len(videos) != 0 and isinstance(videos[0], str):  # video paths
                videos = [os.path.join(self.image_dir, video) for video in videos]

            processed_videos = [] if len(videos) != 0 else None  # text-only data
            for video in videos:
                processed_videos.append(process_video(video, self.min_pixels, self.max_pixels, self.video_fps))

            model_inputs = self.processor(
                videos=processed_videos, text=[prompt], add_special_tokens=False, return_tensors="pt"
            )
            return model_inputs["input_ids"].size(-1) <= self.max_prompt_length
        else:
            input_ids = self.tokenizer.apply_chat_template(messages, add_generation_prompt=True)
            return len(input_ids) <= self.max_prompt_length

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        example: dict = self.dataset[index]
        messages = self._build_messages(example)
        example.pop(self.prompt_key, None)

        if self.image_key in example:
            prompt = self.processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
            images = example.pop(self.image_key)
            if self.image_dir is not None and len(images) != 0 and isinstance(images[0], str):  # image paths
                images = [os.path.join(self.image_dir, image) for image in images]

            processed_images = [] if len(images) != 0 else None  # text-only data
            for image in images:
                processed_images.append(process_image(image, self.min_pixels, self.max_pixels))

            model_inputs = self.processor(processed_images, [prompt], add_special_tokens=False, return_tensors="pt")
            input_ids = model_inputs.pop("input_ids")[0]
            attention_mask = model_inputs.pop("attention_mask")[0]
            example["multi_modal_data"] = {"images": images}
        elif self.video_key in example:
            prompt = self.processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
            videos = example.pop(self.video_key)
            if self.image_dir is not None and len(videos) != 0 and isinstance(videos[0], str):  # video paths
                videos = [os.path.join(self.image_dir, video) for video in videos]

            processed_videos = [] if len(videos) != 0 else None  # text-only data
            video_fps_list = []
            for video in videos:
                processed_video, video_fps = process_video(
                    video, self.min_pixels, self.max_pixels, self.video_fps, return_fps=True
                )
                processed_videos.append(processed_video)
                video_fps_list.append(video_fps)

            model_inputs = self.processor(
                videos=processed_videos, text=[prompt], add_special_tokens=False, return_tensors="pt"
            )
            if "second_per_grid_ts" in self.processor.model_input_names:
                model_inputs["second_per_grid_ts"] = [2.0 / video_sample_fps for video_sample_fps in video_fps_list]

            input_ids = model_inputs.pop("input_ids")[0]
            attention_mask = model_inputs.pop("attention_mask")[0]
            example["multi_modal_data"] = {"videos": videos}
        else:
            prompt = self.tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
            model_inputs = self.tokenizer([prompt], add_special_tokens=False, return_tensors="pt")
            input_ids = model_inputs.pop("input_ids")[0]
            attention_mask = model_inputs.pop("attention_mask")[0]

        if self.processor is not None and "Qwen2VLImageProcessor" in self.processor.image_processor.__class__.__name__:
            # qwen2vl mrope
            vision_position_ids = get_rope_index(
                self.processor,
                input_ids=input_ids,
                image_grid_thw=model_inputs.get("image_grid_thw", None),
                video_grid_thw=model_inputs.get("video_grid_thw", None),
                second_per_grid_ts=model_inputs.get("second_per_grid_ts", None),
                attention_mask=attention_mask,
            )  # (3, seq_length)
            text_position_ids = torch.arange(len(input_ids)).unsqueeze(0)  # (1, seq_length)
            position_ids = torch.cat((text_position_ids, vision_position_ids), dim=0)  # (4, seq_length)
        else:
            position_ids = torch.clip(attention_mask.cumsum(dim=0) - 1, min=0, max=None)  # (seq_length,)

        input_ids, attention_mask, position_ids = VF.postprocess_data(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            max_length=self.max_prompt_length,
            pad_token_id=self.tokenizer.pad_token_id,
            left_pad=True,
            truncation=self.truncation,
        )
        raw_prompt_ids = self.tokenizer.encode(prompt, add_special_tokens=False)
        if len(raw_prompt_ids) > self.max_prompt_length:
            if self.truncation == "left":
                raw_prompt_ids = raw_prompt_ids[-self.max_prompt_length :]
            elif self.truncation == "right":
                raw_prompt_ids = raw_prompt_ids[: self.max_prompt_length]
            elif self.truncation == "error":
                raise RuntimeError(f"Prompt length {len(raw_prompt_ids)} is longer than {self.max_prompt_length}.")

        example["input_ids"] = input_ids
        example["attention_mask"] = attention_mask
        example["position_ids"] = position_ids
        example["raw_prompt_ids"] = raw_prompt_ids
        example["ground_truth"] = example.pop(self.answer_key)
        return example




class RankLLMRerankDataset(Dataset):
    """
    RankLLM reranking 전용 Dataset (12-frame uniform sampling).

    기대 JSON(권장):
    {
      "metadata": {"frames_per_candidate": 12, "default_num_candidates": 5},
      "samples": [
        {
          "sample_id": "0_2",
          "query": "...",
          "num_candidates": 5,                 # optional
          "frames_per_candidate": 12,          # optional
          "candidates": [
            {
              "candidate_id": 1,               # 1..N (프롬프트 [1]..[N]과 1:1)
              "video_id": "videoXXXX",
              "is_correct": false,             # optional (평가용)
              "sim_score": 1.23,               # optional (분석)
              "gt_rank_top10": 3,              # optional (분석)
              "frames": ["/abs/path/f_00.jpg", "... x12 ..."]
              # 또는 "video_path": "/abs/path/videoXXXX.mp4"
            },
            ...
          ],
          "ground_truth": {
            "order_sim_desc": [2,4,3,5,1],     # cand_id 순열 (sim 내림차순 등)
            "correct_cand_id": 2
          },
          "case_type": "random_2",             # optional
          "seed": 123456                       # optional
        }
      ]
    }

    반환 키:
      - input_ids, attention_mask, position_ids, raw_prompt_ids
      - ground_truth (dict), multi_modal_data (frames or decoded images)
      - query, sample_id, num_candidates, frames_per_candidate
    """

    def __init__(
        self,
        data_path: str,
        tokenizer: PreTrainedTokenizer,
        processor: ProcessorMixin,
        jinja_prompt_path: str,              # easy-r1 Jinja 템플릿 경로
        image_root: Optional[str] = None,    # 상대경로 prefix
        frames_per_candidate: Optional[int] = None,  # None이면 metadata/샘플 → 기본 12
        max_prompt_length: int = 4096,
        truncation: str = "error",
        min_pixels: Optional[int] = None,
        max_pixels: Optional[int] = None,
        video_fps: float = 2.0,
        allow_video_decode: bool = False,    # frames 없고 video_path만 있을 때 디코드 허용
        filter_overlong_prompts: bool = True,
        filter_workers: int = 16,            # (datasets.filter의 num_proc에 사용할 수 있음)
    ):
        self.tokenizer = tokenizer
        self.processor = processor
        self.image_root = image_root
        self.default_fpc = frames_per_candidate
        self.max_prompt_length = max_prompt_length
        self.truncation = truncation
        self.min_pixels = min_pixels
        self.max_pixels = max_pixels
        self.video_fps = video_fps
        self.allow_video_decode = allow_video_decode

        # 데이터 로드
        if os.path.isdir(data_path):
            file_type = os.path.splitext(os.listdir(data_path)[0])[-1][1:].replace("jsonl", "json")
            ds = load_dataset(file_type, data_dir=data_path, split="train")
            # 샘플 접근: {"samples": [...]} 구조면 그 배열을, 아니면 ds 자체를
            self.metadata = ds[0].get("metadata", None) if len(ds) and "metadata" in ds.features else None
            self.samples = ds["samples"] if "samples" in ds.features else ds
        elif os.path.isfile(data_path):
            file_ext = os.path.splitext(data_path)[-1][1:].lower()
            if file_ext == "json":
                # JSON 파일을 직접 로드
                import json
                with open(data_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                self.metadata = data.get("metadata", None)
                self.samples = data.get("samples", [])
            else:
                # JSONL이나 다른 형태는 datasets 라이브러리 사용
                file_type = file_ext.replace("jsonl", "json")
                ds = load_dataset(file_type, data_files=data_path, split="train")
                self.metadata = ds[0].get("metadata", None) if len(ds) and "metadata" in ds.features else None
                self.samples = ds["samples"] if "samples" in ds.features else ds
        else:
            ds = load_dataset(data_path, split="train")
            self.metadata = ds[0].get("metadata", None) if len(ds) and "metadata" in ds.features else None
            self.samples = ds["samples"] if "samples" in ds.features else ds

        with open(jinja_prompt_path, "r", encoding="utf-8") as f:
            self.jinja_template = Template(f.read().strip())

        # 길이 필터(빠른 1차 점검; 실제 이미지 로드 없이 텍스트 길이만 확인)
        if filter_overlong_prompts:
            self._indices = []
            for idx in range(len(self.samples)):
                try:
                    if self._check_length(idx):
                        self._indices.append(idx)
                except Exception:
                    continue
        else:
            self._indices = list(range(len(self.samples)))

    # ---------- helpers ----------
    def __len__(self) -> int:
        return len(self._indices)

    def _resolve_fpc(self, sample: dict) -> int:
        if self.default_fpc is not None:
            return int(self.default_fpc)
        if "frames_per_candidate" in sample:
            return int(sample["frames_per_candidate"])
        if self.metadata and "frames_per_candidate" in self.metadata:
            return int(self.metadata["frames_per_candidate"])
        return 12  # 기본값

    def _resolve_num_candidates(self, sample: dict) -> int:
        if "num_candidates" in sample:
            return int(sample["num_candidates"])
        if self.metadata and "default_num_candidates" in self.metadata:
            return int(self.metadata["default_num_candidates"])
        return len(sample["candidates"])

    def _render_prompt(self, sample: dict, N: int, F: int) -> str:
        return self.jinja_template.render(
            query=sample["query"],
            num_candidates=N,
            frames_per_candidate=F,
        ).strip()

    def _gather_frames_uniform(
        self,
        sample: dict,
        F: int,
    ) -> list[Union[str, Image.Image]]:
        """
        cand_id 오름차순으로 정렬 후,
        각 후보에서 균등 간격으로 F장 선택(부족하면 마지막 프레임 반복).
        """
        cands = sorted(sample["candidates"], key=lambda x: int(x["candidate_id"]))
        items: list[Union[str, Image.Image]] = []

        for c in cands:
            # 1) frames 목록 제공 시 → 거기서 균등 F장
            if "frames" in c and c["frames"]:
                paths = c["frames"]
                if self.image_root is not None and isinstance(paths[0], str) and not os.path.isabs(paths[0]):
                    paths = [os.path.join(self.image_root, p) for p in paths]
                total = len(paths)
                if total >= F:
                    idxs = np.linspace(0, total - 1, F, dtype=int)
                    picked = [paths[i] for i in idxs]
                else:
                    picked = paths + [paths[-1]] * (F - total)
                items.extend(picked)

            # 2) video_path만 있을 때 → 디코드 + 균등 F장
            elif "video_path" in c and self.allow_video_decode:
                vid = c["video_path"]
                if self.image_root is not None and isinstance(vid, str) and not os.path.isabs(vid):
                    vid = os.path.join(self.image_root, vid)
                frames = _decode_video_uniform(vid, self.min_pixels, self.max_pixels, self.video_fps, F)
                if len(frames) == 0:
                    raise RuntimeError(f"Video decode failed or empty: {vid}")
                items.extend(frames[:F])

            else:
                raise RuntimeError(
                    "Each candidate must provide 'frames' or (allow_video_decode=True and 'video_path')."
                )

        return items

    def _pack_inputs(
        self,
        prompt: str,
        images_or_paths: list[Union[str, Image.Image]],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict]:
        """
        processor를 통해 input_ids / attention_mask / position_ids 생성.
        Qwen2VL이면 mRoPE position_ids 포함.
        """
        messages = [{"role": "user", "content": prompt}]
        prompt_text = self.processor.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=False
        )

        # 이미지 로드/리사이즈
        if len(images_or_paths) == 0:
            processed_images = None
        else:
            processed_images: list[ImageObject] = []
            for im in images_or_paths:
                if isinstance(im, (str, bytes, dict, Image.Image)):
                    try:
                        processed_img = _process_image(im, self.min_pixels, self.max_pixels)
                        processed_images.append(processed_img)
                    except Exception as e:
                        processed_images.append(im)  # 원본 사용
                else:
                    processed_images.append(im)  # PIL.Image 가정
        
        model_inputs = self.processor(
            images=processed_images,
            text=[prompt_text],
            add_special_tokens=False,
            return_tensors="pt",
        )
        input_ids = model_inputs.pop("input_ids")[0]
        attention_mask = model_inputs.pop("attention_mask")[0]
        

        # Qwen2VL mRoPE
        if self.processor is not None and "Qwen2VLImageProcessor" in self.processor.image_processor.__class__.__name__:
            vision_position_ids = get_rope_index(
                self.processor,
                input_ids=input_ids,
                image_grid_thw=model_inputs.get("image_grid_thw", None),
                video_grid_thw=model_inputs.get("video_grid_thw", None),
                second_per_grid_ts=model_inputs.get("second_per_grid_ts", None),
                attention_mask=attention_mask,
            )  # (3, seq_len)
            text_position_ids = torch.arange(len(input_ids)).unsqueeze(0)  # (1, seq_len)
            position_ids = torch.cat((text_position_ids, vision_position_ids), dim=0)  # (4, seq_len)
        else:
            position_ids = torch.clip(attention_mask.cumsum(dim=0) - 1, min=0)

        # 좌/우 절단+패딩
        input_ids, attention_mask, position_ids = VF.postprocess_data(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            max_length=self.max_prompt_length,
            pad_token_id=self.tokenizer.pad_token_id,
            left_pad=True,
            truncation=self.truncation,
        )
        return input_ids, attention_mask, position_ids, model_inputs

    def _check_length(self, idx: int) -> bool:
        sample = self.samples[idx]
        F = self._resolve_fpc(sample)
        N = self._resolve_num_candidates(sample)
        prompt = self._render_prompt(sample, N, F)
        messages = [{"role": "user", "content": prompt}]
        prompt_text = self.processor.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=False
        )
        input_ids = self.tokenizer([prompt_text], add_special_tokens=False, return_tensors="pt")["input_ids"][0]
        return input_ids.size(0) <= self.max_prompt_length

    # ---------- standard Dataset ----------
    def __getitem__(self, i: int) -> dict[str, Any]:
        idx = self._indices[i]
        sample = self.samples[idx]

        F = self._resolve_fpc(sample)
        N = self._resolve_num_candidates(sample)

        # 1) 프롬프트(Jinja) 렌더
        prompt = self._render_prompt(sample, N, F)

        # 2) cand_id 오름차순 × 후보당 F장 "균등" 샘플링
        images_or_paths = self._gather_frames_uniform(sample, F)

        # 3) processor로 입력 구성
        input_ids, attention_mask, position_ids, model_inputs = self._pack_inputs(prompt, images_or_paths)

        # 4) raw_prompt_ids (길이 체크용)
        raw_prompt_ids = self.tokenizer.encode(prompt, add_special_tokens=False)
        if len(raw_prompt_ids) > self.max_prompt_length:
            if self.truncation == "left":
                raw_prompt_ids = raw_prompt_ids[-self.max_prompt_length:]
            elif self.truncation == "right":
                raw_prompt_ids = raw_prompt_ids[: self.max_prompt_length]
            elif self.truncation == "error":
                raise RuntimeError(
                    f"Prompt length {len(raw_prompt_ids)} exceeds max_prompt_length {self.max_prompt_length}."
                )

        ground_truth = sample.get("ground_truth", {})
        
        # 각 cand_id별 sim_score를 ground_truth에 추가
        cand_sim_scores = {}
        for candidate in sample.get("candidates", []):
            cand_id = candidate.get("candidate_id")
            sim_score = candidate.get("sim_score")
            if cand_id is not None and sim_score is not None:
                cand_sim_scores[cand_id] = sim_score
        
        if cand_sim_scores:
            ground_truth["cand_sim_scores"] = cand_sim_scores
        
        multi_modal_data = {"frames": images_or_paths}

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "position_ids": position_ids,
            "raw_prompt_ids": raw_prompt_ids,
            "ground_truth": ground_truth,
            "multi_modal_data": multi_modal_data,
            "query": sample.get("query"),
            "sample_id": sample.get("sample_id"),
            "num_candidates": N,
            "frames_per_candidate": F,
        }