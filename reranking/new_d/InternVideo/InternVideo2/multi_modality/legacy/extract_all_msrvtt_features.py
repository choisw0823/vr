"""
MSRVTT 전체 비디오 및 텍스트 특징 추출 스크립트
- 모든 비디오 특징 추출 (10,000개)
- 모든 텍스트 특징 추출 (학습용 + 테스트용)
- 나중에 빠른 similarity 계산을 위해 저장
"""

import json
import logging
import os
import sys
import pickle
import ast
from collections import defaultdict
from tqdm import tqdm

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.distributed as dist
from torch.utils.data import DataLoader, Dataset

from dataset import pt_dataset, ret_dataset
from models.internvideo2_stage2_audiovisual import InternVideo2_Stage2_audiovisual
from models.criterions import get_sim
from tasks.shared_utils import setup_model
from utils.config_utils import setup_main
from utils.distributed import get_rank, is_main_process

logger = logging.getLogger(__name__)

class MSRVTTFeatureDataset(Dataset):
    """MSRVTT 전체 데이터셋을 위한 커스텀 데이터셋"""
    
    def __init__(self, video_root, annotation_files, tokenizer, max_txt_length=77):
        self.video_root = video_root
        self.tokenizer = tokenizer
        self.max_txt_length = max_txt_length
        
        # 모든 annotation 파일 로드
        self.video_data = {}  # video_id -> video_info
        self.text_data = []   # 모든 텍스트 쿼리
        
        for ann_file in annotation_files:
            logger.info(f"Loading annotation file: {ann_file}")
            with open(ann_file, 'r') as f:
                data = json.load(f)
            
            for item in data:
                video_id = item['video_id']
                video_path = os.path.join(video_root, item['video'])
                
                # 비디오 정보 저장 (중복 제거)
                if video_id not in self.video_data:
                    self.video_data[video_id] = {
                        'video_id': video_id,
                        'video_path': video_path,
                        'video_file': item['video']
                    }
                
                # 텍스트 쿼리 저장
                text_item = {
                    'video_id': video_id,
                    'caption': item['caption'],
                    'source': item.get('source', 'MSR-VTT'),
                    'category': item.get('category', 0),
                    'split': self._get_split_from_filename(ann_file)
                }
                self.text_data.append(text_item)
        
        self.video_ids = list(self.video_data.keys())
        logger.info(f"Total videos: {len(self.video_ids)}")
        logger.info(f"Total text queries: {len(self.text_data)}")
    
    def _get_split_from_filename(self, filename):
        """파일명에서 split 정보 추출"""
        if 'test' in filename:
            return 'test'
        elif 'train_7k' in filename:
            return 'train_7k'
        elif 'train_9k' in filename:
            return 'train_9k'
        else:
            return 'unknown'
    
    def __len__(self):
        return len(self.video_ids)
    
    def get_video_info(self, idx):
        """비디오 정보 반환"""
        video_id = self.video_ids[idx]
        return self.video_data[video_id]
    
    def get_texts_for_video(self, video_id):
        """특정 비디오의 모든 텍스트 쿼리 반환"""
        return [item for item in self.text_data if item['video_id'] == video_id]

def extract_video_features(model, video_path, device, config):
    """단일 비디오의 특징 추출 - 기존 retrieval_utils 사용"""
    try:
        # 기존 eval과 동일한 방식으로 비디오 로드
        from dataset.video_utils import VIDEO_READER_FUNCS
        
        # 비디오 파일 읽기 (올바른 인자 전달)
        video_reader = VIDEO_READER_FUNCS['decord']
        video_input = config.inputs.video_input
        
        # num_frames를 정수로 변환
        num_frames = int(video_input.num_frames) if hasattr(video_input, 'num_frames') else 8
        logger.info(f"Loading video {video_path} with {num_frames} frames")
        
        video_data = video_reader(video_path, num_frames, sample='rand', fix_start=None, max_num_frames=-1, client=None, trimmed30=False)
        
        # video_data는 tuple이고 첫 번째 요소가 비디오 텐서 (T, 3, H, W)
        if isinstance(video_data, tuple):
            frames = video_data[0]  # (T, 3, H, W)
        elif isinstance(video_data, torch.Tensor):
            frames = video_data
        else:
            frames = torch.tensor(video_data)
        
        # 기존 eval과 동일한 전처리: PIL Image로 변환 후 transform 적용
        from PIL import Image
        from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize, InterpolationMode
        
        transform = Compose([
            Resize(224, interpolation=InterpolationMode.BICUBIC),
            CenterCrop(224),
            lambda image: image.convert("RGB"),
            ToTensor(),
            Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
        ])
        
        # 각 프레임을 PIL Image로 변환 후 transform 적용
        processed_frames = []
        for i in range(frames.shape[0]):
            frame = frames[i].permute(1, 2, 0)  # (3, H, W) -> (H, W, 3)
            frame_np = (frame * 255).byte().numpy()  # uint8로 변환
            pil_image = Image.fromarray(frame_np)
            transformed_frame = transform(pil_image)
            processed_frames.append(transformed_frame)
        
        # 텐서로 스택하고 배치 차원 추가
        video_tensor = torch.stack(processed_frames).unsqueeze(0).to(device)  # (1, T, 3, 224, 224)
        
        with torch.no_grad():
            # 비디오 특징 추출 - 기존 eval과 동일 (extract_vision_feats 방식)
            # DistributedDataParallel로 래핑된 경우 model.module 사용
            actual_model = model.module if hasattr(model, 'module') else model
            vision_feat, pooled_vision_feat = actual_model.encode_vision(video_tensor, test=True)
            video_proj = actual_model.vision_proj(pooled_vision_feat)
            
        return video_proj.cpu()
        
    except Exception as e:
        logger.warning(f"Failed to extract features for {video_path}: {e}")
        return None

def extract_text_features(model, texts, tokenizer, device, video_feature=None):
    """텍스트 리스트의 특징 추출 - 극도 메모리 절약 버전"""
    from tasks.retrieval_utils import extract_text_feats
    
    try:
        # 기존 eval과 동일한 방식으로 텍스트 특징 추출
        # DistributedDataParallel로 래핑된 경우 model.module 사용
        actual_model = model.module if hasattr(model, 'module') else model
        
        # 디버깅: 텍스트 길이 확인 (간소화)
        logger.info(f"Extracting text features for {len(texts)} texts")
        
        # 극도 메모리 절약을 위해 하나씩 처리
        all_text_proj = []
        
        for i, text in enumerate(texts):
            # 메모리 정리
            torch.cuda.empty_cache()
            
            # 하나씩 처리
            result = extract_text_feats([text], 77, tokenizer, actual_model, device)
            
            # 반환값이 tuple인지 확인하고 적절히 처리
            if isinstance(result, tuple):
                if len(result) == 2:
                    text_feats, text_atts = result
                elif len(result) == 1:
                    text_feats = result[0]
                    text_atts = None
                else:
                    logger.warning(f"Unexpected tuple length: {len(result)}")
                    continue
            else:
                text_feats = result
                text_atts = None
            
            # projection 후 결과 저장 (기존 eval과 동일)
            text_proj = actual_model.text_proj(text_feats[:, 0])  # CLS token만 사용
            
            # CPU로 이동하여 메모리 절약
            all_text_proj.append(text_proj.cpu())
            
            # 메모리 정리
            del text_feats, text_proj
            if text_atts is not None:
                del text_atts
            torch.cuda.empty_cache()
        
        if all_text_proj:
            # 모든 배치 결과를 합치기
            final_text_proj = torch.cat(all_text_proj, dim=0)
            logger.info(f"Final text_proj shape: {final_text_proj.shape}")
            
            # 메모리 정리
            del all_text_proj
            torch.cuda.empty_cache()
            
            return final_text_proj
        else:
            return None
            
    except Exception as e:
        logger.warning(f"Failed to extract text features: {e}")
        return None

def _load_single_video_feature(video_feat_file, video_id):
    """단일 비디오 특징만 로드"""
    try:
        with open(video_feat_file, 'rb') as f:
            all_video_features = pickle.load(f)
        
        # 단일 비디오 특징만 반환
        if video_id in all_video_features:
            feature = all_video_features[video_id]
            # 원본 데이터 해제
            del all_video_features
            return feature
        else:
            del all_video_features
            return None
    except Exception as e:
        logger.warning(f"Failed to load video feature for {video_id}: {e}")
        return None

def _load_video_features_lazy(video_feat_file, video_ids):
    """비디오 특징을 lazy loading으로 필요한 것만 로드"""
    try:
        with open(video_feat_file, 'rb') as f:
            all_video_features = pickle.load(f)
        
        # 필요한 비디오 특징만 반환
        needed_features = {}
        for video_id in video_ids:
            if video_id in all_video_features:
                needed_features[video_id] = all_video_features[video_id]
        
        # 원본 데이터 해제
        del all_video_features
        return needed_features
    except Exception as e:
        logger.warning(f"Failed to load video features: {e}")
        return {}

def _save_single_text_feature(video_id, features, texts, output_dir):
    """단일 텍스트 특징 즉시 저장"""
    feature_file = os.path.join(output_dir, f"text_features_{video_id}.pkl")
    metadata_file = os.path.join(output_dir, f"text_metadata_{video_id}.pkl")
    
    with open(feature_file, 'wb') as f:
        pickle.dump({video_id: features}, f)
    with open(metadata_file, 'wb') as f:
        pickle.dump({video_id: texts}, f)

def _save_intermediate_results(features, metadata, output_dir, prefix):
    """중간 결과 저장 (메모리 절약)"""
    if not features:
        return
    
    # 중간 저장 디렉토리 생성
    intermediate_dir = os.path.join(output_dir, "intermediate")
    os.makedirs(intermediate_dir, exist_ok=True)
    
    # 파일명에 타임스탬프 추가
    import time
    timestamp = int(time.time())
    
    feat_file = os.path.join(intermediate_dir, f"{prefix}_features_{timestamp}.pkl")
    meta_file = os.path.join(intermediate_dir, f"{prefix}_metadata_{timestamp}.pkl")
    
    with open(feat_file, 'wb') as f:
        pickle.dump(features, f)
    with open(meta_file, 'wb') as f:
        pickle.dump(metadata, f)
    
    logger.info(f"Intermediate {prefix} features saved: {len(features)} items")

def main(config):
    # Setup basic logging
    logging.basicConfig(level=logging.INFO)
    logger.info(f"Config: {config}")
    
    # 텍스트만 추출할지 확인 (명령행 인자로 제어)
    text_only_mode = getattr(config, 'text_only', False)
    logger.info(f"Text-only mode: {text_only_mode}")
    
    # Setup model
    model_cls = InternVideo2_Stage2_audiovisual
    model, model_without_ddp, optimizer, scheduler, scaler, tokenizer, start_epoch, global_step = setup_model(
        config, model_cls, pretrain=False, find_unused_parameters=config.model.get("find_unused_parameters", False)
    )
    
    model.eval()
    
    # MSRVTT annotation 파일들
    annotation_files = [
        "/home/work/smoretalk/seo/reranking/dataset/MSR-VTT/msrvtt_test_1k.json",
        "/home/work/smoretalk/seo/reranking/dataset/MSR-VTT/msrvtt_train_9k.json"
    ]
    
    video_root = "/home/work/smoretalk/seo/reranking/dataset/MSR-VTT/video"
    
    # 데이터셋 생성
    dataset = MSRVTTFeatureDataset(video_root, annotation_files, tokenizer)
    
    # 출력 디렉토리 설정
    output_dir = config.output_dir if hasattr(config, 'output_dir') else './outputs/msrvtt_features'
    os.makedirs(output_dir, exist_ok=True)
    
    logger.info("Starting feature extraction...")
    
    # 비디오 특징 추출 (배치별 처리) - text_only_mode가 아닌 경우에만
    if not text_only_mode:
        logger.info("Extracting video features for all videos (batch processing)...")
        video_features = {}
        video_metadata = {}
        
        batch_size = 1000  # 배치 크기
        total_videos = len(dataset)
        
        for batch_start in tqdm(range(0, total_videos, batch_size), desc="Processing video batches"):
            batch_end = min(batch_start + batch_size, total_videos)
            logger.info(f"Processing videos {batch_start}-{batch_end-1}...")
            
            for i in range(batch_start, batch_end):
                video_info = dataset.get_video_info(i)
                video_id = video_info['video_id']
                video_path = video_info['video_path']
                
                if os.path.exists(video_path):
                    features = extract_video_features(model, video_path, config.device, config)
                    if features is not None:
                        video_features[video_id] = features
                        video_metadata[video_id] = video_info
                else:
                    logger.warning(f"Video file not found: {video_path}")
            
            # 배치별 중간 저장 (메모리 절약)
            if batch_start % (batch_size * 5) == 0:  # 5배치마다 중간 저장
                logger.info(f"Intermediate save at batch {batch_start//batch_size}")
                _save_intermediate_results(video_features, video_metadata, output_dir, "video")
    else:
        # 텍스트만 추출하는 경우: 비디오 특징 파일 경로만 저장 (메모리 절약)
        logger.info("Text-only mode: Using lazy loading for video features...")
        
        # 기존 비디오 특징 파일 찾기
        video_feat_file = None
        search_dirs = [
            "/home/work/smoretalk/seo/reranking/new_d/InternVideo/InternVideo2/multi_modality/outputs/feature_result",
            os.path.join(output_dir, "msrvtt_features"),
            "./outputs/msrvtt_features/extract_all_features_20250920_212209/msrvtt_features",
            "./outputs/msrvtt_features"
        ]
        
        for search_dir in search_dirs:
            potential_file = os.path.join(search_dir, "video_features.pkl")
            if os.path.exists(potential_file):
                video_feat_file = potential_file
                break
        
        if not video_feat_file or not os.path.exists(video_feat_file):
            logger.error(f"Video features file not found in any of the search directories")
            logger.error("Search directories:")
            for search_dir in search_dirs:
                logger.error(f"  - {search_dir}")
            logger.error("Please run without text_only mode first to extract video features")
            return
        
        logger.info(f"Video features file found: {video_feat_file}")
        video_metadata = {}  # 텍스트만 추출하는 경우 메타데이터는 필요 없음
    
    # 텍스트 특징 추출 (배치별 처리)
    logger.info("Extracting text features for all videos (batch processing)...")
    # 텍스트 특징은 즉시 저장하므로 딕셔너리 불필요
    # text_features = {}
    # text_metadata = {}
    
    # 비디오별로 텍스트 그룹화
    video_texts = defaultdict(list)
    for text_item in dataset.text_data:
        video_texts[text_item['video_id']].append(text_item)
    
    # 모든 비디오 ID들 처리 (배치별로 처리하여 메모리 절약)
    if text_only_mode:
        # 텍스트만 추출하는 경우: 모든 비디오 ID를 텍스트 데이터에서 가져오기
        all_video_ids = list(set(item['video_id'] for item in dataset.text_data))
    else:
        # 비디오 특징도 추출하는 경우: 기존 방식
        all_video_ids = list(video_features.keys())
    
    batch_size = getattr(config, 'text_batch_size', 3)  # 텍스트 처리 배치 크기 (극도로 작게)
    
    for batch_start in tqdm(range(0, len(all_video_ids), batch_size), desc="Processing text batches"):
        batch_end = min(batch_start + batch_size, len(all_video_ids))
        batch_video_ids = all_video_ids[batch_start:batch_end]
        
        logger.info(f"Processing text batch {batch_start//batch_size + 1}: videos {batch_start}-{batch_end-1}")
        
        # 비디오 특징은 하나씩 로드하므로 배치 로딩 제거
        
        for video_idx, video_id in enumerate(batch_video_ids):
            # 텍스트만 추출하므로 비디오 특징 불필요
                texts = video_texts[video_id]
                # 캡션을 리스트로 변환 (test/train 구분)
                captions = []
                for item in texts:
                    caption = item['caption']
                    split = item.get('split', 'unknown')
                    
                    if isinstance(caption, str):
                        # test 데이터: 문자열 그대로 사용
                        if split == 'test':
                            captions.append(caption)
                        else:
                            # train 데이터: 문자열을 리스트로 변환 시도
                            try:
                                caption_list = ast.literal_eval(caption)
                                if isinstance(caption_list, list):
                                    captions.extend(caption_list)
                                else:
                                    captions.append(caption)
                            except:
                                # 변환 실패시 단일 캡션으로 처리
                                captions.append(caption)
                    elif isinstance(caption, list):
                        # train 데이터: 리스트 그대로 사용
                        captions.extend(caption)
                    else:
                        # 기타 경우: 문자열로 변환
                        captions.append(str(caption))
                
                if captions:  # 캡션이 있는 경우만 처리
                    # 메모리 정리
                    torch.cuda.empty_cache()
                    
                    features = extract_text_features(model, captions, tokenizer, config.device)
                    
                    if features is not None:
                        # 즉시 저장하고 메모리에서 제거
                        _save_single_text_feature(video_id, features, texts, output_dir)
                        logger.info(f"Extracted and saved {len(captions)} text features for video {video_id}")
                        
                        # 텍스트 특징 메모리 해제
                        del features
                    else:
                        logger.warning(f"No valid text features extracted for video {video_id}")
                else:
                    logger.warning(f"No captions found for video {video_id}")
            
            # 메모리 정리
            torch.cuda.empty_cache()
            
            # 주기적 메모리 정리 (매 10개 비디오마다)
            if (video_idx + 1) % 10 == 0:
                torch.cuda.empty_cache()
                logger.info(f"Memory cleanup after processing {video_idx + 1} videos in current batch")
        
        # 배치별 중간 저장 (메모리 절약) - 텍스트 특징은 이미 개별 저장됨
        if batch_start % (batch_size * 2) == 0:  # 2배치마다 로그
            logger.info(f"Completed batch {batch_start//batch_size + 1}")
    
    # 결과 저장
    output_dir = os.path.join(config.output_dir, "msrvtt_features")
    os.makedirs(output_dir, exist_ok=True)
    
    # 비디오 특징 저장 (text_only_mode가 아닌 경우에만)
    if not text_only_mode:
        video_feat_file = os.path.join(output_dir, "video_features.pkl")
        with open(video_feat_file, 'wb') as f:
            pickle.dump(video_features, f)
        
        video_meta_file = os.path.join(output_dir, "video_metadata.pkl")
        with open(video_meta_file, 'wb') as f:
            pickle.dump(video_metadata, f)
    
    # 텍스트 특징 저장
    text_feat_file = os.path.join(output_dir, "text_features.pkl")
    with open(text_feat_file, 'wb') as f:
        pickle.dump(text_features, f)
    
    text_meta_file = os.path.join(output_dir, "text_metadata.pkl")
    with open(text_meta_file, 'wb') as f:
        pickle.dump(text_metadata, f)
    
    # 통계 정보 저장
    stats = {
        'total_videos': len(dataset.video_ids),
        'extracted_videos': len(video_features) if not text_only_mode else len(all_video_ids),
        'total_texts': len(dataset.text_data),
        'extracted_texts': sum(len(metadata) for metadata in text_metadata.values()),
        'video_feature_dim': list(video_features.values())[0].shape[1] if video_features and not text_only_mode else 0,
        'text_feature_dim': list(text_features.values())[0].shape[1] if text_features else 0,
        'text_only_mode': text_only_mode
    }
    
    stats_file = os.path.join(output_dir, "extraction_stats.json")
    with open(stats_file, 'w') as f:
        json.dump(stats, f, indent=2)
    
    logger.info(f"✅ Feature extraction completed!")
    logger.info(f"📊 Videos: {stats['extracted_videos']}/{stats['total_videos']}")
    logger.info(f"📝 Texts: {stats['extracted_texts']}/{stats['total_texts']}")
    logger.info(f"💾 Results saved to {output_dir}")

if __name__ == "__main__":
    # 기본 config 파일 설정
    if len(sys.argv) == 1:
        sys.argv.extend([
            'scripts/evaluation/stage2/zero_shot/6B/config_msrvtt.py',
            'output_dir', './outputs/msrvtt_features/extraction',
            'evaluate', 'True',
            'pretrained_path', '/home/work/smoretalk/seo/reranking/new_d/InternVideo2-Stage2_6B-224p-f4/internvideo2-s2_6b-224p-f4_with_audio_encoder.pt'
        ])
    
    # text_only 옵션 확인
    text_only = '--text_only' in sys.argv
    if text_only:
        sys.argv.remove('--text_only')
    
    config = setup_main()
    
    # text_only 옵션을 config에 추가
    config.text_only = text_only
    
    main(config)
