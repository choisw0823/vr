# Qwen VL 2.5 Reranking Evaluation

Qwen VL 2.5 모델을 사용한 video-text reranking 평가 코드입니다.

## 파일 구조

- `evaluate_qwen_vl_reranking.py`: 메인 평가 코드
- `run_evaluation.sh`: 평가 실행 스크립트
- `README.md`: 사용법 설명

## 사용법

### 1. 환경 설정

```bash
# 필요한 패키지 설치
pip install torch transformers qwen-vl-utils pillow tqdm numpy
```

### 2. 경로 설정

`run_evaluation.sh` 파일에서 다음 경로들을 수정하세요:

```bash
MODEL_PATH="/path/to/your/qwen-vl-2.5-model"  # 훈련된 모델 경로
VIDEO_BASE_PATH="/path/to/msrvtt/videos"      # MSR-VTT 비디오 파일 경로
```

### 3. 평가 실행

```bash
# 스크립트 실행 권한 부여
chmod +x run_evaluation.sh

# 평가 실행
./run_evaluation.sh
```

### 4. 직접 실행

```bash
python3 evaluate_qwen_vl_reranking.py \
    --model_path "/path/to/your/model" \
    --data_path "/path/to/test_data.pkl" \
    --video_base_path "/path/to/videos" \
    --output_path "results.json" \
    --max_samples 1000
```

## 평가 방식

1. **데이터 로드**: test 데이터에서 1000개 샘플 사용
2. **후보 생성**: 각 쿼리마다 정답 비디오 + 4개 랜덤 비디오 (총 5개)
3. **모델 예측**: Qwen VL 2.5가 5개 비디오를 순위 매김
4. **평가 지표**: R@1, R@5, R@10 계산

## 출력 형식

```json
{
  "recalls": {
    "1": 0.533,
    "5": 0.753,
    "10": 0.830
  },
  "results": [...],
  "total_samples": 100,
  "model_path": "/path/to/model"
}
```

## 주요 기능

- **비디오 프레임 추출**: 각 비디오에서 12개 프레임 자동 추출
- **순위 파싱**: 모델 응답에서 `<answer>` 태그 내 순위 추출
- **에러 처리**: 비디오 로드 실패 시 기본값 처리
- **배치 처리**: GPU 메모리 효율적인 처리

## 주의사항

1. **비디오 경로**: MSR-VTT 비디오 파일들이 올바른 경로에 있어야 함
2. **메모리**: GPU 메모리 부족 시 `max_samples`로 샘플 수 조절
3. **모델 형식**: Qwen VL 2.5 모델이 올바른 형식으로 저장되어 있어야 함

## 문제 해결

### 메모리 부족
```bash
# 샘플 수 줄이기
--max_samples 50
```

### 비디오 로드 실패
- 비디오 파일 경로 확인
- 파일 권한 확인
- 비디오 형식 지원 확인

### 모델 로드 실패
- 모델 경로 확인
- `trust_remote_code=True` 설정 확인
- GPU 메모리 확인

