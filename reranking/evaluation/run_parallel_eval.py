#!/usr/bin/env python3
"""
병렬 평가 실행 스크립트
"""

import os
import json
import subprocess
import argparse
from typing import List, Dict, Any
import time
from tqdm import tqdm
import threading
import queue

def split_dataset(dataset_path: str, num_chunks: int = 7, output_dir: str = "./output") -> List[str]:
    """평가 데이터셋을 청크로 분할"""
    
    # 원본 데이터셋 로드
    with open(dataset_path, 'r') as f:
        dataset = json.load(f)
    
    samples = dataset['samples']
    total_samples = len(samples)
    chunk_size = total_samples // num_chunks
    
    print(f"총 {total_samples}개 샘플을 {num_chunks}개 청크로 분할 (청크당 ~{chunk_size}개)")
    
    chunk_files = []
    
    for i in range(num_chunks):
        start_idx = i * chunk_size
        if i == num_chunks - 1:  # 마지막 청크는 나머지 모두 포함
            end_idx = total_samples
        else:
            end_idx = (i + 1) * chunk_size
        
        chunk_samples = samples[start_idx:end_idx]
        
        # 청크 데이터셋 생성
        chunk_dataset = {
            "metadata": {
                "total_samples": len(chunk_samples),
                "chunk_id": i,
                "total_chunks": num_chunks,
                "original_total": total_samples
            },
            "samples": chunk_samples
        }
        
        # 청크 파일 저장
        chunk_file = f"{output_dir}/evaluation_chunk_{i}.json"
        with open(chunk_file, 'w') as f:
            json.dump(chunk_dataset, f, indent=2)
        
        chunk_files.append(chunk_file)
        print(f"청크 {i}: {len(chunk_samples)}개 샘플 -> {chunk_file}")
    
    return chunk_files

def run_parallel_evaluation(chunk_files: List[str], model_path: str, output_dir: str):
    """병렬 평가 실행"""
    
    os.makedirs(output_dir, exist_ok=True)
    
    # 각 청크별 평가 프로세스 시작
    processes = []
    output_files = []
    chunk_start_time = {}  # 함수 시작 시 초기화
    
    # 전체 샘플 수 계산
    total_samples = 0
    for chunk_file in chunk_files:
        with open(chunk_file, 'r') as f:
            chunk_data = json.load(f)
            total_samples += chunk_data['metadata']['total_samples']
    
    for i, chunk_file in enumerate(chunk_files):
        output_file = os.path.join(output_dir, f"chunk_{i}_results.json")
        output_files.append(output_file)
        
        # 평가 명령어
        cmd = [
            "python3", "/home/work/smoretalk/seo/reranking/evaluation/evaluate_from_dataset.py",
            "--dataset", chunk_file,
            "--model_path", model_path,
            "--output", output_file,
            "--max_samples", "0"  # 모든 샘플 평가
        ]
        
        print(f"청크 {i} 평가 시작")
        
        # 백그라운드에서 프로세스 시작
        process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        processes.append(process)
        chunk_start_time[i] = time.time()
    
    # 전체 진행률 표시를 위한 tqdm
    print(f"\n{len(processes)}개 프로세스 실행 중...")
    
    # 진행률 업데이트를 위한 큐
    progress_queue = queue.Queue()
    
    def monitor_process(process, chunk_id):
        """프로세스 모니터링"""
        stdout, stderr = process.communicate()
        progress_queue.put((chunk_id, process.returncode == 0, stderr.decode() if process.returncode != 0 else ""))
    
    def get_total_progress(output_files, total_samples):
        """전체 진행률 계산"""
        try:
            total_completed = 0
            for output_file in output_files:
                if os.path.exists(output_file):
                    with open(output_file, 'r') as f:
                        data = json.load(f)
                        if 'results' in data:
                            total_completed += len(data['results'])
            
            if total_samples > 0:
                return (total_completed / total_samples) * 100
            return 0
        except:
            return 0
    
    # 각 프로세스에 대한 모니터링 스레드 시작
    threads = []
    for i, process in enumerate(processes):
        thread = threading.Thread(target=monitor_process, args=(process, i))
        thread.start()
        threads.append(thread)
    
    # 각 청크별 개별 진행률 표시 (이미 위에서 초기화됨)
    
    # tqdm으로 진행률 표시 (N/3000 형식)
    with tqdm(total=total_samples, desc="병렬 평가 진행률", unit="샘플") as pbar:
        completed_chunks = 0
        while completed_chunks < len(processes):
            try:
                chunk_id, success, error = progress_queue.get(timeout=1)
                if success:
                    elapsed = time.time() - chunk_start_time.get(chunk_id, time.time())
                    pbar.set_postfix({
                        "완료": f"청크 {chunk_id}",
                        "소요시간": f"{elapsed:.1f}s"
                    })
                else:
                    pbar.set_postfix({
                        "실패": f"청크 {chunk_id}",
                        "오류": error[:30] + "..." if len(error) > 30 else error
                    })
                completed_chunks += 1
            except queue.Empty:
                # 전체 진행률 업데이트
                total_completed = get_total_progress(output_files, total_samples) * total_samples / 100
                pbar.n = int(total_completed)
                pbar.refresh()
                continue
    
    # 모든 스레드 완료 대기
    for thread in threads:
        thread.join()
    
    return output_files

def merge_results(output_files: List[str], final_output: str):
    """결과 병합"""
    
    all_results = []
    total_metrics = {
        "r1_sim_correct": 0,
        "r1_truth_correct": 0,
        "total_samples": 0,
        "ndcg_scores": [],
        "case_type_stats": {}
    }
    
    # tqdm으로 결과 병합 진행률 표시
    with tqdm(total=len(output_files), desc="결과 병합 중", unit="파일") as pbar:
        for output_file in output_files:
            if os.path.exists(output_file):
                with open(output_file, 'r') as f:
                    chunk_results = json.load(f)
                
                all_results.extend(chunk_results['detailed_results'])
                
                # 메트릭 누적
                metrics = chunk_results['metrics']
                total_metrics['r1_sim_correct'] += metrics['r1_sim_correct']
                total_metrics['r1_truth_correct'] += metrics['r1_truth_correct']
                total_metrics['total_samples'] += metrics['total_samples']
                total_metrics['ndcg_scores'].extend(metrics['ndcg_scores'])
                
                # 케이스 타입별 통계 누적
                for case_type, stats in metrics['case_type_stats'].items():
                    if case_type not in total_metrics['case_type_stats']:
                        total_metrics['case_type_stats'][case_type] = {
                            'count': 0, 'r1_sim': 0, 'r1_truth': 0, 'ndcg_scores': []
                        }
                    
                    total_metrics['case_type_stats'][case_type]['count'] += stats['count']
                    total_metrics['case_type_stats'][case_type]['r1_sim'] += stats['r1_sim']
                    total_metrics['case_type_stats'][case_type]['r1_truth'] += stats['r1_truth']
                    total_metrics['case_type_stats'][case_type]['ndcg_scores'].extend(stats['ndcg_scores'])
            
            pbar.update(1)
    
    # 최종 메트릭 계산
    final_metrics = {
        "overall": {
            "R@1 (Sim GT)": total_metrics['r1_sim_correct'] / total_metrics['total_samples'],
            "R@1 (Truth)": total_metrics['r1_truth_correct'] / total_metrics['total_samples'],
            "nDCG@5": sum(total_metrics['ndcg_scores']) / len(total_metrics['ndcg_scores']),
            "total_samples": total_metrics['total_samples']
        },
        "by_case_type": {}
    }
    
    # 케이스 타입별 메트릭 계산
    for case_type, stats in total_metrics['case_type_stats'].items():
        if stats['count'] > 0:
            final_metrics['by_case_type'][case_type] = {
                "R@1 (Sim GT)": stats['r1_sim'] / stats['count'],
                "R@1 (Truth)": stats['r1_truth'] / stats['count'],
                "nDCG@5": sum(stats['ndcg_scores']) / len(stats['ndcg_scores']),
                "count": stats['count']
            }
    
    # 최종 결과 저장
    final_results = {
        "metadata": {
            "total_chunks": len(output_files),
            "total_samples": total_metrics['total_samples'],
            "evaluation_time": time.strftime("%Y-%m-%d %H:%M:%S")
        },
        "metrics": final_metrics,
        "results": all_results
    }
    
    with open(final_output, 'w') as f:
        json.dump(final_results, f, indent=2)
    
    print(f"\n=== 최종 결과 ===")
    print(f"총 샘플: {final_metrics['overall']['total_samples']}")
    print(f"R@1 (Sim GT): {final_metrics['overall']['R@1 (Sim GT)']:.4f}")
    print(f"R@1 (Truth): {final_metrics['overall']['R@1 (Truth)']:.4f}")
    print(f"nDCG@5: {final_metrics['overall']['nDCG@5']:.4f}")
    
    print(f"\n=== 케이스 타입별 결과 ===")
    for case_type, metrics in final_metrics['by_case_type'].items():
        print(f"{case_type}: R@1(Sim)={metrics['R@1 (Sim GT)']:.4f}, R@1(Truth)={metrics['R@1 (Truth)']:.4f}, nDCG@5={metrics['nDCG@5']:.4f}")

def main():
    parser = argparse.ArgumentParser(description="병렬 평가 실행")
    parser.add_argument("--dataset", required=True, help="평가 데이터셋 경로")
    parser.add_argument("--model_path", required=True, help="모델 경로")
    parser.add_argument("--output_dir", default="/tmp/parallel_eval", help="출력 디렉토리")
    parser.add_argument("--final_output", default="parallel_evaluation_results.json", help="최종 결과 파일")
    parser.add_argument("--num_chunks", type=int, default=7, help="청크 수")
    
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    
    print("=== 병렬 평가 시작 ===")
    print(f"데이터셋: {args.dataset}")
    print(f"모델: {args.model_path}")
    print(f"청크 수: {args.num_chunks}")
    
    # 1. 데이터셋 분할
    print("\n1. 데이터셋 분할 중...")
    chunk_files = split_dataset(args.dataset, args.num_chunks, args.output_dir)
    
    # 2. 병렬 평가 실행
    print("\n2. 병렬 평가 실행 중...")
    output_files = run_parallel_evaluation(chunk_files, args.model_path, args.output_dir)
    
    # 3. 결과 병합
    print("\n3. 결과 병합 중...")
    merge_results(output_files, args.final_output)
    
    print(f"\n평가 완료! 결과: {args.final_output}")

if __name__ == "__main__":
    main()
