#!/usr/bin/env python3
import argparse
import json
from typing import List, Tuple
import numpy as np
import faiss
import time
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
import os

def load_simplified_nq_data(file_path, max_samples=None) -> Tuple[List[str], List[str], List[int]]:
    """NQ 데이터셋 로드 (simplified 버전)"""
    print(f"Loading simplified NQ data from {file_path}...")
    
    documents = []
    questions = []
    document_ids = []
    doc_id = 0
    
    with open(file_path, 'r') as f:
        for i, line in enumerate(tqdm(f, desc="Loading NQ data")):
            if max_samples and i >= max_samples:
                break
                
            try:
                example = json.loads(line)
                
                # 질문 추출
                if "question_text" in example and example["question_text"]:
                    questions.append(example["question_text"])
                
                # 문서 추출
                paragraphs = []
                if "long_answer_candidates" in example and "document_tokens" in example:
                    # 일단 top_level이 True인 후보들만 먼저 고려
                    top_level_candidates = [c for c in example["long_answer_candidates"] if c.get("top_level", False)]
                    
                    # top_level 후보가 없으면 모든 후보 사용
                    if not top_level_candidates:
                        candidates_to_use = example["long_answer_candidates"]
                    else:
                        candidates_to_use = top_level_candidates
                    
                    doc_tokens = example["document_tokens"]
                    
                    for candidate in candidates_to_use:
                        start = candidate.get("start_token", 0)
                        end = candidate.get("end_token", 0)
                        
                        # 유효한 범위인지 확인
                        if start < end and end <= len(doc_tokens):
                            # 후보 문단의 토큰 추출
                            tokens = [token.get("token", "") for token in doc_tokens[start:end]]
                            text = " ".join(tokens)
                            
                            # 의미 있는 길이의 텍스트만 저장 (최소 10단어 이상)
                            if len(text.split()) >= 10:
                                paragraphs.append(text)
                
                for para in paragraphs:
                    documents.append(para)
                    document_ids.append(doc_id)
                doc_id += 1
            except json.JSONDecodeError:
                print(f"Error decoding JSON at line {i+1}")
                continue
    
    print(f"Loaded {len(documents)} paragraphs from {doc_id} documents")
    print(f"Loaded {len(questions)} questions")
    
    return documents, questions, document_ids


def build_index(embeddings, args):
    """Build various types of index"""
    dimension = embeddings.shape[1]

    if args.index_type == "flat":
        index = faiss.IndexFlatL2(dimension)
        index.add(embeddings)
        return index
    elif args.index_type == "ivfflat":
        quantizer = faiss.IndexFlatL2(dimension)
        index = faiss.IndexIVFFlat(quantizer, dimension, args.nlist)
        
        if not index.is_trained:
            print(f"Training IVF index with {args.nlist} clusters...")
            index.train(embeddings)
        
        print("Adding vectors to index...")
        index.add(embeddings)
        index.nprobe = args.nprobe
        return index
    elif args.index_type == "ivfpq":
        # IVF PQ 인덱스
        quantizer = faiss.IndexFlatL2(dimension)
        index = faiss.IndexIVFPQ(quantizer, dimension, args.nlist, args.m, args.nbits)
        
        if not index.is_trained:
            print(f"Training IVFPQ index with {args.nlist} clusters, {args.m} subquantizers...")
            index.train(embeddings)
        
        print("Adding vectors to index...")
        index.add(embeddings)
        index.nprobe = args.nprobe
        return index
    else:
        raise NotImplementedError(args.index_type)


def working_test(args):
    print(f"Starting NQ-FAISS {args.index_type.upper()} Vector Database Test")
    
    print("Loading embedding model...")
    model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
    
    # 데이터 로드 (필요에 따라 max_samples 조정)
    documents, questions, document_ids = load_simplified_nq_data(args.input, max_samples=args.max_samples)
    
    # 문서 임베딩 생성
    print("Creating document embeddings...")
    document_embeddings = model.encode(documents, batch_size=args.batch_size, show_progress_bar=True)
    
    # 인덱스 구축
    print(f"Building FAISS {args.index_type.upper()} index...")
    index = build_index(document_embeddings, args)
    print(f"Index contains {index.ntotal} vectors")
    
    # 인덱스 저장
    index_filename = os.path.join(args.index_path, f"{args.index_type}.bin")
    faiss.write_index(index, index_filename)
    print(f"Index saved to {index_filename}")
    
    # 검색 테스트를 위한 쿼리 선택 (처음 5개 질문)
    test_queries = questions[:5]
    
    # 쿼리 임베딩 생성
    print("Creating query embeddings...")
    query_embeddings = model.encode(test_queries, show_progress_bar=True)
    
    # 검색 수행
    print("Performing search...")
    k = 5  # top-k 결과
    start_time = time.time()
    distances, indices = index.search(query_embeddings, k)
    search_time = time.time() - start_time
    
    print(f"Search completed in {search_time:.4f} seconds for {len(test_queries)} queries")
    print(f"Average search time per query: {search_time/len(test_queries):.6f} seconds")
    
    # 결과 출력
    results = []
    for i, (query, idxs, dists) in enumerate(zip(test_queries, indices, distances)):
        result = {
            "query": query,
            "results": []
        }
        
        print(f"\nQuery {i+1}: {query}")
        for rank, (idx, dist) in enumerate(zip(idxs, dists)):
            if idx < len(documents):
                doc_text = documents[idx][:200] + "..." if len(documents[idx]) > 200 else documents[idx]
                doc_id = document_ids[idx]
                
                print(f"  Result {rank+1}: [Document {doc_id}] Distance: {dist:.4f}")
                print(f"  {doc_text}")
                
                result["results"].append({
                    "rank": rank+1,
                    "document_id": int(doc_id),
                    "distance": float(dist),
                    "text": doc_text
                })
        
        results.append(result)
    
    # 결과 저장
    results_filename = f"{args.out_path}/{args.index_type}_search_results.json"
    with open(results_filename, 'w') as f:
        json.dump(results, f, indent=2)

    index_info = {
        "index_type": args.index_type,
        "vectors": index.ntotal,
        "dimension": document_embeddings.shape[1],
        "parameters": {
            "nlist": getattr(args, "nlist", None),
            "nprobe": getattr(args, "nprobe", None),
            "m": getattr(args, "m", None),
            "nbits": getattr(args, "nbits", None),
            "M": getattr(args, "M", None),
            "efConstruction": getattr(args, "efConstruction", None),
            "efSearch": getattr(args, "efSearch", None)
        },
        "search_time": search_time,
        "avg_query_time": search_time/len(test_queries) if test_queries else None
    }

    index_info_file_name = f"{args.out_path}/{args.index_type}_index_info.json"
    with open(index_info_file_name, 'w') as f:
        json.dump(index_info, f, indent=2)    
    
    print("\nTest completed. Results saved to", results_filename)
    return index, document_embeddings

def run_io_performance_test(args):
    """Performance test for disk I/O and access latency"""
    try:
        import psutil
    except ImportError:
        print("psutil module not found. Install with: pip install psutil")
        return
    
    print("\n===== RUNNING DISK I/O PERFORMANCE TEST =====")
    process = psutil.Process(os.getpid())
    
    # 임베딩 모델 로드
    print("Loading embedding model...")
    model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
    
    # 데이터 로드
    print(f"Loading data (max {args.max_samples} samples)...")
    documents, questions, _ = load_simplified_nq_data(args.input, max_samples=args.max_samples)
    
    # 1. 임베딩 생성 시간 측정
    print("Creating document embeddings...")
    start_embed_time = time.time()
    document_embeddings = model.encode(documents, batch_size=args.batch_size, show_progress_bar=True)
    embed_time = time.time() - start_embed_time
    print(f"Embedding generation time: {embed_time:.4f} seconds")
    
    # 2. 인덱스 구축 시간 측정
    print("Building FAISS index...")
    start_build_time = time.time()
    index = build_index(document_embeddings, args)
    build_time = time.time() - start_build_time
    print(f"Index build time: {build_time:.4f} seconds")
    
    # 3. 인덱스 저장 전 디스크 I/O 측정
    index_filename = os.path.join(args.index_path, f"{args.index_type}.bin")
    print("Saving index to disk...")
    io_before = process.io_counters()
    start_save_time = time.time()
    faiss.write_index(index, index_filename)
    save_time = time.time() - start_save_time
    io_after = process.io_counters()
    
    # I/O 통계 계산
    write_bytes = io_after.write_bytes - io_before.write_bytes
    file_size = os.path.getsize(index_filename)
    
    print(f"I/O time for saving index: {save_time:.4f} seconds")
    print(f"Index file size: {file_size / (1024 * 1024):.4f} MB")
    print(f"Actual disk write: {write_bytes / (1024 * 1024):.4f} MB")
    print(f"Write throughput: {file_size / (save_time * 1024 * 1024):.4f} MB/s")
    
    # 4. 인덱스 로드 측정 (캐시 효과 테스트 포함)
    print("\nLoading index from disk...")
    if args.cache_test:
        print("\nTesting cold start load (trying to clear cache)...")
        try:
            print("Attempting to clear cache (may require root privileges)...")
            os.system("sync")
            os.system("echo 3 > /proc/sys/vm/drop_caches")
        except:
            print("Could not clear cache (need root permissions)")
    
    # 인덱스 로드 측정
    # index_filename = os.path.join(args.index_path, f"{args.index_type}.bin")
    io_before = process.io_counters()
    start_load_time = time.time()
    loaded_index = faiss.read_index(index_filename)
    load_time = time.time() - start_load_time
    io_after = process.io_counters()
    
    # I/O 통계 계산
    read_bytes = io_after.read_bytes - io_before.read_bytes
    
    print(f"Index load time: {load_time:.4f} seconds")
    print(f"Actual disk read: {read_bytes / (1024 * 1024):.4f} MB")
    print(f"Read throughput: {file_size / (load_time * 1024 * 1024):.4f} MB/s")
    
    if args.cache_test:
        # 캐시 효과 측정 (웜 스타트)
        print("\nTesting cache effect (warm start)...")
        start_time = time.time()
        _ = faiss.read_index(index_filename)
        warm_load_time = time.time() - start_time
        print(f"Warm load time: {warm_load_time:.4f} seconds")
        print(f"Cache speedup: {load_time / warm_load_time:.4f}x")
    
    # 검색 성능 측정 (샘플 쿼리)
    if len(questions) > 0:
        print("\nTesting search performance...")
        test_queries = questions[:5]
        query_embeddings = model.encode(test_queries, show_progress_bar=True)
        
        start_search_time = time.time()
        _, _ = loaded_index.search(query_embeddings, 5)
        search_time = time.time() - start_search_time
        
        print(f"Search time for {len(test_queries)} queries: {search_time:.4f} seconds")
        print(f"Average search time per query: {search_time/len(test_queries):.6f} seconds")
    
    # 전체 시간 비율 계산
    total_time = embed_time + build_time + save_time + load_time
    print("\n===== PERFORMANCE SUMMARY =====")
    print(f"Total processing time: {total_time:.4f} seconds")
    print(f"Embedding: {embed_time:.4f}s ({embed_time/total_time*100:.2f}%)")
    print(f"Index building: {build_time:.4f}s ({build_time/total_time*100:.2f}%)")
    print(f"Index saving: {save_time:.4f}s ({save_time/total_time*100:.2f}%)")
    print(f"Index loading: {load_time:.4f}s ({load_time/total_time*100:.2f}%)")

    # 인덱스 정보 저장
    perf_info = {
        "index_type": args.index_type,
        "vectors": index.ntotal,
        "dimension": document_embeddings.shape[1],
        "file_size_mb": file_size / (1024 * 1024),
        "parameters": {
            "nlist": getattr(args, "nlist", None),
            "nprobe": getattr(args, "nprobe", None),
            "m": getattr(args, "m", None),
            "nbits": getattr(args, "nbits", None),
            "M": getattr(args, "M", None),
            "efConstruction": getattr(args, "efConstruction", None),
            "efSearch": getattr(args, "efSearch", None)
        },
        "timings": {
            "embedding_time": embed_time,
            "build_time": build_time,
            "save_time": save_time,
            "load_time": load_time,
            "search_time": search_time if len(questions) > 0 else None,
            "avg_query_time": search_time/len(test_queries) if len(questions) > 0 else None
        },
        "io": {
            "write_bytes_mb": write_bytes / (1024 * 1024),
            "write_throughput_mbs": file_size / (save_time * 1024 * 1024),
            "read_bytes_mb": read_bytes / (1024 * 1024),
            "read_throughput_mbs": file_size / (load_time * 1024 * 1024)
        }
    }
    
    if args.cache_test:
        perf_info["cache"] = {
            "cold_load_time": load_time,
            "warm_load_time": warm_load_time,
            "speedup": load_time / warm_load_time
        }
    
    # 성능 정보 저장
    results_filename = f"{args.out_path}/{args.index_type}_index_info.json"
    with open(results_filename, 'w') as f:
        json.dump(perf_info, f, indent=2)
        
    print("==============================")

def parse_arguments():
    """Parse the command arguments"""
    parser = argparse.ArgumentParser(description="FAISS Index Test for NQ")

    # Basic info
    parser.add_argument("--out_path", "-o", type=str, default="/home/dhmin/faiss/dh_demos/results", help="Path for storing results")
    parser.add_argument("--input", "-i", type=str, default="/mnt/sda/nq_dataset/v1.0-simplified_nq-dev-all.jsonl", help="Path for loading datasets")
    parser.add_argument("--index_path", "-index", type=str, default="/mnt/sda/nq_dataset/", help="Path for storing index")
    parser.add_argument("--index_type", "-t", default="flat", choices=["flat", "ivf", "ivfpq", "ivfflat", "hnsw", "ivfhnsw"], help="Type of FAISS index to use")
    parser.add_argument("--batch_size", type=int, default=32, help="Maximum batch size")
    
    # Info for IVF-based index
    parser.add_argument("--nlist", type=int, default=100, help="Number of clusters of IVF-based indexes")
    parser.add_argument("--nprobe", type=int, default=10, help="Number of clusters to visit during search for IVF-based indexes")
    
    # Info for IVFPQ-based index
    parser.add_argument("--m", type=int, default=8, help="Number of subquantizers for PQ-based indexes")
    parser.add_argument("-nbits", type=int, default=8, help="Number of bits per subquantizer (usually 8)")
    
    # Performance test-related info
    parser.add_argument("--perf_test", action="store_true", help="Run performance tests for disk I/O and latency")
    parser.add_argument("--cache_test", action="store_true", help="Test cache effects on loading index")
    parser.add_argument("--max_samples", type=int, default=None, help="Maximum number of samples to use for testing")

    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = parse_arguments()
    os.makedirs(args.out_path, exist_ok=True)

    if args.perf_test:
        run_io_performance_test(args)
    else:
        working_test(args)