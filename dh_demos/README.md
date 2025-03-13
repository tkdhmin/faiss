# Getting started 
```bash
# 기본 Flat 인덱스 테스트
python script.py --index_type flat

# IVF Flat 인덱스 테스트 (클러스터 수 및 탐색 수 조정)
python script.py --index_type ivfflat --nlist 200 --nprobe 20

# IVFPQ 인덱스 테스트
python script.py --index_type ivfpq --nlist 100 --m 16 --nbits 8

# HNSW 인덱스 테스트
python script.py --index_type hnsw --M 32 --efConstruction 200 --efSearch 128

# 성능 테스트 (IVFPQ 인덱스)
python script.py --index_type ivfpq --nlist 100 --m 8 --perf_test --max_samples 10000
```
