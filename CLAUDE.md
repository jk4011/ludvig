# LUDVIG 프로젝트 구조 분석

## 개요
**LUDVIG (Learning-Free Uplifting of 2D Visual Features to Gaussian Splatting Scenes)**는 2D vision foundation model의 features를 3D Gaussian Splatting scene으로 uplifting하는 learning-free 방법입니다.

## 핵심 개념

### 1. Uplifting (2D → 3D)
- **목표**: 2D 이미지의 feature map을 3D Gaussian으로 변환
- **방법**: Weighted aggregation
  - 각 Gaussian이 렌더링 시 기여하는 픽셀들의 feature를 가중 평균
  - 가중치는 splatting rendering weight `w_i(d,p)` 사용
  - 수식: `f_i = Σ [w_i(d,p) / Σw_i(d,p)] * F_{d,p}`
- **구현**: `utils/solver.py::uplifting()`
  - 각 view에서 2D feature를 추출하고 3D로 aggregation
  - Optional: 낮은 weight를 가진 Gaussian pruning

### 2. Graph Diffusion (Refinement)
- **목표**: Uplifted feature를 geometry와 feature similarity를 이용해 refine
- **방법**:
  - k-NN graph 구성 (3D 공간상 nearest neighbors)
  - Edge weight: feature similarity + regularization
  - Iterative propagation: `g^(t+1) = A * g^t`
- **구현**: `diffusion/base.py::GraphDiffusion`
  - RBF kernel로 feature similarity 계산
  - Sparse matrix를 사용한 효율적인 diffusion

## 프로젝트 구조

```
ludvig/
├── ludvig_base.py              # 기본 클래스 (Gaussian 로딩, 렌더링)
├── ludvig_uplift.py            # Uplifting 메인 클래스
├── ludvig_clip.py              # CLIP 기반 open-vocabulary localization
├── demo.py                     # 데모 실행 스크립트
│
├── gaussiansplatting/          # 3D Gaussian Splatting 구현
│   ├── scene/
│   │   ├── gaussian_model.py  # GaussianModel 클래스
│   │   ├── cameras.py         # Camera 클래스
│   │   └── camera_scene.py    # Scene 로딩
│   └── gaussian_renderer.py   # Rendering 함수
│
├── utils/
│   ├── solver.py              # **핵심**: uplifting() 함수
│   ├── graph.py               # k-NN graph 구성, similarity 계산
│   ├── visualization.py       # 시각화 유틸리티
│   └── image.py               # 이미지 처리
│
├── diffusion/
│   ├── base.py                # **핵심**: GraphDiffusion 클래스
│   ├── segmentation.py        # DINOv2 기반 segmentation diffusion
│   └── clip.py                # CLIP 기반 diffusion
│
├── predictors/                # 2D Feature Extractors
│   ├── base.py                # BaseDataset 클래스
│   ├── dino.py                # DINOv2 feature 추출
│   ├── clip.py                # CLIP feature 추출
│   ├── sam.py                 # SAM mask 추출
│   └── scannet.py             # ScanNet semantic segmentation
│
├── clip_utils/                # CLIP 관련 유틸리티
│   ├── openclip_encoder.py   # OpenCLIP 인코더
│   ├── lerf.py                # LERF 데이터셋 로딩
│   └── visualization.py       # CLIP 시각화
│
├── dinov2/                    # DINOv2 모델 및 평가
│   ├── model.py
│   ├── eval.py
│   └── segmentation_head.py
│
├── evaluation/                # 평가 스크립트
│   ├── spin_nvos/            # SPIn-NeRF NVOS 벤치마크
│   └── removal/              # Object removal 평가
│
└── configs/                   # 실험 설정 파일들
    ├── demo.yaml
    ├── lerf_clip.yaml
    └── ...
```

## 핵심 클래스

### 1. LUDVIGBase (`ludvig_base.py`)
**역할**: 기본 인프라 제공
- Gaussian Splatting scene 로딩 (`init_gaussians()`)
- RGB 렌더링 (`render_rgb()`)
- Feature 렌더링 (`render()`)
  - 3D feature (N, D)를 2D feature map (D, H, W)로 렌더링
  - 3개씩 나눠서 RGB 채널로 렌더링 후 결합
- 시각화 저장 (`save_images()`)

### 2. LUDVIGUplift (`ludvig_uplift.py`)
**역할**: 2D → 3D uplifting 수행
- `uplift()`:
  - Predictor로부터 2D feature 추출
  - `utils.solver.uplifting()` 호출하여 3D로 변환
  - Optional: L2 normalization
- `save()`:
  - Features를 `.npy`로 저장
  - Visualization 생성
  - Evaluation 실행

**사용 예시**:
```python
model = LUDVIGUplift(args)
features = model.uplift()  # 2D → 3D
model.save()                # 저장 및 평가
```

### 3. LUDVIGCLIP (`ludvig_clip.py`)
**역할**: Open-vocabulary object localization & segmentation
- CLIP features를 사용한 text query 기반 object localization
- LERF 벤치마크 평가
- Optional: SAM을 이용한 mask refinement
- Optional: Graph diffusion을 통한 segmentation 개선

**주요 메서드**:
- `compute_relevancies()`: Text query와 3D CLIP feature의 similarity 계산
- `evaluate_base()`: Uplifted CLIP feature로 평가
- `run_diffusion()`: Graph diffusion 실행
- `evaluate_diffusion()`: Diffusion 후 평가

## 핵심 알고리즘

### Uplifting Algorithm (`utils/solver.py`)
```python
def uplifting(loader, gaussian, ...):
    """
    1. 각 view에 대해:
       - 2D feature map 추출 (predictor)
       - Gaussian의 rendering weight 계산
       - weight * feature를 3D에 누적
    2. 누적된 값을 총 weight로 나누어 평균
    3. Optional: 낮은 weight의 Gaussian pruning
    """
    weights = zeros(num_gaussians)
    features_3d = zeros(num_gaussians, feature_dim)

    for feat, cam in loader:
        gaussian.apply_weights(cam, features_3d, weights, feat)

    features_3d /= weights + eps

    if prune_gaussians:
        keep = select_top_gaussians(weights)
        gaussian.prune_points(keep)
        features_3d = features_3d[keep]

    return features_3d
```

### Graph Diffusion (`diffusion/base.py`)
```python
class GraphDiffusion:
    def __call__(self, features):
        # 1. k-NN graph 구성
        knn_indices = query_neighbors(gaussian._xyz, k)

        # 2. Feature similarity 계산
        similarities = compute_rbf_similarity(features, knn_indices)

        # 3. Adjacency matrix 구성
        A = build_sparse_matrix(similarities, knn_indices)

        # 4. Iterative diffusion
        f = initial_features
        for t in range(num_iterations):
            f = A @ f
            f = normalize(f)

        return f
```

## Feature Predictors

### DINOv2 (`predictors/dino.py`)
- Self-supervised vision transformer features
- Dense feature extraction
- Semantic segmentation에 효과적

### CLIP (`predictors/clip.py`)
- Text-image alignment features
- Open-vocabulary localization
- Text query와의 similarity로 relevancy 계산

### SAM (`predictors/sam.py`)
- Segment Anything Model
- High-quality segmentation masks
- Point/box prompts 기반 segmentation

## 주요 실험 설정

### 1. DINOv2 Segmentation
```yaml
# configs/demo.yaml
feature:
  name: predictors.dino.DinoDataset
  model_type: dinov2_vitb14_reg
  layer: 11
  facet: token
```

### 2. CLIP Open-Vocabulary
```yaml
# configs/lerf_clip.yaml
feature:
  name: predictors.clip.CLIPDataset
  arch: ViT-B-16
  pretrained: laion2b_s34b_b88k
```

### 3. SAM Multi-view Segmentation
```yaml
# configs/sam_NVOS.yaml
feature:
  name: predictors.sam.SAMDataset
  ckpt: weights/sam_vit_h_4b8939.pth
  mode: masks
```

## 데이터 플로우

```
1. Input Images (Multi-view)
   ↓
2. 2D Feature Extraction (DINOv2/CLIP/SAM)
   → Feature maps: (C, H, W) per view
   ↓
3. Uplifting (Weighted Aggregation)
   → 3D Features: (N, C) per Gaussian
   ↓
4. (Optional) Graph Diffusion Refinement
   → Refined 3D Features
   ↓
5. Rendering to Novel Views
   → 2D Feature maps: (C, H, W)
   ↓
6. Downstream Tasks
   - Segmentation (with threshold)
   - Localization (argmax)
   - Open-vocabulary queries
```

## 중요 파일 위치

### 핵심 알고리즘
- **Uplifting**: `utils/solver.py:8-71`
- **Graph Diffusion**: `diffusion/base.py:82-183`
- **Rendering**: `ludvig_base.py:88-113`

### 주요 클래스
- **LUDVIGBase**: `ludvig_base.py:29-168`
- **LUDVIGUplift**: `ludvig_uplift.py:15-103`
- **LUDVIGCLIP**: `ludvig_clip.py:24-413`
- **GraphDiffusion**: `diffusion/base.py:9-210`

### Feature Extractors
- **DINOv2**: `predictors/dino.py`
- **CLIP**: `predictors/clip.py`
- **SAM**: `predictors/sam.py`

## 특징 및 장점

1. **Learning-free**:
   - Scene-specific training 불필요
   - 단순한 weighted aggregation만 사용
   - 빠르고 메모리 효율적

2. **Generic**:
   - 다양한 2D feature source 지원 (DINOv2, CLIP, SAM)
   - 여러 downstream task에 적용 가능

3. **Efficient**:
   - Single forward pass (no optimization loop)
   - Transpose rendering으로 해석 가능

4. **Optional Refinement**:
   - Graph diffusion으로 추가 개선 가능
   - Geometry와 feature similarity 활용

## 제한사항

1. 3D Gaussian Splatting representation이 미리 필요
2. Rendering quality에 의존
3. Gaussian 밀도가 feature resolution 결정
4. Multi-view consistency가 중요

## 평가 벤치마크

- **LERF**: Open-vocabulary localization & segmentation
- **SPIn-NeRF NVOS**: Novel-view object segmentation
- **ScanNet**: Semantic segmentation
- **Object Removal**: Inpainting-based removal

## 참고사항

- Python 3.11 환경
- CUDA 필요 (모든 연산이 GPU에서 수행)
- Gaussian Splatting submodule 포함
- 논문: Marrie et al., 2024
