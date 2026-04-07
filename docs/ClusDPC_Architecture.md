# ClusDPC: ClusPro + DPC + MSCI + VAPS + DHNO

## 1. 핵심 아이디어

CZSL에서 **비전 쪽**과 **텍스트 쪽** 모두를 개선하는 4중 적응 모델이다.

- **ClusPro** (ICLR 2025): 비전 쪽에서 속성/객체를 분리하고 다중 prototype으로 다양성 포착
- **DPC-Alpha** (본인 연구): 텍스트 쪽에서 dual prompt로 seen/unseen 균형 조절
- **MSCI** (본인 아이디어): ViT 중간 레이어에서 f_local 추출 → fine-grained 시각 정보
- **VAPS** (본인 아이디어): f_local → prompt_shifter → 이미지별 동적 프롬프트 shift
- **DHNO** (본인 아이디어): Dynamic branch에 hard negative contrastive loss 적용

ClusDPC는 이 다섯 가지를 하나의 프레임워크로 결합한다.

```
┌──────────────────────────────────────────────────────────────────┐
│                      ClusDPC Architecture                         │
│                                                                    │
│  ┌───────────────────────────┐  ┌────────────────────────────────┐│
│  │ Vision Side (ClusPro)     │  │ Text Side (DPC + MSCI + VAPS) ││
│  │                           │  │                                ││
│  │ • Visual Adapter (ViT)    │  │ • Primitive Branch (frozen)    ││
│  │ • Attr/Obj Disentangler   │  │ • Dynamic Branch:              ││
│  │ • K-Prototype Clustering  │  │   ctx_emb + shift(f_local)    ││
│  │ • Contrastive + HSIC Loss │  │ • Alpha Gating (entropy)       ││
│  │ • MSCI: f_local mid-layer │  │ • DHNO hard neg loss           ││
│  └───────────────────────────┘  └────────────────────────────────┘│
│                       │                    │                       │
│                       └────────┬───────────┘                       │
│                                ▼                                   │
│                     Score Fusion + Inference                       │
└──────────────────────────────────────────────────────────────────┘
```


## 2. ClusPro에서 가져온 것 (비전 쪽)

### 2.1 Visual Adapter
- CLIP ViT의 **매 블록**에 경량 LoRA-style adapter 삽입
- MHA 이후 + FFN 이후, 총 2 × num_blocks개
- CLIP 파라미터는 frozen, adapter만 학습
- 효과: ViT를 CZSL 태스크에 맞게 fine-tune

### 2.2 Attr/Obj Disentangler
- 글로벌 이미지 feature를 두 개의 MLP로 분리:
  - `attr_disentangler(f)` → 속성 정보 추출
  - `obj_disentangler(f)` → 객체 정보 추출
- 효과: "old car"에서 "old"와 "car"의 시각적 정보를 분리

### 2.3 Prototype Clustering
- 속성당 K=5개, 객체당 K=5개의 prototype 유지
- Momentum 업데이트: `proto = 0.99 * proto + 0.01 * new_centroid`
- 배치마다 similarity 기반으로 클러스터 할당 → centroid 갱신
- 효과: "old"의 다양한 외형 (늙은 사람, 낡은 차, ...)을 K개로 나눠 기억

### 2.4 ClusPro 손실 함수
- **Prototype Contrastive Loss**: disentangled feature ↔ prototype 매칭
- **HSIC Decorrelation**: attr feature에 obj 정보가 섞이지 않도록 독립성 보장


## 3. DPC에서 가져온 것 (텍스트 쪽)

### 3.1 Dual Soft Embeddings
- **Primitive** (`soft_att_obj_prim`): 범용적 의미 → unseen 조합에 강함
- **Contextual** (`soft_att_obj_ctx`): seen 데이터에 특화 → seen 조합에 강함
- 두 세트 모두 comp/attr/obj 3가지 프롬프트를 생성

### 3.2 Dynamic Alpha Gating
- Primitive branch의 **entropy**(불확실성)와 **max logit**(자신감)을 MLP에 입력
- 이미지마다 다른 α를 예측:
  - 불확실하면 → contextual 비중 ↑ (seen 쪽에 의지)
  - 확신 있으면 → primitive 비중 ↑ (범용 의미 신뢰)
- `final_comp = α * prim_logits + (1-α) * ctx_logits`


## 4. 결합 구조: Forward Flow

```
Image
  │
  ▼
CLIP ViT + Adapter (매 블록)
  │
  ├─→ f_global (CLS token)
  │      │
  │      ├─→ attr_disentangler → f_attr ──→ attr_proj ──┐
  │      │                                                │
  │      ├─→ obj_disentangler  → f_obj  ──→ obj_proj ──┐ │
  │      │                                              │ │
  │      │   ┌─ Prototype Update (momentum) ◄───────────┤ │
  │      │   │   attr_queue[k]: [K, D] per attr         │ │
  │      │   │   obj_queue[k]:  [K, D] per obj          │ │
  │      │   └─→ Contrastive Loss + HSIC Loss           │ │
  │      │                                              │ │
  │      └─→ Text Encoding (DPC Dual Branch)            │ │
  │           │                                         │ │
  │           ├─ Primitive: soft_att_obj_prim            │ │
  │           │   → comp_text_prim, attr_text_prim, ... │ │
  │           │                                         │ │
  │           ├─ Contextual: soft_att_obj_ctx            │ │
  │           │   → comp_text_ctx, attr_text_ctx, ...   │ │
  │           │                                         │ │
  │           └─ 3 branches × 2 prompts = 6 text feats  │ │
  │                                                     │ │
  │   Logit Computation:                                │ │
  │     comp_prim = f_global ⊗ comp_text_prim           │ │
  │     comp_ctx  = f_global ⊗ comp_text_ctx            │ │
  │     attr_logits = f_attr_proj ⊗ attr_text_ctx       │ │
  │     obj_logits  = f_obj_proj  ⊗ obj_text_ctx      ◄─┘ │
  │                                                       │
  │   Alpha Gating:                                       │
  │     α = sigmoid(MLP(entropy, conf, f_global))         │
  │     comp_logits = α * comp_prim + (1-α) * comp_ctx    │
  │                                                       │
  │   Final (Inference):                                  │
  │     score = comp_logits                               │
  │           + λ_attr * softmax(attr_logits)             │
  │           + λ_obj  * softmax(obj_logits)              │
  └───────────────────────────────────────────────────────┘
```


## 5. 손실 함수

```
L_total = L_comp_CE + L_attr_CE + L_obj_CE
        + λ_contrastive * L_prototype_contrastive
        + λ_hsic * L_decorrelation
```

| 손실 | 출처 | 역할 |
|------|------|------|
| L_comp_CE | 기본 | 조합 분류 (dual prompt 기반) |
| L_attr_CE | ClusPro | 속성 분류 (disentangled feature) |
| L_obj_CE | ClusPro | 객체 분류 (disentangled feature) |
| L_contrastive | ClusPro | prototype와 feature 정렬 |
| L_decorrelation | ClusPro | attr↔obj 독립성 보장 |


## 6. 왜 ClusPro 단독보다 나아야 하는가

| 상황 | ClusPro 단독 | ClusDPC |
|------|-------------|---------|
| Seen pair (학습에서 본 조합) | 단일 텍스트 프롬프트로 매칭 | **Contextual prompt**가 seen에 특화 |
| Unseen pair (못 본 조합) | 단일 텍스트 프롬프트 의존 | **Primitive prompt**가 범용적 매칭 + α gating 자동 조절 |
| 속성 다양성 | prototype K개로 커버 | prototype K개 + **dual prompt로 텍스트도 이중화** |
| seen/unseen 균형 | bias term으로 수동 조절 | **entropy 기반 α가 자동 조절** |


## 7. 학습 명령어

```bash
cd /home/dkkim/.gemini/antigravity/scratch/ClusDPC
python train.py --yml_path config/clus_dpc_mit.yml
```


## 8. 프로젝트 구조

```
ClusDPC/
├── config/
│   └── clus_dpc_mit.yml         # MIT-States, ViT-B/16 설정
├── clip_modules/                 # CLIP 관련 (Troika에서)
│   ├── clip_model.py
│   └── tokenization_clip.py
├── model/
│   ├── clus_dpc.py              # ★ 핵심 모델
│   ├── common.py                # CustomTextEncoder 등
│   ├── nce_loss.py              # Prototype contrastive loss
│   ├── hsic.py                  # HSIC decorrelation
│   ├── otgcc.py                 # Cluster assignment
│   └── model_factory.py
├── train.py                      # 학습 스크립트
├── test.py                       # 평가 스크립트
├── dataset.py                    # 데이터셋 로더
├── parameters.py                 # argparse 정의
└── utils.py                      # 유틸리티
```
