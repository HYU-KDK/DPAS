# ClusDPC 연구 로그

## 1. 연구 배경

### 출발점: DPC-Alpha (CSP 기반)
- CSP의 "a photo of [attr] [obj]" 구조에 Dual Prompt를 적용
- Primitive branch (frozen, 범용) + Contextual branch (learnable, 특화) + Alpha Gating
- Hard Negative Loss (DHNO) 추가
- **결과**: CSP 대비 성능 향상 확인 (MIT-States, ViT-L/14 기준 AUC 0.1928 → 0.1977)
- 이를 기반으로 더 강한 baseline(SOTA급)에 DPC를 얹으려고 시도

---

## 2. ClusPro + DPC 시도 (실패)

### 2.1 ClusPro Baseline 구현
CSP 대비 Vision쪽을 대폭 강화한 모델:
- Visual Adapter (LoRA-style, ViT 매 블록)
- Attr/Obj Disentangler (MLP로 f_global에서 분리)
- K-Prototype Clustering (K=5, momentum update)
- Prototype Contrastive Loss (NCE) + HSIC Decorrelation Loss

### 2.2 실험 결과 (MIT-States, ViT-B/16, 15 epochs)

| 모델 | Best AUC | Best HM | Seen | Unseen |
|------|----------|---------|------|--------|
| **ClusPro Baseline** | **0.1977** | **0.3659** | 0.4431 | 0.5253 |
| ClusPro + Dual (freeze) | 0.1926 | 0.3648 | 0.4382 | 0.5164 |
| ClusDPC (full: +MSCI+VAPS+DHNO) | 0.1787 | 0.3504 | 0.4273 | 0.499 |

### 2.3 실패 원인 분석

**핵심: DPC가 해결하려는 문제를 ClusPro가 이미 Vision쪽에서 해결하고 있었음**

1. **역할 중복**: ClusPro의 Visual Adapter + Disentangler가 이미 compositional context를 vision쪽에서 처리. DPC가 text쪽에서 같은 일을 하니 중복.

2. **Gradient 충돌**: ClusPro의 contrastive + HSIC loss와 DPC의 alpha gating + dual prompt가 같은 soft embedding을 통해 서로 다른 방향의 gradient를 보냄.

3. **Alpha predictor 신호 무의미화**: ClusPro의 Visual Adapter가 feature를 이미 잘 적응시켜놔서, primitive branch와 contextual branch의 logit 분포 차이가 작아지고, alpha predictor가 유의미한 gating을 학습하기 어려움.

4. **Primitive freeze의 역설**: Visual Adapter가 contextual branch의 CE loss에 맞춰 최적화되는데, primitive branch는 frozen인 채로 같은 adapted feature를 씀. 결과적으로 primitive branch의 logit quality 저하.

---

## 3. 방향 전환: 역할 분리 설계

### 3.1 핵심 아이디어
Vision쪽과 Text쪽의 역할을 명확히 분리:
- **Vision**: feature quality 개선만 (compositional reasoning X)
- **Text (DPC)**: compositional reasoning 전담
- **Bridge**: VAPS (f_local → prompt shift)가 vision→text 유일한 연결 통로

### 3.2 Vision-side context vs Text-side context 논의

| 측면 | Vision-side | Text-side |
|------|-------------|-----------|
| 강점 | 직접적 visual evidence 포착 | compositional 일반화 자연스러움 |
| 약점 | seen 조합에 overfitting 위험 | visual detail 부족 |
| 실험 근거 | ClusPro Baseline: Seen 높고 Unseen 상대적 낮음 | DPC-Alpha: Unseen 향상 |

결론: 둘 다 필요하되 하는 일이 달라야 함.

---

## 4. AdaptDPC v1 (Adapter + DPC, Disentangler 없음)

### 4.1 설계
- Visual Adapter만 유지 (feature quality)
- Disentangler, Prototype, Contrastive, HSIC 전부 제거
- f_global 하나로 comp/attr/obj logit 전부 계산
- DPC (dual prompt + alpha gating + VAPS + DHNO) 전담

### 4.2 피드백: Disentanglement 부재 문제
- f_global 안에 attr 정보와 obj 정보가 섞여 있어, attr logit 계산 시 obj가 noise로 작용
- CZSL 핵심인 "속성과 객체를 분리해서 일반화"가 vision쪽에 전혀 없음
- DHNO만으로는 attr-obj 독립성 제약 불가
- text쪽에서 아무리 잘해도 vision feature 자체가 entangled면 한계

### 4.3 학습 상태
- GPU 1에서 학습 시작했으나, v2 설계 후 epoch 1 도중 중단

---

## 5. AdaptDPC v2 (현재 실험 중)

### 5.1 v1 대비 변경사항

#### (1) Lightweight Disentanglement 추가
```
f_global → attr_disentangler → f_attr → attr logit 계산
f_global → obj_disentangler  → f_obj  → obj logit 계산
f_global (그대로)             →         → comp logit 계산
```
- ClusPro와 동일한 Disentangler 구조 (Linear + BN + ReLU)
- 단, Prototype/Contrastive는 빼고 Disentangler만 가볍게 유지

#### (2) HSIC → Cosine Decorrelation
```python
def cosine_decorrelation(f_attr, f_obj):
    f_a = normalize(f_attr)
    f_o = normalize(f_obj)
    cos_sim = (f_a * f_o).sum(dim=-1)  # [B]
    return cos_sim.abs().mean()
```
- RBF 커널 HSIC는 소배치(B=4)에서 4x4 커널 행렬로 추정 불안정
- Cosine decorrelation은 배치 크기에 무관하게 안정적
- f_attr ⊥ f_obj 독립성을 직접 강제

#### (3) Alpha Gating을 comp/attr/obj 모두에 적용
```python
# Before (v1): comp만 gating
comp = α·prim + (1-α)·ctx
attr = ctx_only
obj  = ctx_only

# After (v2): 세 branch 모두 gating
α_comp, α_attr, α_obj = alpha_predictor(input)  # MLP → [B, 3]
comp = α_c·prim + (1-α_c)·ctx
attr = α_a·prim + (1-α_a)·ctx
obj  = α_o·prim + (1-α_o)·ctx
```
- unseen 조합에서 primitive의 범용적 attr/obj 의미 활용 가능
- 각 branch별 gating_param bias 독립적 → 자동으로 최적 비율 학습

### 5.2 최종 아키텍처 요약

```
Vision쪽 (feature 제공만):
  - Visual Adapter: 도메인 적응
  - Disentangler: f_attr, f_obj 분리
  - Cosine Decorrelation: f_attr ⊥ f_obj

Text쪽 (compositional reasoning 전담):
  - Primitive branch: frozen general embeddings
  - Contextual branch: learnable + VAPS shift
  - Alpha Gating: comp/attr/obj별 독립 gating
  - DHNO: contextual branch hard negative

Bridge:
  - VAPS: f_local (mid-layer) → prompt shift → contextual prompt
  - f_global/f_attr/f_obj × text features → logits

Loss:
  CE(comp) + CE(attr) + CE(obj) + λ_decorr·CosineDecorr + λ_dhno·DHNO
```

### 5.3 ClusPro 대비 차이

| | ClusPro Baseline | AdaptDPC v2 |
|---|---|---|
| Visual Adapter | O | O |
| Disentangler | O | O |
| Prototype (K=5) | O | **X** |
| Contrastive (NCE) | O | **X** |
| HSIC (prototype 기반) | O | **Cosine Decorr (feature 직접)** |
| Dual Prompt | X | **O** |
| Alpha Gating | X | **O (comp/attr/obj)** |
| VAPS | X | **O** |
| DHNO | X | **O** |
| Vision gradient 압력 | CE + Contrastive + HSIC | **CE + CosineDecorr만** |

### 5.4 VAPS 속도 이슈
- per-image text encoding loop가 가장 큰 병목
- B=4일 때 매 step마다 text encoder를 4회 호출 (각 1262 시퀀스)
- 배치로 묶어 한번에 처리 시도했으나 OOM (B*1262 = 5048+)
- 청크 방식으로 전환 (N개씩 나눠서 처리)
- 근본적 해결은 어렵고, epoch당 ~1.5시간 소요

### 5.5 학습 설정
- Dataset: MIT-States
- Backbone: ViT-B/16
- Batch: 4, Gradient Accumulation: 16 (effective 64)
- Optimizer: Adam, lr=0.0001
- Scheduler: StepLR (step=5, gamma=0.5)
- Epochs: 15
- Loss weights: CE=1.0, CosineDecorr=0.1, DHNO=0.3

### 5.6 학습 상태
- 2026-04-06 GPU 1에서 학습 시작
- tmux 세션: `adapt_dpc_v2`
- 예상 소요: ~30시간
- 체크포인트: `checkpoint/adapt_dpc_v2_b16_mit/`

---

## 6. 향후 실험 계획

### 우선순위 높음
- [ ] v2 학습 결과 확인 (vs ClusPro Baseline)
- [ ] Unseen 성능이 특히 올랐는지 확인 (DPC의 일반화 효과 검증)

### Ablation 실험
- [ ] f_local layer 변경 (4 / 6 / 8) → VAPS 최적 layer 탐색
- [ ] f_local detach 여부 → adapter가 VAPS에 유용한 feature를 만들도록 유도
- [ ] VAPS 제거 버전 → Disentangler + Alpha Gating + DHNO만으로 충분한지

### 추가 개선 방향
- [ ] LLM 기반 초기 Prompt 생성 (미구현)
- [ ] Backbone 변경 (ViT-L/14) → 더 큰 모델에서 효과 확인
- [ ] UT-Zappos, C-GQA 데이터셋 확장 실험

---

## 7. 주요 교훈

1. **강한 baseline에 무작정 얹으면 안 된다**: CSP에서 효과 있던 DPC가 ClusPro에서는 역효과. 각 모듈의 역할이 겹치는지 먼저 분석해야.

2. **역할 분리가 핵심**: Vision과 Text가 같은 일을 하면 gradient 충돌. 각각 뭘 담당할지 명확히 분리해야.

3. **Vision-side disentanglement는 필요하다**: f_global 하나로 attr/obj를 구분하는 건 구조적 한계. 최소한의 분리 메커니즘은 있어야.

4. **HSIC는 배치 크기에 민감**: RBF 커널 HSIC는 소배치에서 불안정. Cosine decorrelation이 더 실용적.

5. **VAPS는 속도 vs 성능 trade-off**: per-image text encoding이 큰 병목. 효과가 있는지 먼저 VAPS 없이 확인하고 추가하는 게 효율적.
