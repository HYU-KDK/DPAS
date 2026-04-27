# 향후 작업 계획 (Future Work)

## 0. 현재 상태 요약 (MIT-States, ViT-B/16 기준)

| Model | Test best_hm | Test AUC | Test Seen | Test Unseen |
|-------|:---:|:---:|:---:|:---:|
| **ClusPro Baseline** | **0.3362** | **0.1651** | 0.4261 | 0.4679 |
| ClusPro Dual | 0.3362 | 0.1661 | 0.4286 | 0.4684 |
| AdaptDPC v2 (no VAPS) | 0.3310 | 0.1647 | 0.4155 | 0.4793 |
| AdaptDPC v2 | 0.3092 | 0.1486 | 0.3971 | 0.4629 |
| ClusDPC (full) | 0.3014 | 0.1408 | 0.3924 | 0.4497 |

### 핵심 관찰
- **ClusPro Baseline을 넘긴 모델이 아직 없음** (Dual은 동률)
- DPC 텍스트 모듈을 더할수록 오히려 성능 하락하는 경향
- AdaptDPC v2에서 VAPS 제거가 성능 향상 (0.3092 → 0.3310) — VAPS가 현재 설정에서 해로움
- Unseen에서는 AdaptDPC v2 (no VAPS)가 최고 (0.4793) — text-side 일반화 가설은 여전히 유효

### 외부 비교 (참고용)
DPC Alpha (CSP 기반, **ViT-L/14**) Test 결과:
- best_hm 0.3510, AUC 0.1851
- ViT-L/14 backbone 효과로 ClusPro Baseline (ViT-B/16) 보다 우위
- 하지만 같은 ViT-B/16에서 비교해야 아키텍처 효과 분리 가능

---

## 1. 우선순위: 높음

### 1.1 ViT-Large 백본 실험
**목표**: ClusPro Baseline 및 AdaptDPC v2를 ViT-L/14로 학습하여 백본 효과 측정

- [ ] ClusPro Baseline (ViT-L/14) — 새 SOTA baseline 확보
- [ ] AdaptDPC v2 / no-VAPS (ViT-L/14) — 백본 키웠을 때 DPC 효과 검증
- [ ] DPC Alpha (ViT-B/16) — 역방향 비교, 백본 통제

**리스크**:
- VAPS의 per-image text encoding이 ViT-L에서 더 느려짐 (epoch당 3시간+ 예상)
- GPU 메모리: RTX 5060 Ti 16GB로 batch_size 4 가능한지 확인 필요
- 우선 no-VAPS 버전으로 시도 권장

**기대 효과**: 같은 backbone에서 아키텍처 기여를 정확히 측정

---

### 1.2 VAPS 재설계
**문제**: VAPS가 AdaptDPC v2에서 -0.022 HM 하락 (no-VAPS가 더 나음)

**가설별 검증 항목**:
- [ ] f_local layer 변경 (현재 6, 4/8 비교)
- [ ] f_local detach 여부 (현재 detach, gradient flow 허용 시 비교)
- [ ] Prompt shift 강도 조절 (residual scale parameter)
- [ ] Text-encoder batch size 늘려 속도 개선 (현재 32, OOM 한계까지)

**대안**: VAPS 대신 더 가벼운 visual conditioning (예: f_global을 prompt prefix로 추가)

---

### 1.3 ClusPro 내부 컴포넌트 ablation
**목표**: ClusPro Baseline의 어떤 컴포넌트가 가장 기여하는지 분리

- [ ] Visual Adapter only (Disentangler/Prototype/Contrastive 제거)
- [ ] Visual Adapter + Disentangler only (Prototype/Contrastive 제거)
- [ ] Prototype K 변경 (1, 3, 5, 10)
- [ ] Contrastive weight sweep (0.05, 0.1, 0.2)
- [ ] HSIC weight sweep

**산출물**: 어느 컴포넌트가 핵심인지 → DPC와 결합 시 어떤 걸 빼야 하는지 결정

---

## 2. 우선순위: 중간

### 2.1 Loss 가중치 튜닝
AdaptDPC v2의 학습 결과가 ClusPro Baseline보다 낮은 원인 중 하나가 loss balance:

- [ ] DHNO weight sweep (0.1, 0.2, 0.3, 0.5)
- [ ] CosineDecorr weight sweep (0.05, 0.1, 0.2, 0.5)
- [ ] CE(attr), CE(obj) weight 조절 (현재 모두 1.0)

---

### 2.2 데이터셋 확장
- [ ] UT-Zappos (단순한 데이터셋, 빠른 검증용)
- [ ] C-GQA (대규모, 일반화 검증)

ClusPro Baseline부터 돌려서 우리 환경에서의 baseline 수치 확보 필요.

---

### 2.3 Open-world 평가
현재 모든 평가가 Closed World. ClusDPC 모델에 `encode_text_for_open` / `forward_for_open` 메서드가 미구현.

- [ ] ClusDPC 계열 모델에 open-world 인터페이스 추가
- [ ] feasibility threshold 기반 open-world 평가 수행
- [ ] Closed vs Open 성능 차이 분석 (compositional generalization 강도 측정)

---

## 3. 우선순위: 낮음 (탐색적)

### 3.1 LLM 기반 Prompt 초기화
- [ ] GPT-4 등으로 attr/obj별 의미 description 생성
- [ ] Description embedding을 primitive soft embedding 초기값으로 사용
- [ ] CLIP의 frozen knowledge와 LLM의 풍부한 의미 결합 가능성

---

### 3.2 Alpha Gating 분석
- [ ] 학습된 alpha 값을 attr/obj별로 시각화
- [ ] Seen vs Unseen에서 alpha 분포 차이 확인
- [ ] α_comp/α_attr/α_obj가 실제로 다른 값을 학습하는지 검증

---

### 3.3 아키텍처 변형
- [ ] DHNO를 contextual branch뿐 아니라 primitive branch에도 적용
- [ ] Cross-attention 기반 vision-text fusion (현재는 cosine similarity만)

---

## 4. 환경/인프라 개선

- [ ] **eval 속도 개선**: `text_first=False` 모드가 1시간 30분+ 소요. ClusDPC 계열에 `encode_text_for_open` 구현하여 `text_first=True` 활성화 (학습 시 사용 인터페이스)
- [ ] **체크포인트 정리**: epoch별 .pt 파일이 누적, val_best.pt와 final_model.pt만 유지
- [ ] **로그 표준화**: train log에서 epoch별 val 결과만 별도 파일로 추출 (현재는 tqdm 출력 섞여 있음)

---

## 5. 정량 목표

ClusPro Baseline을 넘기기 위한 구체적 타깃 (Test):

| 지표 | 현재 (ClusPro) | 1차 목표 | 2차 목표 (SOTA급) |
|------|:---:|:---:|:---:|
| best_hm | 0.3362 | 0.35+ | 0.37+ |
| AUC | 0.1651 | 0.18+ | 0.20+ |
| best_unseen | 0.4679 | 0.49+ | 0.51+ |

---

## 6. 실험 우선순위 추천

가장 ROI 높은 순서:
1. **ClusPro 내부 ablation** (1.3) — 핵심 컴포넌트 식별, 1주
2. **VAPS 재설계** (1.2) — 현재 명확한 손실 요인, 1~2주
3. **ViT-Large** (1.1) — 단순 백본 키우기, GPU 시간 많이 소모
4. **Loss tuning** (2.1) — 한계 효율, 다른 문제 해결 후 진행
5. **데이터셋 확장** (2.2) — 일반화 검증, 위 결과 확정 후
