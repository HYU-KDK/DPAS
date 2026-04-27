"""ClusPro Baseline - Seen vs Unseen 개선폭 분석 그래프"""
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

plt.rcParams['font.family'] = ['DejaVu Sans']

# ── ClusPro Baseline (15 epochs + 1 test) ──
cluspro_epoch = list(range(1, 16))
cluspro_seen   = [0.372, 0.3861, 0.410, 0.4208, 0.4132, 0.4197, 0.4187,
                  0.4279, 0.4328, 0.4355, 0.4366, 0.442, 0.4463, 0.4447, 0.4431]
cluspro_unseen = [0.5142, 0.5156, 0.5143, 0.5185, 0.5185, 0.5196, 0.5216,
                  0.5252, 0.5204, 0.5223, 0.5231, 0.5264, 0.5257, 0.5227, 0.5253]
cluspro_auc    = [0.1633, 0.1686, 0.1781, 0.1833, 0.1795, 0.1836, 0.1839,
                  0.189, 0.1891, 0.1916, 0.192, 0.196, 0.1977, 0.1957, 0.197]
cluspro_hm     = [0.3319, 0.3364, 0.3451, 0.3495, 0.3475, 0.3495, 0.3509,
                  0.3549, 0.3579, 0.3602, 0.3587, 0.3658, 0.3651, 0.3655, 0.3659]

# ── AdaptDPC v2 (11 epochs so far) ──
v2_epoch = list(range(1, 12))
v2_seen   = [0.32, 0.3243, 0.3498, 0.3492, 0.3742, 0.3726, 0.3856,
             0.3861, 0.3877, 0.4018, 0.4089]
v2_unseen = [0.4862, 0.492, 0.5008, 0.5136, 0.5108, 0.5181, 0.5161,
             0.5118, 0.512, 0.5126, 0.5125]
v2_auc    = [0.13, 0.1357, 0.1486, 0.1533, 0.1612, 0.1643, 0.1688,
             0.1686, 0.1704, 0.176, 0.1784]
v2_hm     = [0.2943, 0.3017, 0.3174, 0.3229, 0.3287, 0.3349, 0.3374,
             0.3401, 0.3438, 0.3506, 0.3531]

fig, axes = plt.subplots(2, 2, figsize=(18, 14), dpi=200)
fig.suptitle('ClusPro Baseline vs AdaptDPC v2  —  Training Curve Comparison\n(MIT-States, ViT-B/16)',
             fontsize=18, fontweight='bold', y=0.98)

# ═══════════════════════════════════════════════
# (1) Seen Accuracy
# ═══════════════════════════════════════════════
ax = axes[0, 0]
ax.plot(cluspro_epoch, cluspro_seen, 'o-', color='#2563EB', linewidth=2.5,
        markersize=7, label='ClusPro Baseline', zorder=3)
ax.plot(v2_epoch, v2_seen, 's--', color='#D97706', linewidth=2.5,
        markersize=7, label='AdaptDPC v2', zorder=3)

# 개선폭 annotation
ax.annotate(f'+{cluspro_seen[-1] - cluspro_seen[0]:.3f}\n(+{(cluspro_seen[-1] - cluspro_seen[0])*100:.1f}%p)',
            xy=(15, cluspro_seen[-1]), xytext=(13, 0.46),
            fontsize=10, fontweight='bold', color='#2563EB',
            arrowprops=dict(arrowstyle='->', color='#2563EB', lw=1.5),
            bbox=dict(boxstyle='round,pad=0.3', facecolor='#DBEAFE', edgecolor='#2563EB'))

ax.set_title('Seen Accuracy', fontsize=14, fontweight='bold')
ax.set_xlabel('Epoch', fontsize=12)
ax.set_ylabel('Accuracy', fontsize=12)
ax.set_xlim(0.5, 15.5)
ax.set_ylim(0.30, 0.48)
ax.legend(fontsize=11, loc='lower right')
ax.grid(True, alpha=0.3)
ax.yaxis.set_major_formatter(mticker.FormatStrFormatter('%.2f'))

# ═══════════════════════════════════════════════
# (2) Unseen Accuracy
# ═══════════════════════════════════════════════
ax = axes[0, 1]
ax.plot(cluspro_epoch, cluspro_unseen, 'o-', color='#2563EB', linewidth=2.5,
        markersize=7, label='ClusPro Baseline', zorder=3)
ax.plot(v2_epoch, v2_unseen, 's--', color='#D97706', linewidth=2.5,
        markersize=7, label='AdaptDPC v2', zorder=3)

# 개선폭 annotation
ax.annotate(f'+{cluspro_unseen[-1] - cluspro_unseen[0]:.3f}\n(+{(cluspro_unseen[-1] - cluspro_unseen[0])*100:.1f}%p)',
            xy=(15, cluspro_unseen[-1]), xytext=(13, 0.535),
            fontsize=10, fontweight='bold', color='#2563EB',
            arrowprops=dict(arrowstyle='->', color='#2563EB', lw=1.5),
            bbox=dict(boxstyle='round,pad=0.3', facecolor='#DBEAFE', edgecolor='#2563EB'))

# plateau 영역 강조
ax.axhspan(0.51, 0.53, alpha=0.15, color='#EF4444', zorder=1)
ax.text(8, 0.509, 'Unseen plateau zone', ha='center', fontsize=10,
        color='#DC2626', fontweight='bold', style='italic')

ax.set_title('Unseen Accuracy', fontsize=14, fontweight='bold')
ax.set_xlabel('Epoch', fontsize=12)
ax.set_ylabel('Accuracy', fontsize=12)
ax.set_xlim(0.5, 15.5)
ax.set_ylim(0.48, 0.54)
ax.legend(fontsize=11, loc='lower right')
ax.grid(True, alpha=0.3)
ax.yaxis.set_major_formatter(mticker.FormatStrFormatter('%.3f'))

# ═══════════════════════════════════════════════
# (3) Seen vs Unseen 개선폭 비교 (Bar Chart) - ClusPro only
# ═══════════════════════════════════════════════
ax = axes[1, 0]

# 3구간으로 나눠서 개선폭 비교
periods = ['Ep 1→5', 'Ep 5→10', 'Ep 10→15']
seen_deltas = [
    cluspro_seen[4] - cluspro_seen[0],    # ep1→5
    cluspro_seen[9] - cluspro_seen[4],    # ep5→10
    cluspro_seen[14] - cluspro_seen[9],   # ep10→15
]
unseen_deltas = [
    cluspro_unseen[4] - cluspro_unseen[0],
    cluspro_unseen[9] - cluspro_unseen[4],
    cluspro_unseen[14] - cluspro_unseen[9],
]

x = np.arange(len(periods))
w = 0.35
bars1 = ax.bar(x - w/2, [d*100 for d in seen_deltas], w, label='Seen',
               color='#3B82F6', edgecolor='#1D4ED8', linewidth=1.5, zorder=3)
bars2 = ax.bar(x + w/2, [d*100 for d in unseen_deltas], w, label='Unseen',
               color='#F59E0B', edgecolor='#D97706', linewidth=1.5, zorder=3)

# 값 표시
for bar, val in zip(bars1, seen_deltas):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
            f'+{val*100:.1f}%p', ha='center', fontsize=10, fontweight='bold', color='#1D4ED8')
for bar, val in zip(bars2, unseen_deltas):
    h = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2, max(h, 0) + 0.1,
            f'+{val*100:.1f}%p', ha='center', fontsize=10, fontweight='bold', color='#D97706')

ax.set_title('ClusPro: Seen vs Unseen Improvement per Period', fontsize=14, fontweight='bold')
ax.set_xlabel('Training Period', fontsize=12)
ax.set_ylabel('Accuracy Change (%p)', fontsize=12)
ax.set_xticks(x)
ax.set_xticklabels(periods, fontsize=11)
ax.legend(fontsize=11)
ax.grid(True, axis='y', alpha=0.3)
ax.axhline(y=0, color='black', linewidth=0.8)
ax.set_ylim(-1, 5)

# 핵심 메시지
ax.text(2, 3.8, 'Seen keeps improving\nUnseen stagnates',
        ha='center', fontsize=11, fontweight='bold', color='#DC2626',
        bbox=dict(boxstyle='round,pad=0.4', facecolor='#FEE2E2', edgecolor='#DC2626'))

# ═══════════════════════════════════════════════
# (4) AUC + HM 비교
# ═══════════════════════════════════════════════
ax = axes[1, 1]
ax.plot(cluspro_epoch, cluspro_auc, 'o-', color='#2563EB', linewidth=2.5,
        markersize=7, label='ClusPro AUC', zorder=3)
ax.plot(v2_epoch, v2_auc, 's--', color='#D97706', linewidth=2.5,
        markersize=7, label='AdaptDPC v2 AUC', zorder=3)
ax.plot(cluspro_epoch, cluspro_hm, '^-', color='#7C3AED', linewidth=2,
        markersize=6, label='ClusPro HM', alpha=0.7, zorder=2)
ax.plot(v2_epoch, v2_hm, 'D--', color='#DB2777', linewidth=2,
        markersize=6, label='AdaptDPC v2 HM', alpha=0.7, zorder=2)

ax.set_title('AUC & HM Comparison', fontsize=14, fontweight='bold')
ax.set_xlabel('Epoch', fontsize=12)
ax.set_ylabel('Score', fontsize=12)
ax.set_xlim(0.5, 15.5)
ax.set_ylim(0.12, 0.40)
ax.legend(fontsize=10, loc='lower right')
ax.grid(True, alpha=0.3)
ax.yaxis.set_major_formatter(mticker.FormatStrFormatter('%.2f'))

plt.tight_layout(rect=[0, 0, 1, 0.95])
out = '/home/dkkim/.gemini/antigravity/scratch/ClusDPC/docs/seen_unseen_analysis.png'
plt.savefig(out, dpi=200, bbox_inches='tight', facecolor='white')
print(f'Saved: {out}')
plt.close()

# ═══════════════════════════════════════════════
# 두 번째 그래프: Epoch별 전체 수치 테이블
# ═══════════════════════════════════════════════
fig2, ax2 = plt.subplots(figsize=(18, 10), dpi=200)
ax2.axis('off')

title = 'ClusPro Baseline: Full Training Results (MIT-States, ViT-B/16)'
ax2.set_title(title, fontsize=16, fontweight='bold', pad=20)

col_labels = ['Epoch', 'Seen', 'Unseen', 'HM', 'AUC',
              'Seen\n(+/-)', 'Unseen\n(+/-)']

row_data = []
for i in range(15):
    s_delta = cluspro_seen[i] - cluspro_seen[max(0, i-1)] if i > 0 else 0
    u_delta = cluspro_unseen[i] - cluspro_unseen[max(0, i-1)] if i > 0 else 0
    row_data.append([
        f'{i+1}',
        f'{cluspro_seen[i]*100:.1f}%',
        f'{cluspro_unseen[i]*100:.1f}%',
        f'{cluspro_hm[i]*100:.1f}%',
        f'{cluspro_auc[i]*100:.2f}%',
        f'+{s_delta*100:.1f}' if s_delta >= 0 else f'{s_delta*100:.1f}',
        f'+{u_delta*100:.1f}' if u_delta >= 0 else f'{u_delta*100:.1f}',
    ])

# Summary row
total_s = cluspro_seen[14] - cluspro_seen[0]
total_u = cluspro_unseen[14] - cluspro_unseen[0]
row_data.append([
    'Total',
    f'{cluspro_seen[0]*100:.1f}% → {cluspro_seen[14]*100:.1f}%',
    f'{cluspro_unseen[0]*100:.1f}% → {cluspro_unseen[14]*100:.1f}%',
    f'{cluspro_hm[0]*100:.1f}% → {cluspro_hm[14]*100:.1f}%',
    f'{cluspro_auc[0]*100:.2f}% → {cluspro_auc[14]*100:.2f}%',
    f'+{total_s*100:.1f}%p',
    f'+{total_u*100:.1f}%p',
])

table = ax2.table(cellText=row_data, colLabels=col_labels,
                  loc='center', cellLoc='center')
table.auto_set_font_size(False)
table.set_fontsize(10)
table.scale(1.0, 1.65)

# Header styling
for j in range(len(col_labels)):
    cell = table[0, j]
    cell.set_facecolor('#1E3A5F')
    cell.set_text_props(color='white', fontweight='bold', fontsize=10)

# Row styling
for i in range(len(row_data)):
    is_total = (i == len(row_data) - 1)
    is_best_auc = (i == 12)  # epoch 13 = best AUC
    bg = '#FFF3E0' if is_total else ('#E8F5E9' if is_best_auc else
         ('#F0F7FF' if i % 2 == 0 else '#FFFFFF'))
    for j in range(len(col_labels)):
        cell = table[i + 1, j]
        cell.set_facecolor(bg)
        if is_total:
            cell.set_text_props(fontweight='bold')
        if is_best_auc and j == 0:
            cell.set_text_props(fontweight='bold', color='#2E7D32')

    # Color unseen delta: red if small/negative
    if not is_total:
        u_val = cluspro_unseen[i] - cluspro_unseen[max(0, i-1)] if i > 0 else 0
        s_val = cluspro_seen[i] - cluspro_seen[max(0, i-1)] if i > 0 else 0
        if i > 0:
            # Seen delta color
            c_s = '#2E7D32' if s_val > 0.005 else ('#DC2626' if s_val < 0 else '#6B7280')
            table[i + 1, 5].set_text_props(color=c_s, fontweight='bold')
            # Unseen delta color
            c_u = '#2E7D32' if u_val > 0.005 else ('#DC2626' if u_val < 0 else '#6B7280')
            table[i + 1, 6].set_text_props(color=c_u, fontweight='bold')
    else:
        table[i + 1, 5].set_text_props(color='#2E7D32', fontweight='bold')
        table[i + 1, 6].set_text_props(color='#DC2626', fontweight='bold')

# Best epoch marker
ax2.text(0.08, 0.22, '* Epoch 13 = Best AUC', fontsize=10,
         color='#2E7D32', fontweight='bold', transform=ax2.transAxes)

# Bottom message
fig2.text(0.5, 0.06,
          'Seen: 37.2% → 44.3%  (+7.1%p)    vs    Unseen: 51.4% → 52.5%  (+1.1%p)\n'
          'Seen improves 6.4x faster than Unseen → Vision-side overfitting to seen compositions',
          ha='center', fontsize=13, fontweight='bold', color='#DC2626',
          bbox=dict(boxstyle='round,pad=0.5', facecolor='#FEE2E2', edgecolor='#DC2626'))

plt.tight_layout(rect=[0, 0.12, 1, 0.95])
out2 = '/home/dkkim/.gemini/antigravity/scratch/ClusDPC/docs/seen_unseen_table.png'
plt.savefig(out2, dpi=200, bbox_inches='tight', facecolor='white')
print(f'Saved: {out2}')
plt.close()
