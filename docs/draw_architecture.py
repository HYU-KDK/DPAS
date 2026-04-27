"""AdaptDPC v2 Architecture Diagram for PPT - Clean Version"""
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import matplotlib.patheffects as pe

fig, ax = plt.subplots(1, 1, figsize=(22, 16), dpi=200)
ax.set_xlim(0, 22)
ax.set_ylim(0, 16)
ax.axis('off')
fig.patch.set_facecolor('white')

# --- Colors ---
C_VIS_BG = '#DBEAFE';  C_VIS_BD = '#2563EB'
C_TXT_BG = '#FEF3C7';  C_TXT_BD = '#D97706'
C_LOG_BG = '#EDE9FE';  C_LOG_BD = '#7C3AED'
C_LOS_BG = '#FCE7F3';  C_LOS_BD = '#DB2777'
C_FROZEN = '#94A3B8';   C_LEARN  = '#3B82F6'
C_BRIDGE = '#EF4444';   C_DECORR = '#10B981'
C_WHITE  = '#FFFFFF'

def box(x, y, w, h, text, fc=C_WHITE, ec='#374151', fs=9, fw='normal',
        tc='#111827', lw=1.5, zorder=3, va_off=0):
    b = FancyBboxPatch((x, y), w, h, boxstyle='round,pad=0.12',
                       facecolor=fc, edgecolor=ec, linewidth=lw, zorder=zorder)
    ax.add_patch(b)
    ax.text(x + w/2, y + h/2 + va_off, text, ha='center', va='center',
            fontsize=fs, fontweight=fw, color=tc, zorder=zorder+1,
            linespacing=1.4)

def arr(x1, y1, x2, y2, c='#374151', lw=1.8, cs='arc3,rad=0', dash=False):
    s = dict(arrowstyle='->', color=c, linewidth=lw,
             connectionstyle=cs, mutation_scale=16, zorder=2)
    if dash:
        s['linestyle'] = 'dashed'
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), **s))

# ================================================================
# TITLE
# ================================================================
ax.text(11, 15.4, 'AdaptDPC v2 Architecture', ha='center', va='center',
        fontsize=24, fontweight='bold', color='#111827')
ax.text(11, 14.9, 'Role-Separated Vision-Text Framework for CZSL',
        ha='center', va='center', fontsize=13, color='#6B7280', style='italic')

# ================================================================
# VISION SIDE (left, x=0.5~8.5)
# ================================================================
ax.add_patch(FancyBboxPatch((0.5, 4.5), 8.0, 9.8, boxstyle='round,pad=0.25',
             facecolor=C_VIS_BG, edgecolor=C_VIS_BD, linewidth=2.5, alpha=0.45, zorder=1))
ax.text(4.5, 14.0, 'Vision Side', ha='center', va='center',
        fontsize=14, fontweight='bold', color=C_VIS_BD)
ax.text(4.5, 13.55, '(Feature Quality Only)', ha='center', va='center',
        fontsize=10, color=C_VIS_BD)

# Image
box(3.2, 12.4, 2.6, 0.7, 'Image', fc='#F1F5F9', ec='#475569', fs=12, fw='bold')

# CLIP ViT block
ax.add_patch(FancyBboxPatch((1.5, 9.6), 6.0, 2.2, boxstyle='round,pad=0.15',
             facecolor='#F8FAFC', edgecolor=C_FROZEN, linewidth=2, zorder=2))
ax.text(4.5, 11.4, 'CLIP ViT-B/16 (frozen)', ha='center', va='center',
        fontsize=11, fontweight='bold', color=C_FROZEN, zorder=3)
ax.text(4.5, 10.85, 'Block 1~12  +  LoRA Adapters (learnable)', ha='center', va='center',
        fontsize=9, color='#64748B', zorder=3)
ax.text(4.5, 10.3, 'Block 6 \u2192 f_local  |  Block 12 \u2192 f_global', ha='center', va='center',
        fontsize=9, color='#475569', fontweight='bold', zorder=3)

arr(4.5, 12.4, 4.5, 11.85)

# f_global
box(1.8, 8.2, 2.8, 0.8, 'f_global', fc='#DBEAFE', ec=C_VIS_BD, fs=12, fw='bold')
# f_local
box(5.5, 8.2, 2.6, 0.8, 'f_local\n(Block 6)', fc='#FEF9C4', ec=C_TXT_BD, fs=9, fw='bold')

arr(3.5, 9.6, 3.2, 9.05)
arr(5.8, 9.6, 6.8, 9.05)

# Disentanglers
box(0.8, 6.4, 3.0, 0.9, 'Attr Disentangler\n(Linear+BN+ReLU)',
    fc=C_WHITE, ec=C_LEARN, fs=8.5, fw='bold')
box(5.2, 6.4, 3.0, 0.9, 'Obj Disentangler\n(Linear+BN+ReLU)',
    fc=C_WHITE, ec=C_LEARN, fs=8.5, fw='bold')

arr(2.6, 8.2, 2.3, 7.35)
arr(3.8, 8.2, 6.7, 7.35)

# f_attr, f_obj
box(1.2, 5.1, 2.0, 0.7, 'f_attr', fc='#DBEAFE', ec=C_VIS_BD, fs=11, fw='bold')
box(5.8, 5.1, 2.0, 0.7, 'f_obj', fc='#DBEAFE', ec=C_VIS_BD, fs=11, fw='bold')

arr(2.3, 6.4, 2.2, 5.85)
arr(6.7, 6.4, 6.8, 5.85)

# Cosine Decorrelation
arr(3.2, 5.45, 5.8, 5.45, c=C_DECORR, lw=2.5, dash=True)
ax.text(4.5, 5.9, 'Cosine Decorrelation', ha='center', va='center',
        fontsize=8.5, fontweight='bold', color=C_DECORR,
        bbox=dict(boxstyle='round,pad=0.15', facecolor='#D1FAE5',
                  edgecolor=C_DECORR, alpha=0.9), zorder=4)
ax.text(4.5, 5.15, 'f_attr \u22A5 f_obj', ha='center', va='center',
        fontsize=8, color=C_DECORR, fontweight='bold', zorder=4)

# ================================================================
# TEXT SIDE (right, x=9.5~18.5)
# ================================================================
ax.add_patch(FancyBboxPatch((9.0, 4.5), 9.5, 9.8, boxstyle='round,pad=0.25',
             facecolor=C_TXT_BG, edgecolor=C_TXT_BD, linewidth=2.5, alpha=0.35, zorder=1))
ax.text(13.75, 14.0, 'Text Side (DPC)', ha='center', va='center',
        fontsize=14, fontweight='bold', color=C_TXT_BD)
ax.text(13.75, 13.55, '(Compositional Reasoning)', ha='center', va='center',
        fontsize=10, color=C_TXT_BD)

# Prompt
box(11.0, 12.4, 5.5, 0.7, '"a photo of [attr] [obj]"',
    fc='#FFFBEB', ec=C_TXT_BD, fs=11, fw='bold')

# Primitive
box(9.5, 10.2, 3.5, 1.5, 'Primitive Branch\n(frozen)\n\nGeneral semantics\nfor unseen pairs',
    fc='#F1F5F9', ec=C_FROZEN, fs=9, fw='bold')

# Contextual
box(14.5, 10.2, 3.5, 1.5, 'Contextual Branch\n(learnable + VAPS shift)\n\nSpecialized\nfor seen pairs',
    fc='#EFF6FF', ec=C_LEARN, fs=9, fw='bold')

arr(12.5, 12.4, 11.25, 11.75)
arr(15.0, 12.4, 16.25, 11.75)

# VAPS bridge
arr(8.1, 8.6, 14.5, 10.7, c=C_BRIDGE, lw=3.0, cs='arc3,rad=-0.25')
ax.text(11.2, 9.15, 'VAPS', ha='center', va='center',
        fontsize=12, fontweight='bold', color='white',
        bbox=dict(boxstyle='round,pad=0.3', facecolor=C_BRIDGE,
                  edgecolor='#B91C1C', alpha=0.95), zorder=5)
ax.text(11.2, 8.6, 'Vision \u2192 Text Bridge', ha='center', va='center',
        fontsize=8, color=C_BRIDGE, fontweight='bold', zorder=5)

# DHNO
box(17.0, 12.4, 1.8, 0.7, 'DHNO', fc='#FEE2E2', ec='#DC2626', fs=10, fw='bold')
arr(17.3, 12.4, 17.0, 11.75, c='#DC2626', lw=1.5)

# Alpha Predictor
box(10.0, 7.7, 7.5, 1.1, 'Alpha Predictor (MLP)\n\u03b1_comp          \u03b1_attr          \u03b1_obj',
    fc='#FFFBEB', ec=C_TXT_BD, fs=11, fw='bold')

arr(11.25, 10.2, 12.5, 8.85)
arr(16.25, 10.2, 15.0, 8.85)

# Per-Branch Blending
box(10.0, 5.5, 7.5, 1.3, 'Per-Branch Blending\n\n\u03b1 \u00b7 Primitive  +  (1 - \u03b1) \u00b7 Contextual',
    fc='#FFFBEB', ec=C_TXT_BD, fs=11, fw='bold')

arr(13.75, 7.7, 13.75, 6.85)

# ================================================================
# LOGIT COMPUTATION
# ================================================================
ax.add_patch(FancyBboxPatch((0.5, 2.0), 18.0, 2.0, boxstyle='round,pad=0.25',
             facecolor=C_LOG_BG, edgecolor=C_LOG_BD, linewidth=2.5, alpha=0.35, zorder=1))
ax.text(9.5, 3.75, 'Logit Computation', ha='center', va='center',
        fontsize=14, fontweight='bold', color=C_LOG_BD)

box(0.8, 2.3, 5.0, 1.1, 'attr_logit\nf_attr  \u00d7  blend(t_attr)',
    fc=C_WHITE, ec=C_LOG_BD, fs=10, fw='bold')
box(6.5, 2.3, 5.5, 1.1, 'comp_logit\nf_global  \u00d7  blend(t_comp)',
    fc=C_WHITE, ec=C_LOG_BD, fs=10, fw='bold')
box(12.7, 2.3, 5.0, 1.1, 'obj_logit\nf_obj  \u00d7  blend(t_obj)',
    fc=C_WHITE, ec=C_LOG_BD, fs=10, fw='bold')

# Vision -> Logits
arr(2.2, 5.1, 3.3, 3.45, c=C_VIS_BD, lw=1.5)
arr(3.2, 8.2, 9.25, 3.45, c=C_VIS_BD, lw=1.5, cs='arc3,rad=0.2')
arr(6.8, 5.1, 15.2, 3.45, c=C_VIS_BD, lw=1.5, cs='arc3,rad=0.2')

# Text -> Logits
arr(11.5, 5.5, 3.3, 3.45, c=C_TXT_BD, lw=1.5, cs='arc3,rad=0.15')
arr(13.75, 5.5, 9.25, 3.45, c=C_TXT_BD, lw=1.5)
arr(16.0, 5.5, 15.2, 3.45, c=C_TXT_BD, lw=1.5, cs='arc3,rad=-0.1')

# ================================================================
# LOSS
# ================================================================
ax.add_patch(FancyBboxPatch((0.5, 0.3), 18.0, 1.3, boxstyle='round,pad=0.2',
             facecolor=C_LOS_BG, edgecolor=C_LOS_BD, linewidth=2.5, alpha=0.4, zorder=1))
ax.text(9.5, 1.25, 'Loss', ha='center', va='center',
        fontsize=14, fontweight='bold', color=C_LOS_BD)
ax.text(9.5, 0.7,
        'L  =  CE(comp) + CE(attr) + CE(obj)  +  0.1 \u00d7 CosineDecorr  +  0.3 \u00d7 DHNO',
        ha='center', va='center', fontsize=12, fontweight='bold', color='#374151',
        bbox=dict(boxstyle='round,pad=0.25', facecolor='white',
                  edgecolor=C_LOS_BD, alpha=0.8), zorder=3)

arr(3.3, 2.3, 6.0, 1.65, c=C_LOG_BD, lw=1.2)
arr(9.25, 2.3, 9.25, 1.65, c=C_LOG_BD, lw=1.2)
arr(15.2, 2.3, 13.0, 1.65, c=C_LOG_BD, lw=1.2)

# ================================================================
# LEGEND
# ================================================================
lx, ly = 19.2, 13.5
ax.text(lx + 1.0, ly + 0.5, 'Legend', ha='center', va='center',
        fontsize=12, fontweight='bold', color='#374151')
for i, (c, label) in enumerate([
    (C_FROZEN, 'Frozen'), (C_LEARN, 'Learnable'),
    (C_BRIDGE, 'VAPS Bridge'), (C_DECORR, 'Decorrelation'),
    ('#DC2626', 'Hard Negative')
]):
    y = ly - i * 0.55
    ax.add_patch(FancyBboxPatch((lx, y - 0.13), 0.6, 0.28,
                 boxstyle='round,pad=0.05', facecolor=c, edgecolor=c, alpha=0.75))
    ax.text(lx + 0.85, y, label, ha='left', va='center', fontsize=9, color='#374151')

# ================================================================
# SAVE
# ================================================================
plt.tight_layout()
out = '/home/dkkim/.gemini/antigravity/scratch/ClusDPC/docs/AdaptDPC_v2_architecture.png'
plt.savefig(out, dpi=200, bbox_inches='tight', facecolor='white', edgecolor='none')
print(f"Saved: {out}")
plt.close()
