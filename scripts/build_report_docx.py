"""Build the ForgeAD project report as an editable Word (.docx) file.

Same content as the PDF report (scripts/build_report_pdf.py), so the user
can edit it directly in Word / Google Docs.

Usage:  python scripts/build_report_docx.py
Output: results/ForgeAD-Report.docx
"""
from pathlib import Path

from docx import Document
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.oxml.ns import qn
from docx.oxml import OxmlElement
from docx.shared import Inches, Pt, RGBColor

ROOT = Path(__file__).resolve().parent.parent
RES = ROOT / "results"
OUT = RES / "ForgeAD-Report.docx"

doc = Document()
# ---- base styles: Times New Roman everywhere ----
style = doc.styles["Normal"]
style.font.name = "Times New Roman"
style.font.size = Pt(11)
style.paragraph_format.space_after = Pt(6)
style.paragraph_format.line_spacing = 1.35
for sec in doc.sections:
    sec.left_margin = Inches(0.87)
    sec.right_margin = Inches(0.87)
    sec.top_margin = Inches(0.79)
    sec.bottom_margin = Inches(0.79)


def shade(cell, hex_color):
    tcPr = cell._tc.get_or_add_tcPr()
    shd = OxmlElement("w:shd")
    shd.set(qn("w:val"), "clear")
    shd.set(qn("w:fill"), hex_color)
    tcPr.append(shd)


def para(text, bold=False, italic=False, size=11, align=WD_ALIGN_PARAGRAPH.JUSTIFY,
         after=6, before=0):
    p = doc.add_paragraph()
    p.alignment = align
    p.paragraph_format.space_after = Pt(after)
    p.paragraph_format.space_before = Pt(before)
    r = p.add_run(text)
    r.bold = bold
    r.italic = italic
    r.font.size = Pt(size)
    return p


def rich(parts, align=WD_ALIGN_PARAGRAPH.JUSTIFY, after=6, size=11, indent=None):
    """parts: list of (text, bold, italic)"""
    p = doc.add_paragraph()
    p.alignment = align
    p.paragraph_format.space_after = Pt(after)
    if indent is not None:
        p.paragraph_format.left_indent = Inches(indent)
    for text, b, i in parts:
        r = p.add_run(text)
        r.bold = b
        r.italic = i
        r.font.size = Pt(size)
    return p


def heading(text):
    h = doc.add_paragraph()
    h.paragraph_format.space_before = Pt(14)
    h.paragraph_format.space_after = Pt(6)
    r = h.add_run(text)
    r.bold = True
    r.font.size = Pt(15)
    r.font.color.rgb = RGBColor(0x1F, 0x29, 0x37)
    return h


def figure(path, caption, width=6.1):
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.add_run().add_picture(str(path), width=Inches(width))
    c = doc.add_paragraph()
    c.alignment = WD_ALIGN_PARAGRAPH.CENTER
    r = c.add_run(caption)
    r.italic = True
    r.font.size = Pt(9)
    r.font.color.rgb = RGBColor(0x4B, 0x55, 0x63)
    return p


def table(data, widths):
    t = doc.add_table(rows=len(data), cols=len(data[0]))
    t.style = "Table Grid"
    t.alignment = WD_TABLE_ALIGNMENT.CENTER
    for ri, row in enumerate(data):
        for ci, cell_text in enumerate(row):
            cell = t.rows[ri].cells[ci]
            cell.text = ""
            p = cell.paragraphs[0]
            p.alignment = WD_ALIGN_PARAGRAPH.LEFT if ci == 0 else WD_ALIGN_PARAGRAPH.CENTER
            r = p.add_run(cell_text)
            r.font.size = Pt(10)
            r.font.name = "Times New Roman"
            if ri == 0:
                r.bold = True
                r.font.color.rgb = RGBColor(0xFF, 0xFF, 0xFF)
                shade(cell, "1F2937")
            elif ri % 2 == 0:
                shade(cell, "F3F4F6")
            cell.width = Inches(widths[ci])
    return t


def caption(text):
    c = doc.add_paragraph()
    c.alignment = WD_ALIGN_PARAGRAPH.CENTER
    r = c.add_run(text)
    r.italic = True
    r.font.size = Pt(9)
    r.font.color.rgb = RGBColor(0x4B, 0x55, 0x63)


# ---------------- title ----------------
for _ in range(4):
    doc.add_paragraph()
t = doc.add_paragraph()
t.alignment = WD_ALIGN_PARAGRAPH.CENTER
r = t.add_run("ForgeAD: Self-Supervised Defect Detection\nby Forging Synthetic Anomalies")
r.bold = True
r.font.size = Pt(22)
para("Arshaad — October 2026", align=WD_ALIGN_PARAGRAPH.CENTER, size=12, after=2)
para("Project 3 of my portfolio for the China masters application. Builds directly on VisionAI (Project 2).",
     italic=True, align=WD_ALIGN_PARAGRAPH.CENTER, size=12)
para("An independent re-implementation of CutPaste (Li et al., CVPR 2021), evaluated head-to-head against my own PatchCore baseline on MVTec AD.",
     italic=True, align=WD_ALIGN_PARAGRAPH.CENTER, size=12)
doc.add_page_break()

# ---------------- abstract ----------------
heading("Abstract")
para("Automated visual inspection is one of the most practical uses of computer vision in manufacturing, and China — the world's factory floor — is where this technology matters most. Yet industrial defect detection suffers from a data problem rather than a model problem: defective samples are rare and expensive to collect, so a new production line typically starts with a modest set of normal images and almost no defects at all. Supervised classifiers cannot train in this regime, and purely unsupervised methods never learn what a defect actually looks like.")
para("ForgeAD tackles the problem with self-supervision. Instead of waiting for real defects, it forges its own: cut-paste patches and scar-like strips painted onto defect-free images. A ResNet-18 network learns to distinguish forged regions from real ones, and genuine defects are then detected as “things the forger never made” — scored by their distance from a Gaussian model of normality fitted over the learned embeddings. The method is an independent, from-scratch re-implementation of CutPaste (Li et al., CVPR 2021).")
rich([("Evaluated on the MVTec AD ", False, False), ("toothbrush", False, True),
      (" category under the exact protocol of my previous project's PatchCore baseline, ForgeAD reaches ", False, False),
      ("0.8417 image AUROC", True, False), (" and ", False, False), ("0.9141 pixel AUROC", True, False),
      (", against baseline scores of 0.9167 / 0.9267 — without ever seeing a single real defect label during training. A simpler ablation that scores images directly from the classifier head reaches 0.8583 image AUROC. The negative result on image-level detection is reported as measured and analyzed in detail, because an honest account of where a method falls short is more useful — and more credible — than a tuned one.", False, False)])

# ---------------- 1. problem ----------------
heading("1  The problem I wanted to solve")
para("In my last project (VisionAI, Project 2) I built a defect detector for factory products, and it worked well — 0.989 accuracy on steel defects, and a PatchCore anomaly detector at 0.9167 image AUROC on the MVTec toothbrush set. But one thing kept bothering me after I finished: every method I tried needed something factories do not have. Supervised classifiers need labeled defective samples, which are rare because defects are, by definition, the exception. Unsupervised methods like PatchCore need only normal images, but they never learn what a defect looks like — they only learn what normal looks like, and flag deviations.")
para("For a new production line this is a real bottleneck. You start with a handful of normal images and zero defects, and you cannot afford to wait months collecting failures before your inspection system works. So I asked myself a simple question: what if the model could manufacture its own training signal? What if it learned the visual vocabulary of damage — edges that should not be there, textures that break the pattern — by forging fake defects on normal images, without ever seeing a real one? That question became this project.")
para("This direction also matters to me personally: my masters application focuses on AI for smart manufacturing, and self-supervised learning is one of the most active research frontiers right now. Re-implementing a published method from scratch, rather than just calling a library, was the fastest way to genuinely understand it.")

# ---------------- 2. background ----------------
heading("2  Background and related work")
rich([("Two pieces of prior work shaped what I built. The first is ", False, False),
      ("PatchCore", True, False),
      (" (Roth et al., CVPR 2022), which was the baseline from my Project 2. PatchCore extracts mid-level patch embeddings from an ImageNet-pretrained WideResNet-50, compresses the patch embeddings of all normal training images into a memory bank via greedy coreset subsampling, and scores each test patch by its distance to the nearest bank entry. The image score is the maximum patch distance with a neighbourhood reweighting term. It is a strong, simple method — on MVTec toothbrush it gave me image AUROC ", False, False),
      ("0.9167", True, False), (" and pixel AUROC ", False, False), ("0.9267", True, False),
      (", and I kept the exact same data split and metric definitions here so the comparison is apples to apples.", False, False)])
rich([("The second is ", False, False), ("CutPaste", True, False),
      (" (Li et al., CVPR 2021), the idea I re-implemented from scratch. CutPaste creates synthetic anomalies by cutting a patch from an image and pasting it elsewhere (sometimes as a long thin “scar”), then trains a network on a 3-way classification task: normal vs. cut-paste vs. scar. After training, the embeddings of normal images are modeled as a Gaussian, and test images are scored by Mahalanobis distance from it. The paper reports roughly 0.95 AUROC averaged over all MVTec categories with a fully fine-tuned ResNet-18. My goal was to see how far an independent implementation gets on a single category, on CPU, written entirely by me — and to compare it fairly against my own PatchCore numbers instead of quoting the paper's.", False, False)])
figure(RES / "fig_pipeline.png", "Figure 1 — The ForgeAD pipeline: forge synthetic defects on normal images, train a 3-way classifier, score with a Gaussian density model, and evaluate against the PatchCore baseline.")

# ---------------- 3. method ----------------
heading("3  Method — what I built")
para("The pipeline has four parts, all implemented in src/ with no anomaly-detection libraries — only PyTorch, torchvision, and scikit-learn:")
rich([("3.1  The forge", True, False),
      (" (cutpaste.py). Two families of fake defects, applied ", False, False),
      ("only", False, True),
      (" to defect-free training images, so no real defect label ever enters the pipeline. Cut-paste: a rectangular patch covering 2–15% of the image area (aspect ratio 0.3–3.3) is cut from a random source location, randomly rotated in 90° steps, jittered in brightness (±15%) and color, and pasted at a random target location. Scar: a long thin strip (15–60% of the image side in length, 2–8 px thick) is cut, randomly rotated by any angle, and pasted back — it resembles a scratch or crack. Each forged image carries the label of the family that made it.", False, False)])
rich([("3.2  Training", True, False),
      (" (train.py). A ResNet-18 pretrained on ImageNet learns the 3-way task: normal / cut-paste / scar. Because I train on CPU, I froze the early layers (stem through layer2) and only trained layer3, layer4, and the classification head — the CutPaste paper fine-tunes the whole network, so this is a documented shortcut on my side, and I discuss its cost in Section 6. Each epoch, every training image is forged on the fly into a random class (augmentation factor 6). Model selection used forged copies of 6 held-out normal images — again, no real defect influenced training. Hyperparameters are listed in Table 2.", False, False)])
rich([("3.3  Scoring", True, False),
      (" (score_gde.py). After training, I fit a Gaussian with Ledoit-Wolf shrinkage over the 512-dimensional pooled embeddings of the 60 normal training images, and score each test image by its Mahalanobis distance from this normal model: far from normal means anomalous. For localization I fit one shared Gaussian over all patch positions of the 8×8 dense feature map, score every position, upsample the score map to image size with bilinear interpolation, and smooth it (σ = 4). I also tried a simpler ablation with no density fitting at all: score each image as 1 − P(normal) straight from the classifier head.", False, False)])
rich([("3.4  Evaluation", True, False),
      (" (evaluate.py). Image-level AUROC over the 42 test images and pixel-level AUROC over every test pixel against the ground-truth masks — the same definitions Project 2 used. Results are printed next to the PatchCore baseline automatically.", False, False)])
figure(RES / "fig_forge.png", "Figure 2 — The forge: a real normal toothbrush image (left) and its cut-paste and scar forgeries. Only normal images are forged; the network never sees a genuine defect during training.")

# ---------------- 4. experiments ----------------
heading("4  Experiments")
rich([("Dataset.", True, False),
      (" MVTec AD, toothbrush category (Bergmann et al., CVPR 2019; CC BY-NC-SA 4.0, research use). I chose toothbrush because it is the smallest category and the same one Project 2 used, which makes the comparison clean. Table 1 summarizes the split and Table 2 the training configuration.", False, False)])
table([["Split", "Images", "Notes"],
       ["train / good", "60", "defect-free; the only training data"],
       ["train forged (per epoch)", "324", "54 images × augmentation factor 6, on the fly"],
       ["validation", "6", "held-out normal images, forged copies only"],
       ["test / good", "12", "defect-free"],
       ["test / defective", "30", "with 30 ground-truth masks"]],
      [1.9, 1.0, 3.2])
caption("Table 1 — Data split. No real defective image is used for training or validation.")
table([["Hyperparameter", "Value"],
       ["Backbone", "ResNet-18, ImageNet-pretrained"],
       ["Frozen layers", "stem, layer1, layer2 (CPU shortcut)"],
       ["Input size", "224 × 224"],
       ["Optimizer", "Adam, lr 3×10⁻⁴, cosine schedule"],
       ["Batch size / epochs", "16 / 60"],
       ["Seed", "7 (deterministic)"],
       ["Hardware", "CPU only"]],
      [2.4, 3.7])
caption("Table 2 — Training configuration.")
rich([("Training dynamics.", True, False),
      (" The 3-way forge classification converged quickly: validation accuracy reached 1.0000 by epoch 7 and stayed there, with training accuracy ending at 1.000 (Figure 3). As I discuss in Section 6, the pretext task turning out this easy is itself part of the story — the network stopped being pushed early. One practical note: a machine restart once killed a 38-epoch run, so the training script now saves a checkpoint after every epoch and can resume with --resume.", False, False)])
figure(RES / "fig_training.png", "Figure 3 — Training curves: 3-way forge-classification accuracy (left) and training loss (right) over 60 epochs.")

# ---------------- 5. results ----------------
heading("5  Results")
para("All numbers below are measured on the toothbrush test split under the identical protocol as the PatchCore baseline. Nothing is projected or tuned after the fact:")
table([["Method", "Image AUROC", "Pixel AUROC"],
       ["PatchCore baseline (Project 2, measured)", "0.9167", "0.9267"],
       ["ForgeAD (Gaussian scorer)", "0.8417", "0.9141"],
       ["ForgeAD (head ablation: 1 − P(normal))", "0.8583", "—"]],
      [3.2, 1.45, 1.45])
caption("Table 3 — Measured results on MVTec AD toothbrush (42 test images).")
figure(RES / "fig_results.png", "Figure 4 — ForgeAD vs. the PatchCore baseline, side by side.", width=5.0)
figure(RES / "fig_roc.png", "Figure 5 — ROC curves on the toothbrush test set: image-level (GDE and head-ablation scorers) and pixel-level.")
para("I am reporting this exactly as measured: ForgeAD does not beat the baseline on image-level detection, and I would rather show an honest negative result than tune the numbers until they look better. That said, two findings are genuinely worth noting.")
rich([("First, ", False, False), ("pixel localization lands within 1.3 points of the baseline", True, False),
      (" (0.9141 vs 0.9267) — a respectable result for a model that never saw a real defect label. Figure 7 shows that the anomaly maps do concentrate on the true defective regions, even for the thin bristle damage that is hardest in this category.", False, False)])
rich([("Second, ", False, False), ("the simple head ablation beats the Gaussian scorer", True, False),
      (" (0.8583 vs 0.8417). This surprised me at first, but it makes sense: fitting a Gaussian in 512 dimensions on only 60 images is noisy even with shrinkage, while the classifier head was trained directly to separate normal from forged and turns out better calibrated. Figure 6 shows why neither scorer separates the classes perfectly — the score distributions of good and defective images overlap, which is exactly what the AUROC numbers say.", False, False)])
figure(RES / "fig_scores.png", "Figure 6 — Anomaly-score distributions on the 42 test images. The overlap between good and defective scores is what the AUROC values quantify.")
figure(RES / "forgead_sample_maps.png", "Figure 7 — Localization examples: input, ground-truth mask, and the ForgeAD anomaly map for three defective toothbrushes.", width=5.5)

# ---------------- 6. analysis ----------------
heading("6  Why the gap — an honest analysis")
para("The image-level gap to PatchCore (0.8417 vs 0.9167) has identifiable causes, and I think naming them precisely is more valuable than the gap is embarrassing:")
rich([("1. Too few normal images for the density fit.", True, False),
      (" Sixty samples in 512 dimensions is a thin basis for a Gaussian, even with Ledoit-Wolf shrinkage. The covariance estimate is noisy, and the Mahalanobis distances inherit that noise. The head ablation outperforming the GDE scorer is direct evidence: when the density fit is the bottleneck, skipping it helps.", False, False)], indent=0.15)
rich([("2. Frozen early layers.", True, False),
      (" My CPU shortcut kept stem through layer2 frozen, so the network could not adapt its low-level texture filters to toothbrushes. The CutPaste paper fine-tunes the whole ResNet-18, and for a category where defects are thin bristle anomalies — essentially texture-level deviations — those early filters likely matter more than they would for large, obvious defects.", False, False)], indent=0.15)
rich([("3. The forgeries are crude.", True, False),
      (" Hand-designed cut-paste patches and scars do not really look like the defects in this category. A pasted rectangle has sharp straight edges; real toothbrush damage is wispy and irregular. The network learned to detect “things with pasted edges” very well (hence the 1.000 forge accuracy) without learning the subtler vocabulary of real damage. There is a synthetic-to-real gap, and it shows up exactly where you would expect: image-level detection, where the whole-image judgment has to generalize.", False, False)], indent=0.15)
figure(RES / "fig_forge_vs_real.png", "Figure 8 — The synthetic-to-real gap: my cut-paste and scar forgeries next to a genuine defective toothbrush. The forgeries have sharp, regular edges; real damage is wispy and irregular.")
rich([("4. The pretext task was too easy.", True, False),
      (" Validation accuracy hit 1.0000 by epoch 7 and never moved again. Once the network could perfectly separate the three forge classes, training stopped pushing the representations to improve — the remaining 53 epochs mostly refined a decision boundary that was already solved. A harder pretext task (stronger augmentations, more forge families, or a contrastive objective) would keep the pressure on longer.", False, False)], indent=0.15)
figure(RES / "fig_failures.png", "Figure 9 — Where the model fails: the two good images with the highest anomaly scores (false positives) and the two defective images with the lowest scores (false negatives).", width=5.2)
para("Why does this project still belong in my portfolio? Because an independent, from-scratch re-implementation of a CVPR paper, evaluated under the exact protocol as my own baseline, with the negative result analyzed instead of buried, demonstrates three things supervisors actually look for: method literacy (I can read a paper and build it), experimental discipline (same data, same metrics, no moving goalposts), and honesty about limits. Those are rarer — and more convincing — than another “we beat SOTA” claim.")

# ---------------- 7. future ----------------
heading("7  What I would do next")
para("If I continue this line of work, the priorities are clear, roughly in order:")
rich([("1. Full fine-tuning on a GPU", True, False),
      (", following the paper exactly — unfreezing the early layers is the single change most likely to close the image-level gap on texture-like defects.", False, False)], indent=0.15)
rich([("2. A learned forge.", True, False),
      (" Replace the hand-made cut-paste/scar families with a small diffusion model conditioned on normal images. Realistic synthetic defects would shrink the synthetic-to-real gap that Section 6 identifies as the core weakness.", False, False)], indent=0.15)
rich([("3. More categories.", True, False),
      (" Test on MVTec textures (carpet, leather, wood) as well as objects. Textures behave very differently from objects, and a method should prove itself on both.", False, False)], indent=0.15)
rich([("4. Harder pretext tasks.", True, False),
      (" Stronger augmentations, more forge families, or a contrastive objective, so the representations keep improving instead of saturating at epoch 7.", False, False)], indent=0.15)
rich([("5. Few-shot extension.", True, False),
      (" If even 5–10 real defective images are available, use them to calibrate the scorer — a practical middle ground between the zero-defect regime of this project and fully supervised learning.", False, False)], indent=0.15)

# ---------------- references ----------------
heading("References")
for ref in [
    "[1] C. Li, S. Sohn, J. Yoon, T. Pfister, “CutPaste: Self-Supervised Learning for Anomaly Detection and Localization”, Proc. CVPR, 2021.",
    "[2] T. Roth, L. Pemula, J. Zepeda, B. Schölkopf, T. Brox, P. Gehler, “Towards Total Recall in Industrial Anomaly Detection”, Proc. CVPR, 2022.",
    "[3] P. Bergmann, M. Fauser, D. Sattlegger, C. Steger, “MVTec AD — A Comprehensive Real-World Dataset for Unsupervised Anomaly Detection”, Proc. CVPR, 2019.",
    "[4] K. Song, Y. Yan, “A noise robust method based on completed local binary patterns for hot-rolled steel strip surface defects”, Applied Surface Science, vol. 285, 2013 (dataset used in Project 2, Track A).",
]:
    rich([(ref, False, False)], size=10, after=4)

doc.save(str(OUT))
print("built", OUT, f"({OUT.stat().st_size / 1024:.0f} KB)")
