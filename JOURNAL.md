## KW 18 - 30/04/2026

### Decision: Proceeding as a Solo Developer for Topic H1+H2
Following the withdrawal of my group members, I have decided to proceed with the project as a solo developer for Topic H1+H2 (XAI-Guided Training for Knee Osteoarthritis Grading). I will try to maintain the original project track but will scale the scope of the project appropriately for a single developer. Since this choice is not easy to make, I will have to look into the topic and what is expected of me, and modify it accordingly. 

### Decision: Escalation of Blocked Dependency (Tutor GitHub Access)
The repository was successfully created, However, the list of tutor GitHub usernames was not visible on my Moodle page. Instead of delaying the repository setup, I initialized the directory structure, committed this first journal entry to establish temporal integrity, and sent an official escalation email to `lehre@iss.uni-saarland.de` requesting the usernames. Tutors will be granted Read access immediately upon receipt of their handles. Later I recieved the username and have added it.


## KW 19 - 07/05/2026

### Decision: Accepting a New Group Member
Following an email exchange with the course instructor (Mr. Deckarm), I decided to accept a late-registering student into our group, bringing the total team size to two members. While this increases our development bandwidth compared to working solo, our group remains significantly smaller than the standard 3-5 member. This late addition requires a slight pivot in our project management and onboarding, but it will allow us to parallelize coding tasks for the upcoming milestones.

### Decision: Narrowing the Scope of Hypothesis 2 (H2)
Because our newly formed two-person group is still operating under standard capacity and initiating collaboration late in the semester, we requested and received explicit approval from the instructor to narrow our project scope. For Hypothesis 2 (H2), the project brief suggests comparing multiple intervention strategies (e.g., augmentation, loss reweighting, attention regularisation). Instead, we decided to select, implement, and evaluate exactly one primary intervention strategy. Our methodology will focus on rigorously arguing why this single chosen strategy is the most appropriate for correcting the unfaithful predictions identified in Hypothesis 1 (H1). This ensures we maintain a realistic, manageable workload without sacrificing methodological quality.

## KW 20 - 12/05/2026

### Decision: Bypass HuggingFace dataset loading script — use raw zip download instead

The initial approach used `load_dataset("SilpaCS/kneeosteoarthritis")` from the HuggingFace `datasets` library. This call failed immediately with a `DatasetGenerationError` caused by the dataset's own loading script. The standard workaround flags such as `verification_mode` do not help because the error occurs during data generation, not during verification. We therefore switched to downloading the raw `data.zip` file directly via `hf_hub_download()` from the `huggingface_hub` library, which bypasses the broken script entirely. The zip was extracted locally to `data/kneeosteoarthritis/` and is excluded from version control via `.gitignore`. This approach is more robust and does not depend on the dataset maintainer fixing the upstream bug.

### Decision: Manual 80/20 train/validation split with fixed random seed

The dataset contains no pre-made train/test split — all 8,260 images are stored flat in five grade subfolders (`0`–`4`). We collected all samples, shuffled with `random.seed(42)`, and split 80% (6,608 images) for training and 20% (1,652 images) for validation. The fixed seed ensures the split is fully reproducible across machines and team members. We considered using stratified splitting (preserving class ratios per fold) but opted for the simple random split at this baseline stage; stratified splitting will be revisited if class imbalance causes instability in validation metrics.

### Decision: Weighted CrossEntropyLoss to address class imbalance

The class distribution is severely skewed: Grade 0 accounts for 3,253 images (39%) while Grade 4 contains only 251 images (3%). An unweighted loss function would bias the model toward majority classes and make Grade 4 predictions unreliable. We computed per-class weights `[0.51, 1.12, 0.76, 1.52, 6.71]` for grades 0–4 respectively, and passed these to `nn.CrossEntropyLoss(weight=class_weights)`. This is a standard and well-supported technique for imbalanced classification. Alternatives considered were oversampling (SMOTE, duplicate sampling) and undersampling, but weighted loss was preferred as it uses all available data without introducing synthetic samples or discarding real ones.

### Decision: ResNet-50 pretrained on ImageNet as baseline architecture

According to Karaelmas et al. (2024) `https://dergipark.org.tr/tr/download/article-file/3912636`, who benchmark multiple CNN architectures on knee osteoarthritis grading using the same KL grade classification task, ResNet-50 achieves accuracy comparable to higher-complexity models such as EfficientNet and DenseNet while being simpler to work with. We therefore selected ResNet-50 as our baseline: it performs nearly as well as the top alternatives and is directly compatible with gradient-based XAI methods via Captum's GuidedGradCAM, which targets `model.layer4[-1]` — the final convolutional block. This compatibility with GradCAM was the deciding factor over other architectures, as saliency analysis is central to both H1 and H2 of this project.

### Decision: Training configuration — Adam, StepLR, 20 epochs, batch size 64

The optimizer was set to Adam with learning rate 1e-4 and weight decay 1e-4. A StepLR scheduler reduces the learning rate by a factor of 0.5 every 5 epochs, preventing the model from overshooting minima in later training stages. We trained for 20 epochs with batch size 64 on a local NVIDIA GPU. Batch size was set to 64 rather than the default 32 to better utilise available VRAM and reduce training time, with no negative effect on convergence. The final training accuracy was 96.28% with a validation accuracy of 67.49%, indicating significant overfitting (~29 percentage point gap). This is expected at the baseline stage with a small imbalanced dataset and standard augmentation. The overfitting is methodologically useful: a model that has memorised training patterns is more likely to rely on spurious image features, which GuidedGradCAM analysis in H1 is designed to expose. Further epoch extensions were rejected as the bottleneck is overfitting, not underfitting — more epochs improve training accuracy only.

### Decision: Data augmentation — horizontal flip, rotation ±10°, ColorJitter deferred

Training augmentations include random horizontal flip and random rotation of ±10°. These are anatomically plausible for knee X-rays (bilateral symmetry, minor patient positioning variation). Stronger augmentations such as ColorJitter, RandomAffine translation, and RandomVerticalFlip were considered but deferred from the baseline run to keep the initial training controlled and comparable to the reference paper's setup. These will be introduced in the improved model (H2) to test whether reduced overfitting and higher validation accuracy accompany improved saliency faithfulness.


### AI Interaction — KW20 — 12.05.2026 — Data loading pipeline (01_data_load.py)

**Task:** Load the `SilpaCS/kneeosteoarthritis` dataset from HuggingFace and build a PyTorch DataLoader pipeline for training.

**Tool:** Perplexity AI (claude.ai / Sonnet 4.6)

**Prompt summary:** Asked how to start coding the project and load the dataset from HuggingFace, providing the full list of installed packages in `ds_env`.

**Output summary:** The AI produced an initial `01_data_load.py` using `load_dataset("SilpaCS/kneeosteoarthritis")` from the `datasets` library, applying HuggingFace transforms and constructing a `collate_fn`-based DataLoader. When the script failed with `DatasetGenerationError: Invalid string class label kneeosteoarthritis@def4c0831fd80df4e0711c3c6590210ba52e7345`, the AI diagnosed the root cause as a bug in the upstream dataset loading script and proposed a replacement approach using `hf_hub_download()` to retrieve the raw zip. A second issue arose because the zip extracts to `data/kneeosteoarthritis/data/0/` rather than having a `train/test` subfolder structure, causing the dataset to load 0 images. The AI identified this from the printed folder structure and provided a further revision that reads directly from the grade subfolders and applies a manual 80/20 split.

**Accepted / rejected / modified:** The final working version was accepted with one modification: `num_workers` was changed from `4` to `0` because Windows multiprocessing via `spawn` requires all DataLoader code to be inside an `if __name__ == '__main__':` guard, which is incompatible with a script run directly. `num_workers=0` is the standard Windows workaround and has negligible performance impact when the GPU is the bottleneck.

**Reasoning:** The AI correctly identified both failure modes and proposed working solutions. The `num_workers` Windows issue was not flagged by the AI in the initial GPU-optimised version and was discovered from the runtime error; the fix was trivial but the decision to apply it is documented here because it affects reproducibility on HPC (where `num_workers=4` will be re-enabled).

**Paper implication:** The methodology section will note that the dataset's HuggingFace loading script contains a bug and that data was loaded via direct zip download, with an 80/20 random split (seed 42) applied manually due to the absence of pre-defined splits.


### AI Interaction — KW20 — 12.05.2026 — Baseline model training (02_baseline_model.py)

**Task:** Implement and train a baseline ResNet-50 classifier for 5-class KL grade prediction with class-weighted loss.

**Tool:** Perplexity AI (claude.ai / Sonnet 4.6)

**Prompt summary:** Asked for the full `02_baseline_model.py` integrating the new data loading approach, with the model architecture, class weighting, training loop, scheduler, and validation evaluation, optimised for local NVIDIA GPU training.

**Output summary:** The AI produced a complete training script reusing the `KneeOADataset` class from `01_data_load.py`, building a ResNet-50 with a replaced fully-connected head, computing class weights from the training split counts, and training with Adam + StepLR over 20 epochs. It included `pin_memory=True` and `non_blocking=True` for faster CPU-to-GPU data transfer, and set `batch_size=64` for the GPU. When `num_workers=4` caused the same Windows multiprocessing error as in `01_data_load.py`, the AI confirmed the fix was simply `num_workers=0`.

**Accepted / rejected / modified:** Accepted with `num_workers=0`. The AI also initially suggested that further accuracy gains were possible with more epochs, but after reviewing the 26.5 percentage point train/val gap, the recommendation to keep the baseline as-is and move on to XAI analysis was accepted rather than continuing to tune.

**Reasoning:** The training results (train acc 90.7%, val acc 64.2%) are consistent with expected baseline behaviour on an imbalanced small medical dataset without strong regularisation. The gap is informative rather than problematic at this stage.

**Paper implication:** The methodology section will report the baseline ResNet-50 results as the starting point for H1 XAI analysis, noting the overfitting as motivation for the XAI-guided training intervention in H2.


## KW 24 - 11/06/2026

### Decision: Case Selection Procedure for Unfaithful Predictions — Documented by Remaining Member

The second group member had been assigned the Grad-CAM implementation and CbU case selection procedure since KW20 but did not deliver any working code. After repeated lack of progress, they eventually revealed they had dropped the course, leaving the entire XAI analysis pipeline incomplete. As a result, this work was started from scratch by me later. The CbU case selection procedure, which had been agreed upon conceptually earlier, was now formally implemented: a CbU case is defined as a validation image where the model predicts correctly but JSER < 0.40. This threshold sits just above the observed mean JSER, balancing sensitivity without excessive false positives.

### Decision: JSER Region Definition — Two Regions Compared, H-Band Selected

Two JSER region definitions were implemented and compared to determine which better captures anatomical joint-space attention: a Square Centre Crop (rows 56–168, cols 56–168) and an Anatomical H-Band (rows 105–160, full width). The H-Band produced lower variance (std 0.0365 vs 0.0937) and a cleaner per-grade CbU gradient, and was selected as the primary metric. Boundaries were visually validated at four candidate positions before being fixed. The square crop is retained as a secondary comparison.


### Decision: Baseline Faithfulness Characterisation Metrics Selected

The following metrics were selected to characterise baseline faithfulness: CbU Rate and Faithfulness Score quantify how often correct predictions are spatially unfaithful; per-class JSER breakdown reveals whether faithfulness degrades with disease severity; and the confidence-faithfulness Pearson correlation tests whether model confidence is a reliable proxy for correct attention. Together these cover global, class-level, and confidence-level dimensions of faithfulness, providing a complete baseline profile against which the XAI-guided model (H2) will be compared.


### AI Interaction — KW24 — 11.06.2026 — Grad-CAM pipeline and JSER implementation (03_gradcam_analysis.py)


**Task:** Implement Grad-CAM over the full validation set and compute JSER faithfulness metrics across both region definitions.

**Tool:** Perplexity AI (claude.ai / Sonnet 4.6)

**Prompt summary:** Worked through the Grad-CAM pipeline, region design, and unfaithful case detection with AI assistance for structuring parts of the code.

**Output summary:** A combined script running Grad-CAM in a single pass for both JSER variants, saving per-image results to JSON and generating worst-case visualisation figures.

**Accepted / rejected / modified:** Modified. The initial output had an indexing error in the JSER region slicing producing inflated energy ratios, and the visualisation figures initially lacked per-image JSER bar chart panels. Both were corrected. Region boundaries were also adjusted independently from 84–140 to 105–160 based on visual validation.

**Paper implication:** Methodology will describe both JSER variants, justify H-Band selection, and report the 105–160 boundary as visually validated.


### AI Interaction — KW24 — 11.06.2026 — Baseline faithfulness metrics (03b_baseline_characterisation.py)


**Task:** Formally compute and report baseline saliency faithfulness metrics.

**Tool:** Perplexity AI (claude.ai / Sonnet 4.6)

**Prompt summary:** Asked the AI to implement the metric formulas I had defined — CbU Rate, Faithfulness Score, per-class JSER breakdown, and confidence-faithfulness Pearson correlation — and produce a paper-ready 4-panel figure.

**Output summary:** The AI implemented the specified formulas and generated the figure from the existing JSON.

**Accepted / rejected / modified:** Modified. The initial figure had inconsistent colour coding across panels, which was corrected. A confidence quartile breakdown was also added after the first version was found insufficient for the analysis.

**Paper implication:** Results section will report the per-class faithfulness table and confidence-faithfulness scatter as evidence that baseline attention is anatomically unreliable, particularly for severe OA grades.