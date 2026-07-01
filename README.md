# XAI-Guided Training for Knee Osteoarthritis Grading

A Data Science mini-project that investigates whether Explainable AI (XAI) can be used not just to explain a model after training, but to actively improve its diagnostic reasoning during training. A ResNet-50 is trained to classify knee X-rays by Kellgren-Lawrence (KL) severity grade, with a custom attention penalty that forces the network to focus on the correct anatomical region — the knee joint space — rather than irrelevant image artifacts.



## Group Members

Mohamed Khalil Abbes



## Research Questions

- **H1:** Does a baseline CNN regularly attend to non-diagnostic regions even when predicting the correct KL grade?
- **H2:** Does adding an explicit attention penalty during training fix this without significantly hurting accuracy?

---

## Dataset

| Property | Details |
|---|---|
| Source | [SilpaCS/kneeosteoarthritis](https://huggingface.co/datasets/SilpaCS/kneeosteoarthritis) via Hugging Face |
| Size | 8,260 knee X-ray images |
| Labels | Kellgren-Lawrence grades 0–4 |
| Split | 80/20 train/val (seed 42) |

---

## Results

| Metric | Baseline | XAI-Guided | Change |
|---|---|---|---|
| Validation Accuracy | 67.5% | 66.0% | −1.5pp |
| Faithfulness Score | 88.4% | 100.0% | +11.6pp |
| CbU Rate | 11.6% | 0.0% | −11.6pp |
| Mean JSER | 0.444 | 0.584 | +0.140 |

**CbU (Correct-but-Unfaithful):** A correct prediction where the model was attending to the wrong region. The baseline produced a 68.9% CbU rate on the most severe grade (KL-4). The XAI-guided model eliminated all unfaithful predictions at a cost of only 1.5 percentage points in overall accuracy.

---

## Interactive Prototype

An interactive Streamlit app built on top of the trained model. Once trained, the model can be applied to any knee X-ray in a similar format to the training dataset (frontal AP view, knee region centered). Users upload an image and receive a KL-grade prediction, a live Grad-CAM heatmap, a JSER score, and a CbU warning if the model's attention falls outside the joint space. A batch scanner is also included to audit the full validation set for faithfulness. 

**To run:** `streamlit run .\code\app.py`

---

## References

- Kellgren & Lawrence (1957). *Radiological assessment of osteo-arthrosis.* Annals of the Rheumatic Diseases, 16(4), 494–502.
- Selvaraju et al. (2017). *Grad-CAM: Visual Explanations from Deep Networks via Gradient-based Localization.* ICCV.
- Ross et al. (2017). *Right for the Right Reasons.* IJCAI.
- Rieger et al. (2020). *Interpretations are useful: penalizing explanations to align neural networks with prior knowledge.* ICML.
- Solak (2024). *Classification of Knee Osteoarthritis Severity by Transfer Learning from X-Ray Images.* Karaelmas Science and Engineering Journal.
