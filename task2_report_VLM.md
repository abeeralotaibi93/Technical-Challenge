# Task 2: Medical Report Generation

**Using Visual Language Model (MedGemma)**

Abeer | February 2026

---

## 1. Model Selection Justification

For this task, I used MedGemma (`google/medgemma-4b-it`), Google's open-source visual language model specifically fine-tuned for medical imaging. The choice was straightforward: MedGemma is trained on medical data, so it already understands clinical terminology and radiology conventions without any extra fine-tuning. Compared to general-purpose VLMs like LLaVA, MedGemma produces more structured and clinically relevant output when given a chest X-ray.

It was loaded directly from Hugging Face using the `AutoModelForImageTextToText` interface, and images were preprocessed to RGB 224×224 as required by the model's vision encoder. The model ran on CPU, which was slower but functional for this evaluation.

## 2. Prompting Strategies Tested and Their Effectiveness

Three prompting strategies were tested, each designed to extract different levels of clinical detail from the model. The goal was to find the right balance between brevity and clinical usefulness.

**Strategy 1 — Diagnostic (Yes/No)**

This was the most minimal prompt: a direct binary question asking if pneumonia is present.

```python
prompt_text = (
    "You are an expert radiologist. "
    "Is pneumonia present in this chest X-ray? "
    "Reply with exactly one word: Yes or No."
)
```

<img width="612" height="539" alt="image" src="https://github.com/user-attachments/assets/1266e7f2-1ea8-4dc8-8865-ef864d6e7518" />


The model returned a single-word answer, which is easy to compare against ground truth labels. This strategy is most comparable to the CNN from Task 1 — both output a binary prediction. However, it provides no reasoning or evidence, making it hard to understand why the model decided one way or the other. Compared to the CNN, the CNN achieved 88% accuracy on the test set. This prompt essentially turns MedGemma into a classifier, but unlike the CNN it has no confidence score and cannot be evaluated quantitatively across the full dataset without significant compute. For the image tested, the model gave a wrong answer. This could happen because MedGemma is designed as a generative report model, not a classifier — it produces text based on learned patterns, so a single-word answer might not fully capture uncertainty or subtle features in the X-ray. Unlike the CNN, which is trained explicitly on labeled examples to minimize classification error, MedGemma may miss fine-grained visual cues or overfit to its language priors.

**Strategy 2 — Descriptive Radiologist Report**

This prompt asks for a structured report as a radiologist would write it, without constraining the format.

```python
prompt_text = (
    "Analyze this chest X-ray as a radiologist would and produce a structured report, "
    "highlighting any signs of pneumonia"
)
```

<img width="1617" height="800" alt="image" src="https://github.com/user-attachments/assets/1ae76262-773e-46fd-b9a2-71eb3231875a" />


The model produced a structured output with Image Analysis and Findings sections. It described lung fields as 'clear bilaterally, with no obvious consolidation, infiltrates, or masses' — which contradicts the ground truth label of Pneumonia. This is a false negative from the VLM's perspective: the image is a true pneumonia case, but the model reported it as normal.
It's worth noting that images were preprocessed and upsampled to 224×224 RGB before being passed to MedGemma, so the model did receive a reasonably sized input. However, upsampling a 28×28 image to 224×224 does not recover any lost detail — it simply scales up the same limited pixel information. The underlying visual content is still constrained by the original low-resolution capture, which means subtle opacities and fine infiltrate patterns remain indistinguishable even at the larger size. The CNN faces the same underlying issue.


**Strategy 3 — RSNA-Guided Structured Report**

This was the most clinically structured prompt, asking the model to follow RSNA pneumonia detection guidelines explicitly.

```python
prompt_text = (
    "You are an expert radiologist. "
    "Generate a chest X-ray report following RSNA pneumonia guidelines. "
    "Include only findings consistent with the standard criteria "
    "(consolidation, effusion, pneumothorax, lung opacity, heart size, "
    "mediastinum, pleura, bones) and provide a structured summary "
    "with Findings and Impression."
)
```

<img width="1631" height="761" alt="image" src="https://github.com/user-attachments/assets/e9e8a0b2-a8c6-4380-8e03-0f8b5093e011" />


The prompt was updated to follow RSNA pneumonia detection guidelines explicitly. Despite this refinement, the model produced the same outcome as Strategy 2, missing the pneumonia diagnosis and returning a report similar in content. This could be due to MedGemma being designed as a generative report model rather than a classifier — it produces text based on learned patterns, so even with structured guidance, it may not reliably capture subtle pathology in low-resolution images.


## 4. Qualitative Analysis: VLM vs Ground Truth vs CNN

Looking across all tested images, a few patterns emerge clearly.

For clear, well-defined pneumonia cases — typically bacterial pneumonia with dense lobar consolidation — the CNN gave more correct answers. As a diagnostic-stage tool, the CNN is better because it is explicitly trained to minimize classification error and directly outputs a binary prediction, making it more reliable for identifying the presence or absence of pneumonia, even when subtle details might be missed by generative models.

The most interesting cases are images where both the CNN and VLM give wrong answers. These tend to be subtle or atypical presentations, such as early-stage viral pneumonia with faint bilateral infiltrates, or borderline normal X-rays with slight haziness. The CNN missed these because fine details are lost at 28×28 resolution. MedGemma missed them for the same reason: when an image is too low resolution, even a medically-trained VLM cannot recover clinical information that was never present.

In the cases where predictions are wrong, the CNN provides no explanation, making it difficult to understand why the image was misclassified. MedGemma, on the other hand, sometimes produces descriptive text that explains its reasoning, even when the prediction is incorrect. This descriptive output can act as a red flag or prompt further investigation, offering more transparency and reducing potential risk compared to an unexplained binary label.

To summarize, the VLM does not replace the CNN’s classification role, but it complements it. The CNN provides a fast, quantifiable prediction, while the VLM explains what is visible in the image, adding interpretability and context.

## 5. Model Strengths and Limitations

**Strengths**

MedGemma's biggest strength is its output format. Even when the diagnosis is wrong, the reports it produces look and read like real radiology reports — structured, systematic, covering all the right anatomical areas. This makes it immediately usable as a draft that a radiologist could review and correct, rather than a black-box binary output.

It also showed sensitivity to image quality — when the X-ray was low resolution, it often noted the limitation implicitly by describing the image as ambiguous, which is actually appropriate clinical behaviour. And unlike the CNN, it requires no training data or fine-tuning to get started.

**Limitations**

The clearest limitation is resolution sensitivity. PneumoniaMNIST images are 28×28 pixels — far below the 1024×1024+ resolution of real clinical X-rays. MedGemma was likely trained on high-resolution DICOM images, so feeding it tiny thumbnails degrades its performance significantly. This is not a flaw in the model; it's a mismatch between the dataset and the model's design.

Another limitation is the lack of quantitative output. The CNN can produce a probability score and be evaluated with AUC, F1, and recall across hundreds of images. The VLM produces text, which requires manual review or an additional NLP layer to evaluate at scale. For this task, qualitative review of 10 images was sufficient, but production deployment would need an automated evaluation pipeline.

Finally, the model is slow on CPU — each image takes a long time to process. For batch evaluation of 624 test images, this would require GPU acceleration or a smaller/quantized model variant.
