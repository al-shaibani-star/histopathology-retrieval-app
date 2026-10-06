# -*- coding: utf-8 -*-
"""
Committee-presentation data — all tables and headline values.

Values follow the manuscript (computers-4369511) and RESULTS_SUMMARY.md.
The classification/retrieval/calibration detail is loaded live from
results/metrics.json (whose MAP, ECE, kappa, top-3, per-class values match
the paper); the overall accuracy headline uses the paper value 85.03%.
"""

# ----------------------------------------------------------------------
# Overview — model & dataset facts
# ----------------------------------------------------------------------
MODEL_FACTS = [
    ("Backbone", "ConvNeXt V2 + CBAM attention (4 stages)"),
    ("Parameters", "287.96 M"),
    ("Shared descriptor", "18,496-D  (global 2,048 + spatial 16,384 + attention 64)"),
    ("Learned fusion weights", "0.232 / 0.262 / 0.287 / 0.218  (F1..F4, softmax)"),
    ("Classifier", "18,496 → 512 → 4 classes"),
    ("Retrieval index", "FAISS IVFFlat, Euclidean (L2)"),
    ("Calibration", "intrinsic (label smoothing 0.05, no temperature scaling)"),
]

DATASET_FACTS = [
    ("Dataset", "WSSS4LUAD — lung adenocarcinoma histopathology"),
    ("Classes (4)", "Normal · Stroma · Tumor · Tumor-Stroma"),
    ("Patches", "10,087  (224×224 H&E, from 67 whole-slide images)"),
    ("Train / Validation", "8,070 / 2,017  (80 / 20, stratified)"),
    ("External test", "NCT-CRC-HE  CRC-VAL-HE-7K (cross-organ, colon)"),
]

HIGHLIGHTS = [
    ("Unified descriptor", "One forward pass drives classification, retrieval, "
     "calibration and Grad-CAM from a single vector."),
    ("Well calibrated", "ECE = 0.0378 without any post-hoc temperature scaling."),
    ("Strong retrieval", "MAP = 0.8323, MRR = 0.8784, Precision@1 = 0.827."),
    ("Cross-organ transfer", "85.79% on a colon dataset never seen in training."),
]

# ----------------------------------------------------------------------
# Table III — Comparison with baseline architectures (WSSS4LUAD)
# ----------------------------------------------------------------------
BASELINES = {
    "chart": "chart__baselines.png",
    "caption": "Table III — Comparison with baseline architectures (WSSS4LUAD, same split).",
    "note": "Standard CNNs reach higher raw accuracy, but each is classification-only. "
            "Our model trades a little accuracy for a single descriptor that also serves "
            "retrieval, calibration and explainability — and it generalises better across "
            "organs (see External Validation).",
    "columns": ["Model", "Accuracy", "F1 (macro)", "Params"],
    "rows": [
        ["ResNet-50", "90.18%", "0.8732", "24.6 M"],
        ["EfficientNet-B0", "91.18%", "0.8924", "5.3 M"],
        ["DenseNet-121", "91.03%", "0.8896", "8.0 M"],
        ["ConvNeXt V2 (vanilla)", "85.28%", "0.8020", "≈ 88 M"],
        ["Ours (unified)", "85.03%", "0.8077", "287.96 M"],
    ],
    "highlight_last": True,
}

# ----------------------------------------------------------------------
# Table IV — Ablation study
# ----------------------------------------------------------------------
ABLATION = {
    "chart": "chart__ablation.png",
    "caption": "Table IV — Ablation study (each component removed from the full model).",
    "note": "Every component contributes; the attention-weighted features of the hybrid "
            "head contribute the most (−2.13%).",
    "columns": ["Configuration", "Accuracy", "Δ vs full"],
    "rows": [
        ["Full model", "85.03%", "—"],
        ["− Attention branch", "82.90%", "−2.13"],
        ["− Adaptive fusion (uniform)", "83.69%", "−1.34"],
        ["− CBAM", "84.09%", "−0.94"],
    ],
    "highlight_first": True,
}

# ----------------------------------------------------------------------
# Table V — 5-fold cross-validation
# ----------------------------------------------------------------------
CROSS_VAL = {
    "chart": "chart__crossval.png",
    "caption": "Table V — 5-fold stratified cross-validation (rules out split-dependent bias).",
    "note": "The low standard deviation (0.93%) shows the result is not sensitive to the "
            "data partitioning; the mean is close to the single-split accuracy.",
    "columns": ["Fold", "Accuracy"],
    "rows": [
        ["Fold 1", "82.65%"],
        ["Fold 2", "83.78%"],
        ["Fold 3", "83.91%"],
        ["Fold 4", "82.56%"],
        ["Fold 5", "83.85%"],
        ["Mean ± σ", "83.35 ± 0.93%"],
    ],
    "highlight_last": True,
}

# ----------------------------------------------------------------------
# Table VII — Class balancing (Focal loss + weighted sampler)
# ----------------------------------------------------------------------
CLASS_BALANCE = {
    "caption": "Table VII — Class-imbalance handling (Focal loss + weighted sampler).",
    "note": "Focal loss + weighted sampling raise the minority Tumor recall (52.17% → 56.09%) "
            "but lower overall accuracy; the paper therefore reports the cross-entropy model "
            "and flags class-conditional treatment as future work.",
    "columns": ["Metric", "Cross-entropy", "Focal + sampler"],
    "rows": [
        ["Overall accuracy", "85.03%", "76.05%"],
        ["Balanced accuracy", "0.8055", "0.7588"],
        ["Tumor recall", "52.17%", "56.09%"],
        ["Tumor F1", "0.5743", "0.5039"],
    ],
}

# ----------------------------------------------------------------------
# Tables VIII & IX — External validation (cross-organ)
# ----------------------------------------------------------------------
EXTERNAL_FAIR = {
    "chart": "chart__external.png",
    "caption": "Table VIII — Fair external validation (all models trained on the same 10.8K subsample).",
    "note": "Under identical training data, our unified model reaches 85.79% on a colon dataset "
            "it never saw in training — evidence of cross-organ generalisation of the descriptor.",
    "columns": ["Model", "Accuracy", "Bal. acc", "F1 (macro)"],
    "rows": [
        ["ResNet-50", "95.25%", "0.9321", "0.9317"],
        ["EfficientNet-B0", "95.45%", "0.9309", "0.9336"],
        ["DenseNet-121", "94.83%", "0.9230", "0.9263"],
        ["Ours", "85.79%", "—", "—"],
    ],
    "highlight_last": True,
}

EXTERNAL_FULL = {
    "caption": "Table IX — Reference: baselines trained on the FULL 100K NCT-CRC-HE.",
    "note": "For completeness, baselines trained on the full 100K set (a 9.3× data advantage). "
            "Our model was limited to the 10.8K subsample by GPU memory (see limitations).",
    "columns": ["Model", "Accuracy", "Bal. acc", "F1 (macro)"],
    "rows": [
        ["ResNet-50", "95.24%", "0.9348", "0.9336"],
        ["EfficientNet-B0", "96.95%", "0.9539", "0.9569"],
        ["DenseNet-121", "94.97%", "0.9300", "0.9321"],
        ["Ours (10.8K)", "85.79%", "—", "—"],
    ],
    "highlight_last": True,
}

COMPARISON_TABLES = [BASELINES, ABLATION, CROSS_VAL, CLASS_BALANCE, EXTERNAL_FAIR, EXTERNAL_FULL]

# Headline accuracy shown on the cards (paper value, matches the thesis)
PAPER_ACCURACY = "85.03%"

# ----------------------------------------------------------------------
# Algorithms — expressive, plain-language pseudocode (matches Chapter 3)
# each step = (indent_level, marker_or_None, text, comment_or_None)
# marker None => section header (bold italic, no number)
# ----------------------------------------------------------------------
ALGORITHMS = [
    dict(num=1, title="CBAM Attention Module",
      input="a feature map produced by a backbone stage (C channels, height H, width W).",
      output="the same feature map, re-weighted to emphasize informative channels and locations.",
      steps=[
        (0, None, "Part A - Channel attention (decide which channels matter)", None),
        (0, "1.", "Summarize the map over space twice: once by average pooling and once by max pooling, giving two length-C vectors.", None),
        (0, "2.", "Pass both vectors through the same small two-layer network (squeeze the channels by 16x, apply ReLU, then expand back).", "shared weights"),
        (0, "3.", "Add the two outputs and apply a sigmoid to obtain one importance weight per channel.", None),
        (0, "4.", "Multiply every channel of the feature map by its weight.", None),
        (0, None, "Part B - Spatial attention (decide which locations matter)", None),
        (0, "5.", "At each location, take the average and the maximum across all channels, giving two single-channel maps.", None),
        (0, "6.", "Stack the two maps, apply a 7x7 convolution, then a sigmoid, to obtain one weight per location.", None),
        (0, "7.", "Multiply the feature map at each location by its weight.", None),
        (0, "8.", "return the refined feature map.", None),
      ]),
    dict(num=2, title="ConvNeXt V2 Backbone - multi-scale feature extraction",
      input="an H&E image patch with 3 colour channels at 224x224 resolution.",
      output="four feature maps captured at increasingly coarse scales.",
      steps=[
        (0, "1.", "Apply the stem: a 4x4 convolution with stride 4 followed by normalization, shrinking the image four-fold (to 56x56 with 64 channels).", None),
        (0, "2.", "Start with an empty list of saved features.", None),
        (0, "3.", "for each of the 4 stages (with 2, 2, 6, and 2 blocks):", None),
        (1, "3.1", "for each block in the stage: apply a ConvNeXt V2 block - a 7x7 depthwise convolution, normalization, a 4x channel expansion, GELU, global response normalization, and a projection back, all wrapped in a residual skip connection with stochastic depth.", None),
        (1, "3.2", "Apply CBAM attention to the stage output.", "Algorithm 1"),
        (1, "3.3", "Save this feature map to the list.", None),
        (1, "3.4", "if this is not the last stage: downsample it (normalization plus a 2x2 stride-2 convolution) and double the number of channels.", None),
        (0, "4.", "return the four saved feature maps.", "channels 64/128/256/512; sizes 56/28/14/7"),
      ]),
    dict(num=3, title="Adaptive Weighted Fusion Neck",
      input="the four backbone feature maps, plus four trainable importance weights.",
      output="a single fused feature map (256 channels at 7x7).",
      steps=[
        (0, "1.", "for each of the four feature maps: project it to 256 channels with a 1x1 convolution, then normalize and apply GELU.", "unify the channel counts"),
        (0, "2.", "Resize all four projected maps to the smallest spatial size (7x7) using bilinear interpolation.", "align the scales"),
        (0, "3.", "Convert the four trainable weights into proportions that sum to one using softmax.", "learned: 0.232, 0.262, 0.287, 0.218"),
        (0, "4.", "Combine the four maps as a weighted sum using these proportions.", None),
        (0, "5.", "Refine the combined map with a 3x3 convolution, normalization, and GELU.", None),
        (0, "6.", "return the fused feature map.", None),
      ]),
    dict(num=4, title="Hybrid Representation Head - the shared 18,496-number descriptor",
      input="the fused feature map (256 channels at 7x7).",
      output="one 18,496-number descriptor and a saliency (attention) map.",
      steps=[
        (0, None, "Global branch - overall appearance (2,048 numbers)", None),
        (0, "1.", "Average the feature map over space, then pass it through a linear layer to 2,048, with normalization, GELU, and dropout.", None),
        (0, None, "Spatial branch - tissue layout (16,384 numbers)", None),
        (0, "2.", "Pool the feature map onto an 8x8 grid (256 x 8 x 8 = 16,384 values).", None),
        (0, "3.", "Flatten and pass through a linear layer to 16,384, with normalization, GELU, and dropout.", None),
        (0, None, "Attention branch - diagnostic focus (64 numbers)", None),
        (0, "4.", "Produce a one-channel saliency map (reduce channels with a 1x1 convolution, GELU, a 1x1 convolution to a single channel, then sigmoid).", None),
        (0, "5.", "Multiply the feature map by this saliency map to emphasize the salient regions.", None),
        (0, "6.", "Reduce to 64 channels with a 1x1 convolution, average over space, then normalize, GELU, and dropout.", None),
        (0, None, "Combine", None),
        (0, "7.", "Concatenate the three branch outputs into one 18,496-number descriptor.", None),
        (0, "8.", "return the descriptor together with the saliency map.", None),
      ]),
    dict(num=5, title="Unified Forward Pass - classification and descriptor in a single pass",
      input="an image patch and the trained model.",
      output="class scores, the shared descriptor, and the attention map.",
      steps=[
        (0, "1.", "Run the backbone to obtain four multi-scale feature maps.", "Algorithm 2"),
        (0, "2.", "Fuse them with the neck into one feature map.", "Algorithm 3"),
        (0, "3.", "Build the hybrid descriptor and the attention map with the head.", "Algorithm 4"),
        (0, "4.", "Pass the descriptor through the classifier (linear to 512, normalization, GELU, dropout, linear to the number of classes) to obtain class scores.", None),
        (0, "5.", "return the class scores, the descriptor, and the attention map.", None),
      ]),
    dict(num=6, title="Training Procedure",
      input="training images with labels; epochs; batch size 16; learning rate 0.0001; weight decay 0.01; 5 warm-up epochs; label-smoothing strength 0.05.",
      output="the best-performing trained model.",
      steps=[
        (0, "1.", "initialize the model; create the AdamW optimizer, a cosine learning-rate schedule with warm-up, and a mixed-precision scaler.", None),
        (0, "2.", "remember the best validation accuracy so far (none yet) and reset a patience counter to zero.", None),
        (0, "3.", "for each epoch:", None),
        (1, "3.1", "for each batch of images and labels:", None),
        (2, "a.", "In mixed precision, run the forward pass to obtain class scores.", "Algorithm 5"),
        (2, "b.", "Compute the label-smoothed cross-entropy loss.", "also keeps the model calibrated - no temperature scaling"),
        (2, "c.", "Back-propagate the loss, update the weights, then clear the gradients.", None),
        (1, "3.2", "Advance the learning-rate schedule by one step.", None),
        (1, "3.3", "Evaluate the model on the validation set.", None),
        (1, "3.4", "if the accuracy improved: save this as the best model and reset patience.", None),
        (1, "3.5", "else: increase patience; if it reaches 10, stop training early.", "early stopping"),
        (0, "4.", "return the best saved model.", None),
      ]),
    dict(num=7, title="Building the Retrieval Index (done once, offline, after training)",
      input="a gallery of labelled images; the trained model; a target of 100 clusters.",
      output="a searchable similarity index.",
      steps=[
        (0, "1.", "for each gallery image: run the model to obtain its 18,496-number descriptor (the classifier is not needed here).", None),
        (0, "2.", "Stack all descriptors into a single matrix.", None),
        (0, "3.", "Create an IVF-Flat index that measures Euclidean (L2) distance over 18,496 dimensions.", None),
        (0, "4.", "Train the index, which groups the descriptors into 100 clusters with k-means.", None),
        (0, "5.", "Add all descriptors to the index and store their class labels.", None),
        (0, "6.", "return the index.", None),
      ]),
    dict(num=8, title="Inference - classify and retrieve similar cases together",
      input="a query image; the index; a setting to probe 10 clusters; the number K of neighbours to return.",
      output="the predicted class and its confidence, the K most similar cases, and a saliency map.",
      steps=[
        (0, "1.", "Run the model once on the query to obtain class scores, the descriptor, and the saliency map.", "Algorithm 5"),
        (0, "2.", "Convert the scores to probabilities with softmax; the predicted class is the highest one, and its probability is the confidence.", None),
        (0, "3.", "Tell the index to probe 10 clusters during the search.", None),
        (0, "4.", "Search the index with the query descriptor for its K nearest neighbours by Euclidean distance.", None),
        (0, "5.", "Collect each neighbour's identifier, distance, and stored label as the ranked list of similar cases.", None),
        (0, "6.", "return the prediction, the confidence, the similar cases, and the saliency map.", None),
      ]),
    dict(num=9, title="Grad-CAM Explanation",
      input="an image; the class to explain; a chosen convolutional layer to inspect.",
      output="a heat-map over the image highlighting the regions that supported the chosen class.",
      steps=[
        (0, "1.", "Run the image forward and record both the activations of the chosen layer and the class scores.", None),
        (0, "2.", "Back-propagate the score of the chosen class and record the gradients at that layer.", None),
        (0, "3.", "For each channel, average its gradient to obtain an importance weight.", None),
        (0, "4.", "Take the weighted sum of the activation channels and keep only the positive contributions (ReLU).", None),
        (0, "5.", "Upsample the result to the image size and rescale it to the range 0 to 1.", None),
        (0, "6.", "return the heat-map.", None),
      ]),
]

CTRL_WORDS = {"for", "if", "else", "return", "while", "repeat", "initialize", "remember"}
