<h1>Pneumonia Detection Using DenseNet-121</h1>

<div class="section">
<h2>1. Project Overview</h2>
<p>
This project presents a deep learning-based system for automatic pneumonia detection 
from chest X-ray images. The system integrates classification, explainability using Grad-CAM, 
and structured diagnostic report generation to enhance clinical interpretability.
</p>
</div>

<div class="section">
<h2>4. Methodology</h2>

<h3>4.1 Data Preprocessing</h3>
<p>
All chest X-ray images were resized to a standardized resolution to ensure dataset consistency.
Pixel normalization was applied to scale values between 0 and 1, enabling stable gradient flow.
</p>

<ul>
<li>Random Rotation</li>
<li>Horizontal Flipping</li>
<li>Zoom Augmentation</li>
<li>Contrast Adjustment</li>
</ul>

<p>
These augmentation techniques simulate realistic medical imaging variations while 
preserving diagnostic features.
</p>

<h3>4.2 Model Architecture</h3>
<p>
The framework utilizes <strong>DenseNet-121</strong>, chosen for efficient feature propagation 
and mitigation of vanishing gradient problems.
</p>

<p>
Each layer receives feature maps from all preceding layers:
</p>

<p><em>H<sub>l</sub>(x) = H<sub>l</sub>([x₀, x₁, ..., x<sub>l−1</sub>])</em></p>

<p>
Transition layers consist of 1×1 convolution followed by 2×2 average pooling.
The final layers include:
</p>

<ul>
<li>Global Average Pooling</li>
<li>Fully Connected Dense Layer</li>
<li>Softmax Output Layer (Binary Classification)</li>
</ul>

<p>
This architecture effectively detects lung abnormalities such as consolidations,
infiltrations, and opacity patterns.
</p>

<h3>4.3 Training Configuration</h3>
<ul>
<li>Loss Function: Cross-Entropy</li>
<li>Optimizer: Adam</li>
<li>Mini-Batch Gradient Descent</li>
<li>Early Stopping</li>
<li>Learning Rate Scheduling</li>
</ul>

<h3>4.4 Explainability Module</h3>
<p>
Grad-CAM was integrated to generate heatmaps highlighting discriminative lung regions.
A circular marker highlights peak activation areas to assist clinician review.
</p>

<h3>4.5 Diagnostic Report Generation</h3>
<p>
For each prediction, a structured PDF report is generated containing:
</p>

<ul>
<li>Original X-ray Image</li>
<li>Grad-CAM Heatmap Overlay</li>
<li>Highlighted Lesion Region</li>
<li>Classification Result</li>
<li>Confidence Score</li>
<li>Optional Patient Details</li>
</ul>

<h3>4.6 System Workflow</h3>
<div class="highlight">
Image Input → Preprocessing → DenseNet-121 Classification → Grad-CAM Visualization → PDF Report Generation
</div>

</div>

<div class="section">
<h2>5. Results</h2>

<h3>Performance Metrics</h3>

<table>
<tr>
<th>Class</th>
<th>Accuracy</th>
<th>Precision</th>
<th>Recall</th>
<th>F1-Score</th>
<th>Support</th>
</tr>

<tr>
<td>NORMAL</td>
<td>84.91%</td>
<td>0.9426</td>
<td>0.8491</td>
<td>0.8934</td>
<td>232</td>
</tr>

<tr>
<td>PNEUMONIA</td>
<td>96.92%</td>
<td>0.9153</td>
<td>0.9692</td>
<td>0.9415</td>
<td>390</td>
</tr>

<tr>
<td><strong>Overall</strong></td>
<td><strong>92.44%</strong></td>
<td>0.9254</td>
<td>-</td>
<td>-</td>
<td>622</td>
</tr>

</table>

</div>
