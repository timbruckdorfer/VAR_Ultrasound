# VAR_Ultrasound

```text
VAR_Ultrasound/
├── Data preprocessing/        # data prep scripts / notes (by us)
├── models/                    # models from github: https://github.com/FoundationVision/VAR
│   └── __init__.py
│   └── basic_vae.py
│   └── basic_var.py
│   └── helpers.py
│   └── quant.py
│   └── var.py
│   └── vqvae.py
├── README.md                  # ← this file
├── sample_vqvae.py            # sampler multi-scale vqvae reconstruction results
├── sample_var.py              # sampler across coarse scales per class
├── train_var.py               # train the VAR (next-scale) model
└── train_vqvae.py             # train+val multi-scale VQ-VAE (with metrics & usage plots)
```


Implementation of Visual Autoregressive Modeling in Ultrasound Images



| Dataset                |   Shape   | Image count | Organ                               | Notes                                                                                                           |
|------------------------|:---------:|------------:|-------------------------------------|-----------------------------------------------------------------------------------------------------------------|
| 105US                  | fan       |         105 | Breast                              | –                                                                                                               |
| AbdomenUS              | fan       |        1543 | Abdomen (multi-organ)               | 100 liver                                                                                                            |
| ACOUSLIC               | –         |          –  | Fetal abdomen                       | Restricted access                                                                                               |
| ASUS                   | –         |          –  | -                                   | can't find                                                                                                      |
| AUL                    | fan       |         735 | Liver                               | –                                                                                                               |
| brachial_plexus        | –         |          –  | Brachial plexus nerve               | Video files                                                                                                     |
| BrEaST                 | rectangular |        256 | Breast                              | –                                                                                                               |
| BUID                   | rectangular |        232 | Breast                              | –                                                                                                               |
| BUS_UC                 | rectangular |        489 | Breast                              |  366 benign images, 489 mali images.                                                                                                               |
| BUS_UCML               | rectangular |        683 | Breast                              | –                                                                                                               |
| BUS-BRA                | rectangular |       1875 | Breast                              | –                                                                                                               |
| BUS (Dataset B)        | –         |          –  | Breast                              | Restricted access                                                                                               |
| BUSI                   | rectangular |        780 | Breast                              | –                                                                                                               |
| CAMUS                  | –         |          –  | Heart                               | Nifty files                                                                                                     |
| CardiacUDC             | –         |          –  | Heart                               | Nifty files                                                                                                     |
| CCAUI                  | rectangular |       1100 | Common carotid artery               | –                                                                                                               |
| DDTI                   | rectangular |        637 | Thyroid                             | –                                                                                                               |
| EchoCP                 | –         |          –  | Heart                               | too big                                                                                                         |
| EchoNet-Dynamic        | –         |          –  | Heart                               | Restricted access                                                                                               |
| EchoNet-Pediatric      | –         |          –  | Heart                               | Restricted access                                                                                               |
| FALLMUD                | rectangular |        813 | Lower-leg muscles                   | –                                                                                                               |
| FASS                   | fan       |        1588 | Fetal abdomen                       | –                                                                                                               |
| Fast-U-Net             | fan       | 412 / 999  | Abdomen / Head circumference        | –                                                                                                               |
| FH-PS-AOP              | –         |          –  | –                                   | Restricted access                                                                                               |
| GIST514-DB             | rectangular |        514 | Endoscopic                          | –                                                                                                               |
| HC                     | fan       |       1334 | Fetal head                          | –                                                                                                               |
| kidneyUS               | fan       |        534 | Kidney                              | –                                                                                                               |
| LUSS_phantom           | –         |          –  | –                                   | Video files                                                                                                     |
| MicroSeg               | –         |          –  | –                                   | Nifty files                                                                                                     |
| MMOTU-2D               | fan       |       1469 | Ovarian Tumor                       | –                                                                                                               |
| MMOTU-3D               | –         |          –  | –                                   | 3D images                                                                                                       |
| MUP                    | –         |          –  | –                                   | Duplicate MicroSeg                                                                                              |
| regPro                 | –         |          –  | –                                   | Nifty files                                                                                                     |
| S1                     | rectangular |        192 | Breast Tumor                        | –                                                                                                               |
| Segthy                 | –         |          –  | –                                   | Nifty files                                                                                                     |
| STMUS_NDA              | rectangular |          – | Neuromuscular disease               | 3 different muscles; healthy vs pathological data – consider merge?                                             |
| STU-Hospital           | –         |          –  | –                                   | Not enough images (≈ 40)                                                                                        |
| TG3K                   | rectangular | 637 / 3585 / 2879 | Thyroid nodules                   | 3 datasets with different volume & diversity                                                                    |
| Thyroid US Cineclip    | –         |          –  | –                                   | Restricted access                                                                                               |
| TN3K                   | –         |          –  | –                                   | Duplicate TG3K                                                                                                  |
| TNSCUI                 | –         |          –  | –                                   | Restricted access                                                                                               |
| UPBD                   | rectangular |        955 | Brachialis plexus                   | –                                                                                                               |
| US nerve Segmentation  | –         |          –  | –                                   | .tif files                                                                                                      |

Selection of organs:
- kidney (Tim)
- fetal head (Tim)
- thyroid (Tim)
- breast (Yiping)
- heart (Yiping)
- liver (Yiping)

Data preprocessing Tim:
- datasets: kidneyUS, HC, TN3K (consisting of 2 datasets: DDTI, TG3K, TN3K)
- created general scripts for ultrasound cropping and resizing
- then customized cropping script for kidney, fetal head and one of the thyroid datasets (others already preprocessed properly) and resized all of them to 256x256 with the resizing script

Data preprocessing Yiping:
- datasets: Breast(consisting of 6 datasets: BUID, BrEaST, BUS_UC, BUS_UCML, BUS-BRA, BUSI) , Heart, Liver (consisting of 3 datasets: AUL, AbdomenUS)
- Shapes & cropping: breast → rectangular, heart → fan-shape, liver → fan-shape, Resize to 256×256, Convert to RGB-format grayscale

Training of VQ-VAE:
- Training pipeline for local running with 20 samples per dataset prepared
- Next step: adjust train script for training with all available data (capped at 1000 images per dataset)

Training of VQ-VAE(Yiping side: class balancing): train_vqvae.py 
- Dataset statistics:<img width="1200" height="600" alt="per_class_by_split" src="https://github.com/user-attachments/assets/842876f5-5494-4fe4-9bdc-27536b4413fe" />
- Input: Preprocessed ultrasound images (train/, val/, test/), resized to 256×256, RGB grayscale.
- Core hyperparams (in code): VOCAB_SIZE=2048, v_patch_nums=(1,2,3,4,5,6,8,10,13,16), beta=0.25
- Method: Multi-scale quantization: from coarse → fine, add residuals across scales to reconstruct rec and compute VQ loss.
- Loss: loss = SSE_per_image + vq_loss (SSE = sum of squared errors per image).
- Output: train & val loss curves, PSNR, SSIM, Reconstruction comparisons, Codebook & vectors usage, best.pth (model weight)


Export per-scale 2D token maps from VQ-VAE: tokenize_multiscale_maps.py
- Inputs: best.pth (include Encoder, quantizer, and codebook vector) from trained VQVAE and Datasets
- Method: Encode each image per scale → 2D token maps; also concatenate all scales into a 1D sequence idx (N, L). 10 scales follow the VQ-VAE.
- Outputs: tokens_multiscale_maps.npz (include 2D token for each scale, list of scales, idx, class labels), classes.json (class names)


Train VAR (Next-Scale prediction): train_var.py 
- Inputs: tokens_multiscale_maps.npz from tokenize_multiscale_maps.py, Trained VQ-VAE codebook (to align token embeddings), classes.json.
- Methods: a VAR transformer is trained via next-scale prediction during training(Teacher Forcing), standard cross-entropy loss is used, Teacher prefix (TP) and COARSE_K are used during sampling（Autoregressive sampling starting from the Kth position）
- Outputs: ar-ckpt-best.pth (model weight), loss curves.


Multi-scale sampling: sample_var.py
- Inputs: ar-ckpt-best.pth from VAR, tokens_multiscale_maps.npz from tokenize_multiscale_maps.py, best.pth from vqvae
- Methods: Sweep across the 10 scale boundaries (K). For each K, generate: TP (Teacher Prefix) mode: random real prefix tokens, NOP (no prefix) mode: prefix from position prior.
- Outputs: decoded images with label

Results:

Multi-scale VQVAE reconstruction results: 

This image reports the multi-scale VQ-VAE training curves decomposed into the sum MSE term (per-image sum of pixel-wise squared errors), the vector-quantization loss, and their total. 
<p align="center">
<img width="459" height="331" alt="image" src="https://github.com/user-attachments/assets/93f90e4c-151e-4c23-94c8-5222619a48d1" />
</p>

This image overlays training and validation reconstruction losses.
<p align="center">
<img width="498" height="333" alt="image" src="https://github.com/user-attachments/assets/0227fab4-8046-490e-bab5-2e3028c2f82f" />
</p>

This image shows the comparison of the reconstructed sample with the original sample.
<p align="center">
  <img src="https://github.com/user-attachments/assets/2290cf50-2a89-45b8-9023-ec168917e473" alt="demo" width="420">
</p>

VAR: Next-Scale Training and Sampling results:

This image plots the mean next-scale cross-entropy over epochs for training and validation
<p align="center">
<img src="https://github.com/user-attachments/assets/e2d21073-a292-4044-b2c7-35c6711500c0" alt="image" width="434" >
</p>
The image shows the generation results for different K. Fetal head(high), Heart(middle), Thyroid-Gland(low) 
<p align="center">
<img src="https://github.com/user-attachments/assets/b668fc6b-696c-4de9-bc7c-8e5d165ebb04" alt="image" width="441" >
</p>

This image shows the Breast-Malignant(1-st row) generation with scale k = 2, Breast-Malignant(2-nd row), Breast–Benign(3-rd row), Kidney-Normal(4-th row), Liver-Benign(5-th row) generation with scale k = 7 and the rightmost image is the real (ground-truth) one.
<p align="center">
<img src="https://github.com/user-attachments/assets/d5cac326-ed12-4c03-ba07-219e31ab7f4f" alt="image" width="432" >
</p>
