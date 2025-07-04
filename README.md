# VAR_Ultrasound
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
- datasets: kidneyUS, HC, TN3K (consisting of 3 datasets: DDTI, TG3K, TN3K)
- created general scripts for ultrasound cropping and resizing
- then customized cropping script for kidney, fetal head and one of the thyroid datasets (others already preprocessed properly) and resized all of them to 256x256 with the resizing script 

Training of VQ-VAE:
- Training pipeline for local running with 20 samples per dataset prepared
- Next step: adjust train script for training with all available data (capped at 1000 images per dataset)

Training of VAR:
Not yet implemented
