# VAR_Ultrasound
Implementation of Visual Autoregressive Modeling in Ultrasound Images

1-21 datasets: https://drive.google.com/file/d/1UNPPyIyMFCuRUkZodLrBcJR1Pt39PqiR/view?usp=drive_link



| Dataset                |   Shape   | Image count | Organ                               | Notes                                                                                                           |
|------------------------|:---------:|------------:|-------------------------------------|-----------------------------------------------------------------------------------------------------------------|
| 105US                  | fan       |         105 | Breast                              | –                                                                                                               |
| AbdomenUS              | fan       |        1543 | Abdomen (multi-organ)               | –                                                                                                               |
| ACOUSLIC               | –         |          –  | Fetal abdomen                       | Restricted access                                                                                               |
| ASUS                   | –         |          –  | Multi-organ (POCUS)                 | can't find                                                                                                      |
| AUL                    | fan       |         200 | Liver                               | –                                                                                                               |
| brachial_plexus        | –         |          –  | Brachial plexus nerve               | Video files                                                                                                     |
| BrEaST                 | rectangular |        256 | Breast                              | –                                                                                                               |
| BUID                   | rectangular |        232 | Breast                              | –                                                                                                               |
| BUS_UC                 | rectangular |        489 | Breast                              | –                                                                                                               |
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


