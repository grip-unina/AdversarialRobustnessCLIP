# AdversarialRobustnessCLIP

[![Github](https://img.shields.io/badge/Github%20webpage-222222.svg?style=for-the-badge&logo=github)](https://github.com/grip-unina/AdversarialRobustnessCLIP)
[![arXiv](https://img.shields.io/badge/-arXiv-B31B1B.svg?style=for-the-badge)](https://arxiv.org/abs/2407.19553)
[![GRIP](https://img.shields.io/badge/-GRIP-0888ef.svg?style=for-the-badge)](https://www.grip.unina.it)

This is the official repository of the paper:
[Exploring the Adversarial Robustness of CLIP for AI-generated Image Detection](https://arxiv.org/abs/2407.19553).

Vincenzo De Rosa, Fabrizio Guillaro, Giovanni Poggi, Davide Cozzolino, and Luisa Verdoliva.

## Overview
In recent years, many forensic detectors have been proposed to detect AI-generated images and prevent their use for malicious purposes. Convolutional neural networks (CNNs) have long been the dominant architecture in this field and have been the subject of intense study. However, recently proposed Transformer-based detectors have been shown to match or even outperform CNN-based detectors, especially in terms of generalization. In this paper, we study the adversarial robustness of AI-generated image detectors, focusing on Contrastive Language-Image Pretraining (CLIP)-based methods that rely on Visual Transformer (ViT) backbones and comparing their performance with CNN-based methods. We study the robustness to different adversarial attacks under a variety of conditions and analyze both numerical results and frequency-domain patterns. CLIP-based detectors are found to be vulnerable to white-box attacks just like CNN-based detectors. However, attacks do not easily transfer between CNN-based and CLIP-based methods. This is also confirmed by the different distribution of the adversarial noise patterns in the frequency domain. Overall, this analysis provides new insights into the properties of forensic detectors that can help to develop more effective strategies.
<center> <img src="./docs/image.png" alt="preview" width="80%" /> </center>

## Dataset
The dataset can be downloaded [here](https://www.grip.unina.it/download/prog/AdversarialRobustnessCLIP/AdversarialRobustnessCLIP.zip) or using the following script:

```
bash ./get_dataset.sh OUTPUT_DIR_PATH
```

## Code
Coming Soon

## License

Copyright (c) 2024 Image Processing Research Group of University Federico II of Naples ('GRIP-UNINA'). 

All rights reserved.

This software should be used, reproduced and modified only for informational and nonprofit purposes.

By downloading and/or using any of these files, you implicitly agree to all the
terms of the license, as specified in the document LICENSE.md
(included in this package)

## Bibtex 

```
@article{de2024exploring,
  title={Exploring the Adversarial Robustness of CLIP for AI-generated Image Detection},
  author={De Rosa, Vincenzo and Guillaro, Fabrizio and Poggi, Giovanni and Cozzolino, Davide and Verdoliva, Luisa},
  journal={arXiv preprint arXiv:2407.19553},
  year={2024}
}
```

## Acknowledgments
We gratefully acknowledge the support of this research by a TUM-IAS Hans Fischer Senior Fellowship and a Google Gift. In addition, this work has received funding by the European Union under the Horizon Europe vera.ai project, Grant Agreement number 101070093, and was partially supported by SERICS (PE00000014) under the MUR National Recovery and Resilience Plan, funded by the European Union - NextGenerationEU. Finally, we thank the partnership and collaboration with the São Paulo Research Foundation (Fapesp) Horus project, Grant #2023/12865-8.
