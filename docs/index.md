---
layout: paper
paper: Exploring the Adversarial Robustness of CLIP for AI-generated Image Detection
github_url: https://github.com/grip-unina/AdversarialRobustnessCLIP
authors:
  - name: Vincenzo De Rosa
    link: https://www.grip.unina.it/members/derosa
    index: 1
  - name: Fabrizio Guillaro
    link: https://www.grip.unina.it/members/guillaro
    index: 1
  - name: Giovanni Poggi
    link: https://www.grip.unina.it/members/poggi
    index: 1
  - name: Davide Cozzolino
    link: https://www.grip.unina.it/members/cozzolino
    index: 1
  - name: Luisa Verdoliva
    link: https://www.grip.unina.it/members/verdoliva
    index: 1
affiliations: 
  - name: University Federico II of Naples, Italy
    index: 1
---

[![Github](https://img.shields.io/badge/Github%20page-222222.svg?style=for-the-badge&logo=github)](https://github.com/grip-unina/AdversarialRobustnessCLIP)
[![arXiv](https://img.shields.io/badge/-arXiv-B31B1B.svg?style=for-the-badge)](https://arxiv.org/abs/2407.19553)
[![GRIP](https://img.shields.io/badge/-GRIP-0888ef.svg?style=for-the-badge)](https://www.grip.unina.it)

In recent years, many forensic detectors have been proposed to detect AI-generated images and prevent their use for malicious purposes. Convolutional neural networks (CNNs) have long been the dominant architecture in this field and have been the subject of intense study. However, recently proposed Transformer-based detectors have been shown to match or even outperform CNN-based detectors, especially in terms of generalization. In this paper, we study the adversarial robustness of AI-generated image detectors, focusing on Contrastive Language-Image Pretraining (CLIP)-based methods that rely on Visual Transformer (ViT) backbones and comparing their performance with CNN-based methods. We study the robustness to different adversarial attacks under a variety of conditions and analyze both numerical results and frequency-domain patterns. CLIP-based detectors are found to be vulnerable to white-box attacks just like CNN-based detectors. However, attacks do not easily transfer between CNN-based and CLIP-based methods. This is also confirmed by the different distribution of the adversarial noise patterns in the frequency domain. Overall, this analysis provides new insights into the properties of forensic detectors that can help to develop more effective strategies.

<center>
<img src="./image.png" alt="images" width="500pt" /><br />
</center>

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