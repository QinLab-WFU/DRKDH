# DRKDH
Source code for TOMM 2026 paper “Deep Relational Knowledge Distillation Hashing via Relaxed Masking Triplet Optimization for Large-scale Image Retrieval”.

## Training

### Processing dataset
- THINGS: https://things-initiative.org
- ImageNet: https://image-net.org/challenges/LSVRC/2012/
- NUS-WIDE: https://lms.comp.nus.edu.sg/wp-content/uploads/2019/research/nuswide/NUS-WIDE.html
- MIRFLICKR-25K: https://press.liacs.nl/mirflickr/

### Start
> cd DRKDH_STAGE1
> python train.py
> python save_embedings.py
> cd ../DRKDH_STAGE2
> python train.py