# VLD: Video-Level Language-Driven Video-Based Visible-Infrared Person Re-Identification

This is the official repository for **VLD**, a novel framework designed for **Video-Based Visible-Infrared Person Re-Identification**.


![framework](https://github.com/user-attachments/assets/6749036a-b93b-48f1-97a1-d7db36440e24)


---
## 🚀 Getting Started

### 🏋️‍♂️ Training
To train VLD on the **VCM** dataset:
```
CUDA_VISIBLE_DEVICES=0 python main.py --mode train --dataset vcm  --pid_num 500 --output_path logs/vcm
```
To train VLD on the **BUPT** dataset:
```
CUDA_VISIBLE_DEVICES=0 python main.py --mode train --dataset bupt  --pid_num 1074 --output_path logs/bupt
```

## :car:Evaluation
To evaluate the model on the **VCM** dataset:
```
CUDA_VISIBLE_DEVICES=0 python main.py --mode test --dataset vcm   --pid_num 500   --resume_test_path logs/vcm/models  --output_path logs/vcm_test
```
To evaluate the model on the **BUPT** dataset:
```
CUDA_VISIBLE_DEVICES=0 python main.py --mode test --dataset bupt  --pid_num 1074  --resume_test_path logs/bupt/models --output_path logs/bupt_test
```


## 💾 Pretrained Models & Training Logs

We provide pretrained weights and training logs for both datasets:

### 🔹 VCM Dataset
- 📦 [Pretrained Model (VCM)](https://drive.google.com/file/d/1x9a4_RasztWPfcA2SH_KvSpSM0kmZRcf/view?usp=drive_link)
- 📑 [Training Log (VCM)](https://drive.google.com/file/d/1JuVrrfBYuJ_o20nYNWAJWCYUO1QNzbau/view?usp=drive_link)

### 🔹 BUPT Dataset
- 📦 [Pretrained Model (BUPT)](https://drive.google.com/file/d/1BIJqfo2GCbkrEuNv-PhjU8eyxBnDeD4T/view?usp=drive_link)
- 📑 [Training Log (BUPT)](https://drive.google.com/file/d/12YC_72gbpD7muVntLZH68QjdNWTQArVQ/view?usp=drive_link)

> Please download the `.pth` model files and place them into the appropriate `logs/[dataset]/models/` directory before running evaluation.


## Citation
If you find VLD useful in your research, please consider citing:
```bibtex

@article{li2025video,
  title={Video-Level Language-Driven Video-Based Visible-Infrared Person Re-Identification},
  author={Li, Shuang and Leng, Jiaxu and Kuang, Changjiang and Tan, Mingpi and Gao, Xinbo},
  journal={IEEE Transactions on Information Forensics and Security},
  year={2025},
  publisher={IEEE}
}
