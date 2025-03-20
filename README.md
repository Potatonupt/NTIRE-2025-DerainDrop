# NTIRE 2025: The First Challenge on Day and Night Raindrop Removal for Dual-Focused Images

This repository contains the solution for the **NTIRE 2025 Challenge on Day and Night Raindrop Removal for Dual-Focused Images**. Our solution is based on the paper:

> **"Learning Weather-General and Weather-Specific Features for Image Restoration Under Multiple Adverse Weather Conditions"** by Zhu et al., CVPR 2023.

---

## ⚡ Runtime and System Information
- **Runtime per video [s]:** 1.38  
- **CPU[1] / GPU[0]:** 0 (CPU-only)  
- **Extra Data [1] / No Extra Data [0]:** 0

---

## 📚 Pretrained Models
We provide three different settings for pretrained models. You can download the models using the following links:

- [WGWS](./stage2res/WGWSS22/)
- [NAFNET](./nafnet_train_log/models/epoch_1700.pth)
---

## 🧪 Test Instructions
You can test the pretrained models using the steps below.

### 1. Modify the Paths
Update the paths for:
- Datasets
- Pre-trained model weights

### 2. Run the Model
#### 2.1 WGWS
```bash
python testing_model_Seting1.py --flag K1 --base_channel 18 --num_block 6 --save_path [path to your save_path]
```
#### 2.2 NAFNET
```bash
python test_nafnet.py 
```



