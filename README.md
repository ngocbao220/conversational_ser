# Speech Emotion Recognition + Reasoning Baseline

Baseline này dùng `AbstractTTS/IEMOCAP`, gom nhãn IEMOCAP về 4 class:

- `neutral -> neutral`
- `happy + excited -> happy`
- `sad -> sad`
- `angry + frustrated -> angry`

Các nhãn minor như `fear`, `surprise`, `disgust`, `other`, `tie_prediction` bị bỏ qua trong loader.

## Cài đặt

```bash
pip install -r requirements.txt
```

## Cài đặt trên cloud NVIDIA GPU CUDA 12.6

Nên cài PyTorch CUDA riêng trước, sau đó cài các dependency còn lại:

```bash
conda create -n speech python=3.11 -y
conda activate speech
pip install -r requirements-cu126-torch.txt
pip install -r requirements-cloud.txt
```

Nếu muốn dùng một file requirements duy nhất trên cloud Linux có NVIDIA GPU:

```bash
conda create -n speech python=3.11 -y
conda activate speech
pip install -r requirements-cloud-cu126.txt
```

Hoặc tạo trực tiếp từ file môi trường:

```bash
conda env create -f environment-cloud-cu126.yml
conda activate speech
```

Kiểm tra GPU:

```bash
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'no cuda')"
```

## Train

```bash
python train.py --config config.yaml
```

Checkpoint tốt nhất theo validation macro F1 được lưu ở:

```text
outputs/ser_baseline/best.pt
```

Log epoch được in dạng JSON và lưu vào `outputs/ser_baseline/history.json`.

## Inference

```bash
python inference.py \
  --audio path/to/audio.wav \
  --checkpoint outputs/ser_baseline/best.pt \
  --transcript "optional transcript"
```

Output gồm emotion dự đoán, confidence, probability từng class, acoustic cues, và explanation text đơn giản.

## Cấu trúc

- `dataset.py`: load `AbstractTTS/IEMOCAP`, map nhãn 8-to-4, bỏ nhãn minor/tie, tạo split nếu dataset chỉ có train.
- `model.py`: SSL encoder Hugging Face + mean/attention pooling + MLP classifier.
- `features.py`: acoustic cues cho explanation.
- `train.py`: training loop, validation metrics, best checkpoint theo macro F1.
- `evaluate.py`: accuracy, macro F1, weighted F1, confusion matrix.
- `inference.py`: predict emotion và tạo explanation.
