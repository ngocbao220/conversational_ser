# Speech Emotion Recognition Baselines

Repo này dùng `AbstractTTS/IEMOCAP`, gom nhãn IEMOCAP về 4 class:

- `neutral -> neutral`
- `happy + excited -> happy`
- `sad -> sad`
- `angry + frustrated -> angry`

Các nhãn minor như `fear`, `surprise`, `disgust`, `other`, `tie_prediction` bị bỏ qua trong loader.

## Cài đặt trên cloud NVIDIA GPU CUDA 12.6

```bash
conda create -n speech python=3.11 -y
conda activate speech
pip install -r requirements.txt
```

Kiểm tra GPU:

```bash
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'no cuda')"
```

Kiểm tra import backbone:

```bash
python -c "from transformers import WavLMModel; print(WavLMModel.__name__)"
```

## B0 - Utterance-Level Baseline

B0 là baseline bắt buộc:

```text
audio utterance
-> frozen WavLM/Wav2Vec2
-> pooling
-> classifier
-> emotion
```

Mặc định B0 dùng `microsoft/wavlm-base`, frozen encoder, mean pooling, MLP classifier.

## Train B0

```bash
python train_b0.py --config config.yaml
```

Checkpoint tốt nhất theo validation macro F1 được lưu ở:

```text
outputs/b0_utterance/best.pt
```

Log dễ đọc được append dần vào `outputs/ser_baseline/train.log`, phù hợp để quan sát bằng:

```bash
tail -f outputs/b0_utterance/train.log
```

Toàn bộ history theo epoch được lưu vào `outputs/b0_utterance/history.json` sau khi train xong.

Muốn bật Weights & Biases, sửa trong config:

```yaml
logging:
  use_wandb: true
  wandb_project: ser-baseline
  wandb_run_name: b0-wavlm
```

Trên server chưa login wandb:

```bash
wandb login
```

## Evaluate B0

```bash
python evaluate_b0.py \
  --config config.yaml \
  --checkpoint outputs/b0_utterance/best.pt \
  --split test
```

Metrics được lưu mặc định tại:

```text
outputs/b0_utterance/test_metrics.json
```

## Inference B0

```bash
python infer_b0.py \
  --audio path/to/audio.wav \
  --checkpoint outputs/b0_utterance/best.pt
```

Output gồm emotion dự đoán, confidence, và probability từng class.

## Cấu trúc

- `dataset.py`: load `AbstractTTS/IEMOCAP`, map nhãn 8-to-4, bỏ nhãn minor/tie, tạo split nếu dataset chỉ có train.
- `b0_model.py`: B0 frozen SSL encoder + mean/attention pooling + MLP classifier.
- `features.py`: acoustic cues cho explanation.
- `train_b0.py`: B0 training loop, validation metrics, best checkpoint theo macro F1.
- `evaluate_b0.py`: B0 accuracy, macro F1, weighted F1, confusion matrix.
- `infer_b0.py`: B0 single-audio prediction.
- `metrics.py`: reusable classification metrics cho các baseline sau.
