# Speech Emotion Recognition Baselines

Repo này có baseline WavLM SER trên Kaggle IEMOCAP local trong `iemocap/`, gom nhãn IEMOCAP về 4 class:

- `ang -> angry`
- `hap + exc -> happy`
- `neu -> neutral`
- `sad -> sad`

Các nhãn ngoài `{ang, hap, exc, neu, sad}` bị bỏ qua.

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

## Experiment 1 - WavLM Baseline No MAL/No TIM

Đây là control condition chính theo `instructions/baseline.md`.

```bash
./scripts/train_wavlm_baseline.sh
```

Config YAML:

```text
configs/wavlm_baseline_no_mal_no_tim.yaml
```

Mặc định:

- dataset local: `iemocap/`
- nếu `iemocap/` chưa tồn tại, config mặc định sẽ tự tải Kaggle dataset `sangayb/iemocap` rồi đặt folder thành `iemocap/`
- LOSO: `test_session: 5`
- validation: 10% dialogue-level split từ train sessions
- model: frozen `microsoft/wavlm-base`
- pooling: attentive statistics pooling
- không dùng dialogue memory, timestamps, speaker, MAL, TIM trong model
- checkpoint tốt nhất chọn theo validation UA

Để auto-download từ Kaggle, cần cài `kaggle` và cấu hình credentials bằng một trong hai cách:

```bash
export KAGGLE_USERNAME="..."
export KAGGLE_KEY="..."
```

hoặc đặt `kaggle.json` tại `~/.kaggle/kaggle.json`.

Outputs:

```text
results/wavlm_baseline_no_mal_no_tim/metrics.json
results/wavlm_baseline_no_mal_no_tim/predictions.csv
results/wavlm_baseline_no_mal_no_tim/config.json
results/wavlm_baseline_no_mal_no_tim/confusion_matrix.csv
results/wavlm_baseline_no_mal_no_tim/confusion_matrix.png
results/wavlm_baseline_no_mal_no_tim/best.pth
results/wavlm_baseline_no_mal_no_tim/last.pth
```

## Legacy B0 - Utterance-Level Baseline

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
./scripts/train_b0.sh
```

Các tham số train nằm ở đầu [scripts/train_b0.sh](/Users/ngocbao/Documents/Document/research/main/speech/exps/demo/scripts/train_b0.sh), ví dụ:

```bash
ENCODER_NAME="microsoft/wavlm-base"
POOLING="mean"
FREEZE_ENCODER=true
BATCH_SIZE=4
EPOCHS=5
LR_SCHEDULER="cosine"
WARMUP_RATIO=0.1
EARLY_STOPPING_PATIENCE=0
```

Checkpoint được lưu theo 2 kiểu:

```text
outputs/b0_utterance/best.pt
outputs/b0_utterance/last.pt
```

- `best.pt`: epoch tốt nhất theo validation macro F1.
- `last.pt`: checkpoint mới nhất sau mỗi epoch.
- `EARLY_STOPPING_PATIENCE=0` nghĩa là tắt early stopping.
- `LR_SCHEDULER` hỗ trợ `linear`, `cosine`, `constant`.

Log dễ đọc được append dần vào `outputs/b0_utterance/train.log`, phù hợp để quan sát bằng:

```bash
tail -f outputs/b0_utterance/train.log
```

Toàn bộ history theo epoch được lưu vào `outputs/b0_utterance/history.json` sau khi train xong.

Muốn bật Weights & Biases, sửa trong `scripts/train_b0.sh`:

```bash
USE_WANDB=true
WANDB_PROJECT="conversational-SER"
WANDB_RUN_NAME="b0-wavlm"
```

Trên server chưa login wandb:

```bash
wandb login
```

## Evaluate B0

```bash
./scripts/evaluate_b0.sh
```

Metrics được lưu mặc định tại:

```text
outputs/b0_utterance/test_metrics.json
```

## B01 - LOSO + Unfreeze 4 SSL Layers

B01 dùng cùng classifier với B0, nhưng:

- `TRAINABLE_ENCODER_LAYERS=4`: unfreeze 4 transformer layer cuối của SSL encoder.
- `SPLIT_STRATEGY="loso"` và `TEST_SESSION="Ses05"`: train trên Ses01-Ses04, test trên Ses05.

```bash
./scripts/train_b01.sh
./scripts/evaluate_b01.sh
```

Output mặc định:

```text
outputs/b01_loso_unfreeze4/
```

## Upload Checkpoint

```bash
hf auth login
./scripts/upload.sh --model b0
```

Mặc định script upload toàn bộ `outputs/b0_utterance` vào `ngocbao05/ser/b0`. Các version sau có thể dùng cùng repo và đổi folder bằng `--model`:

```bash
./scripts/upload.sh --model b1
```

## Download And Evaluate

```bash
hf auth login
./scripts/download.sh --model b0
```

Mặc định script tải `ngocbao05/ser/b0` về `outputs/hf_checkpoints/b0`, sau đó evaluate `best.pt` và ghi metrics tại `outputs/hf_checkpoints/b0/test_metrics.json`.

## Inference B0

```bash
./scripts/infer_b0.sh
```

Output gồm emotion dự đoán, confidence, và probability từng class.

## Cấu trúc

- `models/`: model definitions, gồm `models/wavlm_baseline.py`.
- `utils/`: config, Kaggle IEMOCAP parser, dataset loader, metrics, feature helpers.
- `scripts/`: train/evaluate/infer/export/upload/download implementations và shell entrypoints.
