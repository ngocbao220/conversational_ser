Hãy xây dựng một baseline Speech Emotion Recognition + Reasoning trên bộ dữ liệu `AbstractTTS/IEMOCAP`, theo thiết kế cơ bản tương đương ADEPT. Trước mắt chưa cần áp dụng Agentic Memory, Self-Evolving, multi-agent hay reflection loop phức tạp.

Yêu cầu chính:

1. Dataset

* Dùng dataset `AbstractTTS/IEMOCAP`.
* Mỗi sample gồm audio waveform, emotion label, và transcript nếu dataset cung cấp.
* Dataset ban đầu có 8 loại cảm xúc, cần quy về 4 class chính:

  * neutral → neutral
  * happy + excited → happy
  * sad → sad
  * angry + frustrated → angry
* Các emotion còn lại như fear, surprise, disgust, other hoặc các minor emotion không thuộc 4 nhóm trên thì bỏ qua.
* Nếu có sample bị `tie_prediction` hoặc label không rõ ràng thì bỏ qua trong version baseline đầu tiên.
* Cần viết label mapping rõ ràng trong code để dễ chỉnh sau này.

2. Model SER baseline

* Input: audio waveform.
* Dùng SSL encoder làm backbone, ví dụ WavLM / HuBERT / wav2vec2.
* Pipeline:
  audio waveform → SSL encoder → hidden states → pooling → MLP classifier → emotion prediction.
* Hỗ trợ ít nhất mean pooling.
* Nếu dễ mở rộng thì thêm attention pooling.
* Ban đầu cho phép freeze SSL encoder, chỉ train classifier.
* Có config để bật/tắt fine-tune SSL encoder.

3. Acoustic features

* Nếu khả thi, trích xuất thêm các đặc trưng cơ bản:

  * pitch
  * energy
  * duration
  * speech rate nếu có transcript
* Các feature này dùng chủ yếu để hỗ trợ phần reasoning/explanation, chưa bắt buộc đưa vào classifier ở version đầu.

4. Reasoning / Explanation module

* Sau khi model dự đoán emotion, tạo explanation dựa trên:

  * predicted emotion
  * confidence/probability
  * transcript nếu có
  * acoustic cues như pitch, energy, duration/speech rate
* Có thể dùng prompt template đơn giản:
  “Given the transcript, predicted emotion, confidence score, and acoustic cues, explain why this utterance may express the predicted emotion.”
* Output inference gồm:

  * predicted emotion
  * confidence/probability
  * explanation text

5. Training & evaluation

* Viết code train/evaluate rõ ràng.
* Metrics:

  * accuracy
  * macro F1
  * weighted F1
  * confusion matrix
* Lưu best checkpoint theo validation macro F1.
* Log train loss, validation loss, metrics theo epoch.
* Có config để chỉnh:

  * dataset path/name
  * encoder name
  * batch size
  * learning rate
  * freeze encoder
  * number of epochs
  * max audio length nếu cần

6. Code structure mong muốn

* dataset.py: load `AbstractTTS/IEMOCAP`, xử lý audio, transcript, label mapping 8→4, bỏ minor emotion/tie label.
* model.py: SSL encoder + pooling + classifier.
* features.py: extract acoustic features.
* train.py: training loop.
* evaluate.py: evaluation metrics.
* inference.py: predict emotion + generate explanation cho một audio.
* config.yaml hoặc argparse để cấu hình.

7. Mục tiêu hiện tại

* Tạo baseline chạy được, sạch, dễ debug, dễ mở rộng.
* Ưu tiên code đơn giản, comment rõ ràng.
* Chưa cần memory, self-evolving, agent framework, hay cơ chế tự cập nhật kinh nghiệm.
