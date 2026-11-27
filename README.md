# GPRT cho VRPTW và DVRPTW xe tải–drone

Dự án nghiên cứu cách sinh **biểu thức heuristic** bằng Genetic Programming (GP) và GPRT: Transformer sinh cây theo chuỗi token, GP tiến hóa cây, còn REINFORCE và elite imitation cập nhật Transformer. Bộ dựng tuyến dùng biểu thức đã chọn để đưa ra quyết định; Transformer không trực tiếp sinh tuyến. Đây là bản chuyển thể cho bài toán định tuyến, không khẳng định tái hiện nguyên trạng mọi chi tiết của bài báo.

Có ba luồng triển khai:

- **Solomon VRPTW tĩnh:** huấn luyện GP/GPRT và chọn heuristic bằng validation.
- **Solomon động:** áp dụng zero-shot heuristic đã học vào kịch bản DVRPSim với đơn xuất hiện dần và có thể hủy.
- **DVRPTW xe tải–drone:** huấn luyện và đánh giá trực tiếp bằng episode DVRPSim trên bộ CSV riêng.

Đợt so sánh năm seed thử cách cho Transformer giữ nguyên một luật tốt rồi học đặc trưng cần cộng hoặc trừ. Trên test Solomon, cấu hình này giảm 6,52% quãng đường và 0,90% số xe so với GP cùng ngân sách chấm train. [Kết quả trên Solomon và xe tải–drone](docs/GPRT_ADDITIVE_QUALITY_VI.md) · [Lệnh tái lập](docs/RUN_GPRT_VRPTW.md#so-sánh-hai-cách-sinh-biểu-thức)

[Tài liệu mô hình](docs/MODEL.md) giải thích thuật toán và các quy ước. [Hướng dẫn chạy](docs/RUN_GPRT_VRPTW.md) bao gồm cả ba luồng. [Xe tải–drone](docs/DVRPSIM_DRONE_VI.md) trình bày chi tiết điều phối và kết quả có sẵn. [Báo cáo 32 cycle](docs/END_TO_END_32CYCLE_RESULTS_VI.md) là phân tích một đợt Solomon trước đây; artifact gốc của đợt đó hiện không có trong workspace. [Ghi chú cá nhân](docs/NOTE.md) được giữ riêng.

## Bắt đầu

Tại thư mục gốc dự án, trên Windows PowerShell:

```powershell
python -m venv .venv
.venv/Scripts/python.exe -m pip install -r requirements-dev.txt
.venv/Scripts/python.exe -m pytest tests -q
.venv/Scripts/python.exe run_dvrpsim_drone.py inspect --fleet 10 --fold 0
```

Có thể cài `requirements-fast.txt` để dùng backend Numba cho bộ dựng tuyến Solomon. Dataset CSV xe tải–drone nằm ở `data/Dataset_DVRPTW_drone_GP_RNN`. Dataset Solomon **hiện chưa có tại** `data/data/Solomon` trong workspace; cần đặt đủ 56 instance ở đó hoặc chỉ đường dẫn khác bằng `--data-dir` trước khi chạy luồng Solomon, kể cả `--dry-run` của end-to-end.

## Bản đồ code

| Đường dẫn | Vai trò |
|---|---|
| `gprt_vrptw/` | Grammar, GP, Transformer, trainer, chọn heuristic và evaluator Solomon |
| `vrptw_baselines/` | Đọc Solomon, dựng tuyến và kiểm tra nghiệm |
| `dvrpsim_dynamic/` | Sinh kịch bản và mô phỏng Solomon động |
| `dvrpsim_drone/` | Đọc CSV, lập lịch và đánh giá episode xe tải–drone |
| `configs/` | Preset cho các luồng; tham số mặc định còn lại nằm trong `gprt_vrptw/config.py` |
| `tests/` | Kiểm thử các thành phần và runner |
| `outputs/` | Artifact thí nghiệm cục bộ, không được Git theo dõi |
