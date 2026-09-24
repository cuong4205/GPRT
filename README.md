# GPRT-VRPTW và baseline bằng Python

Project hiện có pipeline GPRT hoàn chỉnh cho Solomon VRPTW bên cạnh baseline cũ. GPRT
dùng Transformer causal sinh cây heuristic prefix, GP tinh chỉnh cây, REINFORCE chỉ học
từ mẫu neural on-policy và elite imitation học từ GP bằng teacher forcing.

Protocol feature-v2 sàng lọc archive trên tập train chung, bắt buộc phủ đủ 8 batch,
đo đa dạng hành vi và gradient từng nhánh, đồng thời thêm ba feature nhìn trước dùng
chung. Kết quả `reduced_main` là pilot 4 chu kỳ lịch sử, không phải kết quả sau sửa.

Xem [hướng dẫn chạy](docs/RUN_GPRT_VRPTW.md),
[tài liệu kỹ thuật diễn giải toàn bộ project](docs/GPRT_PROJECT_TECHNICAL_DOCUMENTATION_VI.md),
[báo cáo thực thi cải thiện](docs/GPRT_VRPTW_IMPROVEMENT_IMPLEMENTATION_VI.md) và
[báo cáo pilot tiếng Việt](docs/GPRT_VRPTW_REPORT_VI.md). Kết quả legacy nằm tại
`outputs/gprt_vrptw/reduced_main`.

[Rà soát chất lượng ngày 17/09](docs/GPRT_DEEP_QUALITY_AUDIT_2026_09_17_VI.md)
đã chạy sáu lượt screening (baseline, routing-rank, elite3 × hai seed), chỉ dùng
train/validation. Chưa có biến thể đạt tiêu chí cải thiện ổn định; giữ nguyên mặc định.

```powershell
python -m unittest discover -s tests -v
python validate_gprt_results.py --results-dir outputs/gprt_vrptw/reduced_main
```

## Baseline trước đó

Project triển khai bốn heuristic xây dựng và một baseline Genetic Programming (GP)
trên riêng bộ dữ liệu Solomon có sẵn trong `data/data/Solomon`.

## Phương pháp

- **FIFO:** chọn khách hàng khả thi có ID nhỏ nhất.
- **Random:** chọn ngẫu nhiên trong tập khách hàng khả thi.
- **Greedy:** chọn khách hàng khả thi gần vị trí hiện tại nhất.
- **EDD:** chọn khách hàng khả thi có hạn phục vụ (`due date`) sớm nhất.
- **GP:** tiến hóa biểu thức ưu tiên từ 11 feature đã chuẩn hóa. GP dùng cùng bộ
  dựng tuyến, kiểm tra tải trọng, cửa sổ thời gian, thời gian phục vụ và giới hạn xe.

Mục tiêu được so sánh theo thứ tự: khả thi → ít xe → tổng quãng đường Euclid.
Không dùng BKS vì thư mục dữ liệu không kèm nghiệm tham chiếu; do đó báo cáo không
gọi chênh lệch quan sát được là optimality gap.

## Chạy lại

```powershell
python -m unittest discover -s tests -v
python run_experiment.py --runs 5 --output outputs/vrptw_baselines/results.json
python validate_results.py --results outputs/vrptw_baselines/results.json
```

Seed mặc định là `20260914..20260918`. Toàn bộ 56 file Solomon được parse và kiểm kê;
dữ liệu Gehring-Homberger không tham gia thí nghiệm. Bảng tổng hợp chính chỉ dùng test split
cố định theo từng họ C1/C2/R1/R2/RC1/RC2 để tránh báo cáo GP trên dữ liệu đã dùng
cho huấn luyện hoặc chọn biểu thức.

Ngân sách GP mặc định là population 10 × 4 thế hệ trên 6 instance train (một
instance từ mỗi họ C1/C2/R1/R2/RC1/RC2 theo round-robin), phù hợp
để chạy baseline 5 seed trên laptop. Các tham số có thể thay đổi bằng
`--gp-population`, `--gp-generations`, `--gp-max-depth` và `--gp-train-limit`. Giữ nguyên cấu hình/seed khi so sánh với
GPRR hoặc GPRT trong các bước tiếp theo của khóa luận.
