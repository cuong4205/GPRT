# Báo cáo thực thi cải thiện GPRT–VRPTW
> **Tài liệu triển khai lịch sử:** ghi lại các sửa đổi và screening trước khi runner được rút gọn. Các ablation nêu trong đây không còn là chế độ chạy hiện hành. Xem [hướng dẫn chạy hiện hành](RUN_GPRT_VRPTW.md).


## Cập nhật 2026-09-16

Đã thực thi feature tính theo nhu cầu, bỏ tạo danh sách khách lặp trong solver và sửa typed grammar/rút gọn/mutation/crossover. Toàn bộ 35 tests qua; benchmark 4 biểu thức × 6 instance train xác nhận cùng nghiệm giữa dense và lazy. Xem [báo cáo kiểm chứng chi tiết](GPRT_LIGHTWEIGHT_IMPLEMENTATION_2026_09_16_VI.md). Đây là cải thiện chi phí evaluator và tính hợp lệ, chưa phải kết quả ma trận huấn luyện mới.

Ngày: 2026-09-15. Phạm vi: protocol feature-v2 sau rà soát pilot 4 chu kỳ.

## Các thay đổi đã thực thi

| Điểm | Thay đổi | Trạng thái |
|---|---|---|
| Archive khác batch | Lưu observation theo batch ID và danh sách instance; lấy quota mỗi batch, chấm lại trên screening train cố định rồi mới validation; áp dụng cả GPRT và GP | Đã cài và test |
| Phủ train/đường học | Trainer từ chối cấu hình chính không phủ đủ batch; batch cân bằng theo family và seed; log coverage/gradient step; thêm preset 8/16/32 | Đã cài và smoke test 8 chu kỳ |
| Reward/baseline/temperature | Reward rank chỉ dùng mẫu neural on-policy; mặc định khai báo rõ batch standardization; EWMA-scale còn là lựa chọn có test riêng; sampling và teacher forcing cùng temperature | Đã cài và test |
| Neural–GP | Elite được chấm lại trên screening set, ưu tiên khác hành vi; log gradient norm hai nhánh, cosine, tie rate, neural-vs-random, survival và elite contribution; thêm alpha 0,01/0,05 | Đã cài; chưa có kết quả main mới |
| Đa dạng/grammar | Log fingerprint tuyến trên probe cố định; rút gọn cây an toàn; thêm typed number/boolean grammar dưới dạng ablation | Đã cài và test |
| Giới hạn feature/engine | Thêm remaining vehicles, next-feasible fraction và urgency rank dùng chung; có ablation tắt feature-v2 | Đã cài; insertion/local search chưa bật |
| Tái sử dụng archive | Thêm `gprt_archive_seeding`: 4 elite đã qua screening chung thay 4 cây ngẫu nhiên, loại trùng mẫu neural và được chấm lại trên batch mới | Đã cài và test; chưa có kết luận chất lượng |

## Ngân sách và đầu ra

Evaluator ghi cả request và cache-miss instance evaluation theo bốn pha: `train`,
`screening`, `validation`, `diagnostic`. Vì chi phí mỗi pha và cache khác nhau, báo cáo
không còn dùng một mình `fitness_requests` để tuyên bố ngân sách bằng nhau.

`candidate_archive.json` mới chứa `observations`, `batch_id`, `instances`, fitness theo
batch và fitness trên screening set. `training_summary.json` chứa coverage count/rate
và số gradient step. Checkpoint cũ không resume được vào feature-v2.

## Kiểm chứng

- `python -m unittest discover -s tests -q`: 26 test qua.
- Smoke test với Transformer/GP nhỏ chạy 8 chu kỳ trên đúng 32 train instance Solomon:
  coverage cuối 32/32, validation và archive mới hoạt động.
- GP baseline mới đã chạy smoke test riêng.
- Validator vẫn xác nhận `outputs/gprt_vrptw/reduced_main` là protocol legacy hợp lệ.

Smoke test chỉ xác nhận pipeline vận hành, không phải kết quả so sánh chất lượng. Ma trận
5 seed ở 8/16/32 chu kỳ chưa được chạy, nên chưa được khẳng định các thay đổi làm GPRT
tốt hơn GP hoặc ablation.

## Bước chạy tiếp theo

Lệnh và thứ tự chạy nằm trong `docs/RUN_GPRT_VRPTW.md`. Trước tiên chạy ma trận lõi
8 chu kỳ trên train/validation; sau đó mới quyết định 16/32 và các ablation alpha,
typed grammar, feature-v2. Không dùng test để chọn cấu hình.

## Screening archive seeding ngày 2026-09-15

Đã chạy `gprt` và `gprt_archive_seeding` với cùng seed 20260914, 8 chu kỳ, đủ
32/32 train instance và không đánh giá test. Biến thể dùng 4 archive seed ở chu kỳ
5–8, thay 4 cây ngẫu nhiên; population vẫn 64 và train instance request bằng nhau.

| Chỉ số | GPRT mốc | Archive seeding |
|---|---:|---:|
| Validation mean vehicles | 8,5833 | 8,5000 |
| Validation normalized distance | 0,2710 | 0,2668 |
| Fitness requests | 3.200 | 3.184 |
| Unique instance evaluations | 8.638 | 8.507 |
| Train instance requests | 11.264 | 11.264 |
| Evaluator time (giây) | 1.268,7 | 1.589,4 |
| Tổng wall time từ learning curve (giây) | 1.276,4 | 1.598,5 |
| Thời gian đầu-cuối gồm checkpoint/output (giây) | 1.287,5 | 1.614,2 |
| Kích thước `last.pt` + `best.pt` (MB) | 32,1 | 30,8 |

Kết quả là tín hiệu dương rất nhỏ trên đúng một seed, chưa chứng minh cải thiện.
Runtime dao động mạnh: bốn chu kỳ đầu cho kết quả thuật toán giống hệt nhưng thời
gian chu kỳ 4 là 174 giây ở mốc và 274 giây ở biến thể. Vì archive dùng ít hơn 131
unique instance evaluation, chưa có bằng chứng bản thân cơ chế mới gây overhead 25%.
Chưa đo peak memory ở lượt đã hoàn thành. Runner mới đã bổ sung sampler RSS bằng
thư viện chuẩn và sẽ ghi `peak_rss_mb` cùng `end_to_end_seconds` ở lượt sau. Vì vậy
chưa chạy seed thứ hai hoặc tăng lên 16/32 chu kỳ.


