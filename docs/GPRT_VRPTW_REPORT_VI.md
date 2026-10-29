# Báo cáo triển khai GPRT-VRPTW: pilot legacy và protocol feature-v2
> **Tài liệu lịch sử:** kết quả và các biến thể dưới đây thuộc pilot/protocol cũ. Runner hiện hành chỉ chạy `gp` hoặc GPRT đầy đủ; archive seeding là tùy chọn cấu hình của GPRT. Xem [hướng dẫn chạy hiện hành](RUN_GPRT_VRPTW.md).


## 1. Phạm vi và trạng thái

Project triển khai GPRT sinh biểu thức heuristic prefix trên 56 instance Solomon
VRPTW. Transformer sinh chương trình chấm điểm khách hàng, không sinh trực tiếp tuyến.
Một bộ dựng tuyến chung áp dụng biểu thức và giữ capacity, time window, service time,
depot horizon cùng fleet limit; solution checker sau đó tính lại nghiệm độc lập.

Split cố định gồm 32 train, 12 validation và 12 test, phân tầng theo sáu họ
C1/C2/R1/R2/RC1/RC2. Khoảng cách là Euclid không làm tròn. Test chỉ được dùng sau khi
validation khóa heuristic. Gehring–Homberger không tham gia thí nghiệm hiện tại.

Báo cáo cần phân biệt hai trạng thái:

| Trạng thái | Nội dung | Có kết quả chất lượng? |
|---|---|---|
| Pilot legacy ngày 2026-09-14 | 4 chu kỳ, mã và archive trước rà soát | Có; nằm trong `outputs/gprt_vrptw/reduced_main` |
| Protocol feature-v2 ngày 2026-09-15 | Mã đã sửa theo bản rà soát, tối thiểu 8 chu kỳ | Chưa có ma trận 5 seed; mới có test và smoke test |

Vì vậy, các bảng số liệu bên dưới chỉ mô tả pilot legacy. Chúng không phải bằng chứng
rằng feature-v2 cải thiện chất lượng nghiệm.

## 2. Cơ chế GPRT trong protocol feature-v2

Mỗi chu kỳ GPRT sinh 32 biểu thức từ Transformer, ghép với cây ngẫu nhiên để tạo quần
thể 64, tiến hóa GP trong 3 thế hệ và cập nhật neural. Chỉ biểu thức được lấy mẫu từ
policy Transformer hiện tại nhận policy gradient. Cá thể GP là dữ liệu off-policy và
chỉ tham gia nhánh elite imitation bằng teacher forcing.

Decoder causal mặc định có 2 layer, `d_model=64`, 4 attention head, FFN 128, dropout
0,1 và pre-layer norm. Biểu thức tối đa 63 token, depth 6. Grammar mask theo dõi số
argument slot, loại slot khi chạy ablation phân kiểu, giới hạn depth/token và chỉ cho
EOS khi cây hoàn chỉnh. Số học được bảo vệ và chặn miền.

Fitness được so sánh theo thứ tự từ điển, nhỏ hơn tốt hơn:

1. số instance bất khả thi;
2. tỷ lệ khách chưa phục vụ;
3. số xe trung bình;
4. khoảng cách chuẩn hóa trung bình;
5. độ phức tạp cây để phá hòa.

Reward REINFORCE feature-v2 chỉ xếp hạng các mẫu neural on-policy trong cùng chu kỳ;
quần thể GP không còn thay đổi reward của neural. Mặc định dùng
`advantage_mode=batch_standardize`, mô tả đúng việc trừ trung bình và chia độ lệch
chuẩn trong batch. `ewma_scale` vẫn tồn tại như một lựa chọn có kiểm thử, nhưng không
phải mặc định. Sampling và teacher forcing dùng cùng temperature.

Loss chuyển thể của project là:

`loss = policy_loss + alpha × elite_NLL - beta × entropy`,

với mặc định `alpha=0,1`, `beta=0,001`. Đây là lựa chọn tái triển khai của project,
không được trình bày như công thức hay hyperparameter nguyên bản đã xác minh từ bài báo.

## 3. Các cải thiện đã thực thi

### 3.1. Sàng lọc archive trên thước đo chung

Archive không còn giữ một fitness tốt nhất rồi so trực tiếp giữa các batch khác nhau.
Mỗi biểu thức lưu observation theo batch ID, tên instance, chu kỳ và fitness. Khi đến
validation, pipeline lấy quota ứng viên từ từng batch, chấm lại chúng trên một screening
set train cố định rồi mới lấy tối đa 16 ứng viên sang validation. Cơ chế này áp dụng cho
cả GPRT và GP baseline.

`candidate_archive.json` mới lưu lịch sử observation và screening fitness. Screening
set cũng được ghi vào checkpoint và `training_summary.json`. Checkpoint legacy không
resume được vào protocol feature-v2.

### 3.2. Phủ dữ liệu train và đường học

32 train instance được chia thành 8 batch, mỗi batch 4 instance. Thứ tự được cân bằng
theo family và seed. Trainer mặc định từ chối cấu hình có ít chu kỳ hơn số batch, nên
protocol mới bắt đầu ở 8 chu kỳ để phủ đủ train. Các mốc 8/16/32 tương ứng một, hai và
bốn lượt phủ dữ liệu; log ghi số instance đã thấy, coverage rate và gradient step.

Preset nằm tại:

- `configs/gprt_screening_8cycle.json`;
- `configs/gprt_learning_curve_16cycle.json`;
- `configs/gprt_main.json`.

File `configs/gprt_reduced_4cycle.json` chỉ còn vai trò mô tả protocol legacy.

### 3.3. Trao đổi giữa Transformer và GP

Elite không còn được học bắt chước chỉ vì tốt trên batch hiện hành. Tối đa 8 ứng viên
elite của batch được đánh giá lại trên screening set; quá trình chọn ưu tiên các
fingerprint tuyến khác nhau trước khi điền phần còn thiếu theo fitness. Log mới gồm:

- gradient norm của policy loss và `alpha × elite_NLL`;
- cosine giữa hai hướng gradient;
- tỷ lệ đồng hạng reward neural;
- chất lượng trung bình của mẫu neural;
- tỷ lệ mẫu neural tốt hơn median của cây ngẫu nhiên;
- tỷ lệ sống trực tiếp của mẫu neural và số mẫu neural trong elite cuối.

Runner có các phương pháp `gprt_alpha_001` và `gprt_alpha_005` để screen alpha 0,01 và
0,05 mà không đồng thời đổi feature hoặc grammar.

### 3.4. Đa dạng hành vi và grammar

`population_diversity` vẫn đo đa dạng chuỗi biểu thức, nhưng protocol mới còn đo
`behavior_diversity` bằng fingerprint tuyến trên probe train cố định. Đây chỉ là bằng
chứng hành vi khác nhau trên probe, không chứng minh hai cây tương đương toàn cục.

GP áp dụng một số phép rút gọn bảo toàn ngữ nghĩa, chẳng hạn `sub(x,x) → 0`,
`add(x,0) → x` và `min(x,x) → x`, để giảm cây phình và loại khác biệt cú pháp hiển
nhiên. `gprt_typed_grammar` là ablation trong đó toán tử số, so sánh, logic và
`if_else` chỉ được đặt vào slot đúng kiểu number/boolean. Grammar phân kiểu chưa được
bật mặc định trước khi có kết quả validation.

### 3.5. Feature nhìn trước và giới hạn engine

Feature-v2 có 16 feature: 13 feature cũ và ba feature dùng chung mới:

- tỷ lệ số xe còn lại;
- tỷ lệ khách còn lại có thể phục vụ ngay sau ứng viên;
- hạng khẩn cấp của ứng viên theo due date.

`gprt_no_lookahead_features` cho phép đo đóng góp của ba feature mới. Bộ dựng tuyến vẫn
nối khách vào cuối tuyến đang mở. Insertion/local search chưa được bật vì đây là thay
đổi engine cần thí nghiệm riêng sau khi đường học 8/16/32 ổn định; nếu triển khai phải
áp dụng cùng engine cho mọi phương pháp và báo cáo riêng trước/sau.

### 3.6. Kế toán ngân sách

Evaluator ghi cả số request và số cache miss thực thi theo bốn pha: train, screening,
validation và diagnostic. Do screening set có thể chứa nhiều instance hơn batch và
cache hit khác nhau giữa phương pháp, không dùng một mình `fitness_requests` để tuyên
bố ngân sách tính toán bằng nhau. Báo cáo chính phải kèm instance evaluations, thời
gian evaluator, thời gian neural/GP và chi phí áp dụng heuristic trên test.

## 4. Protocol và kết quả pilot legacy

Pilot cũ dùng 5 seed độc lập `20260914..20260918`, population 64, 32 mẫu Transformer,
3 thế hệ GP, batch 4 và 4 chu kỳ. Mỗi phương pháp học có 1.424 fitness request/seed theo
cách đếm cũ. Bốn chu kỳ chỉ dùng 16/32 instance cho fitness huấn luyện và chỉ tạo bốn
gradient step, nên đây là pilot end-to-end chứ không phải thí nghiệm kết luận.

Các số sau là trung bình ± độ lệch chuẩn qua năm training seed trên 12 test instance:

Hai ablation thành phần trong bảng chỉ là kết quả pilot legacy được lưu để đối chiếu;
chúng đã được gỡ khỏi runner hiện hành và không thuộc protocol chạy mới.

| Phương pháp | Khả thi | Xe | Khoảng cách | Khoảng cách chuẩn hóa |
|---|---:|---:|---:|---:|
| EDD | 100,0% ± 0,0% | 11,000 ± 0,000 | 2.407,4 ± 0,0 | 0,4370 ± 0,0000 |
| GP thuần | 100,0% ± 0,0% | 9,750 ± 0,748 | 1.725,0 ± 399,8 | 0,3135 ± 0,0724 |
| Transformer + RL | 100,0% ± 0,0% | 10,083 ± 0,970 | 1.874,6 ± 516,6 | 0,3408 ± 0,0940 |
| GPRT đầy đủ | 100,0% ± 0,0% | 9,967 ± 0,603 | 1.662,4 ± 233,2 | 0,3024 ± 0,0429 |
| GPRT không seeding | 100,0% ± 0,0% | 9,617 ± 0,774 | 1.618,9 ± 440,8 | 0,2942 ± 0,0798 |
| GPRT không elite imitation | 100,0% ± 0,0% | 9,883 ± 0,679 | 1.521,8 ± 155,1 | 0,2767 ± 0,0280 |

Theo mục tiêu ưu tiên số xe trước khoảng cách, GPRT đầy đủ không thắng GP, biến thể bỏ
seeding hoặc biến thể bỏ elite imitation trong trung bình pilot. So sánh ghép cặp GPRT
với từng đối chứng trên cùng seed và instance cho kết quả:

| Đối chứng | GPRT thắng | GPRT thua | Hòa |
|---|---:|---:|---:|
| GP | 29 | 31 | 0 |
| Bỏ elite imitation | 6 | 18 | 36 |
| Bỏ Transformer seeding | 17 | 31 | 12 |
| Transformer + RL | 28 | 20 | 12 |

Đây là thống kê mô tả trên các phép đo của năm training run, không phải 60 training run
độc lập. Kết quả chỉ cho thấy lợi ích của seeding và elite imitation chưa xuất hiện rõ
trong pilot ngắn; không chứng minh hai thành phần này gây hại nói chung.

Không có nghiệm test bất khả thi trong output legacy. Không dùng BKS vì chưa có nghiệm
tham chiếu đã xác minh cùng fleet, metric và rounding, nên không gọi chênh lệch là
optimality gap.

## 5. Kiểm chứng sau sửa đổi

- `python -m unittest discover -s tests -q`: 26 test qua, gồm kiểm tra archive nhiều
  batch, EWMA/batch advantage, temperature nhất quán, grammar phân kiểu, feature-v2,
  checkpoint/resume và các kiểm tra evaluator/solver trước đó.
- Smoke test dùng Transformer/GP nhỏ đã chạy đủ 8 chu kỳ trên đúng 32 train instance
  Solomon, đạt coverage 32/32 và hoàn tất validation/archive mới.
- GP baseline với cơ chế screening mới đã chạy smoke test riêng.
- Validator vẫn xác nhận output cũ gồm 6 phương pháp × 5 seed × 12 test = 360 dòng và
  nhận diện nó là `legacy_4cycle`.

Smoke test chỉ xác nhận pipeline vận hành; kích thước mô hình và ngân sách nhỏ nên fitness
của nó không được đưa vào bảng kết quả hoặc dùng để chọn cấu hình.

## 6. Kế hoạch thí nghiệm tiếp theo

1. Chạy ma trận lõi GP, Transformer+RL, GPRT và hai ablation ở 8 chu kỳ × 5 seed với
   `--skip-test`.
2. Dùng riêng train/validation để xem đường học và khóa alpha; không dùng test để tuning.
3. Nếu đường học còn cải thiện, chạy tiếp 16 rồi 32 chu kỳ với cùng split và seed.
4. Chạy typed grammar và look-ahead như các ablation riêng, không gộp với alpha screening.
5. Sau khi khóa cấu hình, đánh giá test một lần; báo cáo so sánh ghép cặp, effect size,
   khoảng tin cậy phản ánh cả seed và instance, cùng chi phí theo từng pha.
6. Chỉ sau đó mới thử insertion/local search hoặc OOD; mọi thay đổi engine phải dùng
   chung cho các phương pháp đối chứng.

Test legacy đã được xem để hình thành giả thuyết cải thiện. Vì vậy, chạy lại trên chính
12 test instance đó không còn là một xác nhận hoàn toàn chưa bị ảnh hưởng bởi lựa chọn
trước; cần diễn giải thận trọng hoặc thiết kế thêm đánh giá xác nhận nếu muốn kết luận
mạnh về tổng quát hóa.

## 7. Giới hạn

Protocol feature-v2 chưa có kết quả nhiều seed, nên chưa thể khẳng định archive mới,
alpha thấp hơn, grammar phân kiểu hay feature nhìn trước cải thiện GPRT. Solomon test là
held-out cùng benchmark, chưa chứng minh chuyển quy mô hoặc chuyển phân phối. Bộ dựng
tuyến vẫn là constructive append-only và giới hạn trần chất lượng heuristic.

GPRR/LSTM có thể được bổ sung qua interface generator chung làm đối chứng kiến trúc,
nhưng không thay thế mô hình cuối GPRT. Kết luận của project phải giới hạn ở cơ chế sinh
heuristic trên VRPTW; không suy ra hiệu quả vận hành cảng thật hoặc điều phối thời gian
thực từ các thí nghiệm này.


