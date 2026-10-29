# Kết quả pilot và đánh giá Solomon VRPTW động

Ngày chạy: 27-10-2026. Kịch bản release là synthetic; báo cáo này đo khả năng giữ
chất lượng zero-shot của heuristic khi thông tin đơn hàng đến dần, không đại diện cho
dữ liệu vận hành thực.

## Protocol đã chạy

- `dvrpsim==1.0.0`; Solomon Euclid không làm tròn; thời gian di chuyển bằng khoảng
  cách. Giữ nguyên fleet, capacity, ready time, due time và service time trong file.
- GPRT và GP-only dùng biểu thức `final_heuristic.txt` được chọn từ validation của từng
  training seed: `20260914`–`20260918`. EDD chọn due date sớm nhất.
- Pilot: 4 validation instance (`C102`, `C206`, `R104`, `RC203`) × 2 release seed, ba
  policy và static control; 132 episode.
- Test cuối: 12 instance held-out × 3 release seed (`20261025`–`20261027`), 5 seed
  huấn luyện cho GPRT và GP, EDD và static control; 528 episode (396 dynamic, 132
  static). Không chọn lại biểu thức sau khi xem test.
- Due time Solomon là latest service start. Vì vậy release được chặn theo
  `min(u, due - depot_distance, depot_horizon - depot_distance - service - return_distance)`.
  Giờ phục vụ, giờ quay về và capacity được kiểm tra độc lập sau mô phỏng.

## Kết quả trung bình trên 12 test instance

Các tỷ lệ là trung bình theo episode; khoảng cách là tổng quãng đường trung bình trên
một episode. `p95 callback` là trung bình p95 độ trễ callback trong từng episode.

| Kịch bản | Policy | Episode | Phục vụ đúng hạn | Quãng đường | Quãng đường/đơn phục vụ | Xe dùng | p95 callback |
|---|---|---:|---:|---:|---:|---:|---:|
| Tĩnh | GPRT | 60 | 100.00% | 1,662.41 | 16.62 | 9.97 | 67.96 ms |
| Tĩnh | GP-only | 60 | 100.00% | 1,725.04 | 17.25 | 9.75 | 88.18 ms |
| Tĩnh | EDD | 12 | 100.00% | 2,407.35 | 24.07 | 11.00 | 28.87 ms |
| Động | GPRT | 180 | 78.88% | 3,829.30 | 48.12 | 25.00 | 0.94 ms |
| Động | GP-only | 180 | 79.07% | 3,871.16 | 48.53 | 25.00 | 0.97 ms |
| Động | EDD | 36 | 79.81% | 3,943.11 | 49.09 | 25.00 | 0.92 ms |

Không có khách được phục vụ trễ trong các episode cuối; tỷ lệ đúng hạn giảm vì đơn
không phục vụ trước khi mô phỏng kết thúc. Tất cả policy đều dùng đủ 25 xe trong trung
bình kịch bản động.

## Chênh lệch ghép cặp động

So sánh ghép cặp theo cùng instance, release seed và training seed; 180 cặp cho mỗi
đối chứng. Chênh lệch là GPRT trừ baseline. Với GP, hai model seed được ghép tương ứng.
Với EDD, 36 episode EDD được ghép lặp lại theo từng GPRT seed; đây là kiểm soát cùng
kịch bản, không phải 180 quan sát baseline độc lập.

| So sánh | Đúng hạn | Tổng quãng đường | Quãng đường/đơn phục vụ |
|---|---:|---:|---:|
| GPRT − GP-only | −0.19 điểm % | −41.85 | −0.41 |
| GPRT − EDD | −0.93 điểm % | −113.81 | −0.97 |

GPRT dùng quãng đường thấp hơn một chút, nhưng tỷ lệ phục vụ cũng thấp hơn một chút.
Vì tỷ lệ phục vụ là tiêu chí ưu tiên và quãng đường thấp có thể đến từ việc bỏ sót đơn,
kết quả này **chưa cho thấy GPRT tốt hơn GP hoặc EDD trong kịch bản động**. Trên static,
GPRT vẫn phục vụ đủ và có tổng quãng đường trung bình thấp hơn GP-only; kết quả đó không
chứng minh được lợi thế tương tự khi release bất đồng bộ.

Theo họ instance, tỷ lệ đúng hạn động trung bình của GPRT là 70.63% trên C1, 85.03%
trên C2, 72.83% trên R1, 93.97% trên R2, 57.50% trên RC1 và 93.30% trên RC2. Khác biệt
này cho thấy khó khăn tập trung ở họ có cửa sổ chặt và phân bố ngẫu nhiên/cụm hỗn hợp.

## Kiểm tra và diễn giải

- 132/132 pilot và 528/528 test episode qua checker độc lập: không có vi phạm capacity,
  ready/due time, giờ quay về depot, gán trùng/thiếu giữa route và sự kiện giao, sai lệch
  quãng đường, hoặc đơn chưa release lọt vào routing callback.
- Checker được thử cố ý trên route vi phạm capacity và service start quá due; cả hai lỗi
  đều được phát hiện. Cả năm kiểm thử adapter đều đạt.
- Độ trễ callback động quanh 1 ms ở môi trường chạy này. Độ trễ static cao hơn vì một
  callback xây route cùng lúc cho tập khách hàng đầy đủ; đây là số đo triển khai cụ thể,
  không phải giới hạn thời gian thực của dvrpsim.
- Phép thử chỉ mô phỏng release bất đồng bộ và xe về depot. Nó không có route insertion,
  hủy đơn, traffic biến thiên hoặc travel time bất định. Các chênh lệch ghép cặp mang
  tính mô tả; 12 instance và 5 training seed chưa đủ để khẳng định ưu thế thống kê.

## Artifacts để tái lập

- Cấu hình: [`configs/dvrpsim_dynamic.json`](../configs/dvrpsim_dynamic.json)
- Runner: [`run_dvrpsim_dynamic.py`](../run_dvrpsim_dynamic.py)
- Adapter/simulator: [`dvrpsim_dynamic/simulator.py`](../dvrpsim_dynamic/simulator.py)
- Pilot: `outputs/dvrpsim_dynamic/pilot/`
- Test cuối: `outputs/dvrpsim_dynamic/final/`
- Tổng hợp: `summary_by_policy.csv`, `paired_comparisons.csv`, `gate_report.json`;
  manifest episode ghi biểu thức/hash và release time từng khách; event log ghi thời
  điểm release, quyết định route, service start, giao hàng và từng leg di chuyển.

Simulator và các callback/release semantics được mô tả trong [repository dvrpsim](https://github.com/sztaki-hu/dvrpsim), [tutorial release order](https://sztaki-hu.github.io/dvrpsim/src/tutorial/tutorial_02.html), [tutorial routing callback](https://sztaki-hu.github.io/dvrpsim/src/tutorial/tutorial_03b.html), và [order service/time-window fields](https://sztaki-hu.github.io/dvrpsim/src/tutorial/tutorial_04.html). Nếu dùng trong khóa luận, trích dẫn bài framework Horváth & Tamási (2025) như ghi trong repository.
