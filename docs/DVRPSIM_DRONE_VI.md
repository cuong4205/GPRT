# Huấn luyện GP và GPRT trên DVRPSim xe tải–drone

Mỗi episode chạy một CSV với một biểu thức cố định. DVRPSim quản lý đồng hồ và sự
kiện đơn hàng. Bộ điều phối của project lập lịch xe tải–drone và phát lại các sự
kiện xuất phát, phóng drone, giao, thu hồi và về depot trên cùng đồng hồ. Fitness
dùng kết quả sau episode, theo thứ tự ưu tiên tỷ lệ giao đúng hạn, quãng đường có
trọng số trên mỗi đơn, số xe dùng và độ dài biểu thức. GP và GPRT dùng cùng số lượt
chấm train. Log tách chấm train, sàng lọc, validation, chẩn đoán và lượt mô phỏng
thực chạy sau cache.

## Chạy

```powershell
.venv/Scripts/python.exe run_dvrpsim_drone.py inspect --fleet 10 --fold 0
.venv/Scripts/python.exe run_dvrpsim_drone.py train --fleet 10 --fold 0 --config configs/dvrpsim_drone_quality.json --output outputs/dvrpsim_drone/quality_replay
.venv/Scripts/python.exe run_dvrpsim_drone.py test --fleet 10 --fold 0 --output outputs/dvrpsim_drone/quality_replay
.venv/Scripts/python.exe run_dvrpsim_drone.py test --fleet 10 --fold 0 --output outputs/dvrpsim_drone/quality_replay --transfer
```

Dùng `--seeds` để chọn seed train và nạp đúng các seed khi test. Dùng output riêng
cho mỗi fold hoặc cấu hình. Bốn fold lần lượt giữ riêng bốn nhóm tọa độ của bộ 100
khách để test, dùng nhóm kế tiếp để validation. Bộ 200/400 được đánh giá riêng
với biểu thức đã khóa. Test kiểm tra hash dữ liệu, split và quy tắc điều phối
trước khi nạp token trong `policy.json`. Báo cáo ghi số đơn giao đúng hạn,
đơn còn lại, km từng phương thức, số lượt drone, số lần đổi gán và độ trễ p95.
Baseline duyệt đơn theo hạn giao giống hai phương pháp học. Greedy ưu tiên quãng
đường tăng thêm và EDD ưu tiên phương án có ít thời gian dư nhất đến hạn của đơn.

Để tiếp tục GPRT sau khi gián đoạn, thêm `--methods gprt --resume` vào lệnh
train và chỉ định đúng thư mục output, seed và cấu hình của lần chạy đó.

## Quy tắc điều phối và giả định pilot

- Đọc cột `time` trực tiếp. Đơn có cùng thời điểm được công bố trước khi chấm
  phương án, kể cả trùng mốc xuất phát. Một đơn chỉ được đưa lên xe tại depot.
- Xét lại các chuyến dự kiến mỗi khi có đơn mới, đến mốc xuất phát hoặc có xe về
  depot. Có thể đổi xe, thứ tự và phương thức giao trước khi xuất phát. Từ lúc
  rời depot đến khi quay lại, toàn bộ chuyến đã chốt. Đơn rút khỏi lịch dự kiến
  trở lại hàng chờ và được xét trong lần điều phối sau.
- Thời gian gom đơn mặc định là 5 phút sau lần gán đầu. Nếu khách đầu chuyến
  chưa mở cửa, xe có thể chờ thêm tại depot đến mốc khởi hành theo giờ mở cửa.
  Mỗi lần đổi lịch, bộ điều phối tính lại mốc này và kiểm tra toàn bộ lịch tại
  mốc xuất phát. Xe xuất phát sớm nếu chờ tới mốc đó làm lịch mất khả thi.
  Dùng `--depart-after-hold` để đo đối chứng chỉ chờ theo thời gian gom đơn.
- Số xe là tham số bắt buộc vì không có trong CSV. Pilot chọn 10 xe/instance.
  Mỗi xe có một drone, mỗi sortie giao một đơn, phóng và thu tại hai đầu của
  một chặng truck, kể cả chặng depot–depot. Một drone mỗi xe cũng là cách
  thiết lập trong [mã tham chiếu](https://raw.githubusercontent.com/friedeggs01/DVRPTW-drone/main/graph/network.py).
  Các sortie thực hiện tuần tự. Khi hai phương tiện đến lệch giờ, phải đợi nhau trước khi thu hồi.
- Dùng khoảng cách Euclid theo km, tốc độ và thời gian phục vụ theo cấu hình,
  kiểm tra **hoàn tất** phục vụ trước hạn đóng. Endurance gồm phóng, bay, phục
  vụ, chờ và thu hồi. Trường ServiceWindow được lưu từ cấu hình. Lịch lấy cửa
  sổ từng khách trực tiếp từ open/close.
- Mỗi đơn được xét theo hạn giao tăng dần. Bộ dựng xét ba khoảng chèn cuối của
  mỗi xe và loại lịch vi phạm trước khi gọi biểu thức. Fitness dùng km truck
  cộng 0,25 lần km drone theo mặc định. Đây là quy ước pilot, không phải phép
  đo phát thải. Các giả định này được lưu cùng manifest và áp dụng cho cả
  train, validation, test và baseline.


## Cấu hình cải thiện heuristic

Runner động mặc định dùng `dvrpsim_drone_quality.json` với quần thể 16,
8 mẫu Transformer, 2 thế hệ GP mỗi chu kỳ, 16 chu kỳ, batch 4 và cây tối đa
31 token, sâu 4. Mỗi 4 chu kỳ, sàng lọc trên 8 instance train và chấm tối đa
8 ứng viên trên validation. Việc chia batch và chọn tập sàng lọc sử dụng họ
C1/C2/R1/R2/RC1/RC2 lấy từ tên CSV có tiền tố quy mô.

GP và GPRT khởi tạo bằng `trip_duration`, `extra_duration`, `extra_cost` và
`service_delay` trong quần thể đã ấn định. Mỗi phương pháp có thể đưa hai elite
đã sàng lọc trên train sang batch mới. GPRT tính policy gradient từ mẫu do
Transformer sinh và học mô phỏng biểu thức elite qua một thành phần loss riêng.

Biểu thức có thể dùng thời gian và chi phí tăng thêm của chuyến, tổng thời
gian chờ phục vụ, biên thời gian nhỏ nhất trên lịch và thời gian đến khi giao
xong đơn mới. Các đặc trưng thời gian mới chuẩn hóa theo thời gian phục vụ
của xe tải. Chi phí tăng thêm chuẩn hóa theo quãng đường xe tải đi được trong
cùng khoảng thời gian.

Manifest ghi phiên bản simulator, vocabulary và cách chia batch. Checkpoint
kiểm tra quy tắc evaluator và các biểu thức khởi tạo khi tiếp tục huấn luyện.
Dùng thư mục output riêng cho phiên bản này. Kết quả test kèm luật
`trip_duration` trên cùng simulator để đối chiếu chất lượng biểu thức.

## Kết quả cải thiện trên fold 0

GP và GPRT hoàn thành 16 chu kỳ cho từng seed 42 và 43. Mỗi lượt đi qua đủ 31
instance train với 4.464 lượt chấm biểu thức–instance, so với 1.092 lượt ở
pilot cũ. Validation dùng 16 instance để chọn biểu thức. Tập test 100 khách
có 9 instance. Đánh giá chuyển quy mô dùng seed 42 với 60 instance mỗi cỡ.
Các phương pháp dùng đội 10 xe trên từng instance.

| Phương pháp | Seed | Biểu thức | Đúng hạn validation | Đúng hạn test 100 |
|---|---:|---|---:|---:|
| GP | 42 | `extra_cost - min_route_slack` | 82,88% | 98,22% |
| GP | 43 | `trip_duration + extra_truck_km` | 81,88% | 96,89% |
| GPRT | 42 và 43 | `extra_truck_km` | 81,13% | 97,89% |

| Tập đánh giá | Pilot cũ GP/GPRT | `trip_duration` trên simulator mới | Greedy mới | EDD mới | GP mới, seed 42 | GPRT mới, seed 42 |
|---|---:|---:|---:|---:|---:|---:|
| 100 khách | 94,67% | 97,78% | 98,78% | 99,89% | 98,22% | 97,89% |
| 200 khách | 78,83% | 78,60% | 80,43% | 76,35% | 78,84% | 80,75% |
| 400 khách | 55,82% | 55,75% | 55,66% | 50,25% | 56,33% | 56,13% |

So với pilot cũ ở 400 khách, GP giao thêm 123 đơn đúng hạn và giảm 7,70%
quãng đường có trọng số. GPRT giao thêm 75 đơn với quãng đường có trọng số
tăng 5,17%. Ở 200 khách, GPRT giao thêm 230 đơn và giảm 4,14% quãng đường
có trọng số. Ở 100 khách, EDD có số đơn đúng hạn cao nhất trên simulator mới,
với tổng quãng đường có trọng số khoảng 29.243 km so với 21.054 km của GPRT.
Kết quả mô tả một fold. Hai seed GPRT chọn cùng biểu thức và GP đạt validation
cao hơn ở cả hai seed.

[Test seed 42](../outputs/dvrpsim_drone/quality_v2/family_fixed_seed42/test_results.json) ·
[Test seed 43](../outputs/dvrpsim_drone/quality_v2/family_fixed_seed43/test_results.json) ·
[Chuyển quy mô seed 42](../outputs/dvrpsim_drone/quality_v2/family_fixed_seed42/transfer_results.json)
