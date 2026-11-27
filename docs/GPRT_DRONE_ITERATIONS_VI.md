# Cải thiện GPRT trên bài xe tải–drone

Thí nghiệm ngày 24/11/2026 dùng dữ liệu CSV của dự án và DVRPSim. Mỗi episode có đội 10 xe, thời gian gom đơn 5 phút và chi phí `km xe tải + 0,25 × km drone`. Chất lượng ưu tiên số đơn giao đúng hạn, sau đó mới đến chi phí quãng đường. GPRT sinh biểu thức prefix bằng Transformer, tiến hóa bằng GP và cập nhật bằng REINFORCE cùng elite imitation.

## Điểm nghẽn đã kiểm tra

- **Fitness phân hạng sai khi hòa về phục vụ.** Cộng `1 - delivered / customer_count` bằng số thực có thể tạo hai tỷ lệ khác nhau cho cùng số đơn chưa giao. Trên batch H100R109, H100R111, H100R204 và H100R211, hai luật cùng bỏ 54/400 đơn. Fitness cũ ưu tiên cây dài hơn dù quãng đường chuẩn hóa lớn hơn, do `0.13499999999999998` được xếp trước `0.135`. Tính tỷ lệ bằng phân số chính xác trước khi chuyển sang số thực đưa thứ tự trở về đúng mục tiêu.
- **Các quyết định điều phối giới hạn phạm vi của biểu thức.** Bộ mô phỏng duyệt đơn theo hạn giao, chấm vị trí chèn khả thi trên xe đang nghỉ, rồi giữ nguyên chuyến sau khi xuất phát. Trên validation ban đầu, toàn bộ 302 đơn GPRT bỏ sót nằm ở RC1. Với lịch chạy đã xảy ra, 265 đơn từng có thời điểm một xe rảnh và chuyến riêng khả thi. Đây là dấu hiệu về cơ hội điều phối, vì những chuyến riêng đó sẽ cạnh tranh xe và thay đổi lịch nếu thực hiện. Tăng đội từ 10 lên 12 xe cũng làm GPRT tăng 71 đơn trên cùng validation, với chi phí quãng đường tăng.
- **Lệch phân phối và ngân sách tìm kiếm.** Fold 0 có 31 instance train thuộc C2/R1/R2, 16 validation thuộc RC1/RC2 và 9 test thuộc C1. Hai seed GPRT cũ cùng chọn `extra_truck_km`. Chấm lại tất cả biểu thức từng qua sàng lọc trong hai kho ứng viên cũ, lần lượt 20 và 12 biểu thức, vẫn chọn luật này. GPRT cũ chạy khoảng 21% ít episode train mới hơn GP sau cache dù bằng số yêu cầu chấm, đồng thời chạy thêm episode sàng lọc và chẩn đoán. Điều này giới hạn kết luận về đóng góp của Transformer.

Ở đợt Solomon 32 cycle đã lưu, GPRT giảm khoảng cách 2,51% nhưng tăng số xe 1,95% so với GP. Độ đa dạng biểu thức Transformer giảm rõ qua huấn luyện. Solomon tĩnh và Solomon động dùng mục tiêu và cơ chế triển khai riêng. [Kết quả Solomon](END_TO_END_32CYCLE_RESULTS_VI.md)

## Kết quả chạy lại trên validation

Mỗi cột có 1.600 đơn trên cùng 16 instance. Hai vòng đầu dùng lại các biểu thức đã khóa để đo từng chỉnh sửa mà không đưa thêm biến thiên do huấn luyện.

| Vòng | Thay đổi | GP 42 | GPRT 42 | GP 43 | GPRT 43 |
|---|---|---:|---:|---:|---:|
| 1 | Tính fitness chính xác | 1.326 | 1.298 | 1.310 | 1.298 |
| 2 | Thử các vị trí chèn sớm hơn khi toàn bộ vị trí cuối đều thất bại | 1.338 | 1.310 | 1.322 | 1.310 |
| 3 | Sàng lọc luật khởi tạo trên train và cho Transformer học trước từ các luật đã chọn | 1.333 | 1.333 | 1.338 | 1.338 |
| 4 | Dùng tiền tố lấy từ biểu thức đã sàng lọc trên train để Transformer hoàn thiện cây | 1.333 | 1.333 | 1.338 | 1.345 |

- Vòng 1 xác nhận sửa thứ hạng fitness. Số đơn của biểu thức cố định giữ nguyên vì thứ tự fitness chỉ được dùng khi tìm và chọn biểu thức.
- Vòng 2 tăng 12 đơn cho cả hai seed của mỗi phương pháp. GP seed 42 giảm khoảng cách có trọng số từ 30.466,81 xuống 30.229,09. GPRT tăng từ 31.111,73 lên 31.177,82. Cả hai phương pháp dùng chung cơ chế chèn.
- Vòng 3 sàng lọc 15 luật trên sáu instance train, đưa bốn luật tốt nhất vào quần thể của cả GP và GPRT, rồi học mô phỏng 16 bước trước khi GPRT bắt đầu vòng GP–RL. Cả hai dùng 8 chu kỳ, quần thể 12 và 1.302 yêu cầu chấm train mỗi seed. Seed 42 chọn `extra_duration − min_route_slack` ở cả hai phương pháp. Seed 43 chọn `extra_cost − min_route_slack` ở cả hai. Vòng 3 dùng cấu hình tìm kiếm khác preset lịch sử nên mức tăng so với vòng 2 phản ánh toàn bộ thay đổi.
- Vòng 4 giữ nguyên GP và ngân sách vòng 3. Ba trong sáu mẫu Transformer được sinh từ tiền tố lấy ở các luật khởi tạo và kho cây đã chấm trên train. Ba mẫu còn lại được sinh tự do. REINFORCE chỉ dùng log-xác suất các token do Transformer chọn. Học mô phỏng elite GP vẫn là một thành phần loss riêng. GPRT hòa seed 42, tăng 7 đơn ở seed 43 và tăng 4,82% khoảng cách có trọng số khi lấy tổng qua hai seed.

Mô phỏng các tuyến động cần cân bằng thời gian còn dư với việc giữ xe sẵn sàng cho đơn mới. Dùng nhiều vị trí chèn như đối chứng cũng phù hợp với nghiên cứu VRPTW. Phần học trước và sinh có hướng dẫn được thử trên bài toán này theo nghiên cứu về neural-guided GP cho tìm kiếm biểu thức. [Solomon (1987)](https://doi.org/10.1287/opre.35.2.254) · [Bent và Van Hentenryck (2004)](https://doi.org/10.1287/opre.1040.0124) · [Mundhenk và cộng sự (2021)](https://proceedings.neurips.cc/paper/2021/hash/d073bb8d0c47f317dd39de9c9f004e9d-Abstract.html)

## Test giữ lại trên fold 0

Khóa biểu thức bằng validation trước khi chạy 9 instance C1 giữ lại. Cùng một nhóm C1 đã được dùng trong báo cáo trước đây của dự án.

| Phương pháp | Seed | Đơn đúng hạn / 900 | Quãng đường có trọng số |
|---|---:|---:|---:|
| GP | 42 | 896 | 25.244,51 |
| GPRT | 42 | 896 | 25.244,51 |
| GP | 43 | 884 | 21.135,31 |
| GPRT | 43 | 897 | 25.448,20 |
| EDD | — | 899 | 29.242,65 |
| Greedy | — | 889 | 21.516,77 |

Trên hai seed, GPRT tăng từ 1.780 lên 1.793 đơn đúng hạn trong tổng số 1.800, tương đương +0,72 điểm phần trăm. Tổng quãng đường có trọng số tăng 9,30%. GPRT seed 43 chọn `extra_duration − min_route_slack − is_new_trip`. Hạng tử cuối hạ điểm khi mở chuyến mới, cho phép thuật toán ưu tiên xe chưa có đơn trong lịch dự kiến. So trực tiếp với cùng biểu thức bỏ hạng tử này, thêm hạng tử làm tăng 12 đơn và giảm 154,05 đơn vị chi phí trên validation, tăng 1 đơn và tăng 203,70 đơn vị chi phí trên test C1. Đây là đối chiếu biểu thức đã chạy với bộ điều phối cố định.

## Đổi họ instance và tăng quy mô

Ở fold 2, huấn luyện lại trên 25 instance C1/RC1/RC2, chọn biểu thức trên 8 instance C2 và test trên 23 instance R1/R2. Hai phương pháp giao đủ 800/800 đơn trên validation ở mỗi seed. Trên test, seed 42 của GP giao 1.922/2.300, còn ba lượt GP seed 43 và hai seed GPRT cùng giao 1.909/2.300. Tính chung hai seed, GPRT kém 13 đơn và tăng 1,18% quãng đường có trọng số. Hai fold dùng các nhóm tọa độ khác nhau để test, với các nhóm train có giao nhau.

Trên 60 instance 200 khách, áp dụng biểu thức đã khóa ở fold 0. GP giao 19.598/24.000, GPRT giao 20.147/24.000. Mức tăng 549 đơn đến từ seed 43. Ở seed đó, GPRT thắng 46/60 cặp theo thứ tự số đơn đúng hạn trước, quãng đường sau. Tổng quãng đường có trọng số qua hai seed tăng 8,76%.

Trên 60 instance 400 khách, GP giao 28.506/48.000, GPRT giao 29.221/48.000. GPRT tăng 715 đơn, tương đương +1,49 điểm phần trăm, và tổng quãng đường có trọng số tăng 8,72%. Seed 43 thắng 44/60 cặp theo thứ tự mục tiêu. Seed 42 hòa toàn bộ ở cả hai quy mô vì GP và GPRT chọn cùng một biểu thức.

| Tập đánh giá | Instance | Đúng hạn GP | Đúng hạn GPRT | Chênh lệch đúng hạn | Chênh lệch quãng đường |
|---|---:|---:|---:|---:|---:|
| C1 100 khách, fold 0 | 9 | 98,89% | 99,61% | +0,72 điểm % | +9,30% |
| 200 khách, chuyển từ fold 0 | 60 | 81,66% | 83,95% | +2,29 điểm % | +8,76% |
| 400 khách, chuyển từ fold 0 | 60 | 59,39% | 60,88% | +1,49 điểm % | +8,72% |
| R1/R2 100 khách, fold 2 | 23 | 83,28% | 83,00% | −0,28 điểm % | +1,18% |

Kết quả cho thấy lợi thế về mức phục vụ trên ba tập áp dụng biểu thức fold 0, tập trung ở một trong hai seed. Ở fold 2, validation C2 đạt trần phục vụ, nên khoảng cách quyết định việc chọn biểu thức. Khi chuyển sang R1/R2, GP tìm được biểu thức phục vụ tốt hơn ở seed 42. Đây là giới hạn khi đánh giá một luật trên một nhóm validation có độ khó khác tập test. Các lượt chuyển quy mô dùng lại cùng hai biểu thức trên 120 instance, nên không đại diện cho 120 lần huấn luyện độc lập.

[Kết quả 200 khách](../outputs/dvrpsim_drone/iterations_20261124/round4_transfer_200.json) · [Kết quả 400 khách](../outputs/dvrpsim_drone/iterations_20261124/round4_transfer_400.json) · [Validation fold 2](../outputs/dvrpsim_drone/iterations_20261124/fold2_round4_validation.json) · [Test fold 2](../outputs/dvrpsim_drone/iterations_20261124/fold2_round4_test.json)

## Tái lập

Preset và hai tùy chọn dòng lệnh dưới đây bật cấu hình vòng 4. Cùng bộ điều phối, đội xe, split và ngân sách chấm được áp dụng cho GP và GPRT.

```powershell
.venv/Scripts/python.exe run_dvrpsim_drone.py train --fleet 10 --fold 0 --seeds 42 43 --config configs/dvrpsim_drone_iterations.json --retry-earlier-gaps --screen-initial-rules --output outputs/dvrpsim_drone/replay_iterations_fold0
.venv/Scripts/python.exe run_dvrpsim_drone.py test --fleet 10 --fold 0 --seeds 42 43 --retry-earlier-gaps --screen-initial-rules --output outputs/dvrpsim_drone/replay_iterations_fold0
```

Với vòng 3, dùng bản sao của preset và đặt `guided_samples` bằng 0. Dùng `--fold 2` cùng thư mục đầu ra mới để tái lập phép kiểm tra chéo. Với test chuyển quy mô, thêm `--transfer` vào lệnh `test`. Số yêu cầu chấm train khớp giữa hai phương pháp trong từng fold. Số lượt mô phỏng thực sau cache và công sức sàng lọc vẫn khác nhau. [Validation vòng 1](../outputs/dvrpsim_drone/iterations_20261124/round1_validation.json) · [Validation vòng 2](../outputs/dvrpsim_drone/iterations_20261124/round2_validation.json) · [Validation vòng 3](../outputs/dvrpsim_drone/iterations_20261124/round3_validation.json) · [Validation vòng 4](../outputs/dvrpsim_drone/iterations_20261124/round4_validation.json) · [Test fold 0](../outputs/dvrpsim_drone/iterations_20261124/round4_test.json)

31 kiểm thử liên quan đã đạt, gồm trường hợp hòa số đơn, thử vị trí chèn sớm và tính log-xác suất phần được lấy mẫu. Bài kiểm thử split Solomon cần bộ dữ liệu ở `data/data/Solomon`, hiện chưa có trong workspace.
