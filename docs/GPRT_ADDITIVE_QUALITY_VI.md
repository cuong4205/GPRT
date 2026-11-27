# Điều chỉnh luật bằng Transformer trong GPRT

## Cách sinh biểu thức

Trong cách sinh cũ, Transformer nhận tiền tố của một cây đã bỏ lá cuối rồi hoàn thiện phần còn lại. Một luật chỉ gồm một đặc trưng không có tiền tố để dùng. Cấu hình mới giữ nguyên toàn bộ luật cha và cho Transformer chọn đặc trưng `x` trong biểu thức `cha + x` hoặc `cha − x`. Các lượt có hướng dẫn luân phiên giữa hai dấu và các luật cha lấy từ tập khởi tạo hoặc kho đã sàng lọc trên train. Khi hết chỗ theo giới hạn độ sâu hoặc số token, cây đó được bỏ qua. Nếu không có cha phù hợp, Transformer sinh tự do.

Một nửa số mẫu vẫn được sinh tự do. Các cây sau đó cùng đi qua GP. Với mẫu có hướng dẫn, REINFORCE dùng log-xác suất của đặc trưng được chọn trong tập đặc trưng hợp lệ. Dấu và cây cha được cố định trước khi lấy mẫu. Loss mô phỏng elite học từ toàn bộ cây. Hiệu quả đo dưới đây thuộc về cấu hình GPRT hoàn chỉnh. Phép so sánh chưa tách riêng vai trò của phân phối Transformer khỏi tác dụng của việc tìm kiếm quanh một luật có sẵn.

## Solomon tĩnh

Chạy trên đúng 56 instance 100 khách của dự án, lấy lại từ commit `f180886` vào thư mục tạm. Split cố định gồm 32 train, 12 validation, 12 test. Cả ba phương pháp dùng cùng bốn luật ban đầu, bộ dựng tuyến, grammar, quần thể 16, hai thế hệ GP mỗi chu kỳ, 32 chu kỳ và 9.216 yêu cầu chấm cây–instance trên train cho mỗi seed. Hai phương pháp GPRT dùng cùng 16 bước học trước, 8 mẫu Transformer và 4 mẫu có hướng dẫn khi có luật cha phù hợp. Năm seed là 20260914 đến 20260918. Mỗi seed chọn một luật bằng validation trước khi chạy test. Nhóm test này cũng xuất hiện trong các thí nghiệm Solomon trước đây của dự án.

| Phương pháp | Tổng số xe qua 60 lượt | Xe trung bình mỗi instance | Quãng đường trung bình mỗi instance | Thời gian tìm luật trung bình mỗi seed |
|---|---:|---:|---:|---:|
| GP | 553 | 9,22 | 1.475,66 | 14,8 giây |
| GPRT hoàn thiện tiền tố | 561 | 9,35 | 1.412,86 | 68,7 giây |
| GPRT điều chỉnh nguyên luật | **548** | **9,13** | **1.379,46** | 68,7 giây |

Mọi lượt test phục vụ đủ 100 khách. So với GP, GPRT mới giảm 0,90% số xe và 6,52% quãng đường. So với GPRT hoàn thiện tiền tố trong cùng thí nghiệm, mức giảm là 2,32% và 2,36%. Từng cặp được xếp theo số xe trước, quãng đường sau. GPRT mới thắng/hòa/thua 32/8/20 cặp so với GP và 31/15/14 so với GPRT hoàn thiện tiền tố. Tách theo seed, GPRT mới dùng ít xe hơn GP ở ba seed và nhiều xe hơn ở hai seed. Baseline hạn giao sớm EDD phục vụ đủ khách, dùng 132 xe trên 12 instance, trung bình 11 xe/instance.

Một ví dụ ở seed 20260915 cho điểm tương đương `d_current + 2 × wait_time + due_date − 2 × next_feasible_fraction` trên các đặc trưng chuẩn hóa. Bộ dựng chọn điểm nhỏ nhất. Luật ưu tiên khách gần, ít phải chờ và đến hạn sớm, đồng thời ưu tiên lựa chọn còn nhiều khách có thể nối tiếp. Ba seed 20260915–20260917 đạt cùng kết quả tổng hợp trên test.

Ước lượng bằng cách lấy mẫu lại 10.000 lần theo cả seed và instance, giữ nguyên ghép cặp, cho khoảng phân vị 95% của chênh lệch số xe GPRT mới trừ GP từ −0,52 đến +0,33 xe/instance. Khoảng này chứa 0. Mức giảm quan sát được đủ để ưu tiên thử cấu hình mới trên Solomon, còn độ chắc chắn về số xe khi đổi sang tập instance khác vẫn thấp. Cả ba cách có cùng số yêu cầu chấm train. Số lượt thực tính sau cache trung bình là 4.964 với GP, 4.129 với GPRT tiền tố và 4.275 với GPRT mới. Thời gian trong bảng được đo trong lúc chạy nhiều thí nghiệm trên cùng máy.

Bộ kiểm tra của dự án đã xác nhận cả ba lượt thực nghiệm năm seed. Phát lại 180 nghiệm bằng bộ dựng Python và kiểm tra độc lập từng tuyến cho cùng số xe và quãng đường với lần chạy Numba, không có vi phạm tải, cửa sổ thời gian hay phục vụ khách. [Kết quả GP](../outputs/gprt_vrptw/solomon_additive_confirm/gp/results.json) · [Kết quả GPRT tiền tố](../outputs/gprt_vrptw/solomon_additive_confirm/prefix/results.json) · [Kết quả GPRT mới](../outputs/gprt_vrptw/solomon_additive_confirm/additive/results.json) · [Kiểm tra tuyến](../outputs/gprt_vrptw/solomon_additive_confirm/route_audit.json)

## Xe tải–drone động

Giữ nguyên giao thức của vòng 4, gồm đội 10 xe, thời gian gom đơn 5 phút, cùng cách chèn, cùng 15 luật được sàng lọc trên train, cùng 8 chu kỳ. Mỗi chu kỳ GPRT dùng 3 mẫu có hướng dẫn và 3 mẫu tự do khi có luật cha phù hợp. Hai cấu hình GPRT chỉ khác cách sinh các mẫu có hướng dẫn. Mục tiêu đầu tiên là số đơn giao đúng hạn, sau đó mới đến quãng đường có trọng số, bằng km xe tải cộng 0,25 lần km drone. So sánh dùng seed 42–46. Với GP và GPRT tiền tố, dùng lại hai seed 42–43 đã lưu và huấn luyện mới ba seed 44–46. GPRT mới được huấn luyện lại cả năm seed trên cả hai fold. Các kết quả dùng chung manifest dữ liệu và điều phối trong từng fold.

| Tập test | Phương pháp | Đơn đúng hạn qua 5 seed | Tổng quãng đường có trọng số |
|---|---|---:|---:|
| C1, fold 0 | GP | 4.468 / 4.500 | 122.113,34 |
| C1, fold 0 | GPRT tiền tố | **4.483 / 4.500** | 125.614,65 |
| C1, fold 0 | GPRT điều chỉnh nguyên luật | 4.481 / 4.500 | 126.422,09 |
| R1/R2, fold 2 | GP | **9.580 / 11.500** | 170.924,32 |
| R1/R2, fold 2 | GPRT tiền tố | 9.545 / 11.500 | 171.572,24 |
| R1/R2, fold 2 | GPRT điều chỉnh nguyên luật | 9.554 / 11.500 | **170.079,02** |

Trên C1, GPRT mới giao thêm 13 đơn so với GP, toàn bộ đến từ seed 43, và tăng 3,53% quãng đường. So với GPRT tiền tố, bản mới giao ít hơn 2 đơn và đi xa hơn 0,64%. Gần trần phục vụ, EDD đạt 899/900 đơn trong một lần chạy trên 9 instance với 29.242,65 km có trọng số. Trên R1/R2, bản mới giao thêm 9 đơn và giảm 0,87% quãng đường so với GPRT tiền tố. GP vẫn giao nhiều hơn bản mới 26 đơn, đổi lại quãng đường dài hơn. Khoảng phân vị 95% ghép cặp cho chênh lệch tỷ lệ đúng hạn giữa GPRT mới và GP là từ 0,00 đến +1,20 điểm phần trăm trên C1 và từ −0,94 đến +0,42 trên R1/R2. Mỗi nhóm dùng lại cùng instance qua năm seed.

Chất lượng trên validation chưa dự báo đầy đủ thứ hạng trên nhóm tọa độ mới. Ở fold 0, GP, GPRT tiền tố và GPRT mới giao lần lượt 6.670, 6.681 và 6.690 đơn trong 8.000 lượt đơn validation. Ở fold 2, cả ba giao đủ 4.000/4.000 trên C2, nên việc chọn luật phụ thuộc vào quãng đường. Sang R1/R2, số đơn đúng hạn lại quyết định thứ hạng. Hai fold này đã có kết quả lịch sử. Đợt đo tập trung vào các instance 100 khách.

Mỗi seed có 1.302 yêu cầu chấm train trên fold 0 và 1.218 trên fold 2 cho cả ba phương pháp. Sau cache, số mô phỏng train thực tế trung bình của GP, GPRT tiền tố và GPRT mới lần lượt là 776,2 / 601,8 / 603,4 trên fold 0 và 725,4 / 563,8 / 552,6 trên fold 2. Sàng lọc 15 luật ban đầu cần 90 episode mỗi fold cho mỗi lần chạy sàng lọc. Các bộ đếm sàng lọc, validation và chẩn đoán được lưu riêng trong từng policy.

## Phạm vi sử dụng

Cấu hình mới là lựa chọn có cơ sở khi ưu tiên đồng thời số xe và quãng đường trên Solomon tĩnh. Với xe tải–drone, chênh lệch về mức phục vụ đổi chiều giữa C1 và R1/R2, nên preset mới được giữ như một cấu hình để so sánh. Bộ kiểm thử của dự án đạt 90/90 trong bản sao có đủ dữ liệu Solomon. [Số liệu từng seed và so sánh ghép cặp](analysis/gprt_additive_quality.json) · [Cấu hình Solomon](../configs/gprt_solomon_additive.json) · [Cấu hình xe tải–drone](../configs/dvrpsim_drone_additive.json) · [Lệnh tái lập](RUN_GPRT_VRPTW.md#so-sánh-hai-cách-sinh-biểu-thức)
