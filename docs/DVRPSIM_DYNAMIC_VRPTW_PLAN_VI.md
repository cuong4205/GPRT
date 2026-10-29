# Kế hoạch đánh giá GPRT trong mô phỏng VRPTW động

> Cập nhật 28/10/2026: yêu cầu mới nhất chọn 70% đơn động và 5% ứng viên hủy thay cho
> quy tắc release uniform bên dưới. Triển khai và hướng dẫn hiện hành ở
> [DVRPSIM_DYNAMIC.md](../DVRPSIM_DYNAMIC.md). Phần bên dưới giữ làm lịch sử thiết kế.

## Mục tiêu và phạm vi

Đánh giá heuristic được chọn từ quá trình huấn luyện GPRT hiện có khi khách hàng chỉ
lộ dần theo thời gian. `dvrpsim` chỉ là môi trường mô phỏng; không đổi pipeline huấn
luyện GPRT và không huấn luyện lại trên dữ liệu test.

Giữ nguyên 56 Solomon, split hiện hành 32/12/12, khoảng cách Euclid không làm tròn,
fleet, capacity, service time và time window. Chỉ 12 instance test dùng cho kết quả
cuối. Thời điểm đơn xuất hiện (`release_date`) là biến mới, tách biệt với `ready_time`
— thời điểm sớm nhất được phục vụ.

`dvrpsim` có order release theo thời gian, callback ra quyết định định tuyến, cửa sổ
phục vụ pickup/delivery và khả năng cập nhật tuyến. Kế hoạch này chỉ dùng phần tối
thiểu: một depot, giao một chiều, xe đồng nhất, các sự kiện đơn mới xuất hiện và xe về
depot. Không sửa tuyến xe đang chạy. Cách dùng đó là một chuyển thể để đánh giá, không
phải benchmark động chuẩn đã có sẵn trong Solomon. [Tài liệu khái niệm](https://sztaki-hu.github.io/dvrpsim/src/introduction/basics.html), [tutorial đơn và thời điểm release](https://sztaki-hu.github.io/dvrpsim/src/tutorial/tutorial_02.html), [routing callback](https://sztaki-hu.github.io/dvrpsim/src/tutorial/tutorial_03b.html).

## Thiết kế tối thiểu

### Mô hình trong simulator

- Mỗi điểm Solomon là một `Location`; depot là điểm pickup và khách hàng là điểm
  delivery của một `Order`.
- Demand là quantity; thời gian dịch vụ ở khách hàng là delivery service duration.
  Xe bắt đầu ở depot, có capacity và số lượng đúng theo instance.
- Override cả travel time lẫn travel distance bằng khoảng cách Euclid Solomon; đặt
  một đơn vị khoảng cách bằng một đơn vị thời gian.
- Ánh xạ `ready_time` thành thời điểm sớm nhất bắt đầu dịch vụ. Giới hạn trên của
  Solomon là thời điểm bắt đầu dịch vụ muộn nhất; kiểm tra nó trong bộ lọc ứng viên
  khả thi và checker độc lập, vì framework để giới hạn trên cho thuật toán quyết định.
- Một đơn chưa được dispatch giữ ở trạng thái chờ; chỉ từ chối nếu không còn khả năng
  phục vụ theo chính sách đã định. Không tính một đơn chờ như đã hoàn thành.

### Chính sách dispatch

Tại sự kiện có đơn mới hoặc xe về depot, lấy các đơn đã release nhưng chưa dispatch.
Nếu có xe rảnh ở depot, dùng bộ dựng tuyến chung để tạo route khả thi cho các xe rảnh;
heuristic xếp hạng các khách hàng đang nhìn thấy bằng cùng grammar và feature của GPRT.
Gán các đơn trong route cho xe, chất hàng tại depot rồi chạy route đến khi quay về.

Để đơn giản, không chèn đơn mới vào tuyến xe đang chạy và không cho xe đang chạy đổi
quyết định đã thực hiện. Đơn xuất hiện lúc xe bận sẽ chờ đến sự kiện dispatch kế tiếp.
Adapter chỉ cần giới hạn tập ứng viên ở các đơn đã release/chưa dispatch và truyền
thời gian, vị trí, tải còn lại hiện tại vào bước chấm điểm; giữ nguyên biểu thức GPRT
đã học. Không dùng dữ liệu của đơn chưa release để chọn khách tiếp theo.

### Tạo kịch bản động

- **Đối chứng tĩnh:** mọi đơn có `release_date = 0`; giữ nguyên time window Solomon.
- **Động:** đặt `H = depot.due_date`; với mỗi instance/scenario seed, lấy
  `u_i ~ Uniform(0, 0.5H)`. Solomon dùng `due_i` làm thời điểm bắt đầu phục vụ muộn
  nhất, nên giới hạn release trực tiếp phải tôn trọng cả cửa sổ dịch vụ lẫn giờ quay về:
  `release_i = min(u_i, due_i - d(depot, i), H - d(depot, i) - service_i - d(i, depot))`.
  Trước khi sinh, xác nhận `max(d(depot,i), ready_i) <= due_i` và
  `max(d(depot,i), ready_i) + service_i + d(i,depot) <= H`. Nếu điều kiện trực tiếp
  này không thỏa hoặc giới hạn release âm, dừng và ghi nhận instance đó thay vì âm thầm
  sửa release time. Không trừ `service_i` khỏi `due_i`, vì như vậy diễn giải due date
  thành hạn hoàn tất, trái với ngữ nghĩa latest service start đã chốt ở trên.
- Lưu seed và release time của từng khách trong manifest. Đây là kịch bản synthetic
  để thử độ bền zero-shot, không phải dữ liệu release thực tế.

## Cách đánh giá heuristic

### Ba chính sách trong cùng simulator

1. **GPRT:** biểu thức validation tốt nhất của từng seed huấn luyện GPRT; khóa biểu
   thức/checkpoint trước khi chạy test động.
2. **GP thuần:** biểu thức validation tốt nhất của từng seed GP-only hiện có; cùng bộ
   dựng tuyến và cùng ngân sách kiểm tra.
3. **EDD:** chọn khách đủ điều kiện có `due_date` sớm nhất; baseline dễ hiểu, không cần
   huấn luyện.

Cả ba dùng cùng thời điểm release, xe, quy tắc khả thi, sự kiện dispatch và cách tính
chi phí. Chạy thêm đối chứng tĩnh `release_date=0` để tách ảnh hưởng của việc lộ đơn
dần khỏi chất lượng biểu thức.

### Bộ đo

Ưu tiên so sánh theo thứ tự:

1. Tỷ lệ khách được phục vụ đúng hạn và số đơn không phục vụ/từ chối.
2. Tổng quãng đường và số xe đã dùng, báo riêng cả trên toàn kịch bản và trên mỗi đơn
   được phục vụ; không dùng quãng đường thấp để thưởng cho chính sách bỏ khách.
3. Số vi phạm capacity/time window (mục tiêu bằng 0), độ trễ routing callback và
   thời gian mô phỏng.

Báo cáo chênh lệch ghép cặp GPRT–GP và GPRT–EDD trên cùng instance/scenario seed,
cùng trung bình và độ phân tán theo instance và seed huấn luyện. Kết luận chính là
heuristic GPRT có giữ được chất lượng và chi phí quyết định khi thông tin đến muộn hay
không; không diễn giải thành bằng chứng về hiệu quả vận hành thực tế.

## Các bước thực hiện

| Bước | Việc làm | Điều kiện để qua bước |
|---|---|---|
| 1. Kiểm tra tương thích | Cài một phiên bản cố định của `dvrpsim`; mô phỏng một xe và hai đơn; xác nhận release, capacity, service time, ready time, due time và khoảng cách. | Checker Solomon độc lập phát hiện được một trường hợp vi phạm do cố ý; giới hạn trên của time window được xử lý cứng và đúng quy ước Solomon. |
| 2. Adapter nhỏ | Tạo model/vehicle/order provider và callback. Gắn bộ dựng tuyến GPRT hiện có; không sửa package upstream. | Một instance nhỏ ở chế độ tĩnh khớp feasibility và tính lại được quãng đường; một đơn release muộn chỉ xuất hiện sau đúng thời điểm. |
| 3. Pilot | Chạy 4 instance validation đại diện, 2 release seed, ba chính sách và đối chứng tĩnh. | Không có đơn chưa release lọt vào quyết định; không có vi phạm constraint bị giấu; đo được callback latency. |
| 4. Thí nghiệm cuối | Chạy đủ 12 test instance × 3 release seed; dùng policy khóa từ validation cho từng training seed GPRT/GP trong protocol 5 seed. Chạy EDD trên cùng kịch bản. | Lưu manifest, log sự kiện, kết quả theo episode và bảng tổng hợp để tái lập. |
| 5. Phân tích | So sánh tỷ lệ phục vụ trước, quãng đường sau; phân tích theo nhóm Solomon và độ trễ callback. | Báo cáo có cả kết quả âm, giới hạn của release synthetic, cùng độ biến thiên qua training seed. |

## Đầu ra cần lưu

- Cấu hình simulator và phiên bản dependency.
- Manifest theo episode: instance, split, policy/checkpoint/hash biểu thức, training
  seed, scenario seed, release time từng đơn và tham số simulator.
- CSV/JSON theo episode: served, on-time, rejected/unserved, violations, distance,
  vehicles used, callback count/latency và wall-clock time.
- Bảng tóm tắt ghép cặp GPRT, GP, EDD cho static và dynamic.

## Rủi ro cần khóa trước khi mở rộng

- **Ngữ nghĩa due date:** tài liệu framework xác định giới hạn latest service start
  là trách nhiệm của thuật toán quyết định ([bài framework](https://doi.org/10.1016/j.ejtl.2025.100159)). Enforce trong bộ lọc ứng viên và checker sau mô phỏng; không dựa vào simulator để tự từ chối một lượt trễ.
- **Chuyển phân phối:** heuristic học trên thông tin đầy đủ được áp dụng cho tập đơn
  đang lộ. Giữ nguyên heuristic để đo zero-shot; không huấn luyện lại trên test. Ghi
  rõ đây là giới hạn của phép thử.
- **Phạm vi động:** bản tối thiểu chỉ đo release bất đồng bộ, không đo giao thông biến
  động, hủy đơn, chèn route giữa chuyến hoặc bất định travel time. Chỉ thêm khi kết quả
  lõi và ngân sách cho phép.

`dvrpsim` được phát hành theo MIT; nếu dùng trong khóa luận, trích dẫn bài framework
được nêu trong [repository](https://github.com/sztaki-hu/dvrpsim) và ghi phiên bản đã
khóa.

## Trạng thái triển khai ngày 2026-10-27

Đã khóa `dvrpsim==1.0.0`, triển khai adapter/callback, checker độc lập, pilot 132 episode
trên validation và đánh giá 528 episode trên đủ 12 test instance. Cả hai gate đều đạt,
0 vi phạm được ghi nhận. Kết quả và giới hạn diễn giải nằm trong
[`DVRPSIM_DYNAMIC_VRPTW_RESULTS_VI.md`](DVRPSIM_DYNAMIC_VRPTW_RESULTS_VI.md). Khi triển
khai, công thức release được hiệu chỉnh để khớp định nghĩa Solomon rằng due date là hạn
bắt đầu phục vụ; lý do và ví dụ C102 được ghi ở tài liệu implementation.
