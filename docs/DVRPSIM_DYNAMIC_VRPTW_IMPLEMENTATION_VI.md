# Triển khai đánh giá VRPTW động bằng dvrpsim

## Mục đích

Mô-đun này áp dụng zero-shot các biểu thức heuristic đã chọn trên validation của
GPRT và GP-only vào kịch bản Solomon có khách hàng được phát hành theo thời gian.
Không huấn luyện lại, không chọn biểu thức bằng test và không sửa tuyến đang chạy.
EDD là baseline không học.

`dvrpsim==1.0.0` mô phỏng thời điểm phát hành đơn, chuyến xe, phục vụ và xe về depot.
Đây là chuyển thể synthetic cho mục tiêu nghiên cứu, không phải bộ benchmark DVRPTW
chuẩn hay dữ liệu vận hành thực.

## Quy tắc đánh giá

- Giữ Solomon, split 32/12/12, fleet, tải, service time và cửa sổ thời gian trong
  instance. Khoảng cách là Euclid không làm tròn; travel time bằng khoảng cách.
- `release_date` được tạo độc lập với `ready_time`. Kịch bản tĩnh đặt mọi release bằng
  0. Kịch bản động sinh `Uniform(0, 0.5H)` rồi chặn theo due time kiểu latest service
  start và giờ quay về depot: `min(u_i, due_i - d(depot,i), H - d(depot,i) -
  service_i - d(i,depot))`. Trước đó kiểm tra khả thi trực tiếp tại release 0; episode
  không khả thi dừng với lỗi được ghi rõ.
- Dispatcher chỉ đọc đơn đã xuất hiện trong `dvrpsim.get_state()`, và chỉ dispatch tại
  release hoặc khi xe ở depot. Đơn chưa được gán xe được chấp nhận và chờ. Mỗi xe nhận
  một route mới khi ở depot; tuyến xe đang di chuyển không bị cập nhật.
- Due time của Solomon là latest service start. Adapter lọc ứng viên theo due time và
  thời điểm xe có thể trở lại depot; checker sau mô phỏng tính lại capacity, ready/due,
  tính đầy đủ/duy nhất của đơn, thời gian phục vụ và quãng đường. Lỗi không được giấu.
- GPRT và GP dùng `final_heuristic.txt` của từng training seed; hash biểu thức,
  checkpoint, instance, cấu hình và release times được ghi trong manifest.

### Hiệu chỉnh ngữ nghĩa release

Kế hoạch ban đầu viết `due_i - d(depot,i) - service_i`, nhưng phần còn lại của kế hoạch
định nghĩa `due_i` là latest service start. Solomon C102 minh họa xung đột: khách 13 có
ready time 30, due time 92, service 90 và cách depot 30.81; đơn có thể bắt đầu phục vụ
đúng hạn ở 92, dù phép trừ service cho ra release bound âm. Bản triển khai giữ ngữ nghĩa
latest service start, đặt điều kiện service time vào kiểm tra quay về depot, và ghi sửa
đổi đó trong manifest/tài liệu để không loại nhầm episode.

## Chạy

Cài dependencies đã pin trong `requirements.txt`, sau đó chạy từ thư mục project:

```powershell
python run_dvrpsim_dynamic.py --phase pilot
python run_dvrpsim_dynamic.py --phase final
python run_dvrpsim_dynamic.py --phase all
```

`pilot` dùng 4 validation instance và 2 scenario seed. `final` chạy toàn bộ 12 test
instance, 3 scenario seed, cả đối chứng tĩnh và 5 training seed cho GPRT/GP. `all`
chạy pilot trước; pilot có vi phạm constraint sẽ dừng, không chạy test.

Artifacts được ghi dưới `outputs/dvrpsim_dynamic/<phase>/`:

- `run_manifest.json`, `episode_manifest.jsonl`: phiên bản, hash artifact và release
  time từng đơn.
- `episodes.csv`, `events.jsonl`: metric theo episode và log sự kiện.
- `summary_by_policy.csv`, `paired_comparisons.csv`: trung bình/độ phân tán và chênh
  lệch ghép cặp.
- `gate_report.json`: kiểm tra vi phạm constraint trước khi kết thúc phase.

Kết quả chạy pilot và đủ 12 test instance được tổng hợp trong
[`DVRPSIM_DYNAMIC_VRPTW_RESULTS_VI.md`](DVRPSIM_DYNAMIC_VRPTW_RESULTS_VI.md).

## Giới hạn diễn giải

Kết quả đo độ bền zero-shot với release tổng hợp. Mô hình không chèn đơn vào route đang
chạy; không có hủy đơn, giao thông thay đổi hoặc thời gian di chuyển bất định. Do đó
không suy ra trực tiếp chất lượng của hệ thống định tuyến vận hành thực tế.
