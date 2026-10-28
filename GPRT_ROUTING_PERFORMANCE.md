# Tăng tốc bộ dựng tuyến khi train GPRT

Ngày 28/10/2026, đo trên Solomon của project và thêm backend CPU Numba cho bộ dựng
tuyến tĩnh dùng trong fitness GP/GPRT. Cùng một biểu thức tiếp tục chấm điểm khách
khả thi; thay phần thực thi vòng lặp, không thay thuật toán học hay chính sách chọn khách.

## Điểm nghẽn đo được

Profile 204 lượt dựng tuyến trên 6 instance train đại diện C1/C2/R1/R2/RC1/RC2, gồm
16 terminal, 16 cây ngẫu nhiên có seed và 2 biểu thức look-ahead. Khoảng cách đã cache.
Trong tổng khoảng 22,5 giây có profiler, hàm priority chiếm khoảng 11,3 giây,
tạo `CandidateContext` khoảng 5,3 giây, chuẩn bị feature theo state khoảng 3,7 giây.
Các thời gian lồng nhau không được cộng tùy tiện. Checker độc lập chỉ khoảng 0,17 giây.

Nguồn đo: `outputs/diagnostics/route_builder_before/profile.txt`. Đây là số có overhead
của profiler, dùng để tìm điểm nghẽn; tốc độ trước–sau được đo riêng, không bật profiler.

## Thay đổi

- `gprt_vrptw/fast_routing.py` chứa một kernel biên dịch dùng chung cho mọi cây.
  Mỗi cây được đóng gói thành chương trình postfix nhỏ, không JIT lại từng heuristic.
- Kernel xử lý liên tục lọc capacity/time window/return horizon, tính các feature
  được biểu thức sử dụng, chấm điểm và chọn khách. Không tạo đối tượng Python cho
  từng ứng viên ở từng bước.
- Mảng customer và khoảng cách được chuẩn bị một lần cho mỗi instance. Cache mảng
  khoảng cách bổ sung bị chặn ở một triệu ô và 64 instance; không đổi sang float32.
- Giữ nguyên Euclid từ ma trận `math.hypot` của loader, thứ tự phép tính float64,
  protected arithmetic, tie-break theo ID, capacity/fleet và nghĩa latest service start.
  Không bật fast-math, không tỉa bớt ứng viên hoặc đơn giản hóa cây để lấy tốc độ.
- Mọi tuyến từ kernel vẫn qua `validate_solution` của Python, với kiểm tra coverage,
  thời gian, capacity và tính lại toàn bộ chi phí.
- `make_priority` vẫn có giao diện chấm một `CandidateContext` để dùng trong simulator
  động. Nhánh tăng tốc toàn tuyến chỉ được `solve` tĩnh dùng; không bỏ qua snapshot
  khách đã release của DVRPSim.

## Bật/tắt

```powershell
.venv/Scripts/python.exe -m pip install -r requirements-fast.txt
.venv/Scripts/python.exe run_gprt_experiment.py --config configs/gprt_main.json --route-backend numba
```

`--route-backend auto` là mặc định: dùng Numba nếu đã cài, nếu thiếu thì dùng Python.
`--route-backend python` buộc dùng bản đối chiếu. `--route-backend numba` báo lỗi nếu
chưa cài dependency. Có thể đặt `route_backend` trong preset JSON hoặc biến môi trường
`GPRT_ROUTE_BACKEND`; CLI ưu tiên hơn preset, preset ưu tiên hơn biến môi trường.
API `FitnessEvaluator(route_backend="python")` và `make_priority(tree, backend="python")`
cho phép đối chiếu trực tiếp trong một process.

Dependency tăng tốc nằm riêng ở `requirements-fast.txt`: Numba 0.64.0, llvmlite 0.46.0.
Môi trường đo là Python 3.13.7 / NumPy 2.4.4 trên Windows. Numba hỗ trợ bộ phiên bản
này theo [bảng tương thích chính thức](https://numba.readthedocs.io/en/0.64.0/user/installing.html).

`route_backend` không thêm vào cấu hình thuật toán `GPRTConfig`, vì không đổi nghĩa
evaluator hoặc ngân sách. Config checkpoint cũ vẫn giữ tương thích. Checkpoint mới
ghi `routing_runtime`; log từng cycle và kết quả runner ghi backend đã chọn, cùng
version dependency trong metadata. Cache nghiệm không bị đổi khóa do backend, vì
kết quả phải giống nhau; kiểm chứng ghép cặp xác nhận điều này.

Instance quá lớn để dùng cache ma trận hoặc dữ liệu không thuộc đầu vào canonical
được chuyển về đường Python hiện có. Numba vẫn có chi phí khởi tạo/biên dịch lần đầu;
disk cache được lưu trong `__pycache__`, không JIT mỗi cá thể hay mỗi cycle.

## Đo lại và kiểm chứng

```powershell
.venv/Scripts/python.exe -m scripts.benchmark_route_builder --repeats 5
.venv/Scripts/python.exe -m scripts.benchmark_gprt_routing
.venv/Scripts/python.exe -m pytest tests -q
```

Benchmark dựng tuyến đo xen kẽ hai backend trên cùng 204 cặp, không dùng fitness
cache; mỗi phép đo bao gồm cả checker. Lần chạy kiểm chứng còn so với JSON chứa toàn
bộ nghiệm được lưu trước khi sửa code, không chỉ so hai đường code mới với nhau.
Kết quả ở `outputs/diagnostics/route_builder_benchmark.json`.

Benchmark train chạy hai job chẩn đoán tuần tự, lấy cấu hình từ `configs/gprt_main.json`,
ghi đè population 16, neural samples 8, GP generations 1 và 8 cycle để phủ 32 train
instance. Hai backend dùng cùng ngân sách và seed. Đây là phép đo engineering;
không thay preset nghiên cứu chính 64/32/3 và không dùng để kết luận chất lượng GPRT.
Chỉ dùng train/validation, không mở test. Cả hai job giới hạn một thread cho các
thư viện CPU để đo nhất quán. Báo cáo tính cả thời gian process, startup và checkpoint,
đồng thời so budget, heuristic, learning trajectory và từng tensor model cuối.
Kết quả ở `outputs/diagnostics/gprt_routing_diagnostic/comparison.json`. Có thể truyền
`--population 64 --transformer-samples 32 --gp-generations 3` để đo đầy đủ preset chính.

Không suy mức tăng tốc riêng của routing thành mức tăng tốc toàn bộ train: Transformer,
autograd, selection và ghi checkpoint vẫn có chi phí. Số đo chi tiết được tổng hợp
trong `outputs/diagnostics/route_builder_optimization_report.md`.
