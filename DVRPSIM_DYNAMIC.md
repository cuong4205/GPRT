# Tích hợp DVRPSim cho GP và GPRT

Triển khai theo `dvrpsim_dvrptw_dynamic_implementation_plan.md` người dùng cung cấp ngày
28/10/2026. Kịch bản 30% động + 5% ứng viên hủy thay cho kịch bản uniform-release trong
kế hoạch cũ. Pipeline huấn luyện GPRT vẫn sinh biểu thức heuristic; mô phỏng sử dụng
biểu thức đã chọn theo validation, không chạy GP/RL lại trên từng instance test.

## Chạy

Dependency: `dvrpsim==1.0.0`, `simpy==4.1.1`, cùng NumPy và PyTorch hiện có.
Hai phiên bản mô phỏng được kiểm tra khi bắt đầu episode. Môi trường `.venv` trong dự
án đã có dependency cần thiết, dùng thư viện ML của Python hiện có qua system-site-packages.

```powershell
.venv/Scripts/python.exe run_dvrpsim_dynamic.py --phase smoke
.venv/Scripts/python.exe run_dvrpsim_dynamic.py --phase pilot
.venv/Scripts/python.exe run_dvrpsim_dynamic.py --phase all
.venv/Scripts/python.exe validate_dvrpsim_results.py --results-dir outputs/dvrpsim_dynamic_30pct_cancel5/final
```

Trên môi trường mới: `python -m pip install -r requirements-dev.txt`, sau đó chạy bằng
`python`. Kiểm thử bằng `python -m pytest tests -q`. Trong môi trường Windows có thư
mục tạm hạn chế quyền, có thể đặt thư mục tạm của Python vào `tmp/test_runtime` trong
project trước khi chạy pytest.

- `smoke`: một validation instance, một scenario seed, một training seed mỗi phương pháp.
- `pilot`: 4 validation instance × (1 static + 2 dynamic seed) × (5 GPRT + 5 GP + 1 EDD) = 132 episode.
- `final`: 12 test instance × (1 static + 3 dynamic seed) × 11 policy = 528 episode.
- `all`: pilot phải qua kiểm tra constraint trước khi chuyển sang final.

Runner đọc `configs/dvrpsim_dynamic.json`. Thay cấu hình bằng `--config`; đổi nơi lưu
bằng `--output-dir`. Chỉ dùng baseline không cần mô hình:

```powershell
python run_dvrpsim_dynamic.py --phase smoke --methods edd greedy
```

## Kết nối heuristic GP và GPRT

Mỗi policy lấy `final_heuristic.txt` tại `<policy_root>/<method>/<training_seed>/`.
Parser chỉ chấp nhận grammar, feature và hằng số của dự án. `make_priority(tree)` dùng
chung evaluator với huấn luyện; điểm nhỏ hơn được ưu tiên, hòa điểm theo ID gốc.
Hai policy chỉ khác biểu thức đã học; cùng feasibility filter, dispatch và dữ liệu.
Không cần tải Transformer để áp dụng heuristic. Checkpoint nếu có vẫn được lưu hash
trong manifest để truy xuất nguồn gốc.

Cấu hình hiện dùng các biểu thức sẵn có trong `outputs/gprt_vrptw/reduced_main`, gồm
5 seed GP và 5 seed GPRT. Đây là artifact lịch sử của pilot (GPRT 4 chu kỳ), không phải
bằng chứng đã huấn luyện xong preset chính 32 chu kỳ. Để dùng kết quả mới, khai báo
`policy_roots` trong config, ví dụ:

```json
{
  "policy_roots": {
    "gprt": "outputs/gprt_vrptw/main",
    "gp": "outputs/gprt_vrptw/gp_main"
  }
}
```

Đây là phần bổ sung vào config đầy đủ. Có thể dùng `--policy-root` nếu hai phương pháp
cùng thư mục gốc, và `--training-seeds 20260914 ...` để chọn seed. Thiếu artifact được
yêu cầu thì runner báo lỗi; không tự thay bằng EDD hay một heuristic khác.

## Sinh và replay kịch bản

`scenario.py` dùng canonical model/parser Solomon hiện có; không thay tọa độ, demand,
fleet, ready time, due date hay service time. Parser kiểm tra ID liên tục, fleet/capacity,
dữ liệu hữu hạn và time window. Due date là **hạn bắt đầu phục vụ**, không phải hạn hoàn tất.

- Số lượng nhóm dùng decimal **round half up**; 0,5 làm tròn lên. Với 100 khách: 30 đơn
  được chọn động, 5 ứng viên hủy. Instance nhỏ có thể có 0 yêu cầu hủy.
- Mỗi đơn động có `release_time = max(0, ready_time - randint(10,15))`. Đơn còn lại
  công khai tại 0. Số thực sự release sau 0 có thể nhỏ hơn 30%.
- Nhóm động, nhóm hủy, lead và delay dùng bốn luồng RNG độc lập, seed con từ SHA-256.
  Hai nhóm được phép giao nhau. `cancel_time = release_time + randint(1,15)`.
- Static control: tất cả release tại 0, không có cancellation.
- Không ép đơn động luôn khả thi bằng cách sửa release hoặc time window. Đơn không
  phục vụ được được giữ trong số liệu và có lý do.

Mỗi scenario được lưu JSON **trước khi chạy các policy**, có hash instance, seed,
quy tắc, ID nhóm và thời điểm sự kiện. Mọi policy dùng cùng object/hash kịch bản.
Scenario cũ trong thư mục output phải khớp kịch bản được yêu cầu; nếu khác thì runner
báo lỗi để chọn thư mục khác.

Replay chính xác một file đã lưu:

```powershell
python run_dvrpsim_dynamic.py --phase smoke --scenario-file outputs/dvrpsim_dynamic_30pct_cancel5/pilot/scenarios/c102_dynamic_20261025.json --output-dir outputs/dvrpsim_replay
```

API Python cho một instance, ví dụ C101:

```python
from dvrpsim_dynamic import generate_scenario, load_selected_policies, run_episode
from vrptw_baselines.data import parse_instance

instance = parse_instance("data/data/Solomon/c101.txt")
scenario = generate_scenario(instance, seed=42)
scenario.save("outputs/c101_scenario42.json")
policies = load_selected_policies(".", training_seeds=(20260914,))
for policy in policies:
    result, events = run_episode(
        instance, scenario.release_times, policy.name, policy=policy,
        scenario=scenario, scenario_mode=scenario.mode, scenario_seed=scenario.seed,
        episode_id=f"c101_{policy.name}_42",
    )
```

## Điều phối, hủy và thông tin được nhìn thấy

Simulator quản lý đồng hồ, travel/service và các chuyển trạng thái. Policy được gọi
khi có đơn mới, hủy thành công hoặc xe về depot. Xe rảnh ở depot dựng tuyến khả thi
bằng heuristic; đơn mới lúc tất cả xe bận chờ lần dispatch tiếp theo. Mỗi chuyến đã
gán được giữ đến khi xe quay về depot. Không có chèn đơn vào xe đang chạy trong phạm vi này.

Đội xe cố định, xe **được tái sử dụng cho nhiều chuyến** và nạp lại capacity ở depot.
Báo cáo tách `vehicles_used` và `trip_count`. Dynamic simulator và solver tĩnh một
chuyến/xe có phạm vi vận hành khác nhau; static control dùng cùng simulator để so sánh công bằng.

Không chỉ lọc tập ứng viên: policy nhận một instance snapshot chỉ gồm depot và khách
đã được công khai. Chuẩn hóa khoảng cách, tỷ lệ khách và look-ahead đều dùng thông tin
đã lộ. Các công thức/grammar giữ nguyên; thang chuẩn hóa có thể đổi khi đơn mới xuất
hiện. Không truyền lịch release/hủy tương lai vào policy.

Hủy đọc trạng thái hiện hành từ `Model.get_state()`. `assigned_vehicle` khác `None`
thì từ chối ngay cả khi xe chưa rời depot. Đơn đã hoàn tất/đã hủy cũng bị từ chối.
Đơn đã release và chưa gán được hủy bằng `Order.cancel()`. Trước release, adapter lưu
tombstone và chặn provider gọi `request_order`; không thêm đơn ẩn vào trạng thái router.
Hủy lặp không đổi trạng thái lần nữa.

Sự kiện cùng timestamp theo priority số của SimPy:

1. Sự kiện vehicle/service/release của framework (thường priority 3; process nội bộ 0/1).
2. Cancellation request (priority 4).
3. Routing sau khi batch sự kiện đã hoàn tất (priority 5).
4. Ghi mốc horizon (priority 6).

Vì vậy, release và hủy cùng thời điểm: đơn được release rồi hủy trước assignment.
Completion và hủy cùng thời điểm: completion xử lý trước, yêu cầu hủy bị từ chối.
Assignment đã diễn ra ở thời điểm trước luôn ngăn hủy.

Horizon phục vụ là `depot.due_date`; feasibility filter buộc xe quay về trước horizon.
Simulator chạy đến hết mọi sự kiện bắt buộc, kể cả yêu cầu hủy sau horizon; mốc horizon
và thời điểm kết thúc event queue được lưu riêng. Đơn chưa phục vụ không biến mất.

## Đầu ra và kiểm chứng

Mỗi phase lưu:

| File | Nội dung |
|---|---|
| `scenarios/*.json` | Kịch bản materialized, dùng lại cho mọi policy |
| `run_manifest.json` | Config thực dùng, version, split, commit, trạng thái sửa cục bộ, hash source/policy/checkpoint |
| `episode_manifest.jsonl` | Policy, seed, instance và đường dẫn/hash scenario mỗi episode |
| `episode_results.jsonl` | Toàn bộ metrics, route, thời điểm assignment/dispatch/return, final state, lý do unserved |
| `events.jsonl` | Release, routing, hủy accepted/rejected, di chuyển, dịch vụ, horizon và kết thúc |
| `episodes.csv` | Metrics mỗi episode để phân tích |
| `summary_by_policy.csv` | Trung bình/độ lệch chuẩn theo policy, training seed, static/dynamic |
| `paired_comparisons.csv` | Chênh lệch GPRT–GP/EDD/greedy ghép cặp trên cùng instance/scenario |
| `gate_report.json` | Kết quả checker constraint của phase |
| `validation_report.json` | Đối chiếu độc lập scenario, event log, routes và metrics bằng validator |

`requested = served + cancelled + unserved`. `cancellation_requests` tách thành
accepted/rejected; success rate tính trên requests. Tỷ lệ served/on-time chính dùng
mẫu số `requested - cancelled`; lưu thêm tỷ lệ served trên toàn bộ đơn gốc. Quãng đường
phải đọc cùng số phục vụ/hủy/chưa phục vụ. Đơn không phục vụ có lý do
`infeasible_at_release`, `no_feasible_dispatch_before_horizon` hoặc `rejected`.
Một đơn infeasible là kết quả dịch vụ, khác với việc xe thực thi một tuyến vi phạm constraint.

Checker tính lại capacity, thời điểm bắt đầu dịch vụ, return horizon, fleet, coverage,
khoảng cách và rò rỉ thông tin. Validator đọc lại log đã lưu để đối chiếu thêm lịch
release/hủy, trạng thái đơn và phép đếm metrics. Test bao gồm cả hủy trước release,
trùng timestamp, trước departure sau assignment, sau completion, hủy lặp, xe bận,
không xe khả thi, hết horizon, replay cùng seed và feature không đọc đơn ẩn.

## API đã đối chiếu trên bản pin

Đã đọc source của wheel 1.0.0 đang dùng, không dựa vào tên API pseudocode trong plan.

| API/source upstream | Hành vi dùng trong adapter |
|---|---|
| `model.py:165–205`, `Model.request_order` | Thêm order vào Model tại release date; gọi request callback và decision point |
| `model.py:265–329`, `_routing` | Debounce bằng timeout priority 5; callback và enforcement cùng thời gian mô phỏng |
| `model.py:331–403`, `get_state` | Đơn open và assignment trong carrying/current/next pickup visits |
| `elements/order.py:207–223`, `Order.cancel` | Đặt trạng thái CANCELED và gọi cancellation callback; chưa tự cấm mọi assignment |
| `environment.py`, `DVRPEnvironment.timeout` | Priority số thấp chạy trước; tên high/low không được diễn giải ngược |
| `elements/vehicle.py:579–590` | Simulator chờ earliest service start; latest start do router/checker enforce |

Nguồn chính thức: [release/request tutorial](https://sztaki-hu.github.io/dvrpsim/src/tutorial/tutorial_02.html),
[routing tutorial](https://sztaki-hu.github.io/dvrpsim/src/tutorial/tutorial_03b.html),
[repository](https://github.com/sztaki-hu/dvrpsim). Test chạy ví dụ năm request với release
8/16/24/32/40 từ tutorial chính thức, cùng các episode có xe/constraint của dự án.
Giữ trích dẫn Horváth & Tamási (2025), *A general modeling and simulation framework
for dynamic vehicle routing*, DOI [10.1016/j.ejtl.2025.100159](https://doi.org/10.1016/j.ejtl.2025.100159).

Đợt kiểm chứng ngày 28/10/2026 lưu tại `outputs/dvrpsim_plan_20261028`; báo cáo tổng
hợp của đợt chạy nằm trong `verification_report.md` ở thư mục này. Kết quả uniform-release
cũ trong `outputs/dvrpsim_dynamic` giữ riêng vì khác quy tắc kịch bản.
