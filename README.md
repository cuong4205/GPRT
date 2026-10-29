# GPRT-VRPTW

Project triển khai GPRT đầy đủ trên 56 instance Solomon VRPTW. Transformer sinh biểu thức heuristic, GP tiến hóa quần thể, REINFORCE học từ mẫu neural on-policy, và elite imitation truyền tín hiệu từ các cây GP đã sàng lọc.

Runner hiện có hai chế độ: GPRT đầy đủ (mặc định) và GP thuần. Archive seeding là tùy chọn cấu hình của GPRT. Các mô hình standalone và các ablation cũ đã được gỡ khỏi luồng chạy.

Hướng dẫn: [chạy thí nghiệm](docs/RUN_GPRT_VRPTW.md), [GP và GPRT](docs/GPRT_MODELS_AND_VARIANTS_VI.md), [tài liệu kỹ thuật](docs/GPRT_PROJECT_TECHNICAL_DOCUMENTATION_VI.md), [kế hoạch đánh giá động với dvrpsim](docs/DVRPSIM_DYNAMIC_VRPTW_PLAN_VI.md). Các [báo cáo pilot](docs/GPRT_VRPTW_REPORT_VI.md) được giữ làm tư liệu lịch sử.

## Chạy

Luồng đầy đủ **GP + GPRT 32 cycle → khóa heuristic text → validation động đủ 12
instance → test tĩnh → test mô phỏng động**:

```powershell
.venv/Scripts/python.exe run_end_to_end.py --output-dir outputs/end_to_end_32cycle/run_01
```

Xem trước bằng `--dry-run`, tiếp tục các stage đã hoàn tất bằng `--resume`.
Chi tiết split, gate, cấu hình và artifact: [luồng end-to-end](docs/END_TO_END_GPRT_GP_VI.md).

Cài dependency và chạy test:

```powershell
python -m pip install -r requirements-dev.txt
python -m pytest tests -q
```

Chạy GPRT đầy đủ theo preset chính:

```powershell
python run_gprt_experiment.py --config configs/gprt_main.json
```

Tăng tốc bộ dựng tuyến trên CPU bằng Numba (đã cài trong `.venv` của dự án):

```powershell
python -m pip install -r requirements-fast.txt
python run_gprt_experiment.py --config configs/gprt_main.json --route-backend numba
```

Mặc định `auto` dùng Numba khi có, còn `--route-backend python` dùng bản Python để
đối chiếu. Hai backend giữ nguyên feature, heuristic, tie-break, feasibility và ngân
sách đánh giá. Chi tiết đo trước–sau: [GPRT_ROUTING_PERFORMANCE.md](GPRT_ROUTING_PERFORMANCE.md).

Chạy riêng GP thuần:

```powershell
python run_gprt_experiment.py --config configs/gp_main.json
```

Chạy GPRT với archive seeding (preset screening dùng hai seed, chỉ train/validation):

```powershell
python run_gprt_experiment.py --config configs/gprt_archive_seed_8cycle.json
```

Có thể override preset bằng tham số CLI, ví dụ `--cycles 16`, `--seeds ...` hoặc `--archive-seed-count 4`. `run_experiment.py` được giữ làm lối gọi tương thích cho chế độ GP thuần; runner hợp nhất là `run_gprt_experiment.py`.

Kiểm tra output của runner hợp nhất:

```powershell
python validate_gprt_results.py --results-dir outputs/gprt_vrptw/main
```

## Mô phỏng DVRPTW động

Đã tích hợp DVRPSim với biểu thức heuristic GP/GPRT đã chọn theo validation. Kịch bản
mặc định chọn 70% đơn công khai động (`max(0, ready_time - lead)`, lead 10–15) và 5%
ứng viên yêu cầu hủy; chỉ hủy đơn chưa được gán xe. GP, GPRT và EDD dùng chung kịch bản,
bộ lọc khả thi và cơ chế dispatch. Có thể thêm greedy qua `--methods`.

```powershell
python run_dvrpsim_dynamic.py --phase smoke
python run_dvrpsim_dynamic.py --phase all
python validate_dvrpsim_results.py --results-dir outputs/dvrpsim_dynamic_70pct_cancel5/final
```

Hướng dẫn cấu hình, replay và quy ước mô phỏng: [DVRPSIM_DYNAMIC.md](DVRPSIM_DYNAMIC.md).
Tài liệu này cũng trình bày đầy đủ sự kiện, thời điểm replan, ràng buộc, cách tích hợp
heuristic và báo cáo performance GP/GPRT ở kịch bản 70% động.
Môi trường `.venv` đã được chuẩn bị cho lần tích hợp này; có thể gọi trực tiếp
`.venv/Scripts/python.exe` thay cho `python`.

## Cấu trúc

- `gprt_vrptw/`: GPRT, GP, Transformer, grammar, evaluator và archive selection.
- `vrptw_baselines/`: loader Solomon, bộ dựng tuyến và kiểm tra khả thi dùng chung.
- `configs/`: preset GPRT chính, archive seeding và GP-only.
- `dvrpsim_dynamic/`: scenario tái lập, adapter DVRPSim và bộ định tuyến động dùng heuristic chung.
- `outputs/`: kết quả thí nghiệm; các kết quả baseline cũ được giữ làm tư liệu, không còn được tạo bởi runner hiện hành.
- `docs/`: hướng dẫn chạy, mô tả mô hình và báo cáo lịch sử.
