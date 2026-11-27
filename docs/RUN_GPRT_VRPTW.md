# Hướng dẫn chạy

Chạy các lệnh từ thư mục gốc. Cài `requirements-dev.txt` theo [README](../README.md); `requirements-fast.txt` bổ sung Numba. Dùng `python` thay `.venv/Scripts/python.exe` nếu đã kích hoạt môi trường. Các lệnh train đầy đủ có thể tốn nhiều thời gian.

## Solomon: GP và GPRT

Cần đủ 56 instance trong `data/data/Solomon` hoặc truyền `--data-dir <đường-dẫn>`. Đường dẫn mặc định hiện chưa có trong workspace. Cấu hình chính dùng 32 cycle và 5 seed; screening tối đa hai seed và nên bỏ test.

```powershell
.venv/Scripts/python.exe run_gprt_experiment.py --config configs/gprt_main.json
.venv/Scripts/python.exe run_gprt_experiment.py --config configs/gp_main.json
.venv/Scripts/python.exe run_gprt_experiment.py --config configs/gprt_archive_seed_8cycle.json
.venv/Scripts/python.exe validate_gprt_results.py --results-dir outputs/gprt_vrptw/main
```

`--route-backend auto` chọn Numba nếu có; `python` và `numba` cho phép chọn rõ backend. Override đường dẫn với `--data-dir`, `--output-dir`. Để xem tham số khác chạy `run_gprt_experiment.py --help`. Preset archive là GPRT screening train/validation, không phải method mới. `run_experiment.py` là wrapper tương thích GP.

## Solomon end-to-end

```powershell
.venv/Scripts/python.exe run_end_to_end.py --data-dir <duong-dan-Solomon> --dry-run
.venv/Scripts/python.exe run_end_to_end.py --data-dir <duong-dan-Solomon> --output-dir outputs/end_to_end_32cycle/run_01
```

Bỏ `--data-dir` nếu dữ liệu đã ở đường dẫn mặc định. Dry-run vẫn đọc và kiểm tra dữ liệu, nhưng không train hoặc ghi output. Pipeline khóa 5 seed cho mỗi phương pháp và 32 cycle: train bỏ test → khóa biểu thức → validation động → test tĩnh → test động → báo cáo. `--resume` yêu cầu cùng output, code, dữ liệu, cấu hình và môi trường; stage hoàn tất được kiểm tra hash và bỏ qua, stage chưa hoàn tất chạy lại từ đầu. Giữ các tùy chọn `--data-dir`, `--device`, `--route-backend`, `--threads` nhất quán.

## Solomon động độc lập

Cần Solomon, split manifest và heuristic GP/GPRT đã train. Đường dẫn mặc định được ghi trong `configs/dvrpsim_dynamic.json`; sửa bản sao config nếu dữ liệu, manifest hoặc policy nằm ở nơi khác, hoặc dùng `--policy-root`. Chạy trong output mới:

```powershell
.venv/Scripts/python.exe run_dvrpsim_dynamic.py --config configs/dvrpsim_dynamic.json --phase smoke --output-dir outputs/dynamic_run
.venv/Scripts/python.exe run_dvrpsim_dynamic.py --config configs/dvrpsim_dynamic.json --phase all --output-dir outputs/dynamic_run
.venv/Scripts/python.exe validate_dvrpsim_results.py --results-dir outputs/dynamic_run/final
```

Các phase gồm `smoke`, `pilot`, `final`, `all` (pilot và final). Có thể replay scenario bằng `--phase smoke --scenario-file <file>`. Mặc định có GP, GPRT, EDD; `--methods` có thêm lựa chọn `greedy`. Tham khảo [mô hình](MODEL.md) để phân biệt chính sách động và benchmark tĩnh.

## Xe tải–drone

Dataset CSV mặc định đã có trong `data/Dataset_DVRPTW_drone_GP_RNN`. Các lệnh sau dùng seed mặc định 42, fold 0 và output riêng:

```powershell
.venv/Scripts/python.exe run_dvrpsim_drone.py inspect --fleet 10 --fold 0
.venv/Scripts/python.exe run_dvrpsim_drone.py train --fleet 10 --fold 0 --config configs/dvrpsim_drone_quality.json --output outputs/dvrpsim_drone/new_run
.venv/Scripts/python.exe run_dvrpsim_drone.py test --fleet 10 --fold 0 --output outputs/dvrpsim_drone/new_run
.venv/Scripts/python.exe run_dvrpsim_drone.py test --fleet 10 --fold 0 --output outputs/dvrpsim_drone/new_run --transfer
```

Dùng cùng `--seeds`, `--methods`, `--fleet`, `--fold` và các tùy chọn điều phối khi train/test. Với output của nhiều seed, chỉ định lại `--seeds` để đánh giá đúng các seed cần dùng. Resume GPRT chưa hoàn thành bằng `train ... --methods gprt --resume` với cùng cấu hình, seed và thư mục; GP không hỗ trợ resume này. Mỗi fold/cấu hình dùng thư mục riêng. Xem [điều phối và cấu hình](DVRPSIM_DRONE_VI.md).

## So sánh hai cách sinh biểu thức

Preset `dvrpsim_drone_additive.json` cho Transformer lấy một luật đã có và học thêm hoặc bớt một đặc trưng khi chấm vị trí chèn. Lệnh sau huấn luyện cả GP và GPRT rồi đánh giá các biểu thức đã khóa trên nhóm tọa độ giữ lại. Đổi cả hai lệnh sang `--fold 2` và thư mục mới để lặp lại với nhóm R1/R2.

```powershell
.venv/Scripts/python.exe run_dvrpsim_drone.py train --fleet 10 --fold 0 --seeds 42 43 44 45 46 --config configs/dvrpsim_drone_additive.json --retry-earlier-gaps --screen-initial-rules --output outputs/dvrpsim_drone/additive_replay_f0
.venv/Scripts/python.exe run_dvrpsim_drone.py test --fleet 10 --fold 0 --seeds 42 43 44 45 46 --retry-earlier-gaps --screen-initial-rules --output outputs/dvrpsim_drone/additive_replay_f0
```

Đối chứng GPRT cũ dùng `dvrpsim_drone_iterations.json` và thư mục đầu ra riêng. Trên Solomon tĩnh, chạy `run_gprt_experiment.py --config configs/gprt_solomon_additive.json --data-dir <duong-dan-Solomon>`. Preset cấp cho GP và GPRT cùng bốn luật ban đầu. Để tái lập đối chứng GPRT tĩnh, đổi riêng `guided_mode` thành `complete_prefix` trong bản sao preset và chọn thư mục đầu ra mới. [Kết quả năm seed](GPRT_ADDITIVE_QUALITY_VI.md)

Nếu workspace thiếu Solomon, có thể lấy lại đúng bộ 56 instance từ lịch sử Git của dự án vào một thư mục đầu ra mới rồi chạy như sau.

```powershell
New-Item -ItemType Directory -Path outputs/solomon_replay_data -Force | Out-Null
git archive --format=zip --output=outputs/solomon_replay_data/instances.zip f180886 data/data/Solomon
Expand-Archive -LiteralPath outputs/solomon_replay_data/instances.zip -DestinationPath outputs/solomon_replay_data
$env:OMP_NUM_THREADS='1'
$env:MKL_NUM_THREADS='1'
.venv/Scripts/python.exe run_gprt_experiment.py --config configs/gprt_solomon_additive.json --data-dir outputs/solomon_replay_data/data/data/Solomon --route-backend numba
```

## Artifact

Solomon runner ghi `results.json`, split manifest, CSV, heuristic và log theo method/seed; GPRT có checkpoint. Pipeline end-to-end ghi manifest, các cấu hình đã khóa, log, kết quả validation/test và báo cáo. Xe tải–drone ghi `benchmark.json`, `policy.json`, heuristic, log/checkpoint và `test_results.json` hoặc `transfer_results.json`. Không dùng kết quả test để chọn lại heuristic. Các kết quả trong `outputs/` là dữ liệu cục bộ và có thể không có trong bản sao repository.
