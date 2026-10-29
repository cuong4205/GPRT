# Chạy GPRT trên Solomon VRPTW

## Phạm vi hiện hành

Runner hợp nhất chỉ có hai chế độ: `gprt` (mặc định) và `gp` (baseline). `gprt` luôn bật Transformer sampling, GP evolution, REINFORCE và elite imitation. Archive seeding là tùy chọn của GPRT, điều khiển bằng `archive_seed_count`; đây không phải một method riêng. Các pipeline standalone và các ablation cũ đã được gỡ khỏi đường chạy.

Dữ liệu phải gồm đúng 56 instance Solomon trong `data/data/Solomon`. Runner chia 32 train, 12 validation và 12 test. Test chỉ được chấm sau khi validation đã khóa heuristic; dùng `skip_test` khi screening.

## Chạy GPRT đầy đủ

Preset chính dùng 32 chu kỳ và 5 seed:

```powershell
python run_gprt_experiment.py --config configs/gprt_main.json
```

Có thể chạy với tham số CLI thay cho preset, ví dụ:

```powershell
python run_gprt_experiment.py --methods gprt --cycles 8 --seeds 20260914 20260915 --screening-run --skip-test
```

Screening chỉ dùng train/validation, tối đa hai seed. Main protocol yêu cầu ít nhất năm seed. Tám chu kỳ phủ đủ tám batch train một lần; cấu hình ít hơn bị từ chối nếu vẫn bật `require_full_train_coverage`.

## Chạy GP thuần trong cùng framework

GP-only dùng chung grammar, evaluator, solver, split và quy tắc validation với GPRT. Nó không tạo mô hình Transformer; 32 lượt ứng viên bổ sung mỗi chu kỳ được dùng như random exploration để cân bằng ngân sách đánh giá.

```powershell
python run_gprt_experiment.py --config configs/gp_main.json
```

Hoặc gọi trực tiếp qua method:

```powershell
python run_gprt_experiment.py --methods gp
```

`run_experiment.py` vẫn là wrapper tương thích cho GP-only. Runner hợp nhất và các file `configs/` là cách chạy được khuyến nghị.

## Archive seeding tùy chọn

Archive seeding thay một số slot random trong population GP bằng cây tốt đã lưu trong archive. Transformer seeding, REINFORCE và elite imitation vẫn hoạt động như trong GPRT đầy đủ. Số slot được cấu hình bằng `archive_seed_count`; mặc định là `0`.

Preset screening dưới đây dùng bốn archive seed, hai seed huấn luyện và bỏ test:

```powershell
python run_gprt_experiment.py --config configs/gprt_archive_seed_8cycle.json
```

Muốn chạy main protocol với archive seeding, đặt `archive_seed_count` qua CLI; ví dụ dùng bốn cây archive:

```powershell
python run_gprt_experiment.py --config configs/gprt_main.json --archive-seed-count 4
```

## Output và kiểm tra kết quả

Mỗi lần chạy ghi `results.json`, `split_manifest.json`, các bảng CSV và thư mục theo method/seed. GPRT còn lưu checkpoint, cấu hình, archive ứng viên, đường học và heuristic tốt nhất trên validation.

```powershell
python validate_gprt_results.py --results-dir outputs/gprt_vrptw/main
```

Validator tiếp tục nhận diện output pilot cũ như dữ liệu legacy; các method cũ trong đó không còn là lựa chọn chạy của runner hiện hành. Tài liệu và số liệu pilot cũ được giữ để tra cứu lịch sử.


