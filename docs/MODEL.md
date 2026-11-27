# Mô hình và quy ước hiện hành

## Cách GP và GPRT tìm heuristic

Một heuristic là cây biểu thức được biểu diễn bằng token prefix. Grammar ràng buộc cấu trúc, độ sâu và số token; phép chia có bảo vệ. Các terminal và toán tử được định nghĩa tại `gprt_vrptw/grammar.py`. Heuristic chấm điểm các quyết định khả thi do bộ dựng tuyến đề xuất; bộ dựng tuyến và kiểm tra ràng buộc thuộc môi trường, không thuộc Transformer.

Trong mỗi cycle, GPRT lấy mẫu biểu thức từ Transformer causal theo grammar, kết hợp chúng với cây ngẫu nhiên và các cây seed được cấu hình, rồi tiến hóa quần thể bằng GP. Fitness của các mẫu Transformer được xếp hạng để tạo reward; REINFORCE dùng log-xác suất của chính các mẫu on-policy đó. Elite từ GP được sàng lọc và truyền vào một loss imitation riêng, không coi chúng là mẫu on-policy. Loss kết hợp policy loss, imitation và entropy. Archive seeding là tùy chọn, `archive_seed_count=0` trong preset Solomon chính.

GP thuần dùng cùng grammar, evaluator và quy tắc chọn trên validation; các slot tương ứng với mẫu Transformer được dành cho khám phá ngẫu nhiên. Cùng cấu hình số slot/cycle không có nghĩa là cùng thời gian chạy hay cùng số lượt mô phỏng thực sau cache. Log phân biệt request, cache hit và số lần đánh giá instance thực tế.

## Solomon tĩnh

Bộ dữ liệu gồm 56 instance Solomon, chia theo instance thành 32 train, 12 validation và 12 test. Train tìm biểu thức; validation chọn biểu thức; test dùng để đánh giá sau khi chọn. Fitness so theo thứ tự: số instance bất khả thi, tỷ lệ khách chưa phục vụ, số xe trung bình, khoảng cách chuẩn hóa và độ phức tạp cây. Khoảng cách Euclid không làm tròn; hạn khách là hạn **bắt đầu** phục vụ. Bộ dựng và kiểm tra dùng giới hạn xe, tải trọng, cửa sổ thời gian và hạn depot của instance.

Preset `configs/gprt_main.json` và `configs/gp_main.json`: 32 cycle, 5 seed, population 64, 3 thế hệ GP/cycle và batch 4. GPRT có 32 mẫu Transformer/cycle. Những thông số này là lựa chọn triển khai của dự án. `configs/gprt_archive_seed_8cycle.json` là preset screening hai seed, không chạy test. Tham số còn lại xem `GPRTConfig`.

## Solomon động

Luồng này **không huấn luyện lại** trong simulator: nó nạp heuristic Solomon đã chọn bằng validation và đánh giá cùng EDD, tùy chọn greedy, trên cùng kịch bản. Theo `configs/dvrpsim_dynamic.json`, 70% đơn được chọn làm ứng viên động với `release = max(0, ready_time - lead)`, lead nguyên từ 10 đến 15; tỷ lệ thực sự xuất hiện sau thời điểm 0 có thể thấp hơn 70%. 5% đơn là ứng viên yêu cầu hủy, không phải tỷ lệ hủy thành công; chỉ đơn chưa được gán mới hủy được. Kịch bản được materialize theo seed để tái lập.

Dispatch khi có release, hủy thành công hoặc xe về depot. Chuyến đã cam kết giữ nguyên đến khi về; xe có thể tiếp tục chuyến khác. Vì vậy không đồng nhất kết quả multi-trip với bài toán Solomon tĩnh một tuyến/xe. Đường đi Euclid không làm tròn, thời gian đi bằng khoảng cách và hạn khách là hạn bắt đầu phục vụ. Các chỉ số và event được lưu để kiểm tra độc lập.

## Xe tải–drone động

Đây là luồng riêng, huấn luyện GP/GPRT trực tiếp trên episode từ CSV. Split theo nhóm tọa độ của bộ 100 khách: một nhóm test, nhóm tiếp theo validation, các nhóm còn lại train; 200/400 khách dùng đánh giá chuyển quy mô. `--fleet` là số xe bắt buộc. Fitness ưu tiên lần lượt tỷ lệ chưa giao, `(km truck + hệ số × km drone) / số khách`, số xe và độ phức tạp; ràng buộc cứng được kiểm tra trước dispatch. Khác Solomon, hạn khách trong luồng này là hạn **hoàn tất** phục vụ. Chi tiết giả định, cách khóa policy và kết quả: [DVRPSIM_DRONE_VI.md](DVRPSIM_DRONE_VI.md).

## Diễn giải

Tách chi phí tìm heuristic offline khỏi thời gian áp dụng online; chọn mô hình bằng validation và báo test theo toàn bộ seed đã định. Không suy ra tối ưu, hiệu quả vận hành thực hay khả năng tổng quát hóa ngoài các điều kiện đã đo. Chỉ so BKS khi khớp định nghĩa bài toán, giới hạn xe, mục tiêu và quy tắc khoảng cách.
