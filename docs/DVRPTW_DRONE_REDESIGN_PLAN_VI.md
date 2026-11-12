# Kế hoạch chuyển GPRT sang DVRPTW xe tải–drone

Ngày 09/11/2026. Tài liệu ghi lại khảo sát và phương án ban đầu.
[Quy tắc đã triển khai và cách chạy](DVRPSIM_DRONE_VI.md) cập nhật hai quyết định:
GP và GPRT chấm trực tiếp bằng DVRPSim, và toàn bộ chuyến được khóa từ lúc xe xuất phát.

## Kết quả kiểm tra dữ liệu

Đã đọc toàn bộ 176 CSV. Mỗi file có một dòng depot đầu tiên (demand và time bằng 0), sau đó là đúng N dòng khách, tổng cộng 41.600 dòng khách. Tám cột thống nhất: x, y, demand, open, close, servicetime, drone_serve, time. Các tên thuộc sáu họ C1, C2, R1, R2, RC1, RC2. Không có ID khách trong CSV, vì vậy loader cần gán ID theo thứ tự dòng trong từng file.

| Khách/file | File | Khách có time > 0 | Khách có drone_serve = 1 | Nhóm trùng bộ tọa độ |
|---:|---:|---:|---:|---|
| 100 | 56 | 54,57% (17–88% mỗi file) | 34,50% | 4 nhóm (9, 8, 23, 16 file) |
| 200 | 60 | 63,48% (16,5–91%) | 26,05% | 4 nhóm (10, 10, 20, 20 file) |
| 400 | 60 | 68,08% (20,5–90,25%) | 22,12% | 4 nhóm (10, 10, 20, 20 file) |

- Mọi dòng khách đều thỏa time ≤ open ≤ close. Loader công khai truyền time vào trường arrival và dùng drone_serve làm cờ đủ điều kiện. Sự kiện release lấy trực tiếp từ CSV, kể cả các đơn có time = 0. [Loader nguồn](https://raw.githubusercontent.com/friedeggs01/DVRPTW-drone/main/data/read_data.py), [mô hình đơn hàng](https://raw.githubusercontent.com/friedeggs01/DVRPTW-drone/main/graph/requests.py).
- Mọi dòng được đánh dấu drone_serve = 1 có demand ≤ 10 kg. Có 3.031 dòng khách tải ≤ 10 kg nhưng cờ bằng 0. Kiểm tra cờ và tải là hai điều kiện riêng.
- CSV ghi servicetime = 5 cho tất cả các dòng, kể cả depot. File cấu hình ghi truck 10 phút, drone 5 phút và ServiceWindow = 5 phút. Cần xác minh vai trò của cả ba trước khi tính khả thi.
- Cấu hình ghi tốc độ truck 50 và drone 80, tải 1.300 và 10 kg, endurance 30 phút, phóng và thu hồi mỗi bước 1 phút. Cần khóa cách đổi tọa độ sang thời gian, cách xử lý khoảng cách/làm tròn và những khoản thời gian tính vào endurance.
- Bộ cấu hình công khai khớp từng dòng với file cục bộ và các dòng đầu của h100c101.csv cũng khớp. Bài của Tran và cộng sự công bố kho dữ liệu và mã nguồn có liên quan. Cần đối chiếu hash và phiên bản CSV trước khi gắn nguồn chính xác cho toàn bộ thư mục. [Bài báo](https://link.springer.com/article/10.1007/s12293-026-00523-4), [cấu hình công khai](https://raw.githubusercontent.com/friedeggs01/DVRPTW-drone/main/data/dvrptw-setting.txt).
- Phép kiểm tra sơ bộ cho chính sách xuất phát từ depot sau release, dùng khoảng cách Euclid, tốc độ truck 50/60 đơn vị/phút và phục vụ 10 phút, cho 0/7/169 dòng khách ở ba quy mô mà chuyến truck đi riêng trượt hạn khách hoặc hạn depot. Đây là phép kiểm tra chính sách depot dưới giả định trên. Xe đã ở ngoài tuyến có thể có vị trí tốt hơn.

## Thiết kế đích

Câu hỏi chính: GPRT có học được quy tắc xếp hạng phương án chèn đơn vào lịch xe–drone tốt hơn GP thuần với cùng ngân sách mô phỏng, và mức hiệu quả thay đổi thế nào ở bố cục và quy mô mới?

Luồng xử lý: CSV → lịch công bố đơn → trạng thái nhìn thấy tại thời điểm quyết định → sinh phương án chèn khả thi → heuristic chấm điểm → cập nhật lịch → bộ kiểm tra độc lập phát lại lịch → fitness cho GP/GPRT.

| Thành phần | Thay đổi dự kiến |
|---|---|
| Dữ liệu | Loader riêng, manifest gồm ID, size, họ, hash file, hash bố cục, tỷ lệ đơn động và cấu hình. |
| Mô phỏng | Tái sử dụng ý tưởng sự kiện, nhật ký và trạng thái chỉ chứa đơn đã công bố từ dvrpsim_dynamic. Thêm trạng thái drone, tải hàng đi cùng xe, lịch phóng/thu và đồng bộ với truck. |
| Điều phối | Xét lại lịch xe tại depot theo mốc công bố, mốc xuất phát và lúc xe về depot. Khóa toàn bộ chuyến khi xe xuất phát. Hạn chế danh sách ứng viên bằng cùng một quy tắc cho các mô hình. |
| Phạm vi drone ban đầu | Đề xuất một drone mỗi xe, mỗi sortie phục vụ một khách, phóng/thu tại các điểm dừng và không có sortie chồng lấn. Áp dụng theo đặc tả nguồn nếu xác minh được. Các giới hạn khác biệt cần được ghi là biến thể thí nghiệm. |
| Bộ kiểm tra | Tính lại thời điểm phục vụ, tải, đúng một lần phục vụ, depot horizon, endurance, phóng/thu, thời điểm gặp xe và phần lịch đã khóa. Tách đơn bị từ chối/không phục vụ khỏi vi phạm lịch. |
| Học heuristic | Giữ cây GP, bộ sinh biểu thức Transformer, REINFORCE cho mẫu neural on-policy và elite imitation riêng. Thêm feature như slack sau khi chèn, chi phí tăng thêm, tuổi đơn, drone còn rảnh, biên endurance và chờ rendezvous. Phiên bản hóa vocabulary và checkpoint. |

## Đối chứng và protocol

- **Bố cục mới ở 100 khách.** Đề xuất luân phiên giữ ngoài một trong bốn nhóm tọa độ. Trong mỗi lượt, tách tiếp một nhóm validation và dùng hai nhóm còn lại để train. Cố định vòng train/validation/test trước khi chạy. Gộp R1/R2 và RC1/RC2 theo tọa độ thực tế. Báo cáo từng nhóm và trung bình theo nhóm vì có bốn bố cục duy nhất, với số file chênh lệch.
- **Chuyển quy mô.** Giữ nguyên các heuristic đã chọn chỉ bằng dữ liệu 100 khách rồi đánh giá ở 200 và 400. Báo cáo đây là chuyển quy mô kèm thay đổi phân phối vì tỷ lệ release dương và khả năng phục vụ bằng drone cũng dịch chuyển. Để tách riêng tác động của quy mô cần thí nghiệm kiểm soát bổ sung.
- **Đối chứng.** Dùng luật ưu tiên hạn gần (EDD) và chèn tham lam, GP-only và GPRT trên cùng simulator, thông tin, tập hành động và tie-break. So sánh truck-only với truck–drone bằng cùng chính sách để đo đóng góp của drone. So sánh không sửa tuyến với rolling horizon để đo đóng góp của điều phối lại, nếu ngân sách cho phép.
- **Chỉ số.** Đề xuất xếp hạng ưu tiên đơn giao đúng hạn, sau đó chi phí vận hành theo đặc tả đã khóa. Công bố tỷ lệ giao đúng hạn trên tổng số đơn, số đơn bị từ chối, km truck và drone riêng, số xe, số chuyến drone, thời gian chờ đồng bộ, độ trễ quyết định p50/p95 và thời gian train. Dùng cùng số lượt chấm biểu thức và episode cho GP/GPRT, đồng thời báo cáo thời gian thực và cache hit. Báo chênh lệch ghép cặp theo instance, nhóm bố cục và seed huấn luyện.
- **Đối chiếu với bài báo.** Mã nguồn tham chiếu có ba cây chính sách nhận đơn, xếp thứ tự và chọn xe, cập nhật theo bước thời gian và trả chi phí phát thải cùng số đơn từ chối. Bản thiết kế này dùng một biểu thức chấm phương án chèn và ra quyết định theo sự kiện. Chỉ so sánh trực tiếp với số liệu bài báo nếu tái lập cùng protocol. [Evaluator nguồn](https://raw.githubusercontent.com/friedeggs01/DVRPTW-drone/main/deployment/evaluation.py).
- **Bước khóa quy tắc.** Xác minh nguồn và đơn vị, fleet hữu hạn hay theo nhu cầu, đơn tải lên truck lúc nào, quy tắc nhận/từ chối đơn, time window xét bắt đầu hay hoàn tất phục vụ, chờ/thu hồi tính vào endurance thế nào, mục tiêu và hệ số chi phí. Ghi rõ phần nào bám nguồn và phần nào là lựa chọn của khóa luận.

## Lộ trình đề xuất

| Tuần | Sản phẩm và điều kiện qua mốc |
|---|---|
| 1 | Đặc tả có nguồn, loader và manifest. Đọc đủ 176 file, phát hiện trường lỗi, khóa split theo nhóm. |
| 2–3 | Simulator truck-only đọc release thật. Kiểm thử tay đơn xuất hiện giữa tuyến, đơn trễ, tải, depot và tính lặp lại. |
| 4–5 | Sortie truck–drone và bộ kiểm tra độc lập. Kiểm thử drone đến trước/sau truck, sát giới hạn endurance, trùng phục vụ, cờ drone bằng 0, và bất biến kết quả trước khi một đơn mới được công bố. |
| 6 | EDD và greedy trên dữ liệu phát triển. Đo thời gian mỗi episode ở ba quy mô, chốt ngân sách theo số liệu và kiểm tra baseline thực sự dùng được drone. |
| 7–8 | Nối GP/GPRT với fitness mới, chạy pilot trên nhóm train/validation. So sánh bằng ngân sách chung, kiểm tra replay và quy trình chọn heuristic. |
| 9–10 | Khóa cấu hình và chạy nhiều seed, đánh giá theo nhóm bố cục và chuyển quy mô. |
| 11–12 | Đối chứng truck-only, phân tích sai lệch và độ trễ, tổng hợp báo cáo. |

Rủi ro chính là số phương án chèn và cặp điểm phóng/thu tăng nhanh. Đo trước khi mở rộng, giới hạn tập ứng viên theo quy tắc dùng chung và ưu tiên thí nghiệm GPRT trên 100 khách. Dùng 200/400 cho suy luận đã khóa để kiểm tra chuyển quy mô với chi phí thấp hơn huấn luyện lại.
