# Kết quả thí nghiệm end-to-end GP và GPRT: 32 cycle

Kế hoạch tiếp theo: [cải tiến GPRT sau 32 cycle](GPRT_POST_32CYCLE_IMPROVEMENT_PLAN_VI.md).

Ngày phân tích: **01/11/2026**. Nguồn: [`end_to_end_32cycle/run_01`](../end_to_end_32cycle/run_01). Đợt chạy được ghi nhận ngày **30/10/2026, giờ Việt Nam/Thái Lan** (manifest lưu UTC ngày 29/10). Tài liệu này báo cáo kết quả đã chạy, bổ sung cho [protocol end-to-end](END_TO_END_GPRT_GP_VI.md).

## 1. Kết luận chính

**GPRT chưa cho thấy ưu thế tổng thể so với GP trong đợt 32 cycle này.** Trên test tĩnh, cả hai phục vụ đủ khách và tạo nghiệm hợp lệ. GPRT giảm quãng đường trung bình **2,51%**, nhưng tăng số xe **1,95%**. Vì fitness ưu tiên số xe trước quãng đường, không thể dùng riêng mức giảm khoảng cách để kết luận GPRT tốt hơn. Trên 60 cặp cùng seed và instance, GPRT thắng/thua/hòa theo thứ tự mục tiêu là **26/30/4**.

Trên mô phỏng động, GPRT giảm khoảng cách trung bình **0,40%**, đồng thời phục vụ ít hơn **0,067 khách/episode**. Chi phí học offline của GPRT cao hơn **3,17 lần**; trung bình peak RSS cao hơn **1,95 lần**. Chưa thấy lợi ích chất lượng ổn định tương xứng với chi phí thêm này.

**Heuristic có khác biệt rõ về biểu thức và tuyến cụ thể, nhưng chưa hình thành hai nhóm chiến lược tách biệt rõ theo phương pháp.** GPRT thường chọn cây ngắn hơn, song vẫn dùng các tín hiệu quen thuộc về thời gian đến, thời gian chờ, hạn phục vụ và khả năng nối khách tiếp theo. Mức khác nhau giữa GP và GPRT tương đương biến thiên giữa các seed trong mỗi phương pháp. Độ ngắn của cây là chỉ báo độ phức tạp, chưa phải bằng chứng tính mới hoặc khả năng tổng quát hóa tốt hơn.

## 2. Dữ liệu và thiết kế thực nghiệm

| Thành phần | Cấu hình/kết quả thực tế |
|---|---|
| Benchmark | Solomon VRPTW, 56 instance, mỗi instance 100 khách |
| Split theo instance gốc | 32 train / 12 validation / 12 test |
| Phương pháp học | GP; GPRT = Transformer + GP + cập nhật RL và elite imitation |
| Training seed | 20260914–20260918, 5 seed/phương pháp |
| Cycle | 32/seed; cả 10 lượt hoàn tất, phủ đủ 32 train instance |
| Population / mẫu Transformer | 64 / 32 |
| Thế hệ GP / cycle | 3 |
| Batch train / validation interval | 4 instance / mỗi 4 cycle |
| Transformer | 2 layer, d_model=64, 4 head, feedforward=128, dropout=0,1 |
| Cập nhật | learning rate=0,001; alpha elite=0,1; beta entropy=0,001; advantage chuẩn hóa theo batch |
| Backend | CPU, Numba, 1 thread; Linux; PyTorch 2.11.0+cpu |
| Chọn heuristic | Bằng validation tĩnh; khóa 10 biểu thức trước validation động và test |
| Test trong training | Tắt |
| Validation simulator | 396 episode = 12 × (2 động + 1 tĩnh) × 11 policy |
| Test benchmark tĩnh | 132 lượt = 12 × (5 GP + 5 GPRT + 1 EDD) |
| Test simulator | 528 episode = 12 × (3 động + 1 tĩnh) × 11 policy |

EDD là quy tắc ưu tiên hạn phục vụ sớm, không cần huấn luyện. EDD có 12 lượt test tĩnh và 36 lượt test động; GP/GPRT có lần lượt 60 và 180 lượt mỗi phương pháp. Không nhân EDD thành năm mô hình độc lập.

Fitness được so theo thứ tự: bất khả thi → tỷ lệ chưa phục vụ → số xe → khoảng cách chuẩn hóa → độ phức tạp biểu thức. GP và GPRT dùng chung grammar, feature, bộ dựng tuyến và bộ kiểm tra. GP có random exploration để đối ứng với các ứng viên neural, nhưng cùng số cycle không bảo đảm cùng số đánh giá thực hay cùng thời gian.

Kịch bản động chọn 70/100 đơn, đặt `release=max(0, ready_time-lead)` với lead nguyên 10–15; 30 đơn còn lại xuất hiện từ đầu. Một số đơn trong nhóm được chọn vẫn có release=0, nên tỷ lệ thực sự xuất hiện sau thời điểm 0 không nhất thiết là 70%. Chọn độc lập 5 đơn làm ứng viên yêu cầu hủy, tại `release+delay`, delay nguyên 1–15. Chỉ đơn chưa gán mới được hủy. Test dùng scenario seed 20261025–20261027.

Benchmark tĩnh giới hạn một tuyến/xe. Simulator cho phép xe về depot chạy nhiều chuyến; tuyến đã cam kết được giữ đến khi xe về. Khoảng cách Euclid không làm tròn, travel time bằng distance; hạn khách là hạn **bắt đầu** dịch vụ. Các con số quãng đường trong tài liệu theo đơn vị tọa độ benchmark, không tự quy đổi thành km.

## 3. Kiểm tra tính đầy đủ và tính hợp lệ

- Pipeline manifest ghi tất cả stage `complete`.
- Training validator ghi `valid`, test đã tắt trong học; cả 10 summary xác nhận 32 cycle và coverage 100%. Với GPRT, mỗi seed có 32 bước gradient và 32 dòng learning curve.
- Gate test tĩnh đạt: **132 dòng, 0 lỗi**.
- Validator simulator validation đạt **396 episode**; test đạt **528 episode**; không có failure, không ghi nhận source thay đổi tại lần kiểm tra đã lưu.
- Các episode test simulator ghi **0 vi phạm**, gồm tải, cửa sổ thời gian và các kiểm tra khác; không khách đã phục vụ nào bị trễ.
- Phân tích này kiểm tra lại SHA-256 của **62 artifact** được liệt kê trong các stage: **61 khớp**, gồm dữ liệu kết quả, log kiểm tra và cả 10 heuristic. Riêng [`end_to_end_report.md`](../end_to_end_32cycle/run_01/end_to_end_report.md) **không khớp hash trong manifest**. Chưa xác định nguyên nhân; không dùng file Markdown này làm nguồn tính số liệu. Giữ nguyên file và manifest để bảo toàn dấu vết.

Việc kiểm hash xác nhận các artifact kết quả hiện tại khớp bản ghi đã lưu, không thay thế một lần chạy lại mô phỏng độc lập. Tài liệu này đọc lại dữ liệu, đối chiếu summary và tính thêm các phép so sánh; không huấn luyện lại và không thay đổi heuristic.

Đối chiếu mã nguồn đang có với hash nguồn trong manifest: `gprt_vrptw/trainer.py` khác hash byte gốc nhưng **khớp sau chuẩn hóa CRLF thành LF**; các file nguồn còn lại khớp trực tiếp. Không thấy khác biệt nội dung nguồn sau chuẩn hóa xuống dòng. Việc này cho phép đối chiếu ý nghĩa metric với implementation hiện tại, nhưng không thay thế việc tái dựng môi trường chạy.

## 4. Hiệu năng chạy

### 4.1. Chi phí tìm heuristic offline

Trung bình ± độ lệch chuẩn mẫu trên **5 training seed**:

| Chỉ số / seed | GP | GPRT | GPRT / GP |
|---|---:|---:|---:|
| Thời gian end-to-end học (s) | 97,12 ± 1,15 | 308,20 ± 9,11 | 3,17× |
| Peak RSS (MB, nhãn log) | 642,22 ± 32,76 | 1.253,54 ± 61,42 | 1,95× |
| Tổng instance evaluation thực | 30.729,8 | 31.824,0 | 1,036× |
| Train instance evaluation | 29.953,4 | 26.072,8 | 0,870× |
| Screening instance evaluation | 279,6 | 1.625,0 | 5,812× |
| Validation instance evaluation | 496,8 | 492,0 | 0,990× |
| Diagnostic instance evaluation | 0 | 3.634,2 | — |
| Fitness request | 11.520 | 12.800 | 1,111× |

GPRT mất thêm khoảng **211,08 s/seed**. Tổng thời gian nội bộ 5 lượt GP khoảng **8,09 phút**, GPRT **25,68 phút**. Stage train theo đồng hồ pipeline là **33,87 phút**; validation simulator **111,27 s**, test tĩnh **1,20 s**, test simulator **179,76 s**. Tổng thời lượng các stage khoảng **38,78 phút**. Thời gian pipeline chứa các bước ngoài phép dựng tuyến; không cộng các cột này lần nữa vào chi phí từng seed.

Số đánh giá tổng gần nhau nhưng phân bố khác: GPRT có nhiều screening và diagnostic hơn, ít train evaluation thực hơn do cache và nội dung ứng viên. Vì vậy đây là so sánh cùng cấu hình cycle/population, **chưa phải thí nghiệm khớp chính xác ngân sách đánh giá hoặc thời gian**. Diagnostic cũng tiêu tốn chi phí dù không trực tiếp chọn nghiệm. Peak RSS là giá trị đo theo tiến trình; việc tăng dần qua seed và trạng thái bộ nhớ có thể ảnh hưởng số đo, nên không coi tỷ số này là bộ nhớ riêng của kiến trúc trong mọi môi trường.

### 4.2. Thời gian áp dụng heuristic

Transformer hoạt động khi tìm heuristic offline. Khi giải instance mới, hệ thống chấm điểm bằng biểu thức đã khóa; không gọi Transformer để sinh trực tiếp tuyến.

| Phép đo | EDD | GP | GPRT |
|---|---:|---:|---:|
| Benchmark tĩnh: runtime TB (ms/lượt) | 10,470 | 8,731 | 1,319 |
| Benchmark tĩnh: trung vị runtime (ms/lượt) | xem dữ liệu gốc | 1,743 | 0,728 |
| Simulator static: TB p95 callback (ms) | 17,698 | 62,608 | 59,551 |
| Simulator dynamic: TB p95 callback (ms) | 4,632 | 4,836 | 4,895 |
| Simulator dynamic: wall-clock TB (s/episode) | 0,267 | 0,259 | 0,284 |

**Không diễn giải 8,731/1,319 thành GPRT nhanh hơn GP 6,6 lần một cách ổn định.** Lượt GP đầu tiên, seed 20260914 trên C105, mất **409,648 ms**, chiếm khoảng 78% tổng runtime GP trên test tĩnh. Runner đo luôn lần gọi đầu vào evaluator; backend dùng Numba. Đây phù hợp với ảnh hưởng khởi động/JIT, nhưng log không tách riêng thời gian biên dịch nên chỉ có thể nêu như một nguyên nhân có khả năng. Trung vị cho thấy chênh lệch nhỏ hơn nhiều. Nếu loại cùng instance C105 ở mọi seed của cả hai phương pháp, trung bình còn **1,959 ms GP** và **1,310 ms GPRT**; đây chỉ là kiểm tra độ nhạy sau chạy, không phải benchmark đã warm-up.

Trong simulator động, GPRT có p95 callback cao hơn GP khoảng **1,21%**, wall-clock episode cao hơn khoảng **9,28%**. Không có bằng chứng GPRT nhanh hơn ở khâu dispatch động. Callback gồm đọc trạng thái, snapshot, lọc khả thi, tính feature/look-ahead, chấm điểm và dựng quyết định. **TB p95 từng episode không phải p95 chung của toàn bộ callback**. Static control thường xử lý cả tập khách trong một callback; không so trực tiếp độ trễ này với callback động có workload nhỏ hơn.

## 5. Chất lượng lời giải tĩnh

### 5.1. Trung bình trên test

| Policy | Số lượt | Nghiệm khả thi | Khách phục vụ / 100 | Xe | Quãng đường |
|---|---:|---:|---:|---:|---:|
| EDD | 12 | 100% | 100 | 11,000 | 2.407,35 |
| GP | 60 | 100% | 100 | 9,417 | 1.561,11 |
| GPRT | 60 | 100% | 100 | 9,600 | 1.521,95 |

So với EDD, GP giảm quãng đường **35,15%**, giảm xe **14,39%**; GPRT giảm quãng đường **36,78%**, giảm xe **12,73%**. Cả hai phương pháp học có ích so với baseline này. So với GP, GPRT giảm **39,16 đơn vị quãng đường**, nhưng thêm **0,183 xe/instance** trung bình.

So cặp cùng training seed và instance, GPRT có **30 lượt khoảng cách ngắn hơn, 26 lượt dài hơn, 4 lượt bằng nhau**. Khi ưu tiên số xe trước khoảng cách, GPRT **thắng 26, thua 30, hòa 4**. Hai cách đếm cho kết luận khác nhau, thể hiện đánh đổi giữa xe và quãng đường.

### 5.2. Biến thiên theo training seed

| Seed | Xe GP | Xe GPRT | Distance GP | Distance GPRT | GPRT thắng/thua/hòa theo mục tiêu |
|---|---:|---:|---:|---:|---:|
| 20260914 | 9,583 | 9,500 | 1.845,68 | 1.332,85 | 8 / 4 / 0 |
| 20260915 | 9,500 | 9,583 | 1.572,60 | 1.529,06 | 5 / 4 / 3 |
| 20260916 | 9,583 | 9,250 | 1.330,43 | 1.421,17 | 7 / 5 / 0 |
| 20260917 | 9,083 | 9,500 | 1.583,01 | 1.401,98 | 5 / 6 / 1 |
| 20260918 | 9,333 | 10,167 | 1.473,82 | 1.924,70 | 1 / 11 / 0 |

GPRT có khoảng cách ngắn hơn ở 3/5 seed, nhưng seed 20260918 kém rõ trên cả xe lẫn distance. GP cũng biến thiên lớn: distance trung bình theo seed từ **1.330,43 đến 1.845,68**, GPRT từ **1.332,85 đến 1.924,70**. Không chọn seed tốt nhất bằng test; cả năm heuristic đã khóa đều được giữ trong thống kê.

### 5.3. Theo họ instance

Mỗi họ có 2 test instance × 5 seed = 10 lượt/phương pháp.

| Họ | Xe GP | Xe GPRT | Distance GP | Distance GPRT |
|---|---:|---:|---:|---:|
| C1 | 10,1 | 10,4 | 1.103,64 | 1.117,68 |
| C2 | 3,1 | 3,3 | 679,07 | 702,31 |
| R1 | 18,6 | 18,0 | 2.012,69 | 1.865,91 |
| R2 | 3,4 | 3,8 | 1.598,79 | 1.583,12 |
| RC1 | 17,2 | 17,3 | 2.138,36 | 2.127,26 |
| RC2 | 4,1 | 4,8 | 1.834,10 | 1.735,44 |

Lợi ích rõ nhất nằm ở **R1**, nơi GPRT giảm cả xe và distance. Trên C1/C2, cả hai chỉ số đều kém GP. R2 và RC2 minh họa đánh đổi: khoảng cách giảm nhưng xe tăng. Hai instance mỗi họ là quá ít để khái quát ưu thế theo phân phối.

Static control trong simulator cho cùng trung bình distance/xe với benchmark tĩnh trong đợt này. Đây là quan sát về kết quả, không làm hai bài toán trở thành tương đương: simulator vẫn cho phép multi-trip. Không dùng kết quả multi-trip để tính gap với BKS single-trip. Đợt dữ liệu này cũng chưa cung cấp đối chiếu BKS đã khớp đầy đủ quy ước, nên không báo gap tối ưu.

## 6. Chất lượng trong môi trường động

Các giá trị là trung bình episode; mỗi episode có 100 yêu cầu ban đầu.

| Chỉ số | EDD (36 episode) | GP (180) | GPRT (180) |
|---|---:|---:|---:|
| Khách phục vụ | 91,861 | 91,861 | 91,794 |
| Chưa phục vụ | 7,861 | 7,889 | 7,956 |
| Hủy thành công | 0,278 | 0,250 | 0,250 |
| Tỷ lệ phục vụ trên đơn không hủy | 92,074% | 92,047% | 91,980% |
| Quãng đường | 4.123,23 | 3.662,97 | 3.648,19 |
| Quãng đường / khách phục vụ, TB episode | 44,319 | 39,643 | 39,455 |
| Số xe sử dụng | 17,194 | 16,717 | 16,883 |
| Số chuyến | 51,139 | 50,494 | 50,661 |
| Khách phục vụ trễ | 0 | 0 | 0 |

Tỷ lệ phục vụ dùng mẫu số `requested-cancelled`, còn số khách phục vụ chia 100 dùng mẫu số toàn bộ yêu cầu; không trộn hai chỉ số. Vì không có phục vụ trễ, tỷ lệ đúng hạn trên đơn không hủy bằng tỷ lệ phục vụ. Nếu chỉ tính trong nhóm đã phục vụ, đúng hạn là 100%; điều đó không xóa vấn đề khoảng 8 khách chưa phục vụ mỗi episode.

GPRT giảm distance so với GP **14,78 đơn vị/episode (0,40%)**, nhưng giảm phục vụ **0,067 khách**, tương đương giảm khoảng **0,067 điểm phần trăm** trên mẫu số không hủy. Số chuyến và số xe cũng hơi cao hơn GP. So với EDD, GP/GPRT giảm distance khoảng **11,16%/11,52%** với mức phục vụ gần nhau.

### So cặp cùng workload

Đối chiếu cùng training seed, instance và scenario seed cho **180 cặp GP–GPRT**:

- Số khách phục vụ: GPRT nhiều hơn **10 cặp**, ít hơn **18**, bằng nhau **152**.
- Chỉ **146/180 cặp** có cùng tập ID khách đã phục vụ và cùng tập ID hủy. Bằng số lượng khách không bảo đảm cùng workload.
- Trong 146 cặp cùng workload, distance GPRT thấp hơn **93**, cao hơn **51**, bằng nhau trong sai số 1e-8 ở **2** cặp.
- Chênh lệch distance GPRT−GP trên nhóm này: trung bình **−10,94**, trung vị **−58,72**, độ lệch chuẩn mẫu **293,67**. Có cặp GPRT dài hơn tới **996,48**, nên số lần thắng nhiều không đồng nghĩa mức tiết kiệm trung bình lớn.

Như vậy, mức giảm distance không hoàn toàn chỉ do phục vụ ít khách hơn, nhưng vẫn **nhỏ và không đồng đều**. Nhóm cùng workload là tập con xác định sau khi quan sát đầu ra, dùng mô tả hành vi, không thay thế đánh giá trên toàn bộ episode.

Trung bình có **5,417 đơn/episode bất khả thi ngay lúc release**, giống nhau ở cả ba policy. Đây là một phần nguyên nhân mức phục vụ không đạt 100%, song không giải thích toàn bộ đơn bỏ sót. Tỷ lệ 5% là **ứng viên yêu cầu hủy**; chỉ khoảng 0,25–0,28 đơn/episode thực sự hủy vì đa số đã gán. Kết quả phản ánh đúng luật hủy này, không phải môi trường có 5% đơn bị hủy thành công.

## 7. Heuristic GP và GPRT khác nhau đến đâu?

### 7.1. Cấu trúc biểu thức

Đếm node trên cây biểu thức, mỗi toán tử/feature/hằng là một node; không đếm dấu ngoặc, dấu phẩy hoặc BOS/EOS. Độ sâu leaf=0.

| Seed | Node GP | Node GPRT | Đặc điểm GPRT |
|---|---:|---:|---|
| 20260914 | 29 | 5 | Khoảng cách + hạn phục vụ + thời gian chờ |
| 20260915 | 7 | 5 | Thời gian đến − 2 × khả năng nối tiếp |
| 20260916 | 13 | 3 | Thời điểm bắt đầu phục vụ sớm nhất |
| 20260917 | 33 | 5 | Thời gian chờ + khoảng cách × hạn phục vụ |
| 20260918 | 5 | 40 | Cây nhiều nhánh, so sánh, logic và chia bảo vệ |
| **Trung bình** | **17,4** | **11,6** | GPRT ít hơn 33,33% node |
| **Trung vị** | **13** | **5** | 4/5 heuristic GPRT chỉ có 3–5 node |

Cả **10 biểu thức khác nhau về chuỗi canonical**; không có cặp GP/GPRT giống hệt. GPRT đơn giản hơn ở 4/5 seed, ngoại lệ seed cuối có cây 40 node và cũng là seed có kết quả test kém. Với chỉ năm seed, không thể suy ra quan hệ nhân quả giữa độ dài cây và chất lượng.

Các feature đã chuẩn hóa theo horizon, thang khoảng cách hoặc capacity trước khi chấm điểm. Bộ dựng tuyến lọc khả thi trước, rồi chọn khách có **điểm nhỏ nhất**, hòa theo customer ID. Diễn giải dưới đây áp dụng cho feature chuẩn hóa; không cộng trực tiếp các đại lượng thô khác đơn vị.

### 7.2. Ý nghĩa của các heuristic ngắn

Đặt `A=arrival_time`, `W=wait_time`, `D=d_current`, `T=due_date`, `R=ready_time`, `F=next_feasible_fraction`, `S=service_time`.

| Phương pháp / seed | Công thức | Diễn giải khi tối thiểu hóa điểm |
|---|---|---|
| GPRT / 20260914 | `D + T + W` | Ưu tiên khách gần, hạn sớm và ít chờ |
| GPRT / 20260915 | `A - 2F` | Đến sớm và để lại nhiều khả năng phục vụ khách kế tiếp |
| GPRT / 20260916 | `max(R, A)` | Ưu tiên thời điểm bắt đầu dịch vụ sớm nhất |
| GPRT / 20260917 | `W + D*T` | Kết hợp chờ với tương tác khoảng cách và hạn phục vụ |
| GP / 20260915 | `A + pdiv(W-F, S)` | Cùng dùng arrival/look-ahead, thêm chờ và thang service time |
| GP / 20260918 | `A + W - F` | Thời điểm bắt đầu dịch vụ trừ khả năng nối tiếp |

Do `W=max(0,R-A)`, có đẳng thức `A+W=max(R,A)`. Vì thế GP seed 20260918 tương ứng với heuristic GPRT seed 20260916 **bổ sung phần thưởng look-ahead `−F`**. GPRT seed 20260915 cũng thuộc cùng nhóm ý tưởng arrival/look-ahead. Đây là bằng chứng hai phương pháp chia sẻ các nguyên tắc ưu tiên, dù cây và hệ số khác nhau.

GP seed 20260916 có thể viết gọn thành `max(W, route_load) + max(W,D) + T`: mẫu số của phép chia là `ge(A,route_load)`, chỉ nhận 0/1; chia bảo vệ trả lại tử khi mẫu bằng 0 và chia 1 cũng giữ tử. Đây gần nhóm khoảng cách/hạn/chờ của GPRT seed 20260914, nhưng không bảo đảm cùng thứ tự khách. Các cây GP dài còn lại và GPRT seed cuối có nhiều nhánh logic; không nên diễn giải chỉ bằng cách nhìn tên feature.

`pdiv(x,y)` trả `x` nếu `|y|<1e-9`, ngược lại trả `x/y`; `and/or` là logic trả 0/1; `if_else` kiểm tra điều kiện khác 0. Evaluator còn chặn giá trị vào khoảng hữu hạn. Khi rút gọn các cây phức tạp phải giữ đúng các quy tắc này.

### 7.3. Khác biệt về hành vi tạo tuyến

Để tránh đánh đồng chuỗi biểu thức với hành vi, phân tích dùng các tuyến test đã lưu:

1. **Trùng nghiệm:** so toàn bộ chuỗi khách trong mỗi tuyến, bỏ qua thứ tự liệt kê các xe nhưng giữ hướng đi.
2. **Độ giống cạnh:** với tập cạnh có hướng gồm cạnh depot, tính Jaccard `|E_GP ∩ E_GPRT| / |E_GP ∪ E_GPRT|`. Giá trị 1 là cùng tập cạnh, 0 là không chung cạnh. Đây là thước đo nghiệm cuối, không phải mức đồng ý trên từng quyết định ở cùng trạng thái.

| Nhóm so sánh | Số cặp | Trùng nghiệm | Jaccard cạnh TB |
|---|---:|---:|---:|
| GP–GPRT, cùng seed và instance | 60 | 4 (6,67%) | 0,391 |
| GP–GP, mọi cặp seed khác nhau trên cùng instance | 120 | 6 (5,00%) | 0,381 |
| GPRT–GPRT, mọi cặp seed khác nhau trên cùng instance | 120 | 3 (2,50%) | 0,363 |

GP và GPRT tạo tuyến khác nhau ở **56/60 cặp**, nên không thể nói hai hệ thống chỉ đổi cách viết của cùng heuristic. Tuy nhiên Jaccard GP–GPRT thậm chí hơi cao hơn mức giống nhau giữa các seed trong từng phương pháp. **Khác biệt giữa phương pháp không lớn hơn rõ rệt khác biệt do seed** theo phép đo này.

Seed giống nhau chỉ tạo cách ghép cặp tái lập; GP và GPRT tiêu thụ số ngẫu nhiên và đi qua không gian tìm kiếm khác nhau, không phải hai cây có chung lịch sử tiến hóa. Jaccard là mô tả trên 12 test instance và không đủ để chứng minh tương đương hoặc tách biệt về chiến lược trên mọi trạng thái.

### 7.4. Dấu hiệu Transformer có học, và giới hạn diễn giải

Learning curve cho thấy số biểu thức neural khác nhau trong mỗi batch 32 mẫu giảm. Trung bình 8 cycle đầu → 8 cycle cuối theo từng seed:

| Seed | Số mẫu neural duy nhất | Entropy |
|---|---:|---:|
| 20260914 | 21,00 → 12,38 | 2,469 → 2,050 |
| 20260915 | 23,25 → 7,50 | 2,491 → 1,873 |
| 20260916 | 20,13 → 6,75 | 2,423 → 1,802 |
| 20260917 | 20,00 → 7,50 | 2,438 → 1,890 |
| 20260918 | 23,00 → 8,00 | 2,447 → 1,825 |

Phân phối sinh đang tập trung hơn, đồng thời xuất hiện nguy cơ mất đa dạng. Tỷ lệ quần thể cuối có biểu thức trùng trực tiếp mẫu neural ở 8 cycle cuối chỉ khoảng **2,34–4,10%**, tùy seed; đây là tỷ lệ trong quần thể cuối, **không phải xác suất sống sót trên số mẫu neural đầu vào**. Metric không tính được mọi hậu duệ sau crossover/mutation, nên không dùng nó để kết luận đóng góp neural bằng không.

Trường `neural_better_than_random_rate` không đo tỷ lệ heuristic cuối thắng GP: implementation so fitness của các mẫu neural on-policy với trung vị ứng viên random trên cùng batch train; phần đánh giá random được ghi vào chi phí diagnostic. Chỉ số này không chứng minh cải thiện trên test. Do không có truy vết đầy đủ nguồn gốc của từng cây cuối và chưa có ablation RL/PQT/Transformer tách riêng trong folder này, chưa thể khẳng định từng heuristic được Transformer trực tiếp phát minh, hoặc quy phần cải thiện cho riêng RL.

## 8. Giới hạn và cách sử dụng kết quả trong khóa luận

- **Đơn vị độc lập:** 60 lượt tĩnh không phải 60 instance độc lập; 180 episode động dùng lại 12 instance, 5 model seed và 3 scenario. Những phép đếm thắng/thua và độ lệch chuẩn trong tài liệu mang tính mô tả. Không báo p-value hoặc ưu thế thống kê từ việc coi tất cả dòng độc lập.
- **Test đã có lịch sử sử dụng:** 12 test instance từng xuất hiện trong pilot cũ. Đợt này khóa heuristic trước test, nhưng không phải một bộ xác nhận hoàn toàn chưa từng thấy.
- **Ngân sách chưa khớp tuyệt đối:** cần thêm đối chứng theo cùng thời gian hoặc cùng số instance evaluation thực để đánh giá hiệu quả chi phí.
- **Tĩnh → động:** mô phỏng dùng sự kiện tổng hợp từ Solomon, không phải dữ liệu vận hành thật; không suy ra năng lực xử lý bất định ở cảng hoặc ngoài thực tế.
- **Thời gian chạy:** thiếu đo warm-up/lặp độc lập và thông tin CPU cụ thể để suy rộng tốc độ; so trong môi trường đã lưu là phù hợp hơn khẳng định tốc độ tuyệt đối.
- **Heuristic:** cần ghi mức đồng ý trên cùng tập trạng thái ứng viên, nguồn gốc cây và ablation để phân biệt tác dụng Transformer/RL với tác dụng GP. Jaccard tuyến chỉ là phép đo hành vi đầu ra hiện có.

Kết luận có thể dùng trong báo cáo: **Pipeline GPRT hoàn tất 32 cycle, sinh heuristic dùng được trên test tĩnh và simulator động, giữ nghiệm hợp lệ và giảm chi phí so với EDD. So với GP, GPRT chưa chứng minh ưu thế tổng thể: tiết kiệm distance nhỏ, đánh đổi số xe hoặc mức phục vụ, chi phí học cao hơn và biến thiên theo seed lớn. Các heuristic khác nhau ở cấu trúc và nghiệm cuối nhưng vẫn chia sẻ nguyên tắc thời gian/khoảng cách/look-ahead; chưa có bằng chứng chúng tạo hai lớp chiến lược tách biệt theo phương pháp.**

## 9. Nguồn số liệu và tái lập phân tích

| Nội dung | Nguồn |
|---|---|
| Stage, cấu hình, hash | [`pipeline_manifest.json`](../end_to_end_32cycle/run_01/pipeline_manifest.json), [`training_config.json`](../end_to_end_32cycle/run_01/training_config.json) |
| Biểu thức và hash | [`heuristics_manifest.json`](../end_to_end_32cycle/run_01/heuristics_manifest.json) và 10 file `final_heuristic.txt` |
| Chi phí offline | [`search_metrics.csv`](../end_to_end_32cycle/run_01/training/search_metrics.csv), summary từng seed |
| Nghiệm tĩnh và runtime | [`test_results.csv`](../end_to_end_32cycle/run_01/static_test/test_results.csv) |
| Episode động/tĩnh đối chứng | [`episodes.csv`](../end_to_end_32cycle/run_01/dynamic/final/episodes.csv) |
| Kiểm tra hợp lệ | `training/validation_report.json`, `static_test/gate_report.json`, `dynamic/{pilot,final}/{gate_report,validation_report}.json` trong thư mục run |
| Thống kê bổ sung, cây và learning curve | [`summary.json`](analysis/end_to_end_32cycle/summary.json) |
| Từng cặp test tĩnh | [`paired_static.csv`](analysis/end_to_end_32cycle/paired_static.csv) |
| Từng cặp test động | [`paired_dynamic.csv`](analysis/end_to_end_32cycle/paired_dynamic.csv) |

Chạy từ thư mục gốc project để tạo lại số liệu phân tích, không chạy training:

```powershell
.venv/Scripts/python.exe scripts/analyze_end_to_end_32cycle.py
```

Script chỉ đọc artifact của run và ghi ba file trong `docs/analysis/end_to_end_32cycle/`. Tài liệu Markdown là phần diễn giải kết quả; không được script tự ghi đè. Công thức tỷ lệ thay đổi dùng `(GPRT−GP)/GP`; phần trăm tiết kiệm đổi dấu để thuận đọc. Bảng làm tròn sau khi tổng hợp từ số gốc.
