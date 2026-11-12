# Kế hoạch cải tiến GPRT sau thí nghiệm 32 cycle

Ngày: **01/11/2026**. Trạng thái: **kế hoạch đề xuất, chưa triển khai và chưa chạy thí nghiệm mới**.

Căn cứ: [báo cáo 32 cycle](END_TO_END_32CYCLE_RESULTS_VI.md), [số liệu phân tích đã lưu](analysis/end_to_end_32cycle/summary.json), mã nguồn hiện tại và [giới hạn tài nguyên](GPRT_COMPUTE_CONSTRAINTS_VI.md). Mục tiêu là cải thiện chất lượng và hiệu quả chi phí của cơ chế **Transformer + RL + GP**, đồng thời đo được nguyên nhân cải thiện.

## 1. Mục tiêu và nguyên tắc quyết định

Ưu tiên **ổn định chất lượng qua seed**, giữ tính khả thi và giảm số xe trước khi xét giảm quãng đường. Kết quả cũ cho thấy GPRT giảm distance 2,51% trên tĩnh nhưng tăng xe 1,95%; trên động giảm distance 0,40% nhưng phục vụ hơi ít hơn. Thời gian học cao hơn GP 3,17 lần. Do đó không lấy giảm distance đơn lẻ làm tiêu chí thành công.

Giữ nguyên trong nhánh chính:

- Solomon VRPTW và split 32/12/12; cùng feature, grammar, bộ dựng tuyến và checker cho các đối chứng.
- Transformer nhỏ hiện tại: 2 layer, d_model=64, 4 head; population 64, 32 mẫu neural/cycle, 3 thế hệ GP/cycle, batch 4.
- Mô hình cuối vẫn là GPRT sinh biểu thức. GP là đối chứng bắt buộc.
- Chọn bằng validation tĩnh; đánh giá động là chuyển giao zero-shot, không dùng để sửa heuristic.
- Không thêm insertion/local search hoặc tăng model trong đợt đầu: các thay đổi engine sẽ cần một thí nghiệm riêng áp dụng cho cả GP và GPRT.

Các tham số và ngưỡng dưới đây là **lựa chọn thử nghiệm của project**, không phải thông số tối ưu đã chứng minh hoặc công thức nguyên bản của bài báo.

## 2. Chẩn đoán cần kiểm chứng

| Quan sát | Giả thuyết | Phép kiểm tra |
|---|---|---|
| 32 mẫu neural chỉ còn khoảng 7–12 biểu thức duy nhất cuối đợt | Mẫu trùng chiếm nhiều vị trí population, giảm khám phá | Log số mẫu raw/unique, số vị trí GP thực, đa dạng hành vi và kết quả validation |
| Entropy và số mẫu unique đều giảm | Cập nhật có thể tập trung quá sớm | Thử giảm learning rate, giữ các yếu tố khác cố định |
| Có seed tốt và seed rất kém | Heuristic dễ chuyên biệt theo batch hoặc đường tìm kiếm | Kết quả từng seed, từng family, coverage và quá trình screening |
| `archive_seed_count=0` | Chưa tận dụng lại một phần elite giữa các cycle | Thử archive nhỏ nếu hai thay đổi đầu chưa đủ; có GP dùng cùng archive |
| GPRT tốn khoảng 308 s/seed, tổng evaluation gần GP | Chi phí neural, diagnostic, checkpoint hoặc I/O có thể đáng kể | Đo thời gian riêng, kể cả phần ngoài `wall_seconds` của learning curve |
| Tuyến GP/GPRT khác nhau nhưng tương đương biến thiên giữa seed | Đổi công thức chưa chắc đổi cơ chế ưu tiên | Đo đồng ý trên cùng trạng thái, thêm nguồn gốc cây và hậu duệ |

Entropy giảm không tự chứng minh lỗi hay overfit. Reward rank đã chuẩn hóa có trung bình gần 0 theo thiết kế; không lấy đường `reward_mean≈0` làm bằng chứng không học. Gradient nhánh elite trong log nhỏ hơn nhánh policy ở một số lượt; **chưa có căn cứ giảm alpha chỉ vì nghi imitation áp đảo**.

## 3. P0 — Hoàn thiện phép đo trước khi đổi thuật toán

**Ưu tiên cao nhất, dự kiến 4–6 giờ.**

### Dữ liệu và tính tái lập

Ở lần kiểm tra cuối trong lúc lập kế hoạch, thư mục `end_to_end_32cycle` không còn chứa `run_01`. Báo cáo và `docs/analysis/end_to_end_32cycle/` vẫn là căn cứ đã lưu. Trước khi tái kiểm tra hoặc chạy nối pipeline, cần khôi phục/định vị bản raw artifact từ nơi lưu gốc, rồi kiểm hash; không dựng lại dữ liệu chi tiết từ summary. Đây là điều kiện cho phần thực nghiệm, không cản việc lập kế hoạch.

Giữ nguyên đợt cũ. Tạo thư mục mới cho đợt cải tiến, lưu source/config/data hash, môi trường, seed và trạng thái stage. Báo cáo Markdown cũ từng lệch hash phải được ghi nhận, không sửa manifest để hợp thức hóa.

### Đo chi phí đúng phạm vi

1. Tách `generate`, GP, screening, validation, diagnostic, backward/optimizer, checkpoint, ghi output và tổng end-to-end. `evaluator_seconds` nằm trong các pha gọi evaluator; không cộng nó lần nữa vào tổng thời gian.
2. Timer cycle hiện ghi trước `save_checkpoint`; bổ sung thời gian lưu checkpoint để tránh bỏ sót I/O. Chỉ giảm tần suất checkpoint nếu profile chứng minh đây là chi phí đáng kể; giữ đầy đủ trạng thái resume tại mốc được lưu.
3. Đo cả cold start và warm-up riêng. Sau warm-up, chạy 5 lần mỗi heuristic/instance trên cùng máy, đổi thứ tự policy bằng RNG riêng; báo trung vị, phân vị và khoảng dao động. Đo callback simulator riêng với thời gian giải tĩnh.
4. Chạy mỗi training seed trong tiến trình riêng để đo peak RSS và tránh bộ nhớ còn lại từ seed trước. Ghi phần cứng, phiên bản thư viện và thread thực.
5. Ghi evaluation thực theo train/screening/validation/diagnostic, cache hit và số request. Diagnostic thưa hơn phải dùng RNG riêng và không tác động selection, cache search hoặc lịch ngẫu nhiên của thuật toán; nếu chưa tách được, giữ lịch cũ và chỉ đo.

### Theo dõi đóng góp neural

Gắn ID cho cá thể: neural/random/archive; phép lai/đột biến và parent ID. Lưu cả nguồn khởi tạo và hậu duệ, không chỉ số biểu thức cuối trùng mẫu neural. Dùng tập trạng thái probe cố định từ **train**, phủ 6 family; trên cùng tập ứng viên khả thi đo tỷ lệ chọn cùng khách và tương quan thứ hạng. Probe được chọn trước khi screening, không sử dụng test.

**Điều kiện qua P0:** khi chỉ thêm log, biểu thức và fitness phải giữ nguyên trên seed kiểm tra xác định; tổng thời gian đo có thể giải thích được; bộ kiểm tra nghiệm vẫn đạt. Kiểm thử tập trung vào RNG, cache, counters, resume và tính hợp lệ; không viết test chỉ để xác nhận tên trường log.

## 4. P1 — Hai cải tiến nhỏ, kiểm tra riêng

### A. Loại trùng khi tạo population GP

Hiện trainer đưa cả 32 mẫu neural vào các vị trí GP, kể cả mẫu trùng. Đề xuất:

- Vẫn sinh **đúng 32 mẫu raw** và giữ nguyên toàn bộ mẫu cùng log-probability cho REINFORCE.
- Riêng nhánh khởi tạo GP: giữ các biểu thức neural canonical khác nhau, sau đó điền đủ population 64 bằng random exploration. Giới hạn số lần thử cây random để bảo đảm kết thúc; ghi fallback/trùng còn lại.
- Không lấy mẫu neural lặp đến khi đủ 32 biểu thức unique; việc này sẽ đổi ngân sách và phân phối lấy mẫu.
- Không lọc trùng mẫu khỏi policy-gradient loss. Dùng các mẫu đã lọc như mẫu on-policy không điều chỉnh sẽ thay đổi estimator.
- GP đối chứng dùng cùng quy tắc loại trùng population ban đầu và cùng giới hạn thử random; ghi rõ thay đổi này so với baseline lịch sử.

**Giả thuyết:** tăng số ứng viên khác nhau trong cùng population giúp GP tìm tốt hơn, đặc biệt khi neural phân phối tập trung. Chi phí evaluation có thể tăng do ít cache hit; phải báo cáo, không mặc định đây là tối ưu tốc độ.

**Chỉ số chính:** validation theo thứ tự mục tiêu; số unique GP ban đầu, đa dạng hành vi, evaluation thực và thời gian. Unique tăng mà chất lượng không tăng thì chưa đủ để giữ thay đổi.

### B. Giảm tốc độ cập nhật Transformer

Thử `learning_rate=3e-4` thay `1e-3`; giữ `alpha=0,1`, `beta=0,001`, temperature=1 và các cấu hình khác. Chỉ đổi một yếu tố so với GPRT mốc.

**Giả thuyết:** cập nhật chậm hơn có thể giảm tập trung sớm và giảm seed thất bại. Cần đọc cả chất lượng và entropy; entropy cao hơn mà heuristic kém hơn không được tính là thành công.

**Chỉ số:** số biểu thức unique và entropy theo cycle, neural-vs-random trên train, chất lượng elite, gradient mỗi nhánh và validation. Kiểm tra ở 8 cycle chỉ dùng loại phương án lỗi hoặc kém rõ; cần mốc 16 cycle trước khi loại phương án có vẻ học chậm.

Không ghép A+B ngay từ đầu. Nếu cả hai có tín hiệu riêng, tổ hợp chỉ được chọn sau khi có một screening riêng; không suy ra tổ hợp tốt từ kết quả của từng thành phần.

## 5. P2 — Một phương án dự phòng, chọn theo chẩn đoán

Không chạy đồng thời mọi hướng. Sau P1, chỉ chọn **một** trong các nhánh sau, tối đa một vòng screening bổ sung:

| Điều kiện quan sát | Nhánh thử | Kiểm soát |
|---|---|---|
| A và B đều có ích riêng | A+B | So với từng thành phần trên cùng seed; cần log ngân sách |
| Mất kiến thức giữa batch dù neural đa dạng hơn | Archive seeding 4 cây | 4 cây đã screening thay cây random; population vẫn 64; GP có cùng archive count |
| LR thấp không khắc phục tập trung sớm | Thử beta=0,01, giữ LR mốc | Thay riêng entropy weight; không ghép thêm temperature/alpha |

Archive seeding đã có hàm chọn và preset screening lịch sử, nhưng runner GP hiện chưa có đối chứng archive tương ứng; phải bổ sung đối chứng trước khi dùng kết quả để quy công cho Transformer. Cây cũ phải chấm lại trên batch mới; không so trực tiếp fitness của hai batch khác nhau. Không đưa cây archive/GP vào loss REINFORCE như mẫu neural on-policy.

Các hướng này là giả thuyết triển khai. Công trình [Mundhenk và cộng sự, NeurIPS 2021](https://arxiv.org/abs/2111.00053) mô tả neural seeding cho GP khởi động lại và nghiên cứu cách phối hợp hai thành phần. Đây là cơ sở tham khảo cho cơ chế lai, **không chứng minh archive seeding hoặc các hyperparameter trên cải thiện Solomon VRPTW**.

## 6. Ma trận thí nghiệm và quy tắc chọn

### Screening chỉ train/validation

| Bước | Cycle | Seed | Cấu hình | Số lượt train |
|---|---:|---:|---|---:|
| S1 | 8 | 20260914, 20260915 | GP cùng quy tắc population, GPRT mốc, A, B | 8 |
| S2 | 16 | Hai seed trên | GP, GPRT mốc, 1–2 ứng viên giữ lại | 6–8 |
| Dự phòng, nếu cần | 8 rồi quyết định có tiếp không | Hai seed trên | Một nhánh P2; archive phải thêm GP-archive | Tối đa 4 lượt ở mốc 8 |

Mỗi cấu hình dùng cùng danh sách instance/batch theo seed, cùng validation interval và giới hạn ứng viên validation. Không screening trên 4 cycle vì chưa phủ đủ 32 train instance. Không quét tích số LR × beta × alpha × archive × kiến trúc.

**Quy tắc chọn ứng viên:** so fitness validation trung bình theo seed với thứ tự ưu tiên đã chốt: bất khả thi → chưa phục vụ → xe → distance chuẩn hóa → độ phức tạp. Trình bày đồng thời từng seed/family và seed kém nhất; chỉ giữ một ứng viên cuối nếu trung bình tốt hơn mốc và không kém mốc theo mục tiêu ở cả hai seed screening. Nếu chỉ thắng nhờ một seed hoặc chưa phân biệt được, ghi chưa rõ và không tự nhận cải tiến.

Ứng viên phải không phát sinh lỗi checker, coverage đủ, không rò rỉ dữ liệu. Với chất lượng hòa theo độ chính xác báo cáo đã định, ưu tiên chi phí thấp hơn. Screening hai seed là công cụ chọn cấu hình, không phải kết luận chính nhiều seed. Không đổi ngưỡng hoặc tiêu chí sau khi xem test.

### Xác nhận 32 cycle

Sau khi khóa một cấu hình mới, chạy **GP + GPRT mốc + GPRT cải tiến**, mỗi cấu hình 32 cycle × 5 seed 20260914–20260918: **15 lượt train**. Chạy lại các mốc trong cùng môi trường để so thời gian; bản cũ giữ làm lịch sử. Nếu không có ứng viên qua screening, dừng nhánh chất lượng, giữ kết quả âm và chỉ hoàn tất cải thiện phép đo/chi phí.

Phải định danh rõ cấu hình bằng `variant_id` và hash. Runner end-to-end hiện chỉ xử lý nhãn GP/GPRT; cần mở rộng manifest và so sánh theo variant, không ghi cả mốc và bản mới vào cùng đường dẫn `gprt/<seed>`.

Khóa 15 heuristic trước mọi đánh giá test. Nếu đánh giá đủ cả ba nhóm và EDD, ma trận mới là:

- Validation simulator: 12 × (2 động + 1 static) × 16 policy = **576 episode**.
- Test tĩnh: 12 × 16 = **192 lượt**.
- Test simulator: 12 × (3 động + 1 static) × 16 = **768 episode**.

Các số này là yêu cầu cho runner mở rộng, không phải đầu ra đã có. Không dùng validation động để chọn bản A/B hoặc đổi công thức release. Test Solomon cũ đã được xem; kết quả mới là đánh giá phát triển có kiểm soát, không gọi là xác nhận hoàn toàn chưa từng thấy. Nếu cần tuyên bố tổng quát hóa mạnh hơn, phải khóa một bộ chưa xem theo quy mô/phân phối và nguồn dữ liệu riêng trước khi chạy; không chỉ đổi tên split của các instance đã xem.

## 7. Công bằng ngân sách và cách kết luận

### Phép so sánh bắt buộc

Đợt đầu giữ **32 cycle** để đối chiếu thí nghiệm cũ. Báo cả số evaluation thực và thời gian. Chỉ gọi là so sánh cùng cycle; không gọi cùng ngân sách chính xác nếu counters còn khác.

Lưu chất lượng incumbent theo validation cùng chi phí lũy kế. Incumbent được chọn theo lịch validation cố định, không chọn lại bằng test. Vẽ chất lượng theo evaluation và theo thời gian để thấy GPRT có cải thiện hiệu quả tìm kiếm hay chỉ tiêu tốn thêm tài nguyên.

### Bổ sung nếu muốn kết luận hiệu quả ngân sách

Cần một lượt đối chứng có bộ dừng theo ngân sách. Dùng screening P0 để chốt trước trần evaluation chung và trần thời gian chung cho **hai phép thử riêng**; không đòi cùng lúc bằng cả hai. Tính chi phí train, screening và lựa chọn validation vào ngân sách; diagnostic tách riêng nhưng vẫn báo tổng end-to-end. Dự trữ ngân sách cho lần chọn heuristic cuối và ghi mức vượt do một đơn vị công việc không thể chia nhỏ. So các incumbent hợp lệ đã phủ đủ train; nếu độ lệch trần không đạt dung sai chốt trước, không nhận là matched-budget.

Phần này cần runner hỗ trợ; không thể chứng minh bằng cách chỉ chia distance cho số giây từ các run cũ. Nếu thiếu thời gian, bỏ kết luận matched-budget thay vì thêm một phép so sánh chưa đúng.

### Báo cáo chất lượng

- Tĩnh: khả thi, phục vụ, xe, distance, thắng/thua/hòa theo đúng mục tiêu; từng seed và family.
- Động: mức phục vụ và hủy trước, rồi distance, distance/served và độ trễ. So chi phí cùng tập ID phục vụ/hủy là phân tích bổ sung, không bỏ các episode khác khỏi bảng chính.
- Neural: nguồn gốc/hậu duệ, số unique, đa dạng hành vi và kết quả trên cùng trạng thái; không đặt “heuristic càng khác GP càng tốt” làm mục tiêu.
- Bất định: thống kê theo seed và instance; nếu dựng khoảng tin cậy, dùng lấy mẫu lại theo cả instance và training seed, giữ các scenario cùng instance trong nhóm. Chỉ 5 seed/12 instance nên vẫn nêu giới hạn. Không coi 180 dòng là 180 quan sát độc lập.

Chỉ gọi **cải thiện chất lượng tĩnh** khi mục tiêu ưu tiên được cải thiện; gọi **cải thiện hiệu quả chi phí** khi đạt chất lượng tương đương/tốt hơn với chi phí thấp hơn theo phép so phù hợp. Cải thiện động phải đọc cùng mức phục vụ. Kết quả không thắng GP vẫn có giá trị nếu chỉ ra được vai trò và giới hạn của cơ chế lai.

## 8. Thứ tự triển khai, công sức và đầu ra

Khung **2 tuần, khoảng 24–30 giờ làm việc**, phù hợp 15–20 giờ/tuần; thời gian máy chạy tính riêng.

| Thứ tự | Công việc | Công sức dự kiến | Đầu ra |
|---|---|---:|---|
| 1 | P0: artifact, timer, counters, nguồn gốc cây, benchmark runtime | 4–6 giờ | Log có thể kiểm tra; baseline phép đo |
| 2 | P1-A và P1-B, cấu hình riêng, kiểm thử phần sampling/GP | 5–6 giờ | Hai biến thể nhỏ, có thể tắt bằng config |
| 3 | S1/S2, phân tích validation, khóa hoặc loại ứng viên | 4–5 giờ | Bảng lựa chọn và lý do; không dùng test |
| 4 | Variant-aware manifest/runner; xác nhận 32 cycle | 5–6 giờ | 15 heuristic khóa, counters, gate đầy đủ |
| 5 | Phân tích tĩnh/động, heuristic và cập nhật báo cáo | 4–5 giờ | Báo cáo có đối chứng và giới hạn |
| Dự phòng | Một nhánh P2 hoặc đối chứng ngân sách | Trong phần thời gian còn lại; giảm phạm vi nếu vượt | Chỉ thực hiện sau các bước bắt buộc |

Tham chiếu thời gian cũ: GP khoảng 1,62 phút/seed và GPRT 5,14 phút/seed; 15 lượt xác nhận riêng phần học khoảng **60 phút** nếu môi trường và chi phí giữ gần như cũ. Đây chỉ là ước lượng định cỡ, không cam kết, vì A có thể tăng cache miss và I/O/máy khác nhau. Sau P0 phải đo lại trước khi chạy toàn ma trận.

Files dự kiến cần thay đổi khi bắt đầu thực thi:

- `gprt_vrptw/config.py`, `trainer.py`, `gp.py`: cờ loại trùng, loss giữ mẫu raw, tracing, timer.
- `run_gprt_experiment.py`: đối chứng GP, tiến trình riêng, variant và profiling.
- `run_end_to_end.py`: quản lý nhiều variant, khóa heuristic và đếm ma trận mới.
- `configs/`: các preset S1/S2 và main đã khóa; không ghi đè cấu hình lịch sử.
- `tests/`: kiểm tra các ranh giới quan trọng của on-policy sampling, population, RNG, budget và manifest.
- `docs/`: cập nhật bảng kết quả, so sánh heuristic và nhật ký quyết định.

**Việc bắt đầu:** hoàn thành P0, rồi thử **A: loại trùng population** và **B: LR=3e-4** riêng biệt. Chưa tăng lên 64/128 cycle, chưa tăng kích thước Transformer. Chỉ tăng ngân sách khi đường validation cho thấy còn cải thiện và đã loại được chi phí không cần thiết.
