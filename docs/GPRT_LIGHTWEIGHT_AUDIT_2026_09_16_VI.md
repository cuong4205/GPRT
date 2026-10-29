# Rà soát bổ sung GPRT: hiệu quả trên laptop/Colab miễn phí
> **Tài liệu audit lịch sử:** một số grammar/feature switch được phân tích ở đây đã bị gỡ khỏi runner hiện hành. Xem [hướng dẫn chạy hiện hành](RUN_GPRT_VRPTW.md).


Ngày 2026-09-16. Đọc lại mã hiện tại; không dùng pilot cũ để suy ra chất lượng bản mới. Archive seeding đã có trong mã, mặc định count=0 và preset thử count=4; không còn coi đây là chức năng chưa triển khai. Chưa sửa mã mô hình hoặc chạy huấn luyện mới trong lần rà soát này.

## 1. Ưu tiên cao nhất về tốc độ: chỉ tính feature cây thực sự dùng

`gprt_vrptw/evaluator.py:53,171`: mỗi lần chấm ứng viên đều dựng toàn bộ vector feature. Khi bật look-ahead, ngay cả cây `d_current` cũng duyệt khách chưa phục vụ để tính next-feasible. Ngay cả khi tắt look-ahead, phép tính `urgency_rank` vẫn chạy trước khi giá trị bị thay bằng 0.

Một heuristic chỉ dùng khoảng cách không cần các phép tính này. Với n khách, việc duyệt các khách còn lại bên trong mỗi lần chấm ứng viên có thể đưa phần tính feature từ bậc hai lên bậc ba theo n ở trường hợp xấu nhất của một lần dựng nghiệm.

### Phép đo nhỏ đã chạy

Chọn một instance train mỗi họ bằng `stratified_sample(train, 6)`, dùng cùng solver và quy tắc hòa, biểu thức `d_current`. Ba lần mỗi chế độ, lấy trung vị giây; không qua fitness cache. Chế độ `only_used` là callback trực tiếp `distance(current,candidate)/distance_scale(instance)`, chưa phải một evaluator lazy tổng quát đã triển khai.

| Instance | Tính đủ, look-ahead bật | Look-ahead tắt | Chỉ feature cần dùng |
|---|---:|---:|---:|
| C101 | 0,327009 | 0,097779 | 0,034696 |
| C202 | 0,443403 | 0,056777 | 0,016840 |
| R102 | 0,280588 | 0,087889 | 0,032511 |
| R201 | 0,234543 | 0,071222 | 0,026899 |
| RC103 | 0,255039 | 0,033880 | 0,012557 |
| RC201 | 0,127172 | 0,028277 | 0,010437 |

Cả ba chế độ tạo đúng cùng các tuyến trên cả sáu instance. Callback chỉ tính feature cần dùng nhanh hơn khoảng 8,6–26,3 lần trong phép đo này. Thứ tự chế độ cố định, số lần đo nhỏ; không suy ra toàn bộ huấn luyện sẽ nhanh lên cùng tỷ lệ, và chưa đo cây thực sự dùng look-ahead.

**Kiểm tra lại sau câu hỏi về độ chắc chắn:** warm-up từng chế độ, chạy 7 lần/instance và đảo ngẫu nhiên thứ tự `full`/`only_used` với seed 42. Tỷ lệ trung vị lần lượt: C101 9,22; C202 10,18; R102 9,22; R201 8,39; RC103 12,22; RC201 8,40. Tuyến vẫn giống nhau ở mọi lần chạy. Khoảng 8,4–12,2 lần là số đo kiểm tra lại đáng tham khảo hơn khoảng ban đầu; biến động cho thấy không nên xem mức 26,3 lần là kỳ vọng ổn định. Cả hai lỗi typed grammar/rút gọn cũng được tái hiện lại độc lập trong lần kiểm tra này.

**Thiết kế đề nghị:** phân tích tập terminal một lần khi compile cây; tính feature theo nhu cầu, bỏ hoàn toàn urgency/next-feasible khi cây không dùng. Chuyển `tuple(sorted(unserved))` trong `solver.py` ra ngoài vòng lặp ứng viên; tính các đại lượng chung cho trạng thái một lần. Không cache xuyên trạng thái nếu thiếu current time, tải, vị trí và tập khách còn lại trong khóa. Kiểm chứng cùng điểm/tuyến cho cây số học, điều kiện và look-ahead trên train.

## 2. Typed grammar chưa bảo đảm hoàn tất cây

**Đã tái hiện:** `Grammar(63,6,typed=True)` chấp nhận tiền tố `add add add add add if_else`, sau đó trả `valid_tokens=[]` dù cây chưa hoàn chỉnh. `if_else` ở độ sâu 5 yêu cầu boolean ở độ sâu 6; boolean chỉ sinh được bằng toán tử, nhưng mọi toán tử tại độ sâu 6 bị cấm. Trường hợp nhỏ nhất là max_depth=1 và tiền tố `if_else`.

**Nguyên nhân:** mask xét arity, số token và độ sâu hiện tại, nhưng chưa xét độ sâu tối thiểu để hoàn tất từng kiểu đối số đang chờ.

**Đề nghị:** dùng bảng chi phí tối thiểu về token/độ sâu theo kiểu number/boolean khi kiểm tra một token. Đây là sửa tính hợp lệ, không cần thêm tham số neural. Chưa bật typed grammar cho thí nghiệm dài trước khi kiểm tra mọi tiền tố nhỏ có đường hoàn tất.

## 3. Rút gọn và mutation chưa đồng nhất với typed grammar

**Đã tái hiện:** cây

```text
if_else(and(le(d_current,d_current),le(demand,remaining_capacity)),d_current,due_date)
```

hợp lệ trong typed grammar. `simplify_tree` đổi `le(d_current,d_current)` thành số `1.0`; kết quả không còn hợp lệ vì `and` yêu cầu boolean. `deduplicate_fill` có thể đưa cây rút gọn vào quần thể mà không kiểm tra lại kiểu.

Ngoài ra, `mutate` luôn gọi `grammar.random_tree` với kiểu gốc number. Khi chọn vị trí boolean, phép thay có thể bị từ chối nhiều lần thay vì sinh đúng kiểu cần thiết.

**Đề nghị:** rút gọn có nhận biết kiểu; hoặc giữ nguyên cây khi kết quả rút gọn không hợp lệ. Sinh subtree theo kiểu và phần ngân sách độ sâu/token còn lại ở vị trí mutation; crossover chọn vị trí cùng kiểu. Đo tỷ lệ mutation/crossover tạo thay đổi hợp lệ thay vì chỉ tỷ lệ được yêu cầu.

Hai vấn đề mục 2–3 ảnh hưởng nhánh `typed_grammar=True`; mặc định hiện tại là False, nên không được quy lỗi này cho kết quả mặc định.

## 4. Imitation đang ưu tiên cây dài qua số token

`transformer.py:166–167`: elite NLL là tổng NLL của tất cả token chia tổng số token. Do đó một elite dài có trọng lượng lớn hơn một elite ngắn trong trung bình token, dù chất lượng hai cây tương tự. Đây là lựa chọn mục tiêu, không phải lỗi toán học.

**Đề nghị thử nhẹ:** tính trung bình NLL theo chiều dài từng cây, sau đó trung bình qua elite. Có thể bổ sung trọng số chất lượng sau khi đã đánh giá trên cùng screening, nhưng thử từng yếu tố. Không cần thêm đánh giá VRPTW nếu tái sử dụng elite đang có. Không áp dụng chuẩn hóa chiều dài này một cách máy móc cho log-probability REINFORCE, vì sẽ đổi mục tiêu của nhánh đó.

## 5. Khử trùng biến reproduction thành bổ sung ngẫu nhiên

`gp.py:58`: mỗi thế hệ khử trùng rồi lấp đầy bằng cây mới. Reproduction sao chép parent, nhưng nhiều bản sao sẽ bị loại; phần chỗ trống được thay bằng random. Vì vậy reproduction_rate=0,1 không có nghĩa 10% thế hệ sau là các bản sao giữ nguyên, và tỷ lệ cây ngẫu nhiên thực tế có thể cao hơn tỷ lệ khởi tạo.

**Đề nghị:** ghi số cây trùng bị loại, số random refill, số offspring hợp lệ/thực sự mới. Thử giới hạn refill hoặc tạo offspring thay cho refill với số lần thử hữu hạn; giữ tổng population/fitness budget. Chưa khẳng định khử trùng hiện tại gây giảm chất lượng: đây là cơ chế cần đo để hiểu đóng góp GP–neural.

## 6. Giảm bộ nhớ neural và chi phí diagnostic khi có bằng chứng cần thiết

Nhánh GPRT sinh tự hồi quy với gradient, mỗi bước forward lại toàn bộ tiền tố và giữ graph đến sau khi GP kết thúc. Nhánh Transformer+RL đã sinh no-grad rồi tính likelihood bằng teacher forcing.

Có thể thử cùng cơ chế hai lượt cho GPRT để giảm bộ nhớ: sampling no-grad, teacher forcing trước optimizer step, cùng trọng số, temperature và grammar mask; dùng dropout=0 hoặc eval mode ở cả hai lượt để log-probability khớp (eval mode vẫn cho tính gradient). Không tự coi việc tính lại với dropout khác là on-policy tương đương. Cần đo RAM và thời gian vì thêm lượt forward, không bảo đảm tăng tốc trên CPU.

Gradient norm hai nhánh hiện cần hai lần `autograd.grad` trước `backward`; có thể đo mỗi vài chu kỳ thay vì mọi chu kỳ sau giai đoạn debug. Không bỏ loss hoặc thay gradient để tiết kiệm diagnostic.

## 7. Thời gian log chưa gồm ghi checkpoint

`trainer.py:528,536`: `wall_seconds` được lấy trước khi lưu checkpoint. Checkpoint còn chứa toàn bộ evaluator cache gồm Solution/routes; archive/cache tăng thì thời gian I/O và dung lượng có thể tăng. Chưa đo mức độ trên run dài.

**Đề nghị:** ghi `checkpoint_seconds` và tổng thời gian thật, tách checkpoint resume đầy đủ khỏi artifact heuristic gọn. Giữ checkpoint định kỳ phù hợp Colab; không loại RNG, optimizer, archive hoặc trạng thái cần resume chỉ để làm file nhỏ. Nếu bỏ cache phải ghi rõ ảnh hưởng đến thời gian và thống kê khi resume.

## Phương án nên làm trước

1. Feature tính theo nhu cầu + gom tính toán chung từng trạng thái: có bằng chứng chi phí và không tăng mạng.
2. Sửa tính hợp lệ typed grammar trước khi đánh giá ablation đó.
3. Thử NLL trung bình theo cây, giữ nguyên dữ liệu elite và cấu hình khác.
4. Chỉ sau profiling mới tối ưu graph sampling, diagnostic và checkpoint.

Tất cả thay đổi evaluator/grammar phải áp dụng nhất quán cho GP và các đối chứng. Tuning bằng train/validation, không mở rộng cấu hình hoặc số seed trước khi ước lượng thời gian thực tế.

