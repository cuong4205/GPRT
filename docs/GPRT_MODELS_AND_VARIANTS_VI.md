# GP và GPRT trong project

## Phạm vi tài liệu

Tài liệu này mô tả cách các mô hình đang được cài đặt trong project Solomon VRPTW, tập trung vào đường chạy hiện tại: `run_gprt_experiment.py` và package `gprt_vrptw/`. Đây là bản triển khai chuyển thể cho VRPTW, không khẳng định là tái lập nguyên trạng mọi chi tiết trong bài báo GPRT gốc. Các kích thước mạng và ngân sách dưới đây là cấu hình của project.

Cần phân biệt ba mức:

1. **Solver vận hành:** dựng tuyến VRPTW và chọn khách hàng tiếp theo.
2. **Heuristic:** biểu thức GP chấm điểm từng khách hàng khả thi.
3. **Bộ học heuristic:** GP-only baseline hoặc GPRT đầy đủ tìm ra biểu thức sẽ dùng ở tầng 2.

Trong GPRT, Transformer không sinh trực tiếp thứ tự khách hàng của một instance. Nó sinh chương trình heuristic; GP tiến hóa chương trình đó; REINFORCE cập nhật Transformer từ các mẫu neural và elite imitation truyền tín hiệu từ cây GP tốt. Solver dùng heuristic đã chọn để dựng tuyến. RL và Transformer chỉ tồn tại như các thành phần phối hợp trong GPRT, không có runner độc lập.

## 1. Bài toán và bộ dựng tuyến dùng chung

### Dữ liệu

Project dùng 56 instance Solomon VRPTW, tách cố định thành 32 train, 12 validation và 12 test. Mỗi khách hàng có tọa độ, nhu cầu, thời điểm sẵn sàng, hạn phục vụ và thời gian phục vụ. Instance cũng có depot, sức chứa xe và giới hạn số xe.

Các phương pháp GP, Transformer+RL và GPRT dùng chung solver xây dựng tuyến trong `vrptw_baselines/solver.py`. Tại mỗi bước, solver tạo tập khách hàng đang chưa phục vụ và khả thi theo tải trọng, thời gian đến, cửa sổ thời gian và khả năng quay về depot. Phương pháp chọn một khách hàng trong tập này; solver cập nhật tải, vị trí và thời gian, rồi tiếp tục nối khách hàng vào cuối tuyến. Khi không còn khách hàng khả thi trên tuyến, solver đóng tuyến và mở tuyến mới nếu còn xe.

Với GP hoặc cây do Transformer sinh:

~~~text
khách hàng được chọn = argmin(priority(instance, trạng thái, khách hàng))
tie-break = customer_id nhỏ hơn
~~~

Điểm ưu tiên thấp hơn được chọn trước. Solver không để heuristic vi phạm các ràng buộc: nghiệm cuối được kiểm tra lại độc lập về số lần ghé thăm, tải trọng, cửa sổ thời gian, số xe và quãng đường.

### Fitness dùng khi học

Fitness trong `gprt_vrptw/evaluator.py` là một tuple được so sánh từ trái sang phải; giá trị nhỏ hơn tốt hơn:

1. Số instance không khả thi trong batch.
2. Tỷ lệ khách hàng chưa phục vụ trung bình.
3. Số xe trung bình.
4. Khoảng cách chuẩn hóa trung bình.
5. Độ phức tạp biểu thức.

Khoảng cách chuẩn hóa trên mỗi instance là:

~~~text
distance / (max(1, số khách) × distance_scale)
distance_scale = max(1, khoảng cách lớn nhất từ depot tới khách hàng)
~~~

Độ phức tạp là số node của cây nhân với `complexity_penalty` (mặc định 0,0001). Do tuple được xếp từ trái sang phải, độ phức tạp chỉ phân xử sau các tiêu chí chất lượng tuyến.

## 2. Ngôn ngữ heuristic dùng chung

GP-only và GPRT dùng cùng một ngôn ngữ biểu thức trong protocol hiện hành. Cây có thể lưu dạng tuple lồng nhau hoặc viết thành prefix token.

~~~text
if_else(ge(slack, 0), add(d_current, 0.5), pdiv(d_depot, 2))
~~~

~~~text
if_else ge slack 0 add d_current 0.5 pdiv d_depot 2
~~~

### Feature đầu vào

13 feature lõi:

| Token | Ý nghĩa |
|---|---|
| `d_current` | Khoảng cách từ vị trí xe hiện tại tới khách hàng ứng viên, chuẩn hóa theo `distance_scale`. |
| `d_depot` | Khoảng cách từ ứng viên tới depot, chuẩn hóa theo `distance_scale`. |
| `demand` | Nhu cầu của ứng viên chia cho sức chứa xe. |
| `ready_time` | Thời điểm sẵn sàng chia cho thời hạn depot. |
| `due_date` | Hạn phục vụ chia cho thời hạn depot. |
| `service_time` | Thời gian phục vụ chia cho thời hạn depot. |
| `arrival_time` | Thời gian dự kiến xe tới ứng viên, chia cho thời hạn depot. |
| `slack` | `(due_date - arrival_time)` chia cho thời hạn depot. |
| `wait_time` | Thời gian chờ nếu xe tới trước `ready_time`, đã chuẩn hóa. |
| `remaining_capacity` | Tải trọng còn lại chia cho sức chứa xe. |
| `route_load` | Tải hiện có chia cho sức chứa xe. |
| `remaining_fraction` | Số khách chưa phục vụ chia cho tổng số khách. |
| `current_time` | Thời gian hiện tại chia cho thời hạn depot. |

Protocol feature-v2 mặc định thêm 3 feature nhìn trước:

| Token | Ý nghĩa |
|---|---|
| `remaining_vehicles` | Tỷ lệ xe còn lại theo giới hạn xe của instance. |
| `next_feasible_fraction` | Tỷ lệ khách khác còn có thể được nối ngay sau ứng viên, xét tải, cửa sổ thời gian và khả năng quay depot. |
| `urgency_rank` | Tỷ lệ khách chưa phục vụ có hạn phục vụ không muộn hơn ứng viên. |

Grammar hiện hành luôn có 16 feature: 13 feature lõi và 3 feature look-ahead. Các feature được tính lười theo những token mà cây thực sự dùng, để giảm thời gian đánh giá.

### Hằng số, phép toán và bảo vệ số học

Hằng số được phép: `-1`, `-0.5`, `0`, `0.5`, `1`, `2`. Các toán tử:

| Nhóm | Toán tử |
|---|---|
| Số học | `add`, `sub`, `mul`, `pdiv`, `min`, `max` |
| So sánh | `ge`, `le` |
| Logic | `and`, `or` |
| Rẽ nhánh | `if_else` |

`pdiv(a,b)` là protected division: nếu mẫu số gần 0, kết quả là tử số. Giá trị phi hữu hạn được thay bằng giá trị hữu hạn bị chặn; mọi kết quả cũng bị chặn trong khoảng [-1e6, 1e6]. Boolean được biểu diễn bằng 0/1.

Runner hiện hành dùng grammar số/boolean không kiểm tra kiểu cứng; các phép so sánh và logic hoạt động qua giá trị số 0/1. Grammar có thể mô tả ràng buộc kiểu như sau, nhưng đây không phải một chế độ runner:

~~~text
ge(number, number) -> boolean
and(boolean, boolean) -> boolean
if_else(boolean, number, number) -> number
~~~

Grammar giới hạn biểu thức mặc định ở tối đa 63 token và độ sâu 6. Cây được rút gọn bằng một số phép bảo toàn ngữ nghĩa như `add(x,0) -> x`, `sub(x,x) -> 0`, `min(x,x) -> x`. Quy tắc này áp dụng sau khi lấy mẫu hoặc biến đổi GP.

## 3. GP thuần

### GP đang dùng làm baseline so sánh

Method `gp` trong `run_gprt_experiment.py` dùng cùng grammar, feature, evaluator, solver và train/validation split như GPRT. Khác biệt cốt lõi là GP không nhận cây do Transformer sinh và không có mạng neural.

Một cycle GP baseline:

1. Chọn batch 4 instance train theo lịch chia đều các họ Solomon.
2. Sinh population ngẫu nhiên đủ 64 cây hợp lệ.
3. Tiến hóa population trong 3 thế hệ.
4. Ghi lại cây và fitness theo batch ID vào archive.
5. Đánh giá thêm 32 cây ngẫu nhiên; các cây này mở rộng thăm dò và làm khớp số yêu cầu đánh giá với 32 mẫu neural của GPRT, nhưng không được ghép vào quần thể đã tiến hóa.
6. Tới mốc validation, sàng lọc archive trên tập train chung rồi chọn biểu thức tốt nhất theo validation.

### Tiến hóa quần thể

Mỗi thế hệ đánh giá cây theo fitness từ điển, giữ nguyên 2 cá thể tốt nhất, rồi tạo phần còn lại bằng tournament selection với tournament size 5:

| Toán tử | Xác suất mặc định | Thao tác |
|---|---:|---|
| Crossover | 0,60 | Chọn một subtree ở cây cha thứ nhất và thay bằng subtree của cây cha thứ hai. |
| Mutation | 0,30 | Chọn một node, tạo subtree ngẫu nhiên mới vừa giới hạn token/độ sâu rồi thay vào cây. |
| Reproduction | 0,10 | Sao chép cây cha sang thế hệ sau. |

Sau biến đổi, cây được rút gọn, kiểm tra grammar và loại trùng theo chuỗi biểu thức. Nếu thiếu cá thể để đủ population, GP sinh thêm cây ngẫu nhiên. Crossover/mutation thử tối đa 20 lần; nếu không tạo được cây hợp lệ, thao tác giữ lại cây gốc.

### GP thuần khác gì GP legacy?

`run_gprt_experiment.py --methods gp` là baseline GP-only trong framework hiện hành, dùng chung grammar, evaluator, solver và split với GPRT. `run_experiment.py` chỉ còn là wrapper tương thích gọi chế độ này. Kết quả trong `outputs/gprt_vrptw/reduced_main` thuộc protocol lịch sử; khi báo cáo cần nêu rõ feature, solver, split và số chu kỳ.

## 4. Transformer sinh biểu thức

### Transformer làm gì?

`CausalTransformer` trong `gprt_vrptw/transformer.py` nhận tiền tố token của một cây và dự đoán token kế tiếp. Mỗi lượt sinh bắt đầu bằng BOS; kết thúc bằng EOS khi grammar xác nhận cây đã đủ các nhánh.

Mạng dùng `nn.TransformerEncoder` với causal mask tam giác trên, nên token vị trí t chỉ nhìn thấy token trước nó. Cấu hình mặc định của GPRT:

| Thành phần | Giá trị |
|---|---:|
| `d_model` | 64 |
| Attention head | 4 |
| Encoder layer | 2 |
| Feed-forward dimension | 128 |
| Dropout | 0,1 |
| Kích thước token biểu thức tối đa | 63 |
| Độ sâu cây tối đa | 6 |
| Temperature | 1,0 |

Chuỗi thực tế còn có BOS và EOS, nên positional embedding cho tối đa 65 vị trí. Từ vựng gồm token đặc biệt, hằng số, feature và toán tử.

Mỗi token được đổi thành tổng của embedding token và embedding vị trí. Trong một attention head, phép tính cốt lõi là:

~~~text
Q = X W_Q,  K = X W_K,  V = X W_V
Attention(X) = softmax((Q Kᵀ / sqrt(d_k)) + causal_mask) V
~~~

`causal_mask` đặt `-∞` lên các vị trí tương lai trước softmax. Nhiều head được ghép lại, qua projection và residual/feed-forward block. Code dùng GELU, dropout, pre-layer normalization và LayerNorm cuối; linear head đổi hidden state của token cuối thành logit cho token kế tiếp. Temperature chia logit trước softmax, sau đó grammar mask loại lựa chọn bất hợp lệ.

### Grammar-constrained decoding

Ở mỗi bước giải mã, `Grammar` theo dõi các slot con còn thiếu và độ sâu của slot. Logit của token không hợp lệ bị loại trước khi lấy mẫu:

1. Token phải thuộc vocabulary.
2. Không được vượt giới hạn độ sâu hoặc token.
3. Còn đủ ngân sách để hoàn tất mọi slot.
4. EOS chỉ hợp lệ khi mọi slot con đã được điền.

Kết quả là một cây heuristic hoàn chỉnh, không phải một route. Khi đánh giá, cùng một cây được chạy tại nhiều trạng thái/instance và chấm ứng viên bằng `argmin` như mục 1.

### Teacher forcing

Teacher forcing chấm xác suất của chuỗi biểu thức đã biết:

~~~text
input  = BOS + các token trừ token cuối
target = các token sau khi dịch sang phải, gồm EOS
~~~

Đây là cách tính negative log-likelihood (NLL) để Transformer học bắt chước các cây elite do GP tìm ra. Đây là nhánh học riêng với policy gradient REINFORCE.

Với chuỗi mục tiêu `y`, loss là trung bình NLL trên token hợp lệ, gồm cả EOS và bỏ qua padding:

~~~text
NLL(y) = -1 / số_token × Σ_t log πθ(y_t | y_<t, grammar)
~~~

## 5. Reinforcement Learning trong project

### RL của GPRT học điều gì?

Trong GPRT, quá trình RL diễn ra ở tầng sinh chương trình:

| Khái niệm RL | Ánh xạ trong GPRT |
|---|---|
| Trạng thái | Tiền tố token hiện có cộng với trạng thái grammar/slot còn thiếu. |
| Hành động | Chọn token tiếp theo trong tập token hợp lệ. |
| Một episode | Sinh xong một biểu thức heuristic. |
| Môi trường đánh giá | Solver VRPTW chạy biểu thức đó trên batch train. |
| Reward | Thứ hạng chất lượng của biểu thức trong nhóm mẫu neural cùng cycle. |

Transformer là policy πθ; xác suất của một biểu thức bằng tích xác suất các token được sinh. Solver không khả vi, nên reward không truyền gradient xuyên qua solver. REINFORCE dùng log-probability đã lưu khi lấy mẫu.

### Reward và policy loss

`rank_rewards` sắp xếp các mẫu neural theo fitness tuple. Cây tốt nhất nhận reward 1, cây kém nhất -1; cây đồng hạng dùng thứ hạng trung bình. Nhờ đó reward không phụ thuộc trực tiếp vào độ lớn của quãng đường giữa các instance khó/dễ.

Mặc định, advantage được chuẩn hóa trong batch (`batch_standardize`):

~~~text
A_i = (reward_i - trung_bình_batch) / độ_lệch_chuẩn_batch
log_probability(cây_i) = Σ_t log πθ(token_t | prefix_t, grammar)
policy_loss = - mean(A_i × log_probability(cây_i))
~~~

Reward rank được ánh xạ chính xác bằng `r_i = 1 - 2 × rank_i / (n - 1)`; hạng của các mẫu đồng điểm được lấy trung bình. `baseline_decay=0.9` cập nhật baseline EWMA mỗi cycle; ở mặc định `batch_standardize`, policy loss dùng trung bình và độ lệch chuẩn của batch hiện tại, còn baseline EWMA được dùng khi chọn `ewma_scale`. Chế độ `none` giữ reward chưa chuẩn hóa. `beta=0.001` là hệ số entropy bonus, giúp policy giữ một mức khám phá khi sinh token.

Điểm tách bạch quan trọng: **chỉ các cây do policy Transformer hiện tại lấy mẫu mới tạo reward cho policy loss.** Cây GP là mẫu off-policy; không được thêm vào nhóm rank REINFORCE. GP có thể dạy Transformer qua elite imitation riêng.

### RL và Transformer luôn phối hợp trong GPRT

REINFORCE tối ưu xác suất các chuỗi token mà Transformer tự lấy mẫu. GP đánh giá và tiến hóa các biểu thức heuristic; những cây GP tốt được đưa vào nhánh elite imitation bằng teacher forcing. Các cây GP là dữ liệu off-policy nên không tham gia nhóm reward của policy gradient.

Runner hiện hành không có chế độ Transformer+RL độc lập và không có actor RL chọn khách hàng trực tiếp. Khi chạy `gprt`, cả GP, Transformer, REINFORCE và elite imitation đều hoạt động trong cùng một vòng huấn luyện.
## 6. GPRT kết hợp các thành phần thế nào?

GPRT ghép ba luồng có vai trò riêng:

- **Transformer:** đề xuất các cây heuristic để khởi tạo GP và nhận policy gradient từ chất lượng chính các cây nó tự sinh.
- **GP:** tìm kiếm lân cận cây bằng crossover, mutation và selection.
- **Elite imitation:** cung cấp tín hiệu từ một nhóm cây GP được sàng lọc để Transformer học xác suất sinh những cấu trúc có ích.

### Sơ đồ một cycle

~~~mermaid
flowchart TD
    A["32 train instance / 8 batch"] --> B["Lấy batch 4 instance"]
    B --> C["Transformer lấy mẫu 32 cây"]
    C --> D["Fitness ban đầu của 32 cây"]
    C --> E["Population 64: neural seeds + random seeds"]
    E --> F["GP tiến hóa 3 thế hệ"]
    F --> G["Archive theo batch ID"]
    D --> H["Rank reward chỉ trên neural samples"]
    H --> L["REINFORCE policy loss"]
    F --> I["Ứng viên GP tốt nhất"]
    I --> J["Chấm lại trên train screening chung"]
    J --> K["Chọn elite đa dạng hành vi"]
    K --> M["Teacher forcing / elite NLL"]
    L --> N["Policy loss + alpha × NLL - beta × entropy"]
    M --> N
    N --> O["Adam update Transformer"]
    O --> C
    G --> P["Sàng lọc archive"]
    P --> Q["Validation 12 instance mỗi 4 cycle"]
    Q --> R["Lưu heuristic validation tốt nhất"]
    R --> S["Test 12 instance sau khi khóa lựa chọn"]
~~~

### Chi tiết theo trình tự thực thi

1. **Batch train:** 32 train instance được chia thành 8 batch, mỗi batch 4 instance, phân tầng theo họ Solomon. Cấu hình hiện tại có thể bắt buộc mỗi train batch được phủ ít nhất một lần.
2. **Neural sampling:** Transformer lấy 32 cây on-policy theo grammar. Cây không phụ thuộc trực tiếp vào trạng thái của một instance khi sinh; chúng là các quy tắc tổng quát rồi được áp lên trạng thái/instance khi đánh giá.
3. **Tạo GP population:** mặc định ghép 32 cây neural làm seed với 32 cây ngẫu nhiên để đủ 64. Nếu bật archive seeding, tối đa 4 cây archive đã sàng lọc thay cho 4 slot ngẫu nhiên. Deduplication có thể sinh lại cây để đủ quần thể.
4. **GP evolution:** population trải qua 3 thế hệ theo cấu hình mặc định. Mỗi cây được đánh giá bằng solver/evaluator chung trên batch hiện tại.
5. **Policy gradient:** fitness ban đầu của 32 cây neural được xếp hạng. Cây GP/off-policy không làm thay đổi reward rank của nhóm này.
6. **Chọn elite:** tối đa 8 cây GP tốt của batch được chấm lại trên fixed screening set gồm 8 train instance. Tối đa 7 cây được lấy cho imitation, ưu tiên cây cho route fingerprint khác nhau trên screening set.
7. **Cập nhật Transformer:** teacher forcing tính NLL trên elite. Loss mặc định:

~~~text
loss = policy_loss + alpha × elite_NLL - beta × entropy
alpha = 0.1
beta  = 0.001
~~~

   Optimizer là Adam, learning rate 0,001; gradient được clip ở norm 1,0. Log có gradient norm từng nhánh và cosine giữa policy gradient với imitation gradient.
8. **Archive và validation:** cây lưu fitness theo batch ID thay vì so trực tiếp điểm của batch khác nhau. Validation định kỳ sàng lọc archive trên tập train chung, lấy shortlist tối đa 16 candidate rồi đánh giá trên 12 validation instance. Heuristic tốt nhất theo validation được checkpoint.
9. **Test:** test set chỉ được dùng sau khi validation đã chốt heuristic/checkpoint. Dùng `--skip-test` cho tuning hoặc screening.

### Khi suy luận

Khi giải instance mới, dùng heuristic tốt nhất đã chọn từ archive/validation. Transformer không cần sinh cây mới ở từng quyết định khách hàng; solver áp dụng cây đã chọn để chấm các khách hàng khả thi. Vì vậy chi phí tìm heuristic offline khác với chi phí giải một instance khi inference.

## 7. Các chế độ runner hiện hành

Runner chỉ nhận hai method: `gprt` (mặc định) và `gp`. Các ablation và pipeline standalone cũ đã được gỡ khỏi runner.

| Method | Thành phần | Mục đích |
|---|---|---|
| `gprt` | Transformer sampling, GP evolution, REINFORCE và elite imitation | Mô hình lai đầy đủ; đây là chế độ mặc định. |
| `gp` | GP evolution, không có Transformer hoặc RL | Baseline GP-only dùng chung grammar, evaluator và solver. |

Archive seeding không phải method riêng. Đặt `archive_seed_count > 0` để thay số cây random tương ứng trong GP population bằng cây được chọn từ archive. Neural seeding, REINFORCE và elite imitation vẫn được giữ nguyên.
## 8. Protocol hiện hành và kết quả lịch sử

Protocol feature-v2 mặc định có 16 feature, population 64, 32 mẫu Transformer, 3 thế hệ GP mỗi cycle, batch 4 instance, validation mỗi 4 cycle và 5 seed cho main run. Preset chính chạy 32 cycle. Archive seeding được bật riêng qua `archive_seed_count`, mặc định bằng 0; preset archive hiện tại là lượt screening 8 cycle, 2 seed và không chấm test.

Các output pilot 4 cycle và các ablation trước đây được giữ trong báo cáo để tra cứu, nhưng không còn là lựa chọn của runner. Không gộp kết quả lịch sử với protocol hiện hành nếu chưa ghi rõ khác biệt về số cycle, feature và ngân sách.
### Kết quả pilot legacy khi so với GP thuần

Bảng pilot 4 cycle, 5 seed trong [báo cáo kết quả](GPRT_VRPTW_REPORT_VI.md) cho thấy GP thuần đạt trung bình 9,750 xe và khoảng cách chuẩn hóa 0,3135; GPRT đầy đủ đạt 9,967 xe và 0,3024. Như vậy, khoảng cách của GPRT thấp hơn khoảng 3,5%, nhưng dùng nhiều hơn trung bình 0,217 xe. GPRT thắng GP ở 29 và thua ở 31 trong 60 phép so sánh seed-instance ghép cặp.

Đây chưa phải lợi thế rõ theo mục tiêu của evaluator, vì số xe được tối ưu trước khoảng cách. Các ablation thành phần trong pilot cũ chỉ có 5 training seed và 4 cycle; kết quả của chúng được giữ trong báo cáo legacy, nhưng không còn là phương pháp chạy hiện hành. Xem đầy đủ độ lệch chuẩn, so sánh ghép cặp và giới hạn của pilot trong báo cáo kết quả.

## 9. Chạy runner

GPRT đầy đủ theo preset chính:

~~~powershell
python run_gprt_experiment.py --config configs/gprt_main.json
~~~

GP-only trong cùng framework:

~~~powershell
python run_gprt_experiment.py --config configs/gp_main.json
~~~

GPRT với archive seeding trong lượt screening train/validation:

~~~powershell
python run_gprt_experiment.py --config configs/gprt_archive_seed_8cycle.json
~~~

Có thể chọn method trực tiếp bằng `--methods gp` hoặc `--methods gprt`. Dùng `--skip-test` trong khi screening/tuning; main protocol yêu cầu tối thiểu 5 seed, còn screening tối đa 2 seed. Runner đọc JSON preset bằng `--config`; tham số CLI được ưu tiên khi trùng khóa.
## 10. Bản đồ code

| File | Vai trò |
|---|---|
| `run_gprt_experiment.py` | Runner hợp nhất; chỉ có `gp` và `gprt`, đọc preset và ghi kết quả. |
| `run_experiment.py` | Wrapper tương thích, mặc định gọi GP-only. |
| `gprt_vrptw/config.py` | Cấu hình GPRT đầy đủ, gồm `archive_seed_count`. |
| `gprt_vrptw/grammar.py` | Token, grammar, cây prefix và simplification. |
| `gprt_vrptw/transformer.py` | Causal Transformer, grammar-constrained sampling và teacher forcing. |
| `gprt_vrptw/gp.py` | Population, tournament, elitism, crossover và mutation. |
| `gprt_vrptw/trainer.py` | Vòng GPRT, REINFORCE, elite imitation, archive, validation và checkpoint. |
| `gprt_vrptw/evaluator.py` | Feature, heuristic, fitness, reward rank và cache. |
| `gprt_vrptw/selection.py` | Archive theo batch, screening và chọn archive seed. |
| `gprt_vrptw/splits.py` | Split và batch phân tầng Solomon. |
| `vrptw_baselines/solver.py` | Bộ dựng tuyến chung và kiểm tra nghiệm. |
| `configs/gprt_main.json` | Preset GPRT đầy đủ. |
| `configs/gprt_archive_seed_8cycle.json` | Preset screening có archive seeding. |
| `configs/gp_main.json` | Preset GP-only trong framework chung. |

Tài liệu kỹ thuật về source, checkpoint, archive và giới hạn implementation: [GPRT_PROJECT_TECHNICAL_DOCUMENTATION_VI.md](GPRT_PROJECT_TECHNICAL_DOCUMENTATION_VI.md). Hướng dẫn chạy: [RUN_GPRT_VRPTW.md](RUN_GPRT_VRPTW.md).



