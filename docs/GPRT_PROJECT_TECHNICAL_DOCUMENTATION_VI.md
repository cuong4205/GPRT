# Tài liệu kỹ thuật project GPRT-VRPTW

## 0. Thông tin tài liệu

- **Ngày rà soát phạm vi runner:** 2026-10-25.
- **Mục tiêu:** giải thích implementation hiện tại từ dữ liệu, bộ dựng tuyến, biểu diễn chương trình, GP, Transformer, RL, trainer, tham số chạy và artefact đầu ra.
- **Mô hình chính:** GPRT = Genetic Programming + Transformer được cập nhật bằng Reinforcement Learning.
- **Bài toán thực nghiệm hiện tại:** Solomon VRPTW.
- **Trạng thái bằng chứng:** runner hiện hành chỉ có `gp` và GPRT đầy đủ; pilot `reduced_main` là dữ liệu lịch sử, không đại diện protocol 32 chu kỳ.

Tài liệu này mô tả những gì code đang làm. Các giá trị được ghi trong kế hoạch nghiên cứu nhưng chưa có trong source hoặc output không được coi là tính năng đã triển khai.

## 1. Tóm tắt ngắn

Project không dùng Transformer để sinh trực tiếp chuỗi khách hàng. Transformer sinh một **biểu thức heuristic dạng cây/prefix**. Biểu thức nhận các feature của một khách hàng đang khả thi và trả về một điểm ưu tiên. Bộ dựng tuyến chung chọn ứng viên có điểm nhỏ nhất, nối khách hàng vào tuyến hiện tại, rồi kiểm tra lại ràng buộc VRPTW.

Luồng chính là:

~~~text
Solomon .txt
    │
    ├─ parse_instance / split_solomon
    │
    ├─ chia train thành batch 4 instance
    │
    ├─ Transformer causal sinh 32 biểu thức prefix
    │
    ├─ ghép với cây ngẫu nhiên → population 64
    │
    ├─ GP tiến hóa 3 thế hệ trên batch hiện tại
    │
    ├─ evaluator chạy bộ dựng tuyến + solution checker
    │
    ├─ reward rank cho mẫu Transformer on-policy
    │
    ├─ elite GP được học bằng teacher forcing
    │
    ├─ cập nhật Transformer bằng policy loss + elite NLL − entropy
    │
    └─ archive / screening / validation / checkpoint
~~~

Các nguyên tắc quan trọng:

1. Mẫu do Transformer sinh là dữ liệu **on-policy** cho REINFORCE.
2. Cá thể do GP tạo ra là dữ liệu **off-policy**; chúng không được đưa vào pool reward REINFORCE. Cây GP tốt luôn được dùng ở nhánh elite imitation trong GPRT đầy đủ.
3. Không có gradient xuyên qua solver VRPTW hoặc qua thao tác GP.
4. Test chỉ được gọi sau khi validation đã khóa heuristic của từng seed.
5. Fitness được so sánh từ điển: khả thi → số khách chưa phục vụ → số xe → khoảng cách chuẩn hóa → độ phức tạp cây.

## 2. Phạm vi và trạng thái hiện tại

### 2.1. Dataset và split

Runner hợp nhất hiện tại chỉ chấp nhận đúng 56 file Solomon trong data/data/Solomon và từ chối dữ liệu không phải Solomon. Mỗi instance có 100 khách hàng, một depot, fleet tối đa 25 xe, capacity thay đổi theo instance. Khoảng cách là Euclid **không làm tròn**.

Split được tạo xác định bởi gprt_vrptw/splits.py:

| Phần | Số instance | Vai trò |
|---|---:|---|
| Train | 32 | GP fitness, Transformer update, archive |
| Validation | 12 | screening candidate và chọn heuristic |
| Test | 12 | chỉ đánh giá cuối sau khi đã khóa heuristic |

Split được phân tầng theo sáu họ Solomon C1, C2, R1, R2, RC1, RC2. Thứ tự trong mỗi họ dựa trên SHA-256 của tên instance, nên không phụ thuộc thứ tự file trong thư mục.

Thư mục data/data/Gehring_Homberger có tồn tại trong project nhưng không thuộc protocol hiện tại.

### 2.2. Hai trạng thái kết quả phải phân biệt

| Trạng thái | Đặc điểm | Cách sử dụng |
|---|---|---|
| outputs/gprt_vrptw/reduced_main | Pilot legacy 4 chu kỳ, 5 seed, 6 phương pháp, 360 dòng test; chỉ phủ 16/32 train instance | Chỉ tham khảo lịch sử, không dùng làm kết luận feature-v2 |
| Protocol feature-v2 | Tối thiểu 8 chu kỳ, đủ 32/32 train instance; có screening train chung, archive theo batch và log gradient | Protocol nên dùng cho thí nghiệm tiếp theo |

Current trainer có require_full_train_coverage=True. Vì vậy, cấu hình cycles < 8 sẽ bị từ chối với dataset 32 train và batch size 4. Artefact 4 chu kỳ hiện có là output lịch sử; không nên coi lệnh legacy trong các tài liệu cũ là lệnh tái chạy hợp lệ với runner mới nếu chưa chủ động tạo GPRTConfig(require_full_train_coverage=False).

### 2.3. Kiểm chứng đã chạy

Tại thời điểm viết tài liệu:

- python -m unittest discover -s tests -v: **37 test passed**.
- python validate_gprt_results.py --results-dir outputs/gprt_vrptw/reduced_main: output legacy hợp lệ theo validator, gồm 6 phương pháp × 5 seed × 12 test.
- Các kết quả screening thử nghiệm trước đây được giữ riêng làm lịch sử; các runner screening theo biến thể đã được gỡ khỏi mã chạy hiện hành.

Kết quả trên không phải bằng chứng GPRT đã tốt hơn một bộ giải VRP tối ưu. Dataset không kèm BKS đã xác minh cùng fleet, distance rule và rounding, nên project không gọi các con số hiện tại là optimality gap.

## 3. Cấu trúc repository

Project không dùng thư mục src/; package được đặt trực tiếp ở root.

| Thành phần | Vai trò |
|---|---|
| run_gprt_experiment.py | Runner hợp nhất; method `gprt` đầy đủ và baseline `gp` |
| gprt_vrptw/config.py | Dataclass cấu hình và kiểm tra tính hợp lệ |
| gprt_vrptw/grammar.py | Tree, prefix token, vocabulary, grammar mask, evaluator số học và simplifier |
| gprt_vrptw/transformer.py | Causal Transformer, constrained sampling và teacher forcing |
| gprt_vrptw/gp.py | Crossover, mutation, deduplication và GP evolution của protocol hiện tại |
| gprt_vrptw/evaluator.py | Feature, compile tree, cache, chạy solver và Fitness |
| gprt_vrptw/selection.py | Archive theo batch, quota, screening và archive seeding |
| gprt_vrptw/splits.py | Split Solomon và tạo batch stratified |
| gprt_vrptw/trainer.py | Vòng lặp GPRT, REINFORCE, elite imitation, validation, checkpoint |
| vrptw_baselines/data.py | Parser Solomon/Gehring-Homberger và dataclass instance |
| vrptw_baselines/solver.py | Bộ dựng tuyến chung, kiểm tra tính khả thi và các baseline FIFO/random/greedy/EDD/GP |
| run_experiment.py | Wrapper tương thích gọi chế độ GP-only trong runner hợp nhất |
| validate_gprt_results.py | Validator output hiện hành và nhận diện schema pilot legacy |
| scripts/ | Công cụ benchmark feature và xử lý kết quả còn dùng |
| configs/ | Preset JSON cho GPRT đầy đủ, archive seeding và GP-only; runner đọc qua `--config` |
| tests/ | Unit test grammar, solver, Transformer, GP và protocol |
| outputs/ | Checkpoint, archive, learning curve, validation/test result và báo cáo máy đọc được |
| docs/ | Hướng dẫn chạy, báo cáo triển khai và audit |

GP hiện hành nằm ở `gprt_vrptw/gp.py`; runner GP-only dùng chung engine này với GPRT.

## 4. Dữ liệu và bộ dựng tuyến VRPTW

### 4.1. Mô hình dữ liệu

vrptw_baselines/data.py định nghĩa:

~~~python
Customer(
    customer_id,
    x, y,
    demand,
    ready_time,
    due_date,
    service_time,
)
~~~

VRPTWInstance giữ:

- name: tên từ dòng đầu file, ví dụ C101;
- path: đường dẫn tuyệt đối;
- max_vehicles: số xe tối đa;
- capacity: tải trọng mỗi xe;
- customers: tuple, trong đó customers[0] là depot;
- sha256: hash byte của file, dùng để nhận diện dữ liệu ổn định;
- source: Solomon hoặc Gehring-Homberger.

parse_instance tìm section VEHICLE và CUSTOMER, đọc dòng vehicle count/capacity, sau đó đọc các dòng có đúng 7 số. ID phải liên tục từ 0. Điều kiện này giúp route checker sử dụng customer ID trực tiếp mà không cần ánh xạ phụ.

discover_instances quét đệ quy các file .txt, sắp xếp theo path rồi parse từng file. Runner GPRT sau đó kiểm tra lại số lượng và source.

### 4.2. Bộ dựng tuyến solve

vrptw_baselines/solver.py dùng một constructive engine cho mọi phương pháp. Nó bắt đầu với:

~~~text
unserved = {1, ..., n}
routes = []
~~~

Mỗi route mới bắt đầu tại depot, current_time là thời gian sẵn sàng của depot, remaining_capacity bằng capacity. Khi đang xây route, _candidate_contexts chỉ đưa một khách hàng vào tập ứng viên nếu:

1. demand của khách hàng không vượt remaining_capacity;
2. thời điểm bắt đầu phục vụ không sau due_date;
3. sau khi phục vụ xong vẫn có thể quay về depot trước depot.due_date.

Candidate context truyền cho heuristic gồm:

- customer hiện tại và customer ứng viên;
- current_time, arrival_time, service_start;
- remaining_capacity;
- số lượng và ID khách chưa phục vụ;
- số route đã dùng.

Sau khi có tập ứng viên khả thi, phương pháp quyết định khách hàng tiếp theo:

| Method | Quy tắc |
|---|---|
| fifo | customer ID nhỏ nhất |
| random | chọn ngẫu nhiên |
| greedy | khoảng cách từ vị trí hiện tại nhỏ nhất, sau đó customer ID |
| edd | due_date sớm nhất, sau đó customer ID |
| gp | giá trị priority nhỏ nhất, sau đó customer ID |

GPRT dùng method gp. Nếu route hiện tại không còn ứng viên, route kết thúc bằng depot. Nếu route rỗng và vẫn còn khách, solver dừng; phần khách còn lại trở thành unserved và nghiệm bị đánh dấu infeasible. Solver không có insertion, 2-opt, local search hoặc backtracking.

### 4.3. Solution checker

validate_solution không tin route được sinh ra mà tính lại độc lập:

- route phải bắt đầu/kết thúc ở depot;
- số route không vượt fleet;
- customer ID phải hợp lệ;
- không có quá tải;
- từng time window phải hợp lệ;
- tính lại distance và waiting time;
- mọi khách phải xuất hiện đúng một lần.

Solution.feasible chỉ là true khi cả route-level checks và coverage đều đúng. Đây là lớp bảo vệ quan trọng vì một heuristic có thể tạo điểm thấp bằng cách bỏ khách hàng nếu evaluator không kiểm tra coverage.

### 4.4. Feature vector hiện tại

gprt_vrptw/grammar.py và gprt_vrptw/evaluator.py dùng 16 feature mặc định. Feature được chuẩn hóa trước khi đi vào biểu thức.

#### 13 feature lõi

| Feature | Cách tính |
|---|---|
| d_current | distance từ vị trí hiện tại tới ứng viên / distance_scale |
| d_depot | distance từ ứng viên tới depot / distance_scale |
| demand | demand / capacity |
| ready_time | ready time / depot horizon |
| due_date | due date / depot horizon |
| service_time | service time / depot horizon |
| arrival_time | arrival time hiện tại / depot horizon |
| slack | (candidate.due_date - arrival_time) / depot horizon |
| wait_time | max(0, ready_time - arrival_time) / depot horizon |
| remaining_capacity | capacity còn lại / capacity |
| route_load | (capacity - remaining_capacity) / capacity |
| remaining_fraction | số khách chưa phục vụ / tổng số khách |
| current_time | thời gian hiện tại / depot horizon |

Trong đó:

~~~text
distance_scale = max(1, max(distance(depot, customer)))
horizon        = max(1, depot.due_date)
~~~

#### 3 feature nhìn trước của feature-v2

| Feature | Ý nghĩa implementation |
|---|---|
| remaining_vehicles | max(0, max_vehicles - routes_used) / max_vehicles |
| next_feasible_fraction | tỷ lệ khách khác còn có thể nối ngay sau ứng viên, xét capacity, time window và khả năng quay depot |
| urgency_rank | tỷ lệ khách chưa phục vụ có due_date <= due_date của ứng viên |

Runner hiện hành luôn dùng đủ 16 feature; không còn chế độ runner tắt ba feature look-ahead.

Evaluator có hai đường tính feature:

1. normalized_features / normalized_feature_vector: đường dense, dễ debug và làm reference.
2. _LazyFeatureVector: chỉ tính terminal thực sự được biểu thức truy cập.

Biểu thức được compile thành Python bytecode nội bộ, nhưng các toán tử được ánh xạ vào _add, _sub, _mul, _pdiv, _min, _max, _ge, _le, _and, _or, _if_else; builtins bị vô hiệu hóa. Test hiện tại so sánh lazy với dense và xác nhận cùng route.

## 5. Biểu diễn chương trình và grammar

### 5.1. Tree và prefix

Kiểu Tree là:

~~~python
str | float | tuple
~~~

Ví dụ heuristic:

~~~text
if_else(
    ge(slack, 0),
    add(d_current, 0.5),
    pdiv(d_depot, 2)
)
~~~

được biểu diễn prefix:

~~~text
if_else ge slack 0 add d_current 0.5 pdiv d_depot 2
~~~

Các hàm chính:

| Hàm | Vai trò |
|---|---|
| tree_to_prefix | cây → token string |
| prefix_to_tree | token string → cây |
| tree_to_string | cây → biểu thức dễ đọc |
| tree_size | số node |
| tree_depth | độ sâu; leaf có depth 0 |
| tree_to_prefix + Grammar.validate_expression | kiểm tra token/cây có hợp lệ |

### 5.2. Vocabulary mặc định

Với feature-v2 và grammar untyped, vocabulary có 36 token:

~~~text
PAD BOS EOS
-1 -0.5 0 0.5 1 2
d_current d_depot demand ready_time due_date service_time
arrival_time slack wait_time remaining_capacity route_load
remaining_fraction current_time remaining_vehicles
next_feasible_fraction urgency_rank
add sub mul pdiv min max ge le and or if_else
~~~

Token ID được tạo theo thứ tự này và được lưu trong vocabulary.json.

### 5.3. Toán tử

| Nhóm | Toán tử | Arity | Kết quả |
|---|---|---:|---|
| Số học | add, sub, mul, pdiv, min, max | 2 | số |
| So sánh | ge, le | 2 | 0 hoặc 1 |
| Logic | and, or | 2 | 0 hoặc 1 |
| Rẽ nhánh | if_else | 3 | giá trị ở nhánh được chọn |

Grammar runner hiện hành không phân loại kiểu ở từng slot. Biểu thức luôn trả một giá trị số; các kết quả so sánh/logic được biểu diễn bằng 0/1.
### 5.4. Grammar state và constrained decoding

GrammarState lưu:

- `pending_depths`: độ sâu của các slot đang chờ;
- `expression_tokens`: số token biểu thức đã sinh, không tính BOS/EOS.

Slot kế tiếp nằm ở cuối list. Khi token operator được chọn, arity của operator thay slot hiện tại bằng các child slot. Grammar chỉ cho token nếu thuộc vocabulary, không vượt giới hạn depth/token, còn đủ ngân sách hoàn tất mọi slot, và chỉ cho EOS khi không còn slot chờ.
max_tokens chỉ tính token biểu thức; chuỗi model thực tế có thêm BOS và EOS nên positional embedding có kích thước max_tokens + 2.

random_tree cũng sử dụng chính grammar này, do đó cây ngẫu nhiên GP và chương trình Transformer dùng cùng không gian tìm kiếm.

### 5.5. Số học bảo vệ và simplification

safe_evaluate và evaluator compiled đều:

- dùng protected division: mẫu số gần 0 trả về tử số;
- chặn kết quả hữu hạn trong khoảng [-1e6, 1e6];
- đổi kết quả boolean thành 0/1;
- thay non-finite bằng giá trị hữu hạn bị chặn.

simplify_tree thực hiện một số biến đổi bảo toàn ngữ nghĩa và vẫn nằm trong vocabulary:

~~~text
add(x, 0)       → x
sub(x, 0)       → x
mul(x, 1)       → x
pdiv(x, 1)      → x
min(x, x)       → x
max(x, x)       → x
sub(x, x)       → 0
if_else(1, a,b) → a
if_else(0, a,b) → b
~~~

Các constant subtree đủ điều kiện còn được constant-fold. Test fuzz hiện tại kiểm tra cây sau simplify vẫn hợp lệ và giữ giá trị trong tập giá trị hữu hạn đã chọn.

## 6. Genetic Programming hiện tại

### 6.1. Ý tưởng

GP không trực tiếp tiến hóa một route. Nó tiến hóa một tree heuristic. Với mỗi tree, solver dùng tree để rank ứng viên tại từng bước xây route, sau đó evaluator đo chất lượng các route trên một batch instance.

### 6.2. Khởi tạo population

Ở GPRT đầy đủ, một cycle bắt đầu với:

~~~text
32 cây Transformer hiện tại
+ cây archive đã screening nếu archive seeding bật
+ cây ngẫu nhiên cho tới đủ 64
→ deduplicate_fill
~~~

deduplicate_fill:

1. simplify từng cây;
2. kiểm tra grammar;
3. loại trùng theo tree_to_string;
4. sinh thêm cây grow/full xen kẽ nếu population chưa đủ;
5. trong grammar cực nhỏ có thể thêm lại cây hợp lệ khi không thể đạt đủ số biểu thức duy nhất.

Với default GPRT không archive seeding: 32 neural seed + 32 random seed. Với archive_seed_count=4: 32 neural + 4 archive + 28 random; bốn archive tree thay bốn random seed, không làm population lớn hơn 64.

### 6.3. Một thế hệ GP

evolve_population lặp gp_generations=3 lần. Mỗi thế hệ:

1. chấm toàn bộ population trên batch hiện tại;
2. sort theo Fitness tuple tăng dần;
3. giữ elite_count=2 tree tốt nhất;
4. chọn parent bằng tournament size 5;
5. sinh phần còn lại theo xác suất:

| Operator | Xác suất |
|---|---:|
| Crossover | 0.60 |
| Mutation | 0.30 |
| Reproduction | 0.10 |

6. simplify, validate và deduplicate để tạo population thế hệ sau.

Fitness tốt hơn là fitness nhỏ hơn. Python NamedTuple thực hiện so sánh từ trái sang phải, không phải tổng có trọng số.

### 6.4. Crossover

crossover(a, b, rng, grammar):

1. chọn random subtree path trong parent a;
2. chọn donor subtree trong b;
3. thay subtree;
4. simplify và validate;
5. thử tối đa 20 lần; nếu không hợp lệ thì trả lại a.
### 6.5. Mutation

mutate(tree, rng, grammar):

1. chọn random node;
2. sinh subtree ngẫu nhiên bằng grammar dùng chung;
3. thay subtree, simplify và kiểm tra lại giới hạn token/depth;
4. thử tối đa 20 lần; nếu không hợp lệ thì giữ cây cũ.
### 6.6. GP archive nội bộ và candidate archive

evolve_population có archive tạm của cycle, lưu fitness tốt nhất của từng expression. Trainer có self.archive dài hạn lớn hơn, lưu observation theo:

~~~text
expression
  ├─ tree
  ├─ tokens
  ├─ observations[batch_id]
  │    ├─ fitness
  │    ├─ cycle
  │    └─ instance names
  └─ screening_fitness
~~~

Không được so trực tiếp fitness của hai batch train khác nhau như thể chúng là cùng một thước đo. Khi validation, archive lấy quota từ các batch rồi chấm lại candidate trên screening set train chung.

## 7. Transformer causal

### 7.1. Kiến trúc

gprt_vrptw/transformer.py định nghĩa CausalTransformer. Dù dùng nn.TransformerEncoder, module được biến thành decoder-only causal model bằng upper-triangular causal mask.

Default:

| Thành phần | Giá trị |
|---|---:|
| Token embedding dimension (d_model) | 64 |
| Attention heads (nhead) | 4 |
| Transformer layers | 2 |
| Feed-forward dimension | 128 |
| Dropout | 0.1 |
| Activation | GELU |
| Normalization | pre-layer norm trong layer, LayerNorm cuối encoder |
| Positional embedding | học được, dài 65 với max_tokens=63 |
| Vocab | 36 token feature-v2 |
| Số parameter hiện tại | 75,876 |

Model nhận input_ids shape [batch, sequence] và trả logits shape [batch, sequence, vocab].

forward:

1. lấy token embedding;
2. cộng position embedding;
3. tạo padding mask từ token PAD nếu caller chưa truyền;
4. áp dụng causal mask để vị trí t không nhìn token ở vị trí > t;
5. chạy TransformerEncoder;
6. linear projection ra logits vocabulary.

Unit test xác nhận thay token tương lai không làm thay đổi logits các vị trí trước và padding không làm thay đổi prefix logits.

### 7.2. Sampling có grammar mask

generate(count) khởi tạo mỗi chuỗi bằng BOS và mỗi grammar state bằng root number.

Ở mỗi bước:

1. chạy model trên prefix hiện tại;
2. lấy logits của vị trí cuối;
3. chia temperature;
4. lấy valid_token_ids từ grammar state;
5. đặt logit token không hợp lệ thành -inf;
6. tạo categorical distribution;
7. sample hoặc argmax nếu greedy;
8. lưu log-probability và entropy;
9. cập nhật grammar state;
10. dừng khi sample EOS.

GeneratedExpression chứa:

- tree: cây đã decode;
- token_ids: chỉ token biểu thức, không gồm BOS/EOS;
- log_probability: tổng log probability của các action, gồm EOS;
- entropy: entropy trung bình trên các bước action.

Nếu sau max_tokens + 1 bước vẫn còn chuỗi active, decoder ném lỗi thay vì âm thầm cắt chương trình.

### 7.3. Teacher forcing

teacher_forced(trees) encode mỗi cây thành:

~~~text
BOS + expression tokens + EOS
~~~

Sau đó:

- input là chuỗi bỏ token cuối;
- target là chuỗi bỏ token đầu;
- batch được pad;
- logits cũng được grammar-mask theo state tương ứng của target;
- sequence_log_prob là tổng log probability token hợp lệ;
- nll là negative log likelihood trung bình trên các target token không phải PAD;
- entropy là entropy trung bình trên target positions.

Sampling và teacher forcing đều chia logits cho cùng temperature. Điều này làm log probability recompute khớp với log probability đã lưu khi greedy sampling; test hiện tại kiểm tra điều đó.

### 7.4. Dropout trong GPRT

GPRT dùng cùng một Transformer cho sampling, policy gradient và elite teacher forcing trong mỗi cycle. Mô hình ở training mode; dropout thuộc policy đang được tối ưu. Không còn nhánh Transformer+RL riêng hoặc cấu hình sample-count độc lập.
## 8. Reinforcement Learning và cập nhật neural

### 8.1. Reward của mẫu Transformer

Với GPRT đầy đủ, chỉ 32 tree vừa sample từ Transformer được đưa vào initial_fitness. GP tree không được đưa vào cùng pool reward.

rank_rewards:

1. sort các Fitness theo thứ tự từ điển;
2. tie nhận average rank;
3. chuyển rank thành reward trong [-1, 1]:

~~~text
reward = 1 − 2 × rank / (N − 1)
~~~

Tree tốt nhất có reward 1, tree tệ nhất có reward -1. Nếu chỉ có một tree, reward là 0. Vì Fitness chứa complexity ở vị trí cuối, hai tree tạo cùng chất lượng route nhưng khác kích thước có thể nhận rank khác nhau.

### 8.2. Advantage và baseline

Default advantage_mode=batch_standardize.

Về code:

~~~text
advantages = detach(rewards − baseline_for_loss)
scale = std(rewards)
batch_standardize:
    advantages = (advantages − mean(advantages)) / scale
ewma_scale:
    advantages = advantages / scale
none:
    giữ advantages chưa scale
~~~

Với batch_standardize, baseline_for_loss được đặt bằng reward mean của batch, nên baseline EWMA không trực tiếp quyết định loss trong mode mặc định. EWMA vẫn được cập nhật và ghi log. Với ewma_scale hoặc none, baseline hiện tại có thể là EWMA sau cycle đầu.

### 8.3. Policy loss

Mỗi GeneratedExpression đã giữ log probability có gradient. Trainer dùng:

~~~text
policy_loss = − mean(advantage_i × log_probability_i)
~~~

REINFORCE luôn được bật trong GPRT đầy đủ; runner không còn cờ tắt policy gradient.

Không có reward backpropagation xuyên evaluator: reward được convert sang tensor và detach trước khi nhân với log probability.

### 8.4. Elite imitation từ GP

Sau GP:

1. sort final population theo batch fitness;
2. lấy tối đa 8 ứng viên đầu (elite_screening_candidates);
3. chấm lại trên screening train set cố định;
4. chọn tối đa ceil(64 × 0.10)=7 tree, ưu tiên khác fingerprint route trên toàn screening set;
5. teacher-force các elite tree.

Nhánh này tạo elite_nll. Cá thể GP không được gọi là on-policy sample; đây là imitation/off-policy learning.

### 8.5. Entropy và loss cuối

Entropy được lấy trung bình giữa entropy của neural sample và teacher-forced elite:

~~~text
entropy = 0.5 × (sample_entropy + teacher_entropy)
~~~

Loss cuối của implementation:

~~~text
loss =
    policy_loss
    + alpha × elite_nll
    − beta × entropy
~~~

Default:

~~~text
alpha = 0.1
beta  = 0.001
~~~

Các lượt screening alpha=0,01 và alpha=0,05 trong output cũ là thí nghiệm lịch sử; runner hiện hành chỉ dùng alpha mặc định 0,1 trong GPRT đầy đủ.

Trainer tính gradient norm của policy branch và alpha×elite branch trước khi cộng loss; cosine giữa hai vector được ghi để xem hai nguồn tín hiệu cùng hướng hay xung đột. Sau backward, toàn bộ gradient được clip ở norm 1.0 rồi cập nhật bằng Adam.

## 9. Một cycle GPRT từ đầu đến cuối

Với default GPRT, một cycle thực hiện như sau:

### Bước 1 — chọn batch

stratified_batches(train_instances, batch_size=4, seed=seed) nhóm instance theo family, shuffle trong family và thứ tự family bằng seed, sau đó round-robin để tạo 8 batch. Cycle c dùng:

~~~python
batch = batches[(c - 1) % len(batches)]
~~~

Vì vậy:

- 8 cycles = một lượt phủ 32 train instance;
- 16 cycles = hai lượt;
- 32 cycles = bốn lượt.

### Bước 2 — Transformer sampling

Transformer hiện tại sinh 32 expression. Các expression này là on-policy samples của policy ở đầu cycle.

### Bước 3 — tạo seed GP

Trong GPRT đầy đủ:

- thêm 32 neural tree nếu use_transformer_seeding=True;
- thêm archive tree nếu archive_seed_count > 0;
- sinh cây random cho đủ population 64.

Archive tree phải đã được screening trên cùng fixed screening set, loại tree trùng expression với neural sample hiện tại và được GP chấm lại trên batch mới.

### Bước 4 — GP evolution

GP chạy 3 thế hệ. Tất cả tree trong population được đánh giá bằng evaluator cache. Kết quả cuối dùng cho final population và elite selection.

### Bước 5 — ghi archive

Cả on-policy tree và final GP population được ghi observation vào candidate archive theo batch ID. Cùng expression trên cùng batch chỉ giữ observation tốt hơn.

### Bước 6 — tính reward và policy branch

Chỉ initial_fitness của on-policy tree được rank. Đây là điểm tách bạch quan trọng giữa neural learning và search refinement.

### Bước 7 — elite screening và teacher forcing

GP elite được chấm lại trên screening set để tránh chọn tree chỉ tốt vì batch hiện tại. Selector elite ưu tiên fingerprint route khác nhau trên toàn screening set; behavior probe riêng được dùng cho log diversity và archive seeding.

### Bước 8 — optimizer step

Ghép policy loss, elite NLL và entropy; đo gradient branch; clip; Adam step.

### Bước 9 — validation định kỳ

Nếu cycle % validation_interval == 0 hoặc là cycle cuối:

1. lấy candidate từ archive qua screen_archive;
2. chấm lại trên 8 screening train instance;
3. chọn tối đa 16 candidate;
4. evaluate trên 12 validation instance;
5. cập nhật best_validation_fitness và best_tree nếu tốt hơn.

### Bước 10 — log/checkpoint

Trainer ghi log JSON, lưu last.pt mỗi cycle và lưu best.pt khi validation best được cập nhật.

## 10. Fitness, cache và ngân sách đánh giá

### 10.1. Fitness tuple

gprt_vrptw/evaluator.py định nghĩa:

~~~text
Fitness(
    infeasible_instances,
    mean_unserved_fraction,
    mean_vehicles,
    mean_normalized_distance,
    complexity,
)
~~~

Chi tiết:

1. infeasible_instances: số instance trong batch có solution không khả thi;
2. mean_unserved_fraction: trung bình len(unserved) / customer_count;
3. mean_vehicles: số route trung bình;
4. mean_normalized_distance:

~~~text
mean(
    solution.distance /
    (max(1, customer_count) × distance_scale)
)
~~~

5. complexity = tree_size(tree) × complexity_penalty.

Default complexity_penalty=1e-4. Complexity chỉ là tie-break sau chất lượng route trong fitness tuple, nhưng vẫn ảnh hưởng rank reward nếu hai tree bằng nhau ở bốn thành phần đầu.

### 10.2. Cache key

Solution cache có key:

~~~text
(expression_string, instance.sha256, evaluator_config_hash)
~~~

Config hash hiện ghi nhận:

- unrounded Euclidean distance;
- tie-break customer ID;
- feature version 1 hoặc 2;
- solver shared_constructive_v2.

Cache giúp tái sử dụng nghiệm giữa train, screening, validation và diagnostic khi expression/instance/evaluator config giống nhau. FitnessEvaluator vẫn tăng fitness_requests cho mỗi request logic và tách riêng số unique_evaluations, instance_evaluations, cache_hits.

### 10.3. Thống kê theo phase

Stats được tách theo:

- train;
- screening;
- validation;
- diagnostic.

Không nên kết luận hai phương pháp có cùng chi phí chỉ từ fitness_requests. Báo cáo nên có thêm unique/physical instance evaluations, cache hits, evaluator seconds, GP seconds, Transformer sample/train seconds và wall-clock.

## 11. Archive selection và validation

### 11.1. Batch ID

instance_set_id tạo hash ổn định từ các instance SHA-256 đã sort. Archive có thể chứa cùng expression được quan sát trên nhiều batch mà không trộn fitness.

### 11.2. Quota candidate

quota_candidates(archive, limit):

1. gom observation theo batch ID;
2. mỗi batch nhận quota ceil(limit / số_batch);
3. lấy tree tốt nhất trong từng batch;
4. nếu chưa đủ do expression trùng, fill bằng candidate có best raw observation.

Bước fill cuối tồn tại trong code để tránh shortlist thiếu phần tử, nhưng raw fitness giữa batch khác nhau không hoàn toàn cùng thước đo. Với default limit=16 và 8 batch, quota thường đủ 16 candidate; audit hiện tại đã kiểm tra nhánh fallback riêng thay vì giả định nó luôn được gọi.

### 11.3. Screening chung

screen_archive chấm candidate trên fixed screening set 8 train instance. Chỉ screening_fitness trên set chung mới được dùng để xếp candidate sang validation. Đây là điểm khác so với pilot cũ, nơi archive có thể dễ bị chi phối bởi batch dễ.

### 11.4. Behavior fingerprint

behavior_fingerprint là tuple các route được quan sát trên probe instance. Nó đo khác biệt hành vi ở probe, không phải chứng minh hai biểu thức khác nhau về mọi instance.

## 12. Toàn bộ tham số GPRTConfig

### 12.1. Transformer và expression

| Tham số | Default | Ý nghĩa |
|---|---:|---|
| d_model | 64 | kích thước embedding/hidden |
| nhead | 4 | số attention head; phải chia hết d_model |
| num_layers | 2 | số TransformerEncoderLayer |
| dim_feedforward | 128 | kích thước FFN |
| dropout | 0.1 | dropout của Transformer |
| max_tokens | 63 | tối đa token expression, không tính BOS/EOS |
| max_depth | 6 | depth tối đa của tree |
| learning_rate | 0.001 | Adam learning rate |
| gradient_clip | 1.0 | max gradient norm |
| temperature | 1.0 | dùng cho sampling và teacher forcing |

### 12.2. Population và lịch GP

| Tham số | Default | Ý nghĩa |
|---|---:|---|
| population_size | 64 | kích thước population |
| transformer_samples | 32 | số expression neural mỗi cycle khi có GP |
| tournament_size | 5 | số contestant tournament |
| elite_count | 2 | số tree GP giữ nguyên mỗi thế hệ |
| crossover_rate | 0.6 | xác suất crossover |
| mutation_rate | 0.3 | xác suất mutation |
| reproduction_rate | 0.1 | xác suất copy parent |
| gp_generations | 3 | số thế hệ mỗi cycle |
| cycles | 32 | số gradient/search cycle |
| batch_size_instances | 4 | số instance trong train batch |
| validation_interval | 4 | validate mỗi 4 cycle |
| validation_candidates | 16 | số archive candidate đưa sang validation |
| screening_train_size | 8 | fixed train screening set |
| elite_screening_candidates | 8 | số GP elite batch được chấm lại |
| behavior_probe_size | 1 | số instance dùng fingerprint |
| archive_seed_count | 0 | số archive tree thay random seed |
| complexity_penalty | 0.0001 | penalty mỗi tree node |

elite_count=2 là GP elitism. Nó khác elite_fraction=0.10, là số tree được chọn cho elite imitation; với population 64, ceil(6.4)=7.

### 12.3. Policy optimization

| Tham số | Default | Ý nghĩa |
|---|---:|---|
| alpha | 0.1 | trọng số elite NLL |
| beta | 0.001 | trọng số entropy bonus |
| elite_fraction | 0.10 | tỷ lệ elite teacher-forcing |
| baseline_decay | 0.9 | EWMA baseline |
| advantage_mode | batch_standardize | ewma_scale, batch_standardize hoặc none |

### 12.4. Cấu hình protocol hiện hành

| Tham số | Mặc định | Tác động |
|---|---:|---|
| `archive_seed_count` | 0 | Số archive tree thay random seed trong GPRT; chỉ áp dụng cho method `gprt`. |
| `require_full_train_coverage` | True | Từ chối số cycle không phủ đủ batch train. |

Feature-v2 và các nhánh neural/GPRT là cố định trong runner hiện hành. Không còn cờ chọn typed grammar, tắt look-ahead, GP hoặc REINFORCE.
GPRTConfig.validate() kiểm tra divisibility, population/rate, kích thước dương, archive seed không vượt số slot random và advantage mode hợp lệ. Temperature dương được kiểm tra khi sampling/teacher forcing.

## 13. Runner và các method

### 13.1. CLI arguments và preset

run_gprt_experiment.py hỗ trợ:

| Argument | Default hiện tại |
|---|---|
| --data-dir | data/data/Solomon |
| --output-dir | outputs/gprt_vrptw/main; `gp` tự chuyển sang `gp_main` |
| --config | tùy chọn; preset JSON |
| --methods | `gprt` mặc định; chọn `gp` hoặc `gprt` |
| --seeds | 20260914 20260915 20260916 20260917 20260918 |
| --cycles | 32 |
| --population | 64 |
| --transformer-samples | 32 |
| --gp-generations | 3 |
| --screening-train-size | 8 |
| --screening-run | false |
| --skip-test | false |
| --device | cpu |

Main runner yêu cầu tối thiểu 5 seed. Nếu bật --screening-run, runner cho tối đa 2 seed. --skip-test nên được bật trong mọi lượt tuning/screening. Archive seeding chỉ áp dụng khi có method `gprt`.

Runner đọc JSON preset bằng `--config`; tham số CLI được ưu tiên khi trùng khóa. Preset GP-only là `configs/gp_main.json`.

### 13.2. Hai method

| Method | Thành phần |
|---|---|
| `gp` | GP-only; dùng grammar/evaluator/solver hiện hành, không có Transformer hoặc RL. |
| `gprt` | GPRT đầy đủ: Transformer sampling, GP evolution, REINFORCE và elite imitation. |

Archive seeding được bật bằng `archive_seed_count` trong preset hoặc `--archive-seed-count`. Đây là cấu hình của GPRT, không phải method thứ ba.
### 13.3. GP baseline trong current runner

run_gp_baseline dùng cùng grammar/evaluator/split với GPRT nhưng:

1. không có model Transformer;
2. mỗi cycle sinh population random;
3. chạy GP 3 thế hệ;
4. ghi archive theo batch;
5. thêm 32 random tree fitness request để tương ứng phần neural sample của GPRT;
6. screening candidate và validation cùng protocol.

Đây là GP-only baseline dùng chung feature-v2 và solver với GPRT.

## 14. Cài đặt và chạy

### 14.1. Dependency

requirements.txt hiện chỉ yêu cầu:

~~~text
torch>=2.2
~~~

Tài liệu chạy hiện tại ghi Python 3.11–3.13; project đã được chạy với PyTorch 2.11 CPU. Các thư viện còn lại là standard library.

PowerShell:

~~~powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m unittest discover -s tests -v
~~~

### 14.2. GPRT đầy đủ

~~~powershell
python run_gprt_experiment.py --config configs/gprt_main.json
~~~

### 14.3. GP-only

~~~powershell
python run_gprt_experiment.py --config configs/gp_main.json
~~~

### 14.4. Archive seeding tùy chọn

~~~powershell
python run_gprt_experiment.py --config configs/gprt_archive_seed_8cycle.json
~~~

Preset archive là screening train/validation, hai seed, tám cycle và không đánh giá test. Để chạy main protocol với bốn archive seed, dùng `python run_gprt_experiment.py --config configs/gprt_main.json --archive-seed-count 4`.
### 14.5. Chạy bằng API và resume

Runner CLI chưa có cờ resume. Có thể resume qua trainer:

~~~python
from pathlib import Path
from gprt_vrptw.config import GPRTConfig
from gprt_vrptw.trainer import GPRTTrainer

config = GPRTConfig(cycles=32)
trainer = GPRTTrainer(config, device="cpu")
result = trainer.train(
    train_instances,
    validation_instances,
    seed=20260914,
    output_dir=Path("outputs/resumed"),
    split_manifest=split_manifest,
    resume=Path("outputs/resumed/last.pt"),
)
~~~

Resume sẽ từ chối nếu config hoặc vocabulary khác checkpoint, seed khác, screening set khác hoặc grammar feature version không tương thích.

## 15. Checkpoint và output

### 15.1. File ở mỗi neural run

| File | Nội dung |
|---|---|
| last.pt | checkpoint cycle gần nhất |
| best.pt | checkpoint tại validation best |
| config.json | toàn bộ GPRTConfig |
| vocabulary.json | runtime vocabulary |
| split_manifest.json | mapping instance → train/validation/test |
| candidate_archive.json | expression, tokens, observation theo batch, screening fitness |
| training_summary.json | seed, completed cycles, best expression, coverage |
| final_heuristic.txt | biểu thức best validation |
| learning_curves.csv | log từng cycle |

### 15.2. Nội dung checkpoint

trainer._state lưu:

- model.state_dict;
- optimizer.state_dict;
- config;
- vocabulary và grammar settings;
- split;
- archive;
- cycle logs;
- training step;
- EWMA baseline;
- training instance đã thấy;
- screening instance names;
- best validation fitness/tree;
- seed;
- Python global/local RNG;
- Torch CPU/CUDA RNG;
- evaluator solution cache và stats.

Do checkpoint giữ cả cache và RNG, test hiện tại xác nhận resume sau một cycle cho cùng tree, fitness, model tensors và output như chạy liền mạch.

### 15.3. Log quan trọng

learning_curves.csv có các nhóm field:

- loss: loss, policy_loss, elite_nll, entropy;
- reward: mean/std, baseline, advantage;
- quality: train/validation fitness, feasibility, vehicles, normalized distance;
- complexity/diversity: expression length/depth, population diversity, behavior diversity;
- interaction: neural seed count, archive seed count, direct survival, neural-vs-random, direct elite count;
- accounting: fitness requests, unique evaluations, per-phase requests/evaluations, cache hits;
- timing: Transformer, GP, evaluator, wall;
- gradient: total norm, policy norm, elite norm, cosine.

## 16. Phân biệt output lịch sử và runner hiện tại

Output `outputs/gprt_vrptw/reduced_main` được tạo bởi protocol pilot cũ, gồm các method và ablation không còn chạy được từ runner hiện hành. Validator giữ nhánh đọc schema này để kiểm tra kết quả đã lưu.

`run_gprt_experiment.py` là runner hiện hành, có hai method `gprt` và `gp`. `run_experiment.py` là wrapper gọi GP-only. Không so sánh trực tiếp output lịch sử và kết quả mới nếu chưa ghi rõ khác biệt về số cycle, feature, ngân sách và method.
## 17. Những giới hạn kỹ thuật cần ghi trong luận văn

### 17.1. Mô hình không phải solver VRPTW tối ưu

GPRT chỉ sinh priority rule cho một constructive append-only solver. Chất lượng bị giới hạn bởi engine: không insertion, không local search, không backtracking. Nếu sau này thêm 2-opt hoặc insertion, đó là một thay đổi engine phải áp dụng cho mọi method và báo cáo trước/sau riêng.

### 17.2. GPRT hiện tại là chuyển thể sang Solomon

Bài GPRT gốc và project này khác môi trường, feature, grammar, population, budget và objective. Không được ghi như tái lập số liệu bài báo. Chỉ nên nói project đánh giá cơ chế GP–Transformer–RL trong VRPTW.

### 17.3. Tổng quát hóa còn giới hạn

Train/validation/test đều nằm trong các họ hình học Solomon đã biết. Split này đánh giá instance chưa thấy trong cùng benchmark, nhưng chưa chứng minh OOD theo layout mới, scale mới hoặc phân phối nhu cầu mới. Cũng không thể suy ra hiệu quả điều phối cảng hoặc dữ liệu thời gian thực.

### 17.4. Không có BKS trong protocol hiện tại

Không được đổi mean_normalized_distance thành optimality gap. Muốn báo cáo gap phải có BKS/optimal reference được xác minh cùng:

- số xe/fleet;
- capacity và time windows;
- distance rule;
- rounding;
- objective ordering.

### 17.5. Checkpoint legacy không resume vào feature-v2

Feature-v2 đổi vocabulary và một số logic evaluator. load_checkpoint so config/vocabulary nên checkpoint cũ sẽ bị từ chối. Không bỏ qua kiểm tra này.

### 17.6. Cách đọc JSON preset

Runner đọc file JSON qua `--config`. Giá trị CLI ghi đè giá trị cùng tên trong preset; các khóa không thuộc `GPRTConfig` được dùng cho runner (method, output path, seeds và chế độ screening/test).
## 18. Checklist tái lập một thí nghiệm

### Trước khi chạy

- [ ] Đúng 56 file Solomon, không dùng Gehring-Homberger.
- [ ] Split là 32/12/12.
- [ ] Khóa method, seed, cycles, population, GP generations, feature và grammar.
- [ ] cycles >= 8 nếu require_full_train_coverage=True.
- [ ] Tuning dùng --skip-test.
- [ ] Không overwrite output cũ.

### Trong khi chạy

- [ ] Theo dõi training_coverage.
- [ ] Kiểm tra screening_instance_names.
- [ ] Ghi train/screening/validation/diagnostic evaluations riêng.
- [ ] Kiểm tra gradient norm, cosine và reward tie rate.
- [ ] Giữ checkpoint last.pt và best.pt.

### Sau khi chạy

- [ ] Chạy unit tests.
- [ ] Kiểm tra training_summary.json và coverage 32/32.
- [ ] Kiểm tra candidate_archive.json có observation theo batch.
- [ ] Chỉ chọn config bằng validation.
- [ ] Test một lần sau khi khóa heuristic.
- [ ] Báo cáo feasibility, unserved, vehicles, distance, runtime và seed variation.
- [ ] Không gọi kết quả là optimality gap nếu chưa có BKS.

## 19. Bản đồ đọc source theo thứ tự

Nếu cần trình bày hoặc debug một pipeline mới, đọc theo thứ tự:

1. vrptw_baselines/data.py — hiểu instance và parser.
2. vrptw_baselines/solver.py — hiểu route construction và feasibility.
3. gprt_vrptw/grammar.py — hiểu không gian expression.
4. gprt_vrptw/evaluator.py — hiểu feature, compile và Fitness.
5. gprt_vrptw/gp.py — hiểu search bằng GP.
6. gprt_vrptw/transformer.py — hiểu sampling/teacher forcing.
7. gprt_vrptw/trainer.py — ghép RL, GP và validation.
8. gprt_vrptw/selection.py — hiểu archive/screening.
9. run_gprt_experiment.py — hiểu method mapping và test policy.
10. tests/ — xem các invariant đã được kiểm chứng.

## 20. Kết luận kỹ thuật

Implementation hiện tại là một pipeline hybrid có ba lớp tách biệt:

1. **Program space:** grammar cố định tạo expression prefix hợp lệ với protected arithmetic và giá trị logic 0/1.
2. **Search/evaluation space:** GP và shared VRPTW solver biến expression thành route behavior rồi đo Fitness.
3. **Learning space:** Transformer học phân phối expression bằng REINFORCE trên mẫu của chính nó và elite imitation từ GP.

Đóng góp kỹ thuật của code hiện tại nằm ở interface giữa ba lớp: cùng grammar cho random/GP/neural, grammar-constrained generation, reward chỉ từ on-policy neural sample, screening archive theo batch, và checkpoint có đủ trạng thái để resume. Chất lượng heuristic cuối cùng vẫn phải được đánh giá theo protocol nhiều seed, cùng engine và validation/test policy; source code hiện tại chưa tự động chứng minh GPRT thắng mọi baseline.









