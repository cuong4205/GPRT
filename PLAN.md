# Kế hoạch khóa luận 12 tuần: GP kết hợp LSTM và học tăng cường cho điều phối xe cảng có khả năng tổng quát hóa

## 1. Đề tài, mục tiêu và phạm vi

**Tên đề tài đề xuất:**  
**“Nghiên cứu khả năng tổng quát hóa của lập trình di truyền kết hợp học tăng cường trong điều phối xe tại cảng container.”**

Kế hoạch dành cho khóa luận đại học, **12 tuần × 15–20 giờ/tuần**, với nền tảng ML và GP cơ bản, chưa học RL; sử dụng laptop và Colab miễn phí; chưa có dữ liệu cảng thực tế.

Tài liệu xuất phát là *Genetic Programming with Reinforcement Learning Trained Transformer for Real-World Dynamic Scheduling Problems*, bản arXiv v2 ngày 05/08/2025. GP trong tài liệu là **Genetic Programming**; GPRR sử dụng LSTM, GPRT sử dụng Transformer. Nguồn đã đọc: :codex-file-citation{path="C:/Users/Admin/Downloads/RNN+GP.pdf" purpose="source"}.

**Mục tiêu thực hiện:** xây dựng một hệ thống mô phỏng điều phối xe chạy được; triển khai GP và GP+LSTM; kiểm tra liệu huấn luyện trên nhiều điều kiện vận hành có giúp heuristic hoạt động tốt hơn khi môi trường thay đổi.

Ba câu hỏi nghiên cứu:

1. **RQ1 — Hiệu quả của mô hình lai:** với cùng ngân sách đánh giá, GP+LSTM có tìm được heuristic tốt hơn GP thuần không?
2. **RQ2 — Tổng quát hóa:** thay đổi điều kiện môi trường trong huấn luyện có cải thiện hiệu quả trên những điều kiện chưa gặp không?
3. **RQ3 — Đánh đổi:** cải thiện đó có làm giảm hiệu quả ở môi trường thông thường, tăng thời gian huấn luyện hoặc làm biểu thức phức tạp hơn không?

**Đóng góp dự kiến** là một thực nghiệm có thể tái lập về tổng quát hóa, kèm simulator và mô hình lai được mô tả rõ. Không đặt giả định rằng phương pháp chắc chắn thắng baseline hoặc rằng domain randomization là ý tưởng mới.

Trong 12 tuần, Transformer nằm trong phần tổng quan và hướng phát triển. Phần triển khai bắt buộc tập trung vào LSTM. Kết luận thực nghiệm giới hạn ở **môi trường mô phỏng**, chưa xác nhận hiệu quả tại một cảng thật.

## 2. Kết quả đọc bài và bản đồ tài liệu tham khảo

### Những điểm cần hiểu và kiểm tra từ bài gốc

Ý tưởng trung tâm là:

> Mạng neural sinh biểu thức điều phối → GP tiến hóa các biểu thức → chất lượng biểu thức cung cấp tín hiệu học cho mạng neural → lặp lại.

**Mạng LSTM ở đây sinh chuỗi token của chương trình.** Nó không mặc nhiên học lịch sử vận hành cảng như một mô hình dự báo chuỗi thời gian. Khi triển khai, biểu thức đã học nhận trạng thái hiện tại và chấm điểm các công việc.

Có bốn vấn đề cần ghi vào phần đánh giá khả năng tái lập:

- **Dữ liệu:** bài dùng lịch sử vận hành cảng Meishan và simulator riêng. Qua các nguồn đã kiểm tra, chưa xác minh được một bộ tải công khai đầy đủ gồm dữ liệu, simulator và mã GPRR/GPRT.
- **Báo cáo thực nghiệm:** phần mô tả nói có 10 tập train và 10 tập test, nhưng Table I trình bày 5 tập mỗi loại; cần làm rõ trước khi tái lập số liệu.
- **Reward và cập nhật:** biểu thức mang tên “loss” chứa gradient; thành phần covariance và chiều ưu tiên của heuristic cần được kiểm tra bằng ví dụ nhỏ trước khi hiện thực.
- **Diễn giải:** số token ít hơn là chỉ báo về độ phức tạp; chưa đủ chứng minh người vận hành hiểu mô hình tốt hơn. Đổi dữ liệu test cũng chưa tự động chứng minh tổng quát hóa ngoài phân phối.

Vì vậy, khóa luận sẽ triển khai **một biến thể thu gọn lấy cảm hứng từ GPRR**, công khai các khác biệt, thay vì nhận là tái lập nguyên bản.

### Nhóm reference đã lần theo và cần ưu tiên

PDF có **59 references**. Việc tra cứu hiện tập trung vào chuỗi công trình trực tiếp quyết định mô hình và thí nghiệm; chưa xác minh toàn văn cả 59 nguồn. Số `[n]` dưới đây giữ nguyên theo PDF. Nguồn chỉ có abstract/metadata sẽ không được dùng để suy ra chi tiết thuật toán chưa đọc được.

| Ưu tiên | Tài liệu và đường dẫn | Vai trò trong khóa luận |
|---|---|---|
| Bắt buộc | [1] Ouelhadj & Petrovic — [A survey of dynamic scheduling in manufacturing systems](https://link.springer.com/article/10.1007/s10951-008-0090-8) | Phân biệt lập lịch tĩnh, phản ứng theo sự kiện và bất định. Đã xác minh metadata/abstract. |
| Bắt buộc | [21] Zhang và cộng sự — [Genetic Programming for Production Scheduling](https://link.springer.com/book/10.1007/978-981-16-4859-5) | Tài liệu hệ thống về biểu diễn heuristic, fitness và thiết kế GP cho scheduling. Đã xác minh sách và mục lục. |
| Bắt buộc | [32] Chen và cộng sự, CEC 2020 — [Data-driven GP cho điều phối xe cảng](https://people.cs.nott.ac.uk/pszrq/files/CEC2020HGP.pdf) | Hiểu bài toán cảng và cách chuyển GP thành quy tắc điều phối; có PDF công khai. |
| Bắt buộc | [53] Chen và cộng sự — [Cooperative Double-Layer GP](https://nottingham-repository.worktribe.com/output/11756174/cooperative-double-layer-genetic-programming-hyper-heuristic-for-online-container-terminal-truck-dispatching) | Hiểu GP có toán tử logic và phân tách tình huống/heuristic. Metadata ghi tập 27(5), năm 2023; DOI mang năm 2022. |
| Bắt buộc | [54] Chen và cộng sự — [Neural Network Assisted GP, ITSC 2023](https://nottingham-repository.worktribe.com/output/31880883/neural-network-assisted-genetic-programming-in-dynamic-container-port-truck-dispatching) | Tiền thân trực tiếp của hướng GP+RNN. Đã xác minh abstract/DOI; chưa xác minh toàn văn công khai. |
| Bắt buộc | [55] Williams, 1992 — [REINFORCE](https://people.cs.umass.edu/~barto/courses/cs687/williams92simple.pdf) | Cơ sở toán của policy gradient, reward và baseline; có PDF công khai. |
| Bắt buộc | [57] Mundhenk và cộng sự, 2021 — [Neural-guided GP population seeding](https://arxiv.org/abs/2111.00053) | Tài liệu quan trọng nhất để hiểu vòng tương tác neural–GP, elite learning và vấn đề mẫu ngoài policy. |
| Nên đọc | [19] Xu và cộng sự, 2024 — [So sánh GP và RL cho dynamic scheduling](https://ieeexplore.ieee.org/abstract/document/10494009/) | Đặt câu hỏi so sánh công bằng, tránh mặc định mô hình lai luôn tốt hơn. |
| Nên đọc | [46] Jin và cộng sự, 2024 — [Real2Sim cho điều phối xe cảng](https://people.cs.nott.ac.uk/pszrq/files/EJOR23PortDRL.pdf) | Hiểu khác biệt giữa mô phỏng được hiệu chỉnh bằng dữ liệu thật và mô phỏng giả định. |
| Nên đọc | [58] Zhang và cộng sự, 2022 — [DRL hyper-heuristic dưới bất định](https://people.cs.nott.ac.uk/pszrq/files/EJOR21-drl-hh.pdf) | Phân biệt RL chọn heuristic với RL sinh heuristic. |
| Nên đọc | [59] Chen và cộng sự — [DRL-assisted GP ensemble](https://people.cs.nott.ac.uk/pszrq/files/TEVC24-GPHH.pdf) | Nhánh kết hợp khác: RL điều phối một tập heuristic GP; dùng trong related work. |
| Mở rộng | [56] Chen và cộng sự — [Transformer Surrogate GP](https://people.cs.nott.ac.uk/pszrq/files/BIC-TA_2024.pdf) | Transformer dự đoán fitness để giảm chi phí đánh giá; vai trò khác Transformer sinh chương trình trong GPRT. |
| Mở rộng | [26] Vaswani và cộng sự — [Attention Is All You Need](https://arxiv.org/abs/1706.03762) | Nền tảng attention, positional encoding và causal masking. |

Bổ sung các nguồn ngoài danh mục gốc:

- [Sutton & Barto — Reinforcement Learning: An Introduction](https://mitpress.mit.edu/9780262039246/reinforcement-learning/): học chọn lọc chương 3, 5 và 13.
- [Deep Symbolic Regression](https://arxiv.org/abs/1912.04871): hiểu sinh biểu thức bằng RNN và tối ưu bằng reward.
- [Survey so sánh GP và RL cho job-shop scheduling](https://link.springer.com/article/10.1007/s10462-024-11059-9): xây khung tổng quan nghiên cứu.
- [Domain Randomization](https://arxiv.org/abs/1703.06907): cơ sở ý tưởng huấn luyện trên nhiều biến thể mô phỏng. Việc áp dụng vào điều phối xe là lựa chọn nghiên cứu của khóa luận, cần tự kiểm chứng.
- [PGU-SGP, 2025](https://arxiv.org/abs/2504.11280): công trình liên quan về giảm chi phí GP tại cảng, giúp định vị hướng mở rộng.
- [Deep Symbolic Optimization — mã nguồn PyTorch](https://github.com/dso-org/deep-symbolic-optimization-pytorch): tham khảo cách sinh biểu thức và huấn luyện; đây không phải mã GPRR/GPRT của bài gốc.

**Cách ghi chép khi đọc:** mỗi bài có một phiếu gồm bài toán, biểu diễn, vai trò neural/GP, reward, dữ liệu, ngân sách, baseline, kết quả, hạn chế và nội dung có thể tái sử dụng. Lưu riêng trạng thái “đã đọc toàn văn”, “chỉ abstract” và “chưa truy cập”.

## 3. Background lý thuyết và bài tập cần hoàn thành

| Khối kiến thức | Nội dung cần nắm | Bằng chứng đã hiểu |
|---|---|---|
| Scheduling và cảng | QC, YC, xe nội bộ; hàng đợi; ràng buộc tài nguyên; dispatching theo sự kiện; makespan và throughput | Tự vẽ và tính tay lịch của 2 xe, 2 cần cẩu, 4 nhiệm vụ |
| Mô phỏng ngẫu nhiên | Discrete-event simulation, phân phối thời gian phục vụ, seed, biến động trong cùng phân phối và thay đổi phân phối | Chạy lại cùng kịch bản cho kết quả giống nhau; giải thích được từng sự kiện |
| GP | Terminal/function set, tree, prefix notation, crossover, mutation, tournament, elitism và bloat | Chuyển qua lại giữa cây và token; đánh giá biểu thức an toàn |
| RL cơ bản | State, action, policy, episode, return; Monte Carlo; exploration; policy gradient | Cài REINFORCE cho một bài toán nhỏ trước khi ghép với cảng |
| LSTM sinh chương trình | Autoregressive generation, embedding, hidden state, teacher forcing, log-probability, grammar mask | Sinh được 1.000 biểu thức hợp lệ |
| Neural–GP | Neural seeding, elite imitation; phân biệt mẫu on-policy và cá thể GP | Viết rõ dữ liệu nào đi vào từng thành phần loss |
| Tổng quát hóa | Train/validation/test, domain randomization, OOD, leakage | Định nghĩa trước điều kiện nào là “chưa gặp” |
| Đánh giá | Paired comparison, effect size, confidence interval, ngân sách tính toán | Tạo bảng so sánh từ nhiều lần chạy, không chỉ báo cáo lần tốt nhất |

Hai tầng quyết định phải được mô tả riêng:

**Tầng vận hành:** heuristic \(h\) chấm điểm công việc khả thi \(j\) cho xe \(v\):

\[
j^*=\arg\min_{j\in\mathcal A(s,v)}h\!\left(x(s,v,j)\right).
\]

**Tầng học:** LSTM sinh các token tạo thành heuristic:

\[
p_\theta(h)=\prod_{t=1}^{L}p_\theta(z_t\mid z_{<t}).
\]

Ở tầng học, trạng thái là tiền tố biểu thức và thông tin cú pháp; hành động là token tiếp theo; reward nhận được khi biểu thức hoàn chỉnh đã chạy trong simulator. Không truyền gradient xuyên qua simulator.

Reward mặc định:

\[
R(h)=\frac{1}{K}\sum_{k=1}^{K}
\left[
\frac{Q(h,\omega_k)}{Q(\mathrm{STT},\omega_k)}-1
\right],
\]

với \(Q\) là TEU/h, \(\omega_k\) là kịch bản và \(K=4\). Chuẩn hóa bằng STT trên cùng kịch bản giúp so sánh chất lượng tương đối khi độ khó môi trường thay đổi.

Loss của biến thể triển khai:

\[
\mathcal L =
-\mathbb E_{h\sim p_\theta}
[(R(h)-b)\log p_\theta(h)]
+\alpha\,\mathcal L_{\text{elite}}
-\beta\,\mathcal H.
\]

Trong đó:

- REINFORCE chỉ dùng các biểu thức vừa được LSTM lấy mẫu.
- \(\mathcal L_{\text{elite}}\) là negative log-likelihood của nhóm biểu thức tốt do GP tìm được.
- \(b\) là baseline trung bình trượt từ các batch trước, không nhận gradient.
- \(\mathcal H\) khuyến khích khám phá; mặc định \(\alpha=0.1,\ \beta=0.001\).

Đây là lựa chọn triển khai có chủ đích. Reference [57] nêu rõ vấn đề khi sử dụng cá thể GP như mẫu on-policy và thảo luận cách học dựa trên likelihood của các biểu thức được tuyển chọn. [Nguồn phương pháp](https://arxiv.org/pdf/2111.00053).

Phần lý thuyết của khóa luận cần tự trình bày được phép suy ra REINFORCE bằng log-derivative trick và điều kiện kết thúc của biểu thức prefix. Không đặt mục tiêu chứng minh hội tụ toàn cục hoặc đảm bảo tổng quát hóa.

## 4. Thiết kế simulator và mô hình thực tế

### Simulator tối thiểu

Dùng Python, **SimPy** cho mô phỏng sự kiện, **DEAP** cho GP và **PyTorch** cho LSTM. Các thành phần này có tài liệu chính thức về [mô phỏng tài nguyên dùng chung](https://simpy.readthedocs.io/en/latest/), [GP](https://deap.readthedocs.io/en/master/tutorials/advanced/gp.html) và [LSTM](https://docs.pytorch.org/docs/stable/generated/torch.nn.LSTM.html).

Thiết lập danh định:

- 4 QC, 8 YC, 16 xe, 300 nhiệm vụ.
- Mỗi nhiệm vụ chở một container; kích thước 1 hoặc 2 TEU.
- Mỗi xe thực hiện một nhiệm vụ tại một thời điểm; mỗi cần cẩu phục vụ một xe tại một thời điểm.
- Đường đi cố định; heuristic chọn nhiệm vụ, không tối ưu tuyến đường.
- Danh sách công việc biết trước; thời gian phục vụ ngẫu nhiên tạo động lực điều phối trực tuyến.
- Giữ thứ tự công việc tại từng QC; mặc định \(q=1\), đơn giản hóa so với cơ chế đổi thứ tự trong bài.
- Thời gian phục vụ dùng lognormal; cấu hình ban đầu có trung bình QC 60 giây, YC 90 giây, hệ số biến thiên 0,2. Đây là **tham số tổng hợp**, không phải ước lượng từ Meishan.

Chu trình nhiệm vụ:

> Xe rảnh → chọn và giữ chỗ nhiệm vụ → chạy tới điểm lấy → đợi và được phục vụ → vận chuyển → đợi và được phục vụ tại điểm trả → xe rảnh.

Các sự kiện đồng thời được xử lý theo thứ tự xác định. Nhiệm vụ được giữ chỗ ngay khi phân công, tránh hai xe nhận cùng việc. Không có công việc khả thi thì simulator chuyển tới sự kiện tiếp theo.

Tính:

\[
Q=3600\,\frac{\sum_j \mathrm{TEU}_j}{C_{\max}-t_0},
\]

khi thời gian tính bằng giây. Dùng cùng mốc bắt đầu \(t_0=0\) cho mọi phương pháp để tránh thay đổi mẫu số do heuristic trì hoãn phục vụ.

### Biểu diễn heuristic

Đầu vào gồm tám đặc trưng quan sát được:

1. Thời gian chạy rỗng tới điểm lấy.
2. Thời gian vận chuyển dự kiến.
3. Số xe đợi tại điểm lấy.
4. Số xe đợi tại điểm trả.
5. Số xe đang được phân công tới QC liên quan.
6. Số nhiệm vụ còn lại của QC.
7. Thời gian phục vụ trung bình tại điểm lấy.
8. Thời gian phục vụ trung bình tại điểm trả.

Dùng tỷ lệ hàng đợi/số xe, công việc còn lại/tổng công việc ban đầu và thang thời gian cố định. Không đưa thời lượng phục vụ ngẫu nhiên chưa xảy ra vào feature.

Function set:

\[
\{+,-,\times,\mathrm{pdiv},\min,\max,
\mathrm{if\_le}(a,b,c,d)\}.
\]

`if_le` trả về \(c\) nếu \(a\le b\), ngược lại trả về \(d\). Hằng số là \(\{-1,0,0.5,1,2\}\). Giới hạn 63 token, độ sâu 6, chia bảo vệ và chặn giá trị số vượt ngưỡng.

Mọi phương pháp sử dụng chung feature, function set và giới hạn biểu thức. Điểm hòa được xử lý bằng ID nhiệm vụ. Nếu biểu thức lỗi số, dùng STT và ghi lại lỗi.

### GP và LSTM

**GP mặc định:** population 64, tournament 5, giữ 2 elite; xác suất crossover/mutation/reproduction là 0,6/0,3/0,1 trên phần quần thể còn lại.

**LSTM mặc định:** một lớp, hidden size 32, embedding 32, Adam với learning rate \(10^{-3}\), gradient clipping 1,0. Mask token dựa trên số đối số còn thiếu, độ sâu và số token còn lại.

Một vòng mô hình lai:

1. LSTM sinh 64 biểu thức hợp lệ.
2. Đánh giá chúng trên cùng batch 4 kịch bản.
3. Dùng chúng khởi tạo GP, tiến hóa 3 thế hệ.
4. Giữ các biểu thức tốt nhất để cập nhật kho ứng viên.
5. Cập nhật REINFORCE từ 64 mẫu LSTM ban đầu.
6. Học likelihood từ 10% cá thể GP tốt nhất.
7. Chuyển batch kịch bản và lặp lại đến hết ngân sách.

Khi vận hành, xuất một biểu thức đã chọn trên validation. Việc điều phối chỉ cần đánh giá biểu thức; không phải chạy lại tiến hóa hoặc lấy mẫu LSTM cho mỗi xe.

### Cải tiến chính: huấn luyện với biến thiên môi trường

Giữ nguyên mô hình, loss và ngân sách; thay đổi phân phối kịch bản huấn luyện:

| Yếu tố | Huấn luyện danh định | Huấn luyện có randomization |
|---|---|---|
| Số xe | 16 | Chọn từ 12, 16, 20 |
| Số nhiệm vụ | 300 | Chọn từ 200, 300, 400 |
| Hệ số thời gian phục vụ | 1,0 | Uniform từ 0,8 đến 1,2 |
| Hệ số biến thiên phục vụ | 0,2 | Uniform từ 0,1 đến 0,3 |
| Số QC/YC | 4/8 | 4/8 |

Cả hai chế độ đều có nhiều seed ngẫu nhiên. Như vậy, thí nghiệm phân biệt được **thêm biến thiên phân phối** với việc đơn giản là chạy thêm các mẫu ngẫu nhiên.

Các interface tối thiểu:

- `evaluate(rule, scenario, seed) -> metrics`
- `score(features) -> priority`
- `generate(batch_size) -> expressions, log_probs`
- `train(config) -> checkpoint, candidate_archive, logs`

Đầu ra phải lưu biểu thức, checkpoint, cấu hình, seed, phiên bản thư viện, số lần đánh giá và thời gian chạy. Cache fitness chỉ được dùng với khóa gồm biểu thức, kịch bản, seed và phiên bản simulator.

## 5. Thí nghiệm, kiểm chứng và tiêu chí hoàn thành

### Các phương pháp so sánh

Thiết kế chính là ma trận \(2\times2\):

| Mã | Phương pháp | Câu hỏi được kiểm tra |
|---|---|---|
| A | GP, huấn luyện danh định | Baseline học |
| B | GP, có randomization | Randomization có giúp GP không? |
| C | GP+LSTM, huấn luyện danh định | Đóng góp của mô hình lai |
| D | GP+LSTM, có randomization | Hiệu quả kết hợp |

Bổ sung FIFO, STT và MTR làm heuristic tham chiếu. Một ablation **LSTM+REINFORCE không có GP**, chạy ở chế độ randomization, kiểm tra vai trò của tiến hóa.

Không đưa DQN, PPO, CDGP và ensemble vào phần triển khai bắt buộc trong 12 tuần.

### Tách dữ liệu và kiểm tra tổng quát hóa

- Sinh trước 64 kịch bản train và 16 kịch bản validation cho mỗi chế độ.
- Dùng cùng ngân sách validation và quy tắc chọn biểu thức cho mọi phương pháp.
- Test có 20 kịch bản mỗi nhóm, seed độc lập; không dùng để chọn cấu hình hoặc checkpoint.
- Test danh định dùng cùng điều kiện với môi trường danh định nhưng nhiệm vụ và thời lượng mới.
- Test OOD gồm: 8 xe; thời gian phục vụ tăng 50%; 800 nhiệm vụ; thay lognormal bằng gamma cùng trung bình/phương sai; mở rộng lên 8 QC, 16 YC, 32 xe và 800 nhiệm vụ.

Các trường hợp ngoài miền train phải được gắn nhãn riêng. Khi so sánh hai phương pháp, cùng kịch bản sử dụng cùng ngẫu nhiên ngoại sinh theo nhiệm vụ và công đoạn, tránh việc thứ tự gọi RNG làm thay đổi bài toán.

### Ngân sách thực nghiệm

Mỗi phương pháp học chạy **5 training seed**. Ngân sách mặc định là **2.048 lượt đánh giá ứng viên**, mỗi lượt trên 4 kịch bản; các bản sao cũng được tính vào ngân sách tương đương.

Trước thí nghiệm chính, dùng pilot để chọn chung một mức trong \(\{512,1024,2048\}\), lấy mức lớn nhất dự kiến hoàn thành trong khoảng 90 phút/lần chạy trên môi trường đã chọn. Cố định mức đó cho tất cả phương pháp; báo cáo thêm thời gian thực và số lượt mô phỏng thực tế.

Ngân sách validation được ghi riêng và giữ bằng nhau. Lưu checkpoint sau mỗi vòng để phục hồi khi Colab ngắt phiên.

### Chỉ số và cách kết luận

Chỉ số chính là **phần trăm cải thiện TEU/h so với STT trên từng kịch bản**. Chỉ số bổ sung gồm makespan, thời gian xe chờ, QC utilization, độ dài/độ sâu biểu thức, thời gian huấn luyện và độ trễ quyết định.

Báo cáo:

- Trung bình và độ phân tán qua các training seed.
- Chênh lệch ghép cặp trên cùng kịch bản.
- Khoảng tin cậy 95% bằng bootstrap hai cấp: training run và scenario.
- Kết quả riêng từng nhóm OOD, không chỉ trung bình gộp.
- Đường cong chất lượng theo lượt đánh giá và theo thời gian.

Không coi mọi lần đo từ cùng một mô hình là quan sát độc lập. Với 5 training seed, kết luận thống kê cần phản ánh mức bất định còn lớn.

### Các kiểm tra bắt buộc

- Không phân công trùng nhiệm vụ hoặc sử dụng một xe/cần cẩu đồng thời cho hai công việc.
- Trường hợp nhỏ khớp kết quả tính tay; cùng seed tái lập được kết quả.
- Cây và chuỗi prefix cho kết quả tương đương; biểu thức sinh ra hợp lệ.
- Mô hình không truy cập dữ liệu tương lai, seed test hoặc tham số ngẫu nhiên ẩn.
- REINFORCE làm tăng xác suất lựa chọn có reward cao trong bài toán thử đơn giản.
- Tách đúng gradient của reward/baseline; không đưa mẫu GP vào nhánh on-policy.
- Heuristic được chọn trên validation trước khi mở test.
- Không đơn giản hóa biểu thức bằng phép biến đổi làm thay đổi ngữ nghĩa của protected division.

**Khóa luận đạt yêu cầu khi** hệ thống chạy và tái lập được, có đủ đối chứng để trả lời RQ1–RQ3, báo cáo trung thực cả kết quả không cải thiện, và giải thích được một số quyết định cụ thể của heuristic. Mục tiêu không phải đạt một tỷ lệ thắng được đặt trước.

## 6. Lịch 12 tuần và sản phẩm bàn giao

| Tuần | Công việc chính | Sản phẩm cần hoàn thành |
|---|---|---|
| 1 | Đọc bài chính, [32], [53]; lập danh mục và phân loại 59 references | Đề cương, bảng literature, đặc tả bài toán và danh sách khác biệt với bài gốc |
| 2 | Học MDP, return, REINFORCE; luyện ví dụ điều phối nhỏ | Notebook RL đơn giản và bản suy ra policy gradient |
| 3 | Xây simulator và kiểm tra tài nguyên, thứ tự công việc | Simulator chạy đúng; FIFO/STT/MTR và event log |
| 4 | Xây feature, biểu thức và GP; chạy pilot | GP baseline, kiểm tra parser, ngân sách tính toán được khóa |
| 5 | Xây LSTM sinh prefix với grammar mask | Bộ sinh biểu thức hợp lệ và notebook giải thích xác suất chuỗi |
| 6 | Ghép REINFORCE với simulator; đọc sâu [55], [57] | LSTM học được trên môi trường nhỏ; kiểm tra reward và gradient |
| 7 | Ghép vòng neural–GP và elite learning | Mô hình lai hoàn chỉnh, checkpoint và log |
| 8 | Thêm randomization; hoàn thiện split và đóng băng cấu hình | Ma trận A–D chạy được; validation và test manifest |
| 9 | Chạy thí nghiệm chính với 5 seed | Kết quả thô, thời gian chạy, biểu thức đã chọn |
| 10 | Chạy OOD, ablation; phân tích thống kê | Bảng so sánh, biểu đồ, phân tích trường hợp thất bại |
| 11 | Viết và chỉnh sửa khóa luận; dựng demo từ event log | Bản thảo đầy đủ, demo so sánh hai heuristic |
| 12 | Tái chạy một cấu hình từ đầu; kiểm tra trích dẫn; luyện bảo vệ | Mã hoàn chỉnh, hướng dẫn, khóa luận và slide |

Viết báo cáo song song từ tuần 1; tuần 11 dành cho tổng hợp và chỉnh sửa.

Các mốc kiểm soát:

- **Cuối tuần 4:** simulator và GP phải hoạt động. Nếu chậm, giảm số nhiệm vụ để phát triển, giữ nguyên ràng buộc vận hành.
- **Cuối tuần 7:** mô hình lai phải chạy hết một thí nghiệm nhỏ. Nếu chất lượng chưa tốt, vẫn giữ hướng nghiên cứu, phân tích nguyên nhân và dùng ngân sách đã định.
- **Cuối tuần 8:** ngừng thêm kiến trúc; khóa thiết kế test.
- **Cuối tuần 10:** ngừng chỉnh model dựa trên test; chuyển sang phân tích và viết.

Sản phẩm cuối cùng gồm:

1. Khóa luận sáu chương: giới thiệu; background/related work; bài toán và simulator; phương pháp; thực nghiệm; kết luận.
2. Kho mã có cấu hình và hướng dẫn tái lập trên laptop/Colab.
3. Bộ kịch bản tổng hợp với train/validation/test và provenance rõ ràng.
4. Checkpoint, các biểu thức tốt, dữ liệu kết quả thô và script tạo bảng/hình.
5. Thư mục tài liệu hợp pháp, bibliography và phiếu đọc các nguồn trọng tâm.
6. Demo phát lại quá trình điều phối, hiển thị hàng đợi, nhiệm vụ được chọn và điểm của heuristic.

**Giả định được chốt:** toàn bộ dữ liệu thực nghiệm ban đầu là dữ liệu tổng hợp; đóng góp chính là đánh giá tổng quát hóa; mô hình bắt buộc là GP+LSTM; việc tiếp cận dữ liệu cảng thật và phát triển Transformer được để ở hướng tiếp nối.
