-RL là thuật toán reinforce, kết hợp với Transformer là poilcy neural -> DRL
-Vì Transformer là mạng neural và được cập nhật theo reward, cách gọi phù hợp là deep reinforcement learning (DRL),
cụ thể là policy-gradient với policy neural.

-RL framework chứa 4 thành phần chính: environment, poilcy, state, reward
environment là bài toán gốc VRPTW, policy là Transformer, state là trạng thái vận hành hiện tải của bài toán,
reward là điểm fitness tính từ cây Heuristic

-Hàm mục tiêu chỉ được tính khi hoàn thành xếp lịch --> delayed reward

reward = ei−1−si−δ·cov(Or, Om)

REINFORCE tính:\[
L*{\text{policy}}=-\frac{1}{B}\sum_i A_i\log \pi*\theta(x*i)
\]với \(x_i\) là chuỗi token của cây, \(\log \pi*\theta(x_i)\) là log-xác suất Transformer đã sinh chuỗi đó, và \(A_i\) là advantage từ reward trừ baseline, có chuẩn hóa theo cấu hình.

---Transformer:
-Transformer không trực tiếp tương tác với môi trường mà thông qua Heuristic mà nó sinh ra.

-Transformer hiện tại có thể attention tới bất kỳ token nào trong prefix đã sinh trước đó.

-Transformer hỗ trợ attention toàn cục trên prefix và xử lý song song các vị trí trong mỗi lần forward; việc giải mã biểu thức vẫn diễn ra tuần tự theo token

-Implementation thực tế tương ứng với Pre-LN Transformer

-Experiment design:
The GP configuration features
a population size of 1024, with crossover, mutation, and
reproduction rates set at 60%, 30%, and 10%, respectively.
The fitness function within the GP is denoted by equation
(4). All algorithms undergo 500 training generations, with the
RNN’s learning rate set at 0.001. Consistent with the GPRR
and GPRT frameworks shown in Fig. 8 and following the
methodology described in [57], K is set at 20, indicating the
RNN’s involvement every 20 GP training iterations. The size
of the GP population, M, equals 1024, and N is set at half
of M, which is 512.
Each algorithm was executed 100 times using different random seeds to ensure robustness.

-GP chỉ giữ một số cây Elite từ Transformer (bật archive_seeding thì sẽ giữ 1 số cây tốt từ chu kỳ trước)

-GPRT là bản cải thiện nhờ nắm bắt phụ thuộc dài hạn
