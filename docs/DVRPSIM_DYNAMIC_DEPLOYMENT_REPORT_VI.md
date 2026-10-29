# Báo cáo cấu hình, cài đặt và tích hợp dvrpsim với GPRT/GP cho VRPTW động

**Ngày lập:** 27-10-2026  
**Phạm vi:** Thử nghiệm zero-shot trên Solomon VRPTW với thời điểm xuất hiện đơn hàng không đồng bộ.

## 1. Tóm tắt

Project dùng **dvrpsim 1.0.0** làm mô phỏng sự kiện rời rạc cho nhánh đánh giá định tuyến động. Dữ liệu lõi vẫn là 56 instance Solomon VRPTW đã chia 32 train, 12 validation và 12 test. Mỗi khách Solomon được biến thành một đơn hàng có thời điểm xuất hiện riêng. Khi đơn xuất hiện hoặc xe trở lại depot, callback lấy trạng thái hiện tại, tạo tuyến khả thi cho xe rảnh và dùng heuristic đã học để xếp thứ tự phục vụ.

GPRT và GP được huấn luyện ngoại tuyến trên pipeline VRPTW hiện có. Trong mô phỏng động, chương trình nạp biểu thức heuristic đã được chọn bằng validation rồi giữ nguyên biểu thức đó; Transformer không sinh lại tuyến ở mỗi sự kiện, GP cũng không tiến hóa lại trên test. Cách này đánh giá khả năng chuyển heuristic từ bài toán Solomon tĩnh sang thông tin đơn hàng đến dần.

Các lựa chọn cấu hình, adapter và runner được lưu tại [configs/dvrpsim_dynamic.json](../configs/dvrpsim_dynamic.json), [dvrpsim_dynamic/simulator.py](../dvrpsim_dynamic/simulator.py) và [run_dvrpsim_dynamic.py](../run_dvrpsim_dynamic.py). Protocol và kết quả đã lưu được mô tả thêm trong [kế hoạch đánh giá động](DVRPSIM_DYNAMIC_VRPTW_PLAN_VI.md) và [báo cáo kết quả](DVRPSIM_DYNAMIC_VRPTW_RESULTS_VI.md).

## 2. Kiến trúc triển khai

~~~mermaid
flowchart LR
    A[Solomon VRPTW] --> B[Ánh xạ depot, khách hàng, xe, đơn]
    C[Thời điểm release theo seed] --> D[dvrpsim: mô phỏng sự kiện]
    B --> D
    E[Heuristic GPRT đã chọn trên validation] --> F[Adapter và bộ dựng tuyến chung]
    G[Heuristic GP đã chọn trên validation] --> F
    H[EDD baseline] --> F
    D -->|đơn mới / xe về depot| F
    F -->|quyết định tuyến và gán đơn| D
    D --> I[Checker độc lập và log]
    I --> J[CSV, JSONL, metrics]
~~~

Simulator quản lý đồng hồ mô phỏng, trạng thái xe, trạng thái đơn và các sự kiện. Adapter của project quản lý logic chọn khách tiếp theo, điều kiện khả thi, gán đơn cho xe và đo kết quả. Việc tách hai lớp giúp giữ dvrpsim như môi trường thực thi, đồng thời dùng chung bộ dựng tuyến và bộ feature của GP/GPRT.

## 3. Cài đặt môi trường

### 3.1. Phụ thuộc đã khóa

File [requirements.txt](../requirements.txt) hiện có:

| Gói | Ràng buộc trong project | Vai trò |
|---|---:|---|
| Python | dvrpsim yêu cầu từ 3.8 trở lên | Môi trường chạy |
| PyTorch | torch>=2.2 | Mô hình GPRT và pipeline học |
| NumPy | numpy>=1.24 | Tính toán số |
| dvrpsim | dvrpsim==1.0.0 | Mô phỏng động |

Để tái lập kết quả của project, cần giữ đúng dvrpsim 1.0.0 thay vì cài phiên bản không khóa. Gói trên PyPI là wheel Python dùng được trên nhiều hệ điều hành và được phát hành theo giấy phép MIT. Python tối thiểu theo metadata của gói là 3.8; phiên bản Python thực tế còn phải tương thích với bản PyTorch được cài.

### 3.2. Windows PowerShell

Từ thư mục gốc project, tạo môi trường riêng và cài các phụ thuộc:

~~~powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
~~~

Kiểm tra phiên bản dvrpsim đã cài:

~~~powershell
.\.venv\Scripts\python.exe -c "from importlib.metadata import version; print(version('dvrpsim'))"
~~~

Kết quả mong đợi là 1.0.0. Dùng trực tiếp Python trong .venv giúp chạy đúng môi trường mà không phụ thuộc việc PowerShell có cho phép kích hoạt script hay không.

### 3.3. Google Colab hoặc Linux

Sau khi đưa/cloning project vào môi trường làm việc:

~~~bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -c "from importlib.metadata import version; print(version('dvrpsim'))"
~~~

Nếu chỉ cần thử riêng thư viện, lệnh cài theo hướng dẫn upstream là python -m pip install dvrpsim; khi chạy thí nghiệm của project, nên dùng requirements đã khóa phiên bản.

### 3.4. Điều kiện dữ liệu và artifact trước khi chạy

Runner yêu cầu:

- Dữ liệu Solomon tại data/data/Solomon, đúng 56 instance.
- Split manifest tại outputs/gprt_vrptw/reduced_main/split_manifest.json.
- Năm artifact GPRT và năm artifact GP-only có file final_heuristic.txt trong cây thư mục outputs/gprt_vrptw/reduced_main/{gprt,gp}/<training_seed>/.
- Các artifact phải đến từ training run tương ứng và đã được chọn bằng validation. Nếu thiếu một trong hai nhóm hoặc thiếu đủ năm seed, runner dừng thay vì tự tạo heuristic thay thế.

Runner ghi hash của biểu thức, checkpoint nếu có, instance, cấu hình và split vào manifest để truy nguyên chính sách đã dùng.

## 4. Cấu hình mô phỏng hiện hành

Cấu hình nằm trong [configs/dvrpsim_dynamic.json](../configs/dvrpsim_dynamic.json). Các giá trị quan trọng:

| Tham số | Giá trị | Ý nghĩa |
|---|---|---|
| Phiên bản simulator | 1.0.0 | Runner kiểm tra phiên bản cài đặt trước khi chạy |
| Khoảng cách | Euclid, không làm tròn | Khớp quy ước Solomon đang dùng trong project |
| Thời gian di chuyển | Bằng khoảng cách | Một đơn vị khoảng cách tương ứng một đơn vị thời gian mô phỏng |
| Chính sách dispatch | Khi đơn release hoặc xe về depot | Không đổi tuyến sau khi xe rời depot |
| release_window_fraction | 0.5 | Sinh thời điểm release trong nửa đầu chân trời depot, rồi giới hạn để giữ khả năng phục vụ trực tiếp |
| Pilot | C102, C206, R104, RC203; 2 seed | Bước kiểm tra trên validation |
| Đánh giá cuối | 12 instance test; seed 20261025–20261027 | Tập test được giữ ngoài bước chọn heuristic |
| Phương pháp | GPRT, GP-only, EDD | Hai heuristic học được và baseline hạn giao sớm nhất |

Chế độ đối chứng tĩnh đặt tất cả release_date = 0. Chế độ động dùng seed để tạo release time có thể tái lập. Với chân trời depot H, khoảng cách depot-khách d_0i, khoảng cách khách-depot d_i0, thời gian phục vụ s_i, và hạn bắt đầu phục vụ muộn nhất b_i, release của khách i được tạo theo:

~~~text
u_i ~ Uniform(0, 0.5 H)
release_i = min(u_i, b_i - d_0i, H - d_0i - s_i - d_i0)
~~~

Trước khi chấp nhận scenario, adapter xác nhận rằng ngay cả khi đơn xuất hiện ở thời điểm 0, xe vẫn có thể đến trong time window và quay lại depot trước chân trời. Nếu không đạt, episode báo lỗi thay vì âm thầm sửa dữ liệu. release_date là thời điểm đơn được biết đến; ready_time là thời điểm sớm nhất được bắt đầu phục vụ. Hai đại lượng này có ý nghĩa khác nhau.

## 5. Ánh xạ Solomon VRPTW sang dvrpsim

| Solomon/project | Đối tượng dvrpsim | Thiết lập trong adapter |
|---|---|---|
| Depot | Location | L000, tọa độ depot |
| Khách hàng | Location | L001, L002, … giữ tọa độ gốc |
| Khách cần giao | Order | Pickup ở depot, delivery tại location khách |
| Nhu cầu | Quantity của order | Dùng để tải xe và kiểm tra sức chứa |
| Fleet | Vehicle | Tạo số xe theo instance; mọi xe bắt đầu tại depot |
| Capacity | Capacity của vehicle | Giữ nguyên capacity trong instance |
| ready_time | earliest_delivery_start | Xe đến sớm phải chờ |
| due_date | latest_delivery_start | Adapter tự lọc và checker độc lập kiểm tra hạn bắt đầu phục vụ |
| service_time | delivery_duration | Thời gian giao hàng tại khách |
| Thời điểm khách xuất hiện | Order.release_date | Thời điểm request được đưa vào mô phỏng |
| Khoảng cách | Callback của vehicle | Euclid không làm tròn |
| Thời gian đi | Callback của vehicle | Bằng Euclid distance |

Trong Solomon, due_date được diễn giải là **hạn bắt đầu dịch vụ**, không phải hạn hoàn thành dịch vụ. Tài liệu dvrpsim mô tả earliest service start là ràng buộc cứng khiến xe phải chờ nếu đến sớm; vì thế adapter cũng phải tự thực thi giới hạn latest service start. Bộ lọc ứng viên kiểm tra service_start <= due_date và khả năng trở về depot trong chân trời. Sau mô phỏng, checker tính lại các ràng buộc từ route/service event độc lập với quyết định của policy.

## 6. Cách GPRT và GP tham gia điều phối động

### 6.1. Nạp heuristic đã học

Adapter đọc final_heuristic.txt từ từng training seed, phân tích biểu thức bằng parser giới hạn theo grammar của project và chuyển thành cây cú pháp. Đây là biểu thức được runner huấn luyện/chọn trước đó, không phải mã Python tùy ý. Mỗi biểu thức có SHA-256 trong manifest.

- **GPRT:** Transformer được dùng ở giai đoạn học để đề xuất biểu thức; GP tinh chỉnh quần thể. Biểu thức được chọn trên validation là policy lúc mô phỏng.
- **GP-only:** dùng cùng pipeline chấm điểm, feature, bộ dựng tuyến và ngân sách chạy động, nhưng không nạp phần Transformer.
- **EDD:** baseline không học; chọn khách đủ điều kiện có due date nhỏ nhất, phá hòa theo mã khách.

Do đó phép thử động đo chất lượng chuyển giao của heuristic, không đo khả năng Transformer phản ứng trực tiếp bằng cách sinh token ở mỗi callback.

### 6.2. Xử lý mỗi sự kiện

Khi dvrpsim yêu cầu định tuyến, callback đọc state hiện tại:

1. Lấy các order đã release và chưa được gán xe. Adapter kiểm tra các order nhìn thấy không chứa khách có release time trong tương lai.
2. Tìm xe rảnh tại depot. Xe đang chạy giữ nguyên route đã nhận.
3. Với từng xe rảnh, bắt đầu route tại depot, khởi tạo thời gian hiện tại và capacity.
4. Tạo CandidateContext cho khách đang chờ. Loại ứng viên vượt capacity, bắt đầu phục vụ sau due date, hoặc không thể quay lại depot kịp.
5. Tính điểm các ứng viên bằng cùng evaluator/feature với VRPTW tĩnh. Chọn điểm nhỏ nhất; nếu hòa, chọn mã khách nhỏ hơn. GP/GPRT dùng biểu thức đã nạp; EDD dùng due date.
6. Lặp đến khi không còn ứng viên khả thi hoặc capacity đã dùng hết.
7. Trả route cho simulator theo thứ tự: depot lấy hàng, các điểm giao, depot quay về. Các đơn được gán cho xe tương ứng.
8. Khi xe về depot, nếu còn đơn đã release nhưng chưa gán, callback được gọi để tạo route kế tiếp.

Như vậy GPRT và GP tác động ở bước **xếp hạng khách hàng khả thi**. Bộ dựng tuyến chuyển thứ hạng thành route và dvrpsim thực thi route theo thời gian mô phỏng. Chỉ các khách đã release được xét; không dùng thông tin tương lai.

### 6.3. Giới hạn chính sách trong nhánh hiện tại

- Không chèn đơn mới vào route của xe đang chạy.
- Không tái tối ưu route giữa chuyến và không thay đổi quyết định đã thực hiện.
- Chỉ phát sinh quyết định khi có đơn mới hoặc xe quay lại depot.
- Không mô phỏng hủy đơn, giao thông biến thiên, thời gian di chuyển ngẫu nhiên hay dữ liệu release từ vận hành thật.
- Heuristic không được cập nhật bằng RL/GP trên test; đây là đánh giá zero-shot với release time synthetic.

## 7. Chạy thí nghiệm và đầu ra

Từ thư mục gốc project, dùng Python của môi trường đã cài:

~~~powershell
python run_dvrpsim_dynamic.py --phase pilot
python run_dvrpsim_dynamic.py --phase final
~~~

Có thể dùng --phase all để chạy tuần tự cả pilot và final. Mặc định đầu ra được ghi vào outputs/dvrpsim_dynamic/. Runner xóa các file đầu ra cũ trong phase đang chạy trước khi bắt đầu; để giữ lại lần chạy cũ, chỉ định thư mục khác:

~~~powershell
python run_dvrpsim_dynamic.py --phase final --output-dir outputs/dvrpsim_dynamic/rerun_01
~~~

Các file chính:

| File | Nội dung |
|---|---|
| run_manifest.json | Phiên bản dvrpsim, hash cấu hình/split/instance, artifact policy và tham số scenario |
| episode_manifest.jsonl | Policy, seed huấn luyện, seed scenario, biểu thức/hash và release time theo khách |
| events.jsonl | Lịch sử release, quyết định tuyến, xe đi, bắt đầu dịch vụ và giao hàng |
| episodes.csv | Metrics từng episode |
| summary_by_policy.csv | Tổng hợp theo policy, seed huấn luyện và loại scenario |
| paired_comparisons.csv | Chênh lệch ghép cặp theo cùng instance và release seed |
| gate_report.json | Số episode, vi phạm, tỷ lệ đúng hạn và độ trễ callback |

Runner dừng nếu phiên bản package sai, split không khớp hoặc checker phát hiện constraint violation. Pilot chỉ dùng validation. Giai đoạn final chạy 12 test instance × 3 release seed, bao gồm các chính sách và đối chứng tĩnh được khóa trước.

## 8. Kết quả đã lưu và cách diễn giải

Theo [báo cáo kết quả hiện có](DVRPSIM_DYNAMIC_VRPTW_RESULTS_VI.md), 132/132 pilot episode và 528/528 test episode qua checker độc lập, không ghi nhận vi phạm capacity, time window, thời điểm release, coverage hoặc tính lại khoảng cách. Tóm tắt kết quả động trên tập test:

| Policy | Đơn phục vụ đúng hạn | Quãng đường trung bình/episode | Quãng đường/đơn phục vụ |
|---|---:|---:|---:|
| GPRT | 78.88% | 3,829.30 | 48.12 |
| GP-only | 79.07% | 3,871.16 | 48.53 |
| EDD | 79.81% | 3,943.11 | 49.09 |

GPRT đi quãng đường ngắn hơn nhẹ nhưng tỷ lệ phục vụ cũng thấp hơn nhẹ so với GP và EDD. Vì vậy kết quả hiện tại **chưa chứng minh GPRT tốt hơn trong kịch bản động**. Không dùng quãng đường thấp đơn lẻ làm bằng chứng cải thiện nếu policy cũng phục vụ ít đơn hơn. Callback động có p95 trung bình khoảng 0.94 ms trong môi trường đã chạy; đây là số đo của môi trường cụ thể, không phải bảo đảm thời gian thực phổ quát.

## 9. Ghi chú tái lập tại workspace

requirements.txt đã khóa dvrpsim 1.0.0 và artifacts pilot/final đã có trong outputs/dvrpsim_dynamic/. Tuy nhiên, tại thời điểm lập báo cáo, Python đang được chọn trong phiên làm việc là 3.13.7 và chưa nhận diện package dvrpsim đã cài. Vì vậy các kết quả trong báo cáo được đọc từ artifacts có sẵn; chưa thực hiện chạy lại từ môi trường đang hoạt động. Sau khi cài requirements.txt, runner sẽ xác nhận version package trước khi chạy.

## 10. Tài liệu tham khảo

1. Repository chính thức và hướng dẫn cài đặt: [sztaki-hu/dvrpsim](https://github.com/sztaki-hu/dvrpsim).
2. Metadata và release 1.0.0 trên PyPI: [dvrpsim 1.0.0](https://pypi.org/project/dvrpsim/1.0.0/).
3. Tài liệu khái niệm, Model, Location, Order, Vehicle: [Basic concepts](https://sztaki-hu.github.io/dvrpsim/src/introduction/basics.html).
4. Tài liệu đặt release date và yêu cầu routing khi order xuất hiện: [Our first orders](https://sztaki-hu.github.io/dvrpsim/src/tutorial/tutorial_02.html).
5. Callback định tuyến, trạng thái xe và route response: [Our first routing callback](https://sztaki-hu.github.io/dvrpsim/src/tutorial/tutorial_03b.html).
6. Service duration và earliest service start: [Order-related features](https://sztaki-hu.github.io/dvrpsim/src/tutorial/tutorial_04.html).
7. Horváth, M., & Tamási, T. (2025). *A general modeling and simulation framework for dynamic vehicle routing*. EURO Journal on Transportation and Logistics, 14, 100159. [DOI: 10.1016/j.ejtl.2025.100159](https://doi.org/10.1016/j.ejtl.2025.100159).

BibTeX theo repository upstream:

~~~bibtex
@article{horvath2025general,
  title={A general modeling and simulation framework for dynamic vehicle routing},
  author={Horv{\'a}th, Mark{\'o} and Tam{\'a}si, T{\'\i}mea},
  journal={EURO Journal on Transportation and Logistics},
  volume={14},
  pages={100159},
  year={2025},
  publisher={Elsevier}
}
~~~
