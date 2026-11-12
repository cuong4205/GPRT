# Script thuyết trình: Tích hợp heuristic GPRT vào DVRPSim

*Ghi chú — không đọc: Bản chỉnh tập trung vào framework triển khai. Phần kết quả thực nghiệm ở slide 10–13 được lược bỏ; giữ số slide để ghép với bản trình chiếu. Các số lượng ở slide 9 mô tả thiết kế so sánh, không phải kết quả chạy.*

## Mở đầu

Kính thưa thầy cô, em là Phan Nguyễn Mạnh Cường. Em xin trình bày cách tích hợp heuristic GPRT vào DVRPSim để điều phối xe khi đơn hàng xuất hiện theo thời gian, có xét tải trọng và khung thời gian phục vụ.

Trước hết, em giới thiệu mô hình bài toán và cơ chế điều phối.

## Slide 1. Mục tiêu đánh giá trong môi trường động

GPRT sinh các biểu thức ưu tiên để dựng tuyến trên bài toán VRPTW tĩnh, trong đó toàn bộ khách hàng được biết ngay từ đầu.

Trong môi trường động, đơn hàng xuất hiện dần và có thể có yêu cầu hủy, trong khi xe đang thực hiện chuyến đã nhận.

Mục tiêu là đánh giá khả năng áp dụng biểu thức đã học sang điều kiện này: khi thông tin đến dần, mức phục vụ và chi phí thay đổi như thế nào?

**Chuyển ý:** Sau đây là vai trò của từng thành phần trong hệ thống.

## Slide 2. Kiến trúc tích hợp GPRT–DVRPSim

Hệ thống gồm kịch bản đầu vào, DVRPSim và bộ điều phối dùng biểu thức GPRT.

Kịch bản quy định dữ liệu khách hàng, thời điểm xuất hiện đơn và yêu cầu hủy. DVRPSim quản lý đồng hồ, trạng thái đơn và hoạt động của xe. Bộ điều phối đọc trạng thái hiện tại, dựng tuyến và trả kế hoạch cho DVRPSim thực thi.

GPRT được huấn luyện trước: Transformer sinh biểu thức, GP tiến hóa biểu thức, còn RL cập nhật bộ sinh. Khi mô phỏng, hệ thống chỉ nạp biểu thức đã chọn để chấm điểm khách hàng; không huấn luyện lại tại mỗi sự kiện.

**Chuyển ý:** Kịch bản động được tạo từ dữ liệu Solomon như sau.

## Slide 3. Tạo kịch bản động từ Solomon

Với các instance Solomon 100 khách, mỗi khách tương ứng một đơn hàng. Hệ thống giữ tọa độ, nhu cầu, giới hạn đội xe, sức chứa và khung thời gian phục vụ.

Trong cấu hình hiện tại, 70 đơn được chọn ngẫu nhiên để áp dụng luật động. Thời điểm xuất hiện bằng thời điểm mở cửa sổ phục vụ trừ khoảng báo trước từ 10 đến 15, rồi chặn dưới tại 0. Ba mươi đơn còn lại xuất hiện ngay từ đầu.

Vì có bước chặn tại 0, không phải cả 70 đơn đều xuất hiện sau thời điểm 0.

Hệ thống chọn độc lập 5 đơn để yêu cầu hủy sau thời điểm xuất hiện từ 1 đến 15 đơn vị thời gian. Đây là số yêu cầu, không phải số đơn chắc chắn hủy thành công.

Thời điểm xuất hiện là lúc hệ thống biết đơn; thời điểm mở cửa sổ là lúc được phép bắt đầu phục vụ. Luật tạo đơn này chưa bảo đảm mọi đơn đều khả thi ngay khi xuất hiện.

**Chuyển ý:** Khi dựng tuyến, mỗi khách phải vượt qua bước kiểm tra ràng buộc.

## Slide 4. Ràng buộc phục vụ một khách hàng

Bộ dựng tuyến kiểm tra ba điều kiện: đủ tải còn lại, bắt đầu phục vụ không muộn hơn hạn của khách, và kịp quay về depot sau khi phục vụ.

Nếu đến sớm, xe chờ đến lúc cửa sổ phục vụ mở. Hạn của khách là hạn bắt đầu dịch vụ, không phải hạn hoàn thành.

Khoảng cách dùng Euclid không làm tròn; thời gian di chuyển bằng khoảng cách. Sau mỗi khách được chọn, bộ dựng tuyến cập nhật vị trí, thời gian và tải còn lại để xét khách tiếp theo.

**Chuyển ý:** Việc gán chuyến và hủy đơn tuân theo chính sách sau.

## Slide 5. Chính sách gán chuyến và hủy đơn

Đơn hàng đi qua các trạng thái chính: chưa xuất hiện, đang chờ, đã gán và hoàn tất. Đơn đang chờ có thể được hủy.

Bộ điều phối chỉ gán chuyến cho xe rảnh tại depot. Tuyến đã gán được giữ đến khi xe quay về; phiên bản hiện tại chưa chèn đơn mới hoặc đổi tuyến giữa chuyến.

Yêu cầu hủy được chấp nhận với đơn đang chờ chưa gán. Nếu đơn đã gán, yêu cầu bị từ chối, kể cả khi xe chưa rời depot.

Đội xe có số lượng cố định. Xe quay về depot có thể nhận hàng cho chuyến tiếp theo, nên một xe có thể thực hiện nhiều chuyến.

**Chuyển ý:** Chính sách này dẫn đến ba sự kiện kích hoạt điều phối.

## Slide 6. Sự kiện kích hoạt bộ điều phối

Chỉ cần một trong ba sự kiện xảy ra là hệ thống yêu cầu điều phối: có đơn mới, hủy đơn thành công, hoặc xe về depot khi còn đơn chưa gán.

Đây là quy tắc cố định theo sự kiện. GPRT quyết định thứ tự chọn khách, không quyết định thời điểm gọi điều phối.

Nhiều yêu cầu cùng thời điểm có thể được gộp thành một lần gọi. Các sự kiện xuất hiện đơn và hoạt động xe được xử lý trước yêu cầu hủy, rồi mới đến điều phối.

Được gọi không có nghĩa là luôn có tuyến mới. Nếu thiếu xe rảnh hoặc không có đơn khả thi, đơn tiếp tục chờ. Giao xong tại một khách không tự kích hoạt điều phối; xe tiếp tục tuyến đã nhận.

**Chuyển ý:** Trong mỗi lần gọi, tuyến được dựng như sau.

## Slide 7. Dựng tuyến bằng heuristic GPRT

Bộ điều phối lấy các đơn đã xuất hiện, còn mở và chưa được gán. Các xe rảnh tại depot được xét lần lượt theo ID.

Với mỗi xe, bộ dựng tuyến lọc khách theo ba ràng buộc, tính điểm bằng biểu thức GPRT và chọn khách có điểm nhỏ nhất. Nếu bằng điểm, chọn theo ID khách.

Sau đó, hệ thống nối khách vào tuyến, cập nhật trạng thái dự kiến rồi lặp lại. Khi không thêm được khách, tuyến kết thúc bằng điểm quay về depot. Nếu không chọn được khách nào, xe chưa nhận chuyến mới.

Biểu thức dùng các đặc trưng đã chuẩn hóa như khoảng cách, tải, thời gian đến, thời gian chờ và độ dư trước hạn. Mọi đặc trưng chỉ sử dụng thông tin đã xuất hiện.

GP dùng biểu thức do GP tìm được; EDD chọn khách có hạn phục vụ sớm nhất. Các phương pháp dùng chung bộ lọc và cách dựng tuyến.

**Chuyển ý:** Ví dụ sau minh họa một lần gọi điều phối chưa tạo được chuyến mới.

## Slide 8. Ví dụ điều phối: một xe, hai đơn

Giả sử chỉ có một xe. Tại thời điểm 0, xe nhận chuyến phục vụ A và dự kiến về depot lúc 40.

Lúc 10, đơn B xuất hiện. Bộ điều phối được gọi nhưng B phải chờ vì xe đang bận. Lúc 15, có yêu cầu hủy A; yêu cầu bị từ chối vì A đã được gán.

Xe hoàn tất phục vụ A lúc 25 rồi quay về depot. Lúc 40, nếu B vẫn đang chờ và còn khả thi, hệ thống gán chuyến mới để phục vụ B.

Ví dụ minh họa chính sách: đơn mới kích hoạt điều phối, nhưng chỉ được gán khi có xe và tuyến phù hợp.

**Chuyển ý:** Cơ chế này được giữ chung khi so sánh các phương pháp.

## Slide 9. Thiết kế so sánh GP, GPRT và EDD

Bộ dữ liệu gồm 56 instance Solomon, chia cố định thành 32 huấn luyện, 12 validation và 12 test.

GP và GPRT có năm seed huấn luyện mỗi phương pháp. Mỗi biểu thức được chọn bằng validation tĩnh rồi giữ nguyên khi mô phỏng. EDD là đối chứng không cần huấn luyện.

Với 12 instance test và ba seed kịch bản, có 36 kịch bản động. Mỗi kịch bản chạy với năm biểu thức GPRT, năm biểu thức GP và một EDD, tương ứng 396 lần mô phỏng.

Đối chứng tĩnh có 132 lần mô phỏng, với mọi đơn xuất hiện tại thời điểm 0 và không có hủy.

Các phương pháp dùng cùng kịch bản, đội xe, ràng buộc và bộ dựng tuyến. Việc đánh giá xét tính hợp lệ trước, tiếp đến mức phục vụ và chi phí. Đơn hủy thành công được thống kê riêng.

## Slide 10–13. Kết quả thực nghiệm

*Ngoài phạm vi bản chỉnh này.*

## Slide 14. Tổng kết framework và hướng mở rộng

Framework tích hợp biểu thức GPRT đã học vào bộ điều phối; DVRPSim thực thi kế hoạch và cập nhật trạng thái theo sự kiện.

Phạm vi hiện tại là kịch bản tổng hợp từ Solomon, với đội xe cố định và nhiều chuyến mỗi xe. Biểu thức được học trên VRPTW tĩnh, sau đó áp dụng nguyên trạng vào mô phỏng động.

Hai yếu tố cần xét khi đánh giá là luật xuất hiện đơn và chính sách giữ tuyến đến khi xe về depot. Hướng mở rộng là thay đổi khoảng báo trước và bổ sung cơ chế chèn đơn giữa chuyến.

Em xin cảm ơn thầy cô đã lắng nghe.

## Câu trả lời ngắn để chuẩn bị trao đổi

**DVRPSim hay GPRT đưa ra quyết định?**

DVRPSim cung cấp trạng thái và thực thi kế hoạch. Bộ điều phối dựng tuyến, dùng biểu thức GPRT để chọn khách trong tập khả thi.

**Cứ có sự kiện là tạo một tuyến mới?**

Không. Sự kiện kích hoạt việc xem xét điều phối. Chỉ tạo tuyến khi có xe rảnh tại depot và có khách khả thi; nhiều yêu cầu cùng thời điểm có thể được gộp.

**GPRT có học lại khi đơn mới xuất hiện không?**

Không. Hệ thống tính lại đặc trưng theo trạng thái hiện tại rồi áp dụng biểu thức đã học. Transformer, GP và RL nằm ở giai đoạn huấn luyện trước mô phỏng.

**Vì sao đơn mới không được chèn ngay vào xe đang chạy?**

Chính sách hiện tại giữ tuyến đã gán đến khi xe về depot. Chèn giữa chuyến là hướng mở rộng, cần kiểm tra lại tải và thời gian phục vụ của các khách phía sau.

**70% động và 5% hủy được hiểu thế nào?**

70% là tỷ lệ đơn được chọn để áp dụng luật xuất hiện động; một số vẫn xuất hiện lúc 0. 5% là tỷ lệ đơn có yêu cầu hủy, không phải tỷ lệ hủy thành công.

**Vì sao một đơn có thể không khả thi ngay khi xuất hiện?**

Thời điểm xuất hiện được tính từ thời điểm mở cửa sổ và khoảng báo trước. Công thức chưa bảo đảm xe từ depot đến kịp hạn phục vụ.

**Không vi phạm ràng buộc có nghĩa là phục vụ đủ đơn không?**

Không. Tính hợp lệ của các chuyến và mức phục vụ là hai tiêu chí riêng. Đơn chưa phục vụ vẫn được ghi nhận.
