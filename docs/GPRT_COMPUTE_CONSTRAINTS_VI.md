# Giới hạn tài nguyên cho thiết kế GPRT

## Yêu cầu người dùng

Người dùng nhấn mạnh trong trao đổi: chạy trên laptop hoặc Colab miễn phí, cấu hình không được quá nặng. Áp dụng cho cả huấn luyện, screening, diagnostic và ma trận thí nghiệm; không chỉ kích thước Transformer.

## Nguyên tắc chọn cải tiến

- Giữ cấu hình Transformer nhỏ hiện tại làm mốc (2 lớp, d_model 64, 4 heads, FFN 128); chưa tăng kiến trúc trước khi có đo đạc.
- Ưu tiên thay thế trong ngân sách hiện có: lấy một phần elite đã lưu thay cho một phần cây khởi tạo, không tăng population. Đánh giá lại elite trên batch mới và tính chi phí này.
- Không mặc định mở rộng screening từ 8 lên 32 cây. Trước tiên thử giữ 8 ứng viên screening nhưng chọn ít elite hơn, hoặc thay một phần ứng viên bằng archive. So sánh bằng validation.
- Nếu thử thêm thông tin cấu trúc cây, dùng embedding nhỏ của độ sâu/cha/vị trí đối số; chưa thêm encoder instance, GNN hoặc mô hình sửa cây riêng.
- Giữ bộ nhớ elite có giới hạn. Chỉ thử thêm một số ít minibatch imitation nếu đo thời gian cho phép; không lặp REINFORCE trên mẫu cũ như mẫu on-policy.
- Diagnostic tốn chi phí có thể đo theo chu kỳ thưa hơn. Ghi riêng thời gian evaluator, neural, screening, diagnostic và lưu checkpoint.
- Giữ feature nhìn trước dưới dạng ablation có đo thời gian; không mặc định thêm insertion/local search vào vòng fitness.

## Trình tự thí nghiệm phù hợp

1. Đo thời gian và bộ nhớ một cấu hình nhỏ nhưng phủ đủ 32 instance train. Với batch 4 cần ít nhất 8 chu kỳ trong lịch hiện tại.
2. Screening ít cấu hình trên train/validation, có thể dùng 1–2 seed để loại phương án; đây không phải kết quả chính nhiều seed. Runner nào yêu cầu 5 seed cho protocol chính vẫn giữ yêu cầu đó; cần chế độ screening riêng nếu triển khai.
3. Thay từng yếu tố, tránh quét tích alpha × tỷ lệ seeding × kiến trúc × số chu kỳ.
4. Sau khi khóa một phương án, mới chạy so sánh chính nhiều seed. Chỉ tăng lên 16/32 chu kỳ nếu thời gian đo được phù hợp; giảm số phương án phụ trước khi tăng cấu hình.

Các thông số trên là định hướng tiết kiệm tài nguyên, chưa phải cấu hình tối ưu hoặc cam kết thời gian chạy. Chưa có thay đổi mã hay huấn luyện mới trong lần ghi nhận này.
