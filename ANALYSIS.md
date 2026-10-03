# Phân tích kết quả Day 17

Chạy `python src/benchmark.py` từ thư mục gốc. Kết quả dưới đây dùng chế độ offline, không cần API key. `Response quality` là điểm heuristic dựa trên độ phủ các fact kỳ vọng và độ dài câu trả lời; đây không phải đánh giá của người đọc hay LLM judge.

| Bộ dữ liệu | Agent | Agent tokens only | Prompt tokens processed | Cross-session recall | Response quality | Memory growth (bytes) | Compactions |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Standard | Baseline | 1.530 | 14.645 | 0% | 20% | 0 | 0 |
| Standard | Advanced | 2.061 | 22.863 | 100% | 99% | 281 | 0 |
| Long-context stress | Baseline | 450 | 23.388 | 0% | 20% | 0 | 0 |
| Long-context stress | Advanced | 609 | 11.542 | 100% | 100% | 258 | 6 |

Baseline giữ lịch sử trong từng thread nhưng các câu hỏi recall được hỏi ở thread mới, nên điểm recall bằng 0. Advanced lưu tên, nghề, nơi ở và preference trong `User.md`, vì vậy vẫn trả lời được sau khi đổi thread. Dữ liệu có đính chính từ Huế sang Đà Nẵng và câu đùa về nghề product manager; quy tắc trích xuất ưu tiên thông tin mới, rõ ràng từ người dùng.

Ở bộ Standard, Advanced xử lý nhiều prompt tokens hơn Baseline (22.863 so với 14.645). Profile và summary làm tăng ngữ cảnh, trong khi hội thoại còn ngắn nên compaction chưa đem lại lợi ích. Ở stress test, sáu lần compact giảm lượng prompt tokens cần xử lý khoảng 50,7% (11.542 so với 23.388). `Agent tokens only` vẫn cao hơn, vì đây là số token trong câu trả lời, không phải chi phí mang lịch sử vào prompt.

`Memory growth` là mức tăng kích thước file profile trong từng lần chạy cô lập. Upsert thay fact cũ khi có đính chính, còn các mối quan tâm mới có thể làm file lớn dần. Trích xuất theo quy tắc vẫn có nguy cơ lưu sai một phát biểu mơ hồ; summary giới hạn kích thước cũng có thể bỏ sót chi tiết cũ. Với hệ thống thật nên bổ sung nguồn gốc fact, confidence và khả năng xem hoặc xóa ký ức đã lưu.

## Bonus: conflict handling và entity extraction

Bonus xử lý lỗi mâu thuẫn khi người dùng đính chính một fact ổn định. `extract_profile_updates()` lấy lần nhắc rõ ràng mới nhất trong message, còn `upsert_fact()` ghi theo key cố định (`name`, `location`, `profession`, ...), nên giá trị mới thay giá trị cũ thay vì giữ đồng thời cả hai. Test `test_correction_replaces_old_persisted_fact` xác nhận `Huế` bị thay hoàn toàn bằng `Đà Nẵng` trong `User.md`.

Điều này cải thiện recall đúng theo thời điểm hiện tại: các câu hỏi ở thread mới không thể trả lời đồng thời nơi ở cũ và mới. Bonus không nhằm giảm token trực tiếp; profile có cấu trúc giữ phần persistent memory nhỏ, dễ đọc và tránh phải suy luận lại các fact cũ từ toàn bộ lịch sử.

Rủi ro là một câu nói mơ hồ hoặc bị trích sai vẫn có thể ghi đè fact đúng. Các guardrail hiện có chỉ nhận mẫu tự mô tả rõ ràng, bỏ qua câu hỏi và vài trường hợp nhiễu. Hệ thống production nên thêm confidence score, provenance, lịch sử thay đổi và thao tác để người dùng sửa hoặc xóa profile.
