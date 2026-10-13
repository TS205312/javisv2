# Changelog

## 2026-10-11
- Đổi waterfall sang Groq Qwen → Kilo → OpenRouter Ultra → Groq fallback → OpenRouter Lightning → Gemini → Ollama.
- Đặt model Ollama 1.5B làm offline backup, timeout 180 giây, cảnh báo phản hồi trên 30 giây; thêm cấu hình concurrency/load giới hạn.
- Mặc định swarm 4 agent khởi tạo bằng Groq, Kilo, OpenRouter và Groq fallback; mỗi agent ưu tiên provider riêng rồi failover qua provider khác.
- Bổ sung test thứ tự tier, fallback provider riêng cho swarm và chỉ kiểm tra hai model OpenRouter được ưu tiên.

## 2026-10-11
- Đặt Ollama ở Tier 7 sau OpenRouter fallback và Gemini; tăng timeout Ollama lên 180 giây, kiểm tra model local trước khi gọi và yêu cầu nội dung thật trong live test.
- Thêm regression test router đi tới Ollama sau khi các tier cloud lỗi.

## 2026-10-11
- Khôi phục `main.py` thành CLI agent, dựa trên API hiện có trong orchestrator sau khi xác nhận không có Git/backup/CLI trong log.
- Thêm bảo vệ đường dẫn và backup timestamp (giữ tối đa 5) cho các thao tác ghi qua file/coder tool; giới hạn skill clone vào `.skills/gh_*` và backup registry metadata.
- Truy vấn catalog OpenRouter live, thay pool bằng 10 ID `:free` đang có trong catalog và bỏ qua OpenRouter nếu catalog rỗng/không khả dụng.

## 2026-10-11
- Thay các model OpenRouter cố định bằng pool cấu hình; kiểm tra catalog API với cache một giờ và block từng model sau ba lỗi trong năm phút.
- Thêm test live cho từng model trong pool, test fallback model kế tiếp; cập nhật README về quản lý pool, trạng thái cooldown và quota.

## 2026-10-11
- Xác minh Python script do coder tool tạo bằng cách chạy tối đa ba lần; sửa qua LLM khi stderr báo lỗi và chỉ trả PASS sau exit code 0.
- Bổ sung arXiv cho truy vấn bài báo/nghiên cứu AI và Bing fallback cho tìm kiếm chung khi DuckDuckGo lỗi hoặc trả rỗng; lỗi tìm kiếm được log và trả danh sách rỗng thay vì làm crash tool.
- Làm aggregator chịu lỗi khi nhận kết quả `partial`, `None`, agent crash hoặc mọi agent fail; swarm truyền tổng số agent để log tỷ lệ kết quả hợp lệ.
- Đặt Groq chính là `qwen/qwen3.8-27b`; đổi fallback sang `openai/gpt-oss-20b` sau khi model `llama-3.1-8b-instant` trả HTTP 404. Đổi OpenRouter fallback sang `nvidia/nemotron-3.5-lightning:free` vì `google/gemini-2.0-flash-exp:free` trả HTTP 404; tài khoản hiện gặp 429 với các model OpenRouter free được kiểm tra.
- Chuẩn hóa `shell_tool` để nhận cả khóa `cmd` và `command` như router đang sinh, tránh retry giả do thiếu tham số lệnh.
- Hỗ trợ các bí danh `file_path`/`filepath` và `content` của router trong coder tool; bổ sung `matplotlib` làm dependency để script biểu đồ chạy trong project venv.

## 2026-10-11
- Sửa gate theo tên provider để Gemini vẫn chạy ở Tier 6 và Ollama optional chỉ chạy ở Tier 7 khi model đã cài; xác nhận model variables đọc đúng từ `.env`.
- Cập nhật README nêu rõ Tier 7 Ollama optional và yêu cầu cài đúng model trước khi dùng.
- Sắp xếp router thành bảy tier: Kilo, OpenRouter primary, Groq primary, Groq fallback, OpenRouter fallback, Gemini, Ollama; giữ tier thành công trong state nhưng không để cache vượt thứ tự ưu tiên mới.
- Chặn OpenRouter trong 60 giây sau khi tier primary/fallback gặp rate limit, nhưng vẫn thử tier fallback trong cùng một lượt.
- Gia cố GitHub hunt với kiểm tra shape kết quả, URL GitHub, metadata repo và graceful skip cho response rỗng/API 403/429; thêm regression tests.
- Cập nhật model ID Kilo/OpenRouter/Gemini/Groq theo cấu hình đã xác minh; thêm model Groq dự phòng.
- Thêm retry theo model cho OpenRouter/Groq, xác thực nội dung phản hồi, cooldown chống gọi lặp OpenRouter và trạng thái router lưu trong `logs/router_state.json`.
- Thêm kiểm tra live provider và kiểm tra fallback/state độc lập.
- Sửa thứ tự fallback LLM cố định sáu tier (Kilo, OpenRouter, Gemini, Groq, OpenRouter fallback, Ollama) với log OK/SKIP/FAIL và xác thực Ollama model trước khi gọi.
- Chạy swarm trong executor thread pool qua `asyncio.gather`/`wait_for`, tách memory/log theo agent và lọc lỗi từng agent trước khi tổng hợp.

## 2026-10-11
- Cập nhật model Groq mặc định sang `qwen/qwen3.8-27b` sau khi xác nhận `llama-3.3-70b-versatile` không còn khả dụng qua API; key được lưu cục bộ trong `.env`.
- Giới hạn chat completion token để tương thích quota token đầu ra của Groq/OpenRouter.
- Thêm Skill Resolver ba tầng: tìm skill local/procedural memory, chấm điểm chất lượng, tìm repo GitHub theo stars và lọc license/cập nhật/fork/topic.
- Thêm CLI `main_skill.py`, cài/import-test repo được chọn, lưu metadata registry, cập nhật outcome/rating; tích hợp resolver vào executor/router và pipeline đơn/swarm.
- Thêm cấu hình token GitHub optional cùng dependencies PyGithub, GitPython, PyYAML.

## 2026-10-11
- Chuyển mặc định LLM sang fallback Ollama → Groq → OpenRouter → Gemini tùy chọn; đặt timeout 60 giây và ghi provider vào session log.
- Thêm cấu hình model/API key cho Ollama, Groq, OpenRouter; giữ Gemini là tùy chọn.
- Cập nhật `.env`, thêm `.env.example`, bổ sung dependency `groq` và hướng dẫn provider trong README.
- Đặt stdout/stderr UTF-8 trên Windows, thêm kiểm tra provider trước khi khởi động agent và mã thoát lỗi khi lập kế hoạch thất bại.

## 2026-10-10
- Di chuyển hàm `run_single_agent` và `run_swarm_mode` từ `main.py` sang `core/orchestrator.py` để tương thích import.
- Bổ sung tham số `cwd` vào `run_single_agent`, `run_swarm_mode`, `main.py`.
- Thêm validate trong `main.py` để từ chối chạy khi `--agents > 8` hoặc `--agents 1 --strategy debate`.
- Cập nhật file `.env` với các API keys trống (như hướng dẫn).
