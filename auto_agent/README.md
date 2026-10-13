# AUTONOMOUS AGENT — Swarm Ensemble

Agent tự hành bằng Python 3.10+ với khả năng chạy **N agent song song**, mỗi agent tự
lập kế hoạch, tự thực thi, tự đánh giá, tự phản chiếu — hoàn toàn độc lập. Kết quả
được tổng hợp bởi Aggregator qua LLM.

## Kiến trúc

```
┌─────────────── main.py ───────────────┐
│  --agents 1 → đơn agent (legacy)      │
│  --agents 4 → swarm ensemble          │
└───────────────────────────────────────┘
         │ (swarm mode)
         ▼
┌─── orchestrator.py ───────────────────┐
│  asyncio.gather(A1, A2, A3, A4)       │
│  timeout 300s/agent                   │
│  crash-safe: agent lỗi → bỏ qua      │
└───────────────────────────────────────┘
         │
    ┌────┼────┬────┐
    ▼    ▼    ▼    ▼
  ┌──┐ ┌──┐ ┌──┐ ┌──┐
  │A1│ │A2│ │A3│ │A4│   ← SwarmAgent (swarm.py)
  └──┘ └──┘ └──┘ └──┘
   │    │    │    │      Mỗi agent có:
   │    │    │    │       • LLM fallback: Kilo → OpenRouter → Groq → Groq fallback → OpenRouter fallback → Gemini → Ollama optional
   │    │    │    │       • Memory riêng (ChromaDB namespace)
   │    │    │    │       • Log riêng (logs/session_*_agentN.json)
   │    │    │    │       • Pipeline: planner → router → executor → critic → reflection
    └────┼────┴────┘
         │
         ▼
┌─── aggregator.py ─────────────────────┐
│  Đọc 4 kết quả → LLM tổng hợp        │
│  Trả: winner, confidence, reason      │
│  Strategy: ensemble | vote | debate   │
│  Log riêng: session_*_aggregator.json │
└───────────────────────────────────────┘
         │
         ▼
    FINAL RESULT
```

## Cài đặt

```bash
cd C:\javis\auto_agent
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
playwright install chromium
```

GitHub skill resolver (token optional; raises GitHub API rate limits):

```bash
python main_skill.py check "code web"    # local skills only; never hunts
python main_skill.py resolve "code web"  # local assessment, then GitHub hunt/install
python main_skill.py list
python main_skill.py upgrade fastapi
python main_skill.py rate github:tiangolo/fastapi 5
```

Discovered repositories are filtered by license, update date, fork status, and
blocked topics before selection. Approved repositories are cloned under
`.skills/`, installed into the active Python environment, import-tested, and
recorded in `skill_registry.json`. Add an optional `GITHUB_TOKEN` to `.env` to
increase GitHub API rate limits.

**Security note:** installing a repository may execute its build/install code.
The license and topic checks are not a security audit; run this autonomous
installer in a disposable virtual environment or sandbox.

### Cài Ollama (tùy chọn nhưng khuyến nghị)

Cài Ollama theo hệ điều hành:

- **Windows:** tải installer từ [ollama.com/download/windows](https://ollama.com/download/windows), cài đặt và mở Ollama.
- **macOS:** tải ứng dụng từ [ollama.com/download/mac](https://ollama.com/download/mac) hoặc cài bằng `brew install ollama`.
- **Linux:** chạy `curl -fsSL https://ollama.com/install.sh | sh`, sau đó khởi động dịch vụ bằng `ollama serve` nếu dịch vụ chưa tự chạy.

Tải model mặc định:

```bash
ollama pull qwen2.5-coder:1.5b
```

Ollama chạy local, không cần API key. Model 1.5B mặc định tiêu thụ ít tài nguyên hơn model 7B; mức RAM/dung lượng thực tế phụ thuộc lượng context và runtime. Có thể đổi sang model Ollama khác bằng `OLLAMA_MODEL` trong `.env`.

### Cấu hình `.env`

```env
OLLAMA_HOST=http://localhost:11434
OLLAMA_NUM_PARALLEL=1
OLLAMA_MAX_LOADED_MODELS=1
KILO_API_KEY=
KILO_MODEL=kilo-auto/free
OLLAMA_MODEL=qwen2.5-coder:1.5b
GROQ_API_KEY=
GROQ_MODEL=qwen/qwen3.8-27b
GROQ_FALLBACK_MODEL=openai/gpt-oss-20b
OPENROUTER_API_KEY=
OPENROUTER_PRIMARY_MODEL=nvidia/nemotron-3-ultra-550b-a55b:free
OPENROUTER_FALLBACK_MODEL=nvidia/nemotron-3.5-lightning:free
OPENROUTER_FREE_POOL=nvidia/nemotron-3-ultra-550b-a55b:free,nvidia/nemotron-3.5-lightning:free,poolside/laguna-s-2.1:free,cohere/north-mini-code:free,nvidia/nemotron-3-nano-omni-30b-a3b-reasoning:free,google/gemma-4-26b-a4b-it:free,google/gemma-4-31b-it:free,nvidia/nemotron-3-super-120b-a12b:free,liquid/lfm-2.5-2.6b:free,thinkingmachines/inkling:free
GEMINI_API_KEY=
GEMINI_MODEL=gemini-3.8-flash
BLENDER_PATH=blender
FFMPEG_PATH=ffmpeg
AGENT_TIMEOUT=300
```

Sao chép `.env.example` thành `.env` nếu cần. Thứ tự ưu tiên được sắp theo latency đã đo; Gemini chỉ được thử sau các tier cloud nhanh hơn. Router lưu tier/model thành công gần nhất, bộ đếm hằng ngày và cooldown vào `logs/router_state.json`. Model Ollama chỉ được gọi khi `ollama list` xác nhận model đã có.

### OpenRouter Free Pool (model free 1M-token theo danh sách yêu cầu)

Catalog OpenRouter được truy vấn trực tiếp ngày 2026-10-11; 10 model sau
có trong catalog và được đặt vào pool theo thứ tự:

1. `nvidia/nemotron-3.5-lightning:free`
2. `poolside/laguna-s-2.1:free`
3. `cohere/north-mini-code:free`
4. `nvidia/nemotron-3-ultra-550b-a55b:free`
5. `nvidia/nemotron-3-nano-omni-30b-a3b-reasoning:free`
6. `google/gemma-4-26b-a4b-it:free`
7. `google/gemma-4-31b-it:free`
8. `nvidia/nemotron-3-super-120b-a12b:free`
9. `liquid/lfm-2.5-2.6b:free`
10. `thinkingmachines/inkling:free`

Router xác minh ID trong catalog trước khi gọi. Nếu catalog không lấy được hoặc
rỗng, OpenRouter bị bỏ qua và router tiếp tục provider kế tiếp. Model vắng mặt,
lỗi HTTP/timeout, 429, hoặc phản hồi rỗng/thinking sẽ được bỏ qua để thử model
kế tiếp. Danh sách model được cache một giờ trong `logs/openrouter_models.json`.
Thêm/xóa model bằng cách sửa
`OPENROUTER_FREE_POOL` trong `.env` và phân cách ID bằng dấu phẩy; pool rỗng sẽ
dùng model legacy `OPENROUTER_MODEL`. `OPENROUTER_MODEL` trong Python vẫn là
alias của model đầu pool để tương thích caller cũ.

Nếu cùng một model lỗi ít nhất ba lần trong năm phút, chỉ model đó bị block
30 phút. Block và lịch sử lỗi lưu trong `logs/router_state.json`. Sự hiện diện
trong catalog không đảm bảo quota hoặc khả dụng tại thời điểm gọi. Hạn mức Free
Tier theo thông tin được cung cấp cho tài khoản này là 50 request/ngày/model
và 20 request/phút; nạp $10 có thể nâng mức lên 1.000 request/ngày. Quota và
điều kiện có thể đổi theo model/tài khoản; xác minh trên dashboard OpenRouter.

| Tier | Provider / model | Latency tham khảo | Vai trò |
|------|------------------|------------------|---------|
| 1 | Groq `qwen/qwen3.8-27b` | ~0.6 s | Cloud primary |
| 2 | Kilo `kilo-auto/free` | ~1.5 s | Cloud fallback |
| 3 | OpenRouter `nvidia/nemotron-3-ultra-550b-a55b:free` | ~1.2 s* | Cloud fallback |
| 4 | Groq `openai/gpt-oss-20b` | ~1.7 s | Groq fallback |
| 5 | OpenRouter `nvidia/nemotron-3.5-lightning:free` | ~3 s* | OpenRouter fallback |
| 6 | Gemini `gemini-3.8-flash` | ~39 s | Cloud cuối |
| 7 | Ollama `qwen2.5-coder:1.5b` | ~2–5 s* | Offline backup |

*Giá trị latency là số tham khảo do cấu hình hiện tại cung cấp, không phải cam kết và thay đổi theo mạng/quota/tải máy. Trên Ryzen 5 4600G, RAM 16 GB, không có GPU rời, ưu tiên cloud để tránh tốn CPU/RAM; Ollama chỉ dùng model 1.5B và chỉ gọi sau khi cả 6 tier cloud fail. Request Ollama timeout 180 giây; nếu phản hồi thành công mất trên 30 giây, router ghi warning. `OLLAMA_NUM_PARALLEL=1` và `OLLAMA_MAX_LOADED_MODELS=1` giữ cấu hình phục vụ nhẹ.

- [Groq API keys](https://console.groq.com/keys)
- [OpenRouter API keys](https://openrouter.ai/keys)

| Provider | Tốc độ | Chất lượng | Giới hạn / chi phí |
|----------|--------|------------|---------------------|
| Ollama local | Phụ thuộc phần cứng; GPU thường nhanh hơn | Phụ thuộc model local (mặc định Qwen 1.5B) | Miễn phí, riêng tư; cần tải model |
| Groq API | Rất nhanh nhờ inference chuyên dụng | Cao, tùy model đã cấu hình | Free tier có quota/rate limit; chính sách có thể thay đổi |
| OpenRouter `:free` | Thay đổi theo model và tải dịch vụ | Thay đổi theo model | Model miễn phí có rate limit/availability riêng; có thể hết quota |
| Gemini API | Nhanh | Cao, tùy model | Chỉ fallback khi có key; quota/giá phụ thuộc tài khoản và model |

### Kiểm tra provider và router

```bash
python tests/test_providers.py       # Gửi "hello" thật tới các model đã cấu hình
python tests/test_router_fallback.py # Kiểm tra fallback, cooldown và state bằng mock
```

Live test chỉ gửi prompt ngắn `hello`, in HTTP status/phân loại/thời gian, không in API key. Ollama được báo `MODEL_UNAVAILABLE` nếu model cấu hình chưa cài.

### Phần mềm ngoài (tuỳ chọn)

| Phần mềm   | Mục đích              | Cài đặt                           |
|-------------|----------------------|------------------------------------|
| Blender     | Render 3D headless   | [blender.org](https://blender.org) |
| ffmpeg      | Xử lý media          | `choco install ffmpeg`             |
| Tesseract   | OCR                  | `choco install tesseract`          |
| Matplotlib  | Tạo biểu đồ bằng Python | `pip install matplotlib`          |
| Ollama      | LLM local (ưu tiên)  | [ollama.com](https://ollama.com)   |
| Playwright  | Browser automation   | `playwright install chromium`      |

## Sử dụng

### Đơn agent (legacy)
```bash
python main.py --goal "viết tool crawl giá vàng lưu CSV"
```

### Swarm ensemble (4 agent song song)
```bash
python main.py --goal "viết tool crawl giá vàng lưu CSV" --agents 4
python main.py --goal "tìm 5 bài AI 2024 lưu output/ai.txt" --agents 4 --strategy ensemble
python main.py --goal "Blender cầu xoay render 2s" --agents 4 --max-steps 10
```

### Chọn model
```bash
python main.py --goal "tìm 5 bài AI 2024" --agents 4 --models "auto,auto,auto,auto"
```

Mặc định mọi request dùng chuỗi fallback tự động. `--models` nhận một giá trị cho mỗi agent; dùng dạng `kilo:model`, `openrouter:model`, `gemini:model`, `groq:model` hoặc `ollama:model` để đặt model riêng cho provider đó. Các override không thay đổi thứ tự bảy tier.

### Tìm kiếm web và bài báo

`search_tool` ưu tiên arXiv cho truy vấn học thuật về bài báo/nghiên cứu AI,
sau đó dùng DuckDuckGo và Bing làm nguồn tìm kiếm chung. Kết quả arXiv gồm
tiêu đề, ngày công bố, URL và tóm tắt để agent có thể ghi nguồn rõ ràng.

### Tham số CLI

| Tham số       | Mặc định   | Mô tả                                                  |
|---------------|------------|---------------------------------------------------------|
| `--goal`      | _(bắt buộc)_ | Mục tiêu tiếng Việt (mơ hồ cũng được)               |
| `--agents`    | `1`        | Số agent (1=đơn, 2-8=swarm)                             |
| `--strategy`  | `ensemble` | Chiến lược tổng hợp: `ensemble`, `vote`, `debate`       |
| `--models`   | `auto`     | Model/provider override cho từng agent, phân cách bằng dấu phẩy |
| `--max-steps` | `50`       | Số bước tối đa mỗi agent                                |
| `--yolo`      | `false`    | Bỏ qua safety check (nguy hiểm)                         |

## Output mẫu

```
[SWARM] 4 agent khởi động song song
[A0] [PLANNING] Đang lập kế hoạch...
[A1] [PLANNING] Đang lập kế hoạch...
[A2] [PLANNING] Đang lập kế hoạch...
[A3] [PLANNING] Đang lập kế hoạch...
[A0] [PLAN] Plan xong → 3 task
[A1] [PLAN] Plan xong → 4 task
[A2] [CRITIC] Fail (1/3): ... Gợi ý sửa: ...
[A2] [CRITIC] Pass: Retry thành công
[A3] [DONE] Xong trong 12.3s (3/3 task)
[SWARM] Tất cả agent hoàn tất trong 18.7s
[AGGREGATOR] Chọn: A1 + merge A2 → confidence 0.87
[DONE] Tổng thời gian: 19.1s
⚠️  Chi phí token: ~4x so với chạy đơn agent
```

## Log files

Mỗi lần chạy swarm tạo ra:
- `logs/session_<ts>_agent0.json` — log agent 0
- `logs/session_<ts>_agent1.json` — log agent 1
- `logs/session_<ts>_agent2.json` — log agent 2
- `logs/session_<ts>_agent3.json` — log agent 3
- `logs/session_<ts>_aggregator.json` — log aggregator
- `logs/session_<ts>_orchestrator.json` — log orchestrator

## ⚠️ Chi phí và hạn mức

Chạy swarm với N agent thực hiện nhiều request hơn chế độ đơn. Ollama không tính phí API. Groq free tier và model `:free` của OpenRouter có quota, giới hạn tốc độ và khả năng sẵn sàng có thể thay đổi; kiểm tra chính sách hiện hành của nhà cung cấp. Gemini không phải backend mặc định, chỉ được thử cuối cùng khi đã cấu hình `GEMINI_API_KEY`.

## Cấu trúc thư mục

```
auto_agent/
├── main.py                 # Entry point CLI
├── registry.py             # Tool registry cho LLM đọc
├── requirements.txt
├── .env                    # API keys (tự tạo)
├── core/
│   ├── config.py           # Cấu hình từ .env
│   ├── llm_helper.py       # LLMClient (fixed seven-tier provider fallback)
│   ├── planner.py          # LLM sinh plan JSON
│   ├── router.py           # LLM chọn tool
│   ├── executor.py         # Chạy tool, lưu memory
│   ├── critic.py           # Đánh giá pass/fail
│   ├── reflection.py       # Tự phản chiếu mỗi 5 bước
│   ├── memory.py           # Bộ nhớ ngắn hạn + ChromaDB
│   ├── safety.py           # Blacklist lệnh nguy hiểm
│   ├── logger.py           # Log JSON per-session/per-agent
│   ├── swarm.py            # SwarmAgent — wrap pipeline thành 1 agent
│   ├── aggregator.py       # Tổng hợp kết quả N agent
│   └── orchestrator.py     # Chạy N agent song song
├── tools/
│   ├── shell_tool.py       # Chạy shell command
│   ├── file_tool.py        # Đọc/ghi file
│   ├── coder_tool.py       # Sinh/sửa code
│   ├── browser_tool.py     # Playwright scrape
│   ├── search_tool.py      # arXiv, DuckDuckGo and Bing search
│   ├── blender_tool.py     # Blender headless
│   ├── media_tool.py       # ffmpeg
│   ├── computer_tool.py    # pyautogui + mss
│   └── vision_tool.py      # OCR + ảnh
├── logs/                   # Session logs (auto-created)
└── chroma_db/              # ChromaDB storage (auto-created)
```

## Chạy trên VM / Sandbox

Khuyến nghị chạy agent trong môi trường sandbox vì agent có khả năng chạy lệnh shell
và điều khiển GUI. Không nên chạy trên máy production.
