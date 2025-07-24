# 🎯 AI Slide Generator

Ứng dụng tạo slide thuyết trình tự động sử dụng AI với kiến trúc multi-agent workflow, hỗ trợ nhiều mô hình AI và xuất file PDF/PPTX.

## 📖 Tổng quan

AI Slide Generator là một ứng dụng web hiện đại cho phép người dùng tạo slide thuyết trình chuyên nghiệp chỉ bằng cách mô tả chủ đề. Ứng dụng sử dụng hệ thống đa tác nhân (multi-agent) thông minh để:

- 🔍 **Tự động nghiên cứu** thông tin và hình ảnh liên quan
- 🎨 **Thiết kế layout** và bảng màu phù hợp với chủ đề  
- 📝 **Tạo nội dung** slide chuyên nghiệp
- ✏️ **Chỉnh sửa trực tiếp** text trên slide
- 📄 **Xuất file** PDF và PowerPoint

## ✨ Tính năng chính

### 🤖 Hệ thống Multi-Agent
- **Supervisor Agent**: Điều phối quy trình làm việc
- **Planner Agent**: Lập kế hoạch chi tiết
- **Outline Agent** (Research Agent): Nghiên cứu và tạo đề cương
- **Artist Agent**: Thiết kế bố cục và màu sắc
- **Slide Agent**: Tạo slide HTML cuối cùng

### 🧠 Tích hợp nhiều mô hình AI
- **Claude 3.7 Sonnet** (AWS Bedrock)
- **GPT-4o & GPT-o3** (Azure OpenAI)  
- **Gemini 2.5 Pro & Flash** (Google)
- Cơ chế fallback tự động giữa các mô hình

### 💬 Giao diện thời gian thực
- Chat interface với Socket.IO
- Streaming responses từ AI
- Hiển thị quá trình tạo slide theo thời gian thực
- Approval workflow cho outline

### 🎨 Thiết kế chuyên nghiệp
- Responsive HTML slides với Tailwind CSS
- Bảng màu thông minh dựa trên chủ đề
- Layout tự động cho từng loại nội dung
- Tương thích Material Design

### ✏️ Chỉnh sửa tương tác
- Rich text editor với toolbar đầy đủ
- Chỉnh sửa text trực tiếp trên slide
- AI Assistant cho chỉnh sửa thông minh
- Preview và edit mode

### 📤 Xuất file đa định dạng
- **PDF**: Playwright với scaling chính xác
- **PPTX**: ConvertAPI integration
- Xuất từng slide riêng lẻ hoặc kết hợp
- Presentation mode toàn màn hình

## 🏗️ Kiến trúc hệ thống

![System Architecture](./images/System_Architecture.png)

### Backend (Flask + Socket.IO)
- **app.py**: Server chính với REST API và WebSocket
- **workflow.py**: LangGraph workflow với các agent
- **models/LLMs.py**: Cấu hình các mô hình AI

### Frontend (Vanilla JavaScript)
- **templates/index.html**: Interface chính
- **static/style.css**: Styling responsive
- Chat panel, slides display, editors

### Utils & Tools
- **utils/tools.py**: Tools cho agents (search, crawl, slide generation)
- **utils/pdf_export.py**: Xuất PDF với Playwright
- **utils/pptx_export.py**: Xuất PPTX với ConvertAPI
- **utils/schemas.py**: Pydantic schemas

## 🚀 Cài đặt

### Yêu cầu hệ thống
- Python 3.11+
- Node.js (cho Playwright)
- Internet connection (cho AI APIs)

### 1. Clone repository
```bash
git clone https://github.com/LongNgHuynh/Slide-Generator.git
cd Slide-Generator
```

### 2. Cài đặt dependencies
```bash
# Sử dụng uv (khuyến nghị)
pip install uv
uv sync

# Hoặc pip
pip install -r requirements.txt
```

### 3. Cài đặt Playwright browsers
```bash
playwright install chromium
```

### 4. Cấu hình environment variables
Tạo file `.env`:

```env
# Azure OpenAI (GPT-4o, GPT-o3)
AZURE_OPENAI_API_KEY=your_azure_key
AZURE_OPENAI_ENDPOINT=your_azure_endpoint

# Google AI (Gemini)
GOOGLE_API_KEY=your_google_key

# AWS Bedrock (Claude)
AWS_ACCESS_KEY_ID=your_aws_key
AWS_SECRET_ACCESS_KEY=your_aws_secret
AWS_REGION_NAME=us-east-1

# ConvertAPI (PPTX export - tùy chọn)
CONVERTAPI_API_KEY=your_convertapi_key

# Langfuse (Tracing - tùy chọn)
LANGFUSE_SECRET_KEY=your_langfuse_secret
LANGFUSE_PUBLIC_KEY=your_langfuse_public
```

### 5. Chạy ứng dụng
```bash
python app.py
```

Truy cập: http://localhost:5000

## 📋 Hướng dẫn sử dụng

### 1. Tạo slide mới
1. Nhập chủ đề vào chat (ví dụ: "Tạo presentation về AI")
2. Hệ thống sẽ tự động:
   - Nghiên cứu thông tin
   - Tạo outline
   - Thiết kế layout
   - Sinh slide HTML

### 2. Chỉnh sửa slide
- **Edit mode**: Click nút "Edit" để bật chế độ chỉnh sửa
- **Text editing**: Click icon ✏️ bên cạnh text để mở rich text editor
- **AI Assistant**: Click "🤖 Ask Assistant" để chỉnh sửa với AI

### 3. Xuất file
- **PDF**: Click "📄 Export" → "Export All Slides (Combined PDF)"
- **PPTX**: Click "📊 Export" → "Export All Slides (Combined PPTX)"
- **Presentation**: Click "📊 Present" để mở chế độ trình chiếu

### 4. Chế độ trình chiếu
- Sử dụng phím mũi tên để điều hướng
- F11 hoặc double-click để fullscreen
- ESC để thoát

## 🔧 Cấu hình nâng cao

### Tùy chỉnh AI Models
Trong `models/LLMs.py`:
```python
# Thay đổi mô hình mặc định
LLM = Gemini()  # Hoặc Claude_3_7_Sonnet(), GPT_4o()

# Cấu hình fallback
LLM_FALLBACKS = [
    ("Claude", LLM_CLAUDE),
    ("Gemini", LLM_GEMINI),
    # ...
]
```

### Tùy chỉnh Design Rules
Chỉnh sửa `rules/html.txt` và `rules/instruction.txt` để thay đổi style và guidelines cho AI.

### Cấu hình Export
```python
# PDF Export settings (utils/pdf_export.py)
export_slides_to_pdf(
    slide_files=files,
    method="playwright",  # hoặc "weasyprint"
    combine=True
)

# PPTX Export (yêu cầu ConvertAPI key)
convert_pdf_to_pptx(pdf_path, output_dir, filename)
```

## 🔍 API Endpoints

### REST API
- `GET /` - Main interface
- `GET /export_pdf/all` - Xuất PDF tất cả slides
- `GET /export_pdf/<slide_number>` - Xuất PDF slide cụ thể
- `GET /convert_all_slides_to_pptx` - Xuất PPTX tất cả slides
- `GET /api/slides/available` - Danh sách slides có sẵn

### Socket.IO Events
- `send_message` - Gửi tin nhắn từ user
- `slide_generated` - Nhận slide mới được tạo
- `text_updated` - Cập nhật text slide
- `ai_edit_slide` - Chỉnh sửa slide với AI

## 🧪 Testing

```bash
# Test PDF export
python -m utils.pdf_export

# Test tools
python -m utils.tools
```

## 📊 Monitoring & Tracing

Ứng dụng tích hợp **Langfuse** để theo dõi:
- AI model usage và performance
- Workflow execution traces
- Token usage và costs
- Error tracking

## 🔒 Bảo mật

- Environment variables cho API keys
- Input validation và sanitization
- Rate limiting cho AI calls
- Secure file handling

## 📈 Performance

- **Parallel tool calls** để tăng tốc
- **Streaming responses** cho UX tốt hơn
- **LLM fallback** để đảm bảo availability
- **Caching** color palettes giữa slides

## 🤝 Đóng góp

1. Fork repository
2. Tạo feature branch: `git checkout -b feature/new-feature`
3. Commit changes: `git commit -m 'Add new feature'`
4. Push to branch: `git push origin feature/new-feature`
5. Tạo Pull Request

## 📄 License

MIT License - xem file [LICENSE](LICENSE) để biết thêm chi tiết.

## 🆘 Hỗ trợ

Nếu gặp vấn đề:
1. Kiểm tra logs trong console
2. Xem file `workflow.log`
3. Tạo issue với thông tin chi tiết
4. Kiểm tra API keys và internet connection

## 🚧 Roadmap

- [ ] Hỗ trợ thêm mô hình AI
- [ ] Template library
- [ ] Collaboration features
- [ ] Mobile responsive
- [ ] Video export
- [ ] Voice narration



