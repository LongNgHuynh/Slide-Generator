# 📡 API Documentation - AI Slide Generator

## 📖 Tổng quan

AI Slide Generator cung cấp RESTful API và WebSocket API cho việc tạo và quản lý slide thuyết trình.

## 🔗 Base URL
```
http://localhost:5000
```

## 🌐 REST API Endpoints

### 1. Main Interface

#### GET `/`
Trả về giao diện web chính của ứng dụng.

**Response:**
- Content-Type: `text/html`
- Status: `200 OK`

---

### 2. Slide Export

#### GET `/export_pdf/all`
Xuất tất cả slides thành file PDF kết hợp.

**Response:**
```json
{
  "success": true,
  "filename": "presentation_20231201_143022.pdf"
}
```

**Download:**
- Content-Type: `application/pdf`
- Content-Disposition: `attachment; filename="presentation.pdf"`

**Error Response:**
```json
{
  "success": false,
  "error": "No slides found to export"
}
```

#### GET `/export_pdf/<int:slide_number>`
Xuất slide cụ thể thành PDF.

**Parameters:**
- `slide_number` (int): Số thứ tự slide (1-based)

**Example:**
```bash
GET /export_pdf/3
```

**Response:**
- Content-Type: `application/pdf`
- Filename: `slide_003_timestamp.pdf`

#### GET `/export_pdf/individual`
Xuất tất cả slides thành các file PDF riêng lẻ, đóng gói trong ZIP.

**Response:**
- Content-Type: `application/zip`
- Filename: `slides_individual_timestamp.zip`

---

### 3. PowerPoint Export

#### GET `/convert_all_slides_to_pptx`
Chuyển đổi tất cả slides thành file PowerPoint.

**Response:**
```json
{
  "success": true,
  "message": "PPTX conversion completed"
}
```

**Download:**
- Content-Type: `application/vnd.openxmlformats-officedocument.presentationml.presentation`
- Filename: `presentation_timestamp.pptx`

**Error Response:**
```json
{
  "success": false,
  "error": "ConvertAPI key not configured"
}
```

#### GET `/convert_single_slide_to_pptx/<int:slide_number>`
Chuyển đổi slide cụ thể thành PowerPoint.

**Parameters:**
- `slide_number` (int): Số thứ tự slide

**Example:**
```bash
GET /convert_single_slide_to_pptx/2
```

#### GET `/convert_individual_slides_to_pptx`
Chuyển đổi tất cả slides thành các file PPTX riêng lẻ.

**Response:**
- Content-Type: `application/zip`
- Filename: `slides_individual_pptx_timestamp.zip`

---

### 4. PDF to PPTX Conversion

#### POST `/convert_to_pptx`
Chuyển đổi file PDF thành PowerPoint (flexible endpoint).

**Request Body:**
```json
{
  "pdf_file_path": "/path/to/file.pdf",
  "output_filename": "my_presentation"
}
```

**Query Parameters (GET):**
- `pdf_file_path`: Đường dẫn đến file PDF
- `output_filename`: Tên file output (optional)

**Response:**
```json
{
  "success": true,
  "pptx_path": "/exports/presentation_timestamp.pptx"
}
```

---

### 5. Slide Information

#### GET `/api/slides/available`
Lấy danh sách slides có sẵn.

**Response:**
```json
{
  "success": true,
  "slides": [
    {
      "slide_number": 1,
      "filename": "slide_001.html",
      "path": "/generated_slides/slide_001.html",
      "exists": true
    },
    {
      "slide_number": 2,
      "filename": "slide_002.html", 
      "path": "/generated_slides/slide_002.html",
      "exists": true
    }
  ],
  "total_slides": 2
}
```

#### GET `/api/conversion/options`
Lấy thông tin về các tùy chọn xuất file.

**Response:**
```json
{
  "success": true,
  "conversion_options": {
    "pdf_export": {
      "available": true,
      "formats": ["single_slide", "all_slides"]
    },
    "pptx_export": {
      "available": false,
      "formats": [],
      "note": "Requires ConvertAPI key",
      "requirements": {
        "convertapi_package": true,
        "api_key": false
      }
    }
  },
  "slides": {
    "total": 5,
    "available_slides": [...]
  },
  "status": {
    "convertapi_configured": false,
    "convertapi_installed": true,
    "slides_available": true
  }
}
```

---

## 🔌 WebSocket API (Socket.IO)

### Connection
```javascript
const socket = io('http://localhost:5000');
```

### 1. Client Events (Emit)

#### `send_message`
Gửi tin nhắn để bắt đầu tạo presentation.

**Data:**
```javascript
{
  message: "Tạo presentation về AI trong giáo dục"
}
```

#### `edit_text`
Chỉnh sửa text trên slide.

**Data:**
```javascript
{
  slide_number: 2,
  text_id: 0,
  new_text: "Nội dung mới",
  original_text: "Nội dung cũ",
  is_rich_text: true
}
```

#### `ai_edit_slide`
Yêu cầu AI chỉnh sửa slide.

**Data:**
```javascript
{
  slide_number: 1,
  user_request: "Thêm màu xanh và làm chữ to hơn",
  current_content: "<html>...</html>"
}
```

### 2. Server Events (Listen)

#### `session_created`
Thông báo session được tạo.

**Data:**
```javascript
{
  session_id: "uuid-string"
}
```

#### `chat_message`
Tin nhắn trong chat.

**Data:**
```javascript
{
  type: "assistant", // hoặc "user", "system", "supervisor"
  message: "Đang bắt đầu tạo presentation...",
  timestamp: "2023-12-01T14:30:22.000Z"
}
```

#### `slide_generated`
Slide mới được tạo.

**Data:**
```javascript
{
  slide_number: 1,
  content: "<html>...</html>",
  timestamp: "2023-12-01T14:30:22.000Z",
  title: "Introduction"
}
```

#### `text_updated`
Text trên slide được cập nhật.

**Data:**
```javascript
{
  slide_number: 2,
  content: "<html>...</html>",
  timestamp: "2023-12-01T14:30:22.000Z"
}
```

#### `ai_slide_edited`
Slide được AI chỉnh sửa xong.

**Data:**
```javascript
{
  slide_number: 1,
  content: "<html>...</html>",
  timestamp: "2023-12-01T14:30:22.000Z",
  user_request: "Thêm màu xanh",
  ai_model_used: "Claude 3.7 Sonnet"
}
```

#### `agent_stream_start`
Bắt đầu streaming từ AI agent.

**Data:**
```javascript
{
  agent: "outline_agent"
}
```

#### `agent_stream_token`
Token streaming từ AI.

**Data:**
```javascript
{
  agent: "outline_agent",
  token: "text token",
  accumulated: "accumulated text so far"
}
```

#### `agent_stream_end`
Kết thúc streaming.

**Data:**
```javascript
{
  agent: "outline_agent",
  final_content: "Complete response text"
}
```

#### `agent_tool_call`
AI agent đang sử dụng tool.

**Data:**
```javascript
{
  agent: "outline_agent",
  tool: "web_search",
  content: "Searching for information about..."
}
```

#### `workflow_complete`
Workflow hoàn thành.

**Data:**
```javascript
{
  total_slides: 5,
  iterations: 12
}
```

#### `error`
Thông báo lỗi.

**Data:**
```javascript
{
  message: "Error description"
}
```

---

## 🔧 API Client Examples

### JavaScript/Node.js

#### Basic Slide Generation
```javascript
const io = require('socket.io-client');
const socket = io('http://localhost:5000');

socket.on('connect', () => {
  console.log('Connected to server');
  
  // Request slide generation
  socket.emit('send_message', {
    message: 'Tạo presentation về Machine Learning'
  });
});

socket.on('slide_generated', (data) => {
  console.log(`Generated slide ${data.slide_number}`);
  // Save or process slide content
});

socket.on('workflow_complete', (data) => {
  console.log(`Completed! Generated ${data.total_slides} slides`);
});
```

#### Export Slides
```javascript
// Export all slides to PDF
fetch('/export_pdf/all')
  .then(response => response.blob())
  .then(blob => {
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = 'presentation.pdf';
    a.click();
  });

// Get available slides
fetch('/api/slides/available')
  .then(response => response.json())
  .then(data => {
    console.log(`Available slides: ${data.total_slides}`);
    data.slides.forEach(slide => {
      console.log(`Slide ${slide.slide_number}: ${slide.filename}`);
    });
  });
```

### Python

#### Using requests and python-socketio
```python
import requests
import socketio

# REST API example
def export_pdf():
    response = requests.get('http://localhost:5000/export_pdf/all')
    if response.status_code == 200:
        with open('presentation.pdf', 'wb') as f:
            f.write(response.content)
        print("PDF exported successfully")

# Socket.IO example
sio = socketio.Client()

@sio.on('connect')
def on_connect():
    print('Connected to server')
    sio.emit('send_message', {'message': 'Create presentation about AI'})

@sio.on('slide_generated')
def on_slide_generated(data):
    print(f"Generated slide {data['slide_number']}")

@sio.on('workflow_complete')
def on_complete(data):
    print(f"Completed! Generated {data['total_slides']} slides")

sio.connect('http://localhost:5000')
sio.wait()
```

### cURL Examples

#### Export PDF
```bash
# Export all slides to PDF
curl -O -J http://localhost:5000/export_pdf/all

# Export specific slide
curl -O -J http://localhost:5000/export_pdf/3

# Get slides information
curl http://localhost:5000/api/slides/available
```

#### Convert to PPTX
```bash
# Convert all slides to PowerPoint
curl -O -J http://localhost:5000/convert_all_slides_to_pptx

# Convert PDF to PPTX
curl -X POST http://localhost:5000/convert_to_pptx \
  -H "Content-Type: application/json" \
  -d '{"pdf_file_path": "/path/to/file.pdf", "output_filename": "my_presentation"}'
```

---

## 🚨 Error Handling

### HTTP Status Codes
- `200 OK`: Successful request
- `400 Bad Request`: Invalid parameters
- `404 Not Found`: Resource not found
- `500 Internal Server Error`: Server error

### Error Response Format
```json
{
  "success": false,
  "error": "Detailed error message",
  "code": "ERROR_CODE" // optional
}
```

### Common Errors
- `SLIDES_NOT_FOUND`: No slides available for export
- `CONVERTAPI_NOT_CONFIGURED`: ConvertAPI key missing
- `EXPORT_FAILED`: Export process failed
- `INVALID_SLIDE_NUMBER`: Slide number out of range

---

## 🔒 Authentication & Security

### API Keys
Một số tính năng yêu cầu API keys được cấu hình:
- **ConvertAPI**: Cho PPTX export
- **AI Models**: Azure OpenAI, Google AI, AWS Bedrock

### Rate Limiting
- Mỗi session có giới hạn về số lượng requests
- AI model calls có throttling để tránh rate limits
- File exports có timeout limits

### Input Validation
- HTML content được sanitize
- File paths được validate
- Input size limits áp dụng

---

## 📊 Monitoring & Analytics

### Langfuse Integration
API tự động track:
- Request/response times
- AI model usage
- Error rates
- User sessions

### Metrics Available
- Slide generation success rate
- Average processing time
- Model performance comparison
- Export success rates

---

**API này cung cấp đầy đủ chức năng để tích hợp AI Slide Generator vào các ứng dụng khác.** 