# So sánh FastAPI và Flask

| Tiêu chí | FastAPI | Flask |
| :--- | :--- | :--- |
| **Cú pháp Route** | `@app.get("/path")` (tách biệt theo method) | `@app.route("/path", methods=["GET"])` |
| **Cơ chế Async** | Native async/await (ASGI) | Synchronous (WSGI), hỗ trợ streaming nhưng không native async |
| **Validation** | Pydantic (type hints tự động sinh validation và docs) | Tự định nghĩa hoặc dùng thư viện bên thứ ba (Flask-WTF, Marshmallow) |
| **Docs** | TiviDoc/OpenAPI tự động sinh | Phải cài thêm thư viện (Flask-RESTX/Swagger) |
| **Tốc độ** | Rất cao (dựa trên Starlette và Pydantic) | Cao, nhưng thường chậm hơn FastAPI do không native async |
| **Mở rộng** | Dễ với backend hiện đại, microservices | Linh hoạt, nhiều extension, phù hợp app monolithic |
| **Cộng đồng** | Lớn và đang tăng trưởng nhanh | Lớn nhất trong hệ sinh thái Python, nhiều tài liệu cũ |

## Ghi chú
- FastAPI tối ưu cho API modern, microservices, và các hệ thống cần async.
- Flask phù hợp cho các ứng dụng nhỏ đến vừa, cần sự đơn giản và linh hoạt.