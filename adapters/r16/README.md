# LoRA Adapter r=16

## Cách lấy `adapter_model.safetensors`

File weights **không thể tạo trên máy không có GPU**. Làm trên Colab:

1. Mở `notebook.ipynb` trên Google Colab (Runtime → GPU)
2. Chạy toàn bộ notebook (hoặc ít nhất đến khi train xong r=16)
3. Chạy cell **Download r16 adapter** (trước Submission Checklist)
4. Giải nén `r16_adapter.zip` và copy vào thư mục này:

```
adapters/r16/
├── adapter_config.json
├── adapter_model.safetensors  ← cần copy từ Colab
└── README.md
```

5. Tạo lại `lab21_2A202600979_NgoThiAnh.zip` và nộp LMS

**Đường dẫn trên Colab:** `/content/lab21_lora_t4/r16/`
