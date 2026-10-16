from pydantic import BaseModel


class PredictionResponse(BaseModel):
    label: str  # "melanoma" hoặc "nevus"
    probability_melanoma: float  # 0-1
    confidence: float  # xác suất của nhãn được chọn (max(p, 1-p))
    disclaimer: str = (
        "Đây là công cụ demo học thuật, KHÔNG phải công cụ chẩn đoán y tế. "
        "Kết quả chỉ mang tính tham khảo - hãy đến gặp bác sĩ da liễu để được chẩn đoán chính xác."
    )
