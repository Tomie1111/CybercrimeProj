# KNN และ LSTM: AWID3 → Real Captures

การเลือก hyperparameters และ threshold ใช้เฉพาะ AWID3 validation หลัง fit preprocessing บน AWID3 train เท่านั้น Real Captures ถูกเปิดประเมินภายหลังและไม่ได้ใช้เลือกโมเดล

## ผล AWID3 temporal test

- KNN: F1 `1.0000`, recall `1.0000`, FPR `0.0000` (ทุก W20 window)
- LSTM: F1 `1.0000`, recall `1.0000`, FPR `0.0000` (causal sequence 5 windows)

## ผล External target AP

- KNN: F1 `0.0261`, recall `0.0133`, FPR `0.0002`
- LSTM: F1 `0.0000`, recall `0.0000`, FPR `0.0000`

Ground truth ใช้ exact scope เท่านั้น: D00–D03 Deco `:1c`, D04 Zyxel `:28`, และ D05 ใช้ทั้งสองตัว โดย D05 Zyxel เป็น Evil Twin; BSSID อื่นถูกตัดก่อน prediction และ metric. LSTM ไม่ pad sequence จึงตัด 4 windows แรกของแต่ละ capture+BSSID และมี support ต่างจาก KNN

## ข้อจำกัด

AWID3 train/validation/test เป็น temporal blocks จาก capture และ device profiles เดียวกัน ดังนั้นคะแนน AWID3 สูงไม่ใช่หลักฐาน independent-device generalization. ผล external เป็น diagnostics แบบ AP-role/window และห้ามนำกลับไปเปลี่ยน candidate จากผลชุดนี้
