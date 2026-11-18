# RF-only Research Evaluation

## ตารางผลหลัก

| Scenario | TN | FP | FN | TP | Precision | Recall | F1 | FPR | บทบาท |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| AWID3 temporal test | 86 | 0 | 0 | 72 | 1.0000 | 1.0000 | 1.0000 | 0.0000 | same-capture temporal test |
| D05-A-R2 | 489 | 0 | 72 | 289 | 1.0000 | 0.8006 | 0.8892 | 0.0000 | previously-inspected cross-dataset test |


## Baseline บน W20 เดียวกัน

| Scenario | Model | N | TN | FP | FN | TP | F1 | FPR |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| AWID3 temporal test | Frozen RF | 158 | 86 | 0 | 0 | 72 | 1.0000 | 0.0000 |
| AWID3 temporal test | CART | 158 | 86 | 0 | 72 | 0 | 0.0000 | 0.0000 |
| AWID3 temporal test | Logistic Regression | 158 | 86 | 0 | 0 | 72 | 1.0000 | 0.0000 |
| AWID3 temporal test | KNN | 158 | 86 | 0 | 0 | 72 | 1.0000 | 0.0000 |
| AWID3 temporal test | LSTM | 150 | 82 | 0 | 0 | 68 | 1.0000 | 0.0000 |
| D05-A-R2 | Frozen RF | 850 | 489 | 0 | 72 | 289 | 0.8892 | 0.0000 |
| D05-A-R2 | CART | 850 | 0 | 489 | 361 | 0 | 0.0000 | 1.0000 |
| D05-A-R2 | Logistic Regression | 850 | 489 | 0 | 361 | 0 | 0.0000 | 0.0000 |
| D05-A-R2 | KNN | 850 | 489 | 0 | 355 | 6 | 0.0327 | 0.0000 |
| D05-A-R2 | LSTM | 842 | 485 | 0 | 357 | 0 | 0.0000 | 0.0000 |


## Lightweight

- Frozen artifact: **126.2 KiB**
- Detection-only latency/W20: P50 **15.99 ms**, P95 **16.88 ms**, P99 **19.35 ms**
- Single-window-call throughput: **62.8 W20/s**
- Python incremental allocation: **4.21 MiB**; isolated process peak RSS: **425.7 MiB**

สรุป: เรียกว่า lightweight ได้เฉพาะในความหมายว่า artifact เล็กและ inference ทันอัตรา Beacon ทั่วไปบนเครื่องทดสอบ แต่ยังไม่ควรอ้างว่า memory-light สำหรับ embedded deployment เพราะ Python process RSS สูง และยังไม่มี benchmark บนอุปกรณ์ปลายทางจริง

## การเปรียบเทียบโมเดล

- RF, CART, Logistic Regression และ KNN ใช้ W20 เดียวกันใน AWID3 test และ D05-A-R2; LSTM ใช้ causal sequence 5 จึง support ต่างกันเล็กน้อย
- AWID3 temporal test: RF, Logistic Regression, KNN และ LSTM ได้ F1=1.0 แต่ CART ได้ 0 แสดงว่าผล internal ยังไวต่อชนิดโมเดลและ temporal block
- D05-A-R2: RF F1=0.8892 และเป็น baseline ที่ดีที่สุดในชุดเปรียบเทียบนี้
- จึงสรุปได้ว่า RF generalize จาก AWID3 ไปยัง D05-A-R2 ได้ดีกว่า baseline ที่ทดลอง โดยยังต้องเปิดเผยว่า D05-A-R2 เป็น previously inspected test

## Window-size ablation

AWID3 W10, W20 และ W50 ได้ F1=1.0 ทั้งหมดในผลเดิม จึงเลือก W20 ตาม protocol ที่ freeze ไว้ ไม่ได้เลือกย้อนหลังจาก D05-A-R2

## การเทียบกับงานวิจัย

ETGuard, ET-spotter, Seo et al. และ Wi-Fi Frame Classification ควรใช้ใน Related Work เท่านั้น ตัวเลขห้ามเทียบเป็น ranking โดยตรง เนื่องจาก dataset, attack construction, observation time และ prediction unit ไม่เหมือนกัน รายละเอียดอยู่ใน `literature_comparison.csv`.

## กราฟ

- `figures/rf_generalization_f1.png`
- `figures/model_comparison_d05ar2.png`
- `figures/rf_latency.png`
- `figures/rf_confusion_matrices.png`

## ข้อจำกัดการอ้างผล

D05-A-R2 เป็น previously inspected cross-dataset test; การเลือก RF เป็น final model หลังเห็นผลนี้ต้องเปิดเผย และไม่ควรเรียกว่า untouched independent model-selection evidence
