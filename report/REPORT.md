# Báo cáo bài thực hành Ngày 1 – Đọc prediction YOLO11 để hiểu định dạng nhãn

**Ngày chạy:**

**Runtime Colab:** CPU/GPU

**Python / PyTorch / Ultralytics:**

**Checkpoint:** `yolo11n-cls.pt`, `yolo11n.pt`, `yolo11n-seg.pt`

**Ngưỡng detection/segmentation dùng khi tạo JSON:** `0.35` (chỉ đổi nếu GV cho phép; nếu đổi hãy ghi rõ)

**Thay đổi so với notebook nguồn:** Không / mô tả rõ thay đổi

> **Cách dùng mẫu này:** copy/điền trên máy, lưu thành `REPORT.md`, rồi **upload** vào panel Files
> của Colab (notebook không tự tạo file này). Ô cuối Colab sẽ đóng ZIP cùng `day1_lab_outputs/`.
> Giải nén và đưa vào `report/` trên GitHub, rồi nộp **link repo** trên LMS.
> Không ghi họ tên, MSSV, email, số điện thoại hoặc dữ liệu cá nhân khác vào báo cáo/output;
> họ tên/MSSV chỉ nằm trong tên repository.

### Từ cần nhớ (đọc nhanh trước khi viết)

| Từ | Nghĩa ngắn |
| --- | --- |
| Prediction | Kết quả model vừa đoán ra; chưa phải nhãn đúng |
| Ground truth | Nhãn do người gán theo quy tắc |
| Taxonomy | Danh sách tên lớp có sẵn kèm model (vd. ImageNet-1K, COCO-80) |
| Checkpoint | File model `.pt` đã huấn luyện sẵn |
| Guideline | Quy tắc gán nhãn do người viết |
| Annotator | Người gán nhãn |
| Reviewer / QC | Người kiểm tra nhãn |
| Escalation | Ca khó → hỏi Lab Coach, không tự bịa |
| Model score | Số “tin” của model; không phải điểm chất lượng nhãn người |

## 1. Phân loại ảnh

**Mở file:** `day1_lab_outputs/classification_predictions.json` và `visuals/classification_top5.png`.  
Chỉ dùng ảnh `sample_id = traffic`, lấy dòng `rank = 1`.

1. Chép record hạng 1: `class_id`= 468 ; `class_name`= "cab" ; `rank`=1 ; `score`= 0.510915 ; `taxonomy_name`= "ImageNet-1K"
2. Record này nói về cả bức ảnh. Bởi vì trong bức ảnh không có bbox hay polygon chỉ có thống kê các loại xe.
3. Danh sách tên lớp (class list) do đâu mà có: model tự nghĩ ra, hay đi kèm taxonomy/checkpoint?
4. Vì 'class_id' chỉ có nghĩa trong một taxonomy cụ thể: cùng một số ID ở taxonomy khác là lớp khác. Tên lớp có thể trùng hoặc viết khác nhau. Ghi đủ 3 trường thì tra lại được và tái lập được kết quả.
5. Khi trong ảnh có nhiều vật hoặc nhiều chủ đề (multi-object / multi-label), guideline (hướng dẫn gán nhãn) cần quy định một hệ thống phân cấp ưu tiên rõ ràng và không mơ hồ để người gán nhãn chọn được một lớp duy nhất.
6. Score chỉ là mức tin của model, chưa được người xác nhận. Model có thể sai mà score vẫn cao. Nếu lấy prediction làm ground truth thì lỗi của model sẽ bị đưa ngược vào dữ liệu huấn luyện.

## 2. Phát hiện vật thể

**Mở file:** `day1_lab_outputs/detection_predictions.json` và `visuals/detection_predictions.png`.  
Chỉ dùng ảnh `sample_id = kitchen`. Chọn **một** record.  
Số lượng prediction theo ngưỡng: xem output ô so sánh `0.20` / `0.35` / `0.60` trong notebook.

1. Chép một record: `class_name`= "person" ; `score`= 0.912625 ; `bbox_xyxy`=[385.33,69.24,498.92,348.92] ; `bbox_width`= 113.58 ; `bbox_height`= 279.68
2. Giải thích box bằng lời thường: góc trên-trái ở 385.33, 69.24, góc dưới-phải 498.92, 348.92, rộng 113.58 px cao 279.68 px.
3. Chọn hai ngưỡng 0.20 và 0.60. Ngưỡng 0.20 có 12 còn 0.60 có 6.
4. Hạ ngưỡng thì giữ thêm nhiều box có score thấp. Reviewer phải kiểm tra thêm: box sai (không có vật thật), box trùng (2 box cho 1 vật), sai class. Bù lại, cách này có thể bắt được vật mà ngưỡng cao bỏ sót. Ngưỡng chỉ là cấu hình lọc prediction, không phải quy tắc bỏ qua vật khi tạo ground truth.
5. Mỗi vật thuộc class list có đúng 1 box. Box ôm sát phần nhìn thấy được: 4 cạnh chạm điểm ngoài cùng của vật, không chừa nền thừa và không cắt mất phần vật. Phần bị che thì không đoán và không vẽ thêm. Vật bị cắt ở mép ảnh thì box dừng ở mép ảnh. Không bỏ qua vật chỉ vì nhỏ hoặc mờ.
6. person 0.61. Hình đã hiện rõ 1 phần cánh tay con người nên có thể xác định vật thể (dùng guideline).

## 3. Phân đoạn theo từng đối tượng

**Mở file:** `day1_lab_outputs/segmentation_predictions.json` và `visuals/segmentation_prediction.png`.  
Chỉ dùng ảnh `sample_id = kitchen`. Chọn **một** record.

1. Chép một record: `instance_id`= "kitchen-006" ; `class_name`= "bowl" ; `score`= 0.528201 ; số điểm polygon=`polygon_point_count`= 84 ; 3–5 điểm đầu của `polygon_xy`= [
        29.0,
        303.0
      ],
      [
        28.0,
        304.0
      ],
      [
        26.0,
        304.0
      ],
      [
        25.0,
        305.0
      ],
      [
        23.0,
        305.0
      ],
2. Box chỉ có 4 số, luôn là hình chữ nhật, nên bao luôn cả phần nền ở 4 góc. Polygon của kitchen-006 có 84 điểm bám theo biên cái bowl. Các điểm liên tiếp chỉ cách nhau 1–2 px. Nhờ vậy polygon cho biết hình dạng thật (miệng bowl tròn hoặc elip), diện tích thật, và tách được vật khỏi nền hay khỏi vật bên cạnh. Box không làm được những việc này.
3. kitchen-006 nghĩa là ảnh kitchen, vật thứ 6 trong danh sách prediction của ảnh đó. Nó dùng để phân biệt từng vật riêng trong output.

Không phải class ID: lớp của vật là bowl với class_id riêng. Nếu ảnh có 2 cái bowl thì chúng cùng class_id nhưng khác instance_id.
Không phải tracking ID: nó không theo dõi cùng một vật qua nhiều ảnh hay frame. Chạy lại hoặc đổi ngưỡng thì số thứ tự có thể thay đổi.
4. Mỗi vật là một polygon riêng, bám sát mép ngoài phần nhìn thấy được. Không tính bóng đổ, phản chiếu hay nền vào polygon. Phần bị che thì không đoán. Hai vật chạm nhau thì vẽ hai polygon riêng, ranh giới đi theo mép thật và không chồng lên nhau. Vật bị cắt ở mép ảnh thì polygon dừng ở mép ảnh.
5. dining table 0.58 chạm với bowl 0.74. Áp dụng guide line vẽ 2 polygon riêng.

## 4. Vòng đời và kiểm tra chất lượng

Lab hôm nay chỉ **xem prediction**. Chuỗi dưới đây là vòng đời production để bạn nối các khái niệm:

`ảnh thô → guideline → ground truth → huấn luyện → prediction → QC/rework`

Điền bảng (viết ngắn, có dẫn chứng từ JSON/PNG khi nói về lỗi):

| Tác vụ | Ground truth trông như thế nào? (đơn vị nhãn) | Lỗi hoặc chỗ khó quyết định bạn thấy trên ảnh/JSON | Annotator làm gì? | Reviewer xem gì? |
| --- | --- | --- | --- | --- |
| Phân loại ảnh | **1 nhãn cho cả ảnh**: `468` + `cab` lấy từ taxonomy ImageNet-1K | `traffic` rank 1 = `[cab]` có thể chưa đúng vì ảnh có nhiều chủ thể nên khó chọn một lớp | Chọn 1 lớp trong taxonomy theo quy tắc chủ thể chính; ca nhiều chủ đề mà rule chưa phủ → escalation | Lớp có nằm trong taxonomy không, `class_id` khớp `class_name` không, có đúng quy tắc ưu tiên không |
| Phát hiện vật thể | **1 box cho mỗi vật**: class + `bbox_xyxy` (pixel, gốc trên-trái) | `cup` score 0.381543, box [119.95, 272.2, 141.78, 304.16] bị che khó nhìn; khi nâng ngưỡng 0.35 → 0.60 thì mất `[6]` | Vẽ box ôm sát phần nhìn thấy cho **mọi** vật thuộc class list, kể cả vật nhỏ, mờ hoặc model bỏ sót | Đối chiếu với ảnh: thiếu box (bỏ sót), box thừa/trùng, box lỏng hoặc cắt mất vật, sai class → gửi rework |
| Instance segmentation | **1 polygon cho mỗi instance**: class + `polygon_xy` bám biên; vật cùng lớp vẫn là instance riêng | `kitchen-011` spoon (score 0.359148): mask chỉ phủ đầu muỗng, phần cán trong box [518.68, 41.52, 532.84, 112.53] không có mask → **thiếu mask**, không thừa ra nền; vật mảnh (~14 px), score sát ngưỡng | Vẽ polygon riêng cho từng vật, bám mép phần nhìn thấy, tách hai vật chạm nhau, dừng ở mép ảnh | Mask lẹm nền hoặc thiếu biên, hai vật bị gộp làm một, mỗi instance có ID riêng; ca chưa có rule → chuyển escalation |

## 5. An toàn dữ liệu

1. Chỉ dùng ba ảnh COCO công khai có sẵn trong notebook và không đưa bất kỳ dữ liệu cá nhân hay nội bộ nào lên Colab hoặc GitHub công khai.
2. Nếu gặp ảnh hoặc dữ liệu không đúng phạm vi, tôi sẽ không chạy tiếp, không upload, không commit, và không tự thay bằng ảnh khác (kể cả khi checksum báo sai). Sau đó tôi báo Lab Coach qua kênh hỗ trợ của lớp, kèm bước đang làm và ảnh chụp thông báo, không gửi password hay token. Nếu dữ liệu đó đã lỡ bị push lên GitHub công khai, tôi gỡ nó khỏi báo cáo và output rồi báo ngay GV/Lab Coach để xử lý.

## 6. Checklist trước khi nộp link repo

Kiểm tra trên GitHub trong thư mục `report/`:

- [ ] `REPORT.md` (file này đã điền)
- [ ] `day1_lab_outputs/classification_predictions.json`
- [ ] `day1_lab_outputs/detection_predictions.json`
- [ ] `day1_lab_outputs/segmentation_predictions.json`
- [ ] `day1_lab_outputs/IMAGE_ATTRIBUTION.md`
- [ ] `day1_lab_outputs/visuals/classification_top5.png`
- [ ] `day1_lab_outputs/visuals/detection_predictions.png`
- [ ] `day1_lab_outputs/visuals/segmentation_prediction.png`
- [ ] Ô validation cuối notebook báo `PASS`
- [ ] Không có họ tên, MSSV hoặc dữ liệu nhạy cảm trong báo cáo/output
- [ ] Đã nộp **link repository** trên LMS
