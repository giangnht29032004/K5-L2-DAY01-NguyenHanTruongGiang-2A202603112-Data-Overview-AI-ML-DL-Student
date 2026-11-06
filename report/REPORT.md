# Báo cáo bài thực hành Ngày 1 – Đọc prediction YOLO11 để hiểu định dạng nhãn

**Ngày chạy:**

**Runtime Colab:** CPU/GPU

**Python / PyTorch / Ultralytics:**

**Checkpoint:** `yolo11n-cls.pt`, `yolo11n.pt`, `yolo11n-seg.pt`

**Ngưỡng detection/segmentation dùng khi tạo JSON:** `0.35` (chỉ đổi nếu GV cho phép; nếu đổi hãy ghi rõ)

**Thay đổi so với notebook nguồn:** Không / mô tả rõ thay đổi

> \*\*Cách dùng mẫu này:\*\* copy/điền trên máy, lưu thành `REPORT.md`, rồi \*\*upload\*\* vào panel Files
> của Colab (notebook không tự tạo file này). Ô cuối Colab sẽ đóng ZIP cùng `day1\_lab\_outputs/`.
> Giải nén và đưa vào `report/` trên GitHub, rồi nộp \*\*link repo\*\* trên LMS.
> Không ghi họ tên, MSSV, email, số điện thoại hoặc dữ liệu cá nhân khác vào báo cáo/output;
> họ tên/MSSV chỉ nằm trong tên repository.

### Từ cần nhớ (đọc nhanh trước khi viết)

|Từ|Nghĩa ngắn|
|-|-|
|Prediction|Kết quả model vừa đoán ra; chưa phải nhãn đúng|
|Ground truth|Nhãn do người gán theo quy tắc|
|Taxonomy|Danh sách tên lớp có sẵn kèm model (vd. ImageNet-1K, COCO-80)|
|Checkpoint|File model `.pt` đã huấn luyện sẵn|
|Guideline|Quy tắc gán nhãn do người viết|
|Annotator|Người gán nhãn|
|Reviewer / QC|Người kiểm tra nhãn|
|Escalation|Ca khó → hỏi Lab Coach, không tự bịa|
|Model score|Số “tin” của model; không phải điểm chất lượng nhãn người|

## 1\. Phân loại ảnh

**Mở file:** `day1\_lab\_outputs/classification\_predictions.json` và `visuals/classification\_top5.png`.  
Chỉ dùng ảnh `sample\_id = traffic`, lấy dòng `rank = 1`.

1. Chép record hạng 1: `class\_id`=468 ; `class\_name`="cab" ; `rank`=1 ; `score`=0.510915 ; `taxonomy\_name`= "ImageNet-1K"
2. Record này nói về  **một vật thể riêng** trong ảnh. Vì còn nhiều record với các classname khác trong ảnh này.
3. Danh sách tên lớp (class list) do đâu mà có:  do đi kèm taxonomy/checkpoint.
4. Vì sao cần ghi cả `class\_id`, `class\_name` và `taxonomy\_name`, không chỉ ghi mỗi tên lớp? Việc ghi cả 3 trường này để đảm bảo dữ liệu chính xác cho máy tính, hiểu được cho con người và rõ ràng về ngữ cảnh.
5. Nếu trong ảnh có nhiều vật / nhiều chủ đề, guideline cần nói rõ điều gì để người gán nhãn chọn **một** lớp? Guideline cần quy định rõ bộ quy tắc ưu tiên và loại trừ để người gán nhán chọn.
6. Vì sao không được lấy `score` của model làm ground truth? Vì đôi khi model đánh dấu sai vật thể.

## 2\. Phát hiện vật thể

**Mở file:** `day1\_lab\_outputs/detection\_predictions.json` và `visuals/detection\_predictions.png`.  
Chỉ dùng ảnh `sample\_id = kitchen`. Chọn **một** record.  
Số lượng prediction theo ngưỡng: xem output ô so sánh `0.20` / `0.35` / `0.60` trong notebook.

1. Chép một record: `class\_name`="cup" ; `score`=0.381542 ; `bbox\_xyxy`=\[119.95,272.2,141.78,304.16] ; `bbox\_width`=21.84 ; `bbox\_height`=31.96
2. Giải thích box bằng lời thường: góc trên-trái ở đâu, góc dưới-phải ở đâu, rộng/cao bao nhiêu pixel? Góc trên - trái của chiếc cốc nằm tại tọa độ: X = 119.95 pixel và Y = 272.2 pixel, Góc dưới - phải của chiếc cốc nằm tại tọa độ: X = 141.78 pixel và Y = 304.16 pixel, rộng 21.84 pixel, cao 31.96 pixel.
3. Chọn hai ngưỡng (vd. 0.20 và 0.60). Mỗi ngưỡng có bao nhiêu prediction? 0.2 có 1 prediction, 0.6 có 0 prediction.
4. Khi hạ ngưỡng (giữ nhiều hộp hơn), người **reviewer** phải làm thêm việc gì? Người reviewer sẽ phải tốn nhiều thời gian và công sức hơn rất nhiều để xóa hộp sai, sửa hộp lệch do mô hình sinh ra.
5. Viết **một quy tắc box** để người gán nhãn làm theo (áp dụng được cho nhiều ảnh, không chỉ ảnh này).
Quy tắc: Khi gán nhãn, người gán nhãn phải vẽ hộp bao vừa khít sát vào các cạnh ngoài cùng (trên, dưới, trái, phải) của vật thể, tuyệt đối không để thừa nhiều bối cảnh và không cắt lẹm vào vật thể. Đối với các vật thể bị che khuất một phần bởi vật khác, hộp bao phải bao quanh toàn bộ phần nhìn thấy được của đối tượng đó. Chỉ gán nhãn những vật thể có thể nhận diện rõ bằng mắt thường và có kích thước lớn hơn 10x10 pixel, bỏ qua các vật thể quá mờ hoặc quá nhỏ ở hậu cảnh để tránh làm nhiễu dữ liệu.
6. Trên ảnh `kitchen`, chỉ ra **một** vật bị che khuất hoặc cắt mép. Bạn sẽ viết thành quy tắc trong guideline, hay đưa lên hỏi Lab Coach (escalation)? Vì sao?
Trong ảnh có 1 cái cốc bị che khuất, tôi chọn viết trường hợp này thành một quy tắc rõ ràng trong guideline vì hiện tượng vật thể bị che khuất (occlusion) hoặc bị cắt mép (truncation) là lỗi phổ biến, lặp đi lặp lại trên hàng nghìn bức ảnh trong các bài toán Computer Vision thực tế. Nếu mỗi lần gặp người gán nhãn đều đưa lên hỏi Lab Coach (escalation), quy trình làm việc sẽ bị nghẽn mạch, gây lãng phí thời gian chờ đợi và làm quá tải người quản lý bằng những câu hỏi trùng lặp. Việc chuẩn hóa hành vi này vào guideline giúp người gán nhãn có thể tự tra cứu, tự tin ra quyết định ngay lập tức, từ đó tối ưu hóa tốc độ dự án và đảm bảo tính nhất quán của toàn bộ tập dữ liệu

## 3\. Phân đoạn theo từng đối tượng

**Mở file:** `day1\_lab\_outputs/segmentation\_predictions.json` và `visuals/segmentation\_prediction.png`.  
Chỉ dùng ảnh `sample\_id = kitchen`. Chọn **một** record.

1. Chép một record: `instance\_id`=… ; `class\_name`=… ; `score`=… ; số điểm polygon=`polygon\_point\_count`=… ; 3–5 điểm đầu của `polygon\_xy`=…
"instance\_id"= "kitchen-002"; "class\_name"= "bowl";  "polygon\_point\_count"= 67; "polygon\_xy": \[
\[
53.0,
344.0
],
\[
52.0,
345.0
],
\[
50.0,
345.0
]
]
2. So với box, polygon cho thêm thông tin gì về hình dạng vật thể?
So với box, polygon cung cấp thông tin chính xác về đường biên, hình khối và độ nghiêng thực tế của vật thể.
3. `instance\_id` dùng để làm gì? Nó **không** phải class ID và **không** phải tracking ID — bạn hiểu câu này thế nào?
instance\_id được dùng để phân biệt các cá thể khác nhau thuộc cùng một lớp trong cùng một bức ảnh.
Câu nói đó có nghĩa là instance\_id chỉ dùng để đếm và tách biệt các cá thể cùng loài trong cùng một bức ảnh. Nó không phải class ID vì class ID là mã chung cho cả nhóm (ví dụ: mọi cái cốc đều chung một class ID), còn instance\_id là số thứ tự riêng của từng cái cốc (cốc số 1, cốc số 2). Nó cũng không phải tracking ID vì tracking ID dùng để theo dõi một vật thể di chuyển xuyên suốt nhiều khung hình video, trong khi instance\_id chỉ có giá trị độc lập và duy nhất trong phạm vi của một bức ảnh tĩnh.
4. Viết **một quy tắc vẽ biên mask** để người gán nhãn làm theo.
Quy tắc vẽ biên mặt nạ (Mask): Khi gán nhãn phân đoạn (segmentation), người gán nhãn phải tô màu hoặc chấm các điểm polygon bám sát theo từng pixel đường viền thực tế của vật thể, phân tách tuyệt đối giữa đối tượng và bối cảnh. Đối với các cạnh cong, răng cưa hoặc có chi tiết phức tạp (như quai cốc, kẽ lá), biên mask phải đi theo đúng độ lồi lõm thực tế, không được cắt góc tạo thành các đường thẳng thô kệch và không được lem ra ngoài phông nền.
5. Trên ảnh `kitchen`, chỉ ra **một** chỗ biên mờ / hai vật chạm nhau / bị che. Bạn dùng guideline hay escalation? Vì sao?
Trong ảnh có 1 cái bát bị che. Tôi chọn dùng guideline có sẵn kết hợp với quy trình escalation một lần duy nhất để cập nhật ảnh mẫu. Lý do là vì biên mờ, vật chạm nhau hay bị che là các "ca khó điển hình" trong dán nhãn phân đoạn, cần áp dụng ngay quy tắc chung trong guideline (như ước lượng khối hoặc vẽ đường cắt đôi ranh giới) để không nghẽn tiến độ. Việc escalate lên Lab Coach đối với ảnh kitchen cụ thể này không phải để hỏi cách làm, mà là để xin quyết định chuẩn (Ground Truth), từ đó lấy bức ảnh này làm mẫu trực quan đưa ngược lại vào guideline, giúp người gán nhãn tự tra cứu và triệt tiêu hoàn toàn việc hỏi lại sau này.

## 4\. Vòng đời và kiểm tra chất lượng

Lab hôm nay chỉ **xem prediction**. Chuỗi dưới đây là vòng đời production để bạn nối các khái niệm:

`ảnh thô → guideline → ground truth → huấn luyện → prediction → QC/rework`

Điền bảng (viết ngắn, có dẫn chứng từ JSON/PNG khi nói về lỗi):

|Tác vụ|Ground truth trông như thế nào? (đơn vị nhãn)|Lỗi hoặc chỗ khó quyết định bạn thấy trên ảnh/JSON|Annotator làm gì?|Reviewer xem gì?|
|-|-|-|-|-|
|Phân loại ảnh|Một nhãn cho toàn bộ ảnh (image-level label), ví dụ: traffic.|Model dự đoán cab 0.51, minibus 0.16. Có nhiều loại phương tiện nên dễ nhầm giữa nhãn đại diện cho cảnh và nhãn phương tiện.|Gán nhãn chính cho toàn ảnh theo nội dung và quy tắc dataset.|Kiểm tra nhãn có phản ánh đúng nội dung tổng thể không, có nhầm giữa chủ đề ảnh và vật thể không.|
|Phát hiện vật thể|Mỗi vật thể có một bounding box (tọa độ x, y, w, h) và class label.|PNG: nhiều box bowl chồng lấn; cup 0.45 có độ tin cậy thấp; một phần người ở mép trái bị cắt. Khó xác định box chính xác cho vật thể bị che khuất.|Vẽ box bao quanh từng vật thể, gán đúng class; xử lý vật thể bị che khuất theo quy định.|Kiểm tra box có ôm sát vật thể không, có bỏ sót, gán nhầm hoặc tạo box trùng lặp không.|
|Instance segmentation|Mỗi cá thể có một segmentation mask riêng và class label.|PNG: mask dining table 0.58 phủ vùng mặt bàn; nhiều mask bowl chồng lấn; spoon 0.49 có độ tin cậy thấp. Ranh giới vật thể nhỏ hoặc bị che khuất khó xác định.|Tô mask riêng cho từng cá thể, bám sát đường viền vật thể và giữ phân biệt giữa các cá thể cùng loại.|Kiểm tra mask có đúng đường viền không, có lấn sang vật thể khác, bỏ sót hoặc gộp nhầm các cá thể không.|

## 5\. An toàn dữ liệu

1. Quy tắc bảo vệ dữ liệu của bạn cho bài lab này là gì? (viết 1 câu)
Chỉ sử dụng dữ liệu phục vụ mục đích thử nghiệm, ẩn danh thông tin nhạy cảm, không chia sẻ dữ liệu ra bên ngoài và chỉ cấp quyền truy cập cho người có trách nhiệm.
2. Nếu gặp ảnh hoặc dữ liệu không đúng phạm vi, bạn sẽ dừng và báo cho ai?
Tôi sẽ dừng xử lý dữ liệu và báo ngay cho trưởng nhóm hoặc người phụ trách dự án để kiểm tra, xác nhận hướng xử lý trước khi tiếp tục.
## 6\. Checklist trước khi nộp link repo

Kiểm tra trên GitHub trong thư mục `report/`:

* \[ ] `REPORT.md` (file này đã điền)
* \[ ] `day1\_lab\_outputs/classification\_predictions.json`
* \[ ] `day1\_lab\_outputs/detection\_predictions.json`
* \[ ] `day1\_lab\_outputs/segmentation\_predictions.json`
* \[ ] `day1\_lab\_outputs/IMAGE\_ATTRIBUTION.md`
* \[ ] `day1\_lab\_outputs/visuals/classification\_top5.png`
* \[ ] `day1\_lab\_outputs/visuals/detection\_predictions.png`
* \[ ] `day1\_lab\_outputs/visuals/segmentation\_prediction.png`
* \[ ] Ô validation cuối notebook báo `PASS`
* \[ ] Không có họ tên, MSSV hoặc dữ liệu nhạy cảm trong báo cáo/output
* \[ ] Đã nộp **link repository** trên LMS

