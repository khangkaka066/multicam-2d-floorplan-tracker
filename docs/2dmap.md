# NGHIÊN CỨU & THIẾT KẾ HỆ THỐNG TẠO BẢN ĐỒ 2D VÀ GIÁM SÁT NGƯỜI ĐA CAMERA (MULTI-CAMERA 2D MAPPING & TRACKING)

Tài liệu này tổng hợp toàn bộ phương pháp luận, danh sách các mô hình AI sẵn có (Pre-trained Models) chuyển đổi từ ảnh sang bản đồ 2D, và quy trình suy luận từng bước (Step-by-step Reasoning) để xây dựng hệ thống giám sát vị trí con người trên bản đồ 2D từ tập dữ liệu nhiều camera (trong cùng một phòng và trải rộng qua nhiều phòng).

---

## PHẦN 1: TƯ DUY HỆ THỐNG & PHÂN TÁCH BÀI TOÁN

Để hệ thống đạt độ chính xác cao nhất và chạy được thời gian thực (Real-time), bài toán bắt buộc phải được tách thành **2 pha độc lập**:

1. **Pha Offline (Khởi tạo Bản đồ 2D & Hiệu chuẩn Không gian - Chạy 1 lần khi lắp đặt hoặc cập nhật góc máy):**
   * **Đầu vào:** Tập ảnh/video từ các camera cố định trong các phòng + (Tùy chọn) Video quay quét nối giữa các phòng hoặc ảnh bản vẽ mặt bằng.
   * **Xử lý:** Tái tạo cấu trúc 3D của toàn bộ không gian $\rightarrow$ Xác định vị trí và góc xoay của từng camera $\left(\mathbf{R}_i, \mathbf{t}_i\right)$ trong một hệ tọa độ toàn cục duy nhất $\rightarrow$ Chiếu vuông góc xuống mặt sàn để vẽ bản đồ 2D (Floorplan) và trích xuất ma trận ánh xạ (Homography $\mathbf{H}_i$).
   * **Đầu ra:** File bản đồ 2D tổng thể (ảnh raster hoặc đa giác vector) + File cấu hình ma trận chiếu của từng camera.

2. **Pha Online (Giám sát & Chiếu vị trí người lên Bản đồ 2D - Chạy Real-time $25\text{--}30\text{ FPS}$):**
   * **Đầu vào:** Luồng video trực tiếp từ $N$ camera.
   * **Xử lý:** Phát hiện người và điểm tiếp đất (cổ chân) $\rightarrow$ Ánh xạ tọa độ điểm ảnh $(u, v)$ sang tọa độ mặt sàn 2D $(X, Y)$ bằng ma trận đã lưu $\rightarrow$ Hợp nhất vị trí của cùng một người khi xuất hiện trên nhiều camera (Multi-Camera Fusion & Re-ID).
   * **Đầu ra:** Bản đồ 2D hiển thị các chấm định danh (ID) di chuyển mượt mà theo thời gian thực.

---

## PHẦN 2: CÁC MÔ HÌNH AI CÓ SẴN (PRE-TRAINED MODELS) NHẬN ẢNH VÀ VẼ SƠ ĐỒ 2D

Không có một mô hình đơn lẻ nào nhận các ảnh rời rạc từ nhiều phòng khác nhau mà tự vẽ được ngay sơ đồ 2D chính xác nếu không qua bước hiểu hình học 3D. Do đó, các mô hình có sẵn hiện nay được chia thành **4 nhóm chuyên biệt** để bạn ghép nối tùy theo dữ liệu đầu vào:

### Nhóm 1: Mô hình Nền tảng Hình học 3D (3D Geometry Foundation Models)
*Đầu vào: Nhiều ảnh RGB thông thường (không cần biết thông số camera) $\rightarrow$ Đầu ra: Đám mây điểm 3D (Point Cloud) + Tư thế Camera (Camera Poses) $\rightarrow$ Cắt lát ngang để lấy Bản đồ 2D.*

Đây là hướng tiếp cận hiện đại và hiệu quả nhất cho camera giám sát (CCTV) thông thường:

1. **VGGT (Visual Geometry Grounded Transformer - Meta AI, CVPR 2025 Best Paper)**
   * **Cơ chế:** Mô hình Transformer truyền thẳng (feed-forward) nhận từ $1$ đến hàng trăm ảnh từ nhiều góc nhìn, dự đoán trực tiếp thông số nội tại (intrinsics $\mathbf{K}$), ngoại tại (extrinsics $\mathbf{R}, \mathbf{t}$), bản đồ độ sâu (depth maps) và đám mây điểm 3D chỉ trong $< 1\text{ giây}$.
   * **Ứng dụng tạo Map 2D:** Đưa ảnh chụp từ các camera trong phòng vào VGGT, lấy đám mây điểm 3D, sau đó chiếu vuông góc (Orthographic Projection) theo trục trọng lực xuống mặt sàn để thu được bản đồ 2D và vị trí chính xác của từng camera trên bản đồ đó.
   * **Mã nguồn & Trọng số:** `facebookresearch/vggt` (GitHub / Hugging Face).

2. **Họ mô hình DUSt3R / MASt3R / MUSt3R (NAVER Labs Europe, 2024 - 2025)**
   * **DUSt3R (CVPR 2024):** Tái tạo bề mặt 3D từ các cặp ảnh bất kỳ mà không cần hiệu chuẩn camera trước (uncalibrated). Rất mạnh khi góc nhìn giữa hai camera trong phòng chênh lệch lớn.
   * **MASt3R (ECCV 2024) & MASt3R-SfM:** Bổ sung nhánh khớp đặc trưng (feature matching) chính xác theo hệ mét (metric scale), giúp ghép hàng chục góc camera có độ chồng lấp thấp thành một không gian 3D thống nhất.
   * **MUSt3R (CVPR 2025):** Mở rộng khả năng xử lý đồng thời nhiều góc nhìn (multi-view) tốc độ cao.
   * **Mã nguồn & Trọng số:** `naver/dust3r`, `naver/mast3r`, `naver/must3r`.

3. **Fast3R & CUT3R (2025)**
   * **Cơ chế:** Xử lý hàng trăm khung hình cùng lúc trong một lần suy luận mà không cần tối ưu hóa lặp (Bundle Adjustment) tốn thời gian như COLMAP truyền thống. Rất phù hợp khi dùng 1 video quay đi dạo qua nhiều phòng để tạo bản đồ tổng.

---

### Nhóm 2: Mô hình Chuyển đổi Đám mây điểm / Ảnh Đa góc nhìn thành Bản đồ 2D Vector (Vectorized Floorplan)
*Đầu vào: Đám mây điểm 3D (từ VGGT/MASt3R) hoặc chuỗi ảnh RGB $\rightarrow$ Đầu ra: Bản vẽ 2D sạch đẹp gồm các đa giác tường, cửa, phòng (Vector CAD/SVG).*

Nếu bạn không muốn bản đồ 2D chỉ là ảnh chụp từ trên xuống bị nhiễu bởi bàn ghế, các model này sẽ "vẽ lại" thành sơ đồ kiến trúc 2D phẳng:

1. **RoomFormer (CVPR 2023) & PolyRoom (ECCV 2024)**
   * **Cơ chế:** Nhận vào một bản đồ mật độ 2D (2D density map - được tạo bằng cách ép phẳng đám mây điểm 3D theo chiều thẳng đứng). Sử dụng kiến trúc *Two-Level Query Transformer* để dự đoán song song tọa độ các góc tường $\mathbf{p}_n^m \in \mathbb{R}^2$ và nối chúng thành các đa giác phòng khép kín, tự động loại bỏ nhiễu nội thất.
   * **Ưu điểm:** Thời gian suy luận cực nhanh ($\approx 0.01\text{ s}$/cảnh), tạo ra bản đồ 2D chuẩn xác cho bài toán nhiều phòng (multi-room).
   * **Mã nguồn:** `ywyue/RoomFormer` (GitHub).

2. **NeuralRecon / PlanarRecon**
   * **Cơ chế:** Nhận chuỗi video/ảnh đa góc nhìn và tái tạo trực tiếp các mặt phẳng 3D (tường, sàn) theo thời gian thực bằng mạng 3D Sparse Convolution và GRU. Khi lấy mặt cắt ngang ở độ cao $1.2\text{ m}$ so với mặt sàn, bạn thu được ngay sơ đồ tường 2D sắc nét.

---

### Nhóm 3: Mô hình Ước lượng Khung Phòng từ 1 Ảnh (Monocular / Panorama Room Layout Estimation)
*Đầu vào: 1 ảnh đơn lẻ (góc thường hoặc góc rộng $360^\circ$) $\rightarrow$ Đầu ra: Đường ranh giới Sàn - Tường - Trần $\rightarrow$ Bung ra sơ đồ 2D của phòng.*

Nếu có những phòng chỉ lắp duy nhất **1 camera** (không thể ghép cặp stereo), bạn sử dụng nhóm model này:

1. **HorizonNet / LGT-Net / DOPNet (Dành cho ảnh góc rộng / Fisheye / Panorama)**
   * **Cơ chế:** Dự đoán trực tiếp góc phương vị của các góc phòng và đường giao tuyến giữa tường với sàn nhà trên ảnh, sau đó giải hình học để xuất ra đa giác 2D của căn phòng.
   * **Mã nguồn:** `sunset1995/HorizonNet`, `zhigangjiang/LGT-Net`.

2. **Depth Anything V2 (Metric Depth) + RANSAC Plane Fitting (Dành cho ảnh camera góc hẹp thông thường)**
   * **Cơ chế:** Dùng bản pre-trained **Depth Anything V2 Metric Indoor** (`depth-anything/Depth-Anything-V2-Metric-Indoor-Large`) để suy ra độ sâu thực tế (tính bằng mét) cho từng pixel từ 1 ảnh RGB duy nhất. Sau đó dùng thuật toán RANSAC tách mặt phẳng sàn nhà (Ground Plane) và mặt phẳng tường để dựng thành bản đồ 2D cục bộ của góc phòng đó.

---

### Nhóm 4: Mô hình Số hóa Bản vẽ có sẵn (Nếu tòa nhà có ảnh chụp sơ đồ thoát hiểm/kiến trúc)
*Đầu vào: Ảnh chụp bản vẽ mặt bằng hoặc sơ đồ thoát hiểm dán trên tường $\rightarrow$ Đầu ra: Bản đồ 2D số hóa.*

* **CubiCasa5k / YOLOv8-Floorplan:** Nếu bạn có thể chụp lại tờ sơ đồ thoát hiểm của tầng nhà, các model này sẽ tách chính xác tường, cửa ra vào, cửa sổ và phòng để tạo ra bản đồ nền 2D chuẩn tỉ lệ $100\%$ trước khi gán camera vào.

---

### Bảng Tóm tắt Lựa chọn Mô hình Tạo Bản đồ 2D

| Kịch bản Dữ liệu Đầu vào | Tổ hợp Mô hình Khuyên dùng | Độ chính xác hình học | Độ khó triển khai |
| :--- | :--- | :--- | :--- |
| Nhiều camera cùng phòng (có góc nhìn giao nhau) | **VGGT** hoặc **MASt3R** $\rightarrow$ Chiếu 2D + **RoomFormer** | Rất cao | Trung bình |
| Nhiều camera ở nhiều phòng (kèm 1 video quay đi bộ nối giữa các phòng) | **VGGT / Fast3R** (xử lý chung video + ảnh CCTV) $\rightarrow$ **RoomFormer** | Rất cao (Đồng bộ toàn cục) | Trung bình - Khó |
| Phòng chỉ có 1 camera duy nhất (không có góc giao) | **Depth Anything V2 (Metric)** + RANSAC tách sàn | Trung bình (Bị khuất sau vật cản) | Dễ |
| Đã có sẵn ảnh bản vẽ mặt bằng / sơ đồ thoát hiểm | **CubiCasa5k** (tạo map) + **SuperPoint/LightGlue** (khớp camera vào map) | Tuyệt đối | Dễ nhất |

---

## PHẦN 3: SUY LUẬN TỪNG BƯỚC (STEP-BY-STEP REASONING) XÂY DỰNG HỆ THỐNG

Dưới đây là chuỗi suy luận kỹ thuật đi từ dữ liệu thô đến kết quả cuối cùng, giải quyết triệt để bài toán **nhiều camera trong cùng phòng** và **nhiều camera ở các phòng tách biệt**.

### Bước 1: Giải quyết bài toán "Cơ sở Tọa độ Chung" (Global Coordinate System)
* **Vấn đề:** Để vẽ được 1 bản đồ 2D duy nhất cho nhiều phòng và chiếu người từ mọi camera lên đó, tất cả các camera $C_1, C_2, \dots, C_K$ phải được quy về **cùng một hệ trục tọa độ thế giới** $\mathcal{W} = (X_w, Y_w, Z_w)$, trong đó mặt sàn của toàn bộ tầng nhà nằm trên mặt phẳng $Z_w = 0$.
* **Trường hợp 1 (Cùng phòng - Overlapping Views):** Các camera nhìn thấy chung các vật thể (bàn, ghế, góc tường). Đưa ảnh của các camera này vào **VGGT** hoặc **MASt3R**. Mô hình sẽ tự động tìm các điểm tương đồng, tái tạo đám mây điểm 3D của phòng và trả về vị trí tương đối giữa các camera.
* **Trường hợp 2 (Khác phòng - Non-overlapping Views):** Camera ở Phòng A và Camera ở Phòng B bị ngăn cách bởi tường, hoàn toàn không có điểm ảnh chung. Nếu chỉ đưa ảnh tĩnh của 2 phòng vào AI, AI sẽ tạo ra 2 bản đồ rời rạc không biết ghép vào nhau ở đâu.
* **Giải pháp chuẩn xác nhất cho Đa Phòng:**
  1. **Cách 1 (Khuyên dùng - Dùng "Video Cầu nối"):** Cầm một chiếc điện thoại quay 1 video liên tục đi bộ từ Phòng A, qua cửa, đi dọc hành lang sang Phòng B (chỉ cần làm 1 lần duy nhất lúc cài đặt hệ thống). Trích xuất các khung hình từ video này và gộp chung với các ảnh chụp từ camera cố định (CCTV) của Phòng A và Phòng B, sau đó đưa tất cả vào **MASt3R-SfM** hoặc **VGGT**. Các khung hình điện thoại đóng vai trò "cầu nối" liên kết không gian Phòng A và Phòng B vào chung một hệ tọa độ 3D duy nhất.
  2. **Cách 2 (Nếu chỉ có Dataset cố định không thể quay thêm video):** Tạo bản đồ 2D riêng cho từng phòng (Local 2D Maps). Sau đó sử dụng **ràng buộc kiến trúc và quỹ đạo người đi bộ (Human Trajectory Stitching)**: Khi một người đi ra khỏi cửa ở Phòng A tại thời điểm $t_1$ và xuất hiện ở cửa hành lang/Phòng B tại $t_2 = t_1 + \Delta t$, vị trí cửa và vận tốc di chuyển sẽ giúp tính được phép biến đổi cứng 2D (2D Rigid Transformation gồm góc xoay $\theta$ và vector tịnh tiến $\mathbf{T} = [t_x, t_y]^T$) để ghép khít cạnh tường của 2 phòng lại với nhau.

---

### Bước 2: Căn chỉnh Mặt phẳng Sàn (Ground Plane Alignment) & Xuất Bản đồ 2D
Sau Bước 1, ta có đám mây điểm 3D $\mathcal{P} = \{\mathbf{X}_j \in \mathbb{R}^3\}$ và thông số của mọi camera. Tuy nhiên, hệ trục tọa độ do AI sinh ra thường bị nghiêng so với mặt đất thực tế.

1. **Tách mặt sàn bằng Semantic Segmentation + RANSAC:**
   * Dùng mô hình phân vùng ngữ nghĩa (như **Mask2Former** hoặc **OneFormer** pre-trained trên bộ dữ liệu ADE20K) để lọc ra các pixel thuộc lớp `floor` (sàn nhà) và `wall` (tường) trên ảnh của các camera.
   * Ánh xạ các pixel `floor` này lên đám mây điểm 3D và dùng thuật toán **RANSAC** để khớp một mặt phẳng toán học:
     $$aX + bY + cZ + d = 0$$
2. **Xoay hệ trục về chuẩn mặt đất ($Z_w = 0$):**
   * Tính ma trận xoay $\mathbf{R}_{\text{align}}$ để đưa vector pháp tuyến của sàn nhà $\mathbf{n} = [a, b, c]^T$ trùng khớp với trục thẳng đứng $\mathbf{z} = [0, 0, 1]^T$, đồng thời dịch chuyển mặt sàn về đúng cao độ $Z_w = 0$.
   * Cập nhật lại toàn bộ tọa độ đám mây điểm $\mathcal{P}$ và ma trận ngoại tại của từng camera $i$:
     $$\mathbf{E}_i^{\text{new}} = \begin{bmatrix} \mathbf{R}_i^{\text{new}} & \mathbf{t}_i^{\text{new}} \\ \mathbf{0}^T & 1 \end{bmatrix}$$
3. **Vẽ Bản đồ 2D (2D Floorplan Generation):**
   * **Cách trực tiếp (Occupancy Grid Map):** Loại bỏ các điểm thuộc trần nhà (ví dụ lọc bỏ các điểm có $Z_w > 2.2\text{ m}$) và mặt sàn ($Z_w < 0.1\text{ m}$). Chiếu toàn bộ các điểm còn lại xuống mặt phẳng $OXY$ bằng cách bỏ qua tọa độ $Z_w$. Chia mặt phẳng $OXY$ thành lưới ô vuông (ví dụ mỗi pixel trên bản đồ 2D tương ứng với $2\text{ cm} \times 2\text{ cm}$ ngoài đời thực). Những vị trí có mật độ điểm thuộc lớp `wall` cao sẽ tạo thành đường viền tường rõ nét.
   * **Cách nâng cao (Vector hóa bằng AI):** Đưa bản đồ mật độ vừa tạo qua mô hình **RoomFormer** để xuất ra bản vẽ 2D sạch đẹp với các đoạn thẳng tường, góc phòng và cửa ra vào.

---

### Bước 3: Tính toán Phép chiếu từ Camera xuống Bản đồ 2D (Camera-to-Map Projection)
Để chuyển đổi một vị trí trên ảnh camera $i$ sang bản đồ 2D nhanh nhất ở pha Online, ta không cần chạy lại mô hình 3D mà rút gọn thành phép toán ma trận:

Với mọi điểm nằm trên mặt sàn thực tế ($Z_w = 0$), mối quan hệ giữa tọa độ điểm ảnh $\mathbf{p} = [u, v, 1]^T$ trên camera $i$ và tọa độ trên mặt phẳng sàn $\mathbf{P}_{\text{map}} = [X_w, Y_w, 1]^T$ được biểu diễn hoàn toàn chính xác thông qua một **Ma trận Homography** $\mathbf{H}_i \in \mathbb{R}^{3 \times 3}$:

$$\lambda \begin{bmatrix} u \\ v \\ 1 \end{bmatrix} = \mathbf{K}_i \begin{bmatrix} \mathbf{r}_{i,1} & \mathbf{r}_{i,2} & \mathbf{r}_{i,3} & \mathbf{t}_i \end{bmatrix} \begin{bmatrix} X_w \\ Y_w \\ 0 \\ 1 \end{bmatrix} = \underbrace{\mathbf{K}_i \begin{bmatrix} \mathbf{r}_{i,1} & \mathbf{r}_{i,2} & \mathbf{t}_i \end{bmatrix}}_{\mathbf{H}_i^{-1}} \begin{bmatrix} X_w \\ Y_w \\ 1 \end{bmatrix}$$

Trong đó:
* $\mathbf{K}_i$ là ma trận nội tại $3 \times 3$ của camera $i$ (do VGGT/MASt3R suy ra).
* $\mathbf{r}_{i,1}, \mathbf{r}_{i,2}$ là cột thứ nhất và cột thứ hai của ma trận xoay $\mathbf{R}_i^{\text{new}}$.
* $\mathbf{t}_i$ là vector tịnh tiến của camera $i$.

Như vậy, để chiếu một điểm từ ảnh camera lên bản đồ 2D, ta chỉ cần nhân với ma trận nghịch đảo $\mathbf{H}_i = \left(\mathbf{K}_i [\mathbf{r}_{i,1}, \mathbf{r}_{i,2}, \mathbf{t}_i]\right)^{-1}$:

$$\begin{bmatrix} X'_w \\ Y'_w \\ w \end{bmatrix} = \mathbf{H}_i \begin{bmatrix} u \\ v \\ 1 \end{bmatrix} \implies X_w = \frac{X'_w}{w}, \quad Y_w = \frac{Y'_w}{w}$$

*(Lưu ý: Nếu ống kính camera là loại góc siêu rộng bị méo hình nặng, cần chạy bước khử méo `cv2.undistortPoints` trước khi nhân với $\mathbf{H}_i$).*

---

### Bước 4: Phát hiện Người & Xác định "Điểm Tiếp Đất" Chính xác (Ground Contact Point)
Đây là lỗi phổ biến nhất khiến việc chiếu người lên bản đồ 2D bị sai lệch:
* **Sai lầm thường gặp:** Lấy trung điểm cạnh đáy của khung hình chữ nhật (Bounding Box bottom-center) làm vị trí đứng của người. Khi một người đứng sau cái bàn hoặc ghế sofa (bị che khuất nửa chân dưới), cạnh đáy của Bounding Box sẽ nằm lơ lửng ở ngang bụng. Khi nhân với ma trận Homography $\mathbf{H}_i$ (vốn giả định điểm đó nằm trên mặt đất $Z_w = 0$), vị trí người trên bản đồ 2D sẽ bị bắn văng ra xa hàng mét về phía sau!
* **Giải pháp chuẩn (Sử dụng Pose Estimation + Khung xương 2D):**
  1. Sử dụng mô hình **YOLOv11-Pose** hoặc **RTMPose** (chạy cực nhanh, $> 100\text{ FPS}$ trên GPU).
  2. Mô hình vừa trả về Bounding Box, vừa trả về $17$ điểm khớp xương (Keypoints), kèm độ tự tin $c_k \in [0, 1]$.
  3. **Quy tắc tính điểm tiếp đất $\mathbf{p}_{\text{ground}} = (u_g, v_g)$:**
     * **Trường hợp 1 (Nhìn thấy rõ chân):** Nếu độ tự tin của khớp mắt cá chân trái ($k = 15$) hoặc phải ($k = 16$) lớn hơn ngưỡng $\tau = 0.5$, lấy trung điểm của hai mắt cá chân cộng thêm một độ lệch nhỏ $\Delta v$ (khoảng $2\%$ chiều cao box) để chạm hẳn xuống đế giày:
       $$u_g = \frac{u_{\text{ankle\_L}} + u_{\text{ankle\_R}}}{2}, \quad v_g = \frac{v_{\text{ankle\_L}} + v_{\text{ankle\_R}}}{2} + \Delta v$$
     * **Trường hợp 2 (Bị bàn ghế che khuất chân - Occlusion):** Nếu mắt cá chân và đầu gối có $c_k < 0.5$ nhưng vẫn thấy rõ vai ($\mathbf{p}_{\text{shoulder}}$) và hông ($\mathbf{p}_{\text{hip}}$), ta ước lượng chiều dài chân bị khuất dựa trên tỉ lệ nhân trắc học cơ thể người (chiều dài từ hông xuống bàn chân xấp xỉ $1.2 \times$ khoảng cách từ vai đến hông trên trục dọc của ảnh):
       $$u_g = u_{\text{hip}}, \quad v_g = v_{\text{hip}} + 1.2 \times \left(v_{\text{hip}} - v_{\text{shoulder}}\right)$$
     * Nhờ công thức bù trừ này, dù người đứng sau bàn làm việc hay quầy lễ tân, điểm $(u_g, v_g)$ vẫn phản ánh đúng vị trí bàn chân chạm sàn dưới gầm bàn, giúp tọa độ chiếu lên Map 2D không bị nhảy loạn xạ.

---

### Bước 5: Hợp nhất Đa Camera & Định danh Liên Phòng (Multi-Camera Fusion & Tracking)
Khi nhiều camera trong cùng một phòng cùng nhìn thấy 1 người, trên bản đồ 2D sẽ xuất hiện nhiều điểm chiếu nằm sát nhau. Khi người đó đi sang phòng khác, hệ thống cần giữ nguyên ID.

1. **Trích xuất Đặc trưng Ngoại hình (Re-Identification - ReID):**
   * Với mỗi Bounding Box người phát hiện được, cắt ảnh người đó (crop) và đưa qua một mạng nhẹ như **OSNet (`osnet_ain_x1_0`)** hoặc **CLIP-ReID** để lấy vector đặc trưng ngoại hình $\mathbf{f} \in \mathbb{R}^{512}$ (đã chuẩn hóa $\|\mathbf{f}\|_2 = 1$).
2. **Hợp nhất trong cùng 1 phòng (Spatial + Appearance Clustering trên mặt phẳng 2D):**
   * Tại mỗi khung hình $t$, giả sử camera $1$ chiếu người $A$ xuống tọa độ $\mathbf{x}_1 = (X_1, Y_1)$ kèm vector $\mathbf{f}_1$, và camera $2$ chiếu người $B$ xuống $\mathbf{x}_2 = (X_2, Y_2)$ kèm vector $\mathbf{f}_2$.
   * Tính ma trận chi phí kết hợp giữa **Khoảng cách hình học 2D (Euclidean Distance)** trên mặt sàn và **Khoảng cách ngoại hình (Cosine Distance)**:
     $$C(1, 2) = \alpha \cdot \frac{\|\mathbf{x}_1 - \mathbf{x}_2\|_2}{D_{\text{max}}} + (1 - \alpha) \cdot \left(1 - \mathbf{f}_1^T \mathbf{f}_2\right)$$
     *(Với $D_{\text{max}} \approx 0.8\text{ m}$ là sai số cho phép trên sàn nhà, và $\alpha = 0.6$)*.
   * Nếu $C(1, 2)$ nhỏ hơn ngưỡng, hai điểm này được gom làm một. Tọa độ cuối cùng của người đó trên bản đồ 2D được tính bằng trung bình có trọng số (camera nào ở gần người hơn và thấy rõ mắt cá chân hơn sẽ có trọng số cao hơn):
     $$\mathbf{x}_{\text{fused}} = \frac{w_1 \mathbf{x}_1 + w_2 \mathbf{x}_2}{w_1 + w_2}$$
3. **Bám vết mượt mà trên Map 2D (2D BEV Kalman Filter) & Chuyển tiếp Liên phòng:**
   * Thay vì chạy thuật toán Tracking trên từng khung hình 2D của từng camera riêng lẻ, hãy đặt **Bộ lọc Kalman (Kalman Filter)** trực tiếp trên mặt phẳng Bản đồ 2D với vector trạng thái:
     $$\mathbf{s}_t = [X_w, Y_w, v_X, v_Y]^T$$
   * Khi một người đi từ Phòng A khuất khỏi camera và đi vào Phòng B, bộ lọc Kalman vẫn dự đoán quỹ đạo người đó tiến về phía cửa, kết hợp với việc so khớp vector Re-ID $\mathbf{f}$ với ngân hàng đặc trưng (Feature Gallery) để gắn lại đúng ID cũ ngay khi người đó vừa bước vào vùng nhìn của camera Phòng B.

---

## PHẦN 4: KIẾN TRÚC PIPELINE TRIỂN KHAI THỰC TẾ (IMPLEMENTATION BLUEPRINT)

Dưới đây là sơ đồ luồng dữ liệu và khung mã nguồn (Python) kết nối toàn bộ hệ thống.

### 1. Sơ đồ Kiến trúc Tổng thể

```
[PHA OFFLINE: TẠO BẢN ĐỒ 2D & MA TRẬN CHIẾU]
Ảnh các Camera + Video đi bộ nối phòng
       │
       ▼
  [Model VGGT / MASt3R-SfM] ──► Đám mây điểm 3D + Camera Poses (K, R, t)
       │
       ▼
  [Mask2Former + RANSAC]    ──► Căn chỉnh mặt sàn về Z_w = 0
       │
       ├──► Nhánh 1: Chiếu vuông góc 3D->2D ──► [Model RoomFormer] ──► Bản đồ 2D (floorplan.png)
       │
       └──► Nhánh 2: Rút gọn ma trận ─────────► Lưu Homography H_i cho từng Camera (homographies.json)

===================================================================================================

[PHA ONLINE: GIÁM SÁT & CHIẾU NGƯỜI REAL-TIME]
Luồng Video Camera 1..N
       │
       ▼
  [YOLOv11-Pose] ──► Phát hiện người + Nội suy điểm tiếp đất (u_g, v_g)
       │
       ├──► [OSNet Re-ID] ──────────────► Vector đặc trưng ngoại hình (f_i)
       │
       └──► [Nhân Ma trận H_i] ─────────► Tọa độ 2D trên sàn (X_i, Y_i)
                                                │
                                                ▼
                                   [Gom cụm Đa Camera + Kalman Filter 2D]
                                                │
                                                ▼
                                   Hiển thị vị trí & quỹ đạo lên Bản đồ 2D
```

### 2. Khung Mã Nguồn Cốt Lõi (Core Python Pipeline)

```python
import cv2
import numpy as np
from scipy.optimize import linear_sum_assignment

class MultiCam2DMapperAndTracker:
    def __init__(self, map_image_path, homographies_dict, map_scale_px_per_meter=50.0, map_origin_px=(500, 500)):
        """
        map_image_path: Đường dẫn tới ảnh Bản đồ 2D đã tạo ở Pha Offline (từ VGGT + RoomFormer)
        homographies_dict: Dict chứa ma trận 3x3 H_i ánh xạ từ pixel ảnh camera i sang mét thực tế (X_w, Y_w)
        map_scale_px_per_meter: Tỉ lệ quy đổi từ mét ngoài đời sang pixel trên ảnh bản đồ 2D (VD: 1m = 50px)
        map_origin_px: Tọa độ pixel gốc (0, 0) trên ảnh bản đồ 2D
        """
        self.base_map = cv2.imread(map_image_path)
        self.homographies = {cam_id: np.array(H, dtype=np.float64) for cam_id, H in homographies_dict.items()}
        self.scale = map_scale_px_per_meter
        self.origin = map_origin_px

    def estimate_ground_point(self, bbox, keypoints, kp_conf_thresh=0.5):
        """
        Tính toán điểm tiếp đất (u_g, v_g) chống sai số khi bị bàn ghế che khuất chân.
        keypoints: Mảng (17, 3) chuẩn COCO [u, v, conf] từ YOLOv11-Pose
        """
        # Index chuẩn COCO: 5,6: Vai trái/phải | 11,12: Hông trái/phải | 15,16: Mắt cá trái/phải
        l_ankle, r_ankle = keypoints[15], keypoints[16]
        l_hip, r_hip = keypoints[11], keypoints[12]
        l_shldr, r_shldr = keypoints[5], keypoints[6]

        box_h = bbox[3] - bbox[1]

        # 1. Trường hợp thấy rõ ít nhất 1 mắt cá chân
        valid_ankles = [kp[:2] for kp in [l_ankle, r_ankle] if kp[2] >= kp_conf_thresh]
        if len(valid_ankles) > 0:
            mean_ankle = np.mean(valid_ankles, axis=0)
            # Cộng thêm 2% chiều cao box để chạm sát mặt sàn (đế giày)
            return float(mean_ankle[0]), float(mean_ankle[1] + 0.02 * box_h), 1.0

        # 2. Trường hợp bị che khuất chân (chỉ thấy thân trên: Vai & Hông)
        valid_hips = [kp[:2] for kp in [l_hip, r_hip] if kp[2] >= kp_conf_thresh]
        valid_shldrs = [kp[:2] for kp in [l_shldr, r_shldr] if kp[2] >= kp_conf_thresh]
        if len(valid_hips) > 0 and len(valid_shldrs) > 0:
            mean_hip = np.mean(valid_hips, axis=0)
            mean_shldr = np.mean(valid_shldrs, axis=0)
            torso_len = max(10.0, mean_hip[1] - mean_shldr[1])
            # Nội suy vị trí bàn chân dựa trên tỉ lệ nhân trắc học (Chân ~ 1.2 * Thân trên)
            u_g = mean_hip[0]
            v_g = mean_hip[1] + 1.2 * torso_len
            return float(u_g), float(v_g), 0.6  # Độ tin cậy giảm xuống 0.6 do nội suy

        # 3. Fallback: Trung điểm cạnh đáy Bounding Box
        u_g = (bbox[0] + bbox[2]) / 2.0
        v_g = bbox[3]
        return float(u_g), float(v_g), 0.3

    def project_to_world_2d(self, cam_id, u_g, v_g):
        """
        Chiếu điểm (u_g, v_g) trên camera sang tọa độ mặt sàn thực tế (X_w, Y_w) tính bằng mét
        """
        H = self.homographies[cam_id]
        pt_img = np.array([u_g, v_g, 1.0], dtype=np.float64)
        pt_world = H @ pt_img
        pt_world /= (pt_world[2] + 1e-8)
        return float(pt_world[0]), float(pt_world[1])

    def world_to_map_pixel(self, X_w, Y_w):
        """
        Chuyển đổi tọa độ mét (X_w, Y_w) sang tọa độ pixel trên ảnh bản đồ 2D để vẽ
        """
        px = int(self.origin[0] + X_w * self.scale)
        py = int(self.origin[1] - Y_w * self.scale)  # Trục Y ảnh ngược chiều trục Y đề-các
        return px, py

    def fuse_multi_camera_detections(self, detections_per_cam, max_dist_m=0.8, alpha=0.6):
        """
        Hợp nhất các điểm chiếu của cùng 1 người từ nhiều camera.
        detections_per_cam: List các dict [{'cam_id', 'pos_2d': (X, Y), 'reid_feat': np.array, 'conf': float}]
        """
        if len(detections_per_cam) <= 1:
            return detections_per_cam

        clusters = []
        used = [False] * len(detections_per_cam)

        for i in range(len(detections_per_cam)):
            if used[i]:
                continue
            current_cluster = [detections_per_cam[i]]
            used[i] = True

            for j in range(i + 1, len(detections_per_cam)):
                if used[j] or detections_per_cam[j]['cam_id'] == detections_per_cam[i]['cam_id']:
                    continue

                # Tính khoảng cách không gian 2D trên mặt sàn
                pos_i = np.array(detections_per_cam[i]['pos_2d'])
                pos_j = np.array(detections_per_cam[j]['pos_2d'])
                spatial_dist = np.linalg.norm(pos_i - pos_j)

                if spatial_dist > max_dist_m:
                    continue

                # Tính khoảng cách ngoại hình Cosine (Re-ID)
                feat_i = detections_per_cam[i]['reid_feat']
                feat_j = detections_per_cam[j]['reid_feat']
                cosine_dist = 1.0 - float(np.dot(feat_i, feat_j))

                combined_cost = alpha * (spatial_dist / max_dist_m) + (1.0 - alpha) * cosine_dist
                if combined_cost < 0.45:
                    current_cluster.append(detections_per_cam[j])
                    used[j] = True

            # Tính trung bình có trọng số theo độ tin cậy của điểm tiếp đất
            weights = np.array([d['conf'] for d in current_cluster])
            positions = np.array([d['pos_2d'] for d in current_cluster])
            fused_pos = np.average(positions, axis=0, weights=weights)

            # Trung bình vector đặc trưng Re-ID và chuẩn hóa lại
            feats = np.array([d['reid_feat'] for d in current_cluster])
            fused_feat = np.average(feats, axis=0, weights=weights)
            fused_feat /= (np.linalg.norm(fused_feat) + 1e-8)

            clusters.append({
                'pos_2d': (float(fused_pos[0]), float(fused_pos[1])),
                'reid_feat': fused_feat,
                'cameras': [d['cam_id'] for d in current_cluster]
            })

        return clusters
```

---

## PHẦN 5: LỘ TRÌNH THỰC THI KHUYẾN NGHỊ (ACTION PLAN)

Để triển khai dự án đạt kết quả nhanh và ít rủi ro nhất, bạn nên thực hiện theo 3 giai đoạn kiểm thử tăng dần:

1. **Giai đoạn 1 (PoC trên 1 Phòng với $2\text{--}4$ Camera):**
   * Lấy $1$ khung hình tĩnh từ mỗi camera trong phòng, chạy qua **VGGT** (hoặc **MASt3R**) để lấy đám mây điểm 3D và ma trận camera.
   * Chiếu đám mây điểm xuống mặt phẳng $OXY$ để xuất ảnh bản đồ 2D của phòng và lưu các ma trận Homography $\mathbf{H}_i$.
   * Chạy **YOLOv11-Pose** trên video của các camera, lấy tọa độ mắt cá chân và nhân với $\mathbf{H}_i$ để kiểm tra xem các chấm từ các camera khác nhau của cùng 1 người đã chụm lại đúng 1 chỗ trên bản đồ 2D chưa.
2. **Giai đoạn 2 (Làm sạch Bản đồ 2D bằng AI):**
   * Đưa bản đồ mật độ điểm ở Giai đoạn 1 vào mô hình **RoomFormer** để chuyển đổi thành sơ đồ tường/cửa 2D dạng vector sắc nét.
3. **Giai đoạn 3 (Mở rộng Đa Phòng & Re-ID):**
   * Quay thêm 1 video đi bộ xuyên qua các hành lang và cửa phòng để ghép toàn bộ camera của các phòng vào chung hệ tọa độ với **VGGT / MASt3R-SfM**.
   * Tích hợp **OSNet Re-ID** và **Kalman Filter 2D** để giữ nguyên ID người dùng khi di chuyển từ phòng này sang phòng khác.
