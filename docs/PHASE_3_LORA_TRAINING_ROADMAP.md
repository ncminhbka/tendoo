# 🎯 KẾ HOẠCH & LỘ TRÌNH HUẤN LUYỆN LORA DiT 4B BASE (PHASE 3 MASTER ROADMAP - BẢN v6 TỐI HẬU)
## ⚔️ TÔN CHỈ CỐT LÕI: HUẤN LUYỆN NĂNG LỰC CẠNH TRANH (COMPETITIVE ATTENTION ROUTING), CHỐNG HỌC VẸT POSTER (ZERO POSTER-OVERFIT)

> ⚡ **TUYÊN NGÔN KỸ THUẬT BẤT BIẾN**:
> - **Mô hình Base 4B đã có sẵn $100\%$ năng lực đọc-viết tiếng Việt zero-shot** ở mọi độ dài text và mọi font khi ở trạng thái cô lập (đã kiểm chứng $100\%$ qua Probe Suite 1).
> - Do đó, **Tendoo AI TUYỆT ĐỐI KHÔNG huấn luyện LoRA để "học vẹt một công thức poster cố định"** (ví dụ: cấm kỵ việc mặc định $t=10$ luôn là chữ ngắn ở trên đỉnh, $t=20$ luôn là 2 dòng ở giữa, $t=30$ luôn là nút CTA ở đáy).
> - **MỤC TIÊU DUY NHẤT CỦA LORA**: Huấn luyện **Năng Lực Giải Quyết Cạnh Tranh Không Gian (Spatial Attention Routing & Disentanglement)**: Khi có nhiều slot cùng kích hoạt đồng thời ($t=10, 20, 30, 40, 50$), mạng Attention tự động phân luồng tách bạch từng thực thể, đưa đúng chữ vào đúng vị trí và quy mô mà Prompt yêu cầu, bất kể câu ngắn hay câu dài!

- **Mô hình Mục Tiêu Duy Nhất**: **`FLUX.2-klein-base-4B`** (Bản Base 50-step, True CFG = 4.0 - 4.5).
- **Hạ tầng Thực thi**: **2x NVIDIA A30 (24GB VRAM $\times 2 = 48$GB VRAM)**, Ampere Architecture, DDP (`accelerate`).
- **Tiến trình Curriculum Lũy Tiến Cạnh Tranh**: **$2\text{ Slots} \longrightarrow 3\text{ Slots} \longrightarrow 4-5\text{ Slots}$** (Độ khó cạnh tranh tăng dần, không lãng phí tài nguyên vào bài toán 1 text đơn lẻ).


---

## 📊 1. MA TRẬN PHÂN TÍCH KỸ THUẬT & ĐỊNH HÌNH THIẾT KẾ (TECHNICAL BASIS)

Dựa trên 61 chuỗi thực nghiệm đối chứng từ `exp01` đến `exp61`, bài test Fourier Phase Aliasing (`probe_rope_phase_aliasing.py`) cùng thực nghiệm đo đạc độ phân giải cấp ký tự (`probe_glyph_resolution_threshold.py`), toàn bộ kiến trúc huấn luyện được xây dựng trên 9 chân lý kỹ thuật đã được chứng minh $100\%$:

| Thành phần Kiến trúc | Phát hiện Thực nghiệm / Chân lý Toán học | Giải pháp Kỹ thuật trong Pipeline Huấn luyện |
| :--- | :--- | :--- |
| **1. Softmax Joint Attention** | Toàn bộ Key $K$ của Canvas, Sản phẩm ($4096$ tokens) và các Glyph bị gom chung vào 1 Softmax duy nhất $\rightarrow$ gây ra tranh chấp Softmax và lấn át khối nhỏ nếu thiếu phân luồng. | Áp dụng **Quy Luật Kích Thước Vừa Đủ (Optimal Tight-Crop Token Sizing)**: Tự động tính toán Box vừa vặn theo độ dài từ và số dòng ($80 - 640\text{ tokens}$), đảm bảo font size nằm trên *Sàn Thực Nghiệm DiT*, triệt tiêu Size Bias và tiết kiệm $>60\%$ sequence length. |
| **2. Target LoRA Layers** | FLUX.2 không có module Cross-Attention riêng; Canvas và Ref dùng chung `img_attn.qkv` (DoubleBlocks) và `linear1` (SingleBlocks). 80% độ sâu mô hình nằm ở 20 SingleBlocks. | Tiêm LoRA trực tiếp vào: `img_attn.qkv` + `txt_attn.qkv` (5 DoubleBlocks) và phần Q, K, V của `linear1` (20 SingleBlocks). Rank $r=32$, $\alpha=32$. |
| **3. Pretrained Discrete Offsets Supremacy** | Thực nghiệm phủ định giả thuyết góc quay số thực liên tục. Trọng số $W_Q, W_K$ của DiT đã được BFL hiệu chuẩn sâu trên các mốc số nguyên rời rạc $t \in \{10, 20, 30, 40, 50\}$. Mốc số thực lẻ ($44.0, 47.1...$) rơi vào Out-of-Distribution (OOD). | **Khóa cứng toàn bộ hệ thống trên các mốc số nguyên bội 10**: $t \in \{10.0, 20.0, 30.0, 40.0, 50.0\}$. Tuyệt đối không dùng các tọa độ float lẻ. |
| **4. Dynamic Context-Aware Slot Assignment** | Vị trí sản phẩm không cố định ở $t=50$, mà được phân bổ linh hoạt theo số lượng khối văn bản thực tế để luôn đạt độ sắc nét cao nhất. | • 1 SP (Đổi background): SP ở $t=10.0$<br>• 1 Text + SP: Text $t=10$, SP $t=20$<br>• 2 Text + SP: Text $t=10, 20$, SP $t=30$<br>• 3 Text + SP: Text $t=10, 20, 30$, SP $t=40$<br>• 4 Text + SP (Full-Power): Text $t=10, 20, 30, 40$, SP $t=50$. |
| **5. True CFG & Chống CFG Drift** | Klein 4B Base dùng True CFG (`use_guidance_embed = False`), nhánh Unconditional giữ nguyên Reference Tokens `img_cond_seq` và chỉ null hóa Text Prompt `ctx = ""`. | Áp dụng **Text Conditioning Dropout ($p=0.10$)**: Thay thế `txt` bằng embedding của chuỗi rỗng `""` với tỉ lệ $10\%$, giữ nguyên $100\%$ Reference Tokens để LoRA học đúng nhánh Unconditional. |
| **6. Contiguous Prefix Sequential Guarantee** | Người dùng không trực tiếp chọn mốc $t$. Backend tự động xếp slot tuần tự từ trước ra sau ($1\rightarrow 2\rightarrow 3\rightarrow 4\rightarrow 5$). Thực tế chỉ xuất hiện các dãy liên tục: $\{10, 20\}$, $\{10, 20, 30\}$, $\{10, 20, 30, 40\}$, $\{10, 20, 30, 40, 50\}$. | Huấn luyện **$100\%$ theo các dãy tiền tố liên tục chuẩn**, triệt tiêu nhiễu rác và tập trung toàn bộ năng lượng gradient vào đúng các cấu hình thực tế của Backend. |
| **7. Pure T2I Parallel Co-existence** | Nếu chỉ train với mỏ neo sản phẩm $4096$ tokens, mô hình bị "nghiện sản phẩm" và lúng túng khi sinh poster sự kiện/thơ ca không có ảnh sản phẩm. | Khóa cứng tỷ lệ **$55\%$ Product-Anchor + $45\%$ Pure T2I** ở CẢ 3 MILESTONES. |
| **8. Masked Product-Region Flow Loss** | Để đảm bảo ở trường hợp cực hạn (Full 5-Slot), chi tiết chữ in và màu sắc nắp sản phẩm ở $t=50.0$ không bị suy thoái $\ge 20\%$. | Áp dụng **Mặt nạ trọng số vùng sản phẩm ($\lambda_{\text{prod}} = 2.0$)** trong hàm Loss Flow Matching cho các pixel thuộc vật thể thật. |
| **9. Empirical DiT Denoise Resolution Floor** | VAE roundtrip chứng minh giữ nguyên $100\%$ nét ở $20\text{pt}$. Hiện tượng gai nét là do động lực học 50 bước ODE Denoise của DiT Base làm mượt chi tiết siêu nhỏ khi thiếu trọng số kích hoạt. | **Đo đạc sàn thực tế qua `probe_dit_font_resolution_floor.py`**: Quét các mốc $[20, 24, 28, 32, 36, 42, 48\text{pt}]$ trên 3 font đại diện để chốt số sàn thực nghiệm, không suy diễn võ đoán. |



---

## 🗂️ 2. THIẾT KẾ DỮ LIỆU & QUY TRÌNH CHẾ TẠO DATASET (DISTILLATION ENGINE)

### 2.1. Quy cách một Training Sample Chuẩn & Khớp Tuyệt Đối (Strict Target-Ref Alignment):

Mỗi mẫu huấn luyện được cấu trúc động và chuẩn hóa đa tầng:

#### 1. `Prompt_clean` — Cấu Trúc 3 Thành Phần Bắt Buộc (The 3-Component Grounding Rule):
* ⚠️ **SỰ THẬT KIẾN TRÚC SỐNG CÒN**: Trong hàm `encode_glyph_to_incontext_tokens`, toạ độ RoPE `h_coords` và `w_coords` của Reference Token **chỉ là toạ độ nội bộ trong chính bounding-box của glyph ($0 \rightarrow H_{\text{glyph}}, 0 \rightarrow W_{\text{glyph}}$), HOÀN TOÀN KHÔNG MANG TOẠ ĐỘ TUYỆT ĐỐI TRÊN CANVAS $1024 \times 1024$!**
  - Bản thân Glyph Token **không thể tự biết** nó phải xuất hiện ở góc trên, ở giữa hay ở đáy canvas.
  - Quyết định vị trí đặt chữ thuộc về **sự tương tác Attention giữa Canvas và Text Prompt qua Qwen3**.
  - Toàn bộ 21 prompt thực tế của tester (`prompt_test.txt`) đều chứa từ chỉ vị trí tường minh. Nếu training data chỉ dùng từ `"poster/banner"` chung chung, mô hình sẽ mất khả năng điều khiển vị trí theo ý muốn người dùng!

* ✅ **3 THÀNH PHẦN BẮT BUỘC TRONG PROMPT CHO MỖI SLOT**:
  1. **Chỉ Dẫn Vị Trí Tường Minh (Explicit Spatial Anchor)**: Bắt buộc mô tả vị trí thực tế trên layout: *"ở góc trên bên trái"*, *"ở giữa bên trái"*, *"ở trung tâm phía trên"*, *"ở góc dưới bên phải"*, *"nằm ở chân poster"*.
  2. **Quy Mô & Vai Trò (Scale & Role Descriptor)**: Giữ các từ định lượng quy mô thị giác: *"dòng chữ tiêu đề lớn nổi bật"*, *"dòng chữ phụ thanh mảnh tinh tế"*, *"huy hiệu ưu đãi nhỏ nhắn"*, *"đoạn trích dẫn nhận xét chi tiết"*. *(Đây là kênh tín hiệu ngữ nghĩa độc lập qua Qwen3, không mâu thuẫn với việc tối ưu token glyph)*.
  3. **Vật Lý, Chất Liệu & Quang Học (Material & Optics)**: *"dập nổi mạ vàng đồng"*, *"đèn neon phát quang xanh ngọc"*, *"khắc chìm trên gỗ"*, *"đổ bóng studio tương phản cao"*.
* ❌ **ĐIỀU CẤM KỴ**: Tuyệt đối **KHÔNG lặp lại nội dung chữ nguyên văn** để triệt tiêu $100\%$ xung đột biểu diễn (Representation Clash) từ Qwen3.

---

#### 2. Hàm Sizing Phổ Quát Dùng Chung Cho Mọi Slot (`compute_optimal_glyph_box`):
* Bỏ hoàn toàn việc gán cứng range token theo slot (`Ref_10` to, `Ref_30` nhỏ). **Mọi slot ($t=10, 20, 30, 40$) đều sử dụng chung 1 hàm sizing phổ quát duy nhất**, chỉ phụ thuộc vào:
  1. **Độ dài ký tự & số dòng thực tế** của mẫu đó.
  2. **Ngưỡng sàn vật lý riêng của từng Font (Per-Font Minimum Floor - KHÓA CỨNG CHÍNH THỨC)**:
     - **`BeVietnamPro-Black`**: **Floor = $32\text{pt}$** (Đã kiểm chứng đối chứng 2x A30 trên cả văn bản ngắn và bài thơ 4 dòng, bảo toàn nét tuyệt đối).
     - **Toàn bộ 15 font còn lại** (`anton`, `gotham`, `lolapeluza`, `gretoon`, `playfair`, `oswald`, `harabaras`, `dancing`, `pacifico`, `sedgwick`, `blowbrush`, `clementine`, `cookies`, `grocery`, `holidays`): **Floor = $36\text{pt}$** (Khóa cứng bất biến toàn bộ Display, Serif, Brush và Script, loại bỏ ước lượng phỏng đoán).

* **Kích thước Token thực tế sau Tight-Crop**:
  - *1 dòng ngắn ($1 - 3$ từ)* (Badge, CTA, Brand Name): **$80 - 140\text{ tokens}$**.
  - *1 dòng vừa ($4 - 6$ từ)* (Slogan 1 dòng, Tiêu đề ngắn): **$130 - 200\text{ tokens}$**.
  - *2 dòng ($6 - 10$ từ)* (Tiêu đề 2 dòng, Subhead): **$220 - 320\text{ tokens}$**.
  - *Đoạn dài ($3 - 4$ dòng / $15 - 25$ từ)* (Quote feedback, bài thơ): **$380 - 640\text{ tokens}$**.

* 💡 **Ý nghĩa Toán Học**:
  - **Tín hiệu phân luồng danh tính (Slot Identity)** đến **DUY NHẤT từ RoPE Time Offset ($t=10, 20, 30, 40$)**, đúng theo thiết kế ban đầu.
  - **Tín hiệu vị trí và kích thước thị giác** đến từ **Prompt Qwen3**.
  - **Tín hiệu hình học và chính tả** đến từ **Glyph VAE tối ưu**.
  - Ba trục tín hiệu hoàn toàn độc lập, trực giao và không gây nhiễu lẫn nhau!




---

### 2.2. Ma Trận Phân Bổ Font Trực Giao & Chống Liên Kết Giả (Orthogonal Font-Domain Decoupling):

> ⚠️ **BÀI HỌC CỐT TỬ**: Nếu cố định cứng $1:1$ giữa Ngành hàng và Font chữ (Thời trang luôn đi với `Playfair`, F&B luôn đi với `Sedgwick`, Công nghệ luôn đi với `Anton`), mô hình DiT sẽ bị **học vẹt mối liên kết giả (Spurious Correlation)**: Nó tưởng rằng *"chữ serif mạ vàng $\Longleftrightarrow$ bối cảnh thời trang"* thay vì học **năng lực đọc-viết Glyph tổng quát**.
> 
> Trong khi đó, **mô hình Base vốn đã có sẵn năng lực tái tạo font zero-shot 100%** (đã chứng minh qua Probe Suite 1). Nhiệm vụ của LoRA là **phân luồng không gian đa slot**, tuyệt đối không phải dạy lại font chữ.

Do đó, toàn bộ $2,500$ mẫu huấn luyện áp dụng cơ chế **Trực Giao Hóa Hoàn Toàn (Zero Mutual Information $I(\text{Font}; \text{Domain}) = 0$)**: Bất kỳ font chữ nào cũng có xác suất xuất hiện bình đẳng ở mọi ngành hàng. Hệ thống huy động trọn vẹn **Pool 16 Font Unicode Tiếng Việt** sẵn có trong kho:

| Nhóm Hình Thái Font (Archetypes) | Tỷ Trọng | Danh Sách Font Thực Tế trong `fonts/` | Phân Bổ Đa Ngành Hàng (Cross-Domain Presence) |
| :--- | :---: | :--- | :--- |
| **1. Clean Sans-Serif**<br>*(Hiện đại, đa dụng, dễ đọc)* | **30%** | `BeVietnamPro-Black`, `SVN-Gotham Ultra`, `SVN-Harabaras`, `Oswald` | Xuất hiện ở: Cả Công nghệ, Thời trang tối giản, FMCG, Bài post feedback Gym/Sofa. |
| **2. Editorial Serif**<br>*(Sang trọng, thanh lịch, cổ điển)* | **20%** | `PlayfairDisplay` | Xuất hiện ở: Cả Thời trang Haute-Couture, Menu Cà phê Vintage, Đồng hồ Luxury, Glamping. |
| **3. Bold Display / Heavy**<br>*(Khỏe khoắn, dập nổi, giật gân)* | **20%** | `Anton-Regular`, `SVN-Lolapeluza Black`, `SVN-Gretoon` | Xuất hiện ở: Cả Tech 5G, Flash Sale Siêu thị, Poster Thể thao, Tiêu đề Tuyển dụng. |
| **4. Script / Calligraphy**<br>*(Mềm mại, cảm xúc, uyển chuyển)* | **15%** | `DancingScript`, `Pacifico`, `SVN-Clementine` | Xuất hiện ở: Cả Spa/Mỹ phẩm, Tiệm Bánh/Trà, Studio Cưới, Lời tri ân khách hàng. |
| **5. Brush / Rounded / Playful**<br>*(Trẻ trung, đường phố, mùa hè)* | **15%** | `SedgwickAveDisplay`, `SVN-Blow Brush`, `SVN-Cookies`, `SVN-Grocery Rounded`, `SVN-Holidays` | Xuất hiện ở: Cả Quán Cafe Street-style, Banner Khai Trương, Đồ ăn vặt, Đồ chơi trẻ em. |

> 🎯 **Quy tắc phân bổ Dataset**: Mỗi font trong pool 16 font chỉ cần xuất hiện từ $50 - 150$ mẫu ngẫu nhiên trải đều qua các domain là đủ để triệt tiêu vĩnh viễn liên kết giả, giúp LoRA tập trung $100\%$ gradient vào bài toán phân luồng Attention đa slot!


---

### 2.3. Ma Trận Đa Dạng Hóa Topology Bố Cục (Spatial Layout Topology Distribution):

Để đảm bảo LoRA không bị "học vẹt" một công thức poster xếp chồng dọc (Top-Mid-Bottom) duy nhất, mà thực sự làm chủ **Năng Lực Phân Luồng Chú Ý Đa Slot Tổng Quát (Universal N-Slot Spatial Routing)**, toàn bộ $2,500$ mẫu huấn luyện được phân bổ nghiêm ngặt theo **4 Dạng Topology Bố Cục**:

| Dạng Topology Bố Cục | Tỷ Trọng | Quy Mô | Cấu Trúc Phân Bổ Không Gian | Nguồn Mẫu & Use-case Nghiệp Vụ Viettel |
| :--- | :---: | :---: | :--- | :--- |
| **1. Poster Dọc Cổ Điển**<br>*(Classic Vertical Stack)* | **35%** | $875$ mẫu | Xếp chồng tuần tự theo trục dọc: Đỉnh (Header lớn) $\rightarrow$ Giữa (Sản phẩm / Slogan) $\rightarrow$ Đáy (CTA Badge / Giá tiền). | Standee hội nghị, Poster sự kiện, Banner F&B trà sữa, Flash Sale siêu thị. |
| **2. Phân Tách Trái - Phải**<br>*(Horizontal Split / Feedback Card)* | **25%** | $625$ mẫu | Chia đôi bố cục theo trục ngang: Nửa trái (Ảnh Before/After, Sản phẩm) $\longleftrightarrow$ Nửa phải (Tiêu đề + Đánh giá 5 sao + Đoạn quote feedback + Badge giảm giá). | Trích xuất từ 11 prompt tester (`prompt_test.txt`): Khách hàng Gym/PT, Spa thú cưng, Khóa học tiếng Anh, Sofa thông minh, Dọn nhà, Nha khoa. |
| **3. Lưới Đều / Menu Danh Mục**<br>*(Equal Grid / Feature Matrix)* | **20%** | $500$ mẫu | Các khối chữ có cỡ tương đương nhau (không có "Hero Title" áp đảo), phân bổ dạng ma trận $2 \times 2$ hoặc danh mục menu $1$ cột đều đặn. | Menu quán ăn/cà phê, Bảng so sánh gói cước Viettel 5G, Bảng thông số kỹ thuật smartwatch (chống nước, đo nhịp tim, pin). |
| **4. Tự Do Bất Đối Xứng & Chữ Nổi**<br>*(Asymmetric Kinetic / Floating Overlap)* | **20%** | $500$ mẫu | Bố cục tự do phi đối xứng: Chữ đặt lệch góc (Top-Left + Mid-Left như Prompt 1), chữ nổi trên khoảng trống âm (Negative Space), badge chéo góc. | Banner tin tuyển dụng (Recruitment), Banner khai trương (Grand Opening), Quảng cáo thời trang dạo phố, Poster nghệ thuật hiện đại. |

> 💡 **Ý nghĩa Sống Còn**: Không làm tăng kích thước dataset (vẫn $2,500$ mẫu), nhưng buộc DiT phải hiểu: *"Tọa độ $t=10, 20, 30, 40$ là các kênh định tuyến độc lập, có thể xuất hiện ở BẤT KỲ ĐÂU trên ảnh theo yêu cầu của Prompt, chứ không nhất thiết $t=10$ là phải nằm ở trên đỉnh!"*


---

### 2.4. Ma Trận Phân Tầng Độ Dài Văn Bản Có Kiểm Soát (Stratified Text-Length Invariance Matrix):

> ⚔️ **VÌ SAO PHẢI "MỞ KHÓA" ĐỘ DÀI CÓ KIỂM SOÁT (STRATIFIED) NGAY TỪ ĐẦU?**
> - **Bằng chứng từ Probe Suite 1**: Một khối chữ đứng riêng lẻ ở bất kỳ $t$ nào ($10, 20, 30, 40$) đều render chính xác $100\%$ bất kể câu ngắn hay câu dài. Năng lực hiểu độ dài đã có sẵn trong mô hình Base!
> - LoRA chỉ cần học: **Giải quyết cạnh tranh khi nhiều slot cùng hoạt động (Attention Disentanglement)**, hoàn toàn không phụ thuộc vào độ dài text.
> - **Nguy cơ nếu overfit**: Nếu cố định $t=10$ luôn là $3-4$ từ ngắn và $t=20$ luôn là $2$ dòng dài, chúng ta sẽ tiêm vào mô hình một **liên kết giả chết người thứ hai (`slot \Longleftrightarrow \text{length}`)**, biến mô hình thành cỗ máy học vẹt poster!
> - **Vì sao không random đều 100%?**: Vì phân bố thực tế có độ lệch tự nhiên (`prompt_test.txt`: headline thường súc tích, quote/bullet-point thường dài). Random đều sẽ pha loãng ngân sách gradient trên các tổ hợp phi thực tế.

Do đó, toàn bộ $2,500$ mẫu huấn luyện áp dụng **Tỷ Lệ Phân Tầng Vàng 75/25 (Golden Stratified Length Ratio)**:

| Phân Tầng Độ Dài (Stratum) | Tỷ Trọng | Cấu Trúc Độ Dài Các Slot | Mục Tiêu & Cơ Sở Khoa Học |
| :--- | :---: | :--- | :--- |
| **1. Phân Bố Tự Nhiên Thương Mại**<br>*(Canonical Commercial Skew)* | **75% – 80%** | • $t=10$ Headline: Ngắn – Vừa ($2 - 6$ từ, $1 - 2$ dòng).<br>• $t=20$ Subtitle: Vừa ($4 - 8$ từ, $1 - 2$ dòng).<br>• $t=30$ CTA Badge: Ngắn gọn, súc tích ($1 - 4$ từ, 1 dòng).<br>• $t=40$ Features / Quote: Dài, danh sách tính năng ($8 - 25$ từ, $2 - 4$ dòng). | Khớp $100\%$ với phân bố sử dụng thực tế trong sản xuất và các prompt thương mại của tester. |
| **2. Chủ Đích Nghịch Đảo Độ Dài**<br>*(Inverted Counter-Skew)* | **20% – 25%** | • **$t=10$ Headline Dài Bất Thường**: $10 - 18$ từ / 3 dòng (Triết lý thương hiệu, trích dẫn tri ân).<br>• **$t=20$ Subtitle Cực Ngắn**: $1 - 2$ từ (*"SIÊU NHẸ"*, *"PRO"*, *"5G"*).<br>• **$t=30$ CTA Dài Hơn Headline**: $8 - 12$ từ (*"NHỮNG VẬT BẤT LY THÂN CỦA BẠN"*).<br>• **$t=40$ Features Cực Ngắn**: $1 - 3$ từ (*"PIN 14 NGÀY"*). | **Phá vỡ vĩnh viễn liên kết giả `slot \Longleftrightarrow \text{length}`**, buộc Attention Heads học đúng nguyên tắc: *"Vai trò chỉ do RoPE time offset và Prompt quyết định, không phụ thuộc vào độ dài câu!"* |

> 🚨 **QUY TẮC KHẮC CỐT GHI TÂM**: **Phân tầng nghịch đảo này PHẢI ĐƯỢC ĐƯA VÀO NGAY TỪ MILESTONE A (2 SLOTS)**! Tuyệt đối không dồn tới Milestone C, vì nếu để mô hình học sai tương quan `slot \Longleftrightarrow \text{length}` ở bài toán 2 slots, các milestone 3-4-5 slots sau đó sẽ kế thừa và khuếch đại sai lệch, rất khó khắc phục!

---

### 2.5. Ma trận Đa Tỉ Lệ Khung Hình (Aspect Ratio Bucketing):

Toàn bộ $2,500$ mẫu được phân bổ đều đặn theo 4 tỉ lệ hiển thị thực tế:

| Tỉ Lệ Bucket | Kích Thước Pixel | Latent Grid ($16\times$) | Token Canvas | Tỷ Trọng | Ứng Dụng Nghiệp Vụ Thương Mại Viettel |
| :---: | :---: | :---: | :---: | :---: | :--- |
| **1:1** (Vuông) | $1024 \times 1024$ | $64 \times 64$ | $4,096$ tokens | **35%** | Bài đăng Feed Facebook, Instagram, E-commerce Post |
| **9:16** (Dọc cao) | $768 \times 1344$ | $48 \times 84$ | $4,032$ tokens | **35%** | **TikTok Ads**, Instagram Reels, Story, Standee quảng cáo Viettel |
| **4:5** (Dọc vừa) | $896 \times 1152$ | $56 \times 72$ | $4,032$ tokens | **15%** | Instagram Portrait Post (Tối ưu diện tích màn hình mobile) |
| **16:9** (Ngang) | $1344 \times 768$ | $84 \times 48$ | $4,032$ tokens | **15%** | Facebook Fanpage Cover, Website Banner, TV Display, Trình chiếu |

---

### 2.6. Quy Trình Chế Tạo Dataset Lũy Tiến Tích Lũy (Progressive Curriculum Pipeline):

```
                                  TỔNG QUY MÔ DATASET: 2,500 MẪU ĐỘC LẬP
                                                    │
         ┌──────────────────────────────────────────┼──────────────────────────────────────────┐
         ▼                                          ▼                                          ▼
 [ 🎯 NHÓM A: 800 MẪU (Milestone A) ]       [ 🎯 NHÓM B: 700 MẪU MỚI (Milestone B) ]   [ 🎯 NHÓM C: 1,000 MẪU MỚI (Milestone C) ]
 • Phục vụ: Milestone A (800 mẫu)           • Phục vụ: Milestone B (1,500 mẫu = A + B) • Phục vụ: Milestone C (2,500 mẫu = A+B+C)
 • 1-Shot Input: 2 Slots Cạnh Tranh         • 1-Shot Input: 3 Slots Cạnh Tranh         • 1-Shot Input: 4-5 Slots Toàn Diện
   - 440 mẫu SP: [Ref_10 + Ref_SP_20]         - 825 mẫu SP: [Ref_10, 20 + Ref_SP_30]     - 1,375 mẫu SP: [Ref_10..40 + Ref_SP_50]
   - 360 mẫu T2I: [Ref_10 + Ref_20]           - 675 mẫu T2I: [Ref_10, 20, 30]            - 1,125 mẫu T2I: [Ref_10, 20, 30, 40]
 • 4 Topologies + 16 Fonts Trực Giao        • 4 Topologies + 16 Fonts Trực Giao        • 4 Topologies + 16 Fonts Trực Giao
 • 75% Tự Nhiên + 25% Đảo Lệch Độ Dài      • 75% Tự Nhiên + 25% Đảo Lệch Độ Dài      • 75% Tự Nhiên + 25% Đảo Lệch Độ Dài
         │                                          │                                          │
         └──────────────────────────────────────────┼──────────────────────────────────────────┘
                                                    ▼
                                   [ AUTOMATED QUALITY ASSURANCE FILTER ]
                                   • OCR Check: Khớp chính xác ký tự tiếng Việt >= 98%
                                   • Độ phân giải chuẩn hóa theo 4 Aspect Ratio Buckets
                                   • Đóng gói thành WebDataset Shards (.tar / .h5)
```

> 📌 **Chi tiết triển khai kỹ thuật Milestone A**: Xem toàn bộ đặc tả ma trận đa chiều, kho 50 sản phẩm tuyển chọn và quy trình chế tạo dữ liệu tại [`docs/SUBPLAN_MILESTONE_A_DATA_PIPELINE.md`](file:///d:/Viettel%20Telecom/Tendoo%20AI/docs/SUBPLAN_MILESTONE_A_DATA_PIPELINE.md).

---

## ⚙️ 3. THIẾT KẾ KIẾN TRÚC LORA & HYPERPARAMETERS

### 3.1. Cấu hình PEFT LoRA Injection:
```python
lora_config = {
    "r": 32,                          # Rank 32 đủ dung lượng học phân luồng đa slot
    "lora_alpha": 32,                 # Scaling factor = 1.0
    "lora_dropout": 0.05,
    "target_modules": [
        "img_attn.qkv",               # 5 DoubleStreamBlocks (Query, Key, Value nhánh ảnh/glyph)
        "txt_attn.qkv",               # 5 DoubleStreamBlocks (Query, Key, Value nhánh text-prompt)
        "linear1",                    # 20 SingleStreamBlocks (Fused Joint QKV Attention + MLP Projection)
    ],
    "bias": "none",
    "dtype": "bfloat16"
}
```

* **Tổng tham số LoRA cần huấn luyện**: $\mathbf{23,592,960\text{ parameters}}$ (**chỉ chiếm $\mathbf{0.58\%}$ mô hình Base 4B**).
* **Kích thước file trọng số LoRA lưu trữ**: $\approx \mathbf{47.2\text{ MB}}$ (`.safetensors`).

---

### 3.2. Bảng Siêu Tham Số Huấn Luyện (Hyperparameters):

| Tham số Huấn luyện | Giá trị Cấu hình | Cơ sở Lý thuyết / Tính toán |
| :--- | :--- | :--- |
| **Optimizer** | `AdamW` ($\beta_1=0.9, \beta_2=0.999$, $\epsilon=10^{-8}$) | Chuẩn cho Diffusion Transformers |
| **Weight Decay** | $0.01$ | Chống overfit trên các font hiếm |
| **Learning Rate** | $1.0 \times 10^{-4}$ | Tốc độ tối ưu cho LoRA Rank 32 |
| **LR Scheduler** | `CosineAnnealingLR` với 150 warmup steps | Đảm bảo ổn định ở các step đầu |
| **Batch Size per GPU** | $1$ sample ($1024 \times 1024$) | Nhờ Tight-Crop, Sequence Length chỉ còn $\sim 9,450$ tokens (tiết kiệm VRAM, an toàn tuyệt đối) |
| **Gradient Accumulation** | $4$ steps | Tạo Effective Batch Size $= 1 \times 2 \times 4 = \mathbf{8}$ |
| **Precision Mode** | `bfloat16` Native Mixed Precision | Tối ưu kiến trúc Tensor Core A30, chống tràn số |
| **Gradient Checkpointing** | Kích hoạt trên toàn bộ 25 Blocks | Giảm bộ nhớ kích hoạt trung gian $>65\%$ |
| **Max Gradient Norm** | $1.0$ (Gradient Clipping) | Chống hiện tượng gradient spike khi gặp glyph phức tạp |
| **Text Conditioning Dropout** | $p = 0.10$ ($10\%$ số step train) | Thay thế `txt` bằng chuỗi rỗng `""`, giữ nguyên $100\%$ Reference Tokens để LoRA học đúng nhánh Unconditional |
| **Cấu Trúc Sequence Huấn Luyện** | Dãy tiền tố liên tục ($100\%$ Contiguous Prefixes) | Tuyệt đối không sinh slot ngắt quãng, khớp $100\%$ với Backend: $\{10, 20\}$, $\{10, 20, 30\}$, $\{10, 20, 30, 40, 50\}$ |
| **Hàm Mất Mát (Loss)** | $\mathcal{L} = (1 + (\lambda_{\text{prod}}-1)\mathbf{M}_{\text{prod}}) \odot \| v_\theta - (x_1 - x_0) \|^2$ | Flow Matching có Masked Focal Loss ($\lambda_{\text{prod}}=2.0$) |

---

## 📈 4. LỘ TRÌNH HUẤN LUYỆN 3 MỐC THỰC CHIẾN (2 $\rightarrow$ 3 $\rightarrow$ 4/5 SLOTS)

```
        [ 🎯 MILESTONE A: 2 SLOTS ]              [ ⚡ MILESTONE B: 3 SLOTS ]              [ 🏆 MILESTONE C: 4-5 SLOTS ]
        Phase Tách Kênh Cốt Lõi                  Phase Mở Rộng 3 Kênh                     Phase Full Production Hoàn Chỉnh
         800 steps (~8.0 epochs)                  1,400 steps (~7.5 epochs)                2,200 steps (~7.04 epochs)
        Tập dữ liệu: 800 samples                 Tập dữ liệu: 1,500 samples               Tập dữ liệu: 2,500 samples
   • Giải quyết dứt điểm tranh chấp          • Mở khóa kênh t=30 (CTA Badge)          • Khóa cứng 4 Text Slots + Sản Phẩm
     giữa 2 Kênh: t=10 và t=20                 và cân bằng phân cấp 3 tầng              ở mốc t=50 (Full-Power Banner)
```

### 📍 Chi tiết từng Milestone:

#### 🔹 Milestone A: Kích hoạt Phân Tách Kênh Đôi ($2\text{ Slots}$) — Quy mô: $800$ mẫu
* **Phân bổ**: $440$ mẫu SP $[t=10\text{ Text} + t=20\text{ SP}]$ + $360$ mẫu Pure T2I $[t=10\text{ Title} + t=20\text{ Subtitle}]$.
* **Mục tiêu**: Dạy LoRA giải quyết bài toán cốt lõi đầu tiên: **Phân luồng Softmax giữa $t=10$ và $t=20$**, triệt tiêu hoàn toàn hiện tượng tràn kênh (Attention Bleeding) và chữ rác Lorem Ipsum.
* **Đặc Điểm Phân Tầng Độ Dài (Bắt Buộc Ngay Từ Milestone A)**:
  - **75% mẫu tự nhiên**: $t=10$ Title ngắn/vừa ($3 - 6$ từ), $t=20$ Subtitle vừa ($5 - 8$ từ).
  - **25% mẫu nghịch đảo độ dài**: $t=10$ Title dài bất thường ($12 - 15$ từ / $2 - 3$ dòng), $t=20$ Subtitle cực ngắn ($1 - 2$ từ) hoặc ngược lại $\implies$ **Phá vỡ vĩnh viễn định kiến `slot \Longleftrightarrow \text{length}` ngay từ gốc!**
* **Số bước**: `800 steps` (~1.2 giờ trên 2x A30).

#### 🔹 Milestone B: Mở Rộng 3 Tầng Thị Giác ($3\text{ Slots}$) — Quy mô: $1,500$ mẫu ($800\text{ cũ} + 700\text{ mới}$)
* **Phân bổ**: $825$ mẫu SP $[t=10, 20\text{ Text} + t=30\text{ SP}]$ + $675$ mẫu Pure T2I $[t=10, 20, 30\text{ Texts}]$.
* **Mục tiêu**: Kích hoạt khả năng nhận diện kênh $t=30$, dạy mô hình tự động bao gói các cụm từ kêu gọi hành động (*"MUA 1 TẶNG 1"*, *"GIẢM 50%"*) thành các khung Badge/Huy hiệu/Neon nhỏ xinh mà **không cần Prompt gợi ý**.
* **Số bước**: `1,400 steps` (~2.0 giờ trên 2x A30).

#### 🔹 Milestone C: Toàn Diện 4–5 Kênh Cực Hạn ($4-5\text{ Slots}$) — Quy mô: $2,500$ mẫu ($1,500\text{ cũ} + 1,000\text{ mới}$)
* **Phân bổ**: $1,375$ mẫu SP $[3-4\text{ Texts} + t=50\text{ SP}]$ + $1,125$ mẫu Pure T2I $[4\text{ Texts Thuần}]$.
* **Mục tiêu**: Khóa cứng toàn bộ ma trận Attention cho mọi tình huống thương mại phức tạp nhất, kích hoạt cơ chế Masked Focal Loss để bảo vệ sản phẩm ở $t=50$ nguyên vẹn $100\%$.
* **Số bước**: `2,200 steps` (~3.5 giờ trên 2x A30).

---

## ⏱️ 5. BẢNG DỰ TOÁN TÀI NGUYÊN & THỜI GIAN

### 5.1. Dự toán Bộ nhớ VRAM trên 2x NVIDIA A30 (48GB Tổng cộng):

| Thành phần Bộ nhớ | Dung lượng GPU 0 | Dung lượng GPU 1 | Ghi chú Kỹ thuật |
| :--- | :---: | :---: | :--- |
| **Trọng số DiT 4B Base (BF16)** | $8.2\text{ GB}$ | $8.2\text{ GB}$ | Đóng băng 100% gradient |
| **Trọng số LoRA + Optimizer States** | $1.2\text{ GB}$ | $1.2\text{ GB}$ | AdamW states (FP32 master weights) |
| **Latent Cache + Text Embeddings** | $1.5\text{ GB}$ | $1.5\text{ GB}$ | Pre-computed embeddings |
| **Activations (Gradient Checkpointed)** | $6.2\text{ GB}$ | $6.2\text{ GB}$ | Sequence dài $\sim 9,450$ tokens |
| **CUDA Workspace & PyTorch Overhead** | $1.0\text{ GB}$ | $1.0\text{ GB}$ | Bộ đệm phân mảnh |
| 📊 **TỔNG VRAM SỬ DỤNG MỖI GPU** | **$18.1\text{ GB}$ / $24\text{ GB}$** | **$18.1\text{ GB}$ / $24\text{ GB}$** | 🟢 **Dư an toàn $\approx 5.9\text{ GB}$ Headroom!** |


---

### 5.2. Dự toán Tiến độ Thực tế (Realistic Timeline Estimates):

```
╔══════════════════════════════════════════════════════════╦══════════════╦═════════════════════════════════╗
║ Hạng Mục Công Việc                                       ║ Thời Gian    ║ Compute / Nhân Lực Cần Thiết    ║
║ 1. Lập trình Script Distillation (`generate_dataset.py`)  ║ 0.5 Ngày     ║ Agent viết code trên Local      ║
║ 2. Sinh Dataset 2,500 mẫu (Async Gemini Teacher + OCR QA)║ 6 – 8 Giờ    ║ Chạy nền qua đêm (Batch script) ║
║ 3. Lập trình Trainer LoRA DDP (`train_lora_dit.py`)      ║ 0.5 Ngày     ║ DDP Accelerate trên 2x A30      ║
║ 4. Huấn luyện Milestone A & B (2,200 steps tổng)         ║ ~3.2 Giờ     ║ 2x GPU A30 chạy liên tục        ║
║ 5. Đánh giá & Điều chỉnh Hyperparameters                 ║ 0.5 Ngày     ║ Chạy Benchmark Suite 20 mẫu     ║
║ 6. Huấn luyện Full Milestone C (4,400 steps tích lũy)    ║ ~3.5 Giờ     ║ 2x GPU A30 chạy qua đêm         ║
║ 7. Đóng gói Serving API & Gradio Web UI                  ║ 1.0 Ngày     ║ Hoàn thiện sản phẩm End-to-End  ║
╠══════════════════════════════════════════════════════════╬══════════════╬═════════════════════════════════╣
║ 🏆 TỔNG THỜI GIAN HOÀN THÀNH GIAI ĐOẠN 3                ║ **3 – 4 NGÀY**║ Compute Server 2x A30 sẵn sàng  ║
╚══════════════════════════════════════════════════════════╩══════════════╩═════════════════════════════════╝
```

---

## 🛡️ 6. BỘ ĐÁNH GIÁ VÀNG (GOLDEN EVALUATION SUITE)

Cứ sau mỗi **500 steps**, trainer tự động tạm dừng và sinh ảnh đánh giá trên **8 Golden Test Cases** bao phủ đủ 5 ngành hàng và bài test chống hồi quy:
0. *Test 0 (Single-Slot Regression Test)*: Duy nhất 1 Headline ở $t=10.0$ (Xác nhận LoRA không phá vỡ độ chính xác 100% vốn có của mô hình Base).
1. *Test 1 (F&B / Cafe)*: Poster Cafe Grand Opening 3 tầng chữ (Gỗ/Neon).
2. *Test 2 (Tech / Audio)*: Poster Tai nghe chụp tai với Headline 3D kim loại + Subtitle mạ bạc + CTA Neon.
3. *Test 3 (Fashion / Clothing)*: Poster Flash Sale Thời trang cao cấp với chất liệu chữ vàng đồng.
4. *Test 4 (Spa / Cosmetics)*: Poster Spa Thảo mộc dưỡng da cao cấp (Chất liệu chữ pastel/tối giản).
5. *Test 5 (Supermarket / FMCG)*: Poster Siêu thị Đại hạ giá cuối tuần (Chất liệu chữ pop-art dập nổi).
6. *Test 6 (Literature / Dense Text)*: Bài thơ Tây Tiến 4 câu (28 từ, kiểm tra độ bền câu dài).
7. *Test 7 (Product Anchor 4096 tokens)*: Giày Sneaker thật $t=40$ + Headline $t=10$ + CTA $t=30$.

---

## 🎯 7. KẾT LUẬN

Bản Roadmap v5 này đã được chuẩn hóa tối hậu:
* **Loại bỏ hoàn toàn công việc thừa thãi** (không train 1 text).
* **Định hình tiến trình thực chiến lũy tiến $2 \rightarrow 3 \rightarrow 4/5$ Slots**.
* **Đồng bộ hóa $100\%$ giữa True CFG, Masked Product Loss và Cấu trúc Dãy Tiền Tố Liên Tục Khép Kín**.
