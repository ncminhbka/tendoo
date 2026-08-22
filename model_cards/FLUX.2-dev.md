![Teaser](../assets/teaser_generation.png)
![Teaser](../assets/teaser_editing.png)

`FLUX.2 [dev]` là một mô hình rectified flow transformer 32 tỷ tham số (32B parameters), có khả năng tạo, chỉnh sửa và kết hợp nhiều hình ảnh dựa trên các câu lệnh văn bản.
Để biết thêm thông tin chi tiết, vui lòng đọc [bài viết trên blog của chúng tôi](https://bfl.ai/blog/flux-2).

# Các tính năng chính
1. Dẫn đầu về chất lượng (SOTA) trong việc sinh ảnh từ văn bản (text-to-image), chỉnh sửa ảnh đơn tham chiếu (single-reference editing) và chỉnh sửa đa tham chiếu (multi-reference editing).
2. Không cần fine-tune: Giữ tham chiếu nhân vật, đồ vật và phong cách mà không cần huấn luyện bổ sung trong cùng một mô hình.
3. Huấn luyện bằng kỹ thuật guidance distillation, giúp `FLUX.2 [dev]` đạt hiệu suất thực thi cao hơn.
4. Mở trọng số (Open weights) nhằm thúc đẩy nghiên cứu khoa học mới và hỗ trợ các nghệ sĩ phát triển các quy trình sáng tạo đột phá.
5. Kết quả sinh ra có thể được sử dụng cho mục đích cá nhân, nghiên cứu khoa học và thương mại, như đã quy định trong [Giấy phép phi thương mại FLUX [dev]](https://github.com/black-forest-labs/flux/blob/main/model_licenses/LICENSE-FLUX1-dev).

# Hướng dẫn sử dụng
Chúng tôi cung cấp bản triển khai tham chiếu của `FLUX.2 [dev]`, cũng như mã lấy mẫu (sampling code), trong một [repository github chuyên dụng](https://github.com/black-forest-labs/flux2).
Các nhà phát triển và nhà sáng tạo muốn xây dựng giải pháp trên nền tảng `FLUX.2 [dev]` được khuyến khích sử dụng repository này làm điểm xuất phát.

`FLUX.2 [dev]` cũng đã được hỗ trợ trên cả [ComfyUI](https://github.com/comfyanonymous/ComfyUI) và [Diffusers](https://github.com/huggingface/diffusers).

### Sử dụng với diffusers 🧨

Để triển khai cục bộ trên các card đồ họa phổ thông dành cho người dùng cá nhân như RTX 4090 hoặc RTX 5090, vui lòng xem [tài liệu diffusers](https://github.com/black-forest-labs/flux2/blob/main/docs/flux2_dev_hf.md) trên trang GitHub của chúng tôi.

Dưới đây là một ví dụ về cách nạp mô hình đã được lượng tử hóa 4-bit với remote text-encoder trên RTX 4090:

```python
import torch
from diffusers import Flux2Pipeline, Flux2Transformer2DModel
from diffusers.utils import load_image
from huggingface_hub import get_token
import requests
import io

repo_id = "diffusers/FLUX.2-dev-bnb-4bit"
device = "cuda:0"
torch_dtype = torch.bfloat16

def remote_text_encoder(prompts):
    response = requests.post(
        "https://remote-text-encoder-flux-2.huggingface.co/predict",
        json={"prompt": prompts},
        headers={
            "Authorization": f"Bearer {get_token()}",
            "Content-Type": "application/json"
        }
    )
    prompt_embeds = torch.load(io.BytesIO(response.content))

    return prompt_embeds.to(device)

pipe = Flux2Pipeline.from_pretrained(
    repo_id, transformer=transformer, text_encoder=None, torch_dtype=torch_dtype
).to(device)

prompt = "Realistic macro photograph of a hermit crab using a soda can as its shell, partially emerging from the can, captured with sharp detail and natural colors, on a sunlit beach with soft shadows and a shallow depth of field, with blurred ocean waves in the background. The can has the text `BFL Diffusers` on it and it has a color gradient that start with #FF5733 at the top and transitions to #33FF57 at the bottom."

image = pipe(
    prompt_embeds=remote_text_encoder(prompt),
    #image=load_image("https://huggingface.co/spaces/zerogpu-aoti/FLUX.1-Kontext-Dev-fp8-dynamic/resolve/main/cat.png") # Tùy chọn truyền ảnh đầu vào
    generator=torch.Generator(device=device).manual_seed(42),
    num_inference_steps=50, # 28 bước có thể là sự cân bằng tốt
    guidance_scale=4,
).images[0]

image.save("flux2_output.png")
```

---

# Các rủi ro và biện pháp giảm thiểu

Black Forest Labs cam kết phát triển và triển khai các mô hình một cách có trách nhiệm. Trước khi phát hành dòng mô hình FLUX.2, chúng tôi đã đánh giá và giảm thiểu hàng loạt rủi ro trong các checkpoint mô hình và dịch vụ lưu trữ, bao gồm việc tạo ra nội dung bất hợp pháp như tài liệu lạm dụng tình dục trẻ em (CSAM) và hình ảnh nhạy cảm không có sự đồng thuận (NCII). Chúng tôi đã triển khai một chuỗi các biện pháp giảm thiểu trước khi phát hành nhằm ngăn chặn việc sử dụng sai mục đích bởi bên thứ ba, cùng các biện pháp bổ sung sau phát hành để xử lý các rủi ro còn lại:

1. **Giảm thiểu trong giai đoạn Tiền huấn luyện (Pre-training)**: Chúng tôi đã lọc dữ liệu tiền huấn luyện cho nhiều danh mục "không an toàn tại nơi làm việc" (NSFW) và các tài liệu lạm dụng tình dục trẻ em (CSAM) đã biết để ngăn người dùng tạo nội dung phi pháp từ các câu lệnh văn bản hoặc hình ảnh tải lên. Chúng tôi đã hợp tác với Internet Watch Foundation, một tổ chức phi lợi nhuận độc lập chuyên ngăn chặn lạm dụng trực tuyến, để lọc các nội dung CSAM đã biết ra khỏi dữ liệu huấn luyện.
2. **Giảm thiểu trong giai đoạn Sau huấn luyện (Post-training)**: Sau đó, chúng tôi đã thực hiện nhiều đợt fine-tuning có chủ đích để tăng cường khả năng chống lại các hành vi lạm dụng tiềm ẩn, bao gồm cả các cuộc tấn công dạng text-to-image (T2I) và image-to-image (I2I). Bằng cách ức chế các hành vi và triệt tiêu một số khái niệm nhất định trong mô hình đã huấn luyện, các kỹ thuật này giúp ngăn chặn việc tạo ra CSAM hoặc NCII tổng hợp từ câu lệnh văn bản, hoặc biến đổi ảnh tải lên thành CSAM hoặc NCII.
3. **Đánh giá liên tục**: Trong suốt quá trình này, chúng tôi đã thực hiện nhiều đợt đánh giá nội bộ và bên thứ ba độc lập đối với các checkpoint của mô hình để xác định thêm các cơ hội giảm thiểu rủi ro. Các đánh giá từ bên thứ ba tập trung vào việc cố gắng kích hoạt CSAM và NCII thông qua kiểm thử đối kháng (adversarial testing) với (i) câu lệnh chỉ có văn bản, (ii) ảnh tham chiếu đơn lẻ kèm câu lệnh văn bản, và (iii) nhiều ảnh tham chiếu kèm câu lệnh văn bản. Dựa trên phản hồi này, chúng tôi đã tiến hành fine-tuning an toàn bổ sung để tạo ra phiên bản mô hình mở trọng số (`FLUX.2 [dev]`).
4. **Quyết định phát hành**: Sau quá trình fine-tuning an toàn và trước khi phát hành, chúng tôi đã tiến hành đánh giá lần cuối từ bên thứ ba đối với checkpoint dự kiến phát hành, tập trung vào khả năng sinh CSAM và NCII tổng hợp ở cả T2I và I2I, bao gồm việc so sánh với các mô hình T2I và I2I mở trọng số khác (tổng số prompt n≈2,800). Checkpoint `FLUX.2 [dev]` cuối cùng đã thể hiện khả năng chống chịu cao trước các dữ liệu đầu vào vi phạm trong các tác vụ tạo và chỉnh sửa ảnh phức tạp, đồng thời vượt trội hơn các mô hình mở trọng số hàng đầu khác ở các hạng mục rủi ro này. Dựa trên những phát hiện đó, chúng tôi đã phê duyệt việc phát hành mô hình FLUX.2 Pro qua API và phát hành mô hình mở trọng số `FLUX.2 [dev]` theo giấy phép phi thương mại để hỗ trợ nghiên cứu và phát triển từ bên thứ ba.
5. **Bộ lọc suy luận (Inference filters)**: Repository của mô hình `FLUX.2 [dev]` tích hợp sẵn các bộ lọc cho nội dung NSFW và vi phạm bản quyền (IP) ở cả đầu vào và đầu ra. Việc sử dụng các bộ lọc hoặc quy trình kiểm duyệt thủ công là bắt buộc theo các điều khoản của Giấy phép phi thương mại FLUX.2 [dev]. Chúng tôi có thể ngẫu nhiên kiểm tra các bên triển khai mô hình `FLUX.2 [dev]` đã biết để xác minh xem các bộ lọc hoặc quy trình kiểm duyệt thủ công có được áp dụng hay không. Ngoài ra, chúng tôi áp dụng nhiều bộ lọc để chặn các câu lệnh văn bản, ảnh tải lên và ảnh đầu ra trên API cho FLUX.2 [pro]. Chúng tôi sử dụng cả bộ lọc nội bộ và bộ lọc do bên thứ ba cung cấp (bao gồm từ Hive và Microsoft) để ngăn chặn đầu ra CSAM và NCII. Chúng tôi cũng cung cấp bộ lọc cho các danh mục nội dung có khả năng gây hại khác, bao gồm hình ảnh bạo lực/kinh dị (gore), có thể được các nhà phát triển điều chỉnh dựa trên hồ sơ rủi ro và các trường hợp sử dụng hợp pháp của họ.
6. **Nguồn gốc nội dung (Content provenance)**: Các tính năng xác thực nguồn gốc nội dung giúp người dùng và nền tảng nhận diện, gắn nhãn và diễn giải tốt hơn nội dung do AI tạo ra trên môi trường mạng. Mã suy luận của `FLUX.2 [dev]` triển khai một ví dụ về thủy vân ở lớp điểm ảnh (pixel-layer watermarking), và repository này bao gồm các liên kết đến tiêu chuẩn siêu dữ liệu của Coalition for Content Provenance and Authenticity (C2PA). API cho FLUX.2 Pro áp dụng siêu dữ liệu C2PA có chữ ký số mã hóa cho nội dung đầu ra nhằm biểu thị rằng hình ảnh được tạo ra từ mô hình của chúng tôi.
7. **Chính sách**: Việc sử dụng các mô hình và truy cập API của chúng tôi chịu sự điều chỉnh bởi Giấy phép phi thương mại FLUX [dev] (dành cho người dùng mở trọng số phi thương mại); Điều khoản dịch vụ dành cho nhà phát triển, Điều khoản giấy phép thương mại tự lưu trữ và Chính sách sử dụng (dành cho người dùng mô hình mở trọng số thương mại); cùng với Điều khoản dịch vụ dành cho nhà phát triển, Điều khoản dịch vụ FLUX API và Chính sách sử dụng (dành cho người dùng API). Các chính sách này nghiêm cấm việc tạo nội dung bất hợp pháp hoặc sử dụng nội dung được tạo cho các mục đích bất hợp pháp, bôi nhọ hoặc lạm dụng. Các nhà phát triển và người dùng phải đồng ý với các điều kiện này để truy cập mô hình `FLUX.2 [dev]` trên Hugging Face.
8. **Giám sát**: Chúng tôi đang theo dõi các hành vi sử dụng vi phạm sau khi phát hành. Chúng tôi tiếp tục gửi và nâng cấp các yêu cầu gỡ bỏ tới các trang web, dịch vụ hoặc doanh nghiệp sử dụng sai mục đích các mô hình của chúng tôi. Ngoài ra, chúng tôi có thể cấm các tài khoản người dùng hoặc nhà phát triển bị phát hiện cố ý và liên tục vi phạm chính sách thông qua FLUX API. Chúng tôi cũng cung cấp một địa chỉ email chuyên dụng (safety@blackforestlabs.ai) để tiếp nhận phản hồi từ cộng đồng. Chúng tôi duy trì mối quan hệ báo cáo với các tổ chức như Internet Watch Foundation và National Center for Missing and Exploited Children, đồng thời hoan nghênh sự phối hợp liên tục với các cơ quan chức năng, nhà phát triển và nhà nghiên cứu để chia sẻ thông tin về các rủi ro mới xuất hiện và phát triển các biện pháp giảm thiểu hiệu quả.

# Giấy phép (License)
Mô hình này được phát hành theo [Giấy phép phi thương mại FLUX [dev]](https://huggingface.co/black-forest-labs/FLUX.2-dev/blob/main/LICENSE.txt).
