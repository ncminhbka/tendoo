# 🧨 Chạy mô hình với thư viện diffusers

## Bắt đầu 

Cài đặt `diffusers` từ nhánh `main` và nâng cấp các thư viện phụ thuộc `transformers`, `accelerate` và `bitsandbytes` lên phiên bản mới nhất:

```sh
pip install git+https://github.com/huggingface/diffusers.git
pip install --upgrade transformers accelerate bitsandbytes
```

Sau khi chấp nhận điều khoản cấp quyền truy cập (gating) trên [repository FLUX.2-dev](https://huggingface.co/black-forest-labs/FLUX.2-dev), đăng nhập tài khoản Hugging Face trên terminal của bạn:
```sh
hf auth login
```

Xem hướng dẫn bên dưới để chạy suy luận (inference) trên các mức cấu hình GPU khác nhau.

---

## 💾 Mức VRAM thấp (~24-32GB) - RTX 4090 và 5090

Những ai sở hữu GPU có 24-32GB VRAM có thể sử dụng mô hình với phương pháp **lượng tử hóa 4-bit (4-bit quantization)**.

### Transformer 4-bit và Remote Text-Encoder (~18GB VRAM)

Đội ngũ diffusers giới thiệu tính năng remote text-encoder cho bản phát hành này.
Text embedding sẽ được tính toán với độ chính xác `bf16` trên đám mây và bạn chỉ cần nạp phần transformer vào VRAM (cấu hình này có thể giảm mức tiêu thụ VRAM xuống chỉ còn khoảng ~18GB):

```py
import torch
from diffusers import Flux2Pipeline
from diffusers.utils import load_image
from huggingface_hub import get_token
import requests
import io

repo_id = "diffusers/FLUX.2-dev-bnb-4bit" # Text-encoder và DiT đã được lượng tử hóa. VAE vẫn ở định dạng bf16
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
    repo_id, text_encoder=None, torch_dtype=torch_dtype
).to(device)

prompt = "Realistic macro photograph of a hermit crab using a soda can as its shell, partially emerging from the can, captured with sharp detail and natural colors, on a sunlit beach with soft shadows and a shallow depth of field, with blurred ocean waves in the background. The can has the text `BFL Diffusers` on it and it has a color gradient that start with #FF5733 at the top and transitions to #33FF57 at the bottom."

#cat_image = load_image("https://huggingface.co/spaces/zerogpu-aoti/FLUX.1-Kontext-Dev-fp8-dynamic/resolve/main/cat.png")
image = pipe(
    prompt_embeds=remote_text_encoder(prompt),
    #image=[cat_image] # Tùy chọn truyền nhiều ảnh đầu vào
    generator=torch.Generator(device=device).manual_seed(42),
    num_inference_steps=50, # 28 bước có thể là mức cân bằng tốt giữa thời gian và chất lượng
    guidance_scale=4,
).images[0]

image.save("flux2_output.png")
```

### Transformer 4-bit và Text-Encoder 4-bit (~20GB VRAM)

Nạp cả text-encoder và transformer ở định dạng 4-bit.
Text-encoder sẽ được chuyển tạm sang RAM (offload) để giải phóng VRAM khi transformer chạy thông qua `pipe.enable_model_cpu_offload()`, đảm bảo cả hai đều vừa trong bộ nhớ GPU.

```py
import torch
from diffusers import Flux2Pipeline, AutoModel
from transformers import Mistral3ForConditionalGeneration
from diffusers.utils import load_image

repo_id = "diffusers/FLUX.2-dev-bnb-4bit" # Text-encoder và DiT đã được lượng tử hóa. VAE vẫn ở định dạng bf16
device = "cuda:0"
torch_dtype = torch.bfloat16

text_encoder = Mistral3ForConditionalGeneration.from_pretrained(
    repo_id, subfolder="text_encoder", torch_dtype=torch.bfloat16, device_map="cpu"
)
dit = AutoModel.from_pretrained(
    repo_id, subfolder="transformer", torch_dtype=torch.bfloat16, device_map="cpu"
)
pipe = Flux2Pipeline.from_pretrained(
    repo_id, text_encoder=text_encoder, transformer=dit, torch_dtype=torch_dtype
)
pipe.enable_model_cpu_offload()

prompt = "Realistic macro photograph of a hermit crab using a soda can as its shell, partially emerging from the can, captured with sharp detail and natural colors, on a sunlit beach with soft shadows and a shallow depth of field, with blurred ocean waves in the background. The can has the text `BFL + Diffusers` on it and it has a color gradient that start with #FF5733 at the top and transitions to #33FF57 at the bottom."
#cat_image = load_image("https://huggingface.co/spaces/zerogpu-aoti/FLUX.1-Kontext-Dev-fp8-dynamic/resolve/main/cat.png")
image = pipe(
    prompt=prompt,
    #image=[cat_image] # Đầu vào nhiều ảnh
    generator=torch.Generator(device=device).manual_seed(42),
    num_inference_steps=50,
    guidance_scale=4,
).images[0]

image.save("flux2_output.png")
``` 

Để hiểu rõ hơn các mức lượng tử hóa khác nhau ảnh hưởng thế nào đến khả năng và chất lượng của mô hình, vui lòng xem bài viết trên blog [FLUX.2 on diffusers](https://huggingface.co/blog/flux-2).

---

## 💿 Mức VRAM cao (80GB+)

Ngay cả một GPU H100 cũng không thể nạp đồng thời toàn bộ text-encoder, transformer và VAE cùng lúc ở kích thước nguyên bản. Tuy nhiên, vì từng thành phần riêng lẻ có thể vừa với VRAM, việc xử lý chỉ đơn giản là kích hoạt `pipe.enable_model_cpu_offload()`.
Đối với các dòng card lớn hơn như H200, B200 hoặc cụm GPU, mọi thứ đều có thể nạp vừa vào VRAM.

```py
import torch
from diffusers import Flux2Pipeline
from diffusers.utils import load_image

repo_id = "black-forest-labs/FLUX.2-dev"
device = "cuda:0"
torch_dtype = torch.bfloat16

pipe = Flux2Pipeline.from_pretrained(
    repo_id, torch_dtype=torch_dtype
)
pipe.enable_model_cpu_offload() # Không cần bật cpu offload đối với GPU có VRAM > 80GB như H200, B200... thay vào đó chỉ cần gọi `pipe.to(device)`

prompt = "Realistic macro photograph of a hermit crab using a soda can as its shell, partially emerging from the can, captured with sharp detail and natural colors, on a sunlit beach with soft shadows and a shallow depth of field, with blurred ocean waves in the background. The can has the text `BFL Diffusers` on it and it has a color gradient that start with #FF5733 at the top and transitions to #33FF57 at the bottom."

#cat_image = load_image("https://huggingface.co/spaces/zerogpu-aoti/FLUX.1-Kontext-Dev-fp8-dynamic/resolve/main/cat.png")
image = pipe(
    prompt=prompt,
    #image=[cat_image] # Đầu vào nhiều ảnh
    generator=torch.Generator(device=device).manual_seed(42),
    num_inference_steps=50,
    guidance_scale=4,
).images[0]

image.save("flux2_output.png")
```

### Remote Text-Encoder + H100
`pipe.enable_model_cpu_offload()` sẽ làm giảm tốc độ một chút. Bạn có thể đạt tốc độ nhanh nhất có thể trên H100 bằng cách kết hợp với remote text-encoder:
```py
import torch
from diffusers import Flux2Pipeline
from diffusers.utils import load_image
from huggingface_hub import get_token
import requests
import io

repo_id = "black-forest-labs/FLUX.2-dev"
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
    assert response.status_code == 200, f"{response.status_code=}"
    prompt_embeds = torch.load(io.BytesIO(response.content))

    return prompt_embeds.to(device)

pipe = Flux2Pipeline.from_pretrained(
    repo_id, text_encoder=None, torch_dtype=torch_dtype
).to(device)

prompt = "Realistic macro photograph of a hermit crab using a soda can as its shell, partially emerging from the can, captured with sharp detail and natural colors, on a sunlit beach with soft shadows and a shallow depth of field, with blurred ocean waves in the background. The can has the text `BFL + Diffusers` on it and it has a color gradient that start with #FF5733 at the top and transitions to #33FF57 at the bottom."

#cat_image = load_image("https://huggingface.co/spaces/zerogpu-aoti/FLUX.1-Kontext-Dev-fp8-dynamic/resolve/main/cat.png")
image = pipe(
    prompt_embeds=remote_text_encoder(prompt),
    #image=[cat_image] # Tùy chọn truyền nhiều ảnh đầu vào
    generator=torch.Generator(device=device).manual_seed(42),
    num_inference_steps=50,
    guidance_scale=4,
).images[0]

image.save("flux2_output.png")
```

## 🧮 Các mức dung lượng VRAM khác

Nếu bạn sử dụng GPU có dung lượng khác, bạn có thể thử nghiệm các mức lượng tử hóa khác nhau. Ví dụ với GPU có 40-48GB VRAM, mức lượng tử hóa 8-bit thay vì 4-bit sẽ mang lại sự cân bằng rất tốt giữa tốc độ và chất lượng. Bạn có thể tìm hiểu thêm chi tiết tại [bài viết phát hành FLUX.2 trên diffusers](https://huggingface.co/blog/flux-2).
