"""
Generates 3 more real (paid, gpt-image-1) test backgrounds representing DIFFERENT scene
families than the one already tested (prompt7: pastel fashion portrait, product_ad).

Picked to stress different aspects of Cap do 2 (detection + MER):
  A) "product_only"   -- prompt_test.txt line 13, product_ad, NO face at all (pure product
                          shot on a pedestal). Tests whether detection+MER behaves better when
                          there's no face-detection noise to deal with.
  B) "multi_face"      -- prompt_test.txt line 39, feedback-shaped, TWO faces close together
                          (wedding couple kissing). Tests multi-obstacle behavior.
  C) "already_clear"   -- prompt_test.txt line 19, landscape hiker-vs-mountains. The person is
                          small/distant and the composition already has a large naturally empty
                          sky region. Sanity/baseline case: MER should NOT invent an artificially
                          narrow safe area when the scene is already clear.

No literal text is requested in any prompt (matches the 100%-overlay architecture: Stage 2
diffusion/generation must never bake in text).
"""
import base64
import os
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()
client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

OUT_DIR = Path("output_hero_selector_test")
OUT_DIR.mkdir(exist_ok=True)

CASES = {
    "case_a_product_only": dict(
        prompt=(
            "Minimalist product photograph of a sleek glossy black smart watch placed on a "
            "glossy black pedestal. A pure white spotlight beams straight down from above, "
            "creating a dramatic highlight. Dark background with depth. Photorealistic, "
            "studio product photography, 8k, no text, no logo, no watermark."
        ),
        size="1536x1024",
    ),
    "case_b_multi_face": dict(
        prompt=(
            "Romantic elegant wedding photography. A bride and groom sharing a passionate kiss "
            "on a beach at sunrise, beautiful golden-hour backlighting silhouetting their "
            "shoulders. White and beige tones, warm golden light, cinematic lighting, "
            "photorealistic, ultra realistic, high detail, no text, no watermark."
        ),
        size="1536x1024",
    ),
    "case_c_already_clear": dict(
        prompt=(
            "A hiker wearing an outdoor smart watch, standing on a ridge facing a majestic "
            "mountain range at dawn, small in the frame relative to the huge sky. Beautiful "
            "sun rays, epic wide landscape, large clear sky area, photorealistic, ultra "
            "realistic, 8k, no text, no watermark."
        ),
        size="1536x1024",
    ),
}


def main():
    for name, cfg in CASES.items():
        out_path = OUT_DIR / f"{name}_bg.png"
        if out_path.exists():
            print(f"[skip] {out_path} already exists")
            continue
        print(f"[gen] {name} ...")
        resp = client.images.generate(
            model="gpt-image-1",
            prompt=cfg["prompt"],
            size=cfg["size"],
            quality="low",
            n=1,
        )
        b64 = resp.data[0].b64_json
        out_path.write_bytes(base64.b64decode(b64))
        print(f"[ok] wrote {out_path}")


if __name__ == "__main__":
    main()
