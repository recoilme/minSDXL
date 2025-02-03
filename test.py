import torch
from diffusers import DiffusionPipeline
from transformers import XLMRobertaTokenizerFast,XLMRobertaModel
from diffusers.models import AutoencoderKL
from diffusers import DDIMScheduler
from diffusers import UNet2DConditionModel

pipe_id = "/home/recoilme/models/waifu-2b"
variant = "fp16"
# tokenizer
tokenizer = XLMRobertaTokenizerFast.from_pretrained(
    pipe_id,
    variant = variant,
    subfolder="tokenizer"
)

# text_encoder
text_encoder = XLMRobertaModel.from_pretrained(
    pipe_id,
    variant = variant,
    subfolder="text_encoder",
    add_pooling_layer=False
).to("cuda")

scheduler = DDIMScheduler(
    beta_schedule="scaled_linear",           # Косинусное расписание
    prediction_type="v_prediction",   # Необходимо для v-предсказания
    num_train_timesteps=1000          # Число шагов, которое использовалось при обучении
)

# VAE
vae = AutoencoderKL.from_pretrained(
    pipe_id,
    variant = variant,
    subfolder="vae"
).to("cuda")

# Загрузка предобученной модели UNet2DConditionModel
#unet = UNet2DConditionModel.from_pretrained(
#    pipe_id,
#    variant = variant,
#    subfolder="unet",
#).to("cuda").to(torch.float16)
#num_params = sum(p.numel() for p in unet.parameters())
#print(f"Количество параметров: {num_params}")

generator = torch.Generator(device="cuda").manual_seed(42)
text_encoder.to(torch.float16)
save = True
if save:
    # Сохранение модели в формате diffusers
    save_directory = "/home/recoilme/models/te"
    text_encoder.save_pretrained(save_directory, safe_serialization=True)

    print(f"Модель успешно сохранена в {save_directory}")

    print(text_encoder)