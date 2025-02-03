from diffusers import DiffusionPipeline
import os

import torch
from diffusers import DiffusionPipeline
from transformers import XLMRobertaTokenizerFast,XLMRobertaModel
from diffusers.models import AutoencoderKL
from diffusers import DDIMScheduler
from diffusers import UNet2DConditionModel

pipe_id = "/home/recoilme/models/waifu-3b"
# tokenizer
tokenizer = XLMRobertaTokenizerFast.from_pretrained(
    pipe_id,
    subfolder="tokenizer"
)

# text_encoder
text_encoder = XLMRobertaModel.from_pretrained(
    pipe_id,
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
    subfolder="vae"
).to("cuda")

# Загрузка предобученной модели UNet2DConditionModel
unet = UNet2DConditionModel.from_pretrained(
    pipe_id,
    subfolder="unet",
).to("cuda")

# Путь к директории модели
pipe_id = "/home/recoilme/models/waifu-4b"

# Создаем pipeline с существующими компонентами


from diffusers import DiffusionPipeline

class CustomPipeline(DiffusionPipeline):
    def __init__(self, tokenizer, text_encoder, vae, unet, scheduler):
        super().__init__()
        self.register_modules(
            tokenizer=tokenizer,
            text_encoder=text_encoder,
            vae=vae,
            unet=unet,
            scheduler=scheduler
        )
        self.text_encoder.pooler = None

    def __call__(self, prompt, num_inference_steps=5, generator=None, **kwargs):

        # Токенизация промпта
        text_inputs = self.tokenizer(
            prompt,
            padding="max_length",
            max_length=self.tokenizer.model_max_length,
            truncation=True,
            add_special_tokens=True,
            return_tensors="pt"
        )
        input_ids = text_inputs.input_ids.to("cuda").to(torch.float16)
        attention_mask = text_inputs.attention_mask.to("cuda").to(torch.float16)

        # Получение текстовых эмбеддингов
        text_embeddings = self.text_encoder(input_ids,attention_mask=attention_mask)[0]

        # Генерация начального шума
        batch_size = text_embeddings.shape[0]
        height = 512  # Размер изображения
        width = 512
        shape = (batch_size, self.unet.config.in_channels, height // 8, width // 8)
        latents = torch.randn(shape, generator=generator, device="cuda")

        # Процесс диффузии
        self.scheduler.set_timesteps(num_inference_steps)
        for t in self.progress_bar(self.scheduler.timesteps):
            latent_model_input = self.scheduler.scale_model_input(latents, t)
            noise_pred = self.unet(latent_model_input, t, encoder_hidden_states=text_embeddings).sample
            latents = self.scheduler.step(noise_pred, t, latents).prev_sample

        # Декодирование латентов в изображение
        image = self.vae.decode(latents / self.vae.config.scaling_factor, return_dict=False)[0]
        image = (image / 2 + 0.5).clamp(0, 1)
        image = image.cpu().permute(0, 2, 3, 1).numpy()

        return image

# Создаем экземпляр кастомного пайплайна
pipeline = CustomPipeline(
    tokenizer=tokenizer,
    text_encoder=text_encoder,
    vae=vae,
    unet=unet,
    scheduler=scheduler
).to("cuda").to(torch.float16)

prompt = "Ваш запрос для генерации изображения"
image = pipeline(prompt, num_inference_steps=5)
print(image.shape)
# Сохраняем pipeline
#if not os.path.exists(pipe_id):
#    os.makedirs(pipe_id)

#pipeline.to(torch.float16)
#pipeline.save_pretrained(pipe_id)


