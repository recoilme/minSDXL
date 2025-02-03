import os

import torch
from diffusers import DiffusionPipeline
from transformers import XLMRobertaTokenizerFast,XLMRobertaModel
from diffusers.models import AutoencoderKL
from diffusers import DDIMScheduler
from diffusers import UNet2DConditionModel

pipe_id = "/home/recoilme/models/waifu-4b"
# tokenizer
tokenizer = XLMRobertaTokenizerFast.from_pretrained(
    pipe_id,
    subfolder="tokenizer"
)

# text_encoder
text_encoder = XLMRobertaModel.from_pretrained(
    pipe_id,
    subfolder="text_encoder",
    torch_dtype=torch.float16,
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
    torch_dtype=torch.float16,
    subfolder="vae"
).to("cuda")

# Загрузка предобученной модели UNet2DConditionModel
unet = UNet2DConditionModel.from_pretrained(
    pipe_id,
    torch_dtype=torch.float16,
    #"/home/recoilme/models/waifu-2b",
    #variant = "fp16",
    subfolder="unet",
).to("cuda")

# Путь к директории модели
pipe_id = "/home/recoilme/models/waifu-4b"

# Создаем pipeline с существующими компонентами

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
        
        # Фикс для совместимости с SDXL
        self.unet.config.addition_time_embed_dim = 256  # Должно совпадать с конфигом модели
        self.unet.config.in_channels = 4 if vae.config.latent_channels is None else vae.config.latent_channels

    def _get_add_time_ids(self, original_size, crops_coords_top_left, target_size, dtype):
        # Формат: [original_h, original_w, crop_top, crop_left, target_h, target_w]
        add_time_ids = list(original_size + crops_coords_top_left + target_size)
        add_time_ids = torch.tensor([add_time_ids], dtype=dtype, device=self.device)
        return add_time_ids

    def __call__(self, prompt, height=1024, width=1024, num_inference_steps=50, guidance_scale=7.5, **kwargs):
        # Токенизация
        text_inputs = self.tokenizer(
            prompt,
            padding="max_length",
            max_length=self.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt"
        )
        input_ids = text_inputs.input_ids.to(self.device)
        
        # Текстовые эмбеддинги
        text_embeddings = self.text_encoder(input_ids)[0]
        #proj = self.text_encoder.text_projector(text_embeddings)
        #print("proj",proj.shape)
        print("text_embeddings",text_embeddings.shape)
        
        # Временные эмбеддинги
        original_size = (height, width)
        crops_coords_top_left = (0, 0)
        target_size = (height, width)
        time_ids = self._get_add_time_ids(
            original_size, 
            crops_coords_top_left, 
            target_size, 
            dtype=text_embeddings.dtype
        )
        
        # Подготовка условий
        added_cond_kwargs = {
            "text_embeds": text_embeddings,
            "time_ids": time_ids.repeat(text_embeddings.shape[0], 1)
        }

        # Генерация латентов
        latents = torch.randn(
            (1, self.unet.config.in_channels, height//8, width//8),
            device=self.device,
            dtype=text_embeddings.dtype
        )
        
        # Процесс диффузии
        self.scheduler.set_timesteps(num_inference_steps)
        for t in self.scheduler.timesteps:
            latent_model_input = self.scheduler.scale_model_input(latents, t)
            
            noise_pred = self.unet(
                latent_model_input,
                t,
                encoder_hidden_states=text_embeddings,
                added_cond_kwargs=added_cond_kwargs
            ).sample
            
            latents = self.scheduler.step(noise_pred, t, latents).prev_sample

        # Декодирование
        image = self.vae.decode(latents / self.vae.config.scaling_factor)[0]
        return image
# Создаем экземпляр кастомного пайплайна
pipeline = CustomPipeline(
    tokenizer=tokenizer,
    text_encoder=text_encoder,
    vae=vae,
    unet=unet,
    scheduler=scheduler
).to("cuda").to(torch.float16)

print(pipeline)
print('unet.config\n',unet.config)
prompt = "Ваш запрос для генерации изображения"
image = pipeline(prompt, num_inference_steps=5)
print(image.shape)
# Сохраняем pipeline
#if not os.path.exists(pipe_id):
#    os.makedirs(pipe_id)

#pipeline.to(torch.float16)
#pipeline.save_pretrained(pipe_id)
