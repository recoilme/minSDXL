from diffusers import DiffusionPipeline
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
        self.original_size = (1024, 1024)

    def _get_add_time_ids(self, original_size, crops_coords_top_left, target_size, dtype):
        add_time_ids = list(original_size + crops_coords_top_left + target_size)
        print("time",add_time_ids,self.unet.config.addition_time_embed_dim,self.unet.config.cross_attention_dim )
        passed_add_embed_dim = (
            self.unet.config.addition_time_embed_dim * len(add_time_ids) + self.unet.config.cross_attention_dim 
        )
        expected_add_embed_dim = self.unet.add_embedding.linear_1.in_features

        if expected_add_embed_dim != passed_add_embed_dim:
            raise ValueError(
                f"Model expects an added time embedding vector of length {expected_add_embed_dim}, but a vector of {passed_add_embed_dim} was created. The model has an incorrect config. Please check `unet.config.time_embedding_type` and `text_encoder_2.config.projection_dim`."
            )

        add_time_ids = torch.tensor([add_time_ids], dtype=dtype)
        return add_time_ids

    def __call__(self, prompt, width=512, height=512, num_inference_steps=5, generator=None, **kwargs):

        # 0. Default height and width to unet
        height = height or self.default_sample_size * self.vae_scale_factor
        width = width or self.default_sample_size * self.vae_scale_factor

        original_size =  (512, 512)
        crops_coords_top_left = (0, 0)
        target_size = (height, width)

        # Токенизация промпта
        text_inputs = self.tokenizer(
            prompt,
            padding="max_length",
            max_length=self.tokenizer.model_max_length,
            truncation=True,
            add_special_tokens=True,
            return_tensors="pt"
        )
        input_ids = text_inputs.input_ids.to("cuda")
        attention_mask = text_inputs.attention_mask.to("cuda")

        # Получение текстовых эмбеддингов
        text_embeddings = self.text_encoder(input_ids,attention_mask=attention_mask)[0]

        time_ids = self._get_add_time_ids(
            original_size, crops_coords_top_left, target_size, dtype=text_embeddings.dtype
        )
        print(text_embeddings.shape,time_ids.shape)
        print("addition_time_embed_dim",self.unet.config)
        # Создание time_ids (дополнительные эмбеддинги для размера изображения)
        #time_ids = torch.tensor([[height, width, height, width]], dtype=torch.float32).to("cuda")
        added_cond_kwargs = {"text_embeds": text_embeddings, "time_ids": time_ids.to("cuda")}

        # Генерация начального шума
        batch_size = text_embeddings.shape[0]
        height = 512  # Размер изображения
        width = 512
        shape = (batch_size, self.unet.config.in_channels, height // 8, width // 8)
        latents = torch.randn(shape, generator=generator, device="cuda")

        # Процесс диффузии
        self.scheduler.set_timesteps(num_inference_steps)
        for t in self.progress_bar(self.scheduler.timesteps):
            latent_model_input = self.scheduler.scale_model_input(latents, t).half()
            #noise_pred = self.unet(latent_model_input, t, encoder_hidden_states=text_embeddings).sample
            # Передача added_cond_kwargs в UNet
            noise_pred = self.unet(
                latent_model_input,
                t,
                encoder_hidden_states=text_embeddings,
                added_cond_kwargs=added_cond_kwargs
            ).sample

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
