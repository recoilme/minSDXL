import torch
from diffusers import UNet2DConditionModel

# Загрузка предобученной модели UNet2DConditionModel
unet = UNet2DConditionModel.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0",
    subfolder="unet",
    torch_dtype=torch.float16,
    variant="fp16",
    use_safetensors=True
).to("cuda")

# Вывод всех параметров модели
#print(unet.config)

# Подсчет числа параметров
num_params = sum(p.numel() for p in unet.parameters())
print(f"Количество параметров: {num_params}")


# Изменение одного из параметров
new_config = unet.config.copy()
new_config["out_channels"] = 16
new_config["in_channels"] = 16 
new_config["cross_attention_dim"] = 1152
new_config["layers_per_block"] = 3
new_config["attention_head_dim"] = [8, 16, 32]  #[4, 8, 16] #num_attention_heads
new_config["transformer_layers_per_block"] = [2, 4, 10] # 2822619856 vs 3217997776 vs 3342013136
new_config["use_linear_projection"] = True
new_config["resnet_time_scale_shift"]= "scale_shift"
new_config["projection_class_embeddings_input_dim"] = 1152 + (6*256) #2688  # 1280(text) + 6*256(time)
new_config["act_fn"] = "relu"#silu?gelu?
new_config["mid_block_scale_factor"]= 1.2 

new_unet = UNet2DConditionModel(**new_config)
# cust to fp16
new_unet.to(torch.float16)

num_params = sum(p.numel() for p in new_unet.parameters())
print(f"Количество параметров: {num_params}")

save = True
if save:
    # Сохранение модели в формате diffusers
    save_directory = "/home/recoilme/models/waifu-4b/unet"
    new_unet.save_pretrained(save_directory, safe_serialization=True)#variant= "fp16")

    print(f"Модель успешно сохранена в {save_directory}")

    print(new_unet)
    #Количество параметров: 2567463684
    #Количество параметров: 3217997776
    # 2845985296
    #3884771856

###
#optimizer = Lion(  # Альтернатива AdamW
#    model.parameters(), 
#    lr=1e-5, 
#    weight_decay=0.01
#)
#scheduler = torch.optim.lr_scheduler.OneCycleLR(
#    optimizer, 
#    max_lr=1e-4,
#    total_steps=10000
#)
#optimizer_params = [
#    {"params": down_blocks.parameters(), "lr": base_lr},
#    {"params": mid_blocks.parameters(), "lr": base_lr * 2},
#    {"params": up_blocks.parameters(), "lr": base_lr * 3}
#]
#scaler = torch.cuda.amp.GradScaler(
#    init_scale=2**14,  # Оптимально для SD-XL
#    growth_interval=200
#)
###