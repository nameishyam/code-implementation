def load_model(model_name, config):
    if model_name == 'srgan_custom':
        from models.srgan_custom.generator import Generator
        model = Generator(config)
    elif model_name == 'srgan_baseline':
        from models.srgan_baseline.generator import Generator
        model = Generator()
    elif model_name == 'srcnn':
        from models.srcnn.srcnn import SRCNN
        model = SRCNN()
    elif model_name == 'esrgan':
        from models.esrgan.esrgan import ESRGAN
        model = ESRGAN()
    elif model_name == 'bilinear':
        from models.interpolation.bilinear import BilinearUpsample
        model = BilinearUpsample()
    else:
        raise ValueError(f"Unsupported model: {model_name}")
    return model
