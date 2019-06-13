
GAN = 0
GANCNN = 1


# Stopping tolerance
tol = 1e-8
min_lr = 1e-8
epsilon = 1e-8
SAVE_EPOCH=20

def get_model_name(model, config):
    if model=='GAN' or model=='GANCNN':
        return get_model_name_AE(model, config)
        
def get_model_name_AE(model, config):
    model_name = model + '_' \
                 + config.dataset_name+ '_'  \
                 + 'latent_dim' + str(config.latent_dim) + '_' \
                 + 'h_dim' + str(config.hidden_dim)  + '_' \
                 + 'h_nl' + str(config.num_layers)
    return model_name
