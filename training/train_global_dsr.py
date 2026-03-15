from models.autoencoder import RNADeepAE
from models.deep_self_representation import DeepSelfRepresentation
from training.train_dsr import train_dsr


def train_global_dsr(kb_embeddings, config, device):
    input_dim = kb_embeddings.shape[1]
    latent_dim = config["autoencoder"]["latent_dim"]

    ae = RNADeepAE(
        input_dim=input_dim,
        latent_dim=latent_dim
    ).to(device)

    dsr = DeepSelfRepresentation(
        n_samples=kb_embeddings.shape[0],
        latent_dim=latent_dim
    ).to(device)

    dsr = train_dsr(
        ae,
        dsr,
        kb_embeddings,
        epochs=config["training"]["epochs"],
        lr=config["training"]["lr"]
    )

    return ae, dsr
    

    return ae, dsr