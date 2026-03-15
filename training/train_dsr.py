import torch
import torch.nn.functional as F


def train_dsr(ae, dsr, H_star, epochs=20, lr=1e-3):
    optimizer = torch.optim.Adam(
        list(ae.parameters()) + list(dsr.parameters()),
        lr=lr
    )

    for epoch in range(epochs):
        optimizer.zero_grad()

        H_latent = ae.encoder(H_star)
        H_rec, loss_sr = dsr(H_latent)
        ae_loss = F.mse_loss(ae.decoder(H_latent), H_star)

        loss = loss_sr + ae_loss
        loss.backward()
        optimizer.step()

        if epoch == 0 or epoch == epochs - 1:
            print(f"Epoch {epoch+1}/{epochs} Loss {loss.item():.4f}")

    return dsr