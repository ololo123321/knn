import tqdm
import torch
import torch.distributed as dist
import numpy as np


def torch_put_2d(x, indices, values):
    """
    torch 2d analogue of np.add.at

    x - [N, D]
    indices - [M]
    values - [M, D]

    return: [N, D]
    """
    values_flat = values.flatten()
    d = x.shape[1]
    indices_flat = (indices[:, None] * d + torch.arange(d, device=indices.device)[None, :]).flatten()
    x.put_(indices_flat, values_flat, accumulate=True)
    return x


if __name__ == "__main__":
    # process group
    dist.init_process_group(backend="nccl", init_method="env://")
    local_rank = dist.get_rank()
    world_size = dist.get_world_size()
    device = torch.device("cuda", local_rank)
    torch.cuda.set_device(device)

    def log(text):
        if local_rank == 0:
            print(text)

    # load train data
    log("generating data...")
    n = 1000000
    d = 768
    x = torch.rand((n, d)).half()

    # start training
    # nc = 1048576
    # nc = 131072
    nc = 1024
    batch_size = 512
    ninit = 1
    niter = 10
    best_score = float("inf")
    best_centroids = None
    for i_init in range(ninit):
        log(f"Initialization try: {i_init}")
        idx = torch.randperm(n)[:nc]
        centroids = x[idx].clone().to(device)
        loss_it = float("inf")
        for i_iter in range(niter):
            log(f"Iteration: {i_iter}")
            dist.broadcast(centroids, 0)
            sums = torch.zeros_like(centroids).float()  # ensure fp32
            counts = torch.zeros(nc, dtype=torch.long, device=device)
            loss = torch.zeros(nc, device=device)  # fp32
            norm_c = torch.square(centroids).sum(1)[None, :]  # [1, nc]
            centroids_t = centroids.transpose(0, 1).half()  # [d, nc], fp16 for faster matmul
            for start in tqdm.trange(0, n, batch_size, position=local_rank, desc=f"rank {local_rank}"):
                end = min(n, start + batch_size)
                xb = x[start:end].to(device)
                s = xb @ centroids_t  # [b, nc]
                norm_x = torch.square(xb).sum(1)[:, None]  # [b, 1]
                dists = norm_x + norm_c - 2.0 * s  # [b, nc]
                dists = torch.maximum(torch.tensor(0.0, device=device), dists)  # [b, nc]
                dists = torch.sqrt(dists)  # [b, nc]
                m = dists.min(1)  # [b], tuple (values, indices)
                torch_put_2d(sums, m.indices, xb.float())
                counts.put_(m.indices, torch.ones_like(m.indices), accumulate=True)
                loss.put_(m.indices, m.values.float(), accumulate=True)
            loss = (loss / (counts.float() + 1e-8)).mean()
            dist.barrier()
            dist.reduce(sums, 0, op=dist.ReduceOp.SUM)
            dist.reduce(counts, 0, op=dist.ReduceOp.SUM)
            dist.reduce(loss, 0, op=dist.ReduceOp.SUM)
            centroids = sums / counts[:, None].float()
            loss_it = (loss / world_size).item()
            log(f"loss: {loss_it}")
            dist.barrier()  # TODO: мб лишний
        if loss_it < best_score:
            best_centroids = centroids
    if local_rank == 0:
        log("save centroids")
        np.save("centroids.npy", best_centroids.to("cpu").numpy())
