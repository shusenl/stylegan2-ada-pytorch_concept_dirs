# Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

"""Generate images using pretrained network pickle."""

import os
import re
from typing import List, Optional

import click
import dnnlib
import numpy as np
import PIL.Image
import torch

import legacy

#----------------------------------------------------------------------------

def num_range(s: str) -> List[int]:
    '''Accept either a comma separated list of numbers 'a,b,c' or a range 'a-c' and return as a list of ints.'''

    range_re = re.compile(r'^(\d+)-(\d+)$')
    m = range_re.match(s)
    if m:
        return list(range(int(m.group(1)), int(m.group(2))+1))
    vals = s.split(',')
    return [int(x) for x in vals]

#----------------------------------------------------------------------------

@click.command()
@click.pass_context
@click.option('--network', 'network_pkl', help='Network pickle filename', required=True)
@click.option('--seeds', type=num_range, help='List of random seeds')
@click.option('--trunc', 'truncation_psi', type=float, help='Truncation psi', default=1, show_default=True)
@click.option('--class', 'class_idx', type=int, help='Class label (unconditional if not specified)')
@click.option('--noise-mode', help='Noise mode', type=click.Choice(['const', 'random', 'none']), default='const', show_default=True)
@click.option('--projected-w', help='Projection result file', type=str, metavar='FILE')
@click.option('--outdir', help='Where to save the output images', type=str, required=True, metavar='DIR')

#### newly added ###
@click.option('--walk_directions', help='file stores the salient direction', type=str, metavar='FILE')
@click.option('--gen_w', help='out file for w samples', type=str, metavar='DIR', default=None)

def generate_images(
    ctx: click.Context,
    network_pkl: str,
    seeds: Optional[List[int]],
    truncation_psi: float,
    noise_mode: str,
    outdir: str,
    class_idx: Optional[int],
    projected_w: Optional[str],
    walk_directions: Optional[str],
    gen_w: Optional[str]
):
    """Generate images using pretrained network pickle.

    Examples:

    \b
    # Generate curated MetFaces images without truncation (Fig.10 left)
    python generate.py --outdir=out --trunc=1 --seeds=85,265,297,849 \\
        --network=https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/pretrained/metfaces.pkl

    \b
    # Generate uncurated MetFaces images with truncation (Fig.12 upper left)
    python generate.py --outdir=out --trunc=0.7 --seeds=600-605 \\
        --network=https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/pretrained/metfaces.pkl

    \b
    # Generate class conditional CIFAR-10 images (Fig.17 left, Car)
    python generate.py --outdir=out --seeds=0-35 --class=1 \\
        --network=https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/pretrained/cifar10.pkl

    \b
    # Generate latent code
    python generate.py --outdir=out --gen_w latent_sample_20M_ffhq.pt --seeds=1-200 --network=https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/pretrained/ffhq.pkl

    \b
    # Render an image from projected W
    python generate.py --outdir=./ --projected_w=projected_w.npz \\
        --network=https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/pretrained/ffhq.pkl
        
    """

    print('Loading networks from "%s"...' % network_pkl)
    device = torch.device('cuda')
    with dnnlib.util.open_url(network_pkl) as f:
        G = legacy.load_network_pkl(f)['G_ema'].to(device) # type: ignore

    os.makedirs(outdir, exist_ok=True)

    # Synthesize the result of a W projection.
    if projected_w is not None:
        if seeds is not None:
            print ('warn: --seeds is ignored when using --projected-w')
        print(f'Generating images from projected W "{projected_w}"')
        ws = np.load(projected_w)['w']
        ws = torch.tensor(ws, device=device) # pylint: disable=not-callable
        assert ws.shape[1:] == (G.num_ws, G.w_dim)
        for idx, w in enumerate(ws):
            img = G.synthesis(w.unsqueeze(0), noise_mode=noise_mode)
            img = (img.permute(0, 2, 3, 1) * 127.5 + 128).clamp(0, 255).to(torch.uint8)
            img = PIL.Image.fromarray(img[0].cpu().numpy(), 'RGB').save(f'{outdir}/proj{idx:02d}.png')
        return

    # Labels.
    label = torch.zeros([1, G.c_dim], device=device)
    if G.c_dim != 0:
        if class_idx is None:
            ctx.fail('Must specify class label with --class when using a conditional network')
        label[:, class_idx] = 1
    else:
        if class_idx is not None:
            print ('warn: --class=lbl ignored when running on an unconditional network')


    if walk_directions is not None:
        if seeds is not None:
            ws_dir = torch.load(walk_directions).to(device)
            print("G.num_ws:", G.num_ws)
            assert ws_dir.shape[1:] == (G.num_ws, G.w_dim)
            
            for seed_idx, seed in enumerate(seeds):
                    print('Generating image for seed %d (%d/%d) ...' % (seed, seed_idx, len(seeds)))
                    z = torch.from_numpy(np.random.RandomState(seed).randn(1, G.z_dim)).to(device)
                    ws = G.mapping(z, label, truncation_psi=truncation_psi, truncation_cutoff=None)
                    img = G.synthesis(ws, noise_mode=noise_mode)
                    img = (img.permute(0, 2, 3, 1) * 127.5 + 128).clamp(0, 255).to(torch.uint8)
                    if img[0].cpu().numpy().shape[-1]==1:
                        PIL.Image.fromarray(np.squeeze(img[0].cpu().numpy()), 'L').save(f'{outdir}/walk_seed{seed:04d}.png')
                    else: 
                        PIL.Image.fromarray(img[0].cpu().numpy(), 'RGB').save(f'{outdir}/walk_seed{seed:04d}.png')
                    print("generated original image ...")
                    # for each direction
                    for idx, w_dir in enumerate(ws_dir):
                        print("w_dir:", idx, w_dir.shape)
                        if idx>15:
                            continue
                        # breakpoint
                        ws_p = ws + 1.0*w_dir
                        ws_m = ws - 1.0*w_dir
                        img = G.synthesis(ws_p, noise_mode=noise_mode)
                        img = (img.permute(0, 2, 3, 1) * 127.5 + 128).clamp(0, 255).to(torch.uint8)
                        if img[0].cpu().numpy().shape[-1]==1:
                            PIL.Image.fromarray(np.squeeze(img[0].cpu().numpy()), 'L').save(f'{outdir}/walk_seed{seed:04d}_dir{idx}.png')
                        else:
                            PIL.Image.fromarray(img[0].cpu().numpy(), 'RGB').save(f'{outdir}/walk_seed{seed:04d}_dir{idx}.png')

                        negative_dir = False
                        if negative_dir:
                            img = G.synthesis(ws_m, noise_mode=noise_mode)
                            img = (img.permute(0, 2, 3, 1) * 127.5 + 128).clamp(0, 255).to(torch.uint8)
                            if img[0].cpu().numpy().shape[-1]==1:
                                PIL.Image.fromarray(np.squeeze(img[0].cpu().numpy()), 'L').save(f'{outdir}/walk_seed{seed:04d}_m_dir{idx}.png')
                            else:
                                PIL.Image.fromarray(img[0].cpu().numpy(), 'RGB').save(f'{outdir}/walk_seed{seed:04d}_m_dir{idx}.png')

            # for idx, w in enumerate(ws):
            #     img = G.synthesis(w.unsqueeze(0), noise_mode=noise_mode)
            #     img = (img.permute(0, 2, 3, 1) * 127.5 + 128).clamp(0, 255).to(torch.uint8)
            #     img = PIL.Image.fromarray(img[0].cpu().numpy(), 'RGB').save(f'{outdir}/proj{idx:02d}.png')
            # return

    if seeds is None:
        ctx.fail('--seeds option is required when not using --projected-w')

    # Generate latent.
    if gen_w is not None:
        ws_list = []
        for seed_idx, seed in enumerate(seeds):            
            z = torch.from_numpy(np.random.RandomState(seed).randn(100000, G.z_dim)).to(device)  
            # z = torch.from_numpy(np.random.RandomState(0).randn(50, G.z_dim)).to(device)  
            # mapping(z, c, truncation_psi=truncation_psi, truncation_cutoff=truncation_cutoff)   
            ws = G.mapping(z, label, truncation_psi=truncation_psi, truncation_cutoff=None)[:,0,:].detach().cpu()
            ws_list.append(ws) 
            print(seed_idx, ws.shape)
        full_ws = torch.cat(ws_list, axis=0)
        print("full ws", full_ws.shape)
        
        torch.save(full_ws, gen_w)
        return 

    # Generate images.
    for seed_idx, seed in enumerate(seeds):
        print('Generating image for seed %d (%d/%d) ...' % (seed, seed_idx, len(seeds)))
        z = torch.from_numpy(np.random.RandomState(seed).randn(1, G.z_dim)).to(device)
        img = G(z, label, truncation_psi=truncation_psi, noise_mode=noise_mode)
        img = (img.permute(0, 2, 3, 1) * 127.5 + 128).clamp(0, 255).to(torch.uint8)
        print(img[0].size())
        # PIL.Image.fromarray(img[0].cpu().numpy(), 'RGB').save(f'{outdir}/seed{seed:04d}.png')
        print("image size:", img[0].cpu().numpy().shape)
        if img[0].cpu().numpy().shape[-1]==1:
            PIL.Image.fromarray(np.squeeze(img[0].cpu().numpy()), 'L').save(f'{outdir}/seed{seed:04d}.png')
        else:
            PIL.Image.fromarray(img[0].cpu().numpy(), 'RGB').save(f'{outdir}/seed{seed:04d}.png')


#----------------------------------------------------------------------------

if __name__ == "__main__":
    generate_images() # pylint: disable=no-value-for-parameter

#----------------------------------------------------------------------------
