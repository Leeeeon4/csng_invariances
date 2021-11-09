    """Concept: 
    1. preprocess images if necessary.
    _, c, w, h = images.shape
    2. src_image = torch.zeros(1, c, w, h, requires_grad=True, device=device)


    """

import torch

def meis(model, images, responses, neurons_to_select, octaves, seed=1, **kwargs):

    device = torch.device("cuda" if next(model.parameters()).is_cuda else "cpu")

    _, c, w, h = images.shape
    image_count, neuron_count = responses.shape
    selected_neurons_count = neurons_to_select.shape
    white_noise_image_tensor = torch.randint(1, c, w, h, requires_grad=True, device=device, seed=seed)
    # TODO should we always use the same white noise tensor to start?
    # TODO normalize this tensor

# TODO compute meis for selected neurons: 
# pick gradient with respect to selected neuron
# for neuron in range(selected neuron coutn): 
#   current mei = compute mei (neuron, white_noise_image_tensor) 
#   meis[neuron,:,:,:] = current mei

    for e, o in enumerate(octaves):
        for i in range(o['iter_n']):
            ox = 0
            oy = 0
            src.data[0].copy_(torch.Tensor(image))

            sigma = o['start_sigma'] + ((o['end_sigma'] - o['start_sigma']) * i) / o['iter_n']
            step_size = o['start_step_size'] + ((o['end_step_size'] - o['start_step_size']) * i) / o['iter_n']

            make_step(net, src, bias=bias, scale=scale, sigma=sigma, step_size=step_size, **step_params)

            if i % 10 == 0:
                print('finished step %d in octave %d' % (i, e))

            # insert modified image back into original image (if necessary)
            image[:, ox:ox + w, oy:oy + h] = src.data[0].cpu().numpy()

    # returning the resulting image
    return unprocess(image, mu=bias, sigma=scale)


def process(x, mu=0.4, sigma=0.224):
    """ Normalize and move channel dim in front of height and width"""
    x = (x - mu) / sigma
    if isinstance(x, torch.Tensor):
        return x.transpose(-1, -2).transpose(-2, -3)
    else:
        return np.moveaxis(x, -1, -3)

def unprocess(x, mu=0.4, sigma=0.224):
    """Inverse of process()"""
    x = x * sigma + mu
    if isinstance(x, torch.Tensor):
        return x.transpose(-3, -2).transpose(-2, -1)
    else:
        return np.moveaxis(x, -3, -1)