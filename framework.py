import torchvision.datasets.mnist as mnist
import torchvision.transforms as tf
import torch.utils.data as data
import torch
import torch.optim.optimizer as optimizer
import numpy as np
from torch import nn
from torch.nn import functional as F
import os
import yaml
import wandb
import matplotlib.pyplot as plt
from functools import partial


def convert_score_to_label(scores, is_logit=True, threshold=0.5) -> torch.Tensor:
    if is_logit:
        p = F.sigmoid(scores)  # convert log(p) back into probabilty range [0, 1] using sigmoid function
    else:
        p = scores
    if threshold:
        p[p > threshold] = 1  # set it to either True or False based on threshold
        p[p < threshold] = 0
        p = p.type(torch.bool)
    return p


def convert_tensor_to_image(x: torch.Tensor) -> np.ndarray:
    # remember that we are outputsing tensors of shape [bs, 784]
    # in order to have them displayed as images we have to reshape them to the disared image format
    x = x.view(-1, 28, 28, 1) # numpy follows the format [bs, H, W, C] whereas torch is of format [bs, C, H, W]
                              # hence the reshape into (-1, 28, 28, 1) instead of (-1, 1, 28, 28)
    x = x.cpu().numpy()
    return x


def create_grid_plot(images, labels) -> plt.Figure:
    # we create a plut with 16 subplots (nrows * ncols)
    fig, axes = plt.subplots(nrows=4, ncols=4, figsize=(56, 56))
    # we slices as many images as subplots we have
    images = convert_tensor_to_image(images)[:4 * 4]
    values = convert_score_to_label(labels)  # this line can be skipped if you want to display the logit
                                             # alternatively if you want to see p(x) instead of the logit
                                             # replace it with the line bellow
    # values = F.sigmoid(logits)
    for idx, image in enumerate(images):
        # we compute our current position in the subplot
        row = idx // 4
        col = idx % 4
        axes[row, col].axis("off")
        axes[row, col].set_title(str(values[idx].item()), size=72) # we plot the image label in the title field
                                                                   # such that each subplot will display its individual labeled value
        axes[row, col].imshow(image, cmap="gray", aspect="auto")
    return fig


def get_activation_function(type: str) -> partial:
    if type == 'relu':
        return partial(nn.ReLU, inplace=True)
    elif type == 'lrelu':
        return partial(nn.LeakyReLU, negative_slope=0.2, inplace=True)
    elif type == 'sigmoid':
        return partial(nn.Sigmoid, inplace=True)
    elif type == 'tanh':
        return partial(nn.Tanh, inplace=True)
    else:
        return partial(IdentityLayer)


def get_normalization_function(type: str) -> partial:
    if type == 'batch':
        # be careful when using batch_norm as it does not work on a batch_size of 1
        return partial(nn.BatchNorm1d, affine=True, track_running_stats=True)
    elif type == 'layer':
        return partial(nn.LayerNorm)
    else:
        return partial(IdentityLayer)


class IdentityLayer(nn.Module):
    def __init__(self, *args):
        super(IdentityLayer, self).__init__()

    def forward(self, x) -> torch.Tensor:
        return x


class Generator(nn.Module):
    def __init__(self, dist_latent, cfg):
        super(Generator, self).__init__()
        self.dist_latent = dist_latent
        self.dim_latent = cfg['g']['dim_latent']
        self.dim_hidden = cfg['g']['dim_hidden']
        self.act = get_activation_function(cfg['g']['act'])
        self.norm = get_normalization_function(cfg['g']['norm'])

        __module_list = [
            nn.Linear(self.dim_latent, self.dim_hidden, bias=True),
            self.norm(self.dim_hidden),
            self.act(),
            nn.Linear(self.dim_hidden, self.dim_hidden * 2, bias=True),
            self.norm(self.dim_hidden * 2),
            self.act(),
            nn.Linear(self.dim_hidden * 2, self.dim_hidden * 4, bias=True),
            self.norm(self.dim_hidden * 4),
            self.act(),
            nn.Linear(self.dim_hidden * 4, 28 * 28, bias=True),
        ]

        self.__net = nn.Sequential(*__module_list)

    def forward(self, z):
        x = self.__net(z)
        return x

    def sample(self, num_samples) -> torch.Tensor:
        # torch.distributions are objects that currently do not support GPU migration
        # alternatives such as torch.randn / torch.rand generate a tensor that still needs to be manually moved on to the gpu
        z = self.dist_latent.sample(sample_shape=(num_samples, self.dim_latent))
        return z.to(self.device)

    def generate_visual_sample(self, num_samples=4):
        z = self.sample(num_samples)
        x = self.forward(z)
        return x

    @property
    def device(self):
        # remember this neat little trick
        # useful for when dealing with parallel/distributed training
        # torch replicates the initialised model across all devices which means
        # that if we store the device ID in self.device when a replica will be placed on another GPU
        # that replica will have its initial device_id
        return next(self.parameters()).device


class Discriminator(nn.Module):
    def __init__(self, dim_input, dim_output, cfg):
        super(Discriminator, self).__init__()
        self.dim_input = dim_input
        self.dim_output = dim_output
        self.act = get_activation_function(cfg['d']['act'])
        self.norm = get_normalization_function(cfg['d']['norm'])

        __module_list = [
            nn.Linear(dim_input, int(dim_input / 2), bias=True),
            self.norm(int(dim_input / 2)),
            self.act(),
            nn.Linear(int(dim_input / 2), int(dim_input / 4), bias=True),
            self.norm(int(dim_input / 4)),
            self.act(),
            nn.Linear(int(dim_input / 4), int(dim_input / 8), bias=True),
            self.norm(int(dim_input / 8)),
            self.act(),
            nn.Linear(int(dim_input / 8), dim_output), # remember that most often than not GANs will include a non-linearity in their
                                                       # training objective and as such we want to have a linear as our final layer in D
                                                       # this is because, from an analytic standpoint linear transformations are not
                                                       # bounded function such as activation functions, as such we can have far more
                                                       # expressiveness in our D behaviour
                                                       # i.e: ReLU([-inf, inf]) -> [0, inf]
                                                       #      Sigmoid(ReLU([-inf, inf]) -> Sigmoid([0, inf]) -> [0.5, 1]
        ]

        self.__net = nn.Sequential(*__module_list)

    def forward(self, x) -> torch.Tensor:
        return self.__net(x)

    @property
    def device(self):
        return next(self.parameters()).device


class GANLoss(nn.Module):
    def __init__(self):
        super(GANLoss, self).__init__()

    def __call__(self, pred, is_real=True, for_discriminator=True) -> torch.Tensor:
        pass


class GANBCEWithLogitsLoss(GANLoss):
    def __init__(self):
        super(GANBCEWithLogitsLoss, self).__init__()
        self.__loss = nn.BCEWithLogitsLoss()
        self.input_shape = 1

    def __get_target_tensor(self, values, is_real):
        target = torch.tensor(int(is_real))  # we compute the target value given is_real
        return target.expand_as(values).type_as(values)  # torch losses require target tensors that have the same shape
        # as the prediction tensor, even if the target is a singular value
        # i.e: 1 or 0, torch losses cannot broadcast tensors
        #
        # torch.abs(torch.ones(size=(1, 4)) - 1) IS A VALID OPERATION
        # however we cannot have  nn.L1Loss(torch.ones(size=(1, 4), 1)
        # even if the two opperations yield the same results from a theoretical standpoint

    def __call__(self, x, is_real=True, for_discriminator=True) -> torch.Tensor:
        # avoiding control flows is a desirable but not mandatory
        # not having control flows allows for model compilation which can sometimes
        # greatly decrease runtimes
        target = self.__get_target_tensor(x, is_real)
        return self.__loss(x, target)


class GANLSELoss(GANLoss):
    def __init__(self):
        super(GANLSELoss, self).__init__()
        self.__loss = nn.MSELoss()
        self.input_shape = 1

    def __get_target_tensor(self, values, is_real):
        # hint, 0 and 1 work just as fine as target labels
        # but is it possible to have them as -1 and 1 without control flow?
        pass

    def __cal__(self, x, is_real, for_discriminator=True) -> torch.Tensor:
        pass

def test_criterion(c: GANLoss):
    if isinstance(c, GANBCEWithLogitsLoss):
        ref_criterion = nn.BCELoss()  # we test against classic BCE loss
        pred_real = torch.log(torch.ones(size=(4, 1)))  # assume that p(real) = 1 -> we use log(p(real)) as D computes log(p)
        pred_fake = torch.log(torch.zeros(size=(4, 1)) + 1e-32)  # assume that p(fake) = 0, we add 1e-32 for numerical stability

        loss_real = c(pred_real, is_real=True)
        loss_fake = c(pred_fake, is_real=False)

        loss_ref_real = ref_criterion(torch.sigmoid(pred_real),
                                      torch.ones_like(pred_real))  # we got back to probability domain by using sigmoid
        loss_ref_fake = ref_criterion(torch.sigmoid(pred_fake),
                                      torch.zeros_like(pred_fake))  # sigmoid [-inf, inf] -> [0, 1]

        # we use allclose to compare within epsilon, due to hardware dependent implementations
        # we can never obtain exact analytic solutions
        # i.e: log(0) -> is undefined
        assert torch.allclose(loss_real, loss_ref_real, rtol=1e-8), 'BCEWithLogits failed for real case'
        assert torch.allclose(loss_fake, loss_ref_fake, rtol=1e-8), 'BCEWithLogits failed for fake case'
    else:
        pass


def test_generator(g: Generator, cfg):
    z = g.sample(4)
    assert z.size() == torch.Size((4, cfg['g']['dim_latent'])), 'failed basic sample test'
    # x = g.forward(z)
    x = g(z)
    assert x.size() == torch.Size((4, 28 * 28)), 'failed forward test'
    y = g.generate_visual_sample()
    assert y.size() == x.size(), 'failed visual generartion test'


def test_discriminator(d: Discriminator, cfg):
    y = torch.randn(size=(4, 784)).to(d.device)
    log_p = d(y)
    assert log_p.size() == torch.Size((4, 1)), 'failed discriminator forward test'


def train_g(g: Generator, d: Discriminator, criterion: GANLoss, cfg) -> dict:
    g_samples = g.generate_visual_sample(cfg['bs'])  # we generate samples with G
    log_p_g_samples = d(g_samples)  # we evaluate log(p(G))
    loss_g = criterion(log_p_g_samples, is_real=True)  # we want p(G) -> 1 so we eval the loss as if the samples should be real
    loss_g.backward()
    return {'g_samples': g_samples, 'loss': loss_g}


def train_d(g: Generator, d: Discriminator, criterion: GANLoss, real: torch.Tensor, cfg) -> dict:
    fake_samples = g.generate_visual_sample(cfg['bs']).detach()  # we detach gradient computation from G samples
    # .detach() can be removed if we disable G's gradients before
    # sample generations
    # i.e
    # the line of code bellow is equivalent if: g.requires_grad_(False), g.eval() are set beforehand
    # fake_samples = g.generate_visual_sample(cfg['bs'])
    log_p_fake_samples = d(fake_samples)
    # this dataset in particular returns input tensor of shape [bs, 1, 28, 28]
    # therefore we reshape them to [-1, 28 * 28], where 28*28 is the input size of the first layer in D
    # and -1 means that batch shaped should be inferred.
    # we infer the batch shape because the dataloader can return batches smaller than batch size
    # if len(dataloader) % bs != 0
    log_p_real_samples = d(real.view(-1, 28 * 28))
    # when evaluating the loss for D we set that p(G) -> 0, and p(real) -> 1
    loss_real = criterion(log_p_real_samples, is_real=True)
    loss_fake = criterion(log_p_fake_samples, is_real=False)
    # the final loss is the mean of the two losses as we want to keep the same gradient magnitude in both G and D
    loss_d = loss_real * 0.5 + loss_fake * 0.5
    loss_d.backward()
    return {'fake_samples': fake_samples, 'logits_fake': log_p_fake_samples, 'logits_real': log_p_real_samples,
            'loss_real': loss_real, 'loss_fake': loss_fake, 'loss': loss_d}


def train_system(g: Generator, d: Discriminator, c: GANLoss, opt_g: optimizer.Optimizer, opt_d: optimizer.Optimizer, real, cfg):
    # train g first
    g.zero_grad()  # we zero out gradients for G
    g.requires_grad_(True)  # we enable gradient computation for G
    g.train()  # we enable training behaviour for G
    # some layers in pytorch have different behaviours when called
    # with train(), or eval()
    # evidently, when optimising the layer we want it to have its train() behaviour

    d.requires_grad_(requires_grad=False)  # we disable gradient computation throught D
    d.eval()  # we set D to eval as theoretically we train G with respect to an optimised G

    # we call the training procedure for g for g_iter times
    for _ in range(cfg['g_iter']):
        results_g = train_g(g, d, c, cfg)
        opt_g.step()

    # train d
    # we perform the opposite gradient and eval rules when optimising D
    d.zero_grad()
    d.requires_grad_(True)
    d.train()

    g.requires_grad_(requires_grad=False)
    g.eval()

    for _ in range(cfg['d_iter']):
        # we train D with the same batch of real samples against different batches of samples generated by G
        # BE CAREFUL: this can be a double edged sword, if 'd_iter' is set too high this can lead to overfitting
        # of D on ilogical patterns in the real data
        # TODO: zero grad
        results_d = train_d(g, d, c, real, cfg)
        opt_d.step()
    return results_g, results_d


def run(g: Generator, d: Discriminator, c: GANLoss, cfg):
    global save_path

    dataset = mnist.FashionMNIST(root='datasets/tutorials', transform=tf.Compose([tf.ToTensor()]), download=True)
    dataloader = torch.utils.data.DataLoader(dataset, num_workers=8, pin_memory=True, batch_size=cfg['bs'])
    opt_g = torch.optim.Adam(params=list(g.parameters()), lr=cfg['lr'] / 2, amsgrad=True)
    opt_d = torch.optim.Adam(params=list(d.parameters()), lr=cfg['lr'] * 2, amsgrad=True)

    # z <- constant
    # g(z) -> monitor if training improves
    eval_latent = g.sample(num_samples=cfg['bs'])

    step = 0
    for _ in range(cfg['num_epochs']):
        for sample in dataloader:
            step += cfg['bs']  # when monitoring we are usually interested in how many examples our network has seen
            # broadly gradient based methods perform with respect to how many examples they have seen
            # not necessarily how many optimization steps they have performed

            sample = sample[0].to(d.device)  # sample [0] <- input, #sample [1] <- label in the case of FashionMNIST
            # real sample structure will vary depending on what kind of dataset you are using

            results_g, results_d = train_system(g, d, c, opt_g, opt_d, sample, cfg)

            # weights and biases works by passing a dictionary where the key represents what we are logging
            # and the values is the value of the logged key at t=step
            if step % cfg['loss_display_freq'] == 0:
                wandb.log({'loss_g': results_g['loss'], 'loss_d': results_d['loss']}, step=step)
            if step % cfg['visual_display_freq'] == 0:
                # we extract and convert values to graphically displayable formats
                real_samples = sample
                fake_samples = results_d['fake_samples']
                logits_real = results_d['logits_real']
                logits_fake = results_d['logits_fake']
                # a convenient way to plot image plots or customs plots is by passing at matplotlib/plotly figure directly to wandb
                # we compute the matplotlib figure and pass it in a simple dict
                fake_grid = create_grid_plot(fake_samples, logits_fake)
                wandb.log({'fake samples': fake_grid}, step=step)
                # if working with matploblib/plotly
                # REMEMBER TO CLOSE FIGURES, wandb does not do this by default and in time
                # it can lead to memory blowup
                plt.close(fake_grid)
                real_grid = create_grid_plot(real_samples, logits_real)
                wandb.log({'real samples': real_grid}, step=step)
            if step % cfg['eval_display_freq'] == 0:
                # when evaluating we pass the initial latent code to G to map it into image space
                eval_visual = g.forward(eval_latent.to(g.device))
                eval_logits = d(eval_visual)
                eval_grid = create_grid_plot(eval_visual, eval_logits)
                wandb.log({'eval samples': eval_grid}, step=step)
            if step % cfg['save_freq'] == 0:
                # we need to recall the device in which our models currently reside
                # and move them to CPU before writting them to disk
                g_device = g.device
                d_device = d.device
                # when working with GANs it is a good rule of thumbs to save the model at different timesteps
                # since the GAN behaviour does not analytically allow for strong convergence, a prior iteration
                # of the system might yield more desirable results
                torch.save(g.cpu(), os.path.join(save_path, 'g_model' + '_{}'.format(step)))
                torch.save(d.cpu(), os.path.join(save_path, 'd_model' + '_{}'.format(step)))
                # we move the model back to their original devices
                g = g.to(g_device)
                d = d.to(d_device)


if __name__ == '__main__':
    # for reproductibility issues
    torch.random.manual_seed(42)  # seed can be anything

    with open('config.yml', 'r') as f:
        config = yaml.load(f)
    run_name = str(config.values())

    save_path = os.path.join('ckpt', run_name)
    if not os.path.isdir(save_path):
        os.mkdir(save_path)

    wandb.init(config=config, project='gan-tut', name=run_name)

    if config['device_id'] != 'cpu':
        device = torch.device('cuda:{}'.format(config['device_id']))
    else:
        device = torch.device('cpu')

    dist_latent = torch.distributions.normal.Normal(loc=0, scale=1)  # Gaussian(0, 1)
    c = GANBCEWithLogitsLoss().to(device)
    g = Generator(dist_latent, config).to(device)
    d = Discriminator(dim_input=28 * 28, dim_output=c.input_shape, cfg=config).to(device)

    test_criterion(c)
    test_generator(g, config)
    test_discriminator(d, config)

    run(g, d, c, config)