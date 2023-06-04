import matplotlib
matplotlib.use('Agg')

import time
from easydict import EasyDict

import torch

import lfigw.ffjord.lib.utils as utils
from lfigw.ffjord.train_misc import standard_normal_logprob
from lfigw.ffjord.train_misc import set_cnf_options, count_nfe, count_parameters, count_total_time
from lfigw.ffjord.train_misc import add_spectral_norm
from lfigw.ffjord.train_misc import build_condition_model_tabular

def get_transforms(model):

    def sample_fn(z, context=None, logpz=None):
        if context is not None:
            if logpz is not None:
                return model(z, context, logpz, reverse=True)
            else:
                return model(z, context, reverse=True)
        else:
            if logpz is not None:
                return model(z, logpz, reverse=True)
            else:
                return model(z, reverse=True)

    def density_fn(x, context=None, logpx=None):
        if context is not None:
            if logpx is not None:
                return model(x, context, logpx, reverse=False)
            else:
                return model(x, context, reverse=False)
        else:
            if logpx is not None:
                return model(x, logpx, reverse=False)
            else:
                return model(x, reverse=False)
            
    return sample_fn, density_fn


def create_ffjord_model(input_dim, 
                        context_dim, 
                        **kwargs
                        ):
    """Build conditioned Ffjord model.

    This models the posterior distribution p(x|y).

    The model consists of
        * a base distribution (StandardNormal, dim(x))
        * a sequence of transforms, each conditioned on y

    Arguments:
        input_dim {int} -- dimensionality of x
        context_dim {int} -- dimensionality of y

    Returns:
        Flow -- the model
    """

    default_config=dict(
        JFrobint=None,
        JdiagFrobint=None,
        JoffdiagFrobint=None,
        l1int=None,
        l2int=None,
        dl2int=None,

        atol=1e-05,
        rtol=1e-05,

        batch_norm=False,
        bn_lag=0,
        dims='64-64-64',
        divergence_fn='brute_force',
        # divergence_fn='approximate',
        
        # gpu=0,
        layer_type='concatsquash',
        # log_freq=10,
        # lr=0.001,
        # niters=10000,
        nonlinearity='tanh',
        num_blocks=1,
        rademacher=False,
        residual=False,

        solver='dopri5',
        step_size=None,
        spectral_norm=False,
        test_atol=None,
        test_rtol=None,
        test_solver=None,
        time_length=0.5,
        train_T=True,
        # val_freq=100,
        # viz_freq=100,
        weight_decay=1e-05,
    )

    config=default_config
    config.update({
        "input_dim":input_dim,
        "context_dim":context_dim,
    })

    config.update(**kwargs)
    easydict_config=EasyDict(config)

    model = build_condition_model_tabular(easydict_config, input_dim, context_dim)
    if easydict_config.spectral_norm: add_spectral_norm(model)
    set_cnf_options(easydict_config, model)

    print(model)
    print("Number of trainable parameters: {}".format(count_parameters(model)))

    # Store hyperparameters. This is for reconstructing model when loading from
    # saved file.

    model.model_hyperparams = config

    return model

class train_ffjord_model:
    def __init__(self):
        self.time_meter = utils.RunningAverageMeter(0.93)
        self.loss_meter = utils.RunningAverageMeter(0.93)
        self.nfef_meter = utils.RunningAverageMeter(0.93)
        self.nfeb_meter = utils.RunningAverageMeter(0.93)
        self.tt_meter = utils.RunningAverageMeter(0.93)

        self.train_itr=0


    def train_epoch(self, flow, train_loader, optimizer, epoch,
                    device=None,
                    output_freq=50):
        """Train model for one epoch.

        Arguments:
            flow {Flow} -- Ffjord model
            train_loader {DataLoader} -- train set data loader
            optimizer {Optimizer} -- model optimizer
            epoch {int} -- epoch number

        Keyword Arguments:
            device {torch.device} -- model device (CPU or GPU) (default: {None})
            output_freq {int} -- frequency for printing status (default: {50})

        Returns:
            float -- average train loss over epoch
        """

        start_time = time.time()
        flow.train()
        train_loss = 0.0
        total_weight = 0.0

        # Change the sampling properties of the dataset over time

        for batch_idx, (h, x, w, snr) in enumerate(train_loader):
            self.train_itr+=1
            optimizer.zero_grad()

            if device is not None:
                h = h.to(device, non_blocking=True)
                x = x.to(device, non_blocking=True)
                w = w.to(device, non_blocking=True)
                snr = snr.to(device, non_blocking=True)

            y = h

            # Compute log prob
            zero = torch.zeros(x.shape[0], 1).to(device)

            # transform to z
            z, delta_logp = flow(x, y, zero)

            # compute log q(z)
            logpz = standard_normal_logprob(z).sum(1, keepdim=True)

            logpx = logpz - delta_logp

            loss = -logpx

            # Keep track of total loss. w is a weight to be applied to each
            # element.
            train_loss += (w * loss.detach()).sum()
            total_weight += w.sum()

            # loss = (w * loss).sum() / w.sum()
            loss = (w * loss).mean()

            self.loss_meter.update(loss.item())

            total_time = count_total_time(flow)
            nfe_forward = count_nfe(flow)

            loss.backward()
            optimizer.step()

            nfe_total = count_nfe(flow)
            nfe_backward = nfe_total - nfe_forward
            self.nfef_meter.update(nfe_forward)
            self.nfeb_meter.update(nfe_backward)

            self.time_meter.update(time.time() - start_time)
            self.tt_meter.update(total_time)

            log_message = (
                'Iter {:04d} | Time {:.4f}({:.4f}) | Loss {:.6f}({:.6f}) | NFE Forward {:.0f}({:.1f})'
                ' | NFE Backward {:.0f}({:.1f}) | CNF Time {:.4f}({:.4f})'.format(
                    self.train_itr, self.time_meter.val, self.time_meter.avg, self.loss_meter.val, self.loss_meter.avg, self.nfef_meter.val, self.nfef_meter.avg,
                    self.nfeb_meter.val, self.nfeb_meter.avg, self.tt_meter.val, self.tt_meter.avg
                )
            )

            print(log_message)

            if (output_freq is not None) and (batch_idx % output_freq == 0):
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.4f}'.format(
                    epoch, batch_idx *
                    train_loader.batch_size, len(train_loader.dataset),
                    100. * batch_idx / len(train_loader),
                    loss.item()))

        train_loss = train_loss.item() / len(train_loader.dataset)
        # train_loss = train_loss.item() / total_weight.item()
        print('Train Epoch: {} \tAverage Loss: {:.4f}'.format(
            epoch, train_loss))
        

        return train_loss


    def test_epoch(self, flow, test_loader, device=None):
        """Calculate test loss for one epoch.

        Arguments:
            flow {Flow} -- Ffjord model
            test_loader {DataLoader} -- test set data loader

        Keyword Arguments:
            device {torch.device} -- model device (CPU or GPu) (default: {None})

        Returns:
            float -- test loss
        """

        with torch.no_grad():
            flow.eval()
            test_loss = 0.0
            total_weight = 0.0
            for h, x, w, snr in test_loader:

                if device is not None:
                    h = h.to(device, non_blocking=True)
                    x = x.to(device, non_blocking=True)
                    w = w.to(device, non_blocking=True)
                    snr = snr.to(device, non_blocking=True)

                y = h


                # Compute log prob

                zero = torch.zeros(x.shape[0], 1).to(x)

                # transform to z
                z, delta_logp = flow(x, y, zero)

                # compute log q(z)
                logpz = standard_normal_logprob(z).sum(1, keepdim=True)

                logpx = logpz - delta_logp

                loss = -logpx

                # Keep track of total loss
                test_loss += (w * loss).sum()
                total_weight += w.sum()

                # loss = (w * loss).sum() / w.sum()
                loss = (w * loss).mean()

            test_loss = test_loss.item() / len(test_loader.dataset)
            # test_loss = test_loss.item() / total_weight.item()
            print('Test set: Average loss: {:.4f}\n'
                .format(test_loss))

            return test_loss

def obtain_samples(flow, y, nsamples, device=None, batch_size=512):
    """Draw samples from the posterior.

    Arguments:
        flow {Flow} -- NSF model
        y {array} -- strain data
        nsamples {int} -- number of samples desired

    Keyword Arguments:
        device {torch.device} -- model device (CPU or GPU) (default: {None})
        batch_size {int} -- batch size for sampling (default: {512})

    Returns:
        Tensor -- samples
    """

    with torch.no_grad():
        flow.eval()

        p_z0 = torch.distributions.MultivariateNormal(
                loc=torch.zeros(15).to(device),
                covariance_matrix=torch.eye(15).to(device)
            )

        y = torch.from_numpy(y).unsqueeze(0).to(device)

        num_batches = nsamples // batch_size
        num_leftover = nsamples % batch_size

        samples = [flow(p_z0.sample((batch_size,)), y.repeat(batch_size,1), torch.zeros(batch_size, 1).to(device), reverse=True) for _ in range(num_batches)]
        if num_leftover > 0:
            samples.append(flow(p_z0.sample((num_leftover,)), y.repeat(num_leftover,1), torch.zeros(num_leftover, 1).to(device), reverse=True))

        # The batching in the nsf package seems screwed up, so we had to do it
        # ourselves, as above. They are concatenating on the wrong axis.

        # samples = flow.sample(nsamples, context=y, batch_size=batch_size)

        x=[item[0] for item in samples]
        logpx=[item[1] for item in samples]
        return torch.cat(x, dim=0)

