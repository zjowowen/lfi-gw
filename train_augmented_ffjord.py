import os

os.environ['OMP_NUM_THREADS'] = str(1)
os.environ['MKL_NUM_THREADS'] = str(1)

import argparse
import time
import wandb

from lfigw.gwpe import PosteriorModel
from lfigw.gwpe import Nestedspace

def parse_args():
    parser = argparse.ArgumentParser(
        description=('Model the gravitational-wave parameter '
                     'posterior distribution with neural networks.'))

    # Since options are often combined, defined parent parsers here and pass
    # them as parents when defining ArgumentParsers.

    dir_parent_parser = argparse.ArgumentParser(add_help=False)
    dir_parent_parser.add_argument('--data_dir', type=str, required=True)
    dir_parent_parser.add_argument('--model_dir', type=str, required=True)
    dir_parent_parser.add_argument('--no_cuda', action='store_false',
                                   dest='cuda')

    activation_parent_parser = argparse.ArgumentParser(add_help=None)
    activation_parent_parser.add_argument(
        '--activation', choices=['relu', 'leaky_relu', 'elu'], default='relu')

    train_parent_parser = argparse.ArgumentParser(add_help=None)
    train_parent_parser.add_argument(
        '--batch_size', type=int, default='512')
    train_parent_parser.add_argument('--lr', type=float, default='0.0001')
    train_parent_parser.add_argument('--lr_anneal_method',
                                     choices=['step', 'cosine', 'cosineWR'],
                                     default='step')
    train_parent_parser.add_argument('--no_lr_annealing', action='store_false',
                                     dest='lr_annealing')
    train_parent_parser.add_argument(
        '--steplr_gamma', type=float, default=0.5)
    train_parent_parser.add_argument('--steplr_step_size', type=int,
                                     default=80)
    train_parent_parser.add_argument('--flow_lr', type=float)
    train_parent_parser.add_argument('--epochs', type=int, required=True)
    train_parent_parser.add_argument(
        '--output_freq', type=int, default='50')
    train_parent_parser.add_argument('--no_save', action='store_false',
                                     dest='save')
    train_parent_parser.add_argument('--no_kl_annealing', action='store_false',
                                     dest='kl_annealing')
    train_parent_parser.add_argument('--detectors', nargs='+')
    train_parent_parser.add_argument('--truncate_basis', type=int)
    train_parent_parser.add_argument('--snr_threshold', type=float)
    train_parent_parser.add_argument('--distance_prior_fn',
                                     choices=['uniform_distance',
                                              'inverse_distance',
                                              'linear_distance',
                                              'inverse_square_distance',
                                              'bayeswave'])
    train_parent_parser.add_argument('--snr_annealing', action='store_true')
    train_parent_parser.add_argument('--distance_prior', type=float,
                                     nargs=2)
    train_parent_parser.add_argument('--bw_dstar', type=float)

    cvae_parent_parser = argparse.ArgumentParser(add_help=False)
    cvae_parent_parser.add_argument(
        '--latent_dim', type=int, required=True)
    cvae_parent_parser.add_argument('--hidden_dims', type=int,
                                    nargs='+', required=True)
    cvae_parent_parser.add_argument('--batch_norm', action='store_true')
    cvae_parent_parser.add_argument(
        '--prior_gaussian_nn', action='store_true')
    cvae_parent_parser.add_argument('--prior_full_cov', action='store_true')

    iaf_parent_parser = argparse.ArgumentParser(add_help=False)
    iaf_parent_parser.add_argument('--iaf.hidden_dims', type=int, nargs='+',
                                   required=True)
    context_group = iaf_parent_parser.add_mutually_exclusive_group(
        required=True)
    context_group.add_argument('--iaf.context_dim', type=int)
    context_group.add_argument('--iaf.context_y', action='store_true')
    iaf_parent_parser.add_argument('--iaf.nflows', type=int, required=True)
    iaf_parent_parser.add_argument('--iaf.batch_norm', action='store_true')
    iaf_parent_parser.add_argument('--iaf.bn_momentum', type=float,
                                   default=0.9)
    iaf_parent_parser.add_argument('--iaf.maf_parametrization',
                                   action='store_false',
                                   dest='iaf.iaf_parametrization')
    iaf_parent_parser.add_argument('--iaf.xcontext', action='store_true')
    iaf_parent_parser.add_argument('--iaf.ycontext', action='store_true')

    maf_prior_parent_parser = argparse.ArgumentParser(add_help=False)
    maf_prior_parent_parser.add_argument('--maf_prior.hidden_dims', type=int,
                                         nargs='+',
                                         required=True)
    maf_prior_parent_parser.add_argument('--maf_prior.nflows', type=int,
                                         required=True)
    maf_prior_parent_parser.add_argument('--maf_prior.no_batch_norm',
                                         action='store_false',
                                         dest='maf_prior.batch_norm')
    maf_prior_parent_parser.add_argument('--maf_prior.bn_momentum', type=float,
                                         default=0.9)
    maf_prior_parent_parser.add_argument('--maf_prior.iaf_parametrization',
                                         action='store_true')

    maf_decoder_parent_parser = argparse.ArgumentParser(add_help=False)
    maf_decoder_parent_parser.add_argument('--maf_decoder.hidden_dims',
                                           type=int,
                                           nargs='+',
                                           required=True)
    maf_decoder_parent_parser.add_argument('--maf_decoder.nflows',
                                           type=int,
                                           required=True)
    maf_decoder_parent_parser.add_argument('--maf_decoder.no_batch_norm',
                                           action='store_false',
                                           dest='maf_decoder.batch_norm')
    maf_decoder_parent_parser.add_argument('--maf_decoder.bn_momentum',
                                           type=float,
                                           default=0.9)
    maf_decoder_parent_parser.add_argument('--maf_decoder.iaf_parametrization',
                                           action='store_true')
    maf_decoder_parent_parser.add_argument('--maf_decoder.zcontext',
                                           action='store_true')

    # Subprograms

    mode_subparsers = parser.add_subparsers(title='mode', dest='mode')
    mode_subparsers.required = True

    train_parser = mode_subparsers.add_parser(
        'train', description=('Train a network.'))

    train_subparsers = train_parser.add_subparsers(dest='model_source')
    train_subparsers.required = True

    train_new_parser = train_subparsers.add_parser(
        'new', description=('Build and train a network.'))

    type_subparsers = train_new_parser.add_subparsers(dest='model_type')
    type_subparsers.required = True

    # Pure MAF

    maf_parser = type_subparsers.add_parser(
        'maf',
        description=('Build and train a MAF.'),
        parents=[activation_parent_parser,
                 dir_parent_parser,
                 train_parent_parser])
    maf_parser.add_argument('--hidden_dims', type=int, nargs='+',
                            required=True)
    maf_parser.add_argument('--nflows', type=int, required=True)
    maf_parser.add_argument(
        '--no_batch_norm', action='store_false', dest='batch_norm')
    maf_parser.add_argument('--bn_momentum', type=float, default=0.9)
    maf_parser.add_argument('--iaf_parametrization', action='store_true')

    # nde (curently just NSFC)

    nde_parser = type_subparsers.add_parser(
        'nde',
        description=('Build and train a flow from the nde package.'),
        parents=[activation_parent_parser,
                 dir_parent_parser,
                 train_parent_parser]
    )
    nde_parser.add_argument('--hidden_dims', type=int, required=True)
    nde_parser.add_argument('--nflows', type=int, required=True)
    nde_parser.add_argument('--batch_norm', action='store_true')
    nde_parser.add_argument('--nbins', type=int, required=True)
    nde_parser.add_argument('--tail_bound', type=float, default=1.0)
    nde_parser.add_argument('--apply_unconditional_transform',
                            action='store_true')
    nde_parser.add_argument('--dropout_probability', type=float, default=0.0)
    nde_parser.add_argument('--num_transform_blocks', type=int, default=2)
    nde_parser.add_argument('--base_transform_type', type=str,
                            choices=['rq-coupling', 'rq-autoregressive'],
                            default='rq-coupling')

    # ffjord

    ffjord_parser = type_subparsers.add_parser(
        'ffjord',
        description=('Build and train a flow from the ffjord package.'),
        parents=[activation_parent_parser,
                    dir_parent_parser,
                    train_parent_parser]
    )
    ffjord_parser.add_argument('--atol', type=float, default=1e-05)
    ffjord_parser.add_argument('--rtol', type=float, default=1e-05)
    ffjord_parser.add_argument('--batch_norm', action='store_true')
    ffjord_parser.add_argument('--bn_lag', type=int, default=0)
    ffjord_parser.add_argument('--dims', type=str, default='64-64-64')
    ffjord_parser.add_argument('--divergence_fn', type=str, default='brute_force')
    ffjord_parser.add_argument('--layer_type', type=str, default='concatsquash')
    ffjord_parser.add_argument('--nonlinearity', type=str, default='tanh')
    ffjord_parser.add_argument('--num_blocks', type=int, default=1)
    ffjord_parser.add_argument('--rademacher', action='store_true')
    ffjord_parser.add_argument('--residual', action='store_true')
    ffjord_parser.add_argument('--solver', type=str, default='dopri5')
    ffjord_parser.add_argument('--step_size', type=float, default=None)
    ffjord_parser.add_argument('--spectral_norm', action='store_true')
    ffjord_parser.add_argument('--time_length', type=float, default=0.5)
    ffjord_parser.add_argument('--train_T', action='store_true')
    ffjord_parser.add_argument('--weight_decay', type=float, default=1e-05)

    # augmented_ffjord

    augmented_ffjord_parser = type_subparsers.add_parser(
        'augmented_ffjord',
        description=('Build and train a flow from the augmented_ffjord package.'),
        parents=[activation_parent_parser,
                    dir_parent_parser,
                    train_parent_parser]
    )
    augmented_ffjord_parser.add_argument('--augment_dim', type=int, default=1)
    augmented_ffjord_parser.add_argument('--atol', type=float, default=1e-05)
    augmented_ffjord_parser.add_argument('--rtol', type=float, default=1e-05)
    augmented_ffjord_parser.add_argument('--batch_norm', action='store_true')
    augmented_ffjord_parser.add_argument('--bn_lag', type=int, default=0)
    augmented_ffjord_parser.add_argument('--dims', type=str, default='64-64-64')
    augmented_ffjord_parser.add_argument('--divergence_fn', type=str, default='brute_force')
    augmented_ffjord_parser.add_argument('--layer_type', type=str, default='concatsquash')
    augmented_ffjord_parser.add_argument('--nonlinearity', type=str, default='tanh')
    augmented_ffjord_parser.add_argument('--num_blocks', type=int, default=1)
    augmented_ffjord_parser.add_argument('--rademacher', action='store_true')
    augmented_ffjord_parser.add_argument('--residual', action='store_true')
    augmented_ffjord_parser.add_argument('--solver', type=str, default='dopri5')
    augmented_ffjord_parser.add_argument('--step_size', type=float, default=None)
    augmented_ffjord_parser.add_argument('--spectral_norm', action='store_true')
    augmented_ffjord_parser.add_argument('--time_length', type=float, default=0.5)
    augmented_ffjord_parser.add_argument('--train_T', action='store_true')
    augmented_ffjord_parser.add_argument('--weight_decay', type=float, default=1e-05)

    # Pure CVAE

    cvae_parser = type_subparsers.add_parser(
        'cvae',
        description=('Build and train a CVAE.'),
        parents=[activation_parent_parser,
                 dir_parent_parser,
                 cvae_parent_parser,
                 train_parent_parser])
    cvae_parser.add_argument('--encoder_diag_cov', action='store_false',
                             dest='encoder_full_cov')
    cvae_parser.add_argument('--decoder_diag_cov', action='store_false',
                             dest='decoder_full_cov')

    # CVAE with IAF

    cvae_iaf_parser = type_subparsers.add_parser(
        'cvae+iaf',
        description=('Build and train a CVAE with IAF encoder.'),
        parents=[activation_parent_parser,
                 dir_parent_parser,
                 cvae_parent_parser,
                 iaf_parent_parser,
                 train_parent_parser])
    cvae_iaf_parser.add_argument('--encoder_full_cov', action='store_true')
    cvae_iaf_parser.add_argument('--decoder_diag_cov', action='store_false',
                                 dest='decoder_full_cov')

    # CVAE with prior MAF

    cvae_maf_prior_parser = type_subparsers.add_parser(
        'cvae+maf_prior',
        description=('Build and train a CVAE with MAF prior.'),
        parents=[activation_parent_parser,
                 dir_parent_parser,
                 cvae_parent_parser,
                 maf_prior_parent_parser,
                 train_parent_parser])
    cvae_maf_prior_parser.add_argument('--encoder_full_cov',
                                       action='store_true')
    cvae_maf_prior_parser.add_argument('--decoder_diag_cov',
                                       action='store_false',
                                       dest='decoder_full_cov')

    # CVAE with decoder MAF

    cvae_maf_decoder_parser = type_subparsers.add_parser(
        'cvae+maf_decoder',
        description=('Build and train a CVAE with MAF decoder.'),
        parents=[activation_parent_parser,
                 dir_parent_parser,
                 cvae_parent_parser,
                 maf_decoder_parent_parser,
                 train_parent_parser])
    cvae_maf_decoder_parser.add_argument('--encoder_diag_cov',
                                         action='store_false',
                                         dest='encoder_full_cov')
    cvae_maf_decoder_parser.add_argument('--decoder_full_cov',
                                         action='store_true')

    # CVAE with IAF + MAF decoder

    cvae_iaf_maf_decoder_parser = type_subparsers.add_parser(
        'cvae+iaf+maf_decoder',
        description=('Build and train a CVAE with IAF encoder'
                     'and MAF decoder.'),
        parents=[activation_parent_parser,
                 dir_parent_parser,
                 cvae_parent_parser,
                 iaf_parent_parser,
                 maf_decoder_parent_parser,
                 train_parent_parser])
    cvae_iaf_maf_decoder_parser.add_argument('--encoder_full_cov',
                                             action='store_true')
    cvae_iaf_maf_decoder_parser.add_argument('--decoder_full_cov',
                                             action='store_true')

    # CVAE with prior MAF + posterior MAF

    cvae_maf_prior_maf_decoder_parser = type_subparsers.add_parser(
        'cvae+maf_prior+maf_decoder',
        description=('Build and train a CVAE with MAF prior'
                     'and MAF decoder.'),
        parents=[activation_parent_parser,
                 dir_parent_parser,
                 cvae_parent_parser,
                 maf_prior_parent_parser,
                 maf_decoder_parent_parser,
                 train_parent_parser])
    cvae_maf_prior_maf_decoder_parser.add_argument('--encoder_full_cov',
                                                   action='store_true')
    cvae_maf_prior_maf_decoder_parser.add_argument('--decoder_full_cov',
                                                   action='store_true')

    # CVAE with IAF + prior MAF + posterior MAF

    cvae_all_parser = type_subparsers.add_parser(
        'cvae+all',
        description=('Build and train a CVAE with IAF, MAF prior'
                     'and MAF decoder.'),
        parents=[activation_parent_parser,
                 dir_parent_parser,
                 cvae_parent_parser,
                 iaf_parent_parser,
                 maf_prior_parent_parser,
                 maf_decoder_parent_parser,
                 train_parent_parser])
    cvae_all_parser.add_argument('--encoder_full_cov',
                                 action='store_true')
    cvae_all_parser.add_argument('--decoder_full_cov',
                                 action='store_true')

    train_subparsers.add_parser(
        'existing',
        description=('Load a network from file and continue training.'),
        parents=[dir_parent_parser, train_parent_parser])

    ns = Nestedspace()

    return parser.parse_args(namespace=ns)


def main():
    args = parse_args()

    if args.mode == 'train':

        print('Waveform directory', args.data_dir)
        print('Model directory', args.model_dir)
        pm = PosteriorModel(model_dir=args.model_dir,
                            data_dir=args.data_dir,
                            use_cuda=args.cuda)
        print('Device', pm.device)
        print('Loading dataset')
        pm.load_dataset(batch_size=args.batch_size,
                        detectors=args.detectors,
                        truncate_basis=args.truncate_basis,
                        snr_threshold=args.snr_threshold,
                        distance_prior_fn=args.distance_prior_fn,
                        distance_prior=args.distance_prior,
                        bw_dstar=args.bw_dstar)
        print('Detectors:', pm.detectors)

        if args.model_source == 'new':

            print('\nConstructing model of type', args.model_type)

            if args.model_type == 'maf':
                pm.construct_model(
                    'maf',
                    hidden_dims=args.hidden_dims,
                    nflows=args.nflows,
                    batch_norm=args.batch_norm,
                    bn_momentum=args.bn_momentum,
                    iaf_parametrization=args.iaf_parametrization,
                    activation=args.activation)

            elif args.model_type == 'nde':
                pm.construct_model(
                    'nde',
                    num_flow_steps=args.nflows,
                    base_transform_kwargs={
                        'hidden_dim': args.hidden_dims,
                        'num_transform_blocks': args.num_transform_blocks,
                        'activation': args.activation,
                        'dropout_probability': args.dropout_probability,
                        'batch_norm': args.batch_norm,
                        'num_bins': args.nbins,
                        'tail_bound': args.tail_bound,
                        'apply_unconditional_transform': args.apply_unconditional_transform,
                        'base_transform_type': args.base_transform_type
                    }
                )

            elif args.model_type == 'ffjord':
                pm.construct_model(
                    'ffjord',
                    weight_decay=args.weight_decay,
                    num_blocks=args.num_blocks,
                    dims=args.dims,
                    divergence_fn=args.divergence_fn,
                    batch_norm=args.batch_norm,
                    layer_type=args.layer_type,
                    solver=args.solver,
                    atol=args.atol,
                    rtol=args.rtol,
                    nonlinearity=args.nonlinearity,
                    rademacher=args.rademacher,
                    residual=args.residual,
                    time_length=args.time_length,
                    train_T=args.train_T,
                )

            elif args.model_type == 'augmented_ffjord':
                pm.construct_model(
                    'augmented_ffjord',
                    augment_dim=args.augment_dim,
                    weight_decay=args.weight_decay,
                    num_blocks=args.num_blocks,
                    dims=args.dims,
                    divergence_fn=args.divergence_fn,
                    batch_norm=args.batch_norm,
                    layer_type=args.layer_type,
                    solver=args.solver,
                    atol=args.atol,
                    rtol=args.rtol,
                    nonlinearity=args.nonlinearity,
                    rademacher=args.rademacher,
                    residual=args.residual,
                    time_length=args.time_length,
                    train_T=args.train_T,
                )

            elif args.model_type == 'cvae':
                pm.construct_model(
                    'cvae',
                    hidden_dims=args.hidden_dims,
                    latent_dim=args.latent_dim,
                    encoder_full_cov=args.encoder_full_cov,
                    decoder_full_cov=args.decoder_full_cov,
                    activation=args.activation,
                    batch_norm=args.batch_norm,
                    prior_gaussian_nn=args.prior_gaussian_nn,
                    prior_full_cov=args.prior_full_cov)

            elif args.model_type == 'cvae+iaf':
                pm.construct_model(
                    'cvae',
                    hidden_dims=args.hidden_dims,
                    latent_dim=args.latent_dim,
                    encoder_full_cov=args.encoder_full_cov,
                    decoder_full_cov=args.decoder_full_cov,
                    activation=args.activation,
                    iaf={
                        'context_dim': args.iaf.context_dim,
                        'hidden_dims': args.iaf.hidden_dims,
                        'nflows': args.iaf.nflows,
                        'batch_norm': args.iaf.batch_norm,
                        'bn_momentum': args.iaf.bn_momentum,
                        'iaf_parametrization': args.iaf.iaf_parametrization
                    },
                    encoder_xcontext=args.iaf.xcontext,
                    encoder_ycontext=args.iaf.ycontext,
                    batch_norm=args.batch_norm,
                    prior_gaussian_nn=args.prior_gaussian_nn,
                    prior_full_cov=args.prior_full_cov)

            elif args.model_type == 'cvae+maf_prior':
                pm.construct_model(
                    'cvae',
                    hidden_dims=args.hidden_dims,
                    latent_dim=args.latent_dim,
                    encoder_full_cov=args.encoder_full_cov,
                    decoder_full_cov=args.decoder_full_cov,
                    activation=args.activation,
                    prior_maf={
                        'hidden_dims': args.maf_prior.hidden_dims,
                        'nflows': args.maf_prior.nflows,
                        'batch_norm': args.maf_prior.batch_norm,
                        'bn_momentum': args.maf_prior.bn_momentum,
                        'iaf_parametrization':
                        args.maf_prior.iaf_parametrization
                    },
                    batch_norm=args.batch_norm,
                    prior_gaussian_nn=args.prior_gaussian_nn,
                    prior_full_cov=args.prior_full_cov)

            elif args.model_type == 'cvae+maf_decoder':
                pm.construct_model(
                    'cvae',
                    hidden_dims=args.hidden_dims,
                    latent_dim=args.latent_dim,
                    encoder_full_cov=args.encoder_full_cov,
                    decoder_full_cov=args.decoder_full_cov,
                    activation=args.activation,
                    decoder_maf={
                        'hidden_dims': args.maf_decoder.hidden_dims,
                        'nflows': args.maf_decoder.nflows,
                        'batch_norm': args.maf_decoder.batch_norm,
                        'bn_momentum': args.maf_decoder.bn_momentum,
                        'iaf_parametrization':
                        args.maf_decoder.iaf_parametrization
                    },
                    decoder_zcontext=args.maf_decoder.zcontext,
                    batch_norm=args.batch_norm,
                    prior_gaussian_nn=args.prior_gaussian_nn,
                    prior_full_cov=args.prior_full_cov)

            elif args.model_type == 'cvae+iaf+maf_decoder':
                pm.construct_model(
                    'cvae',
                    hidden_dims=args.hidden_dims,
                    latent_dim=args.latent_dim,
                    encoder_full_cov=args.encoder_full_cov,
                    decoder_full_cov=args.decoder_full_cov,
                    activation=args.activation,
                    iaf={
                        'context_dim': args.iaf.context_dim,
                        'hidden_dims': args.iaf.hidden_dims,
                        'nflows': args.iaf.nflows,
                        'batch_norm': args.iaf.batch_norm,
                        'bn_momentum': args.iaf.bn_momentum,
                        'iaf_parametrization': args.iaf.iaf_parametrization
                    },
                    decoder_maf={
                        'hidden_dims': args.maf_decoder.hidden_dims,
                        'nflows': args.maf_decoder.nflows,
                        'batch_norm': args.maf_decoder.batch_norm,
                        'bn_momentum': args.maf_decoder.bn_momentum,
                        'iaf_parametrization':
                        args.maf_decoder.iaf_parametrization
                    },
                    encoder_xcontext=args.iaf.xcontext,
                    encoder_ycontext=args.iaf.ycontext,
                    decoder_zcontext=args.maf_decoder.zcontext,
                    batch_norm=args.batch_norm,
                    prior_gaussian_nn=args.prior_gaussian_nn,
                    prior_full_cov=args.prior_full_cov)

            elif args.model_type == 'cvae+maf_prior+maf_decoder':
                pm.construct_model(
                    'cvae',
                    hidden_dims=args.hidden_dims,
                    latent_dim=args.latent_dim,
                    encoder_full_cov=args.encoder_full_cov,
                    decoder_full_cov=args.decoder_full_cov,
                    activation=args.activation,
                    prior_maf={
                        'hidden_dims': args.maf_prior.hidden_dims,
                        'nflows': args.maf_prior.nflows,
                        'batch_norm': args.maf_prior.batch_norm,
                        'bn_momentum': args.maf_prior.bn_momentum,
                        'iaf_parametrization':
                        args.maf_prior.iaf_parametrization
                    },
                    decoder_maf={
                        'hidden_dims': args.maf_decoder.hidden_dims,
                        'nflows': args.maf_decoder.nflows,
                        'batch_norm': args.maf_decoder.batch_norm,
                        'bn_momentum': args.maf_decoder.bn_momentum,
                        'iaf_parametrization':
                        args.maf_decoder.iaf_parametrization
                    },
                    decoder_zcontext=args.maf_decoder.zcontext,
                    batch_norm=args.batch_norm,
                    prior_gaussian_nn=args.prior_gaussian_nn,
                    prior_full_cov=args.prior_full_cov)

            elif args.model_type == 'cvae+all':
                pm.construct_model(
                    'cvae',
                    hidden_dims=args.hidden_dims,
                    latent_dim=args.latent_dim,
                    encoder_full_cov=args.encoder_full_cov,
                    decoder_full_cov=args.decoder_full_cov,
                    activation=args.activation,
                    iaf={
                        'context_dim': args.iaf.context_dim,
                        'hidden_dims': args.iaf.hidden_dims,
                        'nflows': args.iaf.nflows,
                        'batch_norm': args.iaf.batch_norm,
                        'bn_momentum': args.iaf.bn_momentum,
                        'iaf_parametrization': args.iaf.iaf_parametrization
                    },
                    prior_maf={
                        'hidden_dims': args.maf_prior.hidden_dims,
                        'nflows': args.maf_prior.nflows,
                        'batch_norm': args.maf_prior.batch_norm,
                        'bn_momentum': args.maf_prior.bn_momentum,
                        'iaf_parametrization':
                        args.maf_prior.iaf_parametrization
                    },
                    decoder_maf={
                        'hidden_dims': args.maf_decoder.hidden_dims,
                        'nflows': args.maf_decoder.nflows,
                        'batch_norm': args.maf_decoder.batch_norm,
                        'bn_momentum': args.maf_decoder.bn_momentum,
                        'iaf_parametrization':
                        args.maf_decoder.iaf_parametrization
                    },
                    encoder_xcontext=args.iaf.xcontext,
                    encoder_ycontext=args.iaf.ycontext,
                    decoder_zcontext=args.maf_decoder.zcontext,
                    batch_norm=args.batch_norm,
                    prior_gaussian_nn=args.prior_gaussian_nn,
                    prior_full_cov=args.prior_full_cov)

            print('\nInitial learning rate', args.lr)
            if args.lr_annealing is True:
                if args.lr_anneal_method == 'step':
                    print('Stepping learning rate by', args.steplr_gamma,
                          'every', args.steplr_step_size, 'epochs')
                elif args.lr_anneal_method == 'cosine':
                    print('Using cosine LR annealing.')
                elif args.lr_anneal_method == 'cosineWR':
                    print('Using cosine LR annealing with warm restarts.')
            else:
                print('Using constant learning rate. No annealing.')
            if args.flow_lr is not None:
                print('Autoregressive flows initial lr', args.flow_lr)
            pm.initialize_training(lr=args.lr,
                                   lr_annealing=args.lr_annealing,
                                   anneal_method=args.lr_anneal_method,
                                   total_epochs=args.epochs,
                                   # steplr=args.steplr,
                                   steplr_step_size=args.steplr_step_size,
                                   steplr_gamma=args.steplr_gamma,
                                   flow_lr=args.flow_lr)

        elif args.model_source == 'existing':

            print('Loading existing model')
            pm.load_model()

        print('\nModel hyperparameters:')
        for key, value in pm.model.model_hyperparams.items():
            if type(value) == dict:
                print(key)
                for k, v in value.items():
                    print('\t', k, '\t', v)
            else:
                print(key, '\t', value)

        if pm.model_type == 'cvae' and args.kl_annealing:
            print('\nUsing cyclic KL annealing')

        print('\nTraining for {} epochs'.format(args.epochs))

        print('Starting timer')
        start_time = time.time()

        wandb.init(project=f"lfi-gw-{args.model_type}", config=vars(args))

        pm.train(args.epochs,
                 output_freq=args.output_freq,
                 kl_annealing=args.kl_annealing,
                 snr_annealing=args.snr_annealing)

        print('Stopping timer.')
        stop_time = time.time()
        print('Training time (including validation): {} seconds'
              .format(stop_time - start_time))

        if args.save:
            print('Saving model')
            pm.save_model()

    print('Program complete')


if __name__ == "__main__":
    main()
