import argparse
from collections import defaultdict

import pickle
import numpy as np
import pandas as pd
import jax.random
import jax.numpy as jnp

import numpyro
from numpyro.infer import MCMC, NUTS
import numpyro.distributions as dist
from numpyro.contrib.funsor import config_enumerate
from numpyro.ops.indexing import Vindex

import infer
from run_number import get_concept_sets, X_to_tensor, get_likelihoods_predictives
from np_utils import get_likelihoods_predictives


def yn_to_tensor(X_s, df, n_support=100):
    X_y_yn = -np.ones((len(X_s), n_support, 2))
    
    for _, row in df.iterrows():
        X, y, yes, no = row['set'], row['target'], row['yes'], row['no']
        X_i = X_s.index(X)

        X_y_yn[X_i, y-1, :] = [yes, yes+no]

    return jnp.array(X_y_yn)


def load_llm_data(data_path):
    all_results, all_queries = pickle.load(open(data_path, 'rb'))

    agg_res = defaultdict(lambda: np.zeros((100, 2)))
    yn_keys = {'Yes': 1, 'No': 0}

    for res, (messages, X, y, yn) in zip(all_results, all_queries):
        yn_lp = res['completion']['prompt'][0]['logprobs']['token_logprobs'][-1]
        yn_01 = yn_keys[yn]
        agg_res[X][y-1, yn_01] = yn_lp

    agg_pred = {}

    for X, y_lp in agg_res.items():
        y_pr = np.exp(y_lp)
        y_pr = y_pr[:, 1] / y_pr.sum(axis=1)
        agg_pred[X] = y_pr

    return agg_pred


def preds_to_array(agg_pred, X_s):
    return jnp.array([agg_pred[X] for X in X_s])


def main(args):

    df_bp = pd.read_csv('./data/bp_number_game/human-data.txt', sep='\t')
    X_s  = sorted(set(df_bp['set']))
    obs_X = X_to_tensor(X_s)
    
    # Load LLM prediction data for  p*(y | X)
    agg_pred = load_llm_data('./out/results_llama-3-1-8b.pk')
    obs_py = preds_to_array(agg_pred, X_s)

    y_prior = obs_py.mean(axis=0)
    y_prior /= y_prior.max()

    # Likelihoods  p(X | h)  for each h
    h_likelihoods, y_predictives = get_likelihoods_predictives(epsilon=0.1, noise=True, y_prior=y_prior)
    h_likelihoods, y_predictives = jnp.array(h_likelihoods), jnp.array(y_predictives)

    mcmc_fname = './out/mcmc_llama8b_3-orig.pk'

    # likelihoods p(X | h)  =  \prod_k  p(x_k | h)
    X_ = obs_X[..., None]
    X_likelihoods = dist.Categorical(probs=h_likelihoods).mask(X_ != -1).log_prob(X_).sum(axis=-2)          ##### shape: (n_hypotheses, n_data)

    # Xk_likelihoods = dist.Categorical(probs=h_likelihoods).mask(X_ != -1).log_prob(X_)                      ##### shape: (n_hypotheses, K, n_data)
    # Xk_likelihoods = jnp.exp(Xk_likelihoods)


    ###################################
    print('Loaded data, running model')

    model = infer.number_bpl_model
    model_kwargs = dict(
        y_predictives=y_predictives,
        X_likelihoods=X_likelihoods,
        # Xk_likelihoods=Xk_likelihoods,
        obs_py=obs_py,

        prior_type='noise2',
    )

    with numpyro.handlers.seed(rng_seed=1):
        trace = numpyro.handlers.trace(model).get_trace(**model_kwargs)
    print(numpyro.util.format_shapes(trace))

    kernel = numpyro.infer.NUTS(model)


    mcmc = MCMC(
        kernel,
        num_warmup=args.num_warmup,
        num_samples=args.num_samples,
        num_chains=args.num_chains,
        progress_bar=True
    )

    ### with jax.profiler.trace("/tmp/jax-trace", create_perfetto_link=True):
    mcmc.run(
        jax.random.PRNGKey(42),
        extra_fields=('potential_energy',),
        **model_kwargs
    )

    samples = mcmc.get_samples()
    mcmc.print_summary()

    with open(mcmc_fname, 'wb') as f:
        pickle.dump(mcmc, f)






if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-n", "--num-samples", default=1000, type=int)
    parser.add_argument("--num-warmup", default=500, type=int)
    parser.add_argument("--num-chains", default=1, type=int)
    parser.add_argument("--device", default='cpu', type=str, help='cpu or gpu')
    args = parser.parse_args()

    # numpyro.set_platform(args.device)
    # numpyro.set_host_device_count(args.num_chains)
    numpyro.set_host_device_count(args.num_chains)

    print('devices: ', jax.device_count())

    main(args)
