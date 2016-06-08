#!/usr/bin/env python

import matplotlib.pyplot as plt
import numpy as np
import cPickle, argparse
import shared as sh


def get_property(stats, prop):
    return np.array([stats[i][prop] for i in range(len(stats))])


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Plot statistics')
    parser.add_argument('-s', '--save', help='whether to save the plots to disk',
        dest='save', action='store_const', default=False, const=True)
    parser.add_argument('-ne', '--no-ensemble', help='do not do ensemble plots',
        dest='no_ensemble', action='store_const', default=False, const=True)
    parser.add_argument('-e', '--ensemble-stats', help='ensemble stats file (w/o .pkl)',
        dest='ensemble_stats', default=sh.ensemble_stats.replace('.pkl', ''))
    parser.add_argument('-nt', '--no-transductor', help='do not do transductor plots',
        dest='no_transductor', action='store_const', default=False, const=True)
    parser.add_argument('-t', '--transductor-stats', help='transductor stats file (w/o .pkl)',
        dest='transductor_stats', default=sh.transductor_stats.replace('.pkl', ''))
    args = parser.parse_args()


    if not args.no_ensemble:
        # Load ensemble stats
        with open(args.ensemble_stats + '.pkl', 'rb') as f:
            ensemble_stats = cPickle.load(f)

        assert len(ensemble_stats) > 0, 'Stats are empty!'
        x = np.arange(len(ensemble_stats[0]['loss']))

        # Plot accuracy over training and validation sets with error bars
        acc = get_property(ensemble_stats, 'acc')
        plt.errorbar(x, acc.mean(axis=0), yerr=acc.std(axis=0),
            label='Accuracy on training set')
        val_acc = get_property(ensemble_stats, 'val_acc')
        plt.errorbar(x, val_acc.mean(axis=0), yerr=val_acc.std(axis=0),
            label='Accuracy on validation set')
        axes = plt.gca()
        axes.set_ylim([0.8, .95])
        plt.legend()

        if args.save:
            plt.savefig(
                args.ensemble_stats.replace(sh.stats_path, sh.imgs_path) + '.png')


    if not args.no_transductor:
        # Load transducers stats
        with open(args.transductor_stats + '.pkl', 'rb') as f:
            transductor_stats = cPickle.load(f)

        plt.figure()
        x = np.arange(len(transductor_stats))
        for prop in ['auc', 'ks', 'cvm', 'loss']:
            y = get_property(transductor_stats, prop)
            plt.plot(x, y, label=prop.upper())
        plt.legend()

        if args.save:
            plt.savefig(
                args.transductor_stats.replace(sh.stats_path, sh.imgs_path) + '.png')


    if (not args.no_ensemble or not args.no_transductor) and not args.save:
        plt.show()
