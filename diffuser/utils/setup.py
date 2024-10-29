import os
import importlib
import random
import numpy as np
import torch
import json
# from tap import 
import argparse
import pdb

from .serialization import mkdir

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def watch(args_to_watch):
    def _fn(args):
        exp_name = []
        for key, label in args_to_watch:
            if not hasattr(args, key):
                continue
            val = getattr(args, key)
            if type(val) == dict:
                val = '_'.join(f'{k}-{v}' for k, v in val.items())
            exp_name.append(f'{label}{val}')
        exp_name = '_'.join(exp_name)
        exp_name = exp_name.replace('/_', '/')
        exp_name = exp_name.replace('(', '').replace(')', '')
        exp_name = exp_name.replace(', ', '-')
        return exp_name
    return _fn

def lazy_fstring(template, args):
    ## https://stackoverflow.com/a/53671539
    return eval(f"f'{template}'")

class Parser(argparse.ArgumentParser):

    def __init__(self, *args, savepath='', **kwargs):
        super().__init__(*args, **kwargs)
        self.add_argument('--config', help='Path to config file')
        self.savepath = savepath

    def save(self, args):
        fullpath = os.path.join(self.savepath, 'args.json')
        with open(fullpath, 'w') as f:
            json.dump(vars(args), f, skipkeys=True)
        # print(f'[ utils/setup ] Saved args to {fullpath}')
        # super().save(fullpath, skip_unpicklable=True)

    def parse_args(self, experiment=None):
        args = super().parse_args()
        args.dataset = self.dataset
        args.config = self.config
        ## if not loading from a config script, skip the result of the setup
        if not hasattr(args, 'config'): return args
        args = self.read_config(args, experiment)
        # self.add_extras(args)
        self.eval_fstrings(args)
        self.set_seed(args)
        self.set_loadbase(args)
        self.generate_exp_name(args)
        self.mkdir(args)
        return args

    def read_config(self, args, experiment):
        '''
            Load parameters from config file
        '''
        dataset = args.dataset.replace('-', '_')
        # print(f'[ utils/setup ] Reading config: {args.config}:{dataset}')
        module = importlib.import_module(args.config)
        params = getattr(module, 'base')[experiment]

        if hasattr(module, dataset) and experiment in getattr(module, dataset):
            # print(f'[ utils/setup ] Using overrides | config: {args.config} | dataset: {dataset}')
            overrides = getattr(module, dataset)[experiment]
            params.update(overrides)
        # else:
            # print(f'[ utils/setup ] Not using overrides | config: {args.config} | dataset: {dataset}')

        self._dict = {}
        for key, val in params.items():
            setattr(args, key, val)
            self._dict[key] = val

        return args

    def add_extras(self, args):
        '''
            Override config parameters with command-line arguments
        '''
        extras = args.extra_args
        if not len(extras):
            return

        # print(f'[ utils/setup ] Found extras: {extras}')
        assert len(extras) % 2 == 0, f'Found odd number ({len(extras)}) of extras: {extras}'
        for i in range(0, len(extras), 2):
            key = extras[i].replace('--', '')
            val = extras[i+1]
            assert hasattr(args, key), f'[ utils/setup ] {key} not found in config: {args.config}'
            old_val = getattr(args, key)
            old_type = type(old_val)
            # print(f'[ utils/setup ] Overriding config | {key} : {old_val} --> {val}')
            if val == 'None':
                val = None
            elif val == 'latest':
                val = 'latest'
            elif old_type in [bool, type(None)]:
                try:
                    val = eval(val)
                except:
                    print(f'[ utils/setup ] Warning: could not parse {val} (old: {old_val}, {old_type}), using str')
            else:
                val = old_type(val)
            setattr(args, key, val)
            self._dict[key] = val

    def eval_fstrings(self, args):
        for key, old in self._dict.items():
            if type(old) is str and old[:2] == 'f:':
                val = old.replace('{', '{args.').replace('f:', '')
                new = lazy_fstring(val, args)
                # print(f'[ utils/setup ] Lazy fstring | {key} : {old} --> {new}')
                setattr(self, key, new)
                self._dict[key] = new
                setattr(args, key, new)     # Added so that it works with argparse

    def set_seed(self, args):
        if not hasattr(args, 'seed') or args.seed is None:
            return
        # print(f'[ utils/setup ] Setting seed: {args.seed}')
        set_seed(args.seed)

    def set_loadbase(self, args):
        if hasattr(args, 'loadbase') and args.loadbase is None:
            # print(f'[ utils/setup ] Setting loadbase: {args.logbase}')
            args.loadbase = args.logbase

    def generate_exp_name(self, args):
        if not 'exp_name' in dir(args):
            return
        exp_name = getattr(args, 'exp_name')
        if callable(exp_name):
            exp_name_string = exp_name(args)
            # print(f'[ utils/setup ] Setting exp_name to: {exp_name_string}')
            setattr(args, 'exp_name', exp_name_string)
            self._dict['exp_name'] = exp_name_string

    def mkdir(self, args):
        if 'logbase' in dir(args) and 'dataset' in dir(args) and 'exp_name' in dir(args):
            args.savepath = os.path.join(args.logbase, args.dataset, args.exp_name, str(args.seed))
            self._dict['savepath'] = args.savepath
            # if 'suffix' in dir(args):
            #     args.savepath = os.path.join(args.savepath, args.suffix)
            if mkdir(args.savepath):
                print(f'[ utils/setup ] Made savepath: {args.savepath}')
            self.save(args)