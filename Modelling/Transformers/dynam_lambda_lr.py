import types
import warnings
from torch.optim.lr_scheduler import LRScheduler, LambdaLR

class DynamicLambdaLR(LRScheduler):
    def __init__(self, optimizer, lr_lambda, last_epoch=-1, verbose=False):
        self.optimizer = optimizer
        if not isinstance(lr_lambda, list) and not isinstance(lr_lambda, tuple):
            self.lr_lambdas = [lr_lambda] * len(optimizer.param_groups)
        else:
            self.lr_lambdas = list(lr_lambda)
        super().__init__(optimizer, last_epoch, verbose)


    def state_dict(self):
        """Returns the state of the scheduler as a :class:`dict`.

        It contains an entry for every variable in self.__dict__ which
        is not the optimizer.
        The learning rate lambda functions will only be saved if they are callable objects
        and not if they are functions or lambdas.

        When saving or loading the scheduler, please make sure to also save or load the state of the optimizer.
        """

        state_dict = {key: value for key, value in self.__dict__.items() if key not in ('optimizer', 'lr_lambdas')}
        state_dict['lr_lambdas'] = [None] * len(self.lr_lambdas)

        for idx, fn in enumerate(self.lr_lambdas):
            if not isinstance(fn, types.FunctionType):
                state_dict['lr_lambdas'][idx] = fn.__dict__.copy()

        return state_dict

    def load_state_dict(self, state_dict):
        """Loads the schedulers state.

        When saving or loading the scheduler, please make sure to also save or load the state of the optimizer.

        Args:
            state_dict (dict): scheduler state. Should be an object returned
                from a call to :meth:`state_dict`.
        """

        lr_lambdas = state_dict.pop('lr_lambdas')
        self.__dict__.update(state_dict)
        # Restore state_dict keys in order to prevent side effects
        # https://github.com/pytorch/pytorch/issues/32756
        state_dict['lr_lambdas'] = lr_lambdas

        for idx, fn in enumerate(lr_lambdas):
            if fn is not None:
                self.lr_lambdas[idx].__dict__.update(fn)

    def get_lr(self):
        len_optimizer_param_groups = len(self.optimizer.param_groups)
        if len(self.base_lrs) != len_optimizer_param_groups:
            warnings.warn("Number of base lrs does not match number of optimizer groups.")
            self.base_lrs = [group['initial_lr'] for group in self.optimizer.param_groups]

        if len(self.lr_lambdas) < len_optimizer_param_groups:
            warnings.warn("Number of lambdas is less than number of optimizer groups.")
            self.lr_lambdas = [self.lr_lambdas[0] for _ in self.optimizer.param_groups]

        return [base_lr * lmbda(self.last_epoch)
                for lmbda, base_lr in zip(self.lr_lambdas, self.base_lrs)]