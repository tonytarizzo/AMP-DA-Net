
import copy
import math

def update_model_inplace(model, par_before, delta, args, cur_iter, momentum_buffer_list, exp_avgs, exp_avg_sqs):
    grads = copy.deepcopy(delta)
    
    iteration = cur_iter + 1  # add 1 is to make sure nonzero denominator in adam calculation
    lr_decay = 1.0  # learning rate decay factor, 1.0 means no decay

    for i, param in enumerate(model.parameters()): 
        grad = grads[i]  # recieve the aggregated (averaged) gradient
        
        # SGD calculation
        if args.optimizer == 'fedavg':
            param.data.add_(param.data, alpha=-1).add_(par_before[i], alpha=1).add_(grad, alpha=args.global_lr * lr_decay)

        # SGD+momentum calculation
        elif args.optimizer == 'fedavgm':
            buf = momentum_buffer_list[i]
            buf.mul_(args.momentum).add_(grad, alpha=1)
            grad = buf
            param.data.add_(param.data, alpha=-1).add_(par_before[i], alpha=1).add_(grad, alpha=args.global_lr * lr_decay)
            
        # Adam calculation
        elif args.optimizer == 'fedadam':
            exp_avg = exp_avgs[i]
            exp_avg_sq = exp_avg_sqs[i]

            bias_correction1 = 1 - args.beta1 ** iteration
            bias_correction2 = 1 - args.beta2 ** iteration

            exp_avg.mul_(args.beta1).add_(grad, alpha=1 - args.beta1)
            exp_avg_sq.mul_(args.beta2).addcmul_(grad, grad.conj(), value=1 - args.beta2)
            denom = (exp_avg_sq.sqrt() / math.sqrt(bias_correction2)).add_(args.eps) # without maximum

            step_size = args.global_lr * lr_decay / bias_correction1
            param.data.add_(param.data, alpha=-1).add_(par_before[i], alpha=1).addcdiv_(exp_avg, denom, value=step_size)
        else:
            exit('unknown optimizer: {}'.format(args.optimizer))