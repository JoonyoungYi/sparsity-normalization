import argparse
import time
import math
import os
import torch
import torch.nn as nn
import torch.onnx

from . import data
from . import models
from common.utils import str2bool


def batchify(data, bsz, device):
    # Work out how cleanly we can divide the dataset into bsz parts.
    nbatch = data.size(0) // bsz
    # Trim off any extra elements that wouldn't cleanly fit (remainders).
    data = data.narrow(0, 0, nbatch * bsz)
    # Evenly divide the data across the bsz batches.
    data = data.view(bsz, -1).t().contiguous()
    return data.to(device)


def train(model, corpus, optimizer, criterion, train_data, args, epoch):
    # Turn on training mode which enables dropout.
    model.train()
    total_loss = 0.
    start_time = time.time()
    ntokens = len(corpus.dictionary)
    if args.model != 'Transformer':
        hidden = model.init_hidden(args.batch_size)
    for batch, i in enumerate(range(0, train_data.size(0) - 1, args.bptt)):
        data, targets = get_batch(train_data, i, args.bptt)
        # Starting each batch, we detach the hidden state from how it was previously produced.
        # If we didn't, the model would try backpropagating all the way to start of the dataset.
        model.zero_grad()
        if args.model == 'Transformer':
            output = model(data)
        else:
            hidden = repackage_hidden(hidden)
            output, hidden = model(data, hidden)
        loss = criterion(output.view(-1, ntokens), targets)

        optimizer.zero_grad()
        loss.backward()

        # # `clip_grad_norm` helps prevent the exploding gradient problem in RNNs / LSTMs.
        # torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
        # for p in model.parameters():
        #     p.data.add_(-lr, p.grad.data)
        if args.clip:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)

        optimizer.step()

        total_loss += loss.item()

        if batch % args.log_interval == 0 and batch > 0:
            cur_loss = total_loss / args.log_interval
            elapsed = time.time() - start_time
            print('| epoch {:3d} | {:5d}/{:5d} batches | ms/batch {:5.2f} | '
                  'loss {:5.2f} | ppl {:8.2f}'.format(
                      epoch, batch,
                      len(train_data) // args.bptt,
                      elapsed * 1000 / args.log_interval, cur_loss,
                      math.exp(cur_loss)))
            total_loss = 0
            start_time = time.time()


def export_onnx(path, batch_size, seq_len, device):
    print('The model is also exported in ONNX format at {}'.format(
        os.path.realpath(args.onnx_export)))
    model.eval()
    dummy_input = torch.LongTensor(seq_len * batch_size).zero_().view(
        -1, batch_size).to(device)
    hidden = model.init_hidden(batch_size)
    torch.onnx.export(model, (dummy_input, hidden), path)


def repackage_hidden(h):
    """Wraps hidden states in new Tensors, to detach them from their history."""

    if isinstance(h, torch.Tensor):
        return h.detach()
    else:
        return tuple(repackage_hidden(v) for v in h)


# get_batch subdivides the source data into chunks of length args.bptt.
# If source is equal to the example output of the batchify function, with
# a bptt-limit of 2, we'd get the following two Variables for i = 0:
# ┌ a g m s ┐ ┌ b h n t ┐
# └ b h n t ┘ └ c i o u ┘
# Note that despite the name of the function, the subdivison of data is not
# done along the batch dimension (i.e. dimension 1), since that was handled
# by the batchify function. The chunks are along dimension 0, corresponding
# to the seq_len dimension in the LSTM.


def get_batch(source, i, bptt):
    seq_len = min(bptt, len(source) - 1 - i)
    data = source[i:i + seq_len]
    target = source[i + 1:i + 1 + seq_len].view(-1)
    return data, target


def evaluate(data_source, criterion, args):
    # Turn on evaluation mode which disables dropout.
    model.eval()
    total_loss = 0.
    ntokens = len(corpus.dictionary)
    if args.model != 'Transformer':
        hidden = model.init_hidden(eval_batch_size)
    with torch.no_grad():
        for i in range(0, data_source.size(0) - 1, args.bptt):
            data, targets = get_batch(data_source, i, args.bptt)
            if args.model == 'Transformer':
                output = model(data)
            else:
                output, hidden = model(data, hidden)
                hidden = repackage_hidden(hidden)
            output_flat = output.view(-1, ntokens)
            total_loss += len(data) * criterion(output_flat, targets).item()
    return total_loss / (len(data_source) - 1)


def main():
    parser = argparse.ArgumentParser(
        description='PyTorch Wikitext-2 RNN/LSTM Language Model')
    parser.add_argument(
        '--data',
        type=str,
        # default='lm/datasets/wikitext-2',
        default='lm/datasets/ptb',
        help='location of the data corpus')
    parser.add_argument('--train', type=str2bool, default=False, help='')
    parser.add_argument(
        '--model',
        type=str,
        default='LSTM',
        help=
        'type of recurrent net (RNN_TANH, RNN_RELU, LSTM, GRU, Transformer)')
    parser.add_argument(
        '--drop_mode',
        type=str,
        default='standard',
        choices=['standard', 'sparsity-normalization'],
        help='')
    parser.add_argument(
        '--emsize', type=int, default=200, help='size of word embeddings')
    parser.add_argument(
        '--nhid',
        type=int,
        default=200,
        help='number of hidden units per layer')
    parser.add_argument(
        '--nlayers', type=int, default=2, help='number of layers')
    parser.add_argument(
        '--lr',
        type=float,
        default=1e-3,
        # default=20,
        help='initial learning rate')
    parser.add_argument(
        '--clip', type=float, default=0.25, help='gradient clipping')
    parser.add_argument(
        '--epochs', type=int, default=40, help='upper epoch limit')
    parser.add_argument(
        '--batch_size', type=int, default=20, metavar='N', help='batch size')
    parser.add_argument('--bptt', type=int, default=35, help='sequence length')
    parser.add_argument(
        '--dropout',
        type=float,
        default=0.2,
        help='dropout applied to layers (0 = no dropout)')
    parser.add_argument(
        '--tied',
        action='store_true',
        help='tie the word embedding and softmax weights')
    parser.add_argument('--seed', type=int, default=1111, help='random seed')
    # parser.add_argument('--cuda', action='store_true', help='use CUDA')
    parser.add_argument(
        '--log-interval',
        type=int,
        default=200,
        metavar='N',
        help='report interval')
    parser.add_argument(
        '--save',
        type=str,
        default='lm/pts/temp.pt',
        help='path to save the final model')
    parser.add_argument(
        '--onnx-export',
        type=str,
        default='',
        help='path to export the final model in onnx format')

    parser.add_argument(
        '--nhead',
        type=int,
        default=2,
        help=
        'the number of heads in the encoder/decoder of the transformer model')

    args = parser.parse_args()
    if args.dropout <= 1e-20:
        assert args.drop_mode == 'standard'

    # Set the random seed manually for reproducibility.
    torch.manual_seed(args.seed)
    # if torch.cuda.is_available():
    #     if not args.cuda:
    #         print(
    #             "WARNING: You have a CUDA device, so you should probably run with --cuda"
    #         )

    device = torch.device("cuda")

    ###############################################################################
    # Load data
    ###############################################################################

    corpus = data.Corpus(args.data)

    # Starting from sequential data, batchify arranges the dataset into columns.
    # For instance, with the alphabet as the sequence and batch size 4, we'd get
    # ┌ a g m s ┐
    # │ b h n t │
    # │ c i o u │
    # │ d j p v │
    # │ e k q w │
    # └ f l r x ┘.
    # These columns are treated as independent by the model, which means that the
    # dependence of e. g. 'g' on 'f' can not be learned, but allows more efficient
    # batch processing.

    eval_batch_size = 10
    train_data = batchify(corpus.train, args.batch_size, device=device)
    val_data = batchify(corpus.valid, eval_batch_size, device=device)
    test_data = batchify(corpus.test, eval_batch_size, device=device)

    ###############################################################################
    # Build the model
    ###############################################################################

    ntokens = len(corpus.dictionary)
    if args.model == 'Transformer':
        model = models.TransformerModel(ntokens, args.emsize, args.nhead,
                                        args.nhid, args.nlayers,
                                        args.dropout).to(device)
    else:
        model = models.RNNModel(
            args.model,
            ntokens,
            args.emsize,
            args.nhid,
            args.nlayers,
            args.dropout,
            args.tied,
            drop_mode=args.drop_mode).to(device)

    criterion = nn.CrossEntropyLoss()

    ###############################################################################
    # Training code
    ###############################################################################

    # Loop over epochs.
    # lr = args.lr
    best_val_loss = None

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    # At any point you can hit Ctrl + C to break out of training early.
    if args.train:
        try:
            for epoch in range(1, args.epochs + 1):
                epoch_start_time = time.time()
                train(
                    model=model,
                    corpus=corpus,
                    optimizer=optimizer,
                    criterion=criterion,
                    train_data=train_data,
                    args=args,
                    epoch=epoch)
                val_loss = evaluate(val_data, criterion=criterion, args=args)
                print('-' * 89)
                print(
                    '| end of epoch {:3d} | time: {:5.2f}s | valid loss {:5.2f} | '
                    'valid ppl {:8.2f}'.format(
                        epoch, (time.time() - epoch_start_time), val_loss,
                        math.exp(val_loss)))
                print('-' * 89)
                # Save the model if the validation loss is the best we've seen so far.
                if not best_val_loss or val_loss < best_val_loss:
                    with open(args.save, 'wb') as f:
                        torch.save(model, f)
                    best_val_loss = val_loss
                # else:
                # Anneal the learning rate if no improvement has been seen in the validation dataset.
                # lr /= 4.0
        except KeyboardInterrupt:
            print('-' * 89)
            print('Exiting from training early')

    # Load the best saved model.
    with open(args.save, 'rb') as f:
        model = torch.load(f)
        # after load the rnn params are not a continuous chunk of memory
        # this makes them a continuous chunk, and will speed up forward pass
        # Currently, only rnn model supports flatten_parameters function.
        if args.model in ['RNN_TANH', 'RNN_RELU', 'LSTM', 'GRU']:
            for rnn in model.rnns:
                rnn.flatten_parameters()

    # Run on test data.
    test_loss = evaluate(test_data, criterion=criterion, args=args)
    print('=' * 89)
    print('| End of training | test loss {:5.2f} | test ppl {:8.2f}'.format(
        test_loss, math.exp(test_loss)))
    print('=' * 89)

    if len(args.onnx_export) > 0:
        # Export the model in ONNX format.
        export_onnx(
            args.onnx_export, batch_size=1, seq_len=args.bptt, device=device)


if __name__ == '__main__':
    main()
