"""Build a gold dictionary."""

from argparse import ArgumentParser


def load_vocab(lang, max_vocab):
    """Load vocabulary from an embedding file"""
    emb_path = 'data/fasttext/wiki.%s.vec' % lang
    vocab = []
    with open(emb_path, 'r') as f:
        for i, line in enumerate(f):
            if i > 0:
                vocab.append(line.split()[0])
            if len(vocab) == max_vocab:
                break
    return vocab


def main():
    parser = ArgumentParser()
    parser.add_argument('--src', required=True, help='source language')
    parser.add_argument('--tgt', required=True, help='target language')
    parser.add_argument('--max_vocab', type=int, default=20000,
                        help='max vocabulary')
    args = parser.parse_args()

    src_vocab = load_vocab(args.src, 200000)
    tgt_vocab = load_vocab(args.tgt, 200000)
    dico_path = ('data/crosslingual/dictionaries/%s-%s.txt' % (args.src, args.tgt))
    dico = dict()
    with open(dico_path, 'r') as f:
        for line in f:
            w1, w2 = line.strip().split()
            if w1 != w2 and w1 in src_vocab and w2 in tgt_vocab and w1 not in dico:
                dico[w1] = w2
                print w1, w2


if __name__ == '__main__':
    main()
