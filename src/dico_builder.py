# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

from logging import getLogger
import torch

from .utils import get_nn_avg_dist

import os
import numpy as np
import io

logger = getLogger()

def AL_leastFrequent(current_dico, emb1, emb2, num_words, dictionary_path, word2id1, word2id2):
    assert os.path.isfile(dictionary_path)
    all_word_pairs = []
    not_found = 0
    not_found1 = 0
    not_found2 = 0

    # get the ground truth dictionary and fill up all_word_pairs
    with io.open(dictionary_path, 'r', encoding='utf-8') as f:
        for _, line in enumerate(f):
            assert line == line.lower()
            word1, word2 = line.rstrip().split()
            if word1 in word2id1 and word2 in word2id2:  # if words exist                
                all_word_pairs.append((word1, word2))

    # select least frequent words from the ground truth dictionary that are not present already    
    all_word_pairs = sorted(all_word_pairs, key=lambda x: -1 * word2id1[x[0]])
    new_dico = []    
    num_words_added = 0
    for word1, word2 in all_word_pairs:
        if num_words_added == num_words:  # add num_words to the dico
            break
        if word1 not in current_dico[:,0]: # if word not already in dictionary (we wouldn't query a user for it)
            new_dico.append((word2id1[word1], word2id2[word2]))
            num_words_added = num_words_added + 1
    
    # make torch version of new_dico
    new_dico_torch = torch.LongTensor(num_words_added, 2) 
    for index, (word1, word2) in enumerate(new_dico):
        new_dico_torch[index, 0] = word1
        new_dico_torch[index, 1] = word2

    # add the words to current dictionary    
    return torch.cat((current_dico,new_dico_torch),0)            

def AL_mostFrequent(current_dico, emb1, emb2, num_words, dictionary_path, word2id1, word2id2):
    assert os.path.isfile(dictionary_path)
    all_word_pairs = []
    not_found = 0
    not_found1 = 0
    not_found2 = 0

    # get the ground truth dictionary and fill up all_word_pairs
    with io.open(dictionary_path, 'r', encoding='utf-8') as f:
        for _, line in enumerate(f):
            assert line == line.lower()
            word1, word2 = line.rstrip().split()
            if word1 in word2id1 and word2 in word2id2:  # if words exist                
                all_word_pairs.append((word1, word2))

    # select most frequent words from the ground truth dictionary that are not present already    
    all_word_pairs = sorted(all_word_pairs, key=lambda x: word2id1[x[0]])
    new_dico = []    
    num_words_added = 0
    for word1, word2 in all_word_pairs:
        if num_words_added == num_words:  # add num_words to the dico
            break
        if word1 not in current_dico[:,0]: # if word not already in dictionary (we wouldn't query a user for it)
            new_dico.append((word2id1[word1], word2id2[word2]))
            num_words_added = num_words_added + 1
    
    # make torch version of new_dico
    new_dico_torch = torch.LongTensor(num_words_added, 2) 
    for index, (word1, word2) in enumerate(new_dico):
        new_dico_torch[index, 0] = word1
        new_dico_torch[index, 1] = word2

    # add the words to current dictionary    
    return torch.cat((current_dico,new_dico_torch),0)            

def AL_random(current_dico, emb1, emb2, num_words, dictionary_path, word2id1, word2id2):
    assert os.path.isfile(dictionary_path)
    all_word_pairs = []
    not_found = 0
    not_found1 = 0
    not_found2 = 0

    # get the ground truth dictionary and fill up all_word_pairs
    with io.open(dictionary_path, 'r', encoding='utf-8') as f:
        for _, line in enumerate(f):
            assert line == line.lower()
            word1, word2 = line.rstrip().split()
            if word1 in word2id1 and word2 in word2id2:  # if words exist                
                all_word_pairs.append((word1, word2))

    # select K random words from the ground truth dictionary that are not present already
    np.random.shuffle(all_word_pairs) # shuffle dictionary
    new_dico = []    
    num_words_added = 0
    for word1, word2 in all_word_pairs:
        if num_words_added == num_words:  # add num_words to the dico
            break
        if word1 not in current_dico[:,0]: # if word not already in dictionary (we wouldn't query a user for it)
            new_dico.append((word2id1[word1], word2id2[word2]))
            num_words_added = num_words_added + 1
    
    # make torch version of new_dico
    new_dico_torch = torch.LongTensor(num_words_added, 2) 
    for index, (word1, word2) in enumerate(new_dico):
        new_dico_torch[index, 0] = word1
        new_dico_torch[index, 1] = word2

    # add the words to current dictionary    
    return torch.cat((current_dico,new_dico_torch),0)        

def get_candidates(emb1, emb2, params):
    """
    Get best translation pairs candidates.
    """
    bs = 128

    all_scores = []
    all_targets = []

    # number of source words to consider
    n_src = emb1.size(0)
    if params.dico_max_rank > 0 and not params.dico_method.startswith('invsm_beta_'):
        n_src = params.dico_max_rank

    # nearest neighbors
    if params.dico_method == 'nn':

        # for every source word
        for i in range(0, n_src, bs):

            # compute target words scores
            scores = emb2.mm(emb1[i:min(n_src, i + bs)].transpose(0, 1)).transpose(0, 1)
            best_scores, best_targets = scores.topk(2, dim=1, largest=True, sorted=True)

            # update scores / potential targets
            all_scores.append(best_scores.cpu())
            all_targets.append(best_targets.cpu())

        all_scores = torch.cat(all_scores, 0)
        all_targets = torch.cat(all_targets, 0)

    # inverted softmax
    elif params.dico_method.startswith('invsm_beta_'):

        beta = float(params.dico_method[len('invsm_beta_'):])

        # for every target word
        for i in range(0, emb2.size(0), bs):

            # compute source words scores
            scores = emb1.mm(emb2[i:i + bs].transpose(0, 1))
            scores.mul_(beta).exp_()
            scores.div_(scores.sum(0, keepdim=True).expand_as(scores))

            best_scores, best_targets = scores.topk(2, dim=1, largest=True, sorted=True)

            # update scores / potential targets
            all_scores.append(best_scores.cpu())
            all_targets.append((best_targets + i).cpu())

        all_scores = torch.cat(all_scores, 1)
        all_targets = torch.cat(all_targets, 1)

        all_scores, best_targets = all_scores.topk(2, dim=1, largest=True, sorted=True)
        all_targets = all_targets.gather(1, best_targets)

    # contextual dissimilarity measure
    elif params.dico_method.startswith('csls_knn_'):

        knn = params.dico_method[len('csls_knn_'):]
        assert knn.isdigit()
        knn = int(knn)

        # average distances to k nearest neighbors
        average_dist1 = torch.from_numpy(get_nn_avg_dist(emb2, emb1, knn))
        average_dist2 = torch.from_numpy(get_nn_avg_dist(emb1, emb2, knn))
        average_dist1 = average_dist1.type_as(emb1)
        average_dist2 = average_dist2.type_as(emb2)

        # for every source word
        for i in range(0, n_src, bs):

            # compute target words scores
            scores = emb2.mm(emb1[i:min(n_src, i + bs)].transpose(0, 1)).transpose(0, 1)
            scores.mul_(2)
            scores.sub_(average_dist1[i:min(n_src, i + bs)][:, None] + average_dist2[None, :])
            best_scores, best_targets = scores.topk(2, dim=1, largest=True, sorted=True)

            # update scores / potential targets
            all_scores.append(best_scores.cpu())
            all_targets.append(best_targets.cpu())

        all_scores = torch.cat(all_scores, 0)
        all_targets = torch.cat(all_targets, 0)

    all_pairs = torch.cat([
        torch.arange(0, all_targets.size(0)).long().unsqueeze(1),
        all_targets[:, 0].unsqueeze(1)
    ], 1)

    # sanity check
    assert all_scores.size() == all_pairs.size() == (n_src, 2)

    # sort pairs by score confidence
    diff = all_scores[:, 0] - all_scores[:, 1]
    reordered = diff.sort(0, descending=True)[1]
    all_scores = all_scores[reordered]
    all_pairs = all_pairs[reordered]

    # max dico words rank
    if params.dico_max_rank > 0:
        selected = all_pairs.max(1)[0] <= params.dico_max_rank
        mask = selected.unsqueeze(1).expand_as(all_scores).clone()
        all_scores = all_scores.masked_select(mask).view(-1, 2)
        all_pairs = all_pairs.masked_select(mask).view(-1, 2)

    # max dico size
    if params.dico_max_size > 0:
        all_scores = all_scores[:params.dico_max_size]
        all_pairs = all_pairs[:params.dico_max_size]

    # min dico size
    diff = all_scores[:, 0] - all_scores[:, 1]
    if params.dico_min_size > 0:
        diff[:params.dico_min_size] = 1e9

    # confidence threshold
    if params.dico_threshold > 0:
        mask = diff > params.dico_threshold
        logger.info("Selected %i / %i pairs above the confidence threshold." % (mask.sum(), diff.size(0)))
        mask = mask.unsqueeze(1).expand_as(all_pairs).clone()
        all_pairs = all_pairs.masked_select(mask).view(-1, 2)
    return all_pairs

def build_dictionary(src_emb, tgt_emb, params, s2t_candidates=None, t2s_candidates=None,
                     collected_dico=None):
    """
    Build a training dictionary given current embeddings / mapping.
    """
    logger.info("Building the train dictionary ...")
    s2t = 'S2T' in params.dico_build
    t2s = 'T2S' in params.dico_build
    assert s2t or t2s

    if s2t:
        if s2t_candidates is None:
            s2t_candidates = get_candidates(src_emb, tgt_emb, params)
    if t2s:
        if t2s_candidates is None:
            t2s_candidates = get_candidates(tgt_emb, src_emb, params)
        t2s_candidates = torch.cat([t2s_candidates[:, 1:], t2s_candidates[:, :1]], 1)

    if params.dico_build == 'S2T':
        dico = s2t_candidates
    elif params.dico_build == 'T2S':
        dico = t2s_candidates
    else:
        s2t_candidates = set([(a, b) for a, b in s2t_candidates.numpy()])
        t2s_candidates = set([(a, b) for a, b in t2s_candidates.numpy()])
        if params.dico_build == 'S2T|T2S':
            final_pairs = s2t_candidates | t2s_candidates
        else:
            assert params.dico_build == 'S2T&T2S'
            final_pairs = s2t_candidates & t2s_candidates
            if len(final_pairs) == 0:
                logger.warning("Empty intersection ...")
                return None
        dico = torch.LongTensor(list([[int(a), int(b)] for (a, b) in final_pairs]))

    # combine collected dictionary with automatically aligned dictionary
    # TODO: maybe remove inconsistent entries between dico and collected_dico
    if collected_dico is not None:
        dico = torch.cat((dico, torch.LongTensor(collected_dico)), 0)

    logger.info('New train dictionary of %i pairs.' % dico.size(0))
    return dico.cuda() if params.cuda else dico
