from logging import log
from typing import Iterable, Union, List, Dict, Optional, Callable, Tuple, Any
import scipy
from nltk import word_tokenize
import tqdm
import numpy as np
from minicons import utils
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoModelForMaskedLM,
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
)
from transformers.utils.logging import set_verbosity_error
from collections import defaultdict
from itertools import chain
from re import sub
import warnings
from icecream import ic

set_verbosity_error()


TOKEN_OPTIONS = [1]
"""
Option 1
----------
whitespace-separated tokens
It's seven a.m. and 20,000 loud, angry people were there. ->
It's seven a.m. and 20,000 loud, angry people were there.

Option 2
----------
separate all punctuation
It's seven a.m. and 20,000 loud, angry people were there. ->
It ' s seven a . m . and 20 , 000 loud , angry people were there .

Option 3
----------
separate non-word-internal punctuation
It's seven a.m. and 20,000 loud, angry people were there. ->
It's seven a.m . and 20,000 loud , angry people were there .
"""


class LMScorer:
    """
    Base LM scorer class intended to store models and tokenizers along
    with methods to facilitate the analysis of language model output scores.
    """

    def __init__(self, model_name: str, device: Optional[str] = "cpu") -> None:
        """
        :param model_name: name of the model, should either be a path
            to a model (.pt or .bin file) stored locally, or a
            pretrained model stored on the Huggingface Model Hub.
        :type model_name: str
        :param device: device type that the model should be loaded on,
            options: `cpu or cuda:{0, 1, ...}`
        :type device: str, optional
        """
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
        self.device = device
        self.vocab = defaultdict(list)
        # {self.vocab[x.strip()].append(i) for x, i in [(self.tokenizer.decode([i]), i) for i in range(self.tokenizer.vocab_size)]}
        for i in range(self.tokenizer.vocab_size):
            decoded = [(self.tokenizer.decode(i), i)]
            for x, j in decoded:
                self.vocab[x.strip()].append(j)

    def align1(self, surprisals_pre, entropies_pre, token_list, sentence):

        sentence = sentence.replace(
            self.tokenizer.bos_token, self.tokenizer.bos_token + " "
        )

        surprisals_post, entropies_post = [], []
        words = sentence.split()

        i = 0
        try:
            # iterate over words/tokens in the original sentence
            for word in words:

                # build up the original input from the outputs
                # aggregate surprisals additively
                # for entropies of multi-token words, just take the first
                s = ""
                incremental_word_surp = []
                incremental_word_ent = []
                while s != word:
                    if i >= len(token_list):
                        raise IndexError(
                            f"s: {s}, word: {word}, token_list[-1]: {token_list[-1]}"
                        )
                    s += token_list[i]

                    # add a space if required: necessary because the model outputs don't retain spaces
                    if utils.remove_prefix(word, s).startswith(" "):
                        s += " "

                    incremental_word_surp.append(surprisals_pre[i])
                    incremental_word_ent.append(entropies_pre[i])
                    i += 1
                surprisals_post.append(sum(incremental_word_surp))
                entropies_post.append(incremental_word_ent[0])

        except IndexError as e:
            print(f"IndexError: {sentence}")
            print(token_list)
            # if there is a problem with aligning the model outputs with
            # the inputs, then return all nans
            surprisals_post = np.full(len(words), np.nan)
            entropies_post = np.full(len(words), np.nan)

        return words, surprisals_post, entropies_post

    def align(self, surprisals_pre, entropies_pre, token_list, sentence):

        surprisals_post, entropies_post = [], []

        sentence = sub("([.,!?()])", r" \1", sentence)
        sentence = sentence.replace("<|startoftext|>", "<|startoftext|> ")
        sentence = sentence.replace("<|endoftext|>", " <|endoftext|>")
        words = sentence.split()

        i = 0
        try:
            # iterate over words/tokens in the original sentence
            for word in words:

                # need to align model outputs with original inputs
                # build up the original input from the outputs
                # while agreggating surprisals additively
                s = ""
                word_surp = 0
                word_ent = 0
                while s != word:
                    if i >= len(token_list):
                        raise IndexError(
                            f"s: {s}, word: {word}, token_list[-1]: {token_list[-1]}"
                        )
                    s += token_list[i]

                    # add a space if required: necessary because the model outputs
                    # don't retain spaces
                    if utils.remove_prefix(word, s).startswith(" "):
                        s += " "

                    word_surp += surprisals_pre[i]
                    word_ent += entropies_pre[i]
                    i += 1
                surprisals_post.append(word_surp)
                entropies_post.append(word_ent)

        except IndexError as e:
            print(f"IndexError: {sentence}")
            print(token_list)
            # if there is a problem with aligning the model outputs with
            # the inputs, then return all nans
            surprisals_post = np.full(len(words), np.nan)
            entropies_post = np.full(len(words), np.nan)

        return words, surprisals_post, entropies_post

    def add_special_tokens(self, text: Union[str, List[str]]) -> Union[str, List[str]]:
        raise NotImplementedError

    def distribution(self, batch: Iterable) -> torch.Tensor:
        raise NotImplementedError

    def topk(self, distribution: torch.Tensor, k: int = 1) -> Tuple:
        top_k = distribution.topk(k)

        probs = top_k.values.squeeze(1).exp().tolist()
        if k == 1:
            tokens = self.decode(top_k.indices.squeeze(1))
        else:
            tokens = [self.decode(x) for x in top_k.indices.squeeze(1)]

        return tokens, probs

    def query(self, distribution: torch.Tensor, queries: List[str]) -> Tuple:
        # this will be self.vocab tho
        query_ids = [self.vocab[a] for a in queries]
        maxlen = max(map(len, query_ids))
        query_ids = [
            q + [self.tokenizer.pad_token_id] * (maxlen - len(q))
            if len(q) < maxlen
            else q
            for q in query_ids
        ]
        current_batch_size = distribution.shape[0]
        probs = (
            distribution[torch.arange(current_batch_size)[:, None], query_ids]
            .max(1)
            .values.exp()
            .tolist()
        )

        inv_ranks = distribution.argsort().argsort() + 1
        ranks = distribution.shape[1] - inv_ranks + 1
        token_ranks = (
            ranks[torch.arange(current_batch_size)[:, None], query_ids]
            .min(1)
            .values.tolist()
        )

        return probs, token_ranks

    def logprobs(
        self, batch: Iterable, rank: bool = False
    ) -> Union[float, List[float]]:
        warnings.warn(
            "logprobs is deprecated, use compute_stats instead", DeprecationWarning
        )
        raise NotImplementedError

    def compute_stats(
        self, batch: Iterable, rank: bool = False
    ) -> Union[Union[float, int], List[Union[float, int]]]:
        raise NotImplementedError

    def prepare_text(self, text: Union[str, List[str]]) -> Union[str, List[str]]:
        raise NotImplementedError

    def prime_text(
        self, preamble: Union[str, List[str]], stimuli: Union[str, List[str]]
    ) -> Tuple:
        raise NotImplementedError

    def token_score(
        self,
        batch: Union[str, List[str]],
        surprisal: bool = False,
        prob: bool = False,
        base_two: bool = False,
        rank: bool = False,
    ) -> Union[List[Tuple[str, float]], List[Tuple[str, float, int]]]:
        """
        For every input sentence, returns a list of tuples in the following format:
            `(token, score)`,

        where score represents the log-probability (by default) of the token given context. Can also return ranks along with scores.

        :param ``Union[str, List[str]]`` batch: a single sentence or a batch of sentences.
        :param ``bool`` surprisal: If `True`, returns per-word surprisals instead of log-probabilities.
        :param ``bool`` prob: If `True`, returns per-word probabilities instead of log-probabilities.
        :param ``bool`` base_two: If `True`, uses log base 2 instead of natural-log (returns bits of values in case of surprisals)
        :param ``bool`` rank: If `True`, also returns the rank of each word in context (based on the log-probability value)

        :return: A `List` containing a `Tuple` consisting of the word, its associated score, and optionally, its rank.
        :rtype: ``Union[List[Tuple[str, float]], List[Tuple[str, float, int]]]``
        """
        raise NotImplementedError

    def score(
        self, batch: Union[str, List[str]], pool: Callable = torch.mean, *args
    ) -> Union[float, List[float]]:
        """
        DEPRECATED as of v 0.1.18. Check out ``sequence_score`` or ``token_score`` instead!

        Pooled estimates of sentence log probabilities, computed by the
        language model. Pooling is usually done using a function that
        is passed to the method.

        :param batch: a list of sentences that will be passed to the
            language model to score.
        :type batch: Union[str, List[str]]
        :param pool: Pooling function, is selected to be
            `torch.mean()` by default.
        :type pool: Callable

        :return: Float or list of floats specifying the log
            probabilities of the input sentence(s).
        :rtype: Union[float, List[float]]
        """
        warnings.warn(
            "score is deprecated, use sequence_score or token_score instead",
            DeprecationWarning,
        )
        result = self.logprobs(self.prepare_text(batch))
        logprob, _ = list(zip(*result))
        pooled = list(map(lambda x: pool(x, *args).tolist(), logprob))

        return pooled

    def adapt_score(
        self,
        preamble: Union[str, List[str]],
        stimuli: Union[str, List[str]],
        pool: Callable = torch.mean,
        *args,
    ) -> None:
        """
        DEPRECATED as of v 0.1.18. Check out ``partial_score`` instead!
        """
        warnings.warn(
            "adapt_score is deprecated, use partial_score or token_score instead",
            DeprecationWarning,
        )

    def partial_score(
        self,
        preamble: Union[str, List[str]],
        stimuli: Union[str, List[str]],
        reduction: Callable = lambda x: x.mean(0).item(),
        **kwargs,
    ) -> List[float]:
        """
        Pooled estimates of sequence log probabilities (or some modification of it), given a preamble. Pooling is usually done using a function that is passed to the method.

        :param preamble: a batch of preambles or primes passed to the
            language model. This is what the sequence is conditioned on, and the model ignores the word probabilities of this part of the input in estimating the overall score.
        :type preamble: ``Union[str, List[str]]``
        :param stimuli: a batch of sequences (same length as preamble)
            that form the main input consisting of the sequence whose
            score you want to calculate.
        :type stimuli: ``Union[str, List[str]]``
        :param reduction: Reduction function, is selected to be
            ``lambda x: x.mean(0).item()`` by default, which stands for the avg. log-probability per token for each sequence in the batch.
        :type reduction: Callable
        :param kwargs: parameters for the ``compute_stats`` call --

            * `prob` (`bool`): Whether the returned value should be a probability (note that the default reduction method will have to be changed to `lambda x: x.prod(0).item()` to get a meaningful return value)

            * `base_two` (`bool`): whether the returned value should be in base 2 (only works when `prob = False`)

            * `surprisal` (`bool`): whether the returned value should be a surprisal (does not work when `prob = True`)


        :return: List of floats specifying the desired score for the stimuli part of the input, e.g., P(stimuli | preamble).
        :rtype: ``List[float]``
        """
        result = self.compute_stats(
            self.prime_text(preamble, stimuli), **kwargs, return_tensors=True
        )
        logprob = result
        reduced = list(map(reduction, logprob))

        return reduced

    def encode(
        self,
        text: Union[str, List[str]],
        manual_special: bool = True,
        return_tensors: Optional[str] = "pt",
    ) -> Dict:
        """
        Encode a batch of sentences using the model's tokenizer.
        Equivalent of calling `model.tokenizer(input)`

        :param ``Union[str, List[str]]`` text: Input batch/sentence to
            be encoded.
        :param manual_special: Specification of whether special tokens
            will be manually encoded.
        :type manual_special: bool
        :param return_tensors: returned tensor format. Default `'pt'`
        :type manual_special: str

        :return: Encoded batch 
        :rtype: ``Dict``
        """
        sentences = [text] if isinstance(text, str) else text

        if manual_special:
            # manually add special tokens
            sentences = self.add_special_tokens(sentences)
            if return_tensors:
                tokens = self.tokenizer.batch_encode_plus(
                    sentences,
                    add_special_tokens=False,
                    padding="longest",
                    return_attention_mask=True,
                    return_tensors=return_tensors,
                )
        else:
            # mostly for masked LMs
            tokens = self.tokenizer.batch_encode_plus(
                sentences, padding="longest", return_attention_mask=True
            )

        return tokens

    def decode(self, idx: List[int]):
        """
        Decode input ids using the model's tokenizer.

        :param ``List[int]`` idx: List of ids.

        :return: Decoded strings
        :rtype: List[str]
        """
        return [
            self.tokenizer.decode([x]).strip()
            for x in self.tokenizer.convert_tokens_to_ids(
                self.tokenizer.convert_ids_to_tokens(idx)
            )
        ]


class MaskedLMScorer(LMScorer):
    """
    Class for Masked Langauge Models such as BERT, RoBERTa, etc.

    :param model_name: name of the model, should either be a path
        to a model (.pt or .bin file) stored locally, or a
        pretrained model stored on the Huggingface Model Hub.
    :type model_name: str
    :param device: device type that the model should be loaded on,
        options: `cpu or cuda:{0, 1, ...}`
    :type device: str, optional
    """

    def __init__(self, model_name: str, device: Optional[str] = "cpu") -> None:
        """
        :param model_name: name of the model, should either be a path
            to a model (.pt or .bin file) stored locally, or a
            pretrained model stored on the Huggingface Model Hub.

        :type model_name: str
        :param device: device type that the model should be loaded on,
            options: `cpu or cuda:{0, 1, ...}`
        :type device: str, optional
        """
        super(MaskedLMScorer, self).__init__(model_name, device)

        self.model = AutoModelForMaskedLM.from_pretrained(model_name, return_dict=True)
        self.model.to(self.device)
        self.model.eval()

        # define CLS and SEP tokens
        self.bos_token_id = self.tokenizer.cls_token_id
        self.eos_token_id = self.tokenizer.sep_token_id
        self.cls_token_id = self.tokenizer.cls_token_id
        self.sep_token_id = self.tokenizer.sep_token_id
        self.mask_token_id = self.tokenizer.mask_token_id
        self.pad_token_id = self.tokenizer.pad_token_id

    def add_special_tokens(self, text: Union[str, List[str]]) -> List[str]:
        """
        Reformats input text to add special model-dependent tokens.

        :param text: single string or batch of strings to be
            modified.
        :type text: ``Union[str, List[str]]``

        :return: Modified input, containing special tokens as per 
            tokenizer specification
        :rtype: ``List[str]``
        """
        sentences = [text] if isinstance(text, str) else text
        sentences = [
            self.tokenizer.cls_token + " " + sentence + " " + self.tokenizer.sep_token
            for sentence in sentences
        ]

        return sentences

    def mask(
        self, sentence_words: Union[Tuple[str, str], List[Tuple[str, str]]]
    ) -> Tuple[str, str, int]:
        """
        Processes a list of (sentence, word) into input that has the
        word masked out of the sentence. 
        
        Note: only works for masked LMs.

        :param ``Union[Tuple[str], List[Tuple[str]]]`` sentence_words:
            Input consisting of `[(sentence, word)]`, where sentence
            is an input sentence, and word is a word present in the
            sentence that will be masked out.

        :return: Tuple `(sentence, word, length)`
        """
        sentence_words = (
            [sentence_words] if isinstance(sentence_words[0], str) else sentence_words
        )
        sentences, words = list(zip(*sentence_words))
        words = list(words)
        length = len(words)

        sentences = [
            sub(
                rf"(?<![\w\/-])({word})(?=[^\w\/-])",
                self.tokenizer.mask_token,
                sentence,
            )
            for sentence, word in sentence_words
        ]

        return (sentences, words, length)

    def cloze(
        self, sentence_words: Union[Tuple[str, str], List[Tuple[str, str]]]
    ) -> torch.Tensor:
        """
        Runs inference on masked input. 
        Note: only works for masked LMs.

        :param ``Union[Tuple[str], List[Tuple[str]]]`` sentence_words:
            Input consisting of `[(sentence, word)]`, where sentence
            is an input sentence, and word is a word present in the
            sentence that will be masked out and inferred.
        
        :return: A tensor with log probabilities for the desired word
            in context
        """
        sentences, words, length = self.mask(sentence_words)

        encoded = self.tokenizer(sentences, return_tensors="pt")
        encoded = encoded.to(self.device)

        idx = torch.nonzero(
            encoded["input_ids"] == self.tokenizer.mask_token_id, as_tuple=False
        )[:, 1].unsqueeze(1)
        word_idx = self.tokenizer(words, add_special_tokens=False)["input_ids"]
        with torch.no_grad():
            masked_logits = (
                self.model(**encoded)
                .logits[torch.arange(length)[:, None], idx]
                .squeeze()
                .detach()
            )
            if len(sentences) > 1:
                logprobs = masked_logits - masked_logits.logsumexp(1).unsqueeze(1)
                masked_logprobs = (
                    logprobs[torch.arange(len(sentences))[:, None], word_idx]
                    .exp()
                    .squeeze()
                )
            else:
                logprobs = masked_logits - masked_logits.logsumexp(0)
                masked_logprobs = logprobs[word_idx].exp().squeeze()

        return masked_logprobs

    def prepare_text(self, text: Union[str, List[str]]) -> Iterable[Any]:
        """
        Prepares a batch of input text into a format fit to run MLM
        scoring on. 

        Borrows preprocessing algorithm from Salazar et al. (2020), and
        modifies code from the following github repository by simonpri:
        https://github.com/simonepri/lm-scorer
        
        :param text: batch of sentences to be prepared for scoring.

        :return: Batch of formatted input that can be passed to `logprob`
        """
        # converts input text to batch of tensors with every position except the cls and sep token masked
        sentences = [text] if isinstance(text, str) else text

        # idea is to tokenize and then create batches of tokenized instances,
        # but with each token in the sequence replaced by the mask token.

        encoded = self.encode(sentences, manual_special=False)

        token_idx = encoded["input_ids"]
        attention_masks = encoded["attention_mask"]

        masked_tensors = []  # token ids, attention masks, lengths

        for token_ids, attention_mask in zip(token_idx, attention_masks):
            token_ids = torch.tensor(token_ids)
            # final_lengths = len(token_ids) - 2
            attention_mask = torch.tensor(attention_mask)

            token_ids_masked_list = []
            attention_masked_list = []

            effective_token_ids = [
                token
                for token in token_ids
                if token != self.pad_token_id
                and token != self.cls_token_id
                and token != self.sep_token_id
            ]
            effective_length = len(effective_token_ids)

            mask_indices = []
            mask_indices = [[mask_pos] for mask_pos in range(effective_length + 2)]

            # We don't mask the [CLS], [SEP] for now for PLL
            mask_indices = mask_indices[1:-1]

            mask_token_id = self.mask_token_id
            for mask_set in mask_indices:
                token_ids_masked = token_ids.clone()
                token_ids_masked[mask_set] = mask_token_id
                attention_masked = attention_mask.clone()

                attention_masked_list.append(attention_masked)
                token_ids_masked_list.append(token_ids_masked)
            masked_tensors.append(
                (
                    torch.stack(token_ids_masked_list),
                    torch.stack(attention_masked_list),
                    effective_token_ids,
                    len(mask_indices),
                    1,
                )
            )

        return masked_tensors

    def prime_text(
        self, preamble: Union[str, List[str]], stimuli: Union[str, List[str]]
    ) -> Iterable[Any]:
        """
        Prepares a batch of input text into a format fit to run LM
        scoring on. 

        Borrows preprocessing algorithm from Salazar et al. (2020), and
        modifies code from the following github repository by simonpri:
        https://github.com/simonepri/lm-scorer

        :param ``Union[str, List[str]]`` preamble: Batch of prefixes/prime/preambles on which the LM is conditioned.
        :param ``Union[str, List[str]]`` stimuli: Batch of continuations that are scored based on the conditioned text (provided in the ``preamble``). The positions of the elements match their counterparts in the ``preamble``.

        :return: Batch of formatted input that can be passed to
            ``compute_stats``
        """
        preamble_text = [preamble] if isinstance(preamble, str) else preamble
        preamble_encoded = self.encode(preamble_text, False)["input_ids"]
        preamble_lens = []
        for preamble_tokens in preamble_encoded:
            preamble_lens.append(
                len(
                    [
                        token
                        for token in preamble_tokens
                        if token != self.pad_token_id and token != self.sep_token_id
                    ]
                )
            )

        sentences = (
            [preamble + " " + stimuli]
            if isinstance(preamble, str)
            else [p + " " + s for p, s in list(zip(preamble, stimuli))]
        )

        # idea is to tokenize and then create batches of tokenized instances,
        # but with each token in the sequence replaced by the mask token.

        encoded = self.encode(sentences, manual_special=False)

        token_idx = encoded["input_ids"]
        attention_masks = encoded["attention_mask"]

        masked_tensors = []  # token ids, attention masks, lengths

        for i, (token_ids, attention_mask) in enumerate(
            zip(token_idx, attention_masks)
        ):
            token_ids = torch.tensor(token_ids)
            # final_lengths = len(token_ids) - 2
            attention_mask = torch.tensor(attention_mask)

            token_ids_masked_list = []
            attention_masked_list = []

            effective_token_ids = [
                token
                for j, token in enumerate(token_ids)
                if token != self.pad_token_id
                and token != self.cls_token_id
                and token != self.sep_token_id
                and j >= preamble_lens[i]
            ]
            effective_length = len(effective_token_ids) + preamble_lens[i]

            mask_indices = []
            mask_indices = [
                [mask_pos] for mask_pos in range(preamble_lens[i], effective_length + 1)
            ]

            # We don't mask the [CLS], [SEP] for now for PLL
            mask_indices = mask_indices[:-1]

            mask_token_id = self.mask_token_id
            for mask_set in mask_indices:
                token_ids_masked = token_ids.clone()
                token_ids_masked[mask_set] = mask_token_id
                attention_masked = attention_mask.clone()

                attention_masked_list.append(attention_masked)
                token_ids_masked_list.append(token_ids_masked)
            masked_tensors.append(
                (
                    torch.stack(token_ids_masked_list),
                    torch.stack(attention_masked_list),
                    effective_token_ids,
                    len(mask_indices),
                    preamble_lens[i],
                )
            )

        return masked_tensors

    def distribution(self, batch: Iterable) -> torch.Tensor:
        """
        Returns a distribution over the vocabulary of the model.

        :param `Iterable` batch: A batch of inputs fit to pass to a
            transformer LM.

        :return: Tensor consisting of log probabilies over vocab items.
        """
        # takes in prepared text and returns scores for each sentence in batch
        token_ids, attention_masks, effective_token_ids, lengths, offsets = list(
            zip(*batch)
        )
        token_ids = torch.cat(token_ids)
        attention_masks = torch.cat(attention_masks)
        token_ids = token_ids.to(self.device)
        attention_masks = attention_masks.to(self.device)
        effective_token_ids = torch.cat([torch.tensor(x) for x in effective_token_ids])

        indices = list(
            chain.from_iterable(
                [list(range(o, o + n)) for n, o in zip(lengths, offsets)]
            )
        )
        with torch.no_grad():
            output = self.model(token_ids, attention_mask=attention_masks)
            logits = output.logits[torch.arange(sum(lengths)), indices].detach()

        logprob_distribution = logits - logits.logsumexp(1).unsqueeze(1)

        return logprob_distribution

    def cloze_distribution(self, queries: Iterable) -> torch.Tensor:

        """
        Accepts as input batch of [(s_i, bw_i)] where s_i is a prompt with an
        abstract token (bw_i) representing a blank word and returns a distribution
        over the vocabulary of the model.

        :param `Iterable` queries: A batch of [(s_i, bw_i)] where s_i is a prompt with an abstract token (bw_i) representing a blank word

        :return: Tensor contisting of log probabilities over vocab items.
        """

        queries = [queries] if isinstance(queries[0], str) else queries
        prompts, words = list(zip(*queries))

        modified_prompts = self.add_special_tokens(prompts)
        splits = [prompt.split(word) for prompt, word in zip(modified_prompts, words)]
        splits = [[x.strip() for x in s] for s in splits]
        pre, post = list(zip(*splits))
        pre_idx = self.tokenizer(list(pre), add_special_tokens=False, padding=False)[
            "input_ids"
        ]
        mask_idx = [len(item) for item in pre_idx]
        masked = [
            m.replace(w, self.tokenizer.mask_token)
            for m, w in zip(modified_prompts, words)
        ]

        with torch.no_grad():
            encoded = self.tokenizer(
                masked, add_special_tokens=False, return_tensors="pt", padding=True
            )
            encoded = encoded.to(self.device)
            logits = self.model(**encoded)
            presoftmax = logits.logits[torch.arange(len(queries)), mask_idx]
            if "cuda" in self.device:
                presoftmax = presoftmax.detach().cpu()
            else:
                presoftmax = presoftmax.detach()

        logprobs = presoftmax - presoftmax.logsumexp(1).unsqueeze(1)

        return logprobs

    def logprobs(
        self, batch: Iterable, rank=False
    ) -> Union[List[Tuple[torch.Tensor, str]], List[Tuple[torch.Tensor, str, int]]]:
        """
        Returns log probabilities

        :param `Iterable` batch: A batch of inputs fit to pass to a
            transformer LM.
        :param rank: Specifies whether to also return ranks of words.
        :type rank: bool

        :return: List of MLM score metrics and tokens.
        :rtype: Union[List[Tuple[torch.Tensor, str]], List[Tuple[torch.Tensor, str, int]]]
        """
        warnings.warn(
            "logprobs is deprecated, use compute_stats instead", DeprecationWarning
        )
        token_ids, attention_masks, effective_token_ids, lengths, offsets = list(
            zip(*batch)
        )
        token_ids = torch.cat(token_ids)
        attention_masks = torch.cat(attention_masks)
        token_ids = token_ids.to(self.device)
        attention_masks = attention_masks.to(self.device)
        effective_token_ids = torch.cat([torch.tensor(x) for x in effective_token_ids])

        sent_tokens = list(
            map(
                lambda x: self.tokenizer.convert_ids_to_tokens(x.tolist()),
                effective_token_ids.split(lengths),
            )
        )

        indices = list(
            chain.from_iterable(
                [list(range(o, o + n)) for n, o in zip(lengths, offsets)]
            )
        )
        with torch.no_grad():
            output = self.model(token_ids, attention_mask=attention_masks)
            logits = output.logits[torch.arange(sum(lengths)), indices]
            if self.device == "cuda:0" or self.device == "cuda:1":
                logits.detach()

            sent_log_probs = logits - logits.logsumexp(1).unsqueeze(1)
            if rank:
                shape = sent_log_probs.shape
                # inv_ranks = (sent_log_probs).argsort().argsort() + 1
                # ranks = shape[1] - inv_ranks + 1
                ranks = (-1.0 * sent_log_probs).argsort().argsort() + 1
                word_ranks = ranks[torch.arange(shape[0]), effective_token_ids].split(
                    lengths
                )
            sent_log_probs = (
                sent_log_probs[torch.arange(sum(lengths)), effective_token_ids]
                .type(torch.DoubleTensor)
                .split(lengths)
            )
            # print(sent_log_probs)
            # sentence_scores = list(map(lambda x: x.sum().tolist(), logprobs))
            # outputs.append((logprobs, sent_tokens))
            if rank:
                return list(zip(sent_log_probs, sent_tokens, word_ranks))

        return list(zip(sent_log_probs, sent_tokens))

    def compute_stats(
        self,
        batch: Iterable,
        rank: bool = False,
        prob=False,
        base_two: bool = False,
        return_tensors: bool = False,
    ) -> Union[Tuple[List[float], List[float]], List[float]]:
        """
        Primary computational method that processes a batch of prepared sentences and returns per-token scores for each sentence. By default, returns log-probabilities.

        :param ``Iterable`` batch: batched input as processed by ``prepare_text`` or ``prime_text``.
        :param ``bool`` rank: whether the model should also return ranks per word (based on the conditional log-probability of the word in context).
        :param ``bool`` prob: whether the model should return probabilities instead of log-probabilities. Can only be `True` when `base_two` is `False`.
        :param ``bool`` base_two: whether the base of the log should be 2 (usually preferred when reporting results in bits). Can only be `True` when `prob` is `False`.
        :param ``bool`` return_tensors: whether the model should return scores as a list of tensors instead of a list of lists. This is important in some other convenient methods used in the package.

        :return: Either a tuple of lists, each containing probabilities and ranks per token in each sentence passed in the input.
        :rtype: ``Union[Tuple[List[float], List[float]], List[float]]``
        """
        assert not (
            base_two and prob
        ), "cannot both use base (which is for a log), and a probability measure at the same time!"

        token_ids, attention_masks, effective_token_ids, lengths, offsets = list(
            zip(*batch)
        )
        token_ids = torch.cat(token_ids)
        attention_masks = torch.cat(attention_masks)
        token_ids = token_ids.to(self.device)
        attention_masks = attention_masks.to(self.device)
        effective_token_ids = torch.cat([torch.tensor(x) for x in effective_token_ids])

        indices = list(
            chain.from_iterable(
                [list(range(o, o + n)) for n, o in zip(lengths, offsets)]
            )
        )

        with torch.no_grad():
            output = self.model(token_ids, attention_mask=attention_masks)
            logits = output.logits.detach()[torch.arange(sum(lengths)), indices]

        logprob_distribution = logits - logits.logsumexp(1).unsqueeze(1)

        if base_two:
            logprob_distribution = logprob_distribution / torch.tensor(2).log()

        if prob:
            logprob_distribution = logprob_distribution.exp()

        if rank:
            shape = logprob_distribution.shape
            """
            Double argsort trick:
            first argsort returns idxes of values that would return a sorted tensor,
            second argsort returns ranks (0 indexed)

            Proof: https://www.berkayantmen.com/rank.html

            TODO: Try to implement ranking in linear time but across arbitrary dimensions:
            https://stackoverflow.com/a/5284703
            """
            word_ranks = (-1.0 * logprob_distribution).argsort().argsort() + 1
            word_ranks = word_ranks[torch.arange(shape[0]), effective_token_ids].split(
                lengths
            )
            word_ranks = [wr.tolist() for wr in word_ranks]

        scores = (
            logprob_distribution[torch.arange(sum(lengths)), effective_token_ids]
            .type(torch.DoubleTensor)
            .split(lengths)
        )
        scores = [s for s in scores]

        if not return_tensors:
            scores = [s.tolist() for s in scores]

        if rank:
            return scores, word_ranks
        else:
            return scores

    def sequence_score(
        self, batch, reduction=lambda x: x.mean(0).item(), base_two=False
    ):
        """
        TODO: reduction should be a string, if it's a function, specify what kind of function. --> how to ensure it is always that type?
        """
        tokenized = self.prepare_text(batch)
        scores = self.compute_stats(
            tokenized, rank=False, base_two=base_two, return_tensors=True
        )
        reduced = list(map(reduction, scores))
        return reduced

    def token_score(
        self,
        batch: Union[str, List[str]],
        surprisal: bool = False,
        prob: bool = False,
        base_two: bool = False,
        rank: bool = False,
    ) -> Union[List[Tuple[str, float]], List[Tuple[str, float, int]]]:
        """
        For every input sentence, returns a list of tuples in the following format:
            `(token, score)`,

        where score represents the log-probability (by default) of the token given context. Can also return ranks along with scores.

        :param ``Union[str, List[str]]`` batch: a single sentence or a batch of sentences.
        :param ``bool`` surprisal: If `True`, returns per-word surprisals instead of log-probabilities.
        :param ``bool`` prob: If `True`, returns per-word probabilities instead of log-probabilities.
        :param ``bool`` base_two: If `True`, uses log base 2 instead of natural-log (returns bits of values in case of surprisals)
        :param ``bool`` rank: If `True`, also returns the rank of each word in context (based on the log-probability value)

        :return: A `List` containing a `Tuple` consisting of the word, its associated score, and optionally, its rank.
        :rtype: ``Union[List[Tuple[str, float]], List[Tuple[str, float, int]]]``
        """
        assert not (
            surprisal and prob
        ), "cannot both evaluate probability and surprisal at the same time!"
        assert not (
            base_two and prob
        ), "cannot both use base (which is for a log), and a probability measure at the same time!"

        tokenized = self.prepare_text(batch)
        if rank:
            scores, ranks = self.compute_stats(
                tokenized, rank=rank, prob=prob, base_two=base_two, return_tensors=True
            )
        else:
            scores = self.compute_stats(
                tokenized, prob=prob, base_two=base_two, return_tensors=True
            )

        if surprisal:
            scores = [-1.0 * s for s in scores]

        scores = [s.tolist() for s in scores]

        indices = [
            [i.item() for i in indexed if i.item() != self.tokenizer.pad_token_id]
            for indexed in list(zip(*tokenized))[2]
        ]
        tokens = [self.decode(idx) for idx in indices]

        if rank:
            assert len(tokens) == len(scores) == len(ranks)
        else:
            assert len(tokens) == len(scores)

        res = []
        if rank:
            for t, s, r in zip(tokens, scores, ranks):
                res.append(list(zip(t, s, r)))
            # return [list(zip(t, s, r)) for t, s, r in zip(tokens, scores, ranks)]
        else:
            for t, s in zip(tokens, scores):
                res.append(list(zip(t, s)))

        return res


class IncrementalLMScorer(LMScorer):
    """
    Class for Autoregressive or Incremental (or left-to-right) language models such as GPT2, etc.

    :param model_name: name of the model, should either be a path
        to a model (.pt or .bin file) stored locally, or a
        pretrained model stored on the Huggingface Model Hub.
    :type model_name: str
    :param device: device type that the model should be loaded on,
        options: `cpu or cuda:{0, 1, ...}`
    :type device: str, optional
    """

    def __init__(
        self, model_name: str, device: Optional[str] = "cpu", token_option: int = 1
    ) -> None:
        """
        :param model_name: name of the model, should either be a path
            to a model (.pt or .bin file) stored locally, or a
            pretrained model stored on the Huggingface Model Hub.

        :type model_name: str
        :param device: device type that the model should be loaded on,
            options: `cpu or cuda:{0, 1, ...}`
        :type device: str, optional
        """
        super(IncrementalLMScorer, self).__init__(model_name, device)

        self.model = AutoModelForCausalLM.from_pretrained(model_name, return_dict=True)
        self.token_option = token_option

        # define CLS and SEP tokens
        if self.tokenizer.pad_token is None:
            self.tokenizer.add_special_tokens(
                {"additional_special_tokens": ["<|pad|>"]}
            )
            self.tokenizer.pad_token = "<|pad|>"

        if self.tokenizer.bos_token is None:
            self.tokenizer.add_special_tokens(
                {"additional_special_tokens": ["<|bos|>"]}
            )
            self.tokenizer.bos_token = "<|bos|>"

        self.model.resize_token_embeddings(len(self.tokenizer))
        self.model.to(self.device)
        self.model.eval()

    def add_special_tokens(self, text: Union[str, List[str]]) -> Union[str, List[str]]:
        """
        Reformats input text to add special model-dependent tokens.

        :param text: single string or batch of strings to be
            modified.
        :type text: Union[str, List[str]]
        
        :return: Modified input, containing special tokens as per 
            tokenizer specification
        :rtype: Union[float, List[float]]:
        """
        sentences = [text] if isinstance(text, str) else text
        sentences = [self.tokenizer.bos_token + sentence for sentence in sentences]

        return sentences

    def encode(self, text: Union[str, List[str]]) -> dict:
        text = [text] if isinstance(text, str) else text
        return self.tokenizer(text, return_tensors="pt", padding=True)

    def prepare_text(self, text: Union[str, List[str]]) -> Tuple:
        """
        Prepares a batch of input text into a format fit to run LM
        scoring on. 

        :param text: batch of sentences to be prepared for scoring.
        
        :return: Batch of formatted input that can be passed to
            ``compute_stats``
        """
        encoded = self.encode(text)
        offsets = [0] * len(encoded["input_ids"])
        return encoded, offsets

    def prime_text(
        self, preamble: Union[str, List[str]], stimuli: Union[str, List[str]]
    ) -> Tuple:
        """
        Prepares a batch of input text into a format fit to run LM
        scoring on. 

        :param ``Union[str, List[str]]`` preamble: Batch of prefixes/prime/preambles on which the LM is conditioned.
        :param ``Union[str, List[str]]`` stimuli: Batch of continuations that are scored based on the conditioned text (provided in the ``preamble``). The positions of the elements match their counterparts in the ``preamble``.

        :return: Batch of formatted input that can be passed to
            ``compute_stats``
        """
        preamble_text = [preamble] if isinstance(preamble, str) else preamble
        preamble_encoded = self.tokenizer(preamble_text)["input_ids"]
        preamble_lens = []
        for preamble_tokens in preamble_encoded:
            preamble_lens.append(
                len(
                    [
                        token
                        for token in preamble_tokens
                        if token != self.tokenizer.pad_token_id
                        and token != self.tokenizer.sep_token_id
                    ]
                )
                - 1
            )

        sentences = (
            [preamble + " " + stimuli]
            if isinstance(preamble, str)
            else [p + " " + s for p, s in list(zip(preamble, stimuli))]
        )

        return self.encode(sentences), preamble_lens

    def distribution(self, batch: Iterable) -> torch.Tensor:
        """
        Returns a distribution over the vocabulary of the model.

        :param `Iterable` batch: A batch of inputs fit to pass to a
            transformer LM.

        :return: Tensor consisting of log probabilies over vocab items.
        """
        batch, offsets = batch
        ids = batch["input_ids"]
        ids = ids.to(self.device)
        attention_masks = batch["attention_mask"]
        attention_masks = attention_masks.to(self.device)
        nopad_mask = ids != self.tokenizer.pad_token_id

        with torch.no_grad():
            outputs = self.model(ids, attention_mask=attention_masks)
            logits = outputs.logits
            if self.device == "cuda:0" or self.device == "cuda:1":
                logits.detach()

        outputs = []
        for sent_index in range(len(ids)):
            sent_nopad_mask = nopad_mask[sent_index]
            # len(tokens) = len(text[sent_index]) + 1
            sent_tokens = [
                tok
                for i, tok in enumerate(batch.tokens(sent_index))
                if sent_nopad_mask[i] and i > offsets[sent_index] + 1
            ]

            # sent_ids.shape = [len(text[sent_index]) + 1]
            # ignore first token (<|eos|>)
            sent_ids = ids[sent_index, sent_nopad_mask][1:]
            # logits.shape = [len(text[sent_index]) + 1, vocab_size]
            sent_logits = logits[sent_index, sent_nopad_mask][:-1, :]
            sent_logits[:, self.tokenizer.pad_token_id] = float("-inf")

            outputs.append(sent_logits[-1])
        return torch.stack(outputs, 0)

    def next_word_distribution(self, queries: List, surprisal: bool = False):
        """
        Returns the log probability distribution of the next word.
        """
        encoded = self.encode(queries)
        encoded = encoded.to(self.device)
        query_ids = [
            [j for j, i in enumerate(instance) if i != self.tokenizer.pad_token_id][-1]
            for instance in encoded["input_ids"].tolist()
        ]

        logits = self.model(**encoded).logits.detach()
        logits[:, :, self.tokenizer.pad_token_id] = float("-inf")

        logits = logits[torch.arange(len(query_ids)), query_ids]
        logprobs = logits - logits.logsumexp(1).unsqueeze(1)

        if surprisal:
            logprobs = -1.0 * logprobs

        return logprobs

    def compute_stats(
        self,
        batch: Iterable,
        rank: bool = False,
        prob: bool = False,
        base_two: bool = False,
        return_tensors: bool = False,
    ) -> Union[Tuple[List[float], List[float]], List[float]]:
        """
        Primary computational method that processes a batch of prepared sentences and returns per-token scores for each sentence. By default, returns log-probabilities.

        :param ``Iterable`` batch: batched input as processed by ``prepare_text`` or ``prime_text``.
        :param ``bool`` rank: whether the model should also return ranks per word (based on the conditional log-probability of the word in context).
        :param ``bool`` prob: whether the model should return probabilities instead of log-probabilities. Can only be `True` when `base_two` is `False`.
        :param ``bool`` base_two: whether the base of the log should be 2 (usually preferred when reporting results in bits). Can only be `True` when `prob` is `False`.
        :param ``bool`` return_tensors: whether the model should return scores as a list of tensors instead of a list of lists. This is important in some other convenient methods used in the package.

        :return: Either a tuple of lists, each containing probabilities and ranks per token in each sentence passed in the input.
        :rtype: ``Union[Tuple[List[float], List[int]], List[float]]``
        """
        assert not (
            base_two and prob
        ), "cannot both use base (which is for a log), and a probability measure at the same time!"

        encoded, offsets = batch
        encoded = encoded.to(self.device)

        ids = [
            [i for i in instance if i != self.tokenizer.pad_token_id]
            for instance in encoded["input_ids"].tolist()
        ]

        # print("IDs:", ids)

        ## Ignore the probabilities of the first token.
        effective_ids = [id[1:] for id in ids]

        with torch.no_grad():
            logits = self.model(**encoded).logits.detach().cpu()

        # print("Original logits shape:", logits.shape)

        logits[:, :, self.tokenizer.pad_token_id] = float("-inf")

        logits = logits.split([1] * len(offsets))

        ## Set up storage variables
        scores = []
        entropies = []
        if rank:
            ranks = []

        for logit, idx, offset in zip(logits, effective_ids, offsets):
            length = len(idx)
            # logit1 = logit.squeeze(0)[:, :-1]
            # logit2 = logit.squeeze(0)[:, :-1]
            logit1 = logit.squeeze(0)
            logit2 = logit.squeeze(0)

            logprob_distribution = logit1 - logit1.logsumexp(1).unsqueeze(1)
            # print("logprob dist 1 shape:", logprob_distribution.shape)
            query_ids = idx[offset:]
            # print("query IDs:", query_ids)
            if base_two:
                """
                Log_2(X) = log_e(X)/log_e(2) (broadcasted)
                """
                score = -1 * (
                    logprob_distribution[torch.arange(length - offset), query_ids]
                    / torch.tensor(2).log()
                )
                score = torch.cat((torch.full((1,), float("nan")), score))
                score = score.tolist()
            else:
                if prob:
                    score = logprob_distribution[
                        torch.arange(length - offset), query_ids
                    ].exp()
                    score = torch.cat((torch.full((1,), float("nan")), score))
                    score = score.tolist()

                else:
                    score = -1 * (
                        logprob_distribution[torch.arange(length - offset), query_ids]
                    )
                    score = torch.cat((torch.full((1,), float("nan")), score))
                    score = score.tolist()

            logprob_distribution2 = (logit2 - logit2.logsumexp(1).unsqueeze(1))[
                torch.arange(length - offset), :
            ]
            entropy = torch.tensor(
                scipy.stats.entropy(logprob_distribution2.exp(), axis=1,)
            )
            entropy = torch.cat((torch.full((1,), float("nan")), entropy)).tolist()

            if rank:
                """
                Double argsort trick:
                first argsort returns idxes of values that would return a sorted tensor,
                second argsort returns ranks (0 indexed)

                Proof: https://www.berkayantmen.com/rank.html

                TODO: Try to implement ranking in linear time but across arbitrary dimensions:
                https://stackoverflow.com/a/5284703
                """
                word_ranks = (-1.0 * logprob_distribution).argsort().argsort() + 1
                # inv_ranks = logprob_distribution.argsort().argsort() + 1
                # word_ranks = shape[1] - inv_ranks + 1
                word_ranks = word_ranks[torch.arange(length - offset), query_ids]
                word_ranks = torch.cat((torch.full((1,), float("nan")), word_ranks))
                word_ranks = word_ranks.tolist()
                ranks.append(word_ranks)

            scores.append(score)
            entropies.append(entropy)

        if return_tensors:
            scores = [torch.tensor(l) for l in scores]

        if rank:
            return scores, entropies, ranks
        else:
            return scores, entropies

    def get_surprisals_and_entropies(self, sentences, base_two=True):
        tokenized = self.prepare_text(sentences)
        encoded, offsets = tokenized
        scores, entropies = self.compute_stats(tokenized, rank=False, base_two=base_two)

        words_aligned, scores_aligned, entropies_aligned = [], [], []

        token_lists = (
            self.decode(encoded.input_ids[i, :]) for i in range(len(sentences))
        )

        for score, entropy, token_list, sentence in zip(
            scores, entropies, token_lists, sentences
        ):
            # ic(len(score), len(entropy))
            if self.token_option == 1:
                a, b, c = self.align1(score, entropy, token_list, sentence)
            else:
                a, b, c = self.align(score, entropy, token_list, sentence)
            words_aligned.append(a)
            scores_aligned.append(b)
            entropies_aligned.append(c)

        return words_aligned, scores_aligned, entropies_aligned

    def sequence_score(
        self, batch, reduction=lambda x: x.mean(0).item(), base_two=False
    ):
        """
        TODO: reduction should be a string, if it's a function, specify what kind of function. --> how to ensure it is always that type?
        """
        tokenized = self.prepare_text(batch)
        scores = self.compute_stats(
            tokenized, rank=False, base_two=base_two, return_tensors=True
        )
        reduced = list(map(reduction, scores))
        return reduced

    def token_score(
        self,
        batch: Union[str, List[str]],
        surprisal: bool = False,
        prob: bool = False,
        base_two: bool = False,
        rank: bool = False,
    ) -> Union[List[Tuple[str, float]], List[Tuple[str, float, int]]]:
        """
        For every input sentence, returns a list of tuples in the following format:
            `(token, score)`,

        where score represents the log-probability (by default) of the token given context. Can also return ranks along with scores.

        :param ``Union[str, List[str]]`` batch: a single sentence or a batch of sentences.
        :param ``bool`` surprisal: If `True`, returns per-word surprisals instead of log-probabilities.
        :param ``bool`` prob: If `True`, returns per-word probabilities instead of log-probabilities.
        :param ``bool`` base_two: If `True`, uses log base 2 instead of natural-log (returns bits of values in case of surprisals)
        :param ``bool`` rank: If `True`, also returns the rank of each word in context (based on the log-probability value)

        :return: A `List` containing a `Tuple` consisting of the word, its associated score, and optionally, its rank.
        :rtype: ``Union[List[Tuple[str, float]], List[Tuple[str, float, int]]]``
        """

        assert not (
            surprisal and prob
        ), "cannot both evaluate probability and surprisal at the same time!"
        assert not (
            base_two and prob
        ), "cannot both use base (which is for a log), and a probability measure at the same time!"

        tokenized = self.prepare_text(batch)
        if rank:
            scores, ranks = self.compute_stats(
                tokenized, rank=rank, prob=prob, base_two=base_two, return_tensors=True
            )
        else:
            scores = self.compute_stats(
                tokenized, prob=prob, base_two=base_two, return_tensors=True
            )

        if surprisal:
            scores = [-1.0 * s for s in scores]

        scores = [s.tolist() for s in scores]

        indices = [
            [i for i in indexed if i != self.tokenizer.pad_token_id]
            for indexed in tokenized[0]["input_ids"].tolist()
        ]
        tokens = [self.decode(idx) for idx in indices]

        if rank:
            assert len(tokens) == len(scores) == len(ranks)
        else:
            assert len(tokens) == len(scores)

        res = []
        if rank:
            for t, s, r in zip(tokens, scores, ranks):
                if len(t) > len(s):
                    diff = len(t) - len(s)
                    sc = [0.0] * diff + s
                    ra = [0] * diff + r
                    res.append(list(zip(t, sc, ra)))
                else:
                    res.append(list(zip(t, sc, ra)))
            # return [list(zip(t, s, r)) for t, s, r in zip(tokens, scores, ranks)]
        else:
            for t, s in zip(tokens, scores):
                if len(t) > len(s):
                    diff = len(t) - len(s)
                    sc = [0.0] * diff + s
                    res.append(list(zip(t, sc)))
                else:
                    res.append(list(zip(t, sc)))

        return res

    def logprobs(self, batch: Iterable, rank=False) -> Union[float, List[float]]:
        """
        Returns log probabilities

        :param `Iterable` batch: A batch of inputs fit to pass to a
            transformer LM.
        :param rank: Specifies whether to also return ranks of words.
        :type rank: bool

        :return: List of LM score metrics (probability and rank)
            and tokens.
        :rtype: Union[List[Tuple[torch.Tensor, str]], List[Tuple[torch.Tensor, str, int]]]
        """
        warnings.warn(
            "logprobs is deprecated, use compute_stats instead", DeprecationWarning
        )
        batch, offsets = batch
        ids = batch["input_ids"]
        ids = ids.to(self.device)
        attention_masks = batch["attention_mask"]
        attention_masks = attention_masks.to(self.device)
        nopad_mask = ids != self.tokenizer.pad_token_id

        with torch.no_grad():
            outputs = self.model(ids, attention_mask=attention_masks)
            logits = outputs.logits
            if self.device == "cuda:0" or self.device == "cuda:1":
                logits.detach()

        outputs = []
        for sent_index in range(len(ids)):
            sent_nopad_mask = nopad_mask[sent_index]
            # len(tokens) = len(text[sent_index]) + 1
            sent_tokens = [
                tok
                for i, tok in enumerate(batch.tokens(sent_index))
                if sent_nopad_mask[i] and i > offsets[sent_index]
            ]

            # sent_ids.shape = [len(text[sent_index]) + 1]
            # ignore first token (<|eos|>)
            sent_ids = ids[sent_index, sent_nopad_mask][1:]
            # logits.shape = [len(text[sent_index]) + 1, vocab_size]
            sent_logits = logits[sent_index, sent_nopad_mask][:-1, :]
            sent_logits[:, self.tokenizer.pad_token_id] = float("-inf")
            # ids_scores.shape = [seq_len + 1]
            # select only the ids present in the sentence out of all vocab items (as a 2d array)
            sent_ids_scores = sent_logits.gather(1, sent_ids.unsqueeze(1)).squeeze(1)
            # log_prob.shape = [seq_len + 1]
            sent_log_probs = sent_ids_scores - sent_logits.logsumexp(1)

            sent_log_probs = sent_log_probs.type(torch.DoubleTensor)
            sent_log_probs = sent_log_probs[offsets[sent_index] :]
            lengths = len(sent_log_probs)
            if rank:
                shape = sent_logits.shape
                inv_ranks = (sent_logits).argsort().argsort() + 1
                ranks = shape[1] - inv_ranks + 1
                word_ranks = ranks[
                    list(range(shape[0]))[offsets[sent_index] :],
                    sent_ids[offsets[sent_index] :].tolist(),
                ].split(lengths)
                word_ranks = [x[0] for x in word_ranks]
                outputs.append((sent_log_probs, sent_tokens, word_ranks))
            else:
                outputs.append((sent_log_probs, sent_tokens))
            # output = (sent_log_probs.sum(), sent_ids, sent_tokens)
            # outputs.append(output)
        return outputs


class Seq2SeqScorer(LMScorer):
    """
    Class for Autoregressive or Incremental (or left-to-right) language models such as GPT2, etc.

    :param model_name: name of the model, should either be a path
        to a model (.pt or .bin file) stored locally, or a
        pretrained model stored on the Huggingface Model Hub.
    :type model_name: str
    :param device: device type that the model should be loaded on,
        options: `cpu or cuda:{0, 1, ...}`
    :type device: str, optional
    """

    def __init__(self, model_name: str, device: Optional[str] = "cpu") -> None:
        """
        :param model_name: name of the model, should either be a path
            to a model (.pt or .bin file) stored locally, or a
            pretrained model stored on the Huggingface Model Hub.

        :type model_name: str
        :param device: device type that the model should be loaded on,
            options: `cpu or cuda:{0, 1, ...}`
        :type device: str, optional
        """
        super(Seq2SeqScorer, self).__init__(model_name, device)

        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name, return_dict=True)

        # define CLS and SEP tokens
        if self.tokenizer.pad_token is None:
            self.tokenizer.add_special_tokens(
                {"additional_special_tokens": ["<|pad|>"]}
            )
            self.tokenizer.pad_token = "<|pad|>"

        if self.tokenizer.bos_token is None:
            self.tokenizer.add_special_tokens(
                {"additional_special_tokens": ["<|bos|>"]}
            )
            self.tokenizer.bos_token = "<|bos|>"

        self.model.resize_token_embeddings(len(self.tokenizer))
        self.model.to(self.device)
        self.model.eval()

    def add_special_tokens(self, text: Union[str, List[str]]) -> Union[str, List[str]]:
        """
        Reformats input text to add special model-dependent tokens.

        :param text: single string or batch of strings to be
            modified.
        :type text: Union[str, List[str]]
        
        :return: Modified input, containing special tokens as per 
            tokenizer specification
        :rtype: Union[float, List[float]]:
        """
        sentences = [text] if isinstance(text, str) else text
        sentences = [self.tokenizer.bos_token + sentence for sentence in sentences]

        return sentences

    def encode(self, text: Union[str, List[str]]) -> dict:
        text = [text] if isinstance(text, str) else text
        return self.tokenizer(text, return_tensors="pt", padding=True)

    def prepare_text(self, text: Union[str, List[str]]) -> Tuple:
        """
        Prepares a batch of input text into a format fit to run LM
        scoring on. 

        :param text: batch of sentences to be prepared for scoring.
        
        :return: Batch of formatted input that can be passed to
            ``compute_stats``
        """
        encoded = self.encode(text)
        offsets = [0] * len(encoded["input_ids"])
        return encoded, offsets

    def prime_text(
        self, preamble: Union[str, List[str]], stimuli: Union[str, List[str]]
    ) -> Tuple:
        """
        Prepares a batch of input text into a format fit to run LM
        scoring on. 

        :param ``Union[str, List[str]]`` preamble: Batch of prefixes/prime/preambles on which the LM is conditioned.
        :param ``Union[str, List[str]]`` stimuli: Batch of continuations that are scored based on the conditioned text (provided in the ``preamble``). The positions of the elements match their counterparts in the ``preamble``.

        :return: Batch of formatted input that can be passed to
            ``compute_stats``
        """
        preamble_text = [preamble] if isinstance(preamble, str) else preamble
        preamble_encoded = self.tokenizer(preamble_text)["input_ids"]
        preamble_lens = []
        for preamble_tokens in preamble_encoded:
            preamble_lens.append(
                len(
                    [
                        token
                        for token in preamble_tokens
                        if token != self.tokenizer.pad_token_id
                        and token != self.tokenizer.sep_token_id
                    ]
                )
                - 1
            )

        sentences = (
            [preamble + " " + stimuli]
            if isinstance(preamble, str)
            else [p + " " + s for p, s in list(zip(preamble, stimuli))]
        )

        return self.encode(sentences), preamble_lens

    def distribution(self, batch: Iterable) -> torch.Tensor:
        """
        Returns a distribution over the vocabulary of the model.

        :param `Iterable` batch: A batch of inputs fit to pass to a
            transformer LM.

        :return: Tensor consisting of log probabilies over vocab items.
        """
        batch, offsets = batch
        ids = batch["input_ids"]
        ids = ids.to(self.device)
        attention_masks = batch["attention_mask"]
        attention_masks = attention_masks.to(self.device)
        nopad_mask = ids != self.tokenizer.pad_token_id

        with torch.no_grad():
            outputs = self.model(ids, attention_mask=attention_masks)
            logits = outputs.logits
            if self.device == "cuda:0" or self.device == "cuda:1":
                logits.detach()

        outputs = []
        for sent_index in range(len(ids)):
            sent_nopad_mask = nopad_mask[sent_index]
            # len(tokens) = len(text[sent_index]) + 1
            sent_tokens = [
                tok
                for i, tok in enumerate(batch.tokens(sent_index))
                if sent_nopad_mask[i] and i > offsets[sent_index] + 1
            ]

            # sent_ids.shape = [len(text[sent_index]) + 1]
            # ignore first token (<|eos|>)
            sent_ids = ids[sent_index, sent_nopad_mask][1:]
            # logits.shape = [len(text[sent_index]) + 1, vocab_size]
            sent_logits = logits[sent_index, sent_nopad_mask][:-1, :]
            sent_logits[:, self.tokenizer.pad_token_id] = float("-inf")

            outputs.append(sent_logits[-1])
        return torch.stack(outputs, 0)

    def next_word_distribution(self, queries: List, surprisal: bool = False):
        """
        Returns the log probability distribution of the next word.
        """
        encoded = self.encode(queries)
        encoded = encoded.to(self.device)
        query_ids = [
            [j for j, i in enumerate(instance) if i != self.tokenizer.pad_token_id][-1]
            for instance in encoded["input_ids"].tolist()
        ]

        logits = self.model(**encoded).logits.detach()
        logits[:, :, self.tokenizer.pad_token_id] = float("-inf")

        logits = logits[torch.arange(len(query_ids)), query_ids]
        logprobs = logits - logits.logsumexp(1).unsqueeze(1)

        if surprisal:
            logprobs = -1.0 * logprobs

        return logprobs

    def compute_stats(
        self,
        batch: Iterable,
        source: Iterable,
        rank: bool = False,
        prob: bool = False,
        base_two: bool = False,
        return_tensors: bool = False,
    ) -> Union[Tuple[List[float], List[float]], List[float]]:
        """
        Primary computational method that processes a batch of prepared sentences and returns per-token scores for each sentence. By default, returns log-probabilities.

        :param ``Iterable`` batch: batched input as processed by ``prepare_text`` or ``prime_text``.
        :param ``bool`` rank: whether the model should also return ranks per word (based on the conditional log-probability of the word in context).
        :param ``bool`` prob: whether the model should return probabilities instead of log-probabilities. Can only be `True` when `base_two` is `False`.
        :param ``bool`` base_two: whether the base of the log should be 2 (usually preferred when reporting results in bits). Can only be `True` when `prob` is `False`.
        :param ``bool`` return_tensors: whether the model should return scores as a list of tensors instead of a list of lists. This is important in some other convenient methods used in the package.

        :return: Either a tuple of lists, each containing probabilities and ranks per token in each sentence passed in the input.
        :rtype: ``Union[Tuple[List[float], List[int]], List[float]]``
        """
        assert not (
            base_two and prob
        ), "cannot both use base (which is for a log), and a probability measure at the same time!"

        source_encoded, source_offsets = source
        target_encoded, target_offsets = batch
        source_ids = source_encoded["input_ids"].to(self.device)
        target_ids = target_encoded["input_ids"].to(self.device)

        source_ids_list = [
            [i for i in instance if i != self.tokenizer.pad_token_id]
            for instance in source_encoded["input_ids"].tolist()
        ]
        target_ids_list = [
            [i for i in instance if i != self.tokenizer.pad_token_id]
            for instance in target_encoded["input_ids"].tolist()
        ]

        ## Ignore the probabilities of the first token.
        source_effective_ids = [id[1:] for id in source_ids_list]
        target_effective_ids = [id[1:] for id in target_ids_list]

        with torch.no_grad():
            logits = self.model(input_ids=source_ids, labels=target_ids).logits.detach()

        logits[:, :, self.tokenizer.pad_token_id] = float("-inf")

        logits = logits.split([1] * len(target_offsets))

        ## Set up storage variables
        scores = []
        if rank:
            ranks = []

        for logit, idx, offset in zip(logits, target_effective_ids, target_offsets):
            length = len(idx)
            logit = logit.squeeze(0)[:, :-1][
                torch.arange(offset, length),
            ]

            logprob_distribution = logit - logit.logsumexp(1).unsqueeze(1)
            query_ids = idx[offset:]
            if base_two:
                """
                Log_2(X) = log_e(X)/log_e(2) (broadcasted)
                """
                score = (
                    logprob_distribution[torch.arange(length - offset), query_ids]
                    / torch.tensor(2).log()
                ).tolist()
            else:
                if prob:
                    score = (
                        logprob_distribution[torch.arange(length - offset), query_ids]
                        .exp()
                        .tolist()
                    )
                else:
                    score = logprob_distribution[
                        torch.arange(length - offset), query_ids
                    ].tolist()

            if rank:
                # shape = logprob_distribution.shape
                """
                Double argsort trick:
                first argsort returns idxes of values that would return a sorted tensor,
                second argsort returns ranks (0 indexed)

                Proof: https://www.berkayantmen.com/rank.html

                TODO: Try to implement ranking in linear time but across arbitrary dimensions:
                https://stackoverflow.com/a/5284703
                """
                word_ranks = (-1.0 * logprob_distribution).argsort().argsort() + 1
                # inv_ranks = logprob_distribution.argsort().argsort() + 1
                # word_ranks = shape[1] - inv_ranks + 1
                word_ranks = word_ranks[
                    torch.arange(length - offset), query_ids
                ].tolist()
                ranks.append(word_ranks)

            scores.append(score)

        if return_tensors:
            scores = [torch.tensor(l) for l in scores]

        if rank:
            return scores, ranks
        else:
            return scores

    def sequence_score(
        self,
        batch,
        reduction=lambda x: x.mean(0).item(),
        base_two=False,
        source_format="blank",
        source=None,
    ):
        """
        TODO: reduction should be a string, if it's a function, specify what kind of function. --> how to ensure it is always that type?
        """
        if source is not None:
            assert len(source) == len(batch)
            source_format = "custom"

        tokenized = self.prepare_text(batch)
        if source_format == "blank":
            source = [""] * len(batch)
        elif source_format == "copy":
            source = batch
        source = self.prepare_text(source)

        scores = self.compute_stats(
            tokenized, source, rank=False, base_two=base_two, return_tensors=True
        )
        reduced = list(map(reduction, scores))
        return reduced

    def token_score(
        self,
        batch: Union[str, List[str]],
        surprisal: bool = False,
        prob: bool = False,
        base_two: bool = False,
        rank: bool = False,
        source_format: str = "blank",
    ) -> Union[List[Tuple[str, float]], List[Tuple[str, float, int]]]:
        """
        For every input sentence, returns a list of tuples in the following format:
            `(token, score)`,

        where score represents the log-probability (by default) of the token given context. Can also return ranks along with scores.

        :param ``Union[str, List[str]]`` batch: a single sentence or a batch of sentences.
        :param ``bool`` surprisal: If `True`, returns per-word surprisals instead of log-probabilities.
        :param ``bool`` prob: If `True`, returns per-word probabilities instead of log-probabilities.
        :param ``bool`` base_two: If `True`, uses log base 2 instead of natural-log (returns bits of values in case of surprisals)
        :param ``bool`` rank: If `True`, also returns the rank of each word in context (based on the log-probability value)

        :return: A `List` containing a `Tuple` consisting of the word, its associated score, and optionally, its rank.
        :rtype: ``Union[List[Tuple[str, float]], List[Tuple[str, float, int]]]``
        """

        assert not (
            surprisal and prob
        ), "cannot both evaluate probability and surprisal at the same time!"
        assert not (
            base_two and prob
        ), "cannot both use base (which is for a log), and a probability measure at the same time!"

        tokenized = self.prepare_text(batch)
        if source_format == "blank":
            source = [""] * len(batch)
        elif source_format == "copy":
            source = batch
        source = self.prepare_text(source)

        if rank:
            scores, ranks = self.compute_stats(
                tokenized,
                source,
                rank=rank,
                prob=prob,
                base_two=base_two,
                return_tensors=True,
            )
        else:
            scores = self.compute_stats(
                tokenized, source, prob=prob, base_two=base_two, return_tensors=True
            )

        if surprisal:
            scores = [-1.0 * s for s in scores]

        scores = [s.tolist() for s in scores]

        indices = [
            [i for i in indexed if i != self.tokenizer.pad_token_id]
            for indexed in tokenized[0]["input_ids"].tolist()
        ]
        tokens = [self.decode(idx) for idx in indices]

        if rank:
            assert len(tokens) == len(scores) == len(ranks)
        else:
            assert len(tokens) == len(scores)

        res = []
        if rank:
            for t, s, r in zip(tokens, scores, ranks):
                if len(t) > len(s):
                    diff = len(t) - len(s)
                    sc = [0.0] * diff + s
                    ra = [0] * diff + r
                    res.append(list(zip(t, sc, ra)))
                else:
                    res.append(list(zip(t, sc, ra)))
            # return [list(zip(t, s, r)) for t, s, r in zip(tokens, scores, ranks)]
        else:
            for t, s in zip(tokens, scores):
                if len(t) > len(s):
                    diff = len(t) - len(s)
                    sc = [0.0] * diff + s
                    res.append(list(zip(t, sc)))
                else:
                    res.append(list(zip(t, sc)))

        return res

    def logprobs(
        self, batch: Iterable, rank=False, source_format: str = "blank"
    ) -> Union[float, List[float]]:
        """
        Returns log probabilities

        :param `Iterable` batch: A batch of inputs fit to pass to a
            transformer LM.
        :param rank: Specifies whether to also return ranks of words.
        :type rank: bool

        :return: List of LM score metrics (probability and rank)
            and tokens.
        :rtype: Union[List[Tuple[torch.Tensor, str]], List[Tuple[torch.Tensor, str, int]]]
        """
        warnings.warn(
            "logprobs is deprecated, use compute_stats instead", DeprecationWarning
        )
        batch, offsets = batch
        ids = batch["input_ids"]
        ids = ids.to(self.device)
        attention_masks = batch["attention_mask"]
        attention_masks = attention_masks.to(self.device)
        nopad_mask = ids != self.tokenizer.pad_token_id

        with torch.no_grad():
            outputs = self.model(ids, attention_mask=attention_masks)
            logits = outputs.logits
            if self.device == "cuda:0" or self.device == "cuda:1":
                logits.detach()

        outputs = []
        for sent_index in range(len(ids)):
            sent_nopad_mask = nopad_mask[sent_index]
            # len(tokens) = len(text[sent_index]) + 1
            sent_tokens = [
                tok
                for i, tok in enumerate(batch.tokens(sent_index))
                if sent_nopad_mask[i] and i > offsets[sent_index]
            ]

            # sent_ids.shape = [len(text[sent_index]) + 1]
            # ignore first token (<|eos|>)
            sent_ids = ids[sent_index, sent_nopad_mask][1:]
            # logits.shape = [len(text[sent_index]) + 1, vocab_size]
            sent_logits = logits[sent_index, sent_nopad_mask][:-1, :]
            sent_logits[:, self.tokenizer.pad_token_id] = float("-inf")
            # ids_scores.shape = [seq_len + 1]
            # select only the ids present in the sentence out of all vocab items (as a 2d array)
            sent_ids_scores = sent_logits.gather(1, sent_ids.unsqueeze(1)).squeeze(1)
            # log_prob.shape = [seq_len + 1]
            sent_log_probs = sent_ids_scores - sent_logits.logsumexp(1)

            sent_log_probs = sent_log_probs.type(torch.DoubleTensor)
            sent_log_probs = sent_log_probs[offsets[sent_index] :]
            lengths = len(sent_log_probs)
            if rank:
                shape = sent_logits.shape
                inv_ranks = (sent_logits).argsort().argsort() + 1
                ranks = shape[1] - inv_ranks + 1
                word_ranks = ranks[
                    list(range(shape[0]))[offsets[sent_index] :],
                    sent_ids[offsets[sent_index] :].tolist(),
                ].split(lengths)
                word_ranks = [x[0] for x in word_ranks]
                outputs.append((sent_log_probs, sent_tokens, word_ranks))
            else:
                outputs.append((sent_log_probs, sent_tokens))
            # output = (sent_log_probs.sum(), sent_ids, sent_tokens)
            # outputs.append(output)
        return outputs
