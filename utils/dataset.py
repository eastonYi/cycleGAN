"""
Generates text samples.

Note:
    - separate_domain: It shifts the cipher domain out of the plaintext domain.
        i.e. plaintext - [1,2,3] and encipher using shift or Vigenere it will result in
        something in the same domain ([1,2,3]) so if you set separate_domain to True it
        will add 3 to every cipher text element putting it all in [4,5,6]. To improve
        discriminator's performance.
    - corpus
        - custom: no padding needed.
        - non-custom: index 0 and index 1 are <pad> and <unk>

See tf.flags defined below for all available options.
"""
import operator
import nltk
from collections import deque
import numpy as np

# "The output directory to write data to."
output_dir = "tmp/dataset/cipher"
# "The filename to store training samples under."
train_name = "data-train"
# "The filename to store testing samples under."
test_name = "data-eval"
# "The filename to write the vocabulary to."
vocab_filename = "vocab.txt"
# "Choice of nltk corpus to use. If 'custom' uses vocab defined in plain_vocab. "
# "A full list of available corpii is available at http://www.nltk.org/nltk_data/"
corpus = "custom"
# "Choice of shift or vigenere"
cipher = "shift"
#  "Whether input and output domains should be separated."
separate_domains = False
# "The maximum number of vocabulary allowed, words beyond that will be counted as an unknown value"
vocab_size = 1000
# "The characters (comma separated) used for the plaintext vocabulary.")
plain_vocab = "0,1,2,3,4,5,6,7,8,9"
# "The characters (comma separated) used for the cipher text vocabulary."
cipher_vocab = "10,11,12,13,14,15,16,17,18,19"
# "The distribution (comma separated) for each character of the vocabularies."
distribution = None
# "The number of characters in each sample."
sample_length = 100
# "The key for Vigenere cipher relates to the Vigenere table."
vigenere_key = '345'
# "The number of training samples to produce for each vocab."
num_train = 50000
# "The number of test samples to produce for each vocab."
num_test = 5000
# "Whether the data is to be tokenized character-wise."
char_level = False
# "The size of the shift for the shift cipher. -1 means random"
shift_amount = 3
# "The number of files to shard data into."
num_shards = 1
# "Insert <unk> if word is unknown due to vocab_size."
insert_unk = True

# reserve 0 for pad
_CROP_AMOUNT = 1
_EXTRA_VOCAB_ITEMS = ["<pad>"]


class Layer():
    """A single layer for shift"""

    def __init__(self, vocab, shift):
        """Initialize shift layer

        Args:
            vocab (list of String): the vocabulary
            shift (Integer): the amount of shift apply to the alphabet. Positive number implies
                        shift to the right, negative number implies shift to the left.
        """
        self.shift = shift
        alphabet = vocab
        shifted_alphabet = deque(alphabet)
        shifted_alphabet.rotate(shift)
        self.encrypt = dict(zip(alphabet, list(shifted_alphabet)))
        self.decrypt = dict(zip(list(shifted_alphabet), alphabet))

    def encrypt_character(self, character):
        return self.encrypt[character]

    def decrypt_character(self, character):
        return self.decrypt[character]


def generate_plaintext_random(plain_vocab, distribution):
    """Generates samples of text from the provided vocabulary.
    Returns:
        train_indices (np.array of Integers): random integers generated for training.
            shape = [num_samples, length]
        test_indices (np.array of Integers): random integers generated for testing.
            shape = [num_samples, length]
        plain_vocab     (list of Integers): unique vocabularies.
    """
    plain_vocab = _EXTRA_VOCAB_ITEMS + plain_vocab.split(',')
    distribution = None if distribution is None else [
            float(x.strip()) for x in distribution.split(',')
    ]
    assert distribution is None or sum(distribution) == 1.0

    train_samples = num_train
    test_samples = num_test
    length = sample_length

    train_indices = np.random.choice(
            range(_CROP_AMOUNT, len(plain_vocab)), (train_samples, length),
            p=distribution)
    test_indices = np.random.choice(
            range(_CROP_AMOUNT, len(plain_vocab)), (test_samples, length),
            p=distribution)

    return train_indices, test_indices, plain_vocab


def generate_plaintext_corpus(character_level=False, insert_unk=True, corpus='brown', percentage_training=0.8):
    """Load the corpus and divide it up into a training set and a test set before
    generating TFRecords
    Returns:
        train_indices (np.array of Integers): sentences generated from corpus for training.
            shape = [num_samples, length]
        test_indices (np.array of Integers): sentences generated from corpus for evaluation.
            shape = [num_samples, length]
        plain_vocab     (list of Integers): unique vocabularies in samples.
    """
    # Sanity check provided corpus is a valid carpus available in nltk
    if nltk.download(corpus):
        corpus_method = getattr(nltk.corpus, corpus)
        word_frequency = determine_frequency(corpus_method, character_level)
        vocabulary = trim_vocab(word_frequency, vocab_size)
        plain_corpus, vocabulary = tokenize_corpus(corpus_method, vocabulary, _EXTRA_VOCAB_ITEMS, character_level, insert_unk)
        cutoff = int(percentage_training * len(plain_corpus))
        train_indices, test_indices = plain_corpus[:cutoff], plain_corpus[cutoff:]

        # truncate sentences
        for i in range(len(train_indices)):
            train_indices[i] = train_indices[i][:sample_length]
        for i in range(len(test_indices)):
            test_indices[i] = test_indices[i][:sample_length]

        save_data(train_indices, test_indices, vocabulary)
    else:
        raise ValueError(
                "The corpus you specified isn't available. Fix your corpus flag.")


def encipher_shift(plaintext, plain_vocab, shift, separate_domains=False):
    """Encrypt plain text with a single shift layer
    Args:
        plaintext (list of list of Strings): a list of plain text to encrypt.
        plain_vocab (list of Integer): unique vocabularies being used.
        shift (Integer): number of shift, shift to the right if shift is positive.
    Returns:
        ciphertext (list of Strings): encrypted plain text.
    """
    ciphertext = []
    cipher = Layer(range(_CROP_AMOUNT, len(plain_vocab)), shift)

    for i, sentence in enumerate(plaintext):
        cipher_sentence = []
        for j, character in enumerate(sentence):
            encrypted_char = cipher.encrypt_character(character)
            if separate_domains:
                encrypted_char += len(plain_vocab) - _CROP_AMOUNT
            cipher_sentence.append(encrypted_char)
        ciphertext.append(cipher_sentence)

    return ciphertext


def encipher_vigenere(plaintext, plain_vocab, key, separate_domains=False):
    """Encrypt plain text with given key
    Args:
        plaintext (list of list of Strings): a list of plain text to encrypt.
        plain_vocab (list of Integer): unique vocabularies being used.
        key (list of Integer): key to encrypt cipher using Vigenere table.
    Returns:
        ciphertext (list of Strings): encrypted plain text.
    """
    # generate Vigenere table
    layers = []
    for i in range(len(plain_vocab)):
        layers.append(Layer(range(_CROP_AMOUNT, len(plain_vocab)), i))

    for i, sentence in enumerate(plaintext):
        cipher_sentence = []
        for j, character in enumerate(sentence):
            key_idx = key[j % len(key)]
            encrypted_char = layers[key_idx].encrypt_character(character)
            if separate_domains:
                encrypted_char += len(plain_vocab) - _CROP_AMOUNT
            cipher_sentence.append(encrypted_char)

        yield sentence, cipher_sentence


def cipher_generator(vocab_path, output_dir='data', cipher="vigenere", separate_domains=False):
    # train_plain, test_plain, plain_vocab = generate_plaintext_random()
    plain_vocab = load_vocab(vocab_path)
    train_plain = load_data(plain_vocab, output_dir + '/train.txt')
    test_plain = load_data(plain_vocab, output_dir + '/test.txt')

    if cipher == "shift":
        shift = shift_amount

        if shift == -1:
            shift = np.random.randint(1e5)

        train_cipher = encipher_shift(train_plain, plain_vocab, shift)
        test_cipher = encipher_shift(test_plain, plain_vocab, shift)
    elif cipher == "vigenere":
        key = [int(c) for c in vigenere_key]
        train_plain_cipher = encipher_vigenere(train_plain, plain_vocab, key)
        test_plain_cipher = encipher_vigenere(test_plain, plain_vocab, key)
    else:
        raise Exception("Unknown cipher %s" % cipher)

    return train_plain_cipher, test_plain_cipher, plain_vocab


def string2index(sentences, vocab):
    """Convert string to its corresponding index
    i.e. A -> 0, B -> 1 ... for vocab [A,B,...]
    Args:
      sentences (np.array of String): list of String to convert.
       shape = [num_samples, length]
      vocab (list of String): list of vocabulary
    Returns:
          index (np.array of Integer): list of Integer after conversion
            shape = [num_samples, length]
    """
    alphabet_index = list(range(len(vocab)))
    mapping = dict(zip(vocab, alphabet_index))
    index = []
    for i in range(len(sentences)):
      sentence = []
      for j in range(len(sentences[i])):
          sentence.append(mapping[sentences[i, j]])
      index.append(sentence)
    return index


def trim_vocab(word_frequency, vocab_size):
    """Given the max vocab size n, trim the word_frequency dictionary to only contain the
    top n occurring words
    Args:
      word_frequency (Dictionary): dictionary of word, frequency pairs.
      vocab_size (Integer): the maximum number of vocabulary allowed.
    Returns:
      retval (Dictionary): dictionary containing the top n occurring words as keys
    """
    sorted_vocab = sorted(
          word_frequency.items(), key=operator.itemgetter(1), reverse=True)
    max_count = min(len(word_frequency), vocab_size)
    retval = [k for k, _ in sorted_vocab[:max_count]]
    return retval


def determine_frequency(corpus, character_level):
    """Go through corpus and determine frequency of each individual word
    Args:
      corpus (CategorizedTaggedCorpusReader): corpus object for the corpus being used
    Returns:
      unique_word_count (Dictionary): dictionary of word keys and corresponding frequency
          value
    """
    unique_word_count = dict()
    lengths = []
    for sentence in corpus.sents():
      if not character_level:
          lengths.append(len(sentence))
      else:
          lengths.append(sum(len(word) for word in sentence))
      for word in sentence:
          if character_level:
            for character in word:
                if not character.lower() in unique_word_count:
                  unique_word_count[character.lower()] = 1
                else:
                  unique_word_count[character.lower()] += 1
          else:
            if not word.lower() in unique_word_count:
                unique_word_count[word.lower()] = 1
            else:
                unique_word_count[word.lower()] += 1
    print("Average sentence length: %d" % (sum(lengths) / len(lengths)))
    print("Max sentence length: %d" % (max(lengths)))
    print("Min sentence length: %d" % (min(lengths)))
    return unique_word_count


def tokenize_corpus(corpus, vocabulary, additional_items, character_level, insert_unk):
    """Translate string words into int ids
    Args:
      corpus (CategorizedTaggedCorpusReader): corpus object for the corpus being used.
      vocabulary (Dictionary): vocabulary being used. Also write vocab mapping to file.
    Returns:
      tokenized_corpus (Dictionary): tokenized corpus.
      ordered_vocab (list of String): ordered vocabulary
    """
    unique_words = dict()
    vocab = dict()
    tokenized_corpus = []

    shift = len(additional_items)
    unique_count = shift + 1
    for i, item in enumerate(additional_items):
      vocab[i] = item
    vocab[shift] = "<unk>"
    for sentence in corpus.sents():
      tokenized_sentence = []
      for word in sentence:
          word = word.lower()
          if character_level:
            for character in word:
                if not character in unique_words:
                  if character in vocabulary:
                      unique_words[character] = unique_count
                      vocab[unique_count] = character
                      unique_count += 1
                  else:
                      # shift reserved for other unknown words
                      unique_words[character] = shift

                if unique_words[character] == shift and insert_unk:
                  tokenized_sentence.append(unique_words[character])
                elif unique_words[character] != shift:
                  tokenized_sentence.append(unique_words[character])
          else:
            if not word in unique_words:
                if word in vocabulary:
                  unique_words[word] = unique_count
                  vocab[unique_count] = word
                  unique_count += 1
                else:
                  # shift reserved for other unknown words
                  unique_words[word] = shift

            if unique_words[word] == shift and insert_unk:
                tokenized_sentence.append(unique_words[word])
            elif unique_words[word] != shift:
                tokenized_sentence.append(unique_words[word])

      if len(tokenized_sentence) > 0:
          tokenized_corpus.append(tokenized_sentence)
    sorted_vocab = sorted(vocab.items(), key=lambda a: a[0])
    ordered_vocab = [v for _, v in sorted_vocab]

    return tokenized_corpus, ordered_vocab


def save_data(train_indices, test_indices, vocabulary, output_dir='data'):

    for name, dataset in zip(['/train.txt', '/test.txt'],
                             [train_indices, test_indices]):
        with open(output_dir + name, 'w') as fw:
            for sent in dataset:
                line = ' '.join(vocabulary[i] for i in sent)
                fw.write(line + "\n")

    with open(output_dir + '/vocab.txt', 'w') as fw:
        for token in vocabulary:
            fw.write(token + '\n')



def load_vocab(path='data/vocab.txt'):
    vocab = []
    with open(path) as f:
        for line in f:
            vocab.append(line.strip())
    return vocab


def load_data(vocab, data_path='data/train.txt'):
    with open(data_path) as f:
        for line in f:
            new_line = []
            for i in line.strip().split():
                try:
                    new_line.append(vocab.index(i))
                except:
                    new_line.append(vocab.index('<unk>'))
            yield new_line


if __name__ == '__main__':
    # generate_plaintext_corpus()
    # train_indices, test_indices, vocab = load_data()
    train_data, test_data, plain_vocab = cipher_generator()
