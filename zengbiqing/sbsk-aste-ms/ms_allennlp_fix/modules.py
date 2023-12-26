from typing import (
    Union, List, Optional, NamedTuple,
    Iterator, Any, cast, Sequence, Tuple, BinaryIO,
    Dict
)
import io, inspect, itertools, re
import logging
import tarfile, zipfile
from tqdm import tqdm
# import h5py

logger = logging.getLogger(__name__)

from .type_define import (
    SpanExtractor, ConfigurationError,  TokenEmbedder,
    Vocabulary, TextFieldEmbedder, Token,
    TokenIndexer, IndexedTokenList
)
from .file_utils import (
    get_file_extension, is_url_or_existing_file
)
from .field_type import TextFieldTensors
import ms_allennlp_fix.util as util
from Hyperparameters import Hyperparameter as hp

import numpy as np

import msadapter.pytorch as torch
import msadapter.pytorch.nn as nn
from msadapter.pytorch.nn import Parameter
from msadapter.pytorch.nn.functional import embedding

from mindformers import BertConfig, BertModel

class FeedForward(torch.nn.Module):
    """
    This `Module` is a feed-forward neural network, just a sequence of `Linear` layers with
    activation functions in between.

    # Parameters

    input_dim : `int`, required
        The dimensionality of the input.  We assume the input has shape `(batch_size, input_dim)`.
    num_layers : `int`, required
        The number of `Linear` layers to apply to the input.
    hidden_dims : `Union[int, List[int]]`, required
        The output dimension of each of the `Linear` layers.  If this is a single `int`, we use
        it for all `Linear` layers.  If it is a `List[int]`, `len(hidden_dims)` must be
        `num_layers`.
    activations : `Union[Activation, List[Activation]]`, required
        The activation function to use after each `Linear` layer.  If this is a single function,
        we use it after all `Linear` layers.  If it is a `List[Activation]`,
        `len(activations)` must be `num_layers`. Activation must have torch.nn.Module type.
    dropout : `Union[float, List[float]]`, optional (default = `0.0`)
        If given, we will apply this amount of dropout after each layer.  Semantics of `float`
        versus `List[float]` is the same as with other parameters.

    # Examples

    ```python
    FeedForward(124, 2, [64, 32], torch.nn.ReLU(), 0.2)
    #> FeedForward(
    #>   (_activations): ModuleList(
    #>     (0): ReLU()
    #>     (1): ReLU()
    #>   )
    #>   (_linear_layers): ModuleList(
    #>     (0): Linear(in_features=124, out_features=64, bias=True)
    #>     (1): Linear(in_features=64, out_features=32, bias=True)
    #>   )
    #>   (_dropout): ModuleList(
    #>     (0): Dropout(p=0.2, inplace=False)
    #>     (1): Dropout(p=0.2, inplace=False)
    #>   )
    #> )
    ```
    """

    def __init__(
        self,
        input_dim: int,
        num_layers: int,
        hidden_dims: Union[int, List[int]],
        activations,
        dropout: Union[float, List[float]] = 0.0,
    ) -> None:

        super().__init__()
        if not isinstance(hidden_dims, list):
            hidden_dims = [hidden_dims] * num_layers  # type: ignore
        if not isinstance(activations, list):
            activations = [activations] * num_layers  # type: ignore
        if not isinstance(dropout, list):
            dropout = [dropout] * num_layers  # type: ignore

        if len(hidden_dims) != num_layers:
            raise ConfigurationError(
                "len(hidden_dims) (%d) != num_layers (%d)" % (len(hidden_dims), num_layers)
            )
        if len(activations) != num_layers:
            raise ConfigurationError(
                "len(activations) (%d) != num_layers (%d)" % (len(activations), num_layers)
            )
        if len(dropout) != num_layers:
            raise ConfigurationError(
                "len(dropout) (%d) != num_layers (%d)" % (len(dropout), num_layers)
            )

        self._activations = torch.nn.ModuleList(activations)
        input_dims = [input_dim] + hidden_dims[:-1]
        linear_layers = []
        for layer_input_dim, layer_output_dim in zip(input_dims, hidden_dims):
            linear_layers.append(torch.nn.Linear(layer_input_dim, layer_output_dim))
        self._linear_layers = torch.nn.ModuleList(linear_layers)
        dropout_layers = [torch.nn.Dropout(p=value) for value in dropout]
        self._dropout = torch.nn.ModuleList(dropout_layers)
        self._output_dim = hidden_dims[-1]
        self.input_dim = input_dim

    def get_output_dim(self):
        return self._output_dim

    def get_input_dim(self):
        return self.input_dim

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:

        output = inputs
        for layer, activation, dropout in zip(
            self._linear_layers, self._activations, self._dropout
        ):
            output = dropout(activation(layer(output)))
        return output


class TimeDistributed(torch.nn.Module):
    """
    Given an input shaped like `(batch_size, time_steps, [rest])` and a `Module` that takes
    inputs like `(batch_size, [rest])`, `TimeDistributed` reshapes the input to be
    `(batch_size * time_steps, [rest])`, applies the contained `Module`, then reshapes it back.

    Note that while the above gives shapes with `batch_size` first, this `Module` also works if
    `batch_size` is second - we always just combine the first two dimensions, then split them.

    It also reshapes keyword arguments unless they are not tensors or their name is specified in
    the optional `pass_through` iterable.
    """

    def __init__(self, module):
        super().__init__()
        self._module = module

     
    def forward(self, *inputs, pass_through: List[str] = None, **kwargs):

        pass_through = pass_through or []

        reshaped_inputs = [self._reshape_tensor(input_tensor) for input_tensor in inputs]

        # Need some input to then get the batch_size and time_steps.
        some_input = None
        if inputs:
            some_input = inputs[-1]

        reshaped_kwargs = {}
        for key, value in kwargs.items():
            if isinstance(value, torch.Tensor) and key not in pass_through:
                if some_input is None:
                    some_input = value

                value = self._reshape_tensor(value)

            reshaped_kwargs[key] = value

        reshaped_outputs = self._module(*reshaped_inputs, **reshaped_kwargs)

        if some_input is None:
            raise RuntimeError("No input tensor to time-distribute")

        # Now get the output back into the right shape.
        # (batch_size, time_steps, **output_size)
        new_size = some_input.size()[:2] + reshaped_outputs.size()[1:]
        outputs = reshaped_outputs.contiguous().view(new_size)

        return outputs

    @staticmethod
    def _reshape_tensor(input_tensor:torch.Tensor):
        input_size = input_tensor.shape
        if len(input_size) <= 2:
            raise RuntimeError(f"No dimension to distribute: {input_size}")
        # Squash batch_size and time_steps into a single axis; result has shape
        # (batch_size * time_steps, **input_size).
        squashed_shape = [-1] + list(input_size[2:])
        return input_tensor.contiguous().view(*squashed_shape)

class Embedding(TokenEmbedder):
    """
    A more featureful embedding module than the default in Pytorch.  Adds the ability to:

        1. embed higher-order inputs
        2. pre-specify the weight matrix
        3. use a non-trainable embedding
        4. project the resultant embeddings to some other dimension (which only makes sense with
           non-trainable embeddings).

    Note that if you are using our data API and are trying to embed a
    [`TextField`](../../data/fields/text_field.md), you should use a
    [`TextFieldEmbedder`](../text_field_embedders/text_field_embedder.md) instead of using this directly.

    Registered as a `TokenEmbedder` with name "embedding".

    # Parameters

    num_embeddings : `int`
        Size of the dictionary of embeddings (vocabulary size).
    embedding_dim : `int`
        The size of each embedding vector.
    projection_dim : `int`, optional (default=`None`)
        If given, we add a projection layer after the embedding layer.  This really only makes
        sense if `trainable` is `False`.
    weight : `torch.FloatTensor`, optional (default=`None`)
        A pre-initialised weight matrix for the embedding lookup, allowing the use of
        pretrained vectors.
    padding_index : `int`, optional (default=`None`)
        If given, pads the output with zeros whenever it encounters the index.
    trainable : `bool`, optional (default=`True`)
        Whether or not to optimize the embedding parameters.
    max_norm : `float`, optional (default=`None`)
        If given, will renormalize the embeddings to always have a norm lesser than this
    norm_type : `float`, optional (default=`2`)
        The p of the p-norm to compute for the max_norm option
    scale_grad_by_freq : `bool`, optional (default=`False`)
        If given, this will scale gradients by the frequency of the words in the mini-batch.
    sparse : `bool`, optional (default=`False`)
        Whether or not the Pytorch backend should use a sparse representation of the embedding weight.
    vocab_namespace : `str`, optional (default=`None`)
        In case of fine-tuning/transfer learning, the model's embedding matrix needs to be
        extended according to the size of extended-vocabulary. To be able to know how much to
        extend the embedding-matrix, it's necessary to know which vocab_namspace was used to
        construct it in the original training. We store vocab_namespace used during the original
        training as an attribute, so that it can be retrieved during fine-tuning.
    pretrained_file : `str`, optional (default=`None`)
        Path to a file of word vectors to initialize the embedding matrix. It can be the
        path to a local file or a URL of a (cached) remote file. Two formats are supported:
            * hdf5 file - containing an embedding matrix in the form of a torch.Tensor;
            * text file - an utf-8 encoded text file with space separated fields.
    vocab : `Vocabulary`, optional (default = `None`)
        Used to construct an embedding from a pretrained file.

        In a typical AllenNLP configuration file, this parameter does not get an entry under the
        "embedding", it gets specified as a top-level parameter, then is passed in to this module
        separately.

    # Returns

    An Embedding module.
    """

    def __init__(
        self,
        embedding_dim: int,
        num_embeddings: int = None,
        projection_dim: int = None,
        weight: torch.FloatTensor = None,
        padding_index: int = None,
        trainable: bool = True,
        max_norm: float = None,
        norm_type: float = 2.0,
        scale_grad_by_freq: bool = False,
        sparse: bool = False,
        vocab_namespace: str = "tokens",
        pretrained_file: str = None,
        vocab: Vocabulary = None,
    ) -> None:
        super().__init__()

        if num_embeddings is None and vocab is None:
            raise ConfigurationError(
                "Embedding must be constructed with either num_embeddings or a vocabulary."
            )

        _vocab_namespace: Optional[str] = vocab_namespace
        if num_embeddings is None:
            num_embeddings = vocab.get_vocab_size(_vocab_namespace)  # type: ignore
        else:
            # If num_embeddings is present, set default namespace to None so that extend_vocab
            # call doesn't misinterpret that some namespace was originally used.
            _vocab_namespace = None  # type: ignore

        self.num_embeddings = num_embeddings
        self.padding_index = padding_index
        self.max_norm = max_norm
        self.norm_type = norm_type
        self.scale_grad_by_freq = scale_grad_by_freq
        self.sparse = sparse
        self._vocab_namespace = _vocab_namespace
        self._pretrained_file = pretrained_file

        self.output_dim = projection_dim or embedding_dim

        if weight is not None and pretrained_file:
            raise ConfigurationError(
                "Embedding was constructed with both a weight and a pretrained file."
            )

        elif pretrained_file is not None:

            if vocab is None:
                raise ConfigurationError(
                    "To construct an Embedding from a pretrained file, you must also pass a vocabulary."
                )

            # If we're loading a saved model, we don't want to actually read a pre-trained
            # embedding file - the embeddings will just be in our saved weights, and we might not
            # have the original embedding file anymore, anyway.

            # TODO: having to pass tokens here is SUPER gross, but otherwise this breaks the
            # extend_vocab method, which relies on the value of vocab_namespace being None
            # to infer at what stage the embedding has been constructed. Phew.
            weight = _read_pretrained_embeddings_file(
                pretrained_file, embedding_dim, vocab, vocab_namespace
            )
            self.weight = torch.nn.Parameter(weight, requires_grad=trainable)

        elif weight is not None:
            self.weight = torch.nn.Parameter(weight, requires_grad=trainable)

        else:
            weight = torch.FloatTensor(num_embeddings, embedding_dim)
            self.weight = torch.nn.Parameter(weight, requires_grad=trainable)
            torch.nn.init.xavier_uniform_(self.weight)

        # Whatever way we have constructed the embedding, it should be consistent with
        # num_embeddings and embedding_dim.
        if self.weight.size() != (num_embeddings, embedding_dim):
            raise ConfigurationError(
                "A weight matrix was passed with contradictory embedding shapes."
            )

        if self.padding_index is not None:
            self.weight.data[self.padding_index].fill_(0)

        if projection_dim:
            self._projection = torch.nn.Linear(embedding_dim, projection_dim)
        else:
            self._projection = None

     
    def get_output_dim(self) -> int:
        return self.output_dim

     
    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        # tokens may have extra dimensions (batch_size, d1, ..., dn, sequence_length),
        # but embedding expects (batch_size, sequence_length), so pass tokens to
        # util.combine_initial_dims (which is a no-op if there are no extra dimensions).
        # Remember the original size.
        original_size = tokens.size()
        tokens = util.combine_initial_dims(tokens)

        embedded = embedding(
            tokens,
            self.weight,
            padding_idx=self.padding_index,
            max_norm=self.max_norm,
            norm_type=self.norm_type,
            scale_grad_by_freq=self.scale_grad_by_freq,
            sparse=self.sparse,
        )

        # Now (if necessary) add back in the extra dimensions.
        embedded = util.uncombine_initial_dims(embedded, original_size)

        if self._projection:
            projection = self._projection
            for _ in range(embedded.dim() - 2):
                projection = TimeDistributed(projection)
            embedded = projection(embedded)
        return embedded

    def extend_vocab(
        self,
        extended_vocab: Vocabulary,
        vocab_namespace: str = None,
        extension_pretrained_file: str = None,
        model_path: str = None,
    ):
        """
        Extends the embedding matrix according to the extended vocabulary.
        If extension_pretrained_file is available, it will be used for initializing the new words
        embeddings in the extended vocabulary; otherwise we will check if _pretrained_file attribute
        is already available. If none is available, they will be initialized with xavier uniform.

        # Parameters

        extended_vocab : `Vocabulary`
            Vocabulary extended from original vocabulary used to construct
            this `Embedding`.
        vocab_namespace : `str`, (optional, default=`None`)
            In case you know what vocab_namespace should be used for extension, you
            can pass it. If not passed, it will check if vocab_namespace used at the
            time of `Embedding` construction is available. If so, this namespace
            will be used or else extend_vocab will be a no-op.
        extension_pretrained_file : `str`, (optional, default=`None`)
            A file containing pretrained embeddings can be specified here. It can be
            the path to a local file or an URL of a (cached) remote file. Check format
            details in `from_params` of `Embedding` class.
        model_path : `str`, (optional, default=`None`)
            Path traversing the model attributes upto this embedding module.
            Eg. "_text_field_embedder.token_embedder_tokens". This is only useful
            to give a helpful error message when extend_vocab is implicitly called
            by train or any other command.
        """
        # Caveat: For allennlp v0.8.1 and below, we weren't storing vocab_namespace as an attribute,
        # knowing which is necessary at time of embedding vocab extension. So old archive models are
        # currently unextendable.

        vocab_namespace = vocab_namespace or self._vocab_namespace
        if not vocab_namespace:
            # It's not safe to default to "tokens" or any other namespace.
            logging.info(
                "Loading a model trained before embedding extension was implemented; "
                "pass an explicit vocab namespace if you want to extend the vocabulary."
            )
            return

        extended_num_embeddings = extended_vocab.get_vocab_size(vocab_namespace)
        if extended_num_embeddings == self.num_embeddings:
            # It's already been extended. No need to initialize / read pretrained file in first place (no-op)
            return

        if extended_num_embeddings < self.num_embeddings:
            raise ConfigurationError(
                f"Size of namespace, {vocab_namespace} for extended_vocab is smaller than "
                f"embedding. You likely passed incorrect vocab or namespace for extension."
            )

        # Case 1: user passed extension_pretrained_file and it's available.
        if extension_pretrained_file and is_url_or_existing_file(extension_pretrained_file):
            # Don't have to do anything here, this is the happy case.
            pass
        # Case 2: user passed extension_pretrained_file and it's not available
        if extension_pretrained_file:
            raise ConfigurationError(
                f"You passed pretrained embedding file {extension_pretrained_file} "
                f"for model_path {model_path} but it's not available."
            )
        # Case 3: user didn't pass extension_pretrained_file, but pretrained_file attribute was
        # saved during training and is available.
        elif is_url_or_existing_file(self._pretrained_file):
            extension_pretrained_file = self._pretrained_file
        # Case 4: no file is available, hope that pretrained embeddings weren't used in the first place and warn
        else:
            extra_info = (
                f"Originally pretrained_file was at " f"{self._pretrained_file}. "
                if self._pretrained_file
                else ""
            )
            # It's better to warn here and not give error because there is no way to distinguish between
            # whether pretrained-file wasn't used during training or user forgot to pass / passed incorrect
            # mapping. Raising an error would prevent fine-tuning in the former case.
            logging.warning(
                f"Embedding at model_path, {model_path} cannot locate the pretrained_file. "
                f"{extra_info} If you are fine-tuning and want to use using pretrained_file for "
                f"embedding extension, please pass the mapping by --embedding-sources argument."
            )

        embedding_dim = self.weight.data.shape[-1]
        if not extension_pretrained_file:
            extra_num_embeddings = extended_num_embeddings - self.num_embeddings
            extra_weight = torch.FloatTensor(extra_num_embeddings, embedding_dim)
            torch.nn.init.xavier_uniform_(extra_weight)
        else:
            # It's easiest to just reload the embeddings for the entire vocab,
            # then only keep the ones we need.
            whole_weight = _read_pretrained_embeddings_file(
                extension_pretrained_file, embedding_dim, extended_vocab, vocab_namespace
            )
            extra_weight = whole_weight[self.num_embeddings :, :]

        device = self.weight.data.device
        extended_weight = torch.cat([self.weight.data, extra_weight.to(device)], dim=0)
        self.weight = torch.nn.Parameter(extended_weight, requires_grad=self.weight.requires_grad)
        self.num_embeddings = extended_num_embeddings


def _read_pretrained_embeddings_file(
    file_uri: str, embedding_dim: int, vocab: Vocabulary, namespace: str = "tokens"
) -> torch.FloatTensor:
    """
    Returns and embedding matrix for the given vocabulary using the pretrained embeddings
    contained in the given file. Embeddings for tokens not found in the pretrained embedding file
    are randomly initialized using a normal distribution with mean and standard deviation equal to
    those of the pretrained embeddings.

    We support two file formats:

        * text format - utf-8 encoded text file with space separated fields: [word] [dim 1] [dim 2] ...
          The text file can eventually be compressed, and even resides in an archive with multiple files.
          If the file resides in an archive with other files, then `embeddings_filename` must
          be a URI "(archive_uri)#file_path_inside_the_archive"

        * hdf5 format - hdf5 file containing an embedding matrix in the form of a torch.Tensor.

    If the filename ends with '.hdf5' or '.h5' then we load from hdf5, otherwise we assume
    text format.

    # Parameters

    file_uri : `str`, required.
        It can be:

        * a file system path or a URL of an eventually compressed text file or a zip/tar archive
          containing a single file.

        * URI of the type `(archive_path_or_url)#file_path_inside_archive` if the text file
          is contained in a multi-file archive.

    vocab : `Vocabulary`, required.
        A Vocabulary object.
    namespace : `str`, (optional, default=`"tokens"`)
        The namespace of the vocabulary to find pretrained embeddings for.
    trainable : `bool`, (optional, default=`True`)
        Whether or not the embedding parameters should be optimized.

    # Returns

    A weight matrix with embeddings initialized from the read file.  The matrix has shape
    `(vocab.get_vocab_size(namespace), embedding_dim)`, where the indices of words appearing in
    the pretrained embedding file are initialized to the pretrained embedding value.
    """
    file_ext = get_file_extension(file_uri)
    if file_ext in [".h5", ".hdf5"]:
        raise ValueError("封印了hdf5读取")
        return None
        # return _read_embeddings_from_hdf5(file_uri, embedding_dim, vocab, namespace)

    return _read_embeddings_from_text_file(file_uri, embedding_dim, vocab, namespace)


def _read_embeddings_from_text_file(
    file_uri: str, embedding_dim: int, vocab: Vocabulary, namespace: str = "tokens"
) -> torch.FloatTensor:
    """
    Read pre-trained word vectors from an eventually compressed text file, possibly contained
    inside an archive with multiple files. The text file is assumed to be utf-8 encoded with
    space-separated fields: [word] [dim 1] [dim 2] ...

    Lines that contain more numerical tokens than `embedding_dim` raise a warning and are skipped.

    The remainder of the docstring is identical to `_read_pretrained_embeddings_file`.
    """
    tokens_to_keep = set(vocab.get_index_to_token_vocabulary(namespace).values())
    vocab_size = vocab.get_vocab_size(namespace)
    embeddings = {}

    # First we read the embeddings from the file, only keeping vectors for the words we need.
    logger.info("Reading pretrained embeddings from file")

    with EmbeddingsTextFile(file_uri) as embeddings_file:
        for line in tqdm(embeddings_file):
            token = line.split(" ", 1)[0]
            if token in tokens_to_keep:
                fields = line.rstrip().split(" ")
                if len(fields) - 1 != embedding_dim:
                    # Sometimes there are funny unicode parsing problems that lead to different
                    # fields lengths (e.g., a word with a unicode space character that splits
                    # into more than one column).  We skip those lines.  Note that if you have
                    # some kind of long header, this could result in all of your lines getting
                    # skipped.  It's hard to check for that here; you just have to look in the
                    # embedding_misses_file and at the model summary to make sure things look
                    # like they are supposed to.
                    logger.warning(
                        "Found line with wrong number of dimensions (expected: %d; actual: %d): %s",
                        embedding_dim,
                        len(fields) - 1,
                        line,
                    )
                    continue

                vector = np.asarray(fields[1:], dtype="float32")
                embeddings[token] = vector

    if not embeddings:
        raise ConfigurationError(
            "No embeddings of correct dimension found; you probably "
            "misspecified your embedding_dim parameter, or didn't "
            "pre-populate your Vocabulary"
        )

    all_embeddings = np.asarray(list(embeddings.values()))
    embeddings_mean = float(np.mean(all_embeddings))
    embeddings_std = float(np.std(all_embeddings))
    # Now we initialize the weight matrix for an embedding layer, starting with random vectors,
    # then filling in the word vectors we just read.
    logger.info("Initializing pre-trained embedding layer")
    embedding_matrix = torch.FloatTensor(vocab_size, embedding_dim).normal_(
        embeddings_mean, embeddings_std
    )
    num_tokens_found = 0
    index_to_token = vocab.get_index_to_token_vocabulary(namespace)
    for i in range(vocab_size):
        token = index_to_token[i]

        # If we don't have a pre-trained vector for this word, we'll just leave this row alone,
        # so the word has a random initialization.
        if token in embeddings:
            embedding_matrix[i] = torch.FloatTensor(embeddings[token])
            num_tokens_found += 1
        else:
            logger.debug(
                "Token %s was not found in the embedding file. Initialising randomly.", token
            )

    logger.info(
        "Pretrained embeddings were found for %d out of %d tokens", num_tokens_found, vocab_size
    )

    return embedding_matrix


# def _read_embeddings_from_hdf5(
#     embeddings_filename: str, embedding_dim: int, vocab: Vocabulary, namespace: str = "tokens"
# ) -> torch.FloatTensor:
#     """
#     Reads from a hdf5 formatted file. The embedding matrix is assumed to
#     be keyed by 'embedding' and of size `(num_tokens, embedding_dim)`.
#     """
#     with h5py.File(embeddings_filename, "r") as fin:
#         embeddings = fin["embedding"][...]

#     if list(embeddings.shape) != [vocab.get_vocab_size(namespace), embedding_dim]:
#         raise ConfigurationError(
#             "Read shape {0} embeddings from the file, but expected {1}".format(
#                 list(embeddings.shape), [vocab.get_vocab_size(namespace), embedding_dim]
#             )
#         )

#     return torch.FloatTensor(embeddings)


def format_embeddings_file_uri(
    main_file_path_or_url: str, path_inside_archive: Optional[str] = None
) -> str:
    if path_inside_archive:
        return "({})#{}".format(main_file_path_or_url, path_inside_archive)
    return main_file_path_or_url


class EmbeddingsFileURI(NamedTuple):
    main_file_uri: str
    path_inside_archive: Optional[str] = None


def parse_embeddings_file_uri(uri: str) -> "EmbeddingsFileURI":
    match = re.fullmatch(r"\((.*)\)#(.*)", uri)
    if match:
        fields = cast(Tuple[str, str], match.groups())
        return EmbeddingsFileURI(*fields)
    else:
        return EmbeddingsFileURI(uri, None)


class EmbeddingsTextFile(Iterator[str]):
    """
    Utility class for opening embeddings text files. Handles various compression formats,
    as well as context management.

    # Parameters

    file_uri : `str`
        It can be:

        * a file system path or a URL of an eventually compressed text file or a zip/tar archive
          containing a single file.
        * URI of the type `(archive_path_or_url)#file_path_inside_archive` if the text file
          is contained in a multi-file archive.

    encoding : `str`
    cache_dir : `str`
    """

    DEFAULT_ENCODING = "utf-8"

    def __init__(
        self, file_uri: str, encoding: str = DEFAULT_ENCODING, cache_dir: str = None
    ) -> None:

        self.uri = file_uri
        self._encoding = encoding
        self._cache_dir = cache_dir
        self._archive_handle: Any = None  # only if the file is inside an archive

        main_file_uri, path_inside_archive = parse_embeddings_file_uri(file_uri)
        main_file_local_path = main_file_uri

        if zipfile.is_zipfile(main_file_local_path):  # ZIP archive
            self._open_inside_zip(main_file_uri, path_inside_archive)

        elif tarfile.is_tarfile(main_file_local_path):  # TAR archive
            self._open_inside_tar(main_file_uri, path_inside_archive)

        else:  # all the other supported formats, including uncompressed files
            if path_inside_archive:
                raise ValueError("Unsupported archive format: %s" + main_file_uri)

            # All the python packages for compressed files share the same interface of io.open
            extension = get_file_extension(main_file_uri)

            # Some systems don't have support for all of these libraries, so we import them only
            # when necessary.
            package = None
            if extension in [".txt", ".vec"]:
                package = io
            elif extension == ".gz":
                import gzip

                package = gzip
            elif extension == ".bz2":
                import bz2

                package = bz2
            elif extension == ".lzma":
                import lzma

                package = lzma

            if package is None:
                logger.warning(
                    'The embeddings file has an unknown file extension "%s". '
                    "We will assume the file is an (uncompressed) text file",
                    extension,
                )
                package = io

            self._handle = package.open(  # type: ignore
                main_file_local_path, "rt", encoding=encoding
            )

        # To use this with tqdm we'd like to know the number of tokens. It's possible that the
        # first line of the embeddings file contains this: if it does, we want to start iteration
        # from the 2nd line, otherwise we want to start from the 1st.
        # Unfortunately, once we read the first line, we cannot move back the file iterator
        # because the underlying file may be "not seekable"; we use itertools.chain instead.
        first_line = next(self._handle)  # this moves the iterator forward
        self.num_tokens = EmbeddingsTextFile._get_num_tokens_from_first_line(first_line)
        if self.num_tokens:
            # the first line is a header line: start iterating from the 2nd line
            self._iterator = self._handle
        else:
            # the first line is not a header line: start iterating from the 1st line
            self._iterator = itertools.chain([first_line], self._handle)

    def _open_inside_zip(self, archive_path: str, member_path: Optional[str] = None) -> None:
        cached_archive_path = archive_path
        archive = zipfile.ZipFile(cached_archive_path, "r")
        if member_path is None:
            members_list = archive.namelist()
            member_path = self._get_the_only_file_in_the_archive(members_list, archive_path)
        member_path = cast(str, member_path)
        member_file = cast(BinaryIO, archive.open(member_path, "r"))
        self._handle = io.TextIOWrapper(member_file, encoding=self._encoding)
        self._archive_handle = archive

    def _open_inside_tar(self, archive_path: str, member_path: Optional[str] = None) -> None:
        cached_archive_path = archive_path
        archive = tarfile.open(cached_archive_path, "r")
        if member_path is None:
            members_list = archive.getnames()
            member_path = self._get_the_only_file_in_the_archive(members_list, archive_path)
        member_path = cast(str, member_path)
        member = archive.getmember(member_path)  # raises exception if not present
        member_file = cast(BinaryIO, archive.extractfile(member))
        self._handle = io.TextIOWrapper(member_file, encoding=self._encoding)
        self._archive_handle = archive

    def read(self) -> str:
        return "".join(self._iterator)

    def readline(self) -> str:
        return next(self._iterator)

    def close(self) -> None:
        self._handle.close()
        if self._archive_handle:
            self._archive_handle.close()

    def __enter__(self) -> "EmbeddingsTextFile":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()

    def __iter__(self) -> "EmbeddingsTextFile":
        return self

    def __next__(self) -> str:
        return next(self._iterator)

    def __len__(self) -> Optional[int]:
        if self.num_tokens:
            return self.num_tokens
        raise AttributeError(
            "an object of type EmbeddingsTextFile implements `__len__` only if the underlying "
            "text file declares the number of tokens (i.e. the number of lines following)"
            "in the first line. That is not the case of this particular instance."
        )

    @staticmethod
    def _get_the_only_file_in_the_archive(members_list: Sequence[str], archive_path: str) -> str:
        if len(members_list) > 1:
            raise ValueError(
                "The archive %s contains multiple files, so you must select "
                "one of the files inside providing a uri of the type: %s."
                % (
                    archive_path,
                    format_embeddings_file_uri("path_or_url_to_archive", "path_inside_archive"),
                )
            )
        return members_list[0]

    @staticmethod
    def _get_num_tokens_from_first_line(line: str) -> Optional[int]:
        """This function takes in input a string and if it contains 1 or 2 integers, it assumes the
        largest one it the number of tokens. Returns None if the line doesn't match that pattern."""
        fields = line.split(" ")
        if 1 <= len(fields) <= 2:
            try:
                int_fields = [int(x) for x in fields]
            except ValueError:
                return None
            else:
                num_tokens = max(int_fields)
                logger.info(
                    "Recognized a header line in the embedding file with number of tokens: %d",
                    num_tokens,
                )
                return num_tokens
        return None

class EndpointSpanExtractor(SpanExtractor):
    """
    提取跨度表示
    可以设置在结尾添加宽度嵌入
    可以提取其断点 x 为跨度开始的嵌入， y 为跨度结束的嵌入
    可以通过表达式指定合并方式，如 `x`, `y`, `x*y`, `x+y`, `x-y`, `x/y`
    默认解析为二进制表示，可以用 , 分隔返回表示， 用[x;y;x*y] 的形式返回 concat

    # Parameters

    input_dim : `int`, required.
        The final dimension of the `sequence_tensor`.
    combination : `str`, optional (default = `"x,y"`).
        The method used to combine the `start_embedding` and `end_embedding`
        representations. See above for a full description.
    num_width_embeddings : `int`, optional (default = `None`).
        Specifies the number of buckets to use when representing
        span width features.
    span_width_embedding_dim : `int`, optional (default = `None`).
        The embedding size for the span_width features.
    bucket_widths : `bool`, optional (default = `False`).
        Whether to bucket the span widths into log-space buckets. If `False`,
        the raw span widths are used.
    use_exclusive_start_indices : `bool`, optional (default = `False`).
        If `True`, the start indices extracted are converted to exclusive indices. Sentinels
        are used to represent exclusive span indices for the elements in the first
        position in the sequence (as the exclusive indices for these elements are outside
        of the the sequence boundary) so that start indices can be exclusive.
        NOTE: This option can be helpful to avoid the pathological case in which you
        want span differences for length 1 spans - if you use inclusive indices, you
        will end up with an `x - x` operation for length 1 spans, which is not good.
    """

    def __init__(
        self,
        input_dim: int,
        combination: str = "x,y",
        num_width_embeddings: int = None,
        span_width_embedding_dim: int = None,
        bucket_widths: bool = False,
        use_exclusive_start_indices: bool = False,
    ) -> None:
        super().__init__()
        self._input_dim = input_dim
        self._combination = combination
        self._num_width_embeddings = num_width_embeddings
        self._bucket_widths = bucket_widths

        self._use_exclusive_start_indices = use_exclusive_start_indices
        if use_exclusive_start_indices:
            self._start_sentinel = Parameter(torch.randn([1, 1, int(input_dim)]))
        
        self._span_width_embedding: Optional[Embedding] = None
        if num_width_embeddings is not None and span_width_embedding_dim is not None:
            self._span_width_embedding = Embedding(
                num_embeddings=num_width_embeddings, embedding_dim=span_width_embedding_dim
            )
        else:
            if num_width_embeddings is not None or span_width_embedding_dim is not None:
                raise ConfigurationError(
                    "To use a span width embedding representation, you must specify both num_width_buckets and span_width_embedding_dim."
                )

    def get_input_dim(self) -> int:
        return self._input_dim

    def get_output_dim(self) -> int:
        combined_dim = util.get_combined_dim(self._combination, [self._input_dim, self._input_dim])
        if self._span_width_embedding is not None:
            return combined_dim + self._span_width_embedding.get_output_dim()
        return combined_dim

     
    def forward(
        self,
        sequence_tensor: torch.FloatTensor,
        span_indices: torch.LongTensor,
        sequence_mask: torch.Tensor = None,
        span_indices_mask: torch.Tensor = None,
    ) -> None:
        # shape (batch_size, num_spans)
        span_starts, span_ends = [index.squeeze(-1) for index in span_indices.split(1, dim=-1)]

        if span_indices_mask is not None:
            # It's not strictly necessary to multiply the span indices by the mask here,
            # but it's possible that the span representation was padded with something other
            # than 0 (such as -1, which would be an invalid index), so we do so anyway to
            # be safe.
            span_starts = span_starts * span_indices_mask
            span_ends = span_ends * span_indices_mask

        if not self._use_exclusive_start_indices:
            if sequence_tensor.size(-1) != self._input_dim:
                raise ValueError(
                    f"Dimension mismatch expected ({sequence_tensor.size(-1)}) "
                    f"received ({self._input_dim})."
                )
            start_embeddings = util.batched_index_select(sequence_tensor, span_starts)
            end_embeddings = util.batched_index_select(sequence_tensor, span_ends)

        else:
            # We want `exclusive` span starts, so we remove 1 from the forward span starts
            # as the AllenNLP `SpanField` is inclusive.
            # shape (batch_size, num_spans)
            exclusive_span_starts = span_starts - 1
            # shape (batch_size, num_spans, 1)
            start_sentinel_mask = (exclusive_span_starts == -1).unsqueeze(-1)
            exclusive_span_starts = exclusive_span_starts * ~start_sentinel_mask.squeeze(-1)

            # We'll check the indices here at runtime, because it's difficult to debug
            # if this goes wrong and it's tricky to get right.
            if (exclusive_span_starts < 0).any():
                raise ValueError(
                    f"Adjusted span indices must lie inside the the sequence tensor, "
                    f"but found: exclusive_span_starts: {exclusive_span_starts}."
                )

            start_embeddings = util.batched_index_select(sequence_tensor, exclusive_span_starts)
            end_embeddings = util.batched_index_select(sequence_tensor, span_ends)

            # We're using sentinels, so we need to replace all the elements which were
            # outside the dimensions of the sequence_tensor with the start sentinel.
            start_embeddings = (
                start_embeddings * ~start_sentinel_mask + start_sentinel_mask * self._start_sentinel
            )

        combined_tensors = util.combine_tensors(
            self._combination, [start_embeddings, end_embeddings]
        )
        if self._span_width_embedding is not None:
            # Embed the span widths and concatenate to the rest of the representations.
            if self._bucket_widths:
                span_widths = util.bucket_values(
                    span_ends - span_starts, num_total_buckets=self._num_width_embeddings  # type: ignore
                )
            else:
                span_widths = span_ends - span_starts

            span_width_embeddings = self._span_width_embedding(span_widths)
            combined_tensors = torch.cat([combined_tensors, span_width_embeddings], -1)

        if span_indices_mask is not None:
            return combined_tensors * span_indices_mask.unsqueeze(-1)

        return combined_tensors

class SelfAttentiveSpanExtractor(SpanExtractor):
    """
    Computes span representations by generating an unnormalized attention score for each
    word in the document. Spans representations are computed with respect to these
    scores by normalising the attention scores for words inside the span.

    Given these attention distributions over every span, this module weights the
    corresponding vector representations of the words in the span by this distribution,
    returning a weighted representation of each span.

    Registered as a `SpanExtractor` with name "self_attentive".

    # Parameters

    input_dim : `int`, required.
        The final dimension of the `sequence_tensor`.

    # Returns

    attended_text_embeddings : `torch.FloatTensor`.
        A tensor of shape (batch_size, num_spans, input_dim), which each span representation
        is formed by locally normalising a global attention over the sequence. The only way
        in which the attention distribution differs over different spans is in the set of words
        over which they are normalized.
    """

    def __init__(self, input_dim: int) -> None:
        super().__init__()
        self._input_dim = input_dim
        self._global_attention = TimeDistributed(torch.nn.Linear(input_dim, 1))

    def get_input_dim(self) -> int:
        return self._input_dim

    def get_output_dim(self) -> int:
        return self._input_dim

     
    def forward(
        self,
        sequence_tensor: torch.FloatTensor,
        span_indices: torch.LongTensor,
        span_indices_mask: torch.Tensor = None,
    ) -> torch.FloatTensor:
        # shape (batch_size, sequence_length, 1)
        global_attention_logits = self._global_attention(sequence_tensor)

        # shape (batch_size, sequence_length, embedding_dim + 1)
        concat_tensor = torch.cat([sequence_tensor, global_attention_logits], -1)

        concat_output, span_mask = util.batched_span_select(concat_tensor, span_indices)

        # Shape: (batch_size, num_spans, max_batch_span_width, embedding_dim)
        span_embeddings = concat_output[:, :, :, :-1]
        # Shape: (batch_size, num_spans, max_batch_span_width)
        span_attention_logits = concat_output[:, :, :, -1]

        # Shape: (batch_size, num_spans, max_batch_span_width)
        span_attention_weights = util.masked_softmax(span_attention_logits, span_mask)

        # Do a weighted sum of the embedded spans with
        # respect to the normalised attention distributions.
        # Shape: (batch_size, num_spans, embedding_dim)
        attended_text_embeddings = util.weighted_sum(span_embeddings, span_attention_weights)

        if span_indices_mask is not None:
            # Above we were masking the widths of spans with respect to the max
            # span width in the batch. Here we are masking the spans which were
            # originally passed in as padding.
            return attended_text_embeddings * span_indices_mask.unsqueeze(-1)

        return attended_text_embeddings

class EmptyEmbedder(TokenEmbedder):
    """
    Assumes you want to completely ignore the output of a `TokenIndexer` for some reason, and does
    not return anything when asked to embed it.

    You should almost never need to use this; normally you would just not use a particular
    `TokenIndexer`. It's only in very rare cases, like simplicity in data processing for language
    modeling (where we use just one `TextField` to handle input embedding and computing target ids),
    where you might want to use this.

    Registered as a `TokenEmbedder` with name "empty".
    """

    def __init__(self) -> None:
        super().__init__()

    def get_output_dim(self):
        return 0

    def forward(self, *inputs, **kwargs) -> torch.Tensor:
        return None

class BasicTextFieldEmbedder(TextFieldEmbedder, nn.ModuleDict):
    """
    This is a `TextFieldEmbedder` that wraps a collection of
    [`TokenEmbedder`](../token_embedders/token_embedder.md) objects.  Each
    `TokenEmbedder` embeds or encodes the representation output from one
    [`allennlp.data.TokenIndexer`](../../data/token_indexers/token_indexer.md). As the data produced by a
    [`allennlp.data.fields.TextField`](../../data/fields/text_field.md) is a dictionary mapping names to these
    representations, we take `TokenEmbedders` with corresponding names.  Each `TokenEmbedders`
    embeds its input, and the result is concatenated in an arbitrary (but consistent) order.

    Registered as a `TextFieldEmbedder` with name "basic", which is also the default.

    # Parameters

    token_embedders : `Dict[str, TokenEmbedder]`, required.
        A dictionary mapping token embedder names to implementations.
        These names should match the corresponding indexer used to generate
        the tensor passed to the TokenEmbedder.
    """

    def __init__(self, token_embedders: Dict[str, TokenEmbedder]) -> None:
        super().__init__()
        # NOTE(mattg): I'd prefer to just use ModuleDict(token_embedders) here, but that changes
        # weight locations in msadapter.pytorch state dictionaries and invalidates all prior models, just for a
        # cosmetic change in the code.
        self._token_embedders = token_embedders
        for key, embedder in token_embedders.items():
            name = "token_embedder_%s" % key
            self.add_module(name, embedder)
        self._ordered_embedder_keys = sorted(self._token_embedders.keys())

    def get_output_dim(self) -> int:
        output_dim = 0
        for embedder in self._token_embedders.values():
            output_dim += embedder.get_output_dim()
        return output_dim

    def forward(
        self, text_field_input: torch.Tensor, num_wrapping_dims: int = 0, **kwargs
    ) -> torch.Tensor:
        # 原文判断是否未提供的 key 均为 EmptyEmbedder，否则配置错误，此处跳过
        embedded_representations = []
        for key in self._ordered_embedder_keys:
            # Note: need to use getattr here so that the pytorch voodoo
            # with submodules works with multiple GPUs.
            embedder = getattr(self, "token_embedder_{}".format(key))

            if isinstance(embedder, EmptyEmbedder):
                continue

            forward_params = inspect.signature(embedder.forward).parameters
            forward_params_values = {}
            missing_tensor_args = set()
            for param in forward_params.keys():
                if param in kwargs:
                    forward_params_values[param] = kwargs[param]
                else:
                    missing_tensor_args.add(param)

            for _ in range(num_wrapping_dims):
                embedder = TimeDistributed(embedder)

            # tensors: Dict[str, torch.Tensor] = text_field_input[key]
            tensors = text_field_input
            if isinstance(tensors, torch.Tensor):
                token_vectors = embedder(tensors, **forward_params_values)
            else:
                # 不推荐的嵌入计算方法
                token_vectors = embedder(**tensors, **forward_params_values)

            if token_vectors is not None:
                # To handle some very rare use cases, we allow the return value of the embedder to
                # be None; we just skip it in that case.
                embedded_representations.append(token_vectors)
        return torch.cat(embedded_representations, dim=-1)

class BertEmbedder(TokenEmbedder):

    def __init__(
            self, seq_len = hp.pretrain_token_embedd_seq_len, 
            pretrain_name = hp.pretrain_token_embedd_name, 
            auto_prefix=True, flags=None):
        super(TokenEmbedder, self).__init__(auto_prefix, flags)
        self._bert_config:BertConfig = BertConfig.from_pretrained(pretrain_name)
        self._bert_config['seq_length'] = seq_len
        self._bert_config['compute_dtype'] = torch.float32
        # self._bert_config.__setattr__('seq_length', seq_len)
        # self._bert_config.__setattr__('compute_dtype', 'float32')

        self._seq_len = seq_len

        self._bert = BertModel(self._bert_config)

    def get_output_dim(self) -> int:
        return self._bert_config.hidden_size
    
    def _process_input(
            self, 
            target_tensor: Union[torch.Tensor, List[torch.Tensor]],
            ) -> Tuple[Union[torch.Tensor, List[torch.Tensor]], int]:
        if target_tensor is None:
            return None, 0
        
        if isinstance(target_tensor, list):
            res = []
            for tt in target_tensor:
                ott, os = self._process_input(tt)
                res.append(ott)
            return res, os
        
        # type
        target_tensor = target_tensor.to(dtype = torch.int32)
        
        # batch
        if len(target_tensor.shape) == 1:
            target_tensor.unsqueeze(0)
        
        org_size = target_tensor.shape[1]
        target_tensor = torch.nn.functional.pad(
            target_tensor, (0, self._seq_len - org_size)
        )

        return target_tensor, org_size
        
    def _rebuild_input(
            self,
            target_tensor: Union[torch.Tensor, List[torch.Tensor]],
            org_size: int,
            unbatch:bool = True,
        ) -> Union[torch.Tensor, List[torch.Tensor]]:
        if target_tensor is None:
            return None

        if isinstance(target_tensor, list):
            return [self._rebuild_input(tt) for tt in target_tensor]
        
        target_tensor = target_tensor[:, :org_size, :]

        if unbatch and len(target_tensor.shape) > 1 and target_tensor.shape[0] == 1:
            target_tensor = target_tensor.squeeze(0)
        
        return target_tensor

    
    def construct(self, input_ids: torch.Tensor, token_type_ids: torch.Tensor = None, attention_mask = None):
        batch_flag = len(input_ids.shape) > 1
        tensors, orign_size = self._process_input(
            [input_ids, token_type_ids, attention_mask]
        )
        input_ids, token_type_ids, attention_mask = tensors[0], tensors[1], tensors[2]

        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)

        squence_embed, pooled_embed, embed_table = self._bert(
            input_ids = input_ids,
            input_mask = attention_mask,
            # token_type_id = token_type_ids
            token_type_ids = token_type_ids # Bert Model
        )
        squence_embed = self._rebuild_input(squence_embed, orign_size, not batch_flag)

        return squence_embed