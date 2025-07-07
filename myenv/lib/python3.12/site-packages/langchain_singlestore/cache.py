from typing import Any, Dict, List, Optional

from langchain_core.caches import RETURN_VAL_TYPE, BaseCache
from langchain_core.embeddings import Embeddings
from langchain_core.load.dump import dumps
from langchain_core.load.load import loads
from langchain_core.outputs import Generation

from langchain_singlestore._utils import hash
from langchain_singlestore.vectorstores import DistanceStrategy, SingleStoreVectorStore


class SingleStoreSemanticCache(BaseCache):
    """Cache that uses SingleStore as a backend"""

    def __init__(
        self,
        embedding: Embeddings,
        *,
        cache_table_prefix: str = "cache_",
        search_threshold: float = 0.2,
        **kwargs: Any,
    ):
        """Initialize with necessary components.

        Args:
            embedding (Embeddings): A text embedding model.
            cache_table_prefix (str, optional): Prefix for the cache table name.
                Defaults to "cache_".
            search_threshold (float, optional): The minimum similarity score for
                a search result to be considered a match. Defaults to 0.2.

            Following arguments pertrain to the SingleStoreDB vector store:

            distance_strategy (DistanceStrategy, optional):
                Determines the strategy employed for calculating
                the distance between vectors in the embedding space.
                Defaults to DOT_PRODUCT.
                Available options are:
                - DOT_PRODUCT: Computes the scalar product of two vectors.
                    This is the default behavior
                - EUCLIDEAN_DISTANCE: Computes the Euclidean distance between
                    two vectors. This metric considers the geometric distance in
                    the vector space, and might be more suitable for embeddings
                    that rely on spatial relationships. This metric is not
                    compatible with the WEIGHTED_SUM search strategy.

            content_field (str, optional): Specifies the field to store the content.
                Defaults to "content".
            metadata_field (str, optional): Specifies the field to store metadata.
                Defaults to "metadata".
            vector_field (str, optional): Specifies the field to store the vector.
                Defaults to "vector".
            id_field (str, optional): Specifies the field to store the id.
                Defaults to "id".

            use_vector_index (bool, optional): Toggles the use of a vector index.
                Works only with SingleStoreDB 8.5 or later. Defaults to False.
                If set to True, vector_size parameter is required to be set to
                a proper value.

            vector_index_name (str, optional): Specifies the name of the vector index.
                Defaults to empty. Will be ignored if use_vector_index is set to False.

            vector_index_options (dict, optional): Specifies the options for
                the vector index. Defaults to {}.
                Will be ignored if use_vector_index is set to False. The options are:
                index_type (str, optional): Specifies the type of the index.
                    Defaults to IVF_PQFS.
                For more options, please refer to the SingleStoreDB documentation:
                https://docs.singlestore.com/cloud/reference/sql-reference/vector-functions/vector-indexing/

            vector_size (int, optional): Specifies the size of the vector.
                Defaults to 1536. Required if use_vector_index is set to True.
                Should be set to the same value as the size of the vectors
                stored in the vector_field.

            Following arguments pertain to the connection pool:

            pool_size (int, optional): Determines the number of active connections in
                the pool. Defaults to 5.
            max_overflow (int, optional): Determines the maximum number of connections
                allowed beyond the pool_size. Defaults to 10.
            timeout (float, optional): Specifies the maximum wait time in seconds for
                establishing a connection. Defaults to 30.

            Following arguments pertain to the database connection:

            host (str, optional): Specifies the hostname, IP address, or URL for the
                database connection. The default scheme is "mysql".
            user (str, optional): Database username.
            password (str, optional): Database password.
            port (int, optional): Database port. Defaults to 3306 for non-HTTP
                connections, 80 for HTTP connections, and 443 for HTTPS connections.
            database (str, optional): Database name.

            Additional optional arguments provide further customization over the
            database connection:

            pure_python (bool, optional): Toggles the connector mode. If True,
                operates in pure Python mode.
            local_infile (bool, optional): Allows local file uploads.
            charset (str, optional): Specifies the character set for string values.
            ssl_key (str, optional): Specifies the path of the file containing the SSL
                key.
            ssl_cert (str, optional): Specifies the path of the file containing the SSL
                certificate.
            ssl_ca (str, optional): Specifies the path of the file containing the SSL
                certificate authority.
            ssl_cipher (str, optional): Sets the SSL cipher list.
            ssl_disabled (bool, optional): Disables SSL usage.
            ssl_verify_cert (bool, optional): Verifies the server's certificate.
                Automatically enabled if ``ssl_ca`` is specified.
            ssl_verify_identity (bool, optional): Verifies the server's identity.
            conv (dict[int, Callable], optional): A dictionary of data conversion
                functions.
            credential_type (str, optional): Specifies the type of authentication to
                use: auth.PASSWORD, auth.JWT, or auth.BROWSER_SSO.
            autocommit (bool, optional): Enables autocommits.
            results_type (str, optional): Determines the structure of the query results:
                tuples, namedtuples, dicts.
            results_format (str, optional): Deprecated. This option has been renamed to
                results_type.

        Examples:
            Basic Usage:

            .. code-block:: python

                import langchain
                from langchain.cache import SingleStoreDBSemanticCache
                from langchain.embeddings import OpenAIEmbeddings

                langchain.llm_cache = SingleStoreDBSemanticCache(
                    embedding=OpenAIEmbeddings(),
                    host="https://user:password@127.0.0.1:3306/database"
                )

            Advanced Usage:

            .. code-block:: python

                import langchain
                from langchain.cache import SingleStoreDBSemanticCache
                from langchain.embeddings import OpenAIEmbeddings

                langchain.llm_cache = = SingleStoreDBSemanticCache(
                    embeddings=OpenAIEmbeddings(),
                    use_vector_index=True,
                    host="127.0.0.1",
                    port=3306,
                    user="user",
                    password="password",
                    database="db",
                    table_name="my_custom_table",
                    pool_size=10,
                    timeout=60,
                )
        """

        self._cache_dict: Dict[str, SingleStoreVectorStore] = {}
        self.embedding = embedding
        self.cache_table_prefix = cache_table_prefix
        self.search_threshold = search_threshold

        # Pass the rest of the kwargs to the connection.
        self.connection_kwargs = kwargs

    def _index_name(self, llm_string: str) -> str:
        hashed_index = hash(llm_string)
        return f"{self.cache_table_prefix}{hashed_index}"

    def _get_llm_cache(self, llm_string: str) -> SingleStoreVectorStore:
        index_name = self._index_name(llm_string)

        # return vectorstore client for the specific llm string
        if index_name not in self._cache_dict:
            self._cache_dict[index_name] = SingleStoreVectorStore(
                embedding=self.embedding,
                table_name=index_name,
                **self.connection_kwargs,
            )
        return self._cache_dict[index_name]

    def lookup(self, prompt: str, llm_string: str) -> Optional[RETURN_VAL_TYPE]:
        """Look up based on prompt and llm_string."""
        llm_cache = self._get_llm_cache(llm_string)
        generations: List = []
        # Read from a Hash
        results = llm_cache.similarity_search_with_score(
            query=prompt,
            k=1,
        )
        if results:
            for document_score in results:
                if (
                    document_score[1] > self.search_threshold
                    and llm_cache.distance_strategy == DistanceStrategy.DOT_PRODUCT
                ) or (
                    document_score[1] < self.search_threshold
                    and llm_cache.distance_strategy
                    == DistanceStrategy.EUCLIDEAN_DISTANCE
                ):
                    generations.extend(loads(document_score[0].metadata["return_val"]))
        return generations if generations else None

    def update(self, prompt: str, llm_string: str, return_val: RETURN_VAL_TYPE) -> None:
        """Update cache based on prompt and llm_string."""
        for gen in return_val:
            if not isinstance(gen, Generation):
                raise ValueError(
                    "SingleStoreDBSemanticCache only supports caching of "
                    f"normal LLM generations, got {type(gen)}"
                )
        llm_cache = self._get_llm_cache(llm_string)
        metadata = {
            "llm_string": llm_string,
            "prompt": prompt,
            "return_val": dumps([g for g in return_val]),
        }
        llm_cache.add_texts(texts=[prompt], metadatas=[metadata])

    def clear(self, **kwargs: Any) -> None:
        """Clear semantic cache for a given llm_string."""
        index_name = self._index_name(kwargs["llm_string"])
        if index_name in self._cache_dict:
            self._cache_dict[index_name].drop()
            del self._cache_dict[index_name]
