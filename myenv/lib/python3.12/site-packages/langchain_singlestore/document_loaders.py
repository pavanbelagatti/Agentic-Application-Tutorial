"""SingleStore document loader."""

import re
from typing import Any, Iterator

import singlestoredb as s2
from langchain_core.document_loaders.base import BaseLoader
from langchain_core.documents import Document
from sqlalchemy.pool import QueuePool


class SingleStoreLoader(BaseLoader):
    """SingleStore document loader.
    Loads documents from a SingleStore database table.
    The table must contain three fields: content, metadata, and id.
    The content field is used to store the document content,
    the metadata field is used to store the document metadata,
    and the id field is used to store the document id.
    Setup:
            Install ``langchain-singlestore``

            .. code-block:: bash

                pip install -U langchain-singlestore

        Instantiate:
            .. code-block:: python

                from langchain_community.document_loaders import SingleStoreLoader

                loader = SingleStoreLoader(
                    host="https://user:password@127.0.0.1:3306/database",
                    table_name="documents",
                    content_field="content",
                    metadata_field="metadata",
                )

        Lazy load:
            .. code-block:: python

                docs = []
                docs_lazy = loader.lazy_load()

                # async variant:
                # docs_lazy = await loader.alazy_load()

                for doc in docs_lazy:
                    docs.append(doc)
                print(docs[0].page_content[:100])
                print(docs[0].metadata)
    """

    def __init__(
        self,
        *,
        table_name: str = "embeddings",
        content_field: str = "content",
        metadata_field: str = "metadata",
        id_field: str = "id",
        pool_size: int = 5,
        max_overflow: int = 10,
        timeout: float = 30,
        **kwargs: Any,
    ):
        """
        Initialize the SingleStore document loader.
        Args:
            table_name (str, optional): Specifies the name of the table in use.
                Defaults to "embeddings".
            content_field (str, optional): Specifies the field to store the content.
                Defaults to "content".
            metadata_field (str, optional): Specifies the field to store metadata.
                Defaults to "metadata".
            id_field (str, optional): Specifies the field to store the id.
                Defaults to "id".

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

        """
        super().__init__()
        self.pool_size = pool_size
        self.max_overflow = max_overflow
        self.timeout = timeout
        self.table_name = self._sanitize_input(table_name)
        self.content_field = self._sanitize_input(content_field)
        self.metadata_field = self._sanitize_input(metadata_field)
        self.id_field = self._sanitize_input(id_field)

        # Pass the rest of the kwargs to the connection.
        self.connection_kwargs = kwargs

        # Add program name and version to connection attributes.
        if "conn_attrs" not in self.connection_kwargs:
            self.connection_kwargs["conn_attrs"] = dict()

        self.connection_kwargs["conn_attrs"]["_connector_name"] = "langchain python sdk"
        self.connection_kwargs["conn_attrs"]["_connector_version"] = "3.0.0"

        # Create connection pool.
        self.connection_pool = QueuePool(
            self._get_connection,
            max_overflow=max_overflow,
            pool_size=pool_size,
            timeout=timeout,
        )

    def _get_connection(self) -> Any:
        return s2.connect(**self.connection_kwargs)

    def _sanitize_input(self, input_str: str) -> str:
        # Remove characters that are not alphanumeric or underscores
        return re.sub(r"[^a-zA-Z0-9_]", "", input_str)

    def lazy_load(self) -> Iterator[Document]:
        """
        Lazy load documents from the SingleStore database.

        Returns:
            Iterator[Document]: An iterator of Document objects.
        """
        conn = self.connection_pool.connect()
        try:
            cur = conn.cursor()
            try:
                query = f"""SELECT {self.content_field},
                        {self.metadata_field}, 
                        {self.id_field}
                        FROM {self.table_name} ORDER BY {self.id_field} ASC"""
                cur.execute(query)
                for row in cur.fetchall():
                    content = row[0]
                    metadata = row[1]
                    doc_id = row[2]
                    yield Document(page_content=content, metadata=metadata, id=doc_id)
            finally:
                cur.close()
        finally:
            conn.close()
