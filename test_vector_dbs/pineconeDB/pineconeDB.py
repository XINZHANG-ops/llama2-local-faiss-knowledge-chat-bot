import pinecone
import logging

logging.basicConfig(level=logging.INFO)


class PineconeOperations:
    def __init__(self, api_key, environment, index_name):
        """
        Initializes the PineconeOperations class.

        Args:
        - api_key (str): The API key to authenticate with Pinecone.
        - environment (str): The Pinecone environment.
        - index_name (str): The name of the Pinecone index.
        """
        self.api_key = api_key
        self.environment = environment
        self.index_name = index_name
        pinecone.init(api_key, environment=self.environment)
        self.index = pinecone.Index(index_name)
        if self.if_index_exist():
            self.dimension = self.index.describe_index_stats()["dimension"]
        else:
            self.dimension = None

    def if_index_exist(self):
        """
        Checks if the Pinecone index associated with this instance exists.

        Returns:
        - bool: True if the index exists, False otherwise.
        """
        if self.index_name in pinecone.list_indexes():
            return True
        else:
            return False

    def create_index(self, dimension, metric='cosine'):
        """
        Creates a new Pinecone index if it doesn't already exist.

        Args:
        - dimension (int): Dimension of the vector.
        - metric (str, optional): The metric to use, defaults to 'cosine'.
        """
        # First, check if our index already exists. If it doesn't, we create it
        if not self.if_index_exist():
            logging.info(f"Creating the index {self.index_name}...")
            pinecone.create_index(
                name=self.index_name,
                metric=metric,
                dimension=dimension
            )
            self.dimension = dimension
        else:
            self.dimension = self.index.describe_index_stats()["dimension"]
            logging.info(f"Index {self.index_name} exist, delete it using delete_index() first for recreation!")

    def delete_index(self):
        """
        Deletes the Pinecone index associated with this instance.
        """
        pinecone.delete_index(self.index_name)
        self.dimension = None

    def add_vector(self, id, values, metadata={}, namespace=''):
        """
        Adds a vector to the Pinecone index.

        Args:
        - id (str): The unique identifier for the vector.
        - values (list): The vector values.
        - metadata (dict, optional): Metadata associated with the vector.
        - namespace (str, optional): Namespace for the vector.

        Returns:
        - dict: The response from Pinecone for the upsert operation.
        """
        pinecone_format = [{"id": id, "values": values, "metadata": metadata}]
        upsert_response = self.index.upsert(
            vectors=pinecone_format,
            namespace=namespace
        )
        return upsert_response

    def add_vector_batch(self, ids, values, metadatas, namespace='', batch_size=500):
        """
        Adds a batch of vectors to the Pinecone index.

        Args:
        - ids (list of str): A list of unique identifiers for the vectors.
        - values (list of lists): A list where each item is the vector values.
        - metadatas (list of dicts): A list where each item is the metadata associated with the vector.
        - namespace (str, optional): Namespace for the vectors, defaults to an empty string.
        - batch_size (int, optional): Number of vectors to add in a single batch, defaults to 500.

        Notes:
        Vectors are added in batches to avoid overloading the Pinecone index. After each batch insertion, a log message is printed.
        """
        pinecone_format = [{"id": id, "values": value, 'metadata': metadata} for id, value, metadata in
                           zip(ids, values, metadatas)]
        start = 0
        while start < len(pinecone_format):
            end = start + batch_size
            sub_list = pinecone_format[start:end]
            upsert_response = self.index.upsert(
                vectors=sub_list,
                namespace=namespace
            )
            start = end
            logging.info(f"{len(sub_list)} vectors added!")

    def remove_vector(self, id):
        """
        Removes a vector from the Pinecone index.

        Args:
        - id (str): The unique identifier for the vector.

        Returns:
        - dict: The response from Pinecone for the delete operation.
        """
        delete_response = self.index.delete(ids=[id], namespace='')
        return delete_response

    def query_vector(self, vector, top_k=3, include_metadata=True, include_values=True, **kwargs):
        """
        Queries the Pinecone index to find vectors close to the given vector.

        Args:
        - vector (list): The query vector.
        - top_k (int, optional): Number of top matches to return.
        - include_metadata (bool, optional): Whether to include metadata in the response.
        - include_values (bool, optional): Whether to include vector values in the response.
        - **kwargs: Additional keyword arguments for the Pinecone query function.

        Returns:
        - dict: The matching vectors from the Pinecone index.
        """
        matches = self.index.query(
            vector=vector,
            top_k=top_k,
            include_metadata=include_metadata,
            include_values=include_values,
            **kwargs
        )
        return matches
