import chromadb


class chromaDB:
    def __init__(self, index_name, metric, dimension, db_name):
        self.index_name = index_name
        self.metric = metric
        self.dimension = dimension
        self.db_name = db_name
        self.chroma_client = chromadb.PersistentClient(path=db_name)
        try:
            # load collection
            self.collection = self.chroma_client.get_collection(name=index_name)
        except ValueError:
            # create collection
            self.collection = self.chroma_client.create_collection(name=index_name, metadata={"hnsw:space": metric})  # l2 is the default

    def add_vectors(self, vectors, ids):
        self.collection.add(
            embeddings=vectors,
            ids=ids
        )

    def query(self, vectors, top_k):
        matches = self.collection.query(
            query_embeddings=vectors,
            n_results=top_k
        )
        return matches
