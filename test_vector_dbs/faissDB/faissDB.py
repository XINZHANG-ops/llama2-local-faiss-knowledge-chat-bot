import faiss
import pickle
import numpy as np
import os
import shutil
import logging
logging.basicConfig(level=logging.INFO)
# https://github.com/facebookresearch/faiss/wiki/Faiss-indexes
# pip install faiss-cpu

type_name_map = {
    "0": "IndexFlatL2",
    "1": "IndexFlatIP",
    "2": "IndexHNSWFlat",
    "3": "IndexIVFFlat",
    "4": "IndexLSH",
    "5": "IndexScalarQuantizer",
    "6": "IndexPQ",
    "7": "IndexIVFScalarQuantizer",
    "8": "IndexIVFPQ",
    "9": "IndexIVFPQR"
}

type_need_train = {
    "0": False,
    "1": False,
    "2": False,
    "3": True,
    "4": False,
    "5": True,
    "6": True,
    "7": True,
    "8": True,
    "9": True
}

param_map = {
    "0": {"d": "dimension_of_data"},
    "1": {"d": "dimension_of_data"},
    "2": {"d": "dimension_of_data",
          "M": "some_value_related_to_HNSW_structure"},
    "3": {"quantizer": "another_index_like_IndexFlatL2",
          "d": "dimension_of_data",
          "nlists": "number_of_inverted_lists"},
    "4": {"d": "dimension_of_data",
          "nbits": "number_of_bits_for_hash"},
    "5": {"d": "dimension_of_data",
          "qtype": "faiss.ScalarQuantizer.QT_8bit, QT_4bit, etc."},
    "6": {"d": "dimension_of_data",
          "M": "number_of_subquantizers",
          "nbits": "bits_per_subquantizer"},
    "7": {"quantizer": "another_index_like_IndexFlatL2",
          "d": "dimension_of_data",
          "nlists": "number_of_inverted_lists",
          "qtype": "faiss.ScalarQuantizer.QT_8bit, QT_4bit, etc."},
    "8": {"quantizer": "another_index_like_IndexFlatL2",
          "d": "dimension_of_data",
          "nlists": "number_of_inverted_lists",
          "M": "number_of_subquantizers",
          "nbits": "bits_per_subquantizer"},
    "9": {"quantizer": "another_index_like_IndexFlatL2",
          "d": "dimension_of_data",
          "nlists": "number_of_inverted_lists",
          "M": "number_of_subquantizers",
          "nbits": "bits_per_subquantizer",
          "M_refine": "some_value_for_refinement",
          "nbits_refine": "some_value_for_refinement_bits"}
}


def create_flatL2(**kwargs):
    return faiss.IndexFlatL2(kwargs['d'])


def create_flatIP(**kwargs):
    return faiss.IndexFlatIP(kwargs['d'])


def create_hnswFlat(**kwargs):
    index = faiss.IndexHNSWFlat(kwargs['d'], kwargs['M'])
    return index


def create_ivfFlat(**kwargs):
    quantizer = kwargs['quantizer']
    index = faiss.IndexIVFFlat(quantizer, kwargs['d'], kwargs['nlists'])
    return index


def create_lsh(**kwargs):
    return faiss.IndexLSH(kwargs['d'], kwargs['nbits'])


def create_scalarQuantizer(**kwargs):
    return faiss.IndexScalarQuantizer(kwargs['d'], kwargs['qtype'])


def create_pq(**kwargs):
    return faiss.IndexPQ(kwargs['d'], kwargs['M'], kwargs['nbits'])


def create_ivfScalarQuantizer(**kwargs):
    quantizer = kwargs['quantizer']
    index = faiss.IndexIVFScalarQuantizer(quantizer, kwargs['d'], kwargs['nlists'], kwargs['qtype'])
    return index


def create_ivfPQ(**kwargs):
    quantizer = kwargs['quantizer']
    index = faiss.IndexIVFPQ(quantizer, kwargs['d'], kwargs['nlists'], kwargs['M'], kwargs['nbits'])
    return index


def create_ivfPQR(**kwargs):
    quantizer = kwargs['quantizer']
    index = faiss.IndexIVFPQR(quantizer, kwargs['d'], kwargs['nlists'], kwargs['M'], kwargs['nbits'], kwargs['M_refine'], kwargs['nbits_refine'])
    return index


type_map_funcs = {
    "0": create_flatL2,
    "1": create_flatIP,
    "2": create_hnswFlat,
    "3": create_ivfFlat,
    "4": create_lsh,
    "5": create_scalarQuantizer,
    "6": create_pq,
    "7": create_ivfScalarQuantizer,
    "8": create_ivfPQ,
    "9": create_ivfPQR
}


class faissDB:
    def __init__(self, index_type, metric, index_name=None, **kwargs):
        if index_type not in type_map_funcs:
            raise ValueError(f"Unsupported index type: {index_type}")

        # Validate provided arguments
        required_args = set(param_map[index_type].keys())
        provided_args = set(kwargs.keys())

        if not required_args.issubset(provided_args):
            missing_args = required_args - provided_args
            raise ValueError(f"Missing required arguments for index type {type_name_map[index_type]}: {', '.join(missing_args)}")

        index_base = type_map_funcs[index_type](**kwargs)
        self.index = faiss.IndexIDMap(index_base) # this allows add with ids
        self.metric = metric
        if index_name:
            self.index_name = index_name
        else:
            self.index_name = type_name_map[index_type]
        self.index_type = index_type

    def add(self, vectors, ids):
        vectors = np.array(vectors)
        ids = np.array(ids)
        if type_need_train[self.index_type]:
            self.index.train(vectors)
        if self.metric.lower().startswith("c"):
            vectors = vectors / np.linalg.norm(vectors, axis=1)[:, np.newaxis]  # Normalize vectors
        self.index.add_with_ids(vectors, ids)

    def query(self, vectors, top_k):
        vectors = np.array(vectors)
        if self.metric.lower().startswith("c"): # "cosine"
            vectors = vectors / np.linalg.norm(vectors, axis=1)[:, np.newaxis]  # Normalize vectors
        return self.index.search(vectors, top_k)

    def delete(self, ids):
        # not work for all types, but type 0 is good
        # https://github.com/facebookresearch/faiss/blob/3888f9bb11046f8ab3ddbd9abea22f20ba77f130/tests/test_index_composite.py#L30C9-L30C9
        remove_ids = np.array(ids)
        self.index.remove_ids(np.array(remove_ids))

    def save(self):
        # Check if directory exists, if not, create it
        if not os.path.exists(self.index_name):
            os.makedirs(self.index_name)

        # Save the index and attributes to the directory
        faiss.write_index(self.index, os.path.join(self.index_name, f"{self.index_name}.index"))

        # Save other attributes
        attributes_dict = self.__dict__.copy()
        del attributes_dict['index']
        with open(os.path.join(self.index_name, f"{self.index_name}.pkl"), 'wb') as f:
            pickle.dump(attributes_dict, f)
        logging.info(f"{self.index_name}' index saved.")

    def remove_index(self):
        if os.path.exists(self.index_name):
            shutil.rmtree(self.index_name)
            logging.info(f"Index '{self.index_name}' removed!")
        else:
            logging.info(f"Index '{self.index_name}' does not exist!")

    @classmethod
    def _empty(cls):
        """
        Creates an "empty" instance of the faissDB object. Used for loading purposes.

        Returns:
        - An "empty" faissDB object.

        Example (Internal use only):
        instance = cls._empty()
        """
        instance = super().__new__(cls)
        instance.__dict__ = {}
        return instance

    @classmethod
    def load(cls, index_name):
        instance = cls._empty()

        # Load Faiss index
        instance.index = faiss.read_index(os.path.join(index_name, f"{index_name}.index"))

        # Load other attributes
        with open(os.path.join(index_name, f"{index_name}.pkl"), 'rb') as f:
            attributes = pickle.load(f)
            instance.__dict__.update(attributes)
        logging.info(f"{index_name}' index loaded.")
        return instance


