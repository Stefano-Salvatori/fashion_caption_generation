#@title Utility classes execute
from typing import List
import random
import torch
from tqdm import tqdm
import h5py
from dataclasses import dataclass
import numpy as np
import os

"""
FashionGen strings encoding
"""
DEFAULT_STRINGS_ENCODING = "ISO-8859-9"

@dataclass
class Product:
    p_id:int
    caption:str
    image:np.ndarray
    category:str
    subcategory:str
    concat_caption:str = None
    name:str = None
    composition:str = None
    department:str = None
    gender:str = None
    msrpUSD: float = None
    season: str = None
    brand:str = None
    pose:str = "id_gridfs_1"
    index:int =-1


    def __lt__(self, other):
        return self.p_id < other.p_id

    def __eq__(self, other):
        return self.p_id == other.p_id and self.pose == other.pose

    def __hash__(self):
        return hash((self.p_id, self.pose))

    def __str__(self):
      return os.linesep.join([f"{a}: {v}" for a,v in self.__dict__.items() if isinstance(v, str) or isinstance(v,float)])


class FashionGenDataset:
    def __init__(self, file_name):
        self.__subcategory_dict = None
        self.__productid_dict = None
        self.__category_dict = None
        self.__loaded_products = []
        self.__products_lodaded = False
        self.dataset = h5py.File(file_name, mode="r")

    def __getitem__(self, index):
        if isinstance(index, slice):
            return [
                self.get_product(i)
                for i in tqdm(
                    range(
                        min(self.length(), index.start),
                        min(self.length(), index.stop),
                        index.step if index.step is not None else 1,
                    )
                )
            ]
        else:
            return self.get_product(index)

    def __len__(self):
        return self.length()

    def load_products(self):
        """Load all products in a python array."""
        if not self.__products_lodaded:
            print("Loading products...")
            self.__loaded_products = [self.get_product(i) for i in tqdm(range(self.length()))]
            self.__products_lodaded = True

    def length(self):
        return len(self.dataset["input_productID"])

    def distinct_products_in_subcategory(self, subcategory: str):
        return set([p_id for _, p_id in self.subcategory_dict()[subcategory]])

    def distinct_products_in_category(self, category: str):
        return set([p_id for _, p_id in self.category_dict()[category]])

    def raw_h5(self):
        return self.dataset

    def close_file(self):
        self.dataset.close()

    def get_product(self, index, string_encoding = DEFAULT_STRINGS_ENCODING):
        """Return the product in the dataset at the specified index"""
        if self.__products_lodaded:
            return self.__loaded_products[index]
        else:
            return Product(
                p_id=self.dataset["input_productID"][index][0], 
                #name=self.dataset["input_name"][index][0].decode(string_encoding), 
                caption=self.dataset["input_description"][index][0].decode(string_encoding),
                #concat_caption=self.dataset["input_concat_description"][index][0].decode(string_encoding), 
                image=self.dataset["input_image"][index], 
                category=self.dataset["input_category"][index][0].decode(string_encoding), 
                subcategory=self.dataset["input_subcategory"][index][0].decode(string_encoding), 
                #pose=self.dataset["input_pose"][index][0].decode(string_encoding), 
                #composition = self.dataset["input_composition"][index][0].decode(string_encoding),
                #department = self.dataset["input_department"][index][0].decode(string_encoding),
                #gender =  self.dataset["input_gender"][index][0].decode(string_encoding),
                #msrpUSD = float(self.dataset["input_msrpUSD"][index][0]),
                #season = self.dataset["input_season"][index][0].decode(string_encoding),
                #brand = self.dataset["input_brand"][index][0].decode(string_encoding),
                index=index
            )

    def get_product_by_id(self, product_id: int, get_random: bool = False):
        """
        Get the product with the specified product id.
        The image of the product is the first one that appears in the dataset if get_random is set to False, Otherwise a random one is selected
        Args:
            product_id: the id of the product to get
            get_random: if False, the first image of the product will be returned, otherwise a random one is selected
        """
        product_indexes = self.productID_dict()[product_id]
        index = random.choice(product_indexes) if get_random else product_indexes[0]
        return self.get_product(index)

    def subcategory_dict(self) -> dict:
        """A dictionary that foreach 'subcategory' keeps a list of products (and indexes)
        that belong to it"""
        if self.__subcategory_dict is None:
            self.__subcategory_dict = self.__create_attribute_dict("input_subcategory")

        return self.__subcategory_dict

    def category_dict(self) -> dict:
        """A dictionary that foreach 'category' keeps a list of products (and indexes)
        that belong to it"""
        if self.__category_dict is None:
            self.__category_dict = self.__create_attribute_dict("input_category")
        return self.__category_dict

    def productID_dict(self) -> dict:
        """A dictionary that foreach productID keeps the list of indexes in the dataset where those products can be found"""
        if self.__productid_dict is None:
            self.__productid_dict = self.__create_productID_dict()
        return self.__productid_dict

    def get_random_product(self, id_different_from: int = None) -> Product:
        """
        Return a random product from the dataset
        Args:
            id_different_from: an id of a product that cannot be taken. Use None if any products can be returned
        Returns:
            A random product with an id different from 'id_different_from' (if specified)
        """
        random_product = self[random.choice(range(self.length() - 1))]
        while id_different_from is not None and random_product.p_id == id_different_from:
            random_product = self[random.choice(range(self.length() - 1))]
        return random_product

    def get_random_products(self, id_different_from: int = None, n: int = 1) -> List[Product]:
        """
        Get a list of random products
        Args:
            id_different_from: an id of a product that cannot be taken. Use None if any products can be returned
            n: number of products to return
        Returns:
            A list of n products with product ids different from id_different_from (is specified)
        """
        # we don't call the 'get_random_product' function n times because we want to avoid taking the same product (and image)
        # multiple times.
        random_indexes = random.sample(range(self.length() - 1), n)
        random_products = [self[index] for index in random_indexes]
        # replace all products that have id equals to id_different_from with other random ones.
        # It can happen after this step that the same negative product is repeated
        random_products = map(
            lambda p: self.get_random_product(id_different_from)
            if id_different_from is not None and p.p_id == id_different_from
            else p,
            random_products,
        )
        return list(random_products)

    def get_same_subcategory_of(self, product: Product, n: int = 1) -> List[Product]:
        """
        Get a list of random products with the same subcategory of the given product (but with different product ids).
        If the given subcategory has less products than the number n specified, some random product will be returned
        Args:
            product: the product for which we want to find other ones with the same subcategory
            n: The number of products to return
        Returns:
            A list of n products with different ids but same subcategory of the given one.

        """
        negative_prods_idxs = [
            index for index, p_id in self.subcategory_dict()[product.subcategory] if p_id != product.p_id
        ]
        return self.__get_random_products_by_indexes(negative_prods_idxs, product.p_id, n)

    def get_same_category_of(self, product: Product, n: int = 1) -> List[Product]:
        """
        Get a list of random products with the same category of the given product (but with different product ids).
        If the given category has less products than the number n specified, some random product will be returned
        Args:
            product: the product for which we want to find other ones with the same category
            n: The number of products to return
        Returns:
            A list of n products with different ids but same category of the given one.
        """
        negative_prods_idxs = [index for index, p_id in self.category_dict()[product.category] if p_id != product.p_id]
        return self.__get_random_products_by_indexes(negative_prods_idxs, product.p_id, n)

    def __get_random_products_by_indexes(
        self, indexes: List[int], id_different_from: int = None, n: int = 1
    ) -> Product:
        """
        Return a given number of products randomly sampling indexes from a given list. If the number 'n' specified
        is bigger the number of indexes, other random products will be takens (with ids different from 'id_different_from')
        """
        if len(indexes) >= n:
            return [self.get_product(idx) for idx in random.sample(indexes, n)]
        else:
            return self.__get_random_products_by_indexes(
                indexes, id_different_from, len(indexes)
            ) + self.get_random_products(id_different_from, n - len(indexes))

    def get_batch(self, features, from_index, to_index):
        """Get a batch of the raw h5 dataset

        Args:
            features(list(str)): list of names of features present in the dataset
                that should be returned.
            batch_number(int): the id of the batch to be returned.
            batch_size(int): the mini-batch size
        Returns:
            A list of numpy arrays of the requested features"""
        batch = {}
        for feature in features:
            batch[feature] = self.dataset[feature][from_index:to_index]
        return batch

    def __create_productID_dict(self):
        print(f"Creating productID dictionary...")
        prodID_to_index = {}
        for i in tqdm(range(self.length()), position=0, leave=True):
            prod_id = self.dataset["input_productID"][i][0]
            if prod_id in prodID_to_index:
                prodID_to_index[prod_id].append(i)
            else:
                prodID_to_index[prod_id] = [i]
        return prodID_to_index

    def __create_attribute_dict(self, attribute):
        print(f"Creating {attribute} dictionary...")
        attribute_to_prod = {}
        for i in tqdm(range(self.length()), position=0, leave=True):
            attribute_value = self.dataset[attribute][i][0]
            prod_id = self.dataset["input_productID"][i][0]
            if attribute_value in attribute_to_prod:
                # indexes = attribute_to_prod[attribute_value]
                # if (prod_id not in [p for i, p in indexes]):
                attribute_to_prod[attribute_value].append((i, prod_id))
            else:
                attribute_to_prod[attribute_value] = [(i, prod_id)]
        return attribute_to_prod
