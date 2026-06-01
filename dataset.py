import csv
import logging
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)

class Dataset(ABC):
    """
    Classe base abstracta per gestionar els conjunts de dades del sistema.
    """

    def __init__(self):
        self._ratings = {}
        self._items = {}
        self._max_rating = 5.0

    @abstractmethod
    def carregar_dades(self):
        """
        Llegeix i estructura els fitxers de dades originals des de disc.
        """
        pass

    def obtenir_ratings_usuari(self, usuari_id):
        return self._ratings.get(usuari_id, {})

    def obtenir_tots_usuaris(self):
        return list(self._ratings.keys())

    def obtenir_tots_items(self):
        return self._items

    def obtenir_info_item(self, item_id):
        return self._items.get(item_id)

    def obtenir_max_rating(self):
        return self._max_rating


class MovieLensDataset(Dataset):
    """
    Gestió específica de fitxers per al conjunt de dades MovieLens100k.
    """

    def __init__(self, ruta_movies="dataset/MovieLens100k/movies.csv", ruta_ratings="dataset/MovieLens100k/ratings.csv"):
        super().__init__()
        self._ruta_movies = ruta_movies
        self._ruta_ratings = ruta_ratings
        self._max_rating = 5.0

    def carregar_dades(self):
        logger.info("Carregant el dataset MovieLens100k de manera manual.")
        try:
            with open(self._ruta_movies, mode='r', encoding='utf-8') as f:
                reader = csv.reader(f)
                next(reader)
                for fila in reader:
                    if len(fila) >= 3:
                        m_id, titol, generes = fila[0], fila[1], fila[2]
                        self._items[m_id] = {
                            'titol': titol,
                            'caracteristiques': generes.replace('|', ' '),
                            'display': f"Pel·lícula: {titol} ({generes})"
                        }
            logger.debug(f"S'han indexat {len(self._items)} pel·lícules.")

            with open(self._ruta_ratings, mode='r', encoding='utf-8') as f:
                reader = csv.reader(f)
                next(reader)
                for fila in reader:
                    if len(fila) >= 3:
                        u_id, m_id, rating = fila[0], fila[1], float(fila[2])
                        if u_id not in self._ratings:
                            self._ratings[u_id] = {}
                        self._ratings[u_id][m_id] = rating
            logger.info("Càrrega de MovieLens100k finalitzada.")
        except FileNotFoundError as e:
            logger.critical(f"No s'ha pogut trobar un fitxer de MovieLens: {e}")
            raise e


class BooksDataset(Dataset):
    """
    Gestió de dades adaptada al Book Recommendation Dataset.
    """

    def __init__(self, ruta_books="dataset/Books/Books.csv", ruta_ratings="dataset/Books/Ratings.csv"):
        super().__init__()
        self._ruta_books = ruta_books
        self._ruta_ratings = ruta_ratings
        self._max_rating = 10.0

    def carregar_dades(self):
        logger.info("Carregant subconjunt del dataset Books (Màxim 10.000 llibres).")
        try:
            limit_llibres = 10000
            with open(self._ruta_books, mode='r', encoding='utf-8', errors='ignore') as f:
                reader = csv.reader(f)
                next(reader)
                comptador = 0
                for fila in reader:
                    if len(fila) >= 5:
                        isbn, titol, autor, editorial = fila[0], fila[1], fila[2], fila[4]
                        self._items[isbn] = {
                            'titol': titol,
                            'caracteristiques': f"{autor} {editorial}",
                            'display': f"Llibre: {titol} - Autor: {autor}"
                        }
                        comptador += 1
                        if comptador >= limit_llibres:
                            break
            logger.debug(f"S'han indexat {len(self._items)} llibres a memòria.")

            with open(self._ruta_ratings, mode='r', encoding='utf-8', errors='ignore') as f:
                reader = csv.reader(f)
                next(reader)
                for fila in reader:
                    if len(fila) >= 3:
                        u_id, isbn, rating = fila[0], fila[1], float(fila[2])
                        if isbn in self._items:
                            if u_id not in self._ratings:
                                self._ratings[u_id] = {}
                            self._ratings[u_id][isbn] = rating
            logger.info("Càrrega del dataset de llibres finalitzada.")
        except FileNotFoundError as e:
            logger.critical(f"No s'ha pogut trobar un fitxer de Books: {e}")
            raise e