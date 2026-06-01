import math
import logging
import numpy as np
from abc import ABC, abstractmethod
from sklearn.feature_extraction.text import TfidfVectorizer

logger = logging.getLogger(__name__)

class RecommenderMethod(ABC):
    """
    Classe base abstracta per als diferents algorismes de recomanació.
    """

    def __init__(self):
        self._dataset = None

    def entrenar(self, dataset):
        self._dataset = dataset

    @abstractmethod
    def preduir_puntuacio(self, usuari_id, item_id):
        pass

    def recomanar(self, usuari_id, top_n=5):
        logger.info(f"Generant top {top_n} recomanacions per a l'usuari {usuari_id}.")
        vots_usuari = self._dataset.obtenir_ratings_usuari(usuari_id)
        tots_items = self._dataset.obtenir_tots_items()
        
        candidats = []
        for item_id in tots_items:
            if vots_usuari.get(item_id, 0) == 0:
                score = self.preduir_puntuacio(usuari_id, item_id)
                candidats.append((item_id, score))

        candidats.sort(key=lambda x: x[1], reverse=True)
        return candidats[:top_n]

    def avaluar_usuari(self, usuari_id):
        vots_reals = self._dataset.obtenir_ratings_usuari(usuari_id)
        items_test = {i: r for i, r in vots_reals.items() if r > 0}

        if not items_test:
            logger.warning(f"L'usuari {usuari_id} no té prou vots reals per fer test d'error.")
            return {"MAE": 0.0, "RMSE": 0.0, "detalls": []}

        suma_abs = 0.0
        suma_quad = 0.0
        detalls = []

        for item_id, valor_real in items_test.items():
            pred = self.preduir_puntuacio(usuari_id, item_id)
            error = pred - valor_real
            suma_abs += abs(error)
            suma_quad += error ** 2
            detalls.append((item_id, pred, valor_real))

        n = len(items_test)
        mae = suma_abs / n
        rmse = math.sqrt(suma_quad / n)

        return {"MAE": mae, "RMSE": rmse, "detalls": detalls}


class SimpleRecommender(RecommenderMethod):
    """
    Sistema de recomanació popular simple amb factor corrector ponderat.
    """

    def __init__(self, min_vots=3):
        super().__init__()
        self._min_vots = min_vots
        self._avg_global = 0.0
        self._item_stats = {}

    def entrenar(self, dataset):
        super().entrenar(dataset)
        logger.info("Entrenant mètode de recomanació Simple popular.")
        tots_items = self._dataset.obtenir_tots_items()
        usuaris = self._dataset.obtenir_tots_usuaris()
        
        # 1. Primer pas: Recollim tots els vots reals dels fitxers
        recompte_vots = {i: [] for i in tots_items}
        for u in usuaris:
            for i, r in self._dataset.obtenir_ratings_usuari(u).items():
                if r > 0 and i in recompte_vots:
                    recompte_vots[i].append(r)

        # CORREGIT: Aquest bloc ara està fora dels bucles for d'usuaris,
        # d'aquesta manera només es calcula una única vegada al final.
        mitjanes_valides = []
        for i, llista in recompte_vots.items():
            nvots = len(llista)
            avg_i = np.mean(llista) if nvots > 0 else 0.0
            self._item_stats[i] = {'num_vots': nvots, 'avg_item': avg_i}
            if nvots >= self._min_vots:
                mitjanes_valides.append(avg_i)

        self._avg_global = np.mean(mitjanes_valides) if mitjanes_valides else 0.0

    def preduir_puntuacio(self, usuari_id, item_id):
        stats = self._item_stats.get(item_id, {'num_vots': 0, 'avg_item': 0.0})
        vots = stats['num_vots']
        avg_i = stats['avg_item']

        if vots < self._min_vots:
            return 0.0

        score = ((vots / (vots + self._min_vots)) * avg_i) + \
                ((self._min_vots / (vots + self._min_vots)) * self._avg_global)
        return score


class CollaborativeRecommender(RecommenderMethod):
    """
    Filtre col·laboratiu basat en usuaris i distància del cosinus.
    """

    def __init__(self, k=2):
        super().__init__()
        self._k = k
        self._user_means = {}

    def entrenar(self, dataset):
        super().entrenar(dataset)
        logger.info("Entrenant mètode col·laboratiu d'usuaris similars.")
        for u in self._dataset.obtenir_tots_usuaris():
            vots = [r for r in self._dataset.obtenir_ratings_usuari(u).values() if r > 0]
            self._user_means[u] = np.mean(vots) if vots else 0.0

    def _calcular_similitud(self, u_id, v_id):
        r_u = self._dataset.obtenir_ratings_usuari(u_id)
        r_v = self._dataset.obtenir_ratings_usuari(v_id)

        comuns = [i for i in r_u if i in r_v and r_u[i] > 0 and r_v[i] > 0]
        if not comuns:
            return 0.0

        num = sum(r_u[i] * r_v[i] for i in comuns)
        den_u = math.sqrt(sum(r_u[i]**2 for i in comuns))
        den_v = math.sqrt(sum(r_v[i]**2 for i in comuns))

        if den_u == 0 or den_v == 0:
            return 0.0
        return num / (den_u * den_v)

    def preduir_puntuacio(self, usuari_id, item_id):
        tots_u = self._dataset.obtenir_tots_usuaris()
        similituds = []

        for v in tots_u:
            if v != usuari_id:
                sim = self._calcular_similitud(usuari_id, v)
                if sim > 0:
                    r_v = self._dataset.obtenir_ratings_usuari(v)
                    if r_v.get(item_id, 0) > 0:
                        similituds.append((v, sim))

        if not similituds:
            return self._user_means.get(usuari_id, 0.0)

        similituds.sort(key=lambda x: x[1], reverse=True)
        veins = similituds[:self._k]

        num_pred = 0.0
        den_pred = 0.0
        for v, sim in veins:
            r_vi = self._dataset.obtenir_ratings_usuari(v)[item_id]
            mu_v = self._user_means[v]
            num_pred += sim * (r_vi - mu_v)
            den_pred += abs(sim)

        mu_u = self._user_means.get(usuari_id, 0.0)
        if den_pred == 0:
            return mu_u

        prediccio = mu_u + (num_pred / den_pred)
        return max(0.0, min(prediccio, self._dataset.obtenir_max_rating()))


class ContentBasedRecommender(RecommenderMethod):
    """
    Sistema basat en contingut (Fase 2) mitjançant matrius de text TF-IDF.
    """

    def __init__(self):
        super().__init__()
        self._vectorizer = TfidfVectorizer(stop_words='english')
        self._tfidf_matrix = None
        self._item_map = {}

    def entrenar(self, dataset):
        super().entrenar(dataset)
        logger.info("Entrenant mètode de contingut textual (TF-IDF).")
        tots_items = self._dataset.obtenir_tots_items()
        corpus = []
        
        for idx, (i_id, info) in enumerate(tots_items.items()):
            corpus.append(info['caracteristiques'])
            self._item_map[i_id] = idx

        if corpus:
            self._tfidf_matrix = self._vectorizer.fit_transform(corpus).toarray()
            logger.debug(f"Dimensions de la matriu de característiques: {self._tfidf_matrix.shape}")

    def preduir_puntuacio(self, usuari_id, item_id):
        if item_id not in self._item_map:
            return 0.0

        r_u = self._dataset.obtenir_ratings_usuari(usuari_id)
        vots_actius = {i: r for i, r in r_u.items() if r > 0 and i in self._item_map}

        if not vots_actius:
            return 0.0

        num_perfil = np.zeros(self._tfidf_matrix.shape[1])
        den_perfil = 0.0

        for i, rating in vots_actius.items():
            idx = self._item_map[i]
            num_perfil += rating * self._tfidf_matrix[idx]
            de_perfil = den_perfil + rating  # den_perfil += rating
            den_perfil += rating

        if den_perfil == 0:
            return 0.0

        perfil = num_perfil / den_perfil

        idx_tgt = self._item_map[item_id]
        vec_item = self._tfidf_matrix[idx_tgt]

        num_cos = np.dot(perfil, vec_item)
        den_cos = (math.sqrt(np.sum(perfil**2)) * math.sqrt(np.sum(vec_item**2)))

        similitud = (num_cos / den_cos) if den_cos > 0 else 0.0
        return similitud * self._dataset.obtenir_max_rating()