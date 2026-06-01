import sys
import os
import pickle
import logging
from datetime import datetime

from dataset import MovieLensDataset, BooksDataset
from recommender import SimpleRecommender, CollaborativeRecommender, ContentBasedRecommender

def inicialitzar_logging():
    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    fitxer_log_personalitzat = f"log_{ts}.txt"

    logger_arrel = logging.getLogger()
    logger_arrel.setLevel(logging.DEBUG)

    formatari = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    fh = logging.FileHandler(fitxer_log_personalitzat, encoding='utf-8')
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(formatari)

    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(formatari)

    logger_arrel.addHandler(fh)
    logger_arrel.addHandler(ch)

    if not os.path.exists("log.txt"):
        with open("log.txt", "w", encoding="utf-8") as f:
            f.write(f"Arxiu vinculat a traces de sessió: {fitxer_log_personalitzat}\n")

def mostrar_menu():
    print("\n" + "="*40)
    print("      SISTEMA DE RECOMANACIÓ - ACCIONS")
    print("="*40)
    print("1. Recomanar ítems (Top 5)")
    print("2. Avaluar mètode per a un usuari")
    print("3. Sortir del programa")
    print("="*40)

def main():
    inicialitzar_logging()
    logging.info("Arrancant l'aplicació principal del sistema de recomanació.")

    if len(sys.argv) < 3:
        logging.error("Falten paràmetres passats per línia de comandes.")
        print("Ús del sistema: python main.py <dataset> <method>")
        print("  <dataset>: movies | books")
        print("  <method> : simple | collaborative | content")
        sys.exit(1)

    p_dataset = sys.argv[1].lower()
    p_method = sys.argv[2].lower()

    nom_pickle = f"recommender_{p_dataset}_{p_method}.dat"
    recommender_instance = None

    if os.path.exists(nom_pickle):
        logging.info(f"S'ha detectat el fitxer binari {nom_pickle}. Carregant estat...")
        try:
            with open(nom_pickle, 'rb') as f:
                recommender_instance = pickle.load(f)
            logging.info("Instància recuperada i llesta per al seu ús immediat.")
        except Exception as e:
            logging.error(f"Error en carregar pickle: {e}. Es carregarà manualment.")

    if recommender_instance is None:
        logging.info("Inicialitzant l'estat i processant fitxers CSV.")
        
        if p_dataset == "movies":
            dt = MovieLensDataset()
        elif p_dataset == "books":
            dt = BooksDataset()
        else:
            logging.critical(f"Dataset no vàlid: {p_dataset}")
            print(f"Error: Dataset '{p_dataset}' no vàlid. Opcions: movies, books")
            sys.exit(1)

        try:
            dt.carregar_dades()
        except Exception as e:
            logging.critical("No s'han pogut indexar les dades CSV d'origen.")
            sys.exit(1)

        if p_method == "simple":
            recommender_instance = SimpleRecommender()
        elif p_method == "collaborative":
            recommender_instance = CollaborativeRecommender()
        elif p_method == "content":
            recommender_instance = ContentBasedRecommender()
        else:
            logging.critical(f"Algorisme no reconegut: {p_method}")
            print(f"Error: Mètode '{p_method}' no vàlid. Opcions: simple, collaborative, content")
            sys.exit(1)

        recommender_instance.entrenar(dt)

        try:
            logging.info(f"Escriptura persistent de seguretat a {nom_pickle}.")
            with open(nom_pickle, 'wb') as f:
                pickle.dump(recommender_instance, f)
        except Exception as e:
            logging.warning(f"No s'ha pogut guardar l'arxiu .dat: {e}")

    while True:
        print("\n" + "-"*50)
        usuari_id = input("Introdueix el ID de l'usuari (deixa en blanc o prem Enter per sortir): ").strip()
        
        if usuari_id == "":
            logging.info("Cadena buida detectada. Aturant execució.")
            print("Sortint del programa. Gràcies per utilitzar el sistema!")
            break

        if usuari_id not in recommender_instance._dataset.obtenir_tots_usuaris():
            logging.warning(f"L'usuari '{usuari_id}' no s'ha trobat en el registre històric.")
            print(f"L'usuari '{usuari_id}' no té cap registre associat en aquest dataset. Reintenta-ho.")
            continue

        mostrar_menu()
        accio = input("Selecciona una opció de menú (1-3): ").strip()

        if accio == "1":
            resultats = recommender_instance.recomanar(usuari_id, top_n=5)
            print(f"\n--- TOP 5 RECOMANACIONS PER A L'USUARI {usuari_id} ---")
            if not resultats:
                print("No hi ha candidats disponibles per recomanar.")
            for index, (item_id, score) in enumerate(resultats, start=1):
                info = recommender_instance._dataset.obtenir_info_item(item_id)
                txt = info['display'] if info else "Sense descripció metadades"
                print(f"{index}. [ID: {item_id}] {txt} -> Predicció Score: {score:.2f}")

        elif accio == "2":
            avaluacio = recommender_instance.avaluar_usuari(usuari_id)
            print(f"\n--- AVALUACIÓ DEL RENDIMENT (Usuari: {usuari_id}) ---")
            print(f"Mean Absolute Error (MAE)    : {avaluacio['MAE']:.4f}")
            print(f"Root Mean Square Error (RMSE): {avaluacio['RMSE']:.4f}")
            
            detalls = avaluacio['detalls']
            if detalls:
                print("\nDetall de control de desviació (Mostra límit de 10 vots):")
                for item_id, pred, real in detalls[:10]:
                    info = recommender_instance._dataset.obtenir_info_item(item_id)
                    titol = info['titol'] if info else "Ítem desconegut"
                    print(f"  - Ítem {item_id} ({titol[:25]}...): Predicció={pred:.2f} | Real={real:.2f}")
            else:
                print("L'usuari no conté suficients vots reals per extreure un llistat de desviació.")

        elif accio == "3":
            logging.info("S'ha sol·licitat manualment la sortida del bucle actiu.")
            print("Sortint del programa de recomanacions.")
            break
        else:
            print("Opció introduïda invàlida o fora de rang (Tria 1, 2 o 3).")

if __name__ == "__main__":
    main()