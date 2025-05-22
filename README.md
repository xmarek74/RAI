# Plant Disease Classification

Tento projekt implementuje konvoluční neuronovou síť (CNN) pro klasifikaci nemocí rostlin na základě obrázků listů.

## Funkce
- Trénování CNN modelu pomocí Keras s augmentací dat
- Vyhodnocení modelu s metrikami jako F1-score, preciznost, recall
- Grafická aplikace v PyQt5 pro snadné načtení modelu a obrázku, spuštění analýzy a zobrazení výsledku
- Podpora standardních formátů modelů (.h5, .keras) a vstupních obrázků

## Požadavky
- Python 3.x
- Keras, TensorFlow
- PyQt5
- Další závislosti jsou uvedeny v `requirements.txt`

## Instalace
1. Vytvořte a aktivujte virtuální prostředí:
   ```bash
   python -m venv venv
   source venv/bin/activate   # Linux/macOS
   venv\Scripts\activate      # Windows
   pip install -r requirements.txt
   py ./main.py -u            # spuštění help zprávy pro použití
   py ./gui.py                # spuštění grafického rozhraní
