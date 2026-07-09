# PROMPT CONSOLIDAT PER AL MODEL DEL IDE
## EVO_SKILL_BUILDER — Auto-segmentació de caràcters per histograma X dins una ROI

Context del projecte:
Estem treballant en una app Python/Streamlit anomenada `EVO_SKILL_BUILDER`. La finalitat és construir datasets de caràcters/dígits a partir de documents escanejats. Fins ara l'usuari havia de retallar manualment cada caràcter dins una ROI, però volem substituir aquesta feina manual per una detecció automàtica/semi-automàtica basada en histograma/projecció vertical sobre l'eix X.

Objectiu principal:
Quan l'usuari selecciona una ROI general que conté una cadena de caràcters, per exemple `5605`, el sistema ha de detectar automàticament on comença i acaba cada caràcter dins de la ROI, generar crops individuals, mostrar-los en pantalla, permetre validar/etiquetar i guardar-los al dataset.

No volem una demo aïllada. Volem integrar-ho de forma neta dins de l'app existent, sense trencar el flux actual manual.

---

# 1. Funcionalitat a implementar

Afegir un mode nou de segmentació:

- `Manual ROI`: comportament actual, l'usuari retalla manualment.
- `Auto X-Histogram`: l'usuari selecciona una ROI gran i el sistema intenta separar els caràcters automàticament.

La idea és:

1. Rebre una imatge ROI, en BGR, RGB o grayscale.
2. Convertir-la a grayscale.
3. Aplicar binarització robusta.
4. Assegurar que la tinta/text sigui blanc i el fons negre.
5. Calcular la projecció vertical:
   - per cada columna X, sumar quants píxels actius/tinta hi ha.
6. Detectar columnes actives.
7. Agrupar columnes consecutives en segments.
8. Fusionar segments separats per gaps molt petits.
9. Filtrar soroll per amplada mínima.
10. Per cada segment X, calcular també el rang Y real del caràcter.
11. Afegir padding configurable.
12. Retornar:
    - crops individuals
    - bounding boxes
    - imatge binària debug
    - histograma/projecció X
    - segments crus i fusionats

---

# 2. Requisits tècnics

Implementa una funció principal:

```python
def auto_split_characters_by_x_histogram(
    roi_img,
    threshold_method="otsu",
    min_char_width=3,
    min_ink_per_col=1,
    gap_merge_px=2,
    pad_x=2,
    pad_y=2,
    invert_if_needed=True,
    apply_morph_open=True,
    apply_morph_close=False,
):
    """
    Segmenta automàticament una ROI en caràcters mitjançant histograma X.

    Args:
        roi_img: imatge ROI en BGR, RGB o grayscale.
        threshold_method: "otsu" o "adaptive".
        min_char_width: amplada mínima d'un segment per considerar-lo caràcter.
        min_ink_per_col: mínim de píxels actius en una columna X.
        gap_merge_px: fusiona segments separats per gaps iguals o menors a aquest valor.
        pad_x: padding horitzontal dels crops.
        pad_y: padding vertical dels crops.
        invert_if_needed: si True, intenta garantir tinta blanca sobre fons negre.
        apply_morph_open: si True, aplica obertura morfològica suau per treure soroll.
        apply_morph_close: si True, aplica tancament morfològic suau per unir fragments.

    Returns:
        crops: list[np.ndarray]
        boxes: list[tuple[int, int, int, int]]  # x1, y1, x2, y2 dins la ROI
        debug: dict amb gray, binary, x_projection, raw_segments, merged_segments
    """
```

La funció ha de ser robusta i no petar si:

- la ROI és buida
- la ROI és massa petita
- no es detecten caràcters
- la imatge ve en grayscale
- la imatge ve en color
- el fons és blanc i el text negre
- el fons és negre i el text blanc

En cas de no detectar res, ha de retornar:

```python
return [], [], debug
```

mai fer crash.

---

# 3. Algoritme esperat

Pseudocodi:

```python
if roi_img is None or roi_img.size == 0:
    return [], [], debug_empty

convertir a grayscale
aplicar blur suau GaussianBlur(3,3)

si threshold_method == "adaptive":
    adaptiveThreshold
sinó:
    Otsu threshold

si invert_if_needed:
    comptar píxels blancs i negres
    si hi ha més blanc que negre, probablement el fons és blanc
    invertir perquè la tinta quedi blanca i el fons negre

si apply_morph_open:
    morphologyEx(binary, MORPH_OPEN, kernel 2x2)

si apply_morph_close:
    morphologyEx(binary, MORPH_CLOSE, kernel 2x2)

ink = binary > 0
x_projection = sum(ink, axis=0)
active_cols = x_projection >= min_ink_per_col

recórrer active_cols i crear segments continus (x_start, x_end)
fusionar segments si gap <= gap_merge_px
filtrar segments amb amplada < min_char_width

per cada segment:
    calcular y_projection dins del segment
    trobar y1, y2 on hi ha tinta
    aplicar padding
    retallar crop original, no la binària
    afegir crop i box

return crops, boxes, debug
```

---

# 4. Funció de preview visual

Implementa també:

```python
def draw_detected_boxes(roi_img, boxes):
    """
    Dibuixa rectangles i índexs sobre la ROI per visualitzar els caràcters detectats.
    Retorna una imatge BGR/RGB compatible amb st.image.
    """
```

Ha de:

- convertir grayscale a color si cal
- dibuixar rectangles verds
- posar número de caràcter 1, 2, 3...
- no petar si boxes és buit

---

# 5. Integració Streamlit

Afegir controls a la UI propers al lloc on ara es gestiona la ROI:

```python
segmentation_mode = st.radio(
    "Mode segmentació caràcters",
    ["Manual ROI", "Auto X-Histogram"],
    index=1
)
```

Quan `segmentation_mode == "Auto X-Histogram"`, mostrar paràmetres ajustables:

```python
threshold_method = st.selectbox("Threshold", ["otsu", "adaptive"])
min_char_width = st.slider("Amplada mínima caràcter", 1, 30, 3)
min_ink_per_col = st.slider("Tinta mínima per columna", 1, 30, 1)
gap_merge_px = st.slider("Fusionar gaps <= px", 0, 20, 2)
pad_x = st.slider("Padding X", 0, 20, 2)
pad_y = st.slider("Padding Y", 0, 20, 2)
apply_morph_open = st.checkbox("Neteja soroll / morph open", value=True)
apply_morph_close = st.checkbox("Unir fragments / morph close", value=False)
```

Quan hi hagi una ROI seleccionada:

```python
crops, boxes, debug = auto_split_characters_by_x_histogram(...)
preview = draw_detected_boxes(roi_img, boxes)
st.image(preview, caption="Auto-detecció per histograma X")
st.write(f"Detectats {len(crops)} possibles caràcters")
```

Mostrar també, opcionalment en `st.expander("Debug histograma")`:

- imatge binària
- gràfic simple de `x_projection`
- raw_segments
- merged_segments

El gràfic pot ser amb matplotlib o st.line_chart.

---

# 6. Etiquetatge ràpid amb string conegut

Afegir una entrada:

```python
label_string = st.text_input("Text real de la ROI / cadena esperada", "")
```

Si l'usuari escriu `5605` i el sistema detecta 4 crops, llavors assignar:

- crop 1 -> 5
- crop 2 -> 6
- crop 3 -> 0
- crop 4 -> 5

Condició:

```python
if len(label_string) == len(crops):
    labels = list(label_string)
else:
    mostrar warning i permetre etiquetar manualment cada crop
```

Si no coincideix la longitud, no guardar automàticament. Mostrar:

```python
st.warning(f"La cadena té {len(label_string)} caràcters però s'han detectat {len(crops)} crops. Revisa paràmetres o etiqueta manualment.")
```

---

# 7. Visualització dels crops

Mostrar crops detectats en files/columnes:

```python
for i, crop in enumerate(crops):
    st.image(crop, caption=f"Crop {i+1}")
```

Si `label_string` coincideix, mostrar també la label assignada.

Si no coincideix, crear inputs manuals:

```python
manual_label = st.text_input(f"Etiqueta crop {i+1}", key=f"auto_label_{i}")
```

---

# 8. Guardat al dataset

Reutilitzar la funció actual de guardat si ja existeix, per exemple `save_character_crop(...)` o equivalent.

Si no existeix una funció clara, crear una funció auxiliar compatible amb l'estructura actual del dataset:

```python
def save_character_crop(crop_img, label, output_base_dir, prefix="auto"):
    """
    Desa el crop dins una carpeta per etiqueta.
    Exemple:
    dataset/5/auto_YYYYMMDD_HHMMSS_uuid.png
    """
```

Requisits:

- crear carpeta de label si no existeix
- nom únic per evitar sobreescriure
- guardar PNG
- validar que label no sigui buida
- sanititzar label per evitar caràcters problemàtics en noms de carpeta

Botó:

```python
if st.button("Guardar crops detectats al dataset"):
    validar labels
    guardar cada crop
    mostrar resum
```

---

# 9. Casos difícils previstos

Problema: caràcters enganxats, per exemple `11`, `00`, `88`.

La primera versió pot detectar-los com un sol bloc. De moment implementa la versió simple per gaps. Però deixa el codi preparat per afegir més tard una funció:

```python
def split_wide_segments_by_valleys(x_projection, segments, expected_count=None):
    """
    Futur: si un segment és massa ample, buscar valls internes per dividir-lo.
    No cal implementar complet ara si complica massa, però deixa TODO clar.
    """
```

Si és fàcil, pots implementar una primera heurística opcional:

- si `expected_count = len(label_string)` i hi ha menys segments que labels
- buscar mínims locals dins segments massa amples
- dividir pels punts amb menys tinta

Però prioritat: no trencar res. Primer versió simple i estable.

---

# 10. Qualitat del codi

Important:

- No reescriure tota l'app.
- No canviar noms de variables globals sense necessitat.
- Integrar en blocs petits.
- Fer funcions pures sempre que es pugui.
- No eliminar el mode manual.
- Afegir comentaris curts i útils.
- Evitar dependències noves si ja tenim `cv2`, `numpy`, `streamlit`, `matplotlib`.
- Si falta algun import, afegir-lo al principi:

```python
import cv2
import numpy as np
import os
import re
import uuid
from datetime import datetime
```

Recordatori: en aquest projecte ja hem tingut errors per imports que faltaven, com `NameError: name 're' is not defined`. Revisa imports abans de donar el codi final.

---

# 11. Resultat esperat

Vull que retornis:

1. El codi de les funcions noves:
   - `auto_split_characters_by_x_histogram`
   - `draw_detected_boxes`
   - si cal, `save_character_crop`

2. El bloc Streamlit que he d'enganxar dins de `app.py`.

3. Instruccions exactes d'on enganxar-ho:
   - imports
   - funcions auxiliars
   - bloc UI dins del flux de ROI

4. No em donis teoria llarga. Dona codi útil i integrable.

5. Mantén compatibilitat amb Windows i rutes locals.

6. Si detectes que necessites veure `app.py`, demana'l només si és imprescindible. Si pots proposar un patch genèric, fes-ho.

---

# 12. Criteri d'èxit

Considerarem que funciona si:

- Selecciono una ROI amb una cadena tipus `5605`.
- L'app detecta 4 crops.
- Mostra la preview amb rectangles.
- Escric `5605` al camp de text.
- El sistema assigna labels automàticament.
- Clico guardar.
- Es creen imatges separades a les carpetes correctes del dataset.
- Si detecta malament, puc ajustar sliders sense reiniciar l'app.

---

# 13. To i preferència

Prioritat absoluta: funcionalitat pràctica. Res de PowerPoint, res de fum, res de refactor gegant innecessari. Implementació robusta, incremental i fàcil de provar.
