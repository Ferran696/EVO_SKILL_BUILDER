# 📖 GUIA RÀPIDA D'USUARI (JOB AID) — EVO Skill Builder

> **Per a qui és aquesta guia?**
> Aquesta guia està pensada perquè **qualsevol persona**, sense saber res d'informàtica ni d'intel·ligència artificial, pugui fer servir l'eina des del primer minut.
>
> **Què fa aquesta eina?**
> Llegeix automàticament el número d'albarà o contracte dels teus documents escanejats (PDF o imatges) i **et retorna tots els fitxers canviats de nom amb el seu codi real** (per exemple, transforma `18.pdf` en `0529820.pdf`), llestos per arxivar o treballar.

---

## 🚀 PAS 0: Com obrir el programa

1. Si l'aplicació ja està oberta al navegador d'Internet (Chrome/Edge), salta directament a la **PART 1**.
2. Si està tancada:
   - Obre la carpeta del projecte `C:\Ferran\01_PROJECTES\EVO_SKILL_BUILDER`.
   - Obre el terminal (PowerShell) i enganxa aquesta línia fent `Enter`:
     ```powershell
     .\.venv\Scripts\streamlit.exe run app.py
     ```
   - S'obrirà automàticament una pestanya al teu navegador amb el títol **🧠 EVO Skill Builder**.

---

## ⚡ PART 1: ÚS DIARI (Processar documents i canviar-los el nom en 2 minuts)

*Fes servir aquesta part quan el projecte ja existeix (per exemple `contractos_cb_mix`) i només vols passar una remesa de PDFs nous.*

### Pas 1.1 · Entra a la pantalla de treball en bloc
1. Mira la **columna de l'esquerra (Navegació)**.
2. Fes clic a l'última opció: **`07 · Run batch`**.
3. A dalt de tot, al desplegable **Projecte**, tria el teu projecte (per exemple: `contractos_cb_mix_...`).

### Pas 1.2 · Comprova que les opcions automàtiques estan marcades
No cal que toquis res complicat; només assegura't que tens marcat el següent:
- **Estratègia de lectura:** `🔄 Enrutador multi-format automàtic` *(així el programa reconeix sol si el document té codi de barres, logo de Carburos o format antic amb punts)*.
- Dins de **⚙️ Opcions avançades de lectura**:
  - **Mètode de segmentació:** `🔲 Components Connectats (Robust a desplaçaments)`
  - **🎯 Ancoratge visual automàtic:** **Marcat (✔)** *(això fa que el programa busqui el número encara que el full s'hagi escanejat tort o desplaçat)*.
  - **📐 Deskewing:** Desmarcat per defecte *(marca-ho només si veus que han escanejat els papers molt inclinats)*.

### Pas 1.3 · Puja els teus PDFs i prem "Processar"
1. Baixa fins a **3) Upload documents nous**.
2. Arrossega tots els teus fitxers PDF (o fes clic a **`Browse files`** i selecciona'ls).
3. Fes clic al botó gran **`▶️ Processar batch`**.
4. Espera uns segons mentre la barra de progrés arriba al 100%.

### Pas 1.4 · Revisa la taula de resultats
Baixa a **4) Resultats batch, Reindexació i Exportació**. Veuràs una taula molt clara amb el resultat de cada PDF:

| Què veuràs a la taula | Què vol dir |
| :--- | :--- |
| **`Exportar` (Casella ✔)** | Si està marcada, aquest document s'inclourà a la descàrrega final amb el nom nou. Es marca sola per a tots els documents `OK` o corregits manualment (`MANUAL_OK`). |
| **`Fitxer original`** | El nom que tenia el document quan l'has pujat (ex: `18.pdf`). |
| **`Codi (editable)`** | El número que ha llegit el programa (ex: `0529820`). **Si fas doble clic a sobre de la casella, pots escriure el número bo i prémer `Enter`**: el document passarà sol a `MANUAL_OK` i es marcarà per exportar! |
| **`Nou nom fitxer`** | Com es dirà el fitxer quan el descarreguis (ex: `0529820.pdf`). Si hi ha dos documents amb el mateix número, el segon es posarà sol com `0529820_2.pdf` perquè mai es perdi cap fitxer. |
| **`Status`** | 🟢 **`OK`** = lectura automàtica fiable.<br>🟠 **`REVIEW`** / 🔴 **`ERROR`** = el programa té algun dubte i et demana que el miris un segon.<br>🔵 **`MANUAL_OK`** = l'has revisat o corregit tu mateix i ja està llest per exportar. |

---

### Pas 1.4.B · 🛠️ Estació de Revisió Ràpida per als casos `REVIEW` (sense sortir de la pantalla)

Si a la taula t'ha sortit algun document en estat **`REVIEW`** o **`ERROR`**, **no cal que obris el PDF fora del programa ni que vagis pujant i baixant per la pàgina**. Baixa directament al bloc **5) Estació de Revisió Ràpida (`REVIEW`) i Detall Visual**:

#### 1. Com funciona el visor de revisió?
- **Cua intel·ligent (`⚠️ Només pendents`):** El programa ja et filtra automàticament només els documents que tenen dubtes (`REVIEW` / `ERROR`). Quan en valides un, **salta sol al següent pendent** fins que el comptador arriba a zero (`🎉 Tots els documents estan validats`).
- **Navegació fàcil:** Pots moure't amb els botons **`⬅️ Anterior`** i **`➡️ Següent`** o triar qualsevol document al desplegable.
- **Tot a la vista (columna esquerra `👁️ Visor`):**
  1. Veus el **retall de la ROI** que ha llegit la IA.
  2. Veus **cada dígit separat** amb el tant per cent de seguretat (ex: `#1 · 5 (99%)`).
  3. Tens obert automàticament el desplegable **`📄 Veure capçalera del document original (amb ROI marcada)`**, on veus la part de dalt del full sencer amb el requadre vermell perquè puguis llegir el número real amb els teus propis ulls.

#### 2. Tres maneres d'arreglar un document `REVIEW` en 2 segons (columna dreta `✏️ Correcció i Validació`):

* **CAS A · El número es llegeix bé a la imatge, però el programa s'ha equivocat en alguna xifra:**
  1. Mira la imatge de l'esquerra per veure quin és el número real.
  2. A la dreta, escriu el número correcte dins la casella **`Codi del document`**.
  3. Prem el botó blau **`✅ Validar codi i marcar per exportar`**.
  4. *Què passa?* El document canvia a **`MANUAL_OK`**, se li activa la casella **`Exportar`**, se li calcula el nou nom de fitxer i la pantalla passa automàticament al següent document pendent!

* **CAS B · Vols validar el número i, de pas, fer que el programa n'aprengui (`🧠 Active Learning`):**
  1. Si veus que els quadradets dels dígits estan ben tallats (un número a cada quadradet) però el programa ha confós, per exemple, un `8` amb un `3`:
  2. Escriu el codi correcte a **`Codi del document`**.
  3. Prem **`🧠 Validar + Guardar dígits per entrenar (Active Learning)`**.
  4. *Què passa?* Fa el mateix que el botó de validar, però a més **guarda els retalls d'aquests números a la memòria d'entrenament** perquè la pròxima vegada que entrenis el model ja no torni a dubtar.

* **CAS C · Han escanejat el paper tan tort o mogut que el requadre vermell no ha agafat el número (`📐 Document rebel`):**
  1. Just a sota, obre el desplegable **`📐 Reajustar ROI d'aquest document rebel i re-llegir amb la CNN`**.
  2. Mou les barres (`x0 · esquerra`, `y0 · dalt`, `x1 · dreta`, `y1 · baix`) mentre mires la **Previsualització en viu** de la dreta fins que el número complet quedi ben centrat dins del retall.
  3. *(Opcional)* Si tots els documents d'aquest tipus vénen igual de moguts, marca `Guardar també aquesta ROI com a plantilla per al format`.
  4. Prem **`🔄 Re-llegir aquest document amb la nova ROI`**. El programa tornarà a llegir només aquest PDF en menys d'un segon i n'actualitzarà el codi!

### Pas 1.5 · Descarrega els documents ja reanomenats ("Ready to Work")
1. Si vols canviar el format del nom (per exemple, posar `ALB_0529820.pdf` o mantenir el nom antic al final `0529820_18.pdf`), tria-ho al desplegable **Patró de nou nom**.
2. Tens dues maneres d'emportar-te els fitxers:
   - **Opció A (Més fàcil):** Prem el botó **`📦 Descarregar ZIP reanomenat`**. Se't baixarà un fitxer `.zip` amb tots els PDFs ja canviats de nom i un Excel (`index_exportacio.csv`) amb el resum.
   - **Opció B (Directe a una carpeta de l'ordinador):** Prem **`📁 Exportar a carpeta local`**, comprova o enganxa la ruta de la carpeta on els vols i fes clic a **`💾 Copiar fitxers reanomenats`**.

**Ja està! Feina feta.** 🎉

---

## 🛠️ PART 2: COM ENSENYAR AL PROGRAMA UN DOCUMENT NOU DES DE ZERO

*Només has de fer això si algun dia vols crear un projecte nou per a un altre tipus de document que el programa encara no conegui.*

Segueix l'ordre numèric del menú de l'esquerra:

### 1️⃣ `01 · Crear projecte`
- **Nom del projecte:** Posa-li un nom clar (ex: `albarans_logistica`).
- **Departament:** Ex: `Logística`.
- **Camp a llegir:** Ex: `numero_albara`.
- **Alfabet permès:** Deixa `0123456789` (si només són números).
- **Longitud fixa del camp:** Posa quants números té el codi (si tens documents barrejats de 6 i 7 xifres, posa sempre el número més gran: **`7`**).
- Prem **`Crear projecte`**.

### 2️⃣ `02 · Upload documents`
- Selecciona el projecte que acabes de crear.
- Puja entre **15 i 30 documents d'exemple** variats i prem **`Guardar uploads`**.

### 3️⃣ `03 · ROI bàsica` (Indicar en quina zona del paper està el número)
- *ROI* vol dir "la finestra on es troba el número".
- Tria un document de mostra al desplegable.
- Mou les barres (**sliders**) de la dreta (`x0 esquerra`, `y0 dalt`, `x1 dreta`, `y1 baix`) fins que el **requadre vermell** envolti el número que vols llegir amb una mica de marge blanc al voltant.
- Comprova a **2) Crop resultant ampliat** que el número es veu sencer i no queda tallat.
- Prem **`💾 Guardar ROI`**.

### 4️⃣ `04 · Dataset / labeling` (Ensenyar exemples de números al programa)
Aquí li diem al programa quin número és cada imatge perquè aprengui:
1. Tria un document al desplegable **Document a etiquetar**.
2. Pots fer-ho en mode **Auto per Histograma-X** (talla els números sol) o **Manual amb Canvas** (dibuixes un requadre vermell amb el ratolí sobre cada número).
3. Escriu el número real que veus al camp **Valor real complet** (per exemple `515159`).
4. Prem **`💾 Guardar`** per desar els exemples.
5. Repeteix-ho amb uns quants documents fins que a la taula superior (**Estat del dataset**) tinguis exemples de tots els números del `0` al `9`.

### 5️⃣ `05 · Training` (Entrenar el "cervell" del programa)
- Quan ja tinguis exemples guardats, entra a **`05 · Training`**.
- Prem el botó **`🚀 Entrenar CNN ara`**.
- En uns segons veuràs el missatge **"Training completat"** i dues gràfiques. El model ja ha après a llegir!

### 6️⃣ `06 · Test skill` (Opcional)
- Serveix per provar un sol número i veure si el model l'encerta amb un 95%–100% de confiança. Un cop comprovat, ja pots anar sempre a **`07 · Run batch`** (PART 1).

---

## ❓ PART 3: PREGUNTES FREQÜENTS I PROBLEMES TÍPICS (SOS)

### 1. *"He fet un canvi però a la pantalla continua sortint el mateix d'abans"*
- Prem la tecla **`F5`** (o recarrega la pàgina del navegador) i torna a prémer el botó **`▶️ Processar batch`**.

### 2. *"En un document el programa s'ha deixat un número o ha sortit com a `REVIEW`"*
- Tens dues opcions igual de ràpides ( sense re-entrenar res):
  1. **Des de l'Estació de Revisió (secció 5 de `07 · Run batch`):** Mira la capçalera del document a l'esquerra, escriu el codi bo a la dreta i prem **`✅ Validar codi i marcar per exportar`** (o **`🧠 Validar + Guardar dígits per entrenar`** si vols que el model aprengui d'aquesta correcció).
  2. **Directament a la taula (secció 4):** Fes **doble clic a la cel·la `Codi (editable)`**, escriu el número correcte i prem `Enter`. L'estat passarà sol a `MANUAL_OK`, es marcarà la casella `Exportar` i el `Nou nom fitxer` s'actualitzarà a l'instant.

### 3. *"En un PDF del batch el requadre vermell ha caigut en blanc perquè l'escaneig està molt desplaçat"*
- Dins de la mateixa pantalla **`07 · Run batch`** (secció 5), obre el desplegable **`📐 Reajustar ROI d'aquest document rebel i re-llegir amb la CNN`**, mou les barres fins a centrar el número a la previsualització i prem **`🔄 Re-llegir aquest document amb la nova ROI`**.

### 4. *"A la pantalla 04 d'etiquetatge no es veu el número sencer dins el requadre"*
- Això passa si la finestra base és massa estreta. Ves un moment a **`03 · ROI bàsica`**, selecciona aquest document, obre una mica més els marges amb els sliders (`x0`, `x1`, `y0`, `y1`) i prem **`💾 Guardar ROI`**.

### 5. *"Què passa si dos PDFs del mateix lot tenen exactament el mateix codi?"*
- Tranquil, **mai s'esborrarà ni es trepitjarà cap document**. El primer es guardarà com `0529820.pdf` i el segon com `0529820_2.pdf` (o pots triar el patró `{codi}_{original}.ext` perquè es diguin `0529820_18.pdf` i `0529820_19.pdf`).
