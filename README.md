# EVO Skill Builder

Eina interna per crear i executar "skills" de lectura visual de camps documentals (albarans, contractes, parts) amb reindexació i exportació automàtica de fitxers.

👉 **[Obrir la Guia Ràpida d'Usuari (Job Aid pas a pas per a tothom)](JOBAID_USUARI.md)**

## Funcionalitats principals

- **Enrutador Multi-Format Automàtic (`core/router.py`):** Classifica automàticament documents de Tipus 1 (Codi de barres · 7 dígits), Tipus 2 (Logo Carburos Metálicos · 6 dígits) i Tipus 3 (Excepció amb punts · 6 dígits nets).
- **Ancoratge Visual Dinàmic i Deskewing (`core/router.py`, `core/segmentation.py`):** Localitza clústers numèrics, codis de barres i capçaleres per re-centrar la ROI encara que el document s'hagi escanejat mogut o inclinat.
- **Segmentació Robusta (`core/segmentation.py`):** Extracció per Components Connectats i divisió per valls (`valley splitting`) per separar dígits enganxats i filtrar soroll, punts o textos paràsits.
- **Entrenament CNN (`SmallCharCNN`):** Entrenament ràpid per projecte amb control de bloqueig de GPU (`training_lock`).
- **Reindexació, Exportació i Reset de Batch (`07 · Run batch`, `core/storage.py`):** Taula interactiva amb edició de codis en temps real, previsualització del nou nom de fitxer, protecció contra duplicats, exportació directa a `.ZIP` o a carpeta local amb índex CSV de traçabilitat i botó `🧹 Netejar / Següent Batch (Reset)` per buidar el carregador i netejar temporals en un sol clic.
- **Estació de Revisió Ràpida (`REVIEW`) i Active Learning (`07 · Run batch`, `core/labels.py`):** Cua filtrada de documents pendents amb visor de capçalera original, validació en 1 clic (`MANUAL_OK` + auto-selecció per exportar), reajust individual de ROI amb re-lectura CNN al vol i guardat directe de dígits corregits a `samples/` + `labels.csv`.

## Important

No pujar documents amb dades personals o sensibles sense autorització.
