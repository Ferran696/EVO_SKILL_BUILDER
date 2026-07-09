import base64
import json
import re
from io import BytesIO
from PIL import Image

def _image_to_base64(image: Image.Image) -> str:
    buffered = BytesIO()
    image.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode("utf-8")

def get_char_boxes_from_openai(image: Image.Image, prompt: str, api_key: str, fixed_length: int) -> dict:
    """
    Envia una imatge i un prompt a OpenAI VLM per obtenir les caixes dels caràcters.
    """
    try:
        from openai import OpenAI
    except ImportError:
        raise ImportError("OpenAI provider no disponible: 'pip install openai'")

    if not api_key:
        raise ValueError("La clau API de OpenAI no pot estar buida.")

    client = OpenAI(api_key=api_key)
    img_str = _image_to_base64(image)

    full_prompt = f"""
    {prompt}
    El camp té exactament {fixed_length} caràcters.
    Analitza la imatge i retorna un objecte JSON amb una única clau 'characters'.
    El valor ha de ser una llista de {fixed_length} objectes.
    Cada objecte ha de contenir:
    - "char": el caràcter llegit (string).
    - "box": una llista de 4 enters [x0, y0, x1, y1] representant la caixa delimitadora en píxels.

    Exemple de format de sortida per a 3 caràcters:
    {{
      "characters": [
        {{"char": "A", "box": [10, 12, 30, 42]}},
        {{"char": "B", "box": [32, 12, 52, 42]}},
        {{"char": "C", "box": [55, 11, 75, 41]}}
      ]
    }}
    """

    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": full_prompt},
                        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{img_str}"}},
                    ],
                }
            ],
            max_tokens=1024,
            response_format={"type": "json_object"},
        )
        content = response.choices[0].message.content
        return json.loads(content)
    except Exception as e:
        raise RuntimeError(f"Error en la trucada a OpenAI VLM: {e}")


def get_char_boxes_from_gemini(image: Image.Image, api_key: str, prompt: str, fixed_length: int, model_name: str = "gemini-pro-vision", mock: bool = False) -> dict:
    """
    Envia una imatge i un prompt a Gemini Vision per obtenir les caixes dels caràcters.
    Inclou un mode mock per provar la UI sense trucar a l'API real.
    """
    if mock:
        # Modo mock: devuelve una respuesta de ejemplo válida
        print("DEBUG: Ejecutando Gemini Vision en modo mock.")
        # Ejemplo para un fixed_length de 7
        example_boxes = [
            {"char": "1", "box": [5, 5, 20, 40]},
            {"char": "2", "box": [25, 5, 40, 40]},
            {"char": "3", "box": [45, 5, 60, 40]},
            {"char": "A", "box": [65, 5, 80, 40]},
            {"char": "B", "box": [85, 5, 100, 40]},
            {"char": "C", "box": [105, 5, 120, 40]},
            {"char": "7", "box": [125, 5, 140, 40]},
        ]
        if fixed_length < len(example_boxes):
            characters = example_boxes[:fixed_length]
        else:
            characters = example_boxes + [{"char": str(i % 10), "box": [10*(i+1), 5, 10*(i+2), 40]} for i in range(len(example_boxes), fixed_length)]

        return {"characters": characters}

    try:
        import google.generativeai as genai
    except ImportError:
        raise ImportError("Gemini provider no disponible: 'pip install google-generativeai'")

    if not api_key:
        raise ValueError("La clau API de Gemini no pot estar buida.")

    genai.configure(api_key=api_key)
    model = genai.GenerativeModel(model_name)

    # Prepare image for Gemini
    img_data = BytesIO()
    image.save(img_data, format="PNG")
    img_part = {
        "mime_type": "image/png",
        "data": img_data.getvalue()
    }

    full_prompt = f"""
    {prompt}
    El camp té exactament {fixed_length} caràcters.
    Analitza la imatge i retorna un objecte JSON amb una única clau 'characters'.
    El valor ha de ser una llista de {fixed_length} objectes.
    Cada objecte ha de contenir:
    - "char": el caràcter llegit (string).
    - "box": una llista de 4 enters [x0, y0, x1, y1] representant la caixa delimitadora en píxels.

    Exemple de format de sortida per a 3 caràcters:
    {{
      "characters": [
        {{"char": "A", "box": [10, 12, 30, 42]}},
        {{"char": "B", "box": [32, 12, 52, 42]}},
        {{"char": "C", "box": [55, 11, 75, 41]}}
      ]
    }}
    """

    try:
        response = model.generate_content([full_prompt, img_part], request_options={"timeout": 120})
        # Gemini Vision might return text in parts. Concatenate them.
        raw_text = "".join([part.text for part in response.candidates[0].content.parts])

        # Try to extract JSON from markdown or raw text
        match = re.search(r"```json\n({.*})\n```", raw_text, re.DOTALL)
        if match:
            json_str = match.group(1)
        else:
            json_str = raw_text # Assume it's raw JSON

        return json.loads(json_str)
    except Exception as e:
        raise RuntimeError(f"Error en la trucada a Gemini VLM: {e}. Resposta crua: {raw_text if 'raw_text' in locals() else 'N/A'}")


def validate_vlm_response(vlm_result: dict, alphabet: str, fixed_length: int, image_width: int, image_height: int) -> tuple[bool, str]:
    """
    Valida l'estructura i el contingut de la resposta del VLM.
    Retorna (True, None) si és vàlid, o (False, missatge_error) si no.
    """
    if not isinstance(vlm_result, dict):
        return False, "La resposta del VLM no és un diccionari."
    if "characters" not in vlm_result:
        return False, "La clau 'characters' no es troba a la resposta del VLM."
    
    characters = vlm_result["characters"]
    if not isinstance(characters, list):
        return False, "'characters' no és una llista."
    
    if fixed_length is not None and len(characters) != fixed_length:
        return False, f"El número de caràcters retornats ({len(characters)}) no coincideix amb la longitud fixa esperada ({fixed_length})."

    for i, item in enumerate(characters):
        if not isinstance(item, dict):
            return False, f"L'element {i} de 'characters' no és un diccionari."
        if "char" not in item:
            return False, f"L'element {i} no té la clau 'char'."
        if "box" not in item:
            return False, f"L'element {i} no té la clau 'box'."
        
        char = str(item["char"]).strip()
        if not char:
            return False, f"El caràcter a l'element {i} està buit."
        if char not in alphabet:
            # Allow validation to pass but give a warning/info that this character will be ignored later.
            # The user requested to ignore characters outside the alphabet when saving.
            # For validation, we can be more lenient here as the user might want to see the box
            # even if the character is not in the alphabet.
            # Returning an error here would prevent the display of valid boxes, which is not ideal.
            pass # The check is done later when saving.

        box = item["box"]
        if not isinstance(box, list) or len(box) != 4:
            return False, f"La 'box' de l'element {i} no és una llista de 4 elements."
        
        try:
            x0, y0, x1, y1 = [int(x) for x in box]
        except ValueError:
            return False, f"Les coordenades de la 'box' de l'element {i} no són nombres enters."
        
        if not (0 <= x0 < x1 <= image_width):
            return False, f"Les coordenades X de la 'box' de l'element {i} estan fora dels límits de la imatge o són invàlides ([{x0}, {x1}] vs ample {image_width})."
        if not (0 <= y0 < y1 <= image_height):
            return False, f"Les coordenades Y de la 'box' de l'element {i} estan fora dels límits de la imatge o són invàlides ([{y0}, {y1}] vs alt {image_height})."
            
    return True, None


def get_char_boxes_from_vlm(
    image: Image.Image,
    prompt: str,
    api_key: str,
    fixed_length: int,
    provider: str = "openai",
    model_name: str = "gpt-4o", # Default for OpenAI, can be overridden for Gemini
    mock: bool = False # Only applicable for Gemini for now
) -> dict:
    """
    Funció unificada per obtenir caixes de caràcters de diferents proveïdors VLM.
    """
    if provider == "openai":
        # Ignore mock and model_name for OpenAI as it's handled internally by get_char_boxes_from_openai
        return get_char_boxes_from_openai(image, prompt, api_key, fixed_length)
    elif provider == "gemini":
        return get_char_boxes_from_gemini(image, api_key, prompt, fixed_length, model_name=model_name, mock=mock)
    elif provider == "mock":
        # If provider is explicitly 'mock', force mock mode for Gemini to return a mock response
        return get_char_boxes_from_gemini(image, "", prompt, fixed_length, mock=True)
    else:
        raise ValueError(f"Proveïdor VLM '{provider}' no suportat. Opcions: 'openai', 'gemini', 'mock'.")