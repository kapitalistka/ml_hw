import re

def read_texts_from_file(filename):
    texts = []
    try:
        with open(filename, 'r', encoding='utf-8') as file:
            texts = [line.strip() for line in file if line.strip()]
        print(f"Загружено {len(texts)} строк из файла.")
    except FileNotFoundError:
        print(f"Файл {filename} не найден.")
    except Exception as e:
        print(f"Произошла ошибка при чтении файла: {e}")

    return texts

def save_texts_to_file(texts, filename):
    try:
        with open(filename, "w", encoding='utf-8') as f:
            for text in texts:
                f.write(text.strip() + '\n')
        print(f"✅ Успешно сохранено {len(texts)} текстов в файл: {filename}")


    except Exception as e:
        raise RuntimeError(f"Ошибка при сохранении файла: {e}")

def clean_text(text):
    text = text.lower()
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    text = re.sub(r'@\w+', '', text)
    text = re.sub(r'[^\w\s.,!?;:()\-]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def clean_texts(texts):
    return [clean_text(text) for text in texts]