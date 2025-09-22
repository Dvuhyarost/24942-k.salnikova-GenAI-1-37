from transformers import pipeline
import re
import random

def get_file_path(filename):
    """
    Returns the absolute path to the file in the script directory.

    Parameters
    ----------
    filename : str
        The name of the file

    Returns
    -------
    str
        Absolute path to the file
    """
    import os
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), filename)

# Инициализация пайплайна для генерации текста с использованием локальной модели
generator = pipeline(
    "text-generation",
    model=get_file_path("."),  # Путь к локальной модели
    tokenizer=get_file_path(".")  # Путь к локальному токенизатору
)

def count_syllables(word):
    """
    Подсчитывает количество слогов в слове по количеству гласных.

    Parameters
    ----------
    word : str
        Слово для анализа

    Returns
    -------
    int
        Количество слогов в слове
    """
    vowels = 'аеёиоуыэюя'
    return sum(1 for char in word.lower() if char in vowels)

def count_syllables_in_last_word(text):
    """
    Подсчитывает количество слогов в последнем слове строки.

    Parameters
    ----------
    text : str
        Текстовая строка для анализа

    Returns
    -------
    int
        Количество слогов в последнем слове
    """
    if not text:
        return 0
    words = text.split()
    if not words:
        return 0
    last_word = words[-1].lower()
    return count_syllables(last_word)

def get_rhyme_vowel(text):
    """
    Извлекает последнюю гласную из последнего слова строки для рифмы.

    Parameters
    ----------
    text : str
        Текстовая строка для анализа

    Returns
    -------
    str or None
        Последняя гласная буква или None если не найдена
    """
    if not text:
        return None
    words = text.split()
    if not words:
        return None
    last_word = words[-1].lower()
    vowels = 'аеёиоуыэюя'
    # Ищем последнюю гласную в слове
    for char in reversed(last_word):
        if char in vowels:
            return char
    return None

def get_rhyme_vowel_group(vowel):
    """
    Возвращает группу фонетически соответствующих гласных.

    Parameters
    ----------
    vowel : str
        Гласная буква

    Returns
    -------
    list
        Список фонетически соответствующих гласных
    """
    rhyme_groups = {
        'а': ['а', 'я'],
        'я': ['а', 'я'],
        'о': ['о', 'ё'],
        'ё': ['о', 'ё'],
        'у': ['у', 'ю'],
        'ю': ['у', 'ю'],
        'ы': ['ы', 'и'],
        'и': ['ы', 'и'],
        'э': ['э', 'е'],
        'е': ['э', 'е']
    }
    return rhyme_groups.get(vowel, [vowel])

def check_rhyme(target_vowel, current_vowel):
    """
    Проверяет, рифмуются ли гласные с учетом фонетических соответствий.

    Parameters
    ----------
    target_vowel : str
        Целевая гласная для рифмы
    current_vowel : str
        Текущая гласная для проверки

    Returns
    -------
    bool
        True если гласные рифмуются, иначе False
    """
    if not target_vowel or not current_vowel:
        return False
    
    allowed_vowels = get_rhyme_vowel_group(target_vowel)
    return current_vowel in allowed_vowels

def get_rhyme_group_display(vowel):
    """
    Форматирует группу рифмующихся гласных для красивого отображения.

    Parameters
    ----------
    vowel : str
        Гласная буква

    Returns
    -------
    str
        Форматированная строка с группой гласных
    """
    groups = {
        'а': 'а/я', 'я': 'а/я',
        'о': 'о/ё', 'ё': 'о/ё', 
        'у': 'у/ю', 'ю': 'у/ю',
        'ы': 'ы/и', 'и': 'ы/и',
        'э': 'э/е', 'е': 'э/е'
    }
    return groups.get(vowel, vowel)

def get_stress_pattern(text):
    """
    Определяет ударение в последнем слове строки.

    Parameters
    ----------
    text : str
        Текстовая строка для анализа

    Returns
    -------
    tuple or None
        Кортеж (гласная, позиция, слово) или None если не удалось определить
    """
    if not text:
        return None
    
    words = text.split()
    if not words:
        return None
    
    last_word = words[-1].lower()
    
    # Простой словарь ударений для распространенных слов
    stress_dict = {
        'весна': 'веснА', 'весны': 'веснЫ', 'весне': 'веснЕ', 'весну': 'веснУ', 'весной': 'веснОй',
        'трава': 'травА', 'травы': 'травЫ', 'траве': 'травЕ', 'траву': 'травУ', 'травой': 'травОй',
        'цветы': 'цветЫ', 'цветов': 'цветОв', 'цветам': 'цветАм', 'цветами': 'цветАми',
        'дождь': 'дОждь', 'дождя': 'дождЯ', 'дождю': 'дождЮ', 'дождем': 'дождЕм',
        'солнце': 'сОлнце', 'солнца': 'сОлнца', 'солнцу': 'сОлнцу', 'солнцем': 'сОлнцем',
        'ветер': 'вЕтер', 'ветра': 'ветрА', 'ветру': 'ветрУ', 'ветром': 'вЕтром',
        'птицы': 'птИцы', 'птиц': 'птИц', 'птицам': 'птицАм', 'птицами': 'птицАми',
        'листья': 'лИстья', 'листьев': 'лИстьев', 'листьям': 'листьЯм', 'листьями': 'листьЯми',
        'деревья': 'дерЕвья', 'деревьев': 'дерЕвьев', 'деревьям': 'деревьЯм', 'деревьями': 'деревьЯми',
        'ручьи': 'ручьИ', 'ручьев': 'ручьЁв', 'ручьям': 'ручьЯм', 'ручьями': 'ручьЯми',
        'поля': 'полЯ', 'полей': 'полЕй', 'полям': 'полЯм', 'полями': 'полЯми',
        'сады': 'садЫ', 'садов': 'садОв', 'садам': 'садАм', 'садами': 'садАми',
        'луга': 'лугА', 'лугов': 'лугОв', 'лугам': 'лугАм', 'лугами': 'лугАми',
        'река': 'рекА', 'реки': 'рекИ', 'реке': 'рекЕ', 'реку': 'рекУ', 'рекой': 'рекОй',
        'озеро': 'Озеро', 'озера': 'Озера', 'озеру': 'Озеру', 'озером': 'Озером',
        'небо': 'нЕбо', 'неба': 'нЕба', 'небу': 'нЕбу', 'небом': 'нЕбом',
        'утро': 'Утро', 'утра': 'Утра', 'утру': 'Утру', 'утром': 'Утром',
        'день': 'дЕнь', 'дня': 'днЯ', 'дню': 'днЮ', 'днем': 'днЕм',
        'ночь': 'нОчь', 'ночи': 'нОчи', 'ночью': 'нОчью',
        'месяц': 'мЕсяц', 'месяца': 'мЕсяца', 'месяцу': 'мЕсяцу', 'месяцем': 'мЕсяцем',
        'звезды': 'звЁзды', 'звезд': 'звёзд', 'звездам': 'звездАм', 'звездами': 'звездАми',
        'свет': 'свЕт', 'света': 'свЕта', 'свету': 'свЕту', 'светом': 'свЕтом',
        'тепло': 'теплО', 'тепла': 'теплА', 'теплу': 'теплУ', 'теплом': 'теплОм',
        'радость': 'рАдость', 'радости': 'рАдости', 'радостью': 'рАдостью',
        'красота': 'красотА', 'красоты': 'красотЫ', 'красоте': 'красотЕ', 'красотой': 'красотОй',
        'любовь': 'любОвь', 'любви': 'любвИ', 'любовью': 'любОвью',
        'надежда': 'надЕжда', 'надежды': 'надЕжды', 'надежде': 'надЕжде', 'надеждой': 'надЕждой',
        'счастье': 'счАстье', 'счастья': 'счАстья', 'счастью': 'счАстью', 'счастьем': 'счАстьем'
    }
    
    # Проверяем, есть ли слово в словаре ударений
    for word_form, stressed_form in stress_dict.items():
        if last_word == word_form:
            # Находим ударную гласную
            for i, char in enumerate(stressed_form):
                if char.isupper():
                    vowel = char.lower()
                    position = i
                    return vowel, position, last_word
    
    # Если слово не найдено в словаре, используем эвристику
    vowels = 'аеёиоуыэюя'
    vowel_positions = []
    
    # Находим все гласные и их позиции
    for i, char in enumerate(last_word):
        if char in vowels:
            vowel_positions.append((char, i))
    
    if not vowel_positions:
        return None
    
    # Эвристика: ударение обычно падает на предпоследний слог
    if len(vowel_positions) >= 2:
        # Берем предпоследнюю гласную
        stressed_vowel, position = vowel_positions[-2]
        return stressed_vowel, position, last_word
    else:
        # Если только одна гласная, она и будет ударной
        stressed_vowel, position = vowel_positions[0]
        return stressed_vowel, position, last_word

def get_rhyme_type_by_stress(position, word_length):
    """
    Определяет тип рифмы по положению ударения.

    Parameters
    ----------
    position : int
        Позиция ударной гласной в слове
    word_length : int
        Длина слова в символах

    Returns
    -------
    str
        Тип рифмы: мужская, женская, дактилическая, гипердактилическая
    """
    if position == word_length - 1:
        return "мужская"
    elif position >= word_length - 2:
        return "женская"
    elif position >= word_length - 3:
        return "дактилическая"
    else:
        return "гипердактилическая"

def generate_spring_poem():
    """
    Генерирует полное весеннее стихотворение.

    Returns
    -------
    tuple
        Кортеж (строки стихотворения, схема рифмовки, информация об ударении, количество слогов)
    """
    
    # Список для хранения сгенерированных строк стихотворения
    poem_lines = []
    first_line_stress_info = None
    first_line_syllables = 0
    
    # Выбираем случайную схему рифмовки (только 1-2 и 3-4)
    rhyme_schemes = ["1-2 и 3-4"]  # только классическая схема
    selected_scheme = random.choice(rhyme_schemes)
    print(f"Генерируем весеннее стихотворение с рифмовкой: {selected_scheme}")
    
    def clean_poem_line(text):
        """
        Очищает строку от всех символов, кроме русских букв и пробелов.

        Parameters
        ----------
        text : str
            Исходный текст

        Returns
        -------
        str
            Очищенный текст
        """
        # Удаляем все, кроме русских букв, пробелов и дефиса
        cleaned = re.sub(r'[^а-яёА-ЯЁ\s-]', '', text)
        # Удаляем лишние пробелы
        cleaned = re.sub(r'\s+', ' ', cleaned).strip()
        # Удаляем дефисы в начале и конце строки
        cleaned = cleaned.strip('-')
        return cleaned
    
    def get_rhyme_target(line_num, scheme):
        """
        Определяет, с какой строкой должна рифмоваться текущая.

        Parameters
        ----------
        line_num : int
            Номер текущей строки
        scheme : str
            Схема рифмовки

        Returns
        -------
        int or None
            Номер строки для рифмы или None
        """
        if scheme == "1-2 и 3-4":
            if line_num == 1: return 0  # 2-я строка рифмуется с 1-й
            if line_num == 3: return 2  # 4-я строка рифмуется с 3-й
        return None
    
    def is_valid_poem_line(text, min_words=3, max_words=8):
        """
        Проверяет валидность строки стихотворения.

        Parameters
        ----------
        text : str
            Текст для проверки
        min_words : int, optional
            Минимальное количество слов
        max_words : int, optional
            Максимальное количество слов

        Returns
        -------
        bool
            True если строка валидна, иначе False
        """
        if not text:
            return False
        words = text.split()
        # Проверяем количество слов и наличие только русских букв
        if len(words) < min_words or len(words) > max_words:
            return False
        # Проверяем, что все слова состоят только из русских букв
        for word in words:
            if not re.match(r'^[а-яёА-ЯЁ-]+$', word):
                return False
        return True
    
    def is_duplicate_line(new_line, existing_lines):
        """
        Проверяет наличие повторений строк.

        Parameters
        ----------
        new_line : str
            Новая строка
        existing_lines : list
            Существующие строки

        Returns
        -------
        bool
            True если есть повторение, иначе False
        """
        if not new_line:
            return False
        new_lower = new_line.lower()
        for existing_line in existing_lines:
            if existing_line and new_lower == existing_line.lower():
                return True
        return False
    
    def check_stress_compatibility(new_line, target_stress_info):
        """
        Проверяет соответствие ударения в новой строке целевому.

        Parameters
        ----------
        new_line : str
            Новая строка
        target_stress_info : tuple
            Информация об ударении целевой строки

        Returns
        -------
        bool
            True если ударение совпадает, иначе False
        """
        if not target_stress_info:
            return True
            
        new_stress = get_stress_pattern(new_line)
        if not new_stress:
            return False
            
        vowel_new, position_new, word_new = new_stress
        vowel_target, position_target, word_target = target_stress_info
        
        # Проверяем тип рифмы (должен совпадать)
        rhyme_type_new = get_rhyme_type_by_stress(position_new, len(word_new))
        rhyme_type_target = get_rhyme_type_by_stress(position_target, len(word_target))
        
        return rhyme_type_new == rhyme_type_target
    
    def evaluate_line_quality(new_line, target_stress_info, target_syllables, rhyme_vowel):
        """
        Оценивает качество строки по нескольким параметрам.

        Parameters
        ----------
        new_line : str
            Новая строка для оценки
        target_stress_info : tuple
            Информация об ударении целевой строки
        target_syllables : int
            Целевое количество слогов
        rhyme_vowel : str
            Целевая гласная для рифмы

        Returns
        -------
        tuple
            Кортеж (оценка качества, список проблем)
        """
        quality_score = 0
        issues = []
        
        # Проверяем рифму (самый важный параметр)
        if rhyme_vowel:
            current_rhyme_vowel = get_rhyme_vowel(new_line)
            if not check_rhyme(rhyme_vowel, current_rhyme_vowel):
                quality_score -= 100  # Большой штраф за плохую рифму
                issues.append("плохая рифма")
        
        # Проверяем ударение
        if target_stress_info:
            if not check_stress_compatibility(new_line, target_stress_info):
                quality_score -= 50  # Штраф за несовпадение ударения
                issues.append("несовпадение ударения")
        
        # Проверяем количество слогов
        new_syllables = count_syllables_in_last_word(new_line)
        syllables_diff = abs(new_syllables - target_syllables)
        if syllables_diff > 0:
            quality_score -= syllables_diff * 10  # Штраф за разницу в слогах
            issues.append(f"разница в слогах: {syllables_diff}")
        
        # Бонус за идеальное соответствие
        if syllables_diff == 0 and not issues:
            quality_score += 20
        
        return quality_score, issues
    
    # Настройки генерации с адаптивными параметрами для разных строк
    generation_settings = {
        "temperature": 0.85,       # Баланс между креативностью и предсказуемостью
        "top_p": 0.92,             # Нуклеус-сэмплинг: учитываем только 92% наиболее вероятных слов
        "repetition_penalty": 1.3, # Штраф за повторения
        "max_new_tokens": 20,      # Максимальная длина генерируемого текста
        "top_k": 50                # Ограничение на количество рассматриваемых вариантов
    }
    
    # Генерация 4 строк стихотворения
    for line_num in range(4):
        rhyme_target_num = get_rhyme_target(line_num, selected_scheme)
        rhyme_vowel = None
        rhyme_target_text = None
        allowed_vowels = []
        stress_requirement = ""
        syllables_requirement = ""
        target_syllables = 0
        target_stress_info = None
        
        if rhyme_target_num is not None and rhyme_target_num < len(poem_lines):
            rhyme_target_text = poem_lines[rhyme_target_num]
            rhyme_vowel = get_rhyme_vowel(rhyme_target_text)
            if rhyme_vowel:
                allowed_vowels = get_rhyme_vowel_group(rhyme_vowel)
            
            # Для первой строки анализируем ударение и слоги
            if rhyme_target_num == 0 and first_line_stress_info is None:
                first_line_stress_info = get_stress_pattern(rhyme_target_text)
                first_line_syllables = count_syllables_in_last_word(rhyme_target_text)
                if first_line_stress_info:
                    vowel, position, word = first_line_stress_info
                    rhyme_type = get_rhyme_type_by_stress(position, len(word))
                    stress_requirement = f" с {rhyme_type} рифмой"
                    syllables_requirement = f" и {first_line_syllables} слогами в последнем слове"
            
            target_syllables = count_syllables_in_last_word(rhyme_target_text)
            target_stress_info = first_line_stress_info if rhyme_target_num == 0 else None
        
        # Формируем промпт в зависимости от номера строки и схемы рифмовки
        if line_num == 0:
            prompt = "Напиши первую строку весеннего стихотворения о природе, 4-7 слов, с выразительным окончанием:"
        
        else:
            previous_lines = "\n".join(poem_lines[:line_num])
            
            if rhyme_vowel and rhyme_target_text and allowed_vowels:
                target_line_num = rhyme_target_num + 1
                vowels_str = "', '".join(allowed_vowels)
                
                # Добавляем информацию об ударении и слогах в промпт
                prompt = f"Продолжи весеннее стихотворение. Строка {line_num+1} должна рифмоваться со строкой {target_line_num}{stress_requirement}{syllables_requirement} (допустимые рифмы: '{vowels_str}'):\n{previous_lines}\nСтрока {line_num+1}:"
            else:
                prompt = f"Продолжи весеннее стихотворение:\n{previous_lines}\nСтрока {line_num+1}:"
        
        # Многократные попытки генерации валидной строки
        max_attempts = 25
        best_line = None
        best_quality = float('-inf')
        best_issues = []
        candidate_lines = []
        
        for attempt in range(max_attempts):
            # Настраиваем параметры для повторных попыток (увеличиваем температуру для разнообразия)
            current_temp = generation_settings["temperature"] + (attempt * 0.02)
            
            # Генерация текста с использованием модели
            result = generator(
                prompt,
                max_new_tokens=generation_settings["max_new_tokens"],
                num_return_sequences=1,
                do_sample=True,
                temperature=min(current_temp, 1.0),
                top_p=generation_settings["top_p"],
                top_k=generation_settings["top_k"],
                repetition_penalty=generation_settings["repetition_penalty"],
                no_repeat_ngram_size=2,
                truncation=True,
                pad_token_id=50256
            )
    
            # Извлечение сгенерированного текста
            full_text = result[0]['generated_text']
            new_line = full_text.replace(prompt, "").strip()
            
            # Очистка строки
            new_line = clean_poem_line(new_line)
            
            # Проверяем валидность и отсутствие повторений
            if (is_valid_poem_line(new_line) and 
                not is_duplicate_line(new_line, poem_lines) and
                len(new_line) >= 8):
                
                # Оцениваем качество строки
                quality_score, issues = evaluate_line_quality(
                    new_line, target_stress_info, target_syllables, rhyme_vowel
                )
                
                candidate_lines.append((new_line, quality_score, issues))
                
                # Сохраняем лучший вариант
                if quality_score > best_quality:
                    best_line = new_line
                    best_quality = quality_score
                    best_issues = issues
                
                # Если идеальное качество - используем сразу
                if quality_score >= 0 and not issues:
                    poem_lines.append(new_line)
                    
                    # Выводим информацию о рифме и ударении
                    if rhyme_vowel:
                        target_line_num = rhyme_target_num + 1
                        current_vowel = get_rhyme_vowel(new_line)
                        
                        # Анализируем ударение
                        stress_info = ""
                        if target_stress_info:
                            new_stress = get_stress_pattern(new_line)
                            if new_stress:
                                vowel_new, position_new, word_new = new_stress
                                rhyme_type = get_rhyme_type_by_stress(position_new, len(word_new))
                                stress_info = f" [{rhyme_type} рифма]"
                        
                        syllables_info = f" [слоги: {count_syllables_in_last_word(new_line)}]"
                        print(f"✓ Строка {line_num+1}: {new_line} [рифма со строкой {target_line_num}: {rhyme_vowel}→{current_vowel}{stress_info}{syllables_info}]")
                    else:
                        print(f"✓ Строка {line_num+1}: {new_line}")
                    
                    break
            
            elif attempt % 5 == 0:
                print(f"Попытка {attempt+1}/{max_attempts} для строки {line_num+1}")
        
        # Если не нашли идеальное соответствие, используем лучший вариант
        if not poem_lines or len(poem_lines) <= line_num:
            if best_line:
                poem_lines.append(best_line)
                if rhyme_vowel:
                    target_line_num = rhyme_target_num + 1
                    current_vowel = get_rhyme_vowel(best_line)
                    syllables_info = f" [слоги: {count_syllables_in_last_word(best_line)}]"
                    issues_info = f" [проблемы: {', '.join(best_issues)}]" if best_issues else ""
                    print(f"✓ Строка {line_num+1} (лучший вариант): {best_line}{syllables_info}{issues_info}")
                else:
                    print(f"✓ Строка {line_num+1}: {best_line}")
            else:
                # Пробуем сгенерировать без строгой проверки
                print(f"⚠ Не удалось найти хорошую рифму, пробуем без строгих ограничений")
                simplified_prompt = prompt.replace("должна рифмоваться", "должна продолжаться")
                simplified_prompt = re.sub(r" с .*? рифмой", "", simplified_prompt)
                simplified_prompt = re.sub(r" и \d+ слогами", "", simplified_prompt)
                
                result = generator(
                    simplified_prompt,
                    max_new_tokens=generation_settings["max_new_tokens"],
                    num_return_sequences=1,
                    do_sample=True,
                    temperature=0.9,
                    top_p=0.95
                )
                full_text = result[0]['generated_text']
                new_line = clean_poem_line(full_text.replace(simplified_prompt, "").strip())
                if (is_valid_poem_line(new_line) and 
                    not is_duplicate_line(new_line, poem_lines)):
                    poem_lines.append(new_line)
                    print(f"✓ Строка {line_num+1} (без строгой рифмы): {new_line}")
                else:
                    poem_lines.append("")
                    print(f"✗ Строка {line_num+1}: [не сгенерировалась]")
    
    return poem_lines, selected_scheme, first_line_stress_info, first_line_syllables

# Основной цикл генерации - повторяем пока не получим полное стихотворение
max_attempts = 8
final_poem = None
final_scheme = None
final_stress_info = None
final_syllables = 0

for attempt in range(1, max_attempts + 1):
    print(f"\n{'='*60}")
    print(f"ПОПЫТКА ГЕНЕРАЦИИ #{attempt}")
    print(f"{'='*60}")
    
    poem_lines, scheme, stress_info, syllables = generate_spring_poem()
    
    # Проверяем, все ли строки сгенерировались
    all_lines_valid = all(line != "" for line in poem_lines)
    
    if all_lines_valid:
        print(f"\n Успешно сгенерировано полное стихотворение за {attempt} попыток!")
        final_poem = poem_lines
        final_scheme = scheme
        final_stress_info = stress_info
        final_syllables = syllables
        break
    else:
        print(f"\n Попытка {attempt}: не все строки сгенерировались, пробуем снова...")
        
        # Подсчитываем количество пропущенных строк
        empty_lines = sum(1 for line in poem_lines if line == "")
        print(f"Пропущено строк: {empty_lines}")
        
        if attempt < max_attempts:
            print("Перезапускаем генерацию...\n")
        else:
            print("Достигнуто максимальное количество попыток.")
            # Используем последнюю попытку, даже если не все строки идеальны
            final_poem = poem_lines
            final_scheme = scheme
            final_stress_info = stress_info
            final_syllables = syllables

# Вывод итогового стихотворения
print("\n" + "="*60)
print("ФИНАЛЬНОЕ ВЕСЕННЕЕ СТИХОТВОРЕНИЕ:")
print(f"Схема рифмовки: {final_scheme}")
if final_stress_info:
    vowel, position, word = final_stress_info
    rhyme_type = get_rhyme_type_by_stress(position, len(word))
    print(f"Тип рифмы: {rhyme_type} (ударение на '{vowel}' в позиции {position+1} слова '{word}')")
    print(f"Количество слогов в первой строке: {final_syllables}")
print("="*60)

if final_poem and any(line != "" for line in final_poem):
    for i, line in enumerate(final_poem, 1):
        if line:
            print(f"{i}. {line}")
        else:
            print(f"{i}. [не сгенерировалась]")
else:
    print("Не удалось сгенерировать стихотворение.")